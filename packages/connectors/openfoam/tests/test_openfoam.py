# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_openfoam import connector as of
from frontier_connector_openfoam.connector import OpenFoamConnector

import frontier_connectors as fc
from frontier_connectors.errors import ConnectorError, SelectionError, UnsupportedSourceError
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    fetch_sample,
    hex_grid,
    open_store,
    sample_cache,
    tet_grid,
)

# -- a programmatic OpenFOAM case writer (blockMesh + decomposePar stand-in) ---------------------

BANNERS = {
    "esi": (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
        "|  \\\\    /   O peration     | Version:  v2312                                 |\n"
        "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
    ),
    "foundation": (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  11\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
    ),
    "extend": (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\\\      /  F ield         | foam-extend: Open Source CFD                    |\n"
        "|  \\\\    /   O peration     | Version:     4.1                                |\n"
        "|   \\\\  /    A nd           | Web:         http://www.foam-extend.org         |\n"
        "|    \\\\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
    ),
}

HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
TET_FACES = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]


@dataclass
class PolyMesh:
    """An OpenFOAM face-addressed mesh: internal faces first (upper-triangular order), then
    the patches; every face normal points out of its owner."""

    points: np.ndarray
    faces: list[tuple[int, ...]]
    owner: np.ndarray
    neighbour: np.ndarray
    patches: list[tuple[str, str, int, int]]  # name, type, nFaces, startFace
    n_cells: int
    cell_addr: np.ndarray | None = None
    point_addr: np.ndarray | None = None
    proc_patches: dict[str, str] = field(default_factory=dict)

    def face_centres(self) -> np.ndarray:
        return np.array([self.points[list(f)].mean(axis=0) for f in self.faces])

    def cell_centres(self) -> np.ndarray:
        """Mean of the vertices of each cell (exact centroid for boxes and tetrahedra)."""
        verts: list[set[int]] = [set() for _ in range(self.n_cells)]
        for i, f in enumerate(self.faces):
            verts[self.owner[i]].update(f)
            if i < len(self.neighbour):
                verts[self.neighbour[i]].update(f)
        return np.array([self.points[sorted(v)].mean(axis=0) for v in verts])


def build_polymesh(
    points: np.ndarray,
    cells: np.ndarray,
    patch_of: Callable[[np.ndarray], str],
    patch_types: dict[str, str],
    *,
    cell_ids: Sequence[int] | None = None,
    proc_of: np.ndarray | None = None,
    me: int = 0,
) -> PolyMesh:
    """Face-addressed mesh of the cells ``cell_ids`` (default all). Faces shared with a cell of
    another processor (``proc_of``) become ``procBoundary<me>to<other>`` patches."""
    table = HEX_FACES if cells.shape[1] == 8 else TET_FACES
    ids = list(range(len(cells))) if cell_ids is None else list(cell_ids)
    local = {c: i for i, c in enumerate(ids)}
    seen: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    all_faces: dict[tuple[int, ...], list[int]] = {}
    for c, cell in enumerate(cells):
        for f in table:
            all_faces.setdefault(tuple(sorted(int(cell[i]) for i in f)), []).append(c)
    internal: list[tuple[int, int, tuple[int, ...]]] = []
    boundary: dict[str, list[tuple[int, tuple[int, ...]]]] = {n: [] for n in patch_types}
    procs: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    for c in ids:
        cell = cells[c]
        if len(cell) == 4:
            a, b, cc, d = points[cell]
            if np.dot(np.cross(b - a, cc - a), d - a) < 0:
                cell = cell[[0, 2, 1, 3]]
        for f in table:
            verts = tuple(int(cell[i]) for i in f)
            key = tuple(sorted(verts))
            others = [o for o in all_faces[key] if o != c]
            if not others:
                boundary[patch_of(points[list(verts)].mean(axis=0))].append((local[c], verts))
            elif others[0] in local:
                if key in seen:
                    own, own_verts = seen[key]
                    internal.append((own, local[c], own_verts))
                else:
                    seen[key] = (local[c], verts)
            else:
                assert proc_of is not None
                name = f"procBoundary{me}to{int(proc_of[others[0]])}"
                procs.setdefault(name, []).append((local[c], verts))
    internal.sort(key=lambda t: (t[0], t[1]))
    faces = [f for _, _, f in internal]
    owner = [o for o, _, _ in internal]
    patches = []
    for name, items in [*boundary.items(), *sorted(procs.items())]:
        items.sort(key=lambda t: t[0])
        ptype = patch_types.get(name, "processor")
        patches.append((name, ptype, len(items), len(faces)))
        faces += [f for _, f in items]
        owner += [o for o, _ in items]
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    remap = np.full((len(points),), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return PolyMesh(
        points=points[used],
        faces=[tuple(int(remap[v]) for v in f) for f in faces],
        owner=np.array(owner, dtype=np.int64),
        neighbour=np.array([n for _, n, _ in internal], dtype=np.int64),
        patches=patches,
        n_cells=len(ids),
        cell_addr=np.array(ids, dtype=np.int64),
        point_addr=used,
        proc_patches={n: n.split("to")[1] for n in procs},
    )


def box_patch(size: tuple[float, float, float], two_d: bool) -> Callable[[np.ndarray], str]:
    def patch_of(c: np.ndarray) -> str:
        if abs(c[0]) < 1e-9:
            return "inlet"
        if abs(c[0] - size[0]) < 1e-9:
            return "outlet"
        if two_d and (abs(c[2]) < 1e-9 or abs(c[2] - size[2]) < 1e-9):
            return "frontAndBack"
        return "walls"

    return patch_of


def header(cls: str, obj: str, fmt: str, flavor: str, note: str = "", arch: str = "") -> bytes:
    lines = [BANNERS[flavor], "FoamFile\n{\n    version     2.0;\n", f"    format      {fmt};\n"]
    if fmt == "binary" and arch:
        lines.append(f'    arch        "{arch}";\n')
    lines.append(f"    class       {cls};\n")
    if note:
        lines.append(f'    note        "{note}";\n')
    lines.append(f"    object      {obj};\n}}\n")
    lines.append(
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )
    return "".join(lines).encode()


@dataclass
class Style:
    fmt: str = "ascii"
    gz: bool = False
    flavor: str = "esi"
    label: int = 32
    scalar: int = 64
    note: bool = True

    @property
    def arch(self) -> str:
        return f"LSB;label={self.label};scalar={self.scalar}"

    @property
    def binary(self) -> bool:
        return self.fmt == "binary"


def put(path: Path, data: bytes, style: Style) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if style.gz:
        with gzip.open(str(path) + ".gz", "wb") as f:
            f.write(data)
    else:
        path.write_bytes(data)


def raw_list(values: np.ndarray, style: Style, kind: str) -> bytes:
    """``N(...)`` body of a primitive list in ASCII or binary."""
    values = np.asarray(values)
    n = len(values)
    if style.binary and n:
        dtype = f"<i{style.label // 8}" if kind == "label" else f"<f{style.scalar // 8}"
        return f"{n}\n(".encode() + values.astype(dtype).tobytes() + b")\n"
    if n == 0:
        return b"0()\n"
    if values.ndim == 1:
        body = "\n".join(str(int(v)) if kind == "label" else f"{v:.17g}" for v in values)
    else:
        body = "\n".join("(" + " ".join(f"{v:.17g}" for v in row) + ")" for row in values)
    return f"{n}\n(\n{body}\n)\n".encode()


def write_mesh(mesh_dir: Path, mesh: PolyMesh, style: Style) -> None:
    n_int = len(mesh.neighbour)
    note = (
        f"nPoints:{len(mesh.points)}  nCells:{mesh.n_cells}  nFaces:{len(mesh.faces)}  "
        f"nInternalFaces:{n_int}"
        if style.note
        else ""
    )

    def head(cls: str, obj: str, note: str = "") -> bytes:
        return header(cls, obj, style.fmt, style.flavor, note, style.arch)

    put(
        mesh_dir / "points",
        head("vectorField", "points") + raw_list(mesh.points, style, "vector"),
        style,
    )
    if style.binary:
        sizes = np.array([len(f) for f in mesh.faces])
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        flat = np.concatenate([np.asarray(f) for f in mesh.faces])
        faces = (
            head("faceCompactList", "faces")
            + raw_list(offsets, style, "label")
            + b"\n"
            + raw_list(flat, style, "label")
        )
    else:
        body = "\n".join(f"{len(f)}(" + " ".join(str(v) for v in f) + ")" for f in mesh.faces)
        faces = head("faceList", "faces") + f"{len(mesh.faces)}\n(\n{body}\n)\n".encode()
    put(mesh_dir / "faces", faces, style)
    put(
        mesh_dir / "owner",
        head("labelList", "owner", note) + raw_list(mesh.owner, style, "label"),
        style,
    )
    put(
        mesh_dir / "neighbour",
        head("labelList", "neighbour", note) + raw_list(mesh.neighbour, style, "label"),
        style,
    )
    rows = [f"{len(mesh.patches)}\n(\n"]
    for name, ptype, n, start in mesh.patches:
        rows.append(f"    {name}\n    {{\n        type            {ptype};\n")
        if ptype in ("wall", "empty"):
            rows.append(f"        inGroups        1({ptype});\n")
        if ptype == "processor":
            me = name[len("procBoundary") :].split("to")[0]
            rows.append(
                f"        myProcNo        {me};\n"
                f"        neighbProcNo    {mesh.proc_patches[name]};\n"
            )
        rows.append(f"        nFaces          {n};\n        startFace       {start};\n    }}\n")
    rows.append(")\n")
    # the boundary file is always ASCII, also in binary cases
    plain = Style(fmt=style.fmt, gz=style.gz, flavor=style.flavor)
    put(mesh_dir / "boundary", head("polyBoundaryMesh", "boundary") + "".join(rows).encode(), plain)
    if mesh.cell_addr is not None and mesh.proc_patches:
        for name, addr in (
            ("cellProcAddressing", mesh.cell_addr),
            ("pointProcAddressing", mesh.point_addr),
        ):
            assert addr is not None
            put(mesh_dir / name, head("labelList", name) + raw_list(addr, style, "label"), style)


SYMM = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def analytic(centres: np.ndarray, t: float) -> dict[str, np.ndarray]:
    f = analytic_fields(centres, t)
    u = f["U"]
    f["R"] = np.stack([u[:, i] * u[:, j] + 0.1 * (i + j) for i, j in SYMM], axis=1)
    f["gradU"] = np.stack([centres[:, i % 3] * (1.0 + i) + t for i in range(9)], axis=1)
    return f


FIELD_SPECS = {
    "p": ("volScalarField", "scalar", "[0 2 -2 0 0 0 0]"),
    "U": ("volVectorField", "vector", "[0 1 -1 0 0 0 0]"),
    "R": ("volSymmTensorField", "symmTensor", "[0 2 -2 0 0 0 0]"),
    "gradU": ("volTensorField", "tensor", "[0 0 -1 0 0 0 0]"),
    "T": ("volScalarField", "scalar", "[0 0 0 1 0 0 0]"),
}
INLET_U = (1.5, 0.0, 0.0)


def write_field(
    path: Path, name: str, mesh: PolyMesh, t: float, style: Style, *, uniform: float | None = None
) -> None:
    cls, prim, dims = FIELD_SPECS[name]
    out = [header(cls, name, style.fmt, style.flavor, arch=style.arch)]
    out.append(f"dimensions      {dims};\n\n".encode())
    centres = mesh.cell_centres()
    fc_ = mesh.face_centres()
    if uniform is not None:
        out.append(f"internalField   uniform {uniform:g};\n\n".encode())
    else:
        out.append(f"internalField   nonuniform List<{prim}> \n".encode())
        out.append(raw_list(analytic(centres, t)[name], style, prim) + b";\n\n")
    out.append(b"boundaryField\n{\n")
    for pname, ptype, n, start in mesh.patches:
        out.append(f"    {pname}\n    {{\n".encode())
        if ptype == "empty":
            out.append(b"        type            empty;\n")
        elif ptype == "processor":
            vals = analytic(fc_[start : start + n], t)[name] if uniform is None else None
            out.append(b"        type            processor;\n")
            if vals is None:
                out.append(f"        value           uniform {uniform:g};\n".encode())
            else:
                out.append(f"        value           nonuniform List<{prim}> \n".encode())
                out.append(raw_list(vals, style, prim) + b";\n")
        elif uniform is not None:
            out.append(
                b"        type            fixedValue;\n"
                + f"        value           uniform {uniform:g};\n".encode()
            )
        elif name == "U" and pname == "inlet":
            v = " ".join(f"{c:g}" for c in INLET_U)
            out.append(
                b"        type            fixedValue;\n"
                + f"        value           uniform ({v});\n".encode()
            )
        elif name == "p" and pname in ("inlet", "walls"):
            out.append(b"        type            zeroGradient;\n")
        elif name == "p" and pname == "outlet":
            out.append(b"        type            fixedValue;\n        value           uniform 0;\n")
        elif name == "U" and pname == "walls":
            out.append(b"        type            noSlip;\n")
        else:
            vals = analytic(fc_[start : start + n], t)[name]
            out.append(b"        type            calculated;\n")
            out.append(f"        value           nonuniform List<{prim}> \n".encode())
            out.append(raw_list(vals, style, prim) + b";\n")
        out.append(b"    }\n")
    out.append(b"}\n")
    put(path, b"".join(out), style)


def write_dicts(
    root: Path, *, application: str, steady: bool, style: Style, turbulence: str
) -> None:
    def head(obj: str) -> str:
        return header("dictionary", obj, "ascii", style.flavor).decode()

    system, const = root / "system", root / "constant"
    system.mkdir(parents=True, exist_ok=True)
    const.mkdir(parents=True, exist_ok=True)
    (system / "controlDict").write_text(
        head("controlDict")
        + f"application     {application};\n\nstartFrom       startTime;\nstartTime       0;\n"
        "stopAt          endTime;\nendTime         1;\ndeltaT          0.005;\n"
        f"writeControl    timeStep;\nwriteInterval   20;\nwriteFormat     {style.fmt};\n"
        "writePrecision  6;\n#includeFunc     residuals\n"
        "functions\n{\n    #includeFunc streamlines(U, p)\n}\n"
    )
    ddt = "steadyState" if steady else "Euler"
    (system / "fvSchemes").write_text(
        head("fvSchemes") + f"ddtSchemes\n{{\n    default         {ddt};\n}}\n"
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
        "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n"
    )
    (system / "fvSolution").write_text(
        head("fvSolution") + "SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n}\n"
    )
    if style.flavor == "extend":
        # foam-extend / OpenFOAM <= 2.2: dimensioned entries carry their name, model in own file
        (const / "transportProperties").write_text(
            head("transportProperties")
            + "transportModel  Newtonian;\n\nnu              nu [0 2 -1 0 0 0 0] 1.5e-05;\n"
            "rho             rho [1 -3 0 0 0 0 0] 1.2;\n"
        )
        (const / "turbulenceProperties").write_text(
            head("turbulenceProperties") + "simulationType  RASModel;\n"
        )
        (const / "RASProperties").write_text(
            head("RASProperties") + f"RASModel        {turbulence};\nturbulence      on;\n"
        )
    elif style.flavor == "foundation":
        (const / "physicalProperties").write_text(
            head("physicalProperties")
            + "viscosityModel  constant;\n\nnu              [0 2 -1 0 0 0 0] 1.5e-05;\n"
            "rho             1.2;\n"
        )
        (const / "momentumTransport").write_text(
            head("momentumTransport")
            + f"simulationType  RAS;\n\nRAS\n{{\n    model           {turbulence};\n"
            "    turbulence      on;\n}\n"
        )
    else:
        (const / "transportProperties").write_text(
            head("transportProperties")
            + "#inputMode merge\nUref            1.5;\ntransportModel  Newtonian;\n"
            "nu              1.5e-05;\nrho             1.2;\n"
            "inletVelocity   ($Uref 0 0);\n"
        )
        (const / "turbulenceProperties").write_text(
            head("turbulenceProperties") + "simulationType  RAS;\n\nRAS\n{\n    // model choice\n"
            f"    RASModel        {turbulence};\n"
            "    turbulence      on;\n    printCoeffs     on;\n}\n"
        )


@dataclass
class Case:
    root: Path
    mesh: PolyMesh
    times: list[str]
    style: Style
    cells: np.ndarray
    points: np.ndarray
    parts: list[PolyMesh] = field(default_factory=list)

    def exact(self, name: str, t: float) -> np.ndarray:
        return analytic(self.mesh.cell_centres(), t)[name].reshape(self.mesh.n_cells, -1)


def write_case(
    root: Path,
    *,
    shape: tuple[int, int, int] = (4, 3, 2),
    tets: bool = False,
    two_d: bool = False,
    times: Sequence[str] = ("0", "0.5", "1"),
    style: Style | None = None,
    application: str = "pimpleFoam",
    steady: bool = False,
    decompose: int = 0,
    keep_serial: bool = False,
    addressing: bool = True,
    fields: Sequence[str] = ("p", "U", "T"),
    turbulence: str = "kOmegaSST",
    region: str | None = None,
) -> Case:
    style = style or Style()
    size = (2.0, 1.5, 1.0)
    if tets:
        points, cells = tet_grid(*shape)
        points = points * np.array(size)
    else:
        points, cells = hex_grid(*shape, size=size)
    types = {"inlet": "patch", "outlet": "patch", "walls": "wall"}
    if two_d:
        types["frontAndBack"] = "empty"
    patch_of = box_patch(size, two_d)
    mesh = build_polymesh(points, cells, patch_of, types)
    write_dicts(root, application=application, steady=steady, style=style, turbulence=turbulence)
    (root / "case.foam").write_text("")

    def populate(part: Path, m: PolyMesh) -> None:
        const = part / "constant" / region if region else part / "constant"
        write_mesh(const / "polyMesh", m, style)
        for tname in times:
            tdir = part / tname / region if region else part / tname
            for name in fields:
                write_field(
                    tdir / name,
                    name,
                    m,
                    float(tname),
                    style,
                    uniform=300.0 if name == "T" else None,
                )

    case = Case(root=root, mesh=mesh, times=list(times), style=style, cells=cells, points=points)
    if decompose:
        centres = mesh.cell_centres()
        proc_of = np.minimum((centres[:, 0] / size[0] * decompose).astype(int), decompose - 1)
        for me in range(decompose):
            ids = [int(c) for c in np.nonzero(proc_of == me)[0]]
            part = build_polymesh(
                points, cells, patch_of, types, cell_ids=ids, proc_of=proc_of, me=me
            )
            if not addressing:
                part.cell_addr = None
            case.parts.append(part)
            populate(root / f"processor{me}", part)
    if not decompose or keep_serial:
        populate(root, mesh)
    return case


def by_centre(sample: Any) -> np.ndarray:
    """Cell centres of a mesh sample (mean of the cell's points)."""
    cells = np.asarray(sample.cells)
    pts = np.asarray(sample.points, dtype=np.float64)
    mask = cells >= 0
    summed = (pts[np.clip(cells, 0, None)] * mask[..., None]).sum(axis=1)
    return summed / mask.sum(axis=1)[:, None]


def rows_sorted(*arrays: np.ndarray) -> np.ndarray:
    a = np.concatenate(
        [np.asarray(x, dtype=np.float64).reshape(len(x), -1) for x in arrays], axis=1
    )
    return a[np.lexsort(np.round(a, 5).T[::-1])]


def connector() -> Any:
    """The connector, typed ``Any`` so the tests can index what it yields freely."""
    return OpenFoamConnector()


def check(*args: Any, **kwargs: Any) -> Any:
    return check_connector(*args, **kwargs)


# -- parser ---------------------------------------------------------------------------------


def test_parser_dictionary_features(tmp_path: Path) -> None:
    (tmp_path / "initialConditions").write_text("flowVelocity (20 0 0);\npressure 0;\n")
    path = tmp_path / "dict"
    path.write_text(
        BANNERS["foundation"]
        + """FoamFile { version 2.0; format ascii; class dictionary; object dict; }
// line comment
#include "initialConditions"
#includeEtc "caseDicts/setConstraintTypes"
#inputMode merge
/* block
   comment */
application simpleFoam;
nu nu [0 2 -1 0 0 0 0] 1e-05;
internalField uniform $flowVelocity;
weights 3(0.25 0.5 0.25);
flags 4{1};
empty 0();
names (inlet "out let" div(phi,U));
blocks ( hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1) );
RAS { RASModel kEpsilon; coeffs { Cmu 0.09; } }
RAS { turbulence on; }
"(left|right)Wall" { type wall; }
code #{ scalar x = 1; #};
noSemicolon { a 1 }
"""
    )
    doc = of.parse_foam_file(path)
    b = doc.body
    assert doc.version == "11" and doc.distribution == "Foundation (openfoam.org)"
    assert doc.cls == "dictionary" and b["application"] == "simpleFoam"
    assert b["flowVelocity"] == [20, 0, 0] and b["internalField"] == ("uniform", [20, 0, 0])
    assert b["nu"][0] == "nu" and b["nu"][1].unit == "m^2/s" and b["nu"][2] == 1e-5
    assert b["weights"] == [0.25, 0.5, 0.25] and b["flags"] == of.FoamUniform(4, 1)
    assert b["empty"] == [] and b["names"] == ["inlet", "out let", "div(phi,U)"]
    assert b["blocks"][:2] == ["hex", [0, 1, 2, 3, 4, 5, 6, 7]]
    assert b["RAS"] == {"RASModel": "kEpsilon", "coeffs": {"Cmu": 0.09}, "turbulence": "on"}
    assert b.patterns == ["(left|right)Wall"] and b["(left|right)Wall"]["type"] == "wall"
    assert "scalar x" in b["code"] and b["noSemicolon"] == {"a": 1}

    for text, message in [
        ("FoamFile { format ascii; class dictionary; }\nRAS { model k;", "not closed by '}'"),
        ("FoamFile { format ascii; class dictionary; }\nvalues (1 2 3", r"not closed by '\)'"),
        ("FoamFile { format ascii; class dictionary; }\n} stray;", "unbalanced"),
    ]:
        path.write_text(text)
        with pytest.raises(ConnectorError, match=message) as err:
            of.parse_foam_file(path)
        assert err.value.hint


@pytest.mark.parametrize(
    "style",
    [
        Style(),
        Style(gz=True),
        Style(fmt="binary"),
        Style(fmt="binary", gz=True),
        Style(fmt="binary", label=64),
        Style(fmt="binary", scalar=32),
    ],
    ids=["ascii", "ascii-gz", "binary", "binary-gz", "binary-label64", "binary-scalar32"],
)
def test_parser_reads_mesh_and_fields_in_every_encoding(tmp_path: Path, style: Style) -> None:
    case = write_case(
        tmp_path / "case", tets=True, shape=(2, 2, 1), style=style, fields=("p", "U", "T", "R")
    )
    mesh_dir = tmp_path / "case" / "constant" / "polyMesh"

    def f(name: str) -> Path:
        found = of.resolve_file(mesh_dir / name)
        assert found is not None
        return found

    tol = 1e-6 if style.scalar == 32 else 1e-14
    np.testing.assert_allclose(of.read_points(f("points")), case.mesh.points, rtol=tol, atol=tol)
    np.testing.assert_array_equal(of.read_labels(f("owner")), case.mesh.owner)
    np.testing.assert_array_equal(of.read_labels(f("neighbour")), case.mesh.neighbour)
    offsets, labels = of.read_faces(f("faces"))
    assert len(offsets) == len(case.mesh.faces) + 1
    assert [
        tuple(labels[a:b]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ] == case.mesh.faces
    patches = of.read_boundary(f("boundary"))
    assert [(p.name, p.type, p.n_faces, p.start_face) for p in patches] == case.mesh.patches
    assert patches[2].groups == ["wall"]
    counts = of.mesh_counts(mesh_dir)
    assert (counts.n_points, counts.n_cells, counts.n_faces, counts.n_internal_faces) == (
        len(case.mesh.points),
        case.mesh.n_cells,
        len(case.mesh.faces),
        len(case.mesh.neighbour),
    )

    tdir = tmp_path / "case" / "0.5"
    for name, comps, unit in [("p", 1, "m^2/s^2"), ("U", 3, "m/s"), ("R", 6, "m^2/s^2")]:
        path = of.resolve_file(tdir / name)
        assert path is not None
        fld = of.read_field(path)
        assert (fld.location, fld.components, fld.units) == ("cell", comps, unit)
        np.testing.assert_allclose(
            fld.values(case.mesh.n_cells), case.exact(name, 0.5), rtol=tol, atol=tol
        )
        lazy = of.read_field(path, load=False)
        assert isinstance(lazy.internal, of.FoamSkipped) or case.mesh.n_cells < 64
        assert set(lazy.boundary) == {"inlet", "outlet", "walls"}
    t_path = of.resolve_file(tdir / "T")
    assert t_path is not None
    temperature = of.read_field(t_path)
    assert temperature.internal == ("uniform", 300) and temperature.units == "K"
    assert (temperature.values(case.mesh.n_cells) == 300.0).all()
    assert of.read_field(of.resolve_file(tdir / "U"), load=False).boundary["inlet"]["value"] == (  # type: ignore[arg-type]
        "uniform",
        [1.5, 0, 0],
    )


def test_mesh_counts_without_header_note(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", style=Style(note=False), times=("0",))
    counts = of.mesh_counts(tmp_path / "case" / "constant" / "polyMesh")
    assert (counts.n_points, counts.n_cells, counts.n_faces) == (60, 24, len(case.mesh.faces))


# -- connector: serial cases --------------------------------------------------------------------


def test_detect_layouts(tmp_path: Path) -> None:
    c = connector()
    case = write_case(tmp_path / "case", times=("0",))
    for target in (case.root, case.root / "case.foam", case.root / "system" / "controlDict"):
        det = c.detect(target)
        assert det is not None and det.confidence == 1.0 and det.format == "openfoam"
    assert fc.detect(case.root)[0].connector_id == "openfoam"
    assert c.detect(tmp_path) is None and c.detect(tmp_path / "nope") is None
    assert c.detect(case.root / "0" / "U") is None
    (tmp_path / "stray.foam").write_text("")
    assert c.detect(tmp_path / "stray.foam") is None
    shutil.rmtree(case.root / "system")
    det = c.detect(case.root)
    assert det is not None and det.confidence == 0.9
    only_dict = tmp_path / "nomesh"
    (only_dict / "system").mkdir(parents=True)
    (only_dict / "system" / "controlDict").write_text("application icoFoam;\n")
    det = c.detect(only_dict)
    assert det is not None and det.confidence == 0.5 and det.needs
    with pytest.raises(ConnectorError, match="no mesh") as err:
        c.describe(only_dict)
    assert "blockMesh" in (err.value.hint or "")
    with pytest.raises(UnsupportedSourceError):
        c.describe(tmp_path / "nope")


def test_transient_hex_case_ascii(tmp_path: Path) -> None:
    case = write_case(tmp_path / "cavity", fields=("p", "U", "T", "R", "gradU"))
    (case.root / "0.orig").mkdir()
    shutil.copy(case.root / "0" / "U", case.root / "0.orig" / "U")
    c = connector()
    rep = check(c, case.root, tmp_path / "out")
    d = rep.description
    assert d.solver == "OpenFOAM" and d.solver_version == "v2312" and d.domain == "cfd"
    assert d.physics["distribution"] == "ESI (openfoam.com)"
    assert d.physics["application"] == "pimpleFoam" and d.physics["steady"] is False
    assert d.physics["simulation_type"] == "RAS" and d.physics["turbulence_model"] == "kOmegaSST"
    assert (
        d.physics["transport_model"] == "Newtonian" and d.physics["pressure"] == "kinematic (p/rho)"
    )
    assert d.physics["compressible"] is False
    assert (
        d.parameters["nu"] == 1.5e-5 and d.parameters["rho"] == 1.2 and d.parameters["Uref"] == 1.5
    )
    assert d.metadata["run"]["deltaT"] == 0.005 and d.metadata["run"]["writeFormat"] == "ascii"
    assert (
        d.time.values == [0.0, 0.5, 1.0] and d.time.quantity == "time" and d.temporal == "transient"
    )
    assert d.units["length"] == "m"

    assert [(z.name, z.kind, z.role, z.bc_type, z.default, z.n_cells) for z in d.zones] == [
        ("internalMesh", "mesh", "volume", None, True, 24),
        ("patch/inlet", "surface", "boundary", "patch", False, 6),
        ("patch/outlet", "surface", "boundary", "patch", False, 6),
        ("patch/walls", "surface", "boundary", "wall", False, 40),
    ]
    assert d.zone("internalMesh").n_points == 60
    fields = {f.name: (f.location, f.components, f.units) for f in d.fields}
    assert fields == {
        "p": ("cell", 1, "m^2/s^2"),
        "U": ("cell", 3, "m/s"),
        "T": ("cell", 1, "K"),
        "R": ("cell", 6, "m^2/s^2"),
        "gradU": ("cell", 9, "1/s"),
    }
    bc = d.boundary_conditions
    assert (
        bc["inlet"]["type"] == "patch"
        and bc["walls"]["type"] == "wall"
        and bc["walls"]["n_faces"] == 40
    )
    assert bc["inlet"]["U"] == {"type": "fixedValue", "value": [1.5, 0, 0]}
    assert bc["inlet"]["p"] == {"type": "zeroGradient"} and bc["walls"]["U"] == {"type": "noSlip"}
    assert bc["outlet"]["p"] == {"type": "fixedValue", "value": 0}
    assert bc["outlet"]["U"]["type"] == "calculated" and bc["walls"]["groups"] == ["wall"]

    steps = rep.samples["cavity"]
    assert [s.time for s in steps] == [0.0, 0.5, 1.0] and [s.time_index for s in steps] == [0, 1, 2]
    first = steps[0]
    assert first.kind == "mesh" and first.zone == "internalMesh"
    assert steps[1].points is None and steps[1].cells is None
    # blockMesh-style numbering survives: same points and VTK hexahedra as the generator
    np.testing.assert_allclose(first.points, case.points, atol=1e-6)
    assert (np.asarray(first.cell_types) == 12).all() and len(first.cells) == 24
    np.testing.assert_allclose(by_centre(first), case.mesh.cell_centres(), atol=1e-6)
    assert [set(row) for row in np.asarray(first.cells)] == [set(row) for row in case.cells]
    from vtkmodules.vtkCommonDataModel import vtkHexahedron

    # VTK node ordering: positive Jacobian for every hexahedron
    for row in np.asarray(first.cells):
        p = case.points[row]
        assert np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[4] - p[0]) > 0
    assert vtkHexahedron().GetNumberOfPoints() == first.cells.shape[1]
    for s in steps:
        assert s.time is not None
        for name in ("p", "U", "R", "gradU"):
            f = s.fields[name]
            assert f.location == "cell"
            np.testing.assert_allclose(f.data, case.exact(name, s.time), rtol=2e-6, atol=1e-6)
        assert (s.fields["T"].data == 300.0).all()
        assert s.fields["U"].units == "m/s" and s.fields["p"].units == "m^2/s^2"
        assert s.fields["R"].component_names == ["xx", "xy", "xz", "yy", "yz", "zz"]
    assert first.metadata["solver"] == "OpenFOAM" and first.metadata["parameters"]["nu"] == 1.5e-5
    assert first.metadata["boundary_conditions"]["inlet"]["U"]["value"] == [1.5, 0, 0]

    # selections
    last = list(c.read(case.root, times="last", fields=["U"]))
    assert len(last) == 1 and last[0].time == 1.0 and list(last[0].fields) == ["U"]
    assert last[0].points is not None
    some = list(c.read(case.root, times=[0.0, 0.9], fields=["p*", "!p"]))
    assert [s.time for s in some] == [0.0, 1.0] and all(not s.fields for s in some)
    with pytest.raises(SelectionError, match="patch/nope") as err:
        list(c.read(case.root, zones=["patch/nope"]))
    assert "patch/walls" in (err.value.hint or "")


def test_patch_zones_carry_face_values(tmp_path: Path) -> None:
    case = write_case(tmp_path / "duct", times=("0", "2"))
    c = connector()
    rep = check(c, case.root, tmp_path / "out", zones=["patch/*"], times="last")
    assert sorted(rep.samples) == ["duct__patch_inlet", "duct__patch_outlet", "duct__patch_walls"]
    mesh = case.mesh
    centres = mesh.face_centres()
    cells = mesh.cell_centres()
    for name, ptype, n, start in mesh.patches:
        s = rep.first(f"duct__patch_{name}")
        assert s.kind == "surface" and s.zone == f"patch/{name}" and s.time == 2.0
        assert s.metadata["bc_type"] == ptype and list(s.metadata["boundary_conditions"]) == [name]
        assert len(s.faces) == 2 * n  # quads fan-triangulated
        parent = np.asarray(s.static["face_parent"].data)[:, 0]
        tri = np.asarray(s.points, dtype=np.float64)[np.asarray(s.faces)]
        # both triangles of a quad lie in the patch face they came from
        quad = np.array([mesh.points[list(f)] for f in mesh.faces[start : start + n]])
        lo, hi = quad.min(axis=1)[parent] - 1e-6, quad.max(axis=1)[parent] + 1e-6
        assert ((tri.min(axis=1) >= lo) & (tri.max(axis=1) <= hi)).all()
        face = slice(start, start + n)
        exact = analytic(centres[face], 2.0)
        adjacent = analytic(cells[mesh.owner[face]], 2.0)
        u, p = np.asarray(s.fields["U"].data), np.asarray(s.fields["p"].data)[:, 0]
        if name == "inlet":
            np.testing.assert_allclose(u, np.tile(INLET_U, (2 * n, 1)), atol=1e-6)
            np.testing.assert_allclose(
                p, adjacent["p"][parent], rtol=1e-5, atol=1e-6
            )  # zeroGradient
        elif name == "outlet":
            np.testing.assert_allclose(u, exact["U"][parent], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(p, 0.0, atol=1e-7)
        assert s.fields["U"].location == "cell" and s.fields["U"].units == "m/s"
    both = list(c.read(case.root, zones=["internalMesh", "patch/walls"], times=0))
    assert [s.zone for s in both] == ["internalMesh", "patch/walls"]
    assert [s.id for s in both] == ["duct", "duct__patch_walls"] or len({s.id for s in both}) == 2


def test_point_data_option_keeps_cell_values(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("1",))
    rep = check(connector(), case.root, tmp_path / "out", options={"point_data": True})
    s = rep.first()
    assert s.time is None and rep.description.temporal == "steady"
    assert s.fields["U"].location == "cell" and s.fields["U__point"].location == "point"
    np.testing.assert_allclose(s.fields["U"].data, case.exact("U", 1.0), rtol=2e-6)
    assert s.fields["U__point"].shape == (60, 3) and s.fields["U__point"].units == "m/s"


@pytest.mark.parametrize(
    "style",
    [
        Style(gz=True),
        Style(fmt="binary"),
        Style(fmt="binary", gz=True, flavor="foundation"),
        Style(fmt="binary", label=64),
        Style(fmt="binary", scalar=32, note=False),
    ],
    ids=["ascii-gz", "binary", "binary-gz", "binary-label64", "binary-scalar32"],
)
def test_encodings_match_ascii(tmp_path: Path, style: Style) -> None:
    ref_case = write_case(tmp_path / "a" / "case", fields=("p", "U", "T", "R"))
    case = write_case(tmp_path / "b" / "case", style=style, fields=("p", "U", "T", "R"))
    c = connector()
    ref = list(c.read(ref_case.root, zones=["*"]))
    rep = check(c, case.root, tmp_path / "out", zones=["*"])
    got = [s for group in rep.samples.values() for s in group]
    assert rep.description.metadata["run"]["writeFormat"] == style.fmt
    assert [(s.zone, s.time) for s in sorted(got, key=lambda s: (s.time, s.zone))] == [
        (s.zone, s.time) for s in sorted(ref, key=lambda s: (s.time, s.zone))
    ]
    tol = 1e-5 if style.scalar == 32 else 1e-7
    index = {(s.zone, s.time): s for s in got}
    for r in ref:
        s = index[(r.zone, r.time)]
        if r.points is not None:
            np.testing.assert_allclose(s.points, r.points, atol=tol)
            if r.kind == "mesh":
                np.testing.assert_array_equal(s.cells, r.cells)
            else:
                np.testing.assert_array_equal(s.faces, r.faces)
        assert sorted(s.fields) == sorted(r.fields)
        for name, f in r.fields.items():
            np.testing.assert_allclose(s.fields[name].data, f.data, rtol=tol, atol=tol)
    if style.flavor == "foundation":
        d = rep.description
        assert d.solver_version == "11" and d.physics["distribution"] == "Foundation (openfoam.org)"
        assert d.physics["turbulence_model"] == "kOmegaSST" and d.parameters["nu"] == 1.5e-5


def test_tetrahedral_two_dimensional_and_foam_extend(tmp_path: Path) -> None:
    c = connector()
    tet = write_case(
        tmp_path / "tets", tets=True, shape=(2, 2, 2), times=("3",), style=Style(flavor="extend")
    )
    rep = check(c, tet.root, tmp_path / "out1")
    s = rep.first()
    assert (np.asarray(s.cell_types) == 10).all() and len(s.cells) == 48 and s.cells.shape[1] == 4
    order = np.lexsort(np.round(by_centre(s), 6).T[::-1])
    want = np.lexsort(np.round(tet.mesh.cell_centres(), 6).T[::-1])
    np.testing.assert_array_equal(order, want)  # same cell numbering as the case
    np.testing.assert_allclose(s.fields["U"].data, tet.exact("U", 3.0), rtol=2e-6, atol=1e-6)
    p = np.asarray(s.points, dtype=np.float64)[np.asarray(s.cells)]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    assert (vol > 0).all()  # VTK tetra orientation
    d = rep.description
    assert d.solver_version == "4.1" and d.physics["distribution"] == "foam-extend"
    assert d.physics["simulation_type"] == "RAS" and d.physics["turbulence_model"] == "kOmegaSST"
    assert d.parameters["nu"] == 1.5e-5 and d.parameters["rho"] == 1.2

    flat = write_case(tmp_path / "flat", shape=(5, 4, 1), two_d=True, times=("0", "1"))
    rep = check(c, flat.root, tmp_path / "out2")
    d = rep.description
    assert d.physics["two_dimensional"] is True
    empty = d.zone("patch/frontAndBack")
    assert empty.bc_type == "empty" and empty.default is False and empty.n_cells == 40
    assert sorted(rep.samples) == ["flat"]  # only the internal mesh is read by default
    assert d.boundary_conditions["frontAndBack"]["U"] == {"type": "empty"}
    assert d.boundary_conditions["frontAndBack"]["groups"] == ["empty"]


def test_steady_solver_iterations(tmp_path: Path) -> None:
    case = write_case(
        tmp_path / "rans",
        times=("0", "500", "1000"),
        application="simpleFoam",
        steady=True,
        turbulence="kEpsilon",
    )
    c = connector()
    rep = check(c, case.root, tmp_path / "out")
    d = rep.description
    assert d.physics["steady"] is True and d.physics["ddt_scheme"] == "steadyState"
    assert d.time.values == [1000.0] and d.time.quantity == "iteration" and d.temporal == "steady"
    assert d.metadata["iterations"] == [0.0, 500.0, 1000.0]
    assert any("only the last one (1000)" in w for w in d.warnings)
    (s,) = rep.samples["rans"]
    assert s.time is None
    np.testing.assert_allclose(s.fields["p"].data, case.exact("p", 1000.0), rtol=2e-6, atol=1e-6)

    rep = check(c, case.root, tmp_path / "out2", options={"iterations": "all"})
    assert rep.description.time.values == [0.0, 500.0, 1000.0]
    assert rep.description.time.quantity == "iteration"
    assert [s.time for s in rep.samples["rans"]] == [0.0, 500.0, 1000.0]
    picked = list(c.read(case.root, times=1, options={"iterations": "all"}))
    assert [s.time for s in picked] == [500.0]
    with pytest.raises(ConnectorError, match="iterations"):
        c.describe(case.root, options={"iterations": "some"})
    # without fvSchemes the solver name decides
    (case.root / "system" / "fvSchemes").unlink()
    assert c.describe(case.root).time.quantity == "iteration"


def test_mesh_only_case_and_moving_points(tmp_path: Path) -> None:
    c = connector()
    bare = write_case(tmp_path / "bare", times=())
    rep = check(c, bare.root, tmp_path / "out1")
    assert rep.description.time.values == [] and not rep.description.fields
    assert any("no time directories" in w for w in rep.description.warnings)
    assert len(rep.first().cells) == 24 and not rep.first().fields

    case = write_case(tmp_path / "moving", times=("0", "1", "2"))
    moved = case.mesh.points + np.array([0.0, 0.0, 0.25])
    head = header("vectorField", "points", "ascii", "esi")
    put(case.root / "1" / "polyMesh" / "points", head + raw_list(moved, Style(), "vector"), Style())
    rep = check(c, case.root, tmp_path / "out2")
    s0, s1, s2 = rep.samples["moving"]
    np.testing.assert_allclose(s1.points, moved, atol=1e-6)
    assert s1.cells is None and s2.points is None and s0.cells is not None
    store = open_store(rep.results["zarr"].out / "samples" / "moving" / "mesh.zarr")
    assert store["points_t"].shape == (3, 60, 3)
    np.testing.assert_allclose(store["points_t"][2], moved, atol=1e-6)


def test_field_vtk_cannot_parse_is_read_by_the_own_parser(tmp_path: Path, capfd: Any) -> None:
    case = write_case(tmp_path / "case", times=("0", "1"), decompose=2, keep_serial=True)
    for part in (case.root, case.root / "processor0", case.root / "processor1"):
        for tname in case.times:
            # spherical tensors without the '(v)' wrapping: VTK rejects the file, the list is fine
            text = (
                (part / tname / "p")
                .read_text()
                .replace("volScalarField", "volSphericalTensorField")
            )
            text = text.replace("List<scalar>", "List<sphericalTensor>")
            (part / tname / "S").write_text(text.replace("object      p;", "object      S;"))
    c = connector()
    for options in (None, {"decomposed": "true"}):
        rep = check(
            c, case.root, tmp_path / "out", zones=["internalMesh", "patch/inlet"], options=options
        )
        assert ("S", "cell", 1) in [
            (f.name, f.location, f.components) for f in rep.description.fields
        ]
        steps = rep.samples["case__internalMesh"]
        for s in steps:
            np.testing.assert_allclose(
                s.fields["S"].data, case.exact("p", s.time), rtol=2e-6, atol=1e-6
            )
            assert s.fields["S"].units == "m^2/s^2"
        assert "S" not in rep.samples["case__patch_inlet"][0].fields  # only the volume is recovered
    out = capfd.readouterr()
    assert "ERR|" not in out.err and "ERR|" not in out.out  # VTK's complaints stay off the console


def test_string_flags_from_the_cli(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("1",), decompose=2, keep_serial=True)
    c = connector()
    assert c.describe(case.root, options={"decomposed": "true"}).format == "openfoam-decomposed"
    assert c.describe(case.root, options={"decomposed": "false"}).format == "openfoam"
    (s,) = c.read(case.root, options={"point_data": "false"})
    assert "U__point" not in s.fields
    (s,) = c.read(case.root, options={"point_data": "true", "decomposed": "true"})
    assert s.fields["U__point"].shape == (60, 3) and s.fields["U"].shape == (24, 3)


def test_fields_missing_in_a_step_are_nan_filled(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("0", "1", "2"), fields=("p", "U", "T", "R"))
    (case.root / "2" / "R").unlink()
    rep = check(connector(), case.root, tmp_path / "out")
    steps = rep.samples["case"]
    assert np.isnan(steps[2].fields["R"].data).all() and steps[2].fields["R"].shape == (24, 6)
    np.testing.assert_allclose(
        steps[1].fields["R"].data, case.exact("R", 1.0), rtol=2e-6, atol=1e-6
    )


def test_multi_region_case(tmp_path: Path) -> None:
    case = write_case(tmp_path / "cht", region="fluid", times=("0", "1"))
    write_case(tmp_path / "cht", region="solid", times=("0", "1"), shape=(2, 2, 2), fields=("T",))
    c = connector()
    det = c.detect(case.root)
    assert det is not None and det.confidence == 1.0 and "regions: fluid, solid" in det.notes
    with pytest.raises(ConnectorError, match="multi-region") as err:
        c.describe(case.root)
    assert "region" in (err.value.hint or "")
    with pytest.raises(SelectionError, match="gas"):
        c.describe(case.root, options={"region": "gas"})
    rep = check(c, case.root, tmp_path / "out", options={"region": "fluid"}, zones=["*"])
    assert rep.description.metadata["region"] == "fluid"
    steps = rep.samples["cht__internalMesh"]
    assert len(steps[0].cells) == 24
    np.testing.assert_allclose(
        steps[1].fields["U"].data, case.exact("U", 1.0), rtol=2e-6, atol=1e-6
    )
    assert "cht__patch_walls" in rep.samples
    solid = list(c.read(case.root, options={"region": "solid"}, times="last"))
    assert len(solid[0].cells) == 8 and list(solid[0].fields) == ["T"]


# -- decomposed cases ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "style", [Style(), Style(fmt="binary", gz=True)], ids=["ascii", "binary-gz"]
)
def test_decomposed_equals_serial(tmp_path: Path, style: Style) -> None:
    serial_case = write_case(tmp_path / "s" / "case", shape=(6, 3, 2), style=style)
    case = write_case(tmp_path / "d" / "case", shape=(6, 3, 2), style=style, decompose=2)
    assert not (case.root / "constant" / "polyMesh").exists()
    assert [m.n_cells for m in case.parts] == [18, 18]
    c = connector()
    det = c.detect(case.root)
    assert det is not None and det.confidence == 1.0 and det.format == "openfoam-decomposed"
    rep = check(c, case.root, tmp_path / "out", zones=["*"])
    d = rep.description
    assert d.format == "openfoam-decomposed" and d.metadata["processors"] == 2
    assert [z.name for z in d.zones] == [
        "internalMesh",
        "patch/inlet",
        "patch/outlet",
        "patch/walls",
    ]
    assert d.zone("internalMesh").n_cells == 36 and d.zone("patch/walls").n_cells == 60
    assert (
        "procBoundary0to1" not in d.boundary_conditions
        and d.boundary_conditions["walls"]["n_faces"] == 60
    )
    assert d.boundary_conditions["inlet"]["U"]["value"] == [1.5, 0, 0]
    assert not any("welded" in w for w in d.warnings)

    ref = {(s.zone, s.time): s for s in c.read(serial_case.root, zones=["*"])}
    got = {(s.zone, s.time): s for group in rep.samples.values() for s in group}
    assert set(got) == set(ref)
    assert got[("internalMesh", 0.0)].metadata["decomposed_merge"] == "addressing"
    for key, r in ref.items():
        s = got[key]
        if key[0] == "internalMesh":
            # the addressing maps restore the undecomposed numbering exactly
            if r.points is not None:
                np.testing.assert_allclose(s.points, r.points, atol=1e-7)
                # same nodes per cell; the hexahedron may start at another corner because
                # the processor mesh orders its faces differently
                np.testing.assert_array_equal(np.sort(s.cells, axis=1), np.sort(r.cells, axis=1))
                np.testing.assert_array_equal(s.cell_types, r.cell_types)
                p = np.asarray(s.points, dtype=np.float64)[np.asarray(s.cells)]
                jac = np.einsum(
                    "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]), p[:, 4] - p[:, 0]
                )
                assert (jac > 0).all()
            for name, f in r.fields.items():
                np.testing.assert_allclose(s.fields[name].data, f.data, rtol=1e-6, atol=1e-7)
        else:
            # patches are merged geometrically: same faces and values up to ordering
            if r.points is not None:
                assert s.points.shape == r.points.shape and s.faces.shape == r.faces.shape
                np.testing.assert_allclose(rows_sorted(s.points), rows_sorted(r.points), atol=1e-6)
            sp, rp = got[(key[0], 0.0)], ref[(key[0], 0.0)]
            cs = np.asarray(sp.points, dtype=np.float64)[np.asarray(sp.faces)].mean(axis=1)
            cr = np.asarray(rp.points, dtype=np.float64)[np.asarray(rp.faces)].mean(axis=1)
            for name, f in r.fields.items():
                np.testing.assert_allclose(
                    rows_sorted(cs, s.fields[name].data),
                    rows_sorted(cr, f.data),
                    rtol=1e-5,
                    atol=1e-6,
                )


def test_decomposed_without_addressing_and_selection_option(tmp_path: Path) -> None:
    case = write_case(
        tmp_path / "case", shape=(6, 3, 2), decompose=3, addressing=False, keep_serial=True
    )
    c = connector()
    # reconstructed and decomposed data side by side: same number of time dirs -> serial wins
    assert c.describe(case.root).format == "openfoam"
    rep = check(c, case.root, tmp_path / "out", options={"decomposed": True}, times="last")
    assert rep.description.metadata["processors"] == 3
    assert any("welded" in w for w in rep.description.warnings)
    s = rep.first()
    assert s.metadata["decomposed_merge"] == "processor-patch weld"
    assert len(s.cells) == 36 and len(s.points) == 84  # coincident processor points welded
    np.testing.assert_allclose(
        rows_sorted(by_centre(s), s.fields["U"].data),
        rows_sorted(case.mesh.cell_centres(), case.exact("U", 1.0)),
        rtol=1e-5,
        atol=1e-6,
    )
    shutil.rmtree(case.root / "0.5")
    shutil.rmtree(case.root / "1")
    assert c.describe(case.root).format == "openfoam-decomposed"  # results only in processor*/
    assert c.describe(case.root, options={"decomposed": False}).time.values == [0.0]


def test_collated_decomposition_is_reported(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("0",))
    shutil.rmtree(case.root / "constant" / "polyMesh")
    (case.root / "processors4" / "constant" / "polyMesh").mkdir(parents=True)
    c = connector()
    det = c.detect(case.root)
    assert det is not None and det.needs and "reconstructPar" in det.needs[0]
    for call in (lambda: c.describe(case.root), lambda: list(c.read(case.root))):
        with pytest.raises(ConnectorError, match="collated") as err:
            call()
        assert "reconstructPar" in (err.value.hint or "")


# -- malformed input ----------------------------------------------------------------------------


def test_missing_owner(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("0",))
    (case.root / "constant" / "polyMesh" / "owner").unlink()
    c = connector()
    for call in (lambda: c.describe(case.root), lambda: list(c.read(case.root))):
        with pytest.raises(ConnectorError, match="no 'owner' file") as err:
            call()
        assert "polyMesh" in (err.value.hint or "")


@pytest.mark.parametrize(
    "target", ["constant/polyMesh/points", "constant/polyMesh/faces", "1/U", "1/p"]
)
@pytest.mark.parametrize("fmt", ["binary", "ascii"])
def test_truncated_list(tmp_path: Path, target: str, fmt: str) -> None:
    case = write_case(tmp_path / "case", shape=(6, 4, 3), times=("0", "1"), style=Style(fmt=fmt))
    path = case.root / target
    data = path.read_bytes()
    marker = data.find(b"\n(", data.find(b"}")) + 2
    path.write_bytes(data[: marker + (len(data) - marker) // 3])
    with pytest.raises(ConnectorError) as err:
        list(connector().read(case.root))
    assert err.value.hint and path.name in err.value.message
    assert "truncated" in str(err.value) or "not closed" in str(err.value)


def test_truncated_gzip_and_wrong_size_field(tmp_path: Path) -> None:
    case = write_case(tmp_path / "gz", times=("0",), style=Style(gz=True))
    path = case.root / "constant" / "polyMesh" / "points.gz"
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ConnectorError) as err:
        list(connector().read(case.root))
    assert err.value.hint and "points.gz" in err.value.message

    other = write_case(tmp_path / "other", shape=(3, 3, 3), times=("0",))
    case = write_case(tmp_path / "mix", times=("0",))
    shutil.copy(other.root / "0" / "p", case.root / "0" / "p")
    with pytest.raises(ConnectorError, match="27 values but the mesh has 24 cells") as err:
        list(connector().read(case.root))
    assert err.value.hint


def test_unknown_field_class(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("0", "1"))
    for tname in case.times:
        orig = (case.root / tname / "p").read_text()
        text = orig.replace("volScalarField", "volFooField")
        (case.root / tname / "weird").write_text(
            text.replace("object      p;", "object      weird;")
        )
        flux = orig.replace("volScalarField", "surfaceScalarField").replace(
            "object      p;", "object      phi;"
        )
        (case.root / tname / "phi").write_text(flux)
    c = connector()
    with pytest.raises(ConnectorError, match="unsupported field class 'volFooField'") as err:
        of.read_field(case.root / "1" / "weird")
    assert "supported" in (err.value.hint or "")
    with pytest.raises(ConnectorError, match="'weird' has class 'volFooField'") as err:
        list(c.read(case.root, fields=["weird"]))
    assert err.value.hint
    with pytest.raises(ConnectorError, match="surfaceScalarField"):
        list(c.read(case.root, fields=["U", "phi"]))
    # not asked for by name: skipped, and describe says so
    rep = check(c, case.root, tmp_path / "out")
    d = rep.description
    assert d.metadata["skipped_fields"] == {"phi": "surfaceScalarField", "weird": "volFooField"}
    assert any("weird (volFooField)" in w for w in d.warnings) and not any(
        "phi" in w for w in d.warnings
    )
    assert sorted(rep.first().fields) == ["T", "U", "p"]
    assert sorted(f.name for f in d.fields) == ["T", "U", "p"]


def test_garbage_files_do_not_crash(tmp_path: Path) -> None:
    case = write_case(tmp_path / "case", times=("0",))
    c = connector()
    (case.root / "constant" / "polyMesh" / "boundary").write_bytes(b"\x00\x01garbage(((")
    with pytest.raises(ConnectorError) as err:
        c.describe(case.root)
    assert err.value.hint
    case = write_case(tmp_path / "case2", times=("0",))
    (case.root / "system" / "controlDict").write_text("application { broken")
    (case.root / "constant" / "transportProperties").write_bytes(b"nu (1 2")
    d = c.describe(case.root)  # broken side dictionaries only cost metadata
    assert len(d.warnings) >= 2 and d.zone("internalMesh").n_cells == 24
    assert c.detect(case.root / "system" / "controlDict") is not None


# -- real cases (network) -----------------------------------------------------------------------


def _unzip(archive: Path, name: str) -> Path:
    """Extract into the sample cache, refusing members that would leave the target directory."""
    dest = archive.parent / name
    if dest.is_dir():
        return dest
    tmp = archive.parent / (name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (tmp / member.filename).resolve()
            if not target.is_relative_to(tmp.resolve()):
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif (member.external_attr >> 16) & 0o170000 != 0o120000:  # no symlinks
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    tmp.rename(dest)
    return dest


PYVISTA_DATA = "https://github.com/pyvista/vtk-data/raw/master/Data/"
VTK_REF = "v9.3.0"


def _vtk_testing_case(case: str) -> Path:
    """One case of VTK's own OpenFOAM regression data (``Testing/Data/OpenFOAM`` in the VTK
    repository holds SHA-512 content links; the files come from VTK's ExternalData store)."""
    import hashlib
    import json
    from urllib.parse import quote

    dest = sample_cache() / "vtk-openfoam" / VTK_REF / case
    if (dest / ".complete").is_file():
        return dest
    base = f"Testing/Data/OpenFOAM/{case}"
    listing: list[dict[str, Any]] = []
    for page in range(1, 20):
        url = (
            "https://gitlab.kitware.com/api/v4/projects/vtk%2Fvtk/repository/tree"
            f"?path={quote(base)}&ref={VTK_REF}&per_page=100&recursive=true&page={page}"
        )
        chunk = json.loads(fetch_sample(url, name=f"tree-{case}-{page}.json").read_text())
        listing += chunk
        if len(chunk) < 100:
            break
    for entry in listing:
        rel = entry["path"][len(base) + 1 :]
        if entry["type"] != "blob" or not rel.endswith(".sha512") or "lagrangian" in rel:
            continue
        link = fetch_sample(
            f"https://gitlab.kitware.com/vtk/vtk/-/raw/{VTK_REF}/{quote(entry['path'])}",
            name=rel.replace("/", "__"),
        )
        digest = link.read_text().strip()
        assert len(digest) == 128 and all(ch in "0123456789abcdef" for ch in digest)
        blob = fetch_sample(f"https://www.vtk.org/files/ExternalData/SHA512/{digest}", name=digest)
        data = blob.read_bytes()
        assert hashlib.sha512(data).hexdigest() == digest
        target = (dest / rel[: -len(".sha512")]).resolve()
        assert target.is_relative_to(dest.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (dest / ".complete").write_text("ok")
    return dest


def _sorted_values(root: Path, part_glob: str, time: str, name: str, n: int) -> np.ndarray:
    """Field values straight from the files with the connector's own parser."""
    rows = []
    for part in sorted(root.glob(part_glob)) if part_glob else [root]:
        path = of.resolve_file(part / time / name)
        assert path is not None
        counts = of.mesh_counts(part / "constant" / "polyMesh")
        rows.append(of.read_field(path).values(counts.n_cells))
    out = np.concatenate(rows)
    assert len(out) == n
    return out


@pytest.mark.network
def test_real_cavity_openfoam_1_7(tmp_path: Path) -> None:
    """The classic lid-driven cavity tutorial as written by icoFoam (OpenFOAM 1.7.x, ASCII),
    from pyvista's example data (``pyvista.examples.download_cavity``)."""
    archive = fetch_sample(PYVISTA_DATA + "OpenFOAM.zip")
    root = _unzip(archive, "OpenFOAM") / "OpenFOAM" / "cavity"
    c = connector()
    assert fc.detect(root / "case.foam")[0].connector_id == "openfoam"
    rep = check(c, root / "case.foam", tmp_path / "out", zones=["*"])
    d = rep.description
    assert d.solver_version == "1.7.x" and d.physics["application"] == "icoFoam"
    assert d.physics["steady"] is False and d.physics["two_dimensional"] is True
    assert d.parameters["nu"] == 0.01 and d.physics["pressure"] == "kinematic (p/rho)"
    assert d.time.values == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert [(z.name, z.bc_type, z.n_cells) for z in d.zones] == [
        ("internalMesh", None, 400),
        ("patch/movingWall", "wall", 20),
        ("patch/fixedWalls", "wall", 60),
        ("patch/frontAndBack", "empty", 800),
    ]
    assert {f.name: (f.components, f.units) for f in d.fields} == {
        "U": (3, "m/s"),
        "p": (1, "m^2/s^2"),
    }
    assert d.metadata["skipped_fields"] == {"phi": "surfaceScalarField"}
    bc = d.boundary_conditions
    assert bc["movingWall"]["U"] == {"type": "fixedValue", "value": [1, 0, 0]}
    assert bc["fixedWalls"]["U"] == {"type": "fixedValue", "value": [0, 0, 0]}
    assert bc["fixedWalls"]["p"] == {"type": "zeroGradient"}

    steps = rep.samples["cavity__internalMesh"]
    assert [s.time for s in steps] == d.time.values
    first, last = steps[0], steps[-1]
    assert first.points.shape == (882, 3) and first.cells.shape == (400, 8)
    assert (np.asarray(first.cell_types) == 12).all()
    np.testing.assert_allclose(first.points, of.read_points(root / "constant/polyMesh/points"))
    assert not np.asarray(first.fields["U"].data).any()  # initial condition: fluid at rest
    # values equal the files as read by the independent parser
    u = np.asarray(last.fields["U"].data, dtype=np.float64)
    p = np.asarray(last.fields["p"].data, dtype=np.float64)[:, 0]
    np.testing.assert_allclose(u, _sorted_values(root, "", "2.5", "U", 400), rtol=1e-6)
    np.testing.assert_allclose(p, _sorted_values(root, "", "2.5", "p", 400)[:, 0], rtol=1e-6)
    # physics of the lid-driven cavity (lid at y = 0.1 moving in +x at 1 m/s)
    y = by_centre(first)[:, 1]
    assert np.abs(u[:, 2]).max() == 0.0 and np.abs(u).max() < 1.0
    assert u[y > 0.095, 0].mean() > 0.4  # dragged along under the lid
    assert u[(y > 0.02) & (y < 0.06), 0].mean() < -0.05  # return flow of the vortex
    assert abs(u[:, 1].mean()) < 1e-3  # no net vertical flux
    x = by_centre(first)[:, 0]
    top = y > 0.095
    assert p[top & (x > 0.09)].mean() > 1.0 > -1.0 > p[top & (x < 0.01)].mean()  # stagnation corner
    lid = rep.samples["cavity__patch_movingWall"][-1]
    assert lid.kind == "surface" and lid.metadata["bc_type"] == "wall"
    np.testing.assert_array_equal(lid.fields["U"].data, np.tile([1.0, 0.0, 0.0], (40, 1)))
    walls = rep.samples["cavity__patch_fixedWalls"][-1]
    assert not np.asarray(walls.fields["U"].data).any()

    # the two-cell binary case shipped next to it (VTK's 64-bit float regression file, no arch)
    tiny = root.parent / "64BitFloats"
    rep = check(c, tiny, tmp_path / "out2")
    s = rep.first()
    assert len(s.cells) == 2 and s.points.shape == (12, 3)
    np.testing.assert_allclose(s.fields["U"].data, _sorted_values(tiny, "", "0", "U", 2))


@pytest.mark.network
def test_real_tubes_simplefoam_gzip(tmp_path: Path) -> None:
    """Steady RANS (simpleFoam, kOmegaSST) on a mixed hex/prism/pyramid/tet mesh, gzip ASCII,
    OpenFOAM v2112: pyvista's ``download_openfoam_tubes`` (38 MB)."""
    archive = fetch_sample(
        "https://github.com/pyvista/data/raw/master/Data/fvm/turbo_incompressible/"
        "Turbo-Incompressible_3-Run_1-SOLUTION_FIELDS.zip"
    )
    root = _unzip(archive, "tubes")
    c = connector()
    rep = check(c, root, tmp_path / "out", zones=["internalMesh", "patch/face[14]"])
    d = rep.description
    assert d.physics["application"] == "simpleFoam" and d.physics["steady"] is True
    assert d.physics["turbulence_model"] == "kOmegaSST" and d.physics["simulation_type"] == "RAS"
    assert d.parameters["nu"] == 7.55e-06 and d.physics["distribution"] == "ESI (openfoam.com)"
    assert d.time.values == [1000.0] and d.time.quantity == "iteration" and d.temporal == "steady"
    assert d.metadata["iterations"] == [0.0, 1000.0]
    assert d.zone("internalMesh").n_cells == 400712 and d.zone("internalMesh").n_points == 163390
    assert len(d.zones) == 11 and d.zone("patch/face4").bc_type == "wall"
    assert d.boundary_conditions["face1"]["U"] == {"type": "fixedValue", "value": [0, 0, -100]}
    assert {f.name: f.units for f in d.fields} == {
        "U": "m/s",
        "k": "m^2/s^2",
        "nut": "m^2/s",
        "omega": "1/s",
        "p": "kg/m/s^2",
    }
    (s,) = rep.samples["tubes__internalMesh"]
    assert s.time is None and s.points.shape == (163390, 3)
    from frontier_connectors.meshutil import census

    assert census(s.cell_types) == {
        "tetra": 231839,
        "hexahedron": 71138,
        "wedge": 76077,
        "pyramid": 21658,
    }
    for name in ("p", "U", "k"):
        np.testing.assert_allclose(
            s.fields[name].data, _sorted_values(root, "", "1000", name, 400712), rtol=1e-6
        )
    (inlet,) = rep.samples["tubes__patch_face1"]
    u = np.asarray(inlet.fields["U"].data)
    np.testing.assert_array_equal(u, np.tile([0.0, 0.0, -100.0], (len(u), 1)))
    (wall,) = rep.samples["tubes__patch_face4"]
    assert not np.asarray(wall.fields["U"].data).any()
    assert (np.asarray(s.fields["k"].data) > 0).all() and (
        np.asarray(s.fields["nut"].data) >= 0
    ).all()
    both = list(c.read(root, zones=["internalMesh"], fields=["p"], options={"iterations": "all"}))
    assert [b.time for b in both] == [0.0, 1000.0] and not np.asarray(
        both[0].fields["p"].data
    ).any()


@pytest.mark.network
def test_real_square_bend_binary(tmp_path: Path) -> None:
    """rhoSimpleFoam ``squareBend`` tutorial written in binary by OpenFOAM v2012 (VTK's own
    regression data, 28 MB)."""
    root = _vtk_testing_case("squareBend")
    c = connector()
    rep = check(c, root, tmp_path / "out", zones=["*"])
    d = rep.description
    assert d.solver_version == "v2012" and d.physics["distribution"] == "ESI (openfoam.com)"
    assert (
        d.physics["application"] == "rhoSimpleFoam" and d.metadata["run"]["writeFormat"] == "binary"
    )
    assert d.time.values == [100.0] and d.metadata["iterations"] == [50.0, 100.0]
    assert [(z.name, z.bc_type, z.n_cells) for z in d.zones] == [
        ("internalMesh", None, 112000),
        ("patch/inlet", "patch", 400),
        ("patch/outlet", "patch", 400),
        ("patch/walls", "wall", 22400),
    ]
    assert {f.name: f.units for f in d.fields} == {
        "T": "K",
        "U": "m/s",
        "p": "kg/m/s^2",
        "rho": "kg/m^3",
    }
    assert d.physics["pressure"] == "static"
    assert d.boundary_conditions["outlet"]["p"] == {"type": "fixedValue", "value": 110000}
    assert d.boundary_conditions["inlet"]["U"]["type"] == "flowRateInletVelocity"
    assert d.boundary_conditions["inlet"]["U"]["value"] == {"nonuniform": 400}
    (s,) = rep.samples["squareBend__internalMesh"]
    assert (np.asarray(s.cell_types) == 12).all() and s.cells.shape == (112000, 8)
    mesh_dir = root / "constant" / "polyMesh"
    np.testing.assert_allclose(s.points, of.read_points(mesh_dir / "points"), rtol=1e-6)
    offsets, labels = of.read_faces(mesh_dir / "faces")
    assert len(offsets) - 1 == 347600 and labels.max() == len(s.points) - 1
    assert of.read_labels(mesh_dir / "owner").max() == 111999
    for name in ("T", "U", "p", "rho"):
        np.testing.assert_allclose(
            s.fields[name].data, _sorted_values(root, "", "100", name, 112000), rtol=1e-6
        )
    p, rho, temp = (np.asarray(s.fields[n].data, dtype=np.float64)[:, 0] for n in ("p", "rho", "T"))
    gas_constant = p / (rho * temp)  # perfect gas: one constant everywhere (air, ~287 J/kg/K)
    assert 280.0 < gas_constant.mean() < 295.0 and gas_constant.std() < 1.0
    (outlet,) = rep.samples["squareBend__patch_outlet"]
    assert (np.asarray(outlet.fields["p"].data) == 110000.0).all()
    (inlet,) = rep.samples["squareBend__patch_inlet"]
    assert (np.asarray(inlet.fields["T"].data) == 1000.0).all()
    (walls,) = rep.samples["squareBend__patch_walls"]
    assert not np.asarray(walls.fields["U"].data).any()


@pytest.mark.network
def test_real_mixer_ggi_decomposed_moving_mesh(tmp_path: Path) -> None:
    """foam-extend 4.1 ``mixerGgi`` on 4 processors (no reconstructed mesh, no addressing
    files, moving points, GGI sliding interface): VTK's own regression data."""
    root = _vtk_testing_case("mixerGgi")
    c = connector()
    det = c.detect(root)
    assert det is not None and det.format == "openfoam-decomposed" and det.confidence == 0.9
    rep = check(c, root, tmp_path / "out", zones=["*"])
    d = rep.description
    assert d.format == "openfoam-decomposed" and d.metadata["processors"] == 4
    assert d.solver_version == "4.1" and d.physics["distribution"] == "foam-extend"
    assert d.time.values == [0.0, 0.5] and d.zone("internalMesh").n_cells == 576
    assert {z.name: z.bc_type for z in d.zones if z.name != "internalMesh"} == {
        "patch/impellerWall": "wall",
        "patch/baffleWall": "wall",
        "patch/insideSlider": "ggi",
        "patch/outsideSlider": "ggi",
        "patch/defaultFaces": "empty",
    }
    assert any("welded" in w for w in d.warnings)
    s0, s1 = rep.samples["mixerGgi__internalMesh"]
    assert s0.metadata["decomposed_merge"] == "processor-patch weld"
    assert (np.asarray(s0.cell_types) == 12).all() and len(s0.cells) == 576
    # the rotor moved: new points, same count and topology
    assert s1.points is not None and s1.cells is None and s1.points.shape == s0.points.shape
    assert np.abs(np.asarray(s1.points) - np.asarray(s0.points)).max() > 1e-3
    n_local = sum(of.mesh_counts(p / "constant/polyMesh").n_points for p in root.glob("processor*"))
    assert len(s0.points) < n_local  # processor seams are welded
    # foam-extend keeps the whole GGI zones on every processor, so some points serve no cell
    assert n_local == 1888 and len(s0.points) == 1784
    assert len(np.unique(np.asarray(s0.cells))) == 1368
    for sid in rep.sample_ids:  # every zone keeps its point count while the mesh moves
        a, b = rep.samples[sid]
        assert b.points is not None and b.points.shape == a.points.shape and b.faces is None
    for name in ("U", "p"):
        want = _sorted_values(root, "processor*", "0.5", name, 576)
        np.testing.assert_allclose(s1.fields[name].data, want, rtol=1e-6, atol=1e-9)
    assert np.isnan(s0.fields["rAU"].data).all() if "rAU" in s0.fields else True
    baffle = rep.samples["mixerGgi__patch_baffleWall"][-1]
    assert not np.asarray(baffle.fields["U"].data).any()
    impeller = rep.samples["mixerGgi__patch_impellerWall"][-1]
    assert np.abs(np.asarray(impeller.fields["U"].data)).max() > 0.5  # rotating wall
