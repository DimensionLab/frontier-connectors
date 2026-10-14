# SPDX-License-Identifier: Apache-2.0
"""Nek5000 / NekRS binary field files (``<case>0.f00001``, optional ``<case>.nek5000``).

Own reader written from the public format description (Nek5000 ``prepost.f`` behaviour as
documented in the Nek5000 manual and the VisIt/ParaView reader notes); no GPL code is used.

File layout::

    132 bytes ASCII  #std <wdsize> <lx1> <ly1> <lz1> <nelt> <nelgt> <time> <istep> <fid>
                     <nfiles> <rdcode> [p0th] [if_press_mesh]
    4 bytes          float32 6.54321 (gives the byte order)
    nelt int32       global element ids
    per variable of rdcode, in the order X U P T S01..Snn:
                     nelt * [component][lz1*ly1*lx1] reals of wdsize bytes
    (3-D only)       per-variable, per-element min/max float32 blocks (ignored here)

Each spectral element's GLL nodes become points and every GLL sub-cell a linear hexahedron
(quadrilateral in 2-D), so the result is an ordinary ``mesh`` sample. Nodes shared by
neighbouring elements are stored once per element unless ``options={"merge_points": True}``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    ZoneInfo,
    as_field,
)
from frontier_connectors.select import match_names, select_time_indices, wants

HEADER_BYTES = 132
ENDIAN_TAG = 6.54321
ZONE = "fluid"

_FIELD_FILE = re.compile(r"^(?P<case>.+?)(?P<fid>\d+)\.f(?P<step>\d{5,})$")
_LEGACY = re.compile(r"\.fld\d*$", re.IGNORECASE)
_DIR = re.compile(r"^[A-Za-z]\d{2,}$")

_FORMAT_HINT = (
    "expected a Nek5000/NekRS binary field file (`#std` header, written with param(66)=6 / "
    "NekRS default); re-dump the case or convert with Nek5000's own tools"
)


@dataclass
class Variable:
    """One block of the file: ``code`` is X, U, P, T or S<k>."""

    code: str
    name: str
    components: int
    offset: int = 0


@dataclass
class Header:
    path: Path
    wdsize: int
    lx1: int
    ly1: int
    lz1: int
    nelt: int
    nelgt: int
    time: float
    istep: int
    fid: int
    nfiles: int
    rdcode: str
    endian: str
    p0th: float | None = None
    press_mesh: bool = False
    variables: list[Variable] = field(default_factory=list)

    @property
    def ndim(self) -> int:
        return 3 if self.lz1 > 1 else 2

    @property
    def npts(self) -> int:
        return self.lx1 * self.ly1 * self.lz1

    @property
    def real(self) -> np.dtype[Any]:
        return np.dtype(f"{self.endian}f{self.wdsize}")

    def variable(self, code: str) -> Variable | None:
        return next((v for v in self.variables if v.code == code), None)

    @property
    def data_bytes(self) -> int:
        n = self.nelt * self.npts * self.wdsize
        return sum(v.components for v in self.variables) * n


def _bad(path: Path, why: str) -> ConnectorError:
    return ConnectorError(f"{path} is not a readable Nek5000 field file: {why}", hint=_FORMAT_HINT)


def parse_rdcode(rdcode: str, ndim: int) -> list[Variable]:
    """``XUPTS02`` -> coordinates, velocity, pressure, temperature, scalar_1, scalar_2."""
    out: list[Variable] = []
    i = 0
    code = rdcode.strip()
    while i < len(code):
        c = code[i]
        i += 1
        if c == "X":
            out.append(Variable("X", "coordinates", ndim))
        elif c == "U":
            out.append(Variable("U", "velocity", ndim))
        elif c == "P":
            out.append(Variable("P", "pressure", 1))
        elif c == "T":
            out.append(Variable("T", "temperature", 1))
        elif c == "S":
            digits = code[i : i + 2]
            if len(digits) != 2 or not digits.isdigit():
                raise ValueError(f"scalar count missing after 'S' in rdcode {rdcode!r}")
            i += 2
            out.extend(Variable(f"S{k}", f"scalar_{k}", 1) for k in range(1, int(digits) + 1))
        elif c.isspace():
            continue
        else:
            raise ValueError(f"unknown variable code {c!r} in rdcode {rdcode!r}")
    return out


def read_header(path: Path) -> Header:
    """Parse the 132-byte header and the endian tag (reads 136 bytes)."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = f.read(HEADER_BYTES + 4)
    except OSError as e:
        raise _bad(path, str(e)) from e
    if len(raw) < HEADER_BYTES + 4 or not raw.startswith(b"#std"):
        raise _bad(path, "missing '#std' header" if raw[:4] != b"#std" else "truncated header")
    tokens = raw[:HEADER_BYTES].decode("ascii", "replace").split()
    try:
        wdsize, lx1, ly1, lz1, nelt, nelgt = (int(t) for t in tokens[1:7])
        time = float(tokens[7].replace("D", "E"))
        istep, fid, nfiles = (int(t) for t in tokens[8:11])
        rdcode = tokens[11]
    except (ValueError, IndexError) as e:
        raise _bad(path, f"malformed header ({e})") from e
    if wdsize not in (4, 8) or min(lx1, ly1, lz1, nelt) < 1 or lx1 < 2 or ly1 < 2:
        raise _bad(path, f"unsupported sizes wdsize={wdsize} lx1={lx1} ly1={ly1} nelt={nelt}")
    endian = ""
    for e in ("<", ">"):
        if abs(float(np.frombuffer(raw[HEADER_BYTES:], dtype=f"{e}f4")[0]) - ENDIAN_TAG) < 1e-4:
            endian = e
    if not endian:
        raise _bad(path, "endian tag 6.54321 not found after the header")
    h = Header(path, wdsize, lx1, ly1, lz1, nelt, nelgt, time, istep, fid, nfiles, rdcode, endian)
    try:
        h.variables = parse_rdcode(rdcode, h.ndim)
    except ValueError as e:
        raise _bad(path, str(e)) from e
    if len(tokens) > 12:
        try:
            h.p0th = float(tokens[12].replace("D", "E"))
        except ValueError:
            h.p0th = None
    h.press_mesh = len(tokens) > 13 and tokens[13].upper().startswith("T")
    offset = HEADER_BYTES + 4 + 4 * nelt
    for v in h.variables:
        v.offset = offset
        offset += v.components * nelt * h.npts * wdsize
    size = path.stat().st_size
    if size < offset:
        raise ConnectorError(
            f"{path} is truncated: header promises {offset} bytes of field data "
            f"({nelt} elements, rdcode {rdcode}), file has {size}",
            hint="the dump was interrupted or the copy is incomplete; re-copy or drop this file",
        )
    return h


def read_element_ids(h: Header) -> np.ndarray:
    return np.fromfile(h.path, dtype=f"{h.endian}i4", count=h.nelt, offset=HEADER_BYTES + 4)


def read_variable(h: Header, v: Variable) -> np.ndarray:
    """One variable as ``(nelt * npts, components)`` in native byte order (element by element,
    GLL nodes x-fastest inside each element)."""
    shape = (h.nelt, v.components, h.npts)
    mm = np.memmap(h.path, dtype=h.real, mode="r", offset=v.offset, shape=shape)
    out = np.ascontiguousarray(mm.transpose(0, 2, 1)).reshape(h.nelt * h.npts, v.components)
    del mm
    return out.astype(out.dtype.newbyteorder("="), copy=False)


def subcell_template(lx1: int, ly1: int, lz1: int) -> np.ndarray:
    """Local connectivity of the linear sub-cells of one element in VTK node order."""
    i, j, k = np.meshgrid(
        np.arange(lx1 - 1), np.arange(ly1 - 1), np.arange(max(lz1 - 1, 1)), indexing="ij"
    )
    i, j, k = (a.transpose(2, 1, 0).ravel() for a in (i, j, k))

    def pid(a: Any, b: Any, c: Any) -> Any:
        return a + lx1 * (b + ly1 * c)

    quad = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k)]
    if lz1 == 1:
        return np.stack(quad, axis=1).astype(np.int64)
    top = [pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)]
    return np.stack(quad + top, axis=1).astype(np.int64)


def merge_coincident(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster points closer than ``tolerance``. Returns ``(merged_points, inverse)``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    n = len(points)
    pairs = KDTree(points).query_pairs(tolerance, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), (n, n))
    n_merged, inverse = connected_components(graph, directed=False)
    merged = _average(points, inverse, n_merged)
    return merged, inverse.astype(np.int64)


def _average(values: np.ndarray, inverse: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(inverse, minlength=n).astype(np.float64)
    cols = [
        np.bincount(inverse, weights=values[:, c], minlength=n) / counts
        for c in range(values.shape[1])
    ]
    return np.stack(cols, axis=1)


# -- series discovery ---------------------------------------------------------------------------


@dataclass
class Series:
    """``steps[i]`` lists the files of dump ``i`` (several when the dump is split, nfiles > 1)."""

    root: Path
    case: str
    numbers: list[int]
    steps: list[list[Path]]
    solver: str = "Nek5000"
    warnings: list[str] = field(default_factory=list)


def _family(root: Path, case: str) -> dict[int, list[Path]]:
    """All field files of ``case`` under ``root`` (and its ``A01/``-style output directories)."""
    found: dict[int, list[Path]] = {}
    dirs = [root]
    try:
        dirs += sorted(d for d in root.iterdir() if d.is_dir() and _DIR.match(d.name))
    except OSError:
        pass
    for d in dirs:
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for f in entries:
            m = _FIELD_FILE.match(f.name)
            if m and m.group("case") == case and f.is_file():
                found.setdefault(int(m.group("step")), []).append(f)
    return found


def _solver_name(root: Path) -> str:
    """NekRS and Nek5000 write the same field files; only the case files tell them apart."""
    try:
        names = {f.suffix.lower() for f in root.iterdir() if f.is_file()}
    except OSError:
        return "Nek5000"
    if names & {".udf", ".oudf", ".okl"}:
        return "NekRS"
    return "Nek5000"


def parse_nek5000_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        raise ConnectorError(f"cannot read {path}: {e}", hint="check the file permissions") from e
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip().lower()] = value.strip()
    if "filetemplate" not in out:
        raise ConnectorError(
            f"{path} has no 'filetemplate:' line",
            hint="a .nek5000 file looks like 'filetemplate: case%01d.f%05d / firsttimestep: 1 / "
            "numtimesteps: 10'; or point at a field file such as case0.f00001 instead",
        )
    return out


def find_series(path: Path, *, single: bool = False) -> Series:
    path = Path(path)
    wanted: set[int] | None = None
    if path.is_dir():
        metas = sorted(path.glob("*.nek5000"))
        if len(metas) == 1:
            return find_series(metas[0])
        cases = sorted(
            {m.group("case") for f in path.iterdir() if (m := _FIELD_FILE.match(f.name))}
        )
        if len(cases) != 1:
            raise UnsupportedSourceError(
                f"{path} holds {len(cases)} Nek5000 field-file families"
                + (f" ({', '.join(cases)})" if cases else ""),
                hint="point at one field file (case0.f00001) or one .nek5000 file",
            )
        root, case = path, cases[0]
    elif path.suffix.lower() == ".nek5000":
        meta = parse_nek5000_metadata(path)
        template = Path(meta["filetemplate"].replace("\\", "/")).name
        m = re.match(r"^(?P<case>.*?)%\d*d\.f%\d*d$", template)
        if not m:
            raise ConnectorError(
                f"{path}: unsupported filetemplate {meta['filetemplate']!r}",
                hint="expected something like 'case%01d.f%05d' or 'A%02d/case%01d.f%05d'",
            )
        root, case = path.parent, m.group("case")
        try:
            first = int(meta.get("firsttimestep", "1"))
            count = int(meta["numtimesteps"]) if "numtimesteps" in meta else None
        except ValueError as e:
            raise ConnectorError(
                f"{path}: firsttimestep/numtimesteps are not integers ({e})",
                hint="fix the .nek5000 file or point at a field file directly",
            ) from e
        if count is not None:
            wanted = set(range(first, first + count))
    else:
        m = _FIELD_FILE.match(path.name)
        if not m:
            if _LEGACY.search(path.name):
                raise UnsupportedSourceError(
                    f"{path} is a legacy Nek5000 .fld file, which is not read (planned)",
                    hint="re-dump in the current binary format (param(66)=6 in the .rea / "
                    "'writeToFieldFile' default in the .par) to get case0.f00001 files",
                )
            raise UnsupportedSourceError(
                f"{path} is not a Nek5000 field file", hint="expected a name like case0.f00001"
            )
        root, case = path.parent, m.group("case")
        if _DIR.match(root.name) and read_header(path).nfiles > 1:
            # a split dump lives in A01/, A02/, ...: the family root is one level up
            root = root.parent
        if single:
            wanted = {int(m.group("step"))}
    family = _family(root, case)
    if wanted is not None:
        missing = sorted(wanted - set(family))
        family = {n: f for n, f in family.items() if n in wanted}
        if not family:
            raise ConnectorError(
                f"no field files of case {case!r} found next to {path}",
                hint="the .nek5000 file must sit in the directory that holds the field files",
            )
    else:
        missing = []
    if not family:
        raise ConnectorError(f"no field files found for {path}", hint=_FORMAT_HINT)
    numbers = sorted(family)
    s = Series(root, case, numbers, [family[n] for n in numbers], solver=_solver_name(root))
    if missing:
        s.warnings.append(f"{len(missing)} dumps listed in the .nek5000 file are missing on disk")
    return s


def step_headers(files: Sequence[Path]) -> list[Header]:
    """Headers of one dump ordered by file id; checks that a split dump is complete."""
    headers = sorted((read_header(f) for f in files), key=lambda h: h.fid)
    h0 = headers[0]
    if h0.nfiles > 1 or len(headers) > 1:
        fids = [h.fid for h in headers]
        if fids != list(range(h0.nfiles)):
            raise ConnectorError(
                f"dump {h0.path.name} is split into {h0.nfiles} files but file ids {fids} were "
                "found",
                hint="copy every A*/ output directory (or all case<k>.f* parts) of the dump",
            )
        for h in headers[1:]:
            same = (h.lx1, h.ly1, h.lz1, h.wdsize, h.rdcode) == (
                h0.lx1, h0.ly1, h0.lz1, h0.wdsize, h0.rdcode,
            )  # fmt: skip
            if not same:
                raise ConnectorError(
                    f"parts of dump {h0.path.name} disagree on polynomial order or variables",
                    hint="the parts belong to different runs; re-copy the dump",
                )
    if any(h.press_mesh for h in headers) and h0.variable("P") is not None:
        raise UnsupportedSourceError(
            f"{h0.path} stores pressure on the PN-2 mesh (if_press_mesh = T), which is not read",
            hint="dump with the pressure interpolated to the velocity mesh (the default), or "
            "select fields without pressure after re-dumping",
        )
    return headers


def _field_name(v: Variable) -> str:
    return v.name


class Nek5000Connector(BaseConnector):
    id = "nek5000"
    name = "Nek5000 / NekRS field files (.f%05d, .nek5000)"
    formats = ["nek5000-fld-binary"]
    vendor = "Argonne National Laboratory (open source)"
    license = "Apache-2.0 (own reader; Nek5000 itself is BSD-3)"
    extensions = [".nek5000", ".f00001"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["mesh"]
    requires: list[str] = []
    export_help = (
        "Nek5000 and NekRS write case0.f00001, case0.f00002, ... directly (plus case.nek5000 from "
        "the visnek script); point the connector at any of them or at the .nek5000 file."
    )
    status_note = (
        "Reads binary #std files: 2-D and 3-D, single and double precision, both byte orders, "
        "split dumps (nfiles > 1); geometry is reused when later dumps omit X. Legacy .fld files "
        "and pressure on the PN-2 mesh are not read. Spectral elements are exported as linear GLL "
        "sub-cells."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            if path.is_dir():
                if any(path.glob("*.nek5000")):
                    return Detection(self.id, path, 0.9, "nek5000", ["contains a .nek5000 file"])
                hits = [f for f in path.iterdir() if _FIELD_FILE.match(f.name)]
                if hits and self._magic(hits[0]):
                    return Detection(
                        self.id, path, 0.8, "nek5000", [f"{len(hits)} field files with #std header"]
                    )
                return None
            if not path.is_file():
                return None
            if path.suffix.lower() == ".nek5000":
                head = path.read_bytes()[:512]
                if b"filetemplate" in head:
                    return Detection(self.id, path, 0.97, "nek5000", ["filetemplate metadata"])
                return None
            if self._magic(path):
                conf = 0.98 if _FIELD_FILE.match(path.name) else 0.9
                return Detection(self.id, path, conf, "nek5000-fld-binary", ["#std header"])
            if _LEGACY.search(path.name) and self._legacy_header(path):
                return Detection(
                    self.id,
                    path,
                    0.3,
                    "nek5000-fld-legacy",
                    ["legacy .fld header"],
                    needs=["re-dump as binary .f%05d field files (legacy .fld is not read)"],
                )
        except OSError:
            return None
        return None

    @staticmethod
    def _magic(path: Path) -> bool:
        with open(path, "rb") as f:
            return f.read(5) == b"#std "

    @staticmethod
    def _legacy_header(path: Path) -> bool:
        with open(path, "rb") as f:
            tokens = f.read(80).split()
        try:
            return (
                len(tokens) >= 6 and all(int(t) > 0 for t in tokens[:4]) and float(tokens[4]) >= 0
            )
        except ValueError:
            return False

    # -- plan ---------------------------------------------------------------------------------
    def _plan(
        self, path: Path, options: Mapping[str, Any] | None
    ) -> tuple[Series, list[float], str, list[Header]]:
        """Series, its time axis, the axis quantity and the headers of the first dump."""
        opts = dict(options or {})
        series = find_series(Path(path), single=opts.get("series") is False)
        heads = [read_header(files[0]) for files in series.steps]
        times = [h.time for h in heads]
        quantity = "time"
        if len(times) > 1 and not all(b > a for a, b in zip(times, times[1:], strict=False)):
            steps = [float(h.istep) for h in heads]
            if all(b > a for a, b in zip(steps, steps[1:], strict=False)):
                times, quantity = steps, "iteration"
                series.warnings.append("header times do not increase: using the step numbers")
            else:
                times, quantity = [float(n) for n in series.numbers], "iteration"
                series.warnings.append("header times do not increase: using the file numbers")
        return series, times, quantity, step_headers(series.steps[0])

    def _geometry_step(self, series: Series) -> int:
        for i, files in enumerate(series.steps):
            if read_header(files[0]).variable("X") is not None:
                return i
        raise ConnectorError(
            f"none of the {len(series.steps)} dumps of case {series.case!r} contains the mesh "
            "(no X in rdcode)",
            hint="copy the first dump of the run (Nek5000 writes coordinates there), or dump "
            "with ifxyo = .true.",
        )

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        series, times, quantity, heads = self._plan(path, options)
        h = heads[0]
        nel = sum(x.nelt for x in heads)
        ncell = (h.lx1 - 1) * (h.ly1 - 1) * max(h.lz1 - 1, 1)
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format="nek5000-fld-binary",
            solver=series.solver,
            domain="cfd",
            time=TimeInfo(values=times if len(times) > 1 else [], units="", quantity=quantity),
            warnings=list(series.warnings),
        )
        desc.physics = {
            "method": "spectral-element",
            "polynomial_order": h.lx1 - 1,
            "dimension": h.ndim,
        }
        desc.metadata = {
            "elements": nel,
            "global_elements": h.nelgt,
            "gll_points_per_element": [h.lx1, h.ly1, h.lz1],
            "precision": "double" if h.wdsize == 8 else "single",
            "byte_order": "little" if h.endian == "<" else "big",
            "files_per_dump": h.nfiles,
            "rdcode": h.rdcode,
        }
        if h.p0th is not None and h.p0th != 0.0:
            desc.metadata["p0th"] = h.p0th
        desc.zones.append(
            ZoneInfo(
                name=ZONE,
                kind="mesh",
                role="volume",
                n_points=nel * h.npts,
                n_cells=nel * ncell,
                dim=h.ndim,
                cell_types={("hexahedron" if h.ndim == 3 else "quad"): nel * ncell},
            )
        )
        seen: dict[str, FieldInfo] = {}
        # variables can differ between dumps (e.g. mesh-only first file): scan the cheap headers
        for files in series.steps:
            for v in read_header(files[0]).variables:
                if v.code != "X" and v.name not in seen:
                    seen[v.name] = FieldInfo(
                        name=_field_name(v),
                        location="point",
                        components=v.components,
                        dtype="float32",
                        description=f"Nek5000 variable {v.code}",
                    )
        desc.fields = list(seen.values())
        if nel != h.nelgt:
            desc.warnings.append(f"dump holds {nel} of {h.nelgt} global elements")
        return desc

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        path = Path(path)
        opts = dict(options or {})
        series, all_times, quantity, _ = self._plan(path, opts)
        match_names([ZONE], zones, what="zone")
        transient = len(all_times) > 1
        indices = select_time_indices(all_times if transient else [], times)
        sample_id = meshutil.default_sample_id(series.root / series.case)
        merge = bool(opts.get("merge_points", False))

        geo_step = self._geometry_step(series)
        geo_heads = step_headers(series.steps[geo_step])
        g0 = geo_heads[0]
        points = np.concatenate([read_variable(h, _must(h, "X")) for h in geo_heads])
        gids = np.concatenate([read_element_ids(h) for h in geo_heads])
        nel = len(gids)
        template = subcell_template(g0.lx1, g0.ly1, g0.lz1)
        cells = (template[None, :, :] + (np.arange(nel) * g0.npts)[:, None, None]).reshape(
            -1, template.shape[1]
        )
        inverse: np.ndarray | None = None
        if merge:
            span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
            tol = float(opts.get("merge_tolerance", 1e-8)) * span
            points, inverse = merge_coincident(points.astype(np.float64), tol)
            cells = inverse[cells]
        cell_types = np.full(
            len(cells),
            meshutil.VTK_HEXAHEDRON if g0.ndim == 3 else meshutil.VTK_QUAD,
            dtype=np.uint8,
        )
        element_of_cell = np.repeat(gids.astype(np.int32), len(template))
        metadata: dict[str, Any] = {
            "solver": series.solver,
            "source": str(path),
            "format": "nek5000-fld-binary",
            "physics": {
                "method": "spectral-element",
                "polynomial_order": g0.lx1 - 1,
                "dimension": g0.ndim,
            },
            "time_units": "",
            "time_quantity": quantity,
            "merged_points": merge,
        }

        first = True
        for ti in indices:
            heads = step_headers(series.steps[ti])
            h0 = heads[0]
            step_ids = np.concatenate([read_element_ids(h) for h in heads])
            if (h0.lx1, h0.ly1, h0.lz1) != (g0.lx1, g0.ly1, g0.lz1) or len(step_ids) != nel:
                raise ConnectorError(
                    f"{h0.path} has {len(step_ids)} elements of {h0.lx1}x{h0.ly1}x{h0.lz1} "
                    f"nodes, the mesh dump has {nel} of {g0.lx1}x{g0.ly1}x{g0.lz1}",
                    hint="the dumps belong to different meshes; convert them separately "
                    '(options={"series": false})',
                )
            order = _element_order(gids, step_ids, h0.path)
            sample = RawSample(
                id=sample_id,
                kind="mesh",
                zone=ZONE,
                time=float(all_times[ti]) if transient else None,
                time_index=ti if transient else None,
                metadata={**metadata, "istep": h0.istep, "nek_time": h0.time},
            )

            def load(code: str, heads: list[Header] = heads, order: Any = order) -> np.ndarray:
                arr = np.concatenate([read_variable(h, _must(h, code)) for h in heads])
                if order is not None:
                    arr = arr.reshape(nel, g0.npts, -1)[order].reshape(nel * g0.npts, -1)
                return arr

            if first:
                sample.points = points
                sample.cells = cells
                sample.cell_types = cell_types
                sample.static["element_id"] = as_field(element_of_cell, "cell")
            elif ti != geo_step and h0.variable("X") is not None:
                moved = load("X")
                if inverse is not None:
                    moved = _average(moved.astype(np.float64), inverse, len(points))
                if not np.array_equal(moved, points):
                    sample.points = moved
            first = False
            for v in h0.variables:
                name = _field_name(v)
                if v.code == "X" or not wants(name, fields):
                    continue
                values = load(v.code)
                if inverse is not None:
                    values = _average(values.astype(np.float64), inverse, len(points))
                comps = ["x", "y", "z"][: v.components] if v.code == "U" else None
                sample.fields[name] = as_field(values, "point", component_names=comps)
            yield sample


def _must(h: Header, code: str) -> Variable:
    v = h.variable(code)
    if v is None:
        raise ConnectorError(
            f"{h.path} has no variable {code!r} (rdcode {h.rdcode})",
            hint="the parts of a split dump must carry the same variables",
        )
    return v


def _element_order(gids: np.ndarray, step_ids: np.ndarray, path: Path) -> np.ndarray | None:
    """Permutation bringing a dump's elements into the mesh dump's order (``None`` = same)."""
    if np.array_equal(gids, step_ids):
        return None
    a, b = np.argsort(gids, kind="stable"), np.argsort(step_ids, kind="stable")
    if not np.array_equal(gids[a], step_ids[b]):
        raise ConnectorError(
            f"{path} lists different global element ids than the mesh dump",
            hint="the dumps belong to different meshes or partitions; convert them separately",
        )
    order = np.empty(len(gids), dtype=np.int64)
    order[a] = b
    return order
