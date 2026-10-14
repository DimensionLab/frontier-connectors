# SPDX-License-Identifier: Apache-2.0
"""Geometry-only sources: STL (ASCII / binary), OBJ, PLY (ASCII / binary), OFF, and CAD
B-rep files (STEP / IGES / BREP) through an optional tessellator.

Every file becomes a steady ``surface`` sample (triangles; polygons are fan-triangulated)
with no solver fields. What the formats do carry is kept: PLY vertex properties (colours,
normals, scalars) become point fields, PLY face properties cell fields, and the solids of a
multi-solid ASCII STL or the ``o`` / ``g`` groups of an OBJ become zones (one sample each;
``options={"merge_zones": True}`` keeps them together). A directory of geometry files is read
as one sample per file (id = file stem), so a geometry sweep converts in one call.

STL stores three vertices per triangle: duplicates are merged (exact match, or within
``options={"merge_tolerance": 1e-6}``) so the surface is indexed and, when the source is
closed, watertight. ``describe`` and ``RawSample.metadata["geometry"]`` report triangle count,
bounds, area, edge-manifold watertightness and the enclosed volume of closed surfaces.

Options: ``units`` (``"mm"``: recorded as ``metadata["units"]["length"]``, nothing is rescaled),
``merge_tolerance``, ``merge_zones``, ``tessellator`` (``"auto"`` | ``"ocp"`` | ``"gmsh"``) and
``tessellation`` (``{"linear_deflection": 0.1, "angular_deflection": 0.5}``).

CAD files need a tessellator, imported only when such a file is read: ``cadquery-ocp``
(Apache-2.0 bindings to OpenCascade, which is LGPL-2.1 with an exception) or, only when asked
for with ``tessellator="gmsh"``, ``gmsh`` (GPL-2.0-or-later).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import (
    ConnectorError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldData,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeSelection,
    ZoneInfo,
    as_field,
    match_extension,
)
from frontier_connectors.select import match_names, wants

MESH_EXTENSIONS = [".stl", ".obj", ".ply", ".off"]
CAD_EXTENSIONS = [".step", ".stp", ".iges", ".igs", ".brep", ".brp"]
#: directories larger than this are described from file headers only
STATS_LIMIT = 16
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


@dataclass
class Part:
    """One surface (a file, an STL solid or an OBJ group)."""

    name: str
    points: np.ndarray
    faces: np.ndarray
    point_fields: dict[str, FieldData] = field(default_factory=dict)
    cell_fields: dict[str, FieldData] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


# -- shared helpers --------------------------------------------------------------------------


def merge_vertices(vertices: np.ndarray, tolerance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of ``vertices`` in order of first appearance and the index of every input
    row into them. ``tolerance > 0`` snaps coordinates to a grid of that size first."""
    v = np.ascontiguousarray(vertices) + 0.0  # -0.0 and 0.0 must compare equal bytewise
    key_src = np.round(v / tolerance).astype(np.int64) if tolerance > 0 else v
    key_src = np.ascontiguousarray(key_src)
    key = key_src.view(np.dtype((np.void, key_src.dtype.itemsize * key_src.shape[1]))).ravel()
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return v[first[order]], rank[np.asarray(inverse).reshape(-1)]


def surface_stats(points: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Triangle count, bounds, area, edge-manifold watertightness and (closed, consistently
    oriented surfaces only) the enclosed volume."""
    f = np.asarray(faces, dtype=np.int64)
    p = np.asarray(points, dtype=np.float64)
    out: dict[str, Any] = {
        "n_points": int(len(p)),
        "n_triangles": int(len(f)),
        "bounds": meshutil.bounds(p),
        "area": float(meshutil.triangle_areas(p, f).sum()),
        "watertight": False,
        "volume": None,
    }
    if not len(f):
        return out
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = int(directed.max()) + 1
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected[:, 0] * n + undirected[:, 1], return_counts=True)
    out["boundary_edges"] = int((counts == 1).sum())
    out["nonmanifold_edges"] = int((counts > 2).sum())
    out["watertight"] = bool((counts == 2).all())
    oriented = len(np.unique(directed[:, 0] * n + directed[:, 1])) == len(directed)
    out["consistently_oriented"] = bool(oriented)
    if out["watertight"] and oriented:
        a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        signed = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        out["volume"] = abs(signed)
        out["normals"] = "outward" if signed >= 0 else "inward"
    return out


def _drop_degenerate(part: Part) -> None:
    f = part.faces
    if not len(f):
        return
    bad = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    if bad.any():
        keep = ~bad
        part.faces = f[keep]
        for fd in part.cell_fields.values():
            fd.data = fd.data[keep]
        part.info["degenerate_triangles_dropped"] = int(bad.sum())


def _corrupt(path: Path, what: str, hint: str) -> ConnectorError:
    return ConnectorError(f"{path.name}: {what}", hint=hint)


# -- STL -------------------------------------------------------------------------------------


def stl_is_binary(path: Path) -> bool | None:
    """``True``/``False`` for binary/ASCII, ``None`` when the file is neither. Decided by the
    size formula ``84 + 50 n`` (binary files may also start with the word ``solid``)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(1024)
    except OSError:
        return None
    if size >= 84:
        n = int(np.frombuffer(head[80:84], dtype="<u4")[0])
        if 84 + 50 * n == size:
            return True
    text = head.lstrip()
    if text[:5].lower() == b"solid" and (b"facet" in head or b"endsolid" in head):
        return False
    return None


def read_stl(path: Path, tolerance: float = 0.0) -> list[Part]:
    kind = stl_is_binary(path)
    if kind is None:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(84)
        n = int(np.frombuffer(head[80:84], dtype="<u4")[0]) if len(head) == 84 else 0
        raise _corrupt(
            path,
            f"not a valid STL: the header announces {n} triangles ({84 + 50 * n} bytes) but the "
            f"file has {size} bytes, and it is not ASCII STL either",
            "the file is truncated or was transferred in text mode; export it again",
        )
    if kind:
        rec = np.fromfile(path, dtype=_STL_RECORD, offset=84)
        with open(path, "rb") as f:
            title = f.read(80).split(b"\x00")[0].decode("ascii", "replace").strip()
        pts, inv = merge_vertices(rec["v"].reshape(-1, 3), tolerance)
        part = Part(path.stem, pts, inv.reshape(-1, 3), info={"stl": "binary", "title": title})
        _drop_degenerate(part)
        return [part]
    data = path.read_bytes()
    starts = [m for m in re.finditer(rb"(?m)^[ \t]*solid\b[ \t]*([^\r\n]*)", data)]
    parts: list[Part] = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(data)
        body = data[m.end() : end]
        found = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", body)
        if not found and len(starts) > 1:
            continue
        try:
            verts = np.array(found, dtype="S").astype(np.float64).reshape(-1, 3)
        except ValueError as e:
            raise _corrupt(path, f"ASCII STL has a non-numeric vertex ({e})", "re-export") from e
        if len(verts) % 3:
            raise _corrupt(
                path,
                f"ASCII STL solid has {len(verts)} vertices, not a multiple of 3",
                "the file is truncated in the middle of a facet; export it again",
            )
        name = m.group(1).decode("utf-8", "replace").strip() or f"solid{i}"
        pts, inv = merge_vertices(verts, tolerance)
        part = Part(meshutil.sanitize_id(name), pts, inv.reshape(-1, 3), info={"stl": "ascii"})
        _drop_degenerate(part)
        parts.append(part)
    if not parts:
        raise _corrupt(path, "ASCII STL without any facet", "the file is empty or truncated")
    return _unique_part_names(parts, path.stem)


def _unique_part_names(parts: list[Part], stem: str) -> list[Part]:
    if len(parts) == 1:
        parts[0].info.setdefault("source_name", parts[0].name)
        parts[0].name = stem
        return parts
    seen: dict[str, int] = {}
    for p in parts:
        n = seen.get(p.name, 0)
        seen[p.name] = n + 1
        if n:
            p.name = f"{p.name}_{n}"
    return parts


# -- OBJ -------------------------------------------------------------------------------------


def read_obj(path: Path) -> list[Part]:
    verts: list[tuple[float, float, float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = "default"
    by_object = False
    no = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                no += 1
                if not line or line[0] in "#\n\r":
                    continue
                tok = line.split()
                if not tok:
                    continue
                key = tok[0]
                if key == "v":
                    verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                elif key == "f":
                    row = []
                    for t in tok[1:]:
                        i = int(t.split("/", 1)[0])
                        row.append(i - 1 if i > 0 else len(verts) + i)
                    if len(row) >= 3:
                        groups.setdefault(current, []).append(row)
                elif key == "o":
                    by_object = True
                    current = " ".join(tok[1:]) or "object"
                elif key == "g" and not by_object:
                    current = " ".join(tok[1:]) or "default"
    except (ValueError, IndexError) as e:
        raise _corrupt(
            path, f"OBJ line {no} cannot be parsed ({e})", "the file is truncated or not OBJ"
        ) from e
    if not verts:
        raise _corrupt(path, "OBJ file has no 'v' vertex lines", "the file is empty or not OBJ")
    points = np.asarray(verts, dtype=np.float64)
    if not groups:
        return [Part(path.stem, points, np.zeros((0, 3), dtype=np.int64), info={"faces": 0})]
    parts = []
    for name, rows in groups.items():
        width = max(len(r) for r in rows)
        cells = np.full((len(rows), width), -1, dtype=np.int64)
        for k, r in enumerate(rows):
            cells[k, : len(r)] = r
        if cells.max() >= len(points) or cells.min() < -1:
            raise _corrupt(
                path,
                f"OBJ group {name!r} references vertex {int(cells.max()) + 1} but only "
                f"{len(points)} vertices are defined",
                "the file is truncated; export it again",
            )
        faces, _ = meshutil.triangulate(cells)
        pts, faces, _used = meshutil.compact_points(points, faces)
        info = {"polygons": len(rows), "max_polygon_size": width}
        parts.append(Part(meshutil.sanitize_id(name), pts, faces, info=info))
    return _unique_part_names(parts, path.stem)


# -- PLY -------------------------------------------------------------------------------------


@dataclass
class _PlyElement:
    name: str
    count: int
    #: (property name, scalar dtype, list count dtype or None)
    props: list[tuple[str, str, str | None]] = field(default_factory=list)

    @property
    def has_list(self) -> bool:
        return any(p[2] is not None for p in self.props)


def ply_header(path: Path) -> tuple[str, list[_PlyElement], int]:
    """``(format, elements, data offset)`` of a PLY file."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    if not head.startswith(b"ply"):
        raise _corrupt(path, "missing 'ply' magic", "the file is not PLY")
    end = head.find(b"end_header")
    if end < 0:
        raise _corrupt(path, "PLY header has no end_header line", "the file is truncated")
    nl = head.find(b"\n", end)
    offset = nl + 1 if nl >= 0 else len(head)
    fmt = ""
    elements: list[_PlyElement] = []
    try:
        for line in head[:end].decode("ascii", "replace").splitlines()[1:]:
            tok = line.split()
            if not tok or tok[0] in ("comment", "obj_info"):
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append(_PlyElement(tok[1], int(tok[2])))
            elif tok[0] == "property":
                if tok[1] == "list":
                    elements[-1].props.append((tok[4], _PLY_TYPES[tok[3]], _PLY_TYPES[tok[2]]))
                else:
                    elements[-1].props.append((tok[2], _PLY_TYPES[tok[1]], None))
    except (IndexError, KeyError, ValueError) as e:
        raise _corrupt(path, f"malformed PLY header ({e!r})", "the file is not valid PLY") from e
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise _corrupt(path, f"unknown PLY format {fmt!r}", "expected ascii or binary_*_endian")
    return fmt, elements, offset


def _ply_binary_element(
    path: Path, buf: memoryview, pos: int, el: _PlyElement, order: str
) -> tuple[dict[str, np.ndarray], list[np.ndarray] | None, int]:
    """Read one binary element: scalar columns, the rows of its (first) list property, and
    the new offset."""
    lists: list[np.ndarray] | None = None
    if not el.has_list:
        dt = np.dtype([(n, order + t) for n, t, _ in el.props])
        need = dt.itemsize * el.count
        if pos + need > len(buf):
            raise _corrupt(
                path,
                f"PLY element '{el.name}' needs {need} bytes, {len(buf) - pos} are left",
                "the file is truncated",
            )
        arr = np.frombuffer(buf, dtype=dt, count=el.count, offset=pos)
        return {n: arr[n] for n in arr.dtype.names or ()}, None, pos + need
    # fast path: every list has the length of the first one (all triangles, all quads)
    if el.count:
        fields = []
        p = pos
        for n, t, ct in el.props:
            if ct is None:
                fields.append((n, order + t))
                p += np.dtype(t).itemsize
                continue
            if p + np.dtype(ct).itemsize > len(buf):
                raise _corrupt(path, f"PLY element '{el.name}' is cut off", "truncated file")
            k = int(np.frombuffer(buf, dtype=order + ct, count=1, offset=p)[0])
            fields.append((n + "__n", order + ct))
            fields.append((n, order + t, (k,)))
            p += np.dtype(ct).itemsize + k * np.dtype(t).itemsize
        dt = np.dtype(fields)
        need = dt.itemsize * el.count
        if pos + need <= len(buf):
            arr = np.frombuffer(buf, dtype=dt, count=el.count, offset=pos)
            uniform = all(
                bool((arr[n + "__n"] == int(np.shape(arr[n])[-1])).all())
                for n, _, ct in el.props
                if ct
            )
            if uniform:
                cols = {n: arr[n] for n, _, ct in el.props if ct is None}
                first = next(n for n, _, ct in el.props if ct)
                return cols, [np.asarray(arr[first])], pos + need
    cols_l: dict[str, list[Any]] = {n: [] for n, _, ct in el.props if ct is None}
    rows: list[np.ndarray] = []
    try:
        for _ in range(el.count):
            taken = False
            for n, t, ct in el.props:
                if ct is None:
                    cols_l[n].append(np.frombuffer(buf, dtype=order + t, count=1, offset=pos)[0])
                    pos += np.dtype(t).itemsize
                    continue
                k = int(np.frombuffer(buf, dtype=order + ct, count=1, offset=pos)[0])
                pos += np.dtype(ct).itemsize
                vals = np.frombuffer(buf, dtype=order + t, count=k, offset=pos)
                pos += k * np.dtype(t).itemsize
                if not taken:
                    rows.append(vals)
                    taken = True
    except ValueError as e:
        raise _corrupt(path, f"PLY element '{el.name}' is cut off ({e})", "truncated file") from e
    lists = rows
    return {n: np.asarray(v) for n, v in cols_l.items()}, lists, pos


def _ply_ascii_element(
    path: Path, tokens: list[bytes], pos: int, el: _PlyElement
) -> tuple[dict[str, np.ndarray], list[np.ndarray] | None, int]:
    try:
        if not el.has_list:
            n = len(el.props) * el.count
            block = np.array(tokens[pos : pos + n], dtype="S").astype(np.float64)
            block = block.reshape(el.count, len(el.props))
            cols = {name: block[:, i].astype(t) for i, (name, t, _) in enumerate(el.props)}
            return cols, None, pos + n
        cols_l: dict[str, list[float]] = {n: [] for n, _, ct in el.props if ct is None}
        rows: list[np.ndarray] = []
        for _ in range(el.count):
            taken = False
            for name, _t, ct in el.props:
                if ct is None:
                    cols_l[name].append(float(tokens[pos]))
                    pos += 1
                    continue
                k = int(tokens[pos])
                vals = np.array(tokens[pos + 1 : pos + 1 + k], dtype="S").astype(np.int64)
                if len(vals) != k:
                    raise IndexError("list runs past the end of the file")
                pos += 1 + k
                if not taken:
                    rows.append(vals)
                    taken = True
        return {n: np.asarray(v) for n, v in cols_l.items()}, rows, pos
    except (ValueError, IndexError) as e:
        raise _corrupt(
            path, f"PLY element '{el.name}' is incomplete ({e})", "the file is truncated"
        ) from e


def _ply_fields(cols: Mapping[str, np.ndarray], location: str) -> dict[str, FieldData]:
    out: dict[str, FieldData] = {}
    used: set[str] = {"x", "y", "z"}

    def group(name: str, members: Sequence[str]) -> None:
        if all(m in cols for m in members):
            out[name] = as_field(
                np.stack([cols[m] for m in members], axis=1), location, component_names=[*members]
            )
            used.update(members)

    group("normal", ("nx", "ny", "nz"))
    for rgb in (("red", "green", "blue", "alpha"), ("red", "green", "blue"), ("r", "g", "b")):
        if "color" not in out:
            group("color", rgb)
    group("texcoord", ("s", "t"))
    group("texcoord", ("u", "v")) if "texcoord" not in out else None
    for name, arr in cols.items():
        if name not in used:
            out[name] = as_field(arr, location)
    return out


def read_ply(path: Path) -> list[Part]:
    fmt, elements, offset = ply_header(path)
    data: dict[str, tuple[dict[str, np.ndarray], list[np.ndarray] | None]] = {}
    if fmt == "ascii":
        with open(path, "rb") as f:
            f.seek(offset)
            tokens = f.read().split()
        pos = 0
        for el in elements:
            cols, rows, pos = _ply_ascii_element(path, tokens, pos, el)
            data[el.name] = (cols, rows)
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        raw = np.memmap(path, dtype=np.uint8, mode="r") if path.stat().st_size else np.zeros(0)
        buf = memoryview(raw)  # type: ignore[arg-type]
        pos = offset
        for el in elements:
            cols, rows, pos = _ply_binary_element(path, buf, pos, el, order)
            data[el.name] = (cols, rows)
    if "vertex" not in data or not {"x", "y", "z"} <= set(data["vertex"][0]):
        raise _corrupt(path, "PLY has no vertex element with x, y, z", "not a surface PLY")
    vcols = data["vertex"][0]
    points = np.stack([vcols["x"], vcols["y"], vcols["z"]], axis=1).astype(np.float64)
    part = Part(path.stem, points, np.zeros((0, 3), dtype=np.int64), info={"ply": fmt})
    part.point_fields = _ply_fields(vcols, "point")
    fcols, rows = data.get("face", ({}, None))
    if rows:
        if len(rows) == 1 and rows[0].ndim == 2:
            cells = rows[0].astype(np.int64)
        else:
            width = max(len(r) for r in rows)
            cells = np.full((len(rows), width), -1, dtype=np.int64)
            for k, r in enumerate(rows):
                cells[k, : len(r)] = r
        if cells.size and (int(cells.max()) >= len(points) or int(cells.min()) < -1):
            raise _corrupt(
                path, "PLY faces reference vertices that do not exist", "the file is corrupt"
            )
        faces, parent = meshutil.triangulate(cells)
        part.faces = faces
        part.info["polygons"] = int(len(cells))
        for name, fd in _ply_fields(fcols, "cell").items():
            fd.data = fd.data[parent]
            part.cell_fields[name] = fd
    return [part]


# -- OFF -------------------------------------------------------------------------------------


def read_off(path: Path) -> list[Part]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    tokens = " ".join(ln for ln in lines if ln).split()
    try:
        head = tokens[0]
        if not head.upper().endswith("OFF"):
            raise ValueError(f"header {head!r}")
        per_vertex = 3 + (3 if "N" in head.upper() else 0) + (4 if "C" in head.upper() else 0)
        nv, nf = int(tokens[1]), int(tokens[2])
        pos = 4
        block = np.array(tokens[pos : pos + nv * per_vertex], dtype=np.float64)
        points = block.reshape(nv, per_vertex)[:, :3]
        pos += nv * per_vertex
        rows = []
        for _ in range(nf):
            k = int(tokens[pos])
            rows.append([int(t) for t in tokens[pos + 1 : pos + 1 + k]])
            if len(rows[-1]) != k:
                raise ValueError("face list runs past the end of the file")
            pos += 1 + k
            # optional per-face colour: skip tokens until the next plausible count
            while pos < len(tokens) and len(rows) < nf and "." in tokens[pos]:
                pos += 1
    except (ValueError, IndexError) as e:
        raise _corrupt(path, f"OFF file cannot be parsed ({e})", "truncated or not OFF") from e
    if not rows:
        return [Part(path.stem, points, np.zeros((0, 3), dtype=np.int64))]
    width = max(len(r) for r in rows)
    cells = np.full((len(rows), width), -1, dtype=np.int64)
    for k, r in enumerate(rows):
        cells[k, : len(r)] = r
    if cells.max() >= nv:
        raise _corrupt(path, "OFF faces reference vertices that do not exist", "corrupt file")
    faces, _ = meshutil.triangulate(cells)
    return [Part(path.stem, points, faces, info={"polygons": nf})]


# -- CAD tessellation ------------------------------------------------------------------------


def tessellate_cad(path: Path, options: Mapping[str, Any]) -> list[Part]:
    """Tessellate a STEP / IGES / BREP file with OpenCascade (``cadquery-ocp``) or, when asked
    for explicitly, gmsh."""
    which = str(options.get("tessellator") or "auto").lower()
    tess = dict(options.get("tessellation") or {})
    if which in ("auto", "ocp", "cadquery", "opencascade"):
        try:
            ocp: Any = importlib.import_module("OCP")
        except ImportError as e:
            raise MissingDependencyError(
                "cadquery-ocp",
                purpose=f"tessellate {path.suffix.lower()} CAD files (alternative: "
                "options={'tessellator': 'gmsh'} with the GPL-2.0-or-later gmsh package)",
                license="Apache-2.0 bindings; OpenCascade itself is LGPL-2.1 with exception",
            ) from e
        del ocp
        return _tessellate_ocp(path, tess)
    if which == "gmsh":
        try:
            gmsh: Any = importlib.import_module("gmsh")
        except ImportError as e:
            raise MissingDependencyError(
                "gmsh", purpose=f"tessellate {path.suffix.lower()} CAD files",
                license="GPL-2.0-or-later",
            ) from e  # fmt: skip
        return _tessellate_gmsh(gmsh, path, tess)
    raise ConnectorError(
        f"unknown tessellator {which!r}", hint="use auto | ocp (Apache-2.0/LGPL) | gmsh (GPL)"
    )


def _static(cls: Any, name: str) -> Any:
    """OCP exposes C++ static methods as ``Name_s`` up to 7.7 and as ``Name`` afterwards."""
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def _tessellate_ocp(path: Path, tess: Mapping[str, Any]) -> list[Part]:
    mod = importlib.import_module
    ext = path.suffix.lower()
    topods: Any = mod("OCP.TopoDS")
    if ext in (".step", ".stp", ".iges", ".igs"):
        name = "STEPControl" if ext in (".step", ".stp") else "IGESControl"
        reader: Any = getattr(mod(f"OCP.{name}"), f"{name}_Reader")()
        status = reader.ReadFile(str(path))
        if int(status) != int(mod("OCP.IFSelect").IFSelect_ReturnStatus.IFSelect_RetDone):
            raise _corrupt(path, "OpenCascade could not read the CAD file", "check the export")
        reader.TransferRoots()
        shape: Any = reader.OneShape()
    else:
        shape = topods.TopoDS_Shape()
        builder: Any = mod("OCP.BRep").BRep_Builder()
        if not _static(mod("OCP.BRepTools").BRepTools, "Read")(shape, str(path), builder):
            raise _corrupt(path, "OpenCascade could not read the BREP file", "check the export")
    if shape.IsNull():
        raise _corrupt(path, "the CAD file holds no shape", "export solids or faces")
    box: Any = mod("OCP.Bnd").Bnd_Box()
    _static(mod("OCP.BRepBndLib").BRepBndLib, "Add")(shape, box)
    diag = float(box.SquareExtent()) ** 0.5 if not box.IsVoid() else 1.0
    lin = float(tess.get("linear_deflection") or 1e-3 * diag)
    ang = float(tess.get("angular_deflection") or 0.5)
    mod("OCP.BRepMesh").BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    top_abs: Any = mod("OCP.TopAbs")
    explorer: Any = mod("OCP.TopExp").TopExp_Explorer(shape, top_abs.TopAbs_FACE)
    brep_tool: Any = mod("OCP.BRep").BRep_Tool
    loc_cls: Any = mod("OCP.TopLoc").TopLoc_Location
    pts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    face_id: list[np.ndarray] = []
    base = 0
    n_faces = 0
    while explorer.More():
        face: Any = _static(topods.TopoDS, "Face")(explorer.Current())
        loc = loc_cls()
        tri: Any = _static(brep_tool, "Triangulation")(face, loc)
        explorer.Next()
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.empty((tri.NbNodes(), 3))
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            nodes[i - 1] = (p.X(), p.Y(), p.Z())
        conn = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
        for i in range(1, tri.NbTriangles() + 1):
            conn[i - 1] = tri.Triangle(i).Get()
        conn -= 1
        if face.Orientation() == top_abs.TopAbs_REVERSED:
            conn = conn[:, ::-1]
        pts.append(nodes)
        tris.append(conn + base)
        face_id.append(np.full(len(conn), n_faces, dtype=np.int32))
        base += len(nodes)
        n_faces += 1
    if not tris:
        raise _corrupt(path, "tessellation produced no triangles", "the shape has no faces")
    scale = max(diag, 1e-30)
    points, inv = merge_vertices(np.concatenate(pts), tolerance=1e-9 * scale)
    part = Part(path.stem, points, inv[np.concatenate(tris)])
    part.cell_fields["cad_face"] = as_field(np.concatenate(face_id), "cell")
    part.info.update(
        tessellator="OpenCascade (cadquery-ocp)",
        linear_deflection=lin,
        angular_deflection=ang,
        cad_faces=n_faces,
    )
    _drop_degenerate(part)
    return [part]


def _tessellate_gmsh(gmsh: Any, path: Path, tess: Mapping[str, Any]) -> list[Part]:
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(path.stem)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        if tess.get("linear_deflection"):
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(tess["linear_deflection"]) * 20)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        _etags, conn = gmsh.model.mesh.getElementsByType(2)
        index = np.full(int(max(tags)) + 1, -1, dtype=np.int64)
        index[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        faces = index[np.asarray(conn, dtype=np.int64)].reshape(-1, 3)
    except Exception as e:  # noqa: BLE001 - gmsh raises bare Exception
        raise _corrupt(path, f"gmsh could not tessellate the CAD file ({e})", "check it") from e
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()
    pts, faces, _ = meshutil.compact_points(points, faces)
    return [Part(path.stem, pts, faces, info={"tessellator": "gmsh"})]


# -- dispatch --------------------------------------------------------------------------------


def read_geometry(path: Path, options: Mapping[str, Any] | None = None) -> list[Part]:
    """Read one geometry file into parts (most files: one part)."""
    opts = options or {}
    ext = path.suffix.lower()
    tol = float(opts.get("merge_tolerance") or 0.0)
    try:
        if ext == ".stl":
            parts = read_stl(path, tol)
        elif ext == ".obj":
            parts = read_obj(path)
        elif ext == ".ply":
            parts = read_ply(path)
        elif ext == ".off":
            parts = read_off(path)
        elif ext in CAD_EXTENSIONS:
            parts = tessellate_cad(path, opts)
        else:
            raise UnsupportedSourceError(
                f"{path.name}: not a supported geometry format",
                hint="supported: " + ", ".join(MESH_EXTENSIONS + CAD_EXTENSIONS),
            )
    except OSError as e:
        raise ConnectorError(f"{path} cannot be read ({e})", hint="check the file") from e
    if opts.get("merge_zones") and len(parts) > 1:
        offset = 0
        faces = []
        zone_id = []
        for k, p in enumerate(parts):
            faces.append(p.faces + offset)
            zone_id.append(np.full(len(p.faces), k, dtype=np.int32))
            offset += len(p.points)
        merged = Part(path.stem, np.concatenate([p.points for p in parts]), np.concatenate(faces))
        merged.cell_fields["zone_id"] = as_field(np.concatenate(zone_id), "cell")
        merged.info["zones"] = [p.name for p in parts]
        parts = [merged]
    return parts


def quick_counts(path: Path) -> tuple[int | None, int | None]:
    """``(n_points, n_triangles)`` from the file header alone, where the format allows."""
    ext = path.suffix.lower()
    try:
        if ext == ".stl" and stl_is_binary(path):
            return None, (path.stat().st_size - 84) // 50
        if ext == ".ply":
            _, elements, _ = ply_header(path)
            counts = {e.name: e.count for e in elements}
            return counts.get("vertex"), counts.get("face")
        if ext == ".off":
            with open(path, encoding="utf-8", errors="replace") as f:
                tok = f.read(4096).split()
            return int(tok[1]), int(tok[2])
    except (ConnectorError, OSError, ValueError, IndexError):
        return None, None
    return None, None


def _geometry_files(path: Path) -> list[Path]:
    exts = MESH_EXTENSIONS + CAD_EXTENSIONS
    return [f for f in sorted(path.iterdir()) if f.is_file() and match_extension(f, exts)]


class GeometryConnector(BaseConnector):
    id = "geometry"
    name = "Geometry only (STL, OBJ, PLY, OFF; STEP / IGES / BREP via optional tessellator)"
    formats = ["stl", "obj", "ply", "off", "step", "iges", "brep"]
    vendor = "open formats"
    license = (
        "Apache-2.0 (optional CAD tessellators: cadquery-ocp Apache-2.0 on OpenCascade "
        "LGPL-2.1-with-exception; gmsh GPL-2.0-or-later, opt-in only)"
    )
    extensions = MESH_EXTENSIONS + CAD_EXTENSIONS
    maturity = "verified-public-sample"
    temporal = ["steady"]
    kinds = ["surface", "point_cloud"]
    requires = ["cadquery-ocp"]
    export_help = (
        "Any CAD or meshing tool exports STL / OBJ / PLY. For STEP / IGES / BREP install "
        "cadquery-ocp (pip install cadquery-ocp) and tune options "
        "tessellation={'linear_deflection': ..., 'angular_deflection': ...}. Put a design sweep in "
        "one directory to convert it as one dataset."
    )
    status_note = (
        "STL, OBJ, PLY and OFF are read with built-in parsers. STEP, IGES and BREP need "
        "cadquery-ocp, or gmsh on request. OBJ materials and textures, PLY tristrips and binary "
        "STL colour attributes are ignored."
    )

    # detect -----------------------------------------------------------------------------
    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        if path.is_dir():
            try:
                entries = [e for e in path.iterdir() if not e.name.startswith(".")]
            except OSError:
                return None
            files = [e for e in entries if e.is_file()]
            geo = [f for f in files if match_extension(f, self.extensions)]
            if not geo or len(geo) < 0.8 * len(entries):
                return None
            return Detection(
                self.id, path, 0.55, "geometry-dir", [f"directory of {len(geo)} geometry files"]
            )
        ext = match_extension(path, self.extensions)
        if ext is None or not path.is_file():
            return None
        try:
            with open(path, "rb") as f:
                head = f.read(512)
        except OSError:
            return None
        ext = ext.lower()
        if ext == ".stl":
            kind = stl_is_binary(path)
            if kind is None:
                return Detection(self.id, path, 0.4, "stl", ["extension .stl, size mismatch"])
            fmt = "stl-binary" if kind else "stl-ascii"
            return Detection(self.id, path, 0.9, fmt, [fmt.replace("-", " ") + " layout"])
        if ext == ".ply":
            if not head.startswith(b"ply"):
                return None
            return Detection(self.id, path, 0.9, "ply", ["'ply' magic"])
        if ext == ".off":
            if b"OFF" not in head[:8].upper():
                return None
            return Detection(self.id, path, 0.85, "off", ["OFF header"])
        if ext == ".obj":
            if b"\x00" in head or not re.search(rb"(?m)^\s*(v|vn|vt|f|o|g|#|mtllib)\s", head):
                return None
            return Detection(self.id, path, 0.8, "obj", ["Wavefront OBJ statements"])
        det = Detection(self.id, path, 0.6, ext.lstrip("."), [f"extension {ext}"])
        if ext in (".step", ".stp"):
            if b"ISO-10303-21" not in head:
                return None
            det.confidence, det.format = 0.9, "step"
        if importlib.util.find_spec("OCP") is None:
            det.needs.append("cadquery-ocp (Apache-2.0; OpenCascade LGPL-2.1) to tessellate CAD")
        return det

    # plan -------------------------------------------------------------------------------
    def _files(self, path: Path) -> list[Path]:
        if path.is_dir():
            files = _geometry_files(path)
            if not files:
                raise UnsupportedSourceError(
                    f"{path} holds no geometry files",
                    hint="supported: " + ", ".join(self.extensions),
                )
            return files
        if not path.is_file():
            raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
        return [path]

    @staticmethod
    def _zone(path: Path, part: Part, n_parts: int, in_dir: bool) -> str:
        if not in_dir:
            return part.name if n_parts > 1 else "surface"
        return path.stem if n_parts == 1 else f"{path.stem}/{part.name}"

    @staticmethod
    def _id(path: Path, part: Part, n_parts: int) -> str:
        base = meshutil.default_sample_id(path)
        return base if n_parts == 1 else f"{base}__{meshutil.sanitize_id(part.name)}"

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        files = self._files(path)
        in_dir = path.is_dir()
        opts = options or {}
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format="geometry-dir" if in_dir else path.suffix.lower().lstrip("."),
            solver="geometry",
            domain="other",
        )
        if opts.get("units"):
            desc.units = {"length": str(opts["units"])}
        stats: dict[str, Any] = {}
        seen: set[tuple[str, str]] = set()
        for k, f in enumerate(files):
            if in_dir and k >= STATS_LIMIT:
                n_pts, n_tri = quick_counts(f)
                desc.zones.append(
                    ZoneInfo(f.stem, "surface", "geometry", n_points=n_pts, n_cells=n_tri)
                )
                continue
            parts = read_geometry(f, opts)
            for part in parts:
                st = surface_stats(part.points, part.faces)
                zone = self._zone(f, part, len(parts), in_dir)
                kind = "surface" if len(part.faces) else "point_cloud"
                z = ZoneInfo(zone, kind, "geometry" if kind == "surface" else "probes")
                z.n_points, z.n_cells, z.bounds = st["n_points"], st["n_triangles"], st["bounds"]
                z.dim = 2 if kind == "surface" else 3
                if len(part.faces):
                    z.cell_types = {"triangle": st["n_triangles"]}
                desc.zones.append(z)
                stats[zone] = {**st, **part.info}
                for group in (part.point_fields, part.cell_fields):
                    for name, fd in group.items():
                        if (name, fd.location) not in seen:
                            seen.add((name, fd.location))
                            desc.fields.append(
                                FieldInfo(
                                    name=name,
                                    location=fd.location,
                                    components=fd.components,
                                    component_names=fd.component_names,
                                    dtype=str(fd.data.dtype),
                                    time_dependent=False,
                                )
                            )
        names = [z.name for z in desc.zones]
        if len(set(names)) != len(names):
            raise ConnectorError(
                f"{path}: two geometry files share a stem ({sorted(set(names))[:5]} ...)",
                hint="sample ids come from file stems; rename one of the files",
            )
        desc.metadata["geometry"] = stats
        desc.metadata["n_files"] = len(files)
        if in_dir and len(files) > STATS_LIMIT:
            desc.warnings.append(
                f"statistics were computed for the first {STATS_LIMIT} of {len(files)} files"
            )
        for zone, st in stats.items():
            if st["n_triangles"] and not st["watertight"]:
                desc.warnings.append(
                    f"{zone}: surface is not watertight ({st.get('boundary_edges', 0)} boundary "
                    f"edges, {st.get('nonmanifold_edges', 0)} non-manifold edges)"
                )
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
        files = self._files(path)
        in_dir = path.is_dir()
        opts = options or {}
        if in_dir:
            # zones are file stems here: filter before parsing anything
            stems = [f.stem for f in files]
            if zones is not None and all("/" not in z for z in zones):
                keep = set(match_names(stems, zones))
                files = [f for f in files if f.stem in keep]
                zones = None
        yielded = 0
        available: list[str] = []
        for f in files:
            parts = read_geometry(f, opts)
            names = [self._zone(f, p, len(parts), in_dir) for p in parts]
            available.extend(names)
            if zones is not None and not in_dir:
                selected = set(match_names(names, zones))
            elif zones is not None:
                selected = {n for n in names if wants(n, zones)}
            else:
                selected = set(names)
            n_sel = len(selected) if not in_dir else len(parts)
            for part, zone in zip(parts, names, strict=True):
                if zone not in selected:
                    continue
                md: dict[str, Any] = {
                    "solver": "geometry",
                    "source": str(f),
                    "format": f.suffix.lower().lstrip("."),
                    "geometry": {**surface_stats(part.points, part.faces), **part.info},
                }
                if opts.get("units"):
                    md["units"] = {"length": str(opts["units"])}
                sample = RawSample(
                    id=self._id(f, part, n_sel),
                    kind="surface" if len(part.faces) else "point_cloud",
                    zone=zone,
                    points=part.points,
                    faces=part.faces if len(part.faces) else None,
                    metadata=md,
                )
                for group in (part.point_fields, part.cell_fields):
                    for name, fd in group.items():
                        if wants(name, fields):
                            sample.fields[name] = fd
                yielded += 1
                yield sample
        if not yielded and zones is not None:
            match_names(available, zones)
