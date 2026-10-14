# SPDX-License-Identifier: Apache-2.0
"""Conduit Mesh Blueprint -> ``RawSample`` bridge.

The Mesh Blueprint is the tree a Catalyst 2 adaptor hands to ``catalyst_execute`` under
``catalyst/channels/<name>/data`` and what Conduit's relay I/O, Ascent and the ``frontier_emit``
spool protocol store. This module reads it from

* real ``conduit.Node`` objects (when the LLNL ``conduit`` Python module is importable; it is
  not on PyPI, the ``conduit`` distribution there is an unrelated project),
* plain nested ``dict`` trees with the same paths and numpy leaves (keys may also be whole
  paths such as ``"coordsets/coords/values/x"``).

Supported (read 2026-10-10 from https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html
and checked against ``conduit.blueprint.mesh.examples`` of Conduit 0.9.5):

* coordsets ``uniform`` (``dims/{i,j,k}``, ``origin/{x,y,z}``, ``spacing/{dx,dy,dz}``),
  ``rectilinear`` (``values/{x,y,z}``) and ``explicit`` (``values/{x,y,z}``, mcarray or an
  ``(N, D)`` array), Cartesian axes only;
* topologies ``points``, ``uniform``, ``rectilinear``, ``structured`` (``elements/dims``) and
  ``unstructured`` with ``elements/shape`` = ``point line tri quad tet hex wedge pyramid``,
  ``polygonal`` (``sizes``/``offsets``), ``polyhedral`` (``subelements``) and ``mixed``
  (``shape_map`` + ``shapes`` + ``sizes`` + ``offsets``), plus the older form with several named
  element blocks. Blueprint prescribes no winding; like MFEM, VisIt, VTK and Ascent the VTK
  node order is assumed;
* fields with ``association`` ``vertex`` / ``element``, scalar ``values`` or an mcarray
  (``values/{u,v,w}`` or interleaved), ``units``;
* ``state/{time,cycle,domain_id}``, multi-domain trees, and the Catalyst wrapper
  (``catalyst/state/{timestep,time,cycle}``, ``catalyst/channels/<name>/{type,data}`` with
  ``type`` ``mesh`` or ``multimesh``; https://docs.paraview.org/en/latest/Catalyst/blueprints.html).

Mapping: uniform and rectilinear topologies become ``kind="grid"`` samples (fields shaped
``(X, Y[, Z], C)``), everything else ``mesh`` (VTK cell types, ``-1`` padded connectivity),
``surface`` when every cell is two-dimensional (fan triangulated, cell data follows through the
parent map) or ``point_cloud``. Each (domain, topology) pair is one zone.

Not supported, and reported instead of dropped: non-Cartesian coordsets, strided structured
topologies, high-order (``grid_function`` / ``basis``) meshes and fields raise
:class:`UnsupportedSourceError`; material-dependent fields (``matset_values``), ``matsets``,
``specsets``, ``adjsets`` and ``nestsets`` are skipped with a warning (the geometry is complete
without them).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors._vtk import cells_to_nodes
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
from frontier_connectors.model import FieldData, FieldInfo, RawSample, ZoneInfo, as_field
from frontier_connectors.select import match_names, wants

log = logging.getLogger("frontier_connectors.catalyst")

AXES = ("x", "y", "z")
IJK = ("i", "j", "k")

#: Blueprint shape name -> (VTK cell type, nodes per element; 0 = variable)
SHAPES: dict[str, tuple[int, int]] = {
    "point": (meshutil.VTK_VERTEX, 1),
    "line": (meshutil.VTK_LINE, 2),
    "tri": (meshutil.VTK_TRIANGLE, 3),
    "quad": (meshutil.VTK_QUAD, 4),
    "tet": (meshutil.VTK_TETRA, 4),
    "hex": (meshutil.VTK_HEXAHEDRON, 8),
    "wedge": (meshutil.VTK_WEDGE, 6),
    "pyramid": (meshutil.VTK_PYRAMID, 5),
    "polygonal": (meshutil.VTK_POLYGON, 0),
    "polyhedral": (meshutil.VTK_POLYHEDRON, 0),
}
_IGNORED_SECTIONS = ("matsets", "specsets", "adjsets", "nestsets")
_HINT_SPEC = "see https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html"


# -- tree normalisation -------------------------------------------------------------------------


def is_conduit_node(obj: Any) -> bool:
    return (
        not isinstance(obj, Mapping)
        and hasattr(obj, "child_names")
        and hasattr(obj, "fetch")
        and hasattr(obj, "number_of_children")
    )


def text(value: Any) -> str:
    """A string leaf as ``str`` (accepts bytes, numpy strings and HDF5 ``|S1`` char arrays)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").rstrip("\x00")
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return b"".join(arr.reshape(-1).tolist()).decode("utf-8", "replace").rstrip("\x00")
    if arr.dtype.kind == "U":
        return "".join(arr.reshape(-1).tolist()).rstrip("\x00")
    raise ConnectorError(f"expected a string, got {type(value).__name__} {arr.dtype}")


def _leaf(value: Any, path: str) -> Any:
    if isinstance(value, str | bytes):
        return text(value)
    if value is None:
        return None
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.dtype.kind in "SU":
        return text(arr)
    if arr.dtype.kind not in "fiub":
        raise ConnectorError(
            f"{path or '<root>'}: unsupported leaf of type {type(value).__name__} ({arr.dtype})",
            hint="Blueprint leaves must be numbers, numeric arrays or strings",
        )
    return arr


def to_tree(node: Any, _path: str = "") -> Any:
    """Normalise a ``conduit.Node`` or a nested mapping into ``dict`` / ``ndarray`` / ``str``.

    Keys that contain ``/`` are expanded into nested dictionaries; lists of mappings (Conduit
    list nodes, e.g. unnamed domains) get the names ``domain_000000`` ...
    """
    if is_conduit_node(node):
        names = list(node.child_names())
        n_children = int(node.number_of_children())
        if names:
            return {str(n): to_tree(node.fetch(n), f"{_path}/{n}") for n in names}
        if n_children:
            return {
                f"domain_{i:06d}": to_tree(node.child(i), f"{_path}/{i}") for i in range(n_children)
            }
        if node.dtype().is_empty():
            return None
        return _leaf(node.value(), _path)
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for key, value in node.items():
            parts = [p for p in str(key).split("/") if p]
            if not parts:
                raise ConnectorError(f"empty key under {_path or '<root>'}")
            sub = to_tree(value, f"{_path}/{key}")
            cur = out
            for p in parts[:-1]:
                nxt = cur.setdefault(p, {})
                if not isinstance(nxt, dict):
                    raise ConnectorError(f"{_path}/{key}: {p!r} is both a leaf and a group")
                cur = nxt
            last = parts[-1]
            if isinstance(sub, dict) and isinstance(cur.get(last), dict):
                _merge(cur[last], sub)
            else:
                cur[last] = sub
        return out
    if isinstance(node, list | tuple) and node and all(isinstance(v, Mapping) for v in node):
        return {f"domain_{i:06d}": to_tree(v, f"{_path}/{i}") for i, v in enumerate(node)}
    return _leaf(node, _path)


def _merge(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def is_domain(tree: Any) -> bool:
    return isinstance(tree, dict) and (
        isinstance(tree.get("coordsets"), dict)
        or isinstance(tree.get("topologies"), dict)
        or (isinstance(tree.get("fields"), dict) and "values" not in tree)
    )


def _scalar(value: Any) -> float | None:
    if value is None or isinstance(value, str | dict):
        return None
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else None


def catalyst_state(tree: Mapping[str, Any]) -> dict[str, float]:
    """``timestep`` / ``cycle`` / ``time`` of a Catalyst execute node (empty for a bare mesh)."""
    cat = tree.get("catalyst", tree) if isinstance(tree, Mapping) else {}
    state = cat.get("state") if isinstance(cat, Mapping) and "channels" in cat else None
    out: dict[str, float] = {}
    if isinstance(state, Mapping):
        for key in ("timestep", "cycle", "time"):
            v = _scalar(state.get(key))
            if v is not None:
                out[key] = v
    return out


def split_domains(tree: Any, _prefix: str = "", _top: bool = True) -> list[tuple[str | None, dict]]:
    """``[(domain name | None, domain tree)]`` of a normalised tree: a single domain, a
    multi-domain mesh, nested groups of domains, or a Catalyst node with channels."""
    if not isinstance(tree, dict):
        raise UnsupportedSourceError("the Blueprint tree is not an object", hint=_HINT_SPEC)
    if _top and isinstance(tree.get("catalyst"), dict):
        tree = tree["catalyst"]
    if _top and isinstance(tree.get("channels"), dict):
        channels = tree["channels"]
        out: list[tuple[str | None, dict]] = []
        for cname, ch in channels.items():
            if not isinstance(ch, dict) or "data" not in ch:
                raise UnsupportedSourceError(
                    f"catalyst channel {cname!r} has no `data`",
                    hint="set catalyst/channels/<name>/type = 'mesh' and put the Blueprint mesh "
                    "under catalyst/channels/<name>/data",
                )
            ctype = text(ch.get("type", "mesh"))
            if ctype not in ("mesh", "multimesh"):
                raise UnsupportedSourceError(
                    f"catalyst channel {cname!r} has type {ctype!r}",
                    hint="only the Conduit Mesh Blueprint channel types 'mesh' and 'multimesh' "
                    "are read",
                )
            subs = split_domains(ch["data"], _top=False)
            single = len(channels) == 1
            for dname, dom in subs:
                if dname is None:
                    name: str | None = None if single else str(cname)
                else:
                    name = dname if single else f"{cname}/{dname}"
                out.append((name, dom))
        return out
    if is_domain(tree):
        return [(_prefix.rstrip("/") or None, tree)]
    out = []
    for name, child in tree.items():
        if name in ("blueprint_index", "protocol", "state") or not isinstance(child, dict):
            continue
        if is_domain(child):
            out.append((f"{_prefix}{name}", child))
        else:
            try:
                out.extend(split_domains(child, f"{_prefix}{name}/", False))
            except UnsupportedSourceError:
                continue
    if not out:
        raise UnsupportedSourceError(
            "no Blueprint mesh found in the tree (no `coordsets` / `topologies`)",
            hint="pass one domain ({coordsets, topologies, fields}), an object of domains, or a "
            "Catalyst node with catalyst/channels/<name>/data; " + _HINT_SPEC,
        )
    if _top and len(out) == 1 and out[0][0] is not None and "/" not in out[0][0]:
        # {"mesh": {...}}: the usual single named mesh of a relay file is not worth a zone prefix
        return [(None, out[0][1])]
    return out


# -- geometry ----------------------------------------------------------------------------------


@dataclass
class Geometry:
    kind: str
    points: np.ndarray | None = None
    cells: np.ndarray | None = None
    cell_types: np.ndarray | None = None
    faces: np.ndarray | None = None
    #: triangle -> source cell map of a triangulated surface
    parent: np.ndarray | None = None
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    #: point dimensions of a grid / structured block
    shape: list[int] | None = None
    n_points: int = 0
    n_cells: int = 0
    dim: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def census(self) -> dict[str, int]:
        if self.kind == "grid" and self.shape:
            name = {1: "line", 2: "quad", 3: "hexahedron"}[len(self.shape)]
            return {name: self.n_cells}
        if self.cell_types is not None:
            return meshutil.census(self.cell_types)
        if self.kind == "surface":
            return {"polygon": self.n_cells}
        return {}

    def bounds(self) -> list[float]:
        if self.kind == "grid":
            out = [0.0] * 6
            for i, ax in enumerate(AXES):
                if ax in self.coords and len(self.coords[ax]):
                    out[2 * i] = float(np.min(self.coords[ax]))
                    out[2 * i + 1] = float(np.max(self.coords[ax]))
            return out
        return meshutil.bounds(self.points)


def _child(tree: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(tree, Mapping) or key not in tree or tree[key] is None:
        raise UnsupportedSourceError(f"{where}: missing `{key}`", hint=_HINT_SPEC)
    return tree[key]


def _ints(value: Any, where: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind == "f":
        if not np.all(arr == np.round(arr)):
            raise ConnectorError(f"{where}: expected integers")
        arr = arr.astype(np.int64)
    if arr.dtype.kind not in "iu":
        raise ConnectorError(f"{where}: expected an integer array, got {arr.dtype}")
    return arr.astype(np.int64, copy=False).reshape(-1)


def _axes_of(names: Sequence[str], where: str) -> list[str]:
    names = [str(n) for n in names]
    if not names or any(n not in AXES for n in names):
        raise UnsupportedSourceError(
            f"{where}: coordinate axes {names} are not Cartesian",
            hint="only x/y/z coordsets are read; convert cylindrical (z, r) or spherical "
            "(r, theta, phi) coordinates to Cartesian in the adaptor",
        )
    return [a for a in AXES if a in names]


def grid_axes(coordset: Mapping[str, Any], where: str) -> dict[str, np.ndarray] | None:
    """1-D axes of a ``uniform`` / ``rectilinear`` coordset, ``None`` for an explicit one."""
    ctype = text(_child(coordset, "type", where))
    if ctype == "uniform":
        dims = _child(coordset, "dims", where)
        if not isinstance(dims, Mapping):
            raise UnsupportedSourceError(f"{where}: `dims` must hold i/j/k", hint=_HINT_SPEC)
        n = [int(_ints(dims[k], f"{where}/dims/{k}")[0]) for k in IJK if k in dims]
        if not n or min(n) < 1:
            raise ConnectorError(f"{where}: bad dims {n}")
        origin = coordset.get("origin") or {}
        spacing = coordset.get("spacing") or {}
        if isinstance(origin, Mapping) and origin:
            _axes_of(list(origin), f"{where}/origin")
        out = {}
        for i, ax in enumerate(AXES[: len(n)]):
            o = _scalar(origin.get(ax)) if isinstance(origin, Mapping) else None
            d = _scalar(spacing.get("d" + ax)) if isinstance(spacing, Mapping) else None
            out[ax] = (o or 0.0) + (1.0 if d is None else d) * np.arange(n[i], dtype=np.float64)
        return out
    if ctype == "rectilinear":
        values = _child(coordset, "values", where)
        if not isinstance(values, Mapping):
            raise UnsupportedSourceError(f"{where}: `values` must hold x/y/z", hint=_HINT_SPEC)
        axes = _axes_of(list(values), f"{where}/values")
        return {a: np.asarray(values[a], dtype=np.float64).reshape(-1) for a in axes}
    if ctype == "explicit":
        return None
    raise UnsupportedSourceError(
        f"{where}: unknown coordset type {ctype!r}",
        hint="supported: uniform, rectilinear, explicit",
    )


def _expand_axes(axes: Mapping[str, np.ndarray]) -> np.ndarray:
    """Grid axes -> ``(N, 3)`` points, first axis fastest (Blueprint / VTK point order)."""
    names = [a for a in AXES if a in axes]
    arrs = [np.asarray(axes[a], dtype=np.float64) for a in names]
    while len(arrs) < 3:
        arrs.append(np.zeros(1))
    zz, yy, xx = np.meshgrid(arrs[2], arrs[1], arrs[0], indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def explicit_points(coordset: Mapping[str, Any], where: str) -> np.ndarray:
    """``(N, 3)`` float64 points of any coordset (2-D sets get ``z = 0``)."""
    axes = grid_axes(coordset, where)
    if axes is not None:
        return _expand_axes(axes)
    values = _child(coordset, "values", where)
    if isinstance(values, Mapping):
        names = _axes_of(list(values), f"{where}/values")
        cols = [np.asarray(values[a], dtype=np.float64).reshape(-1) for a in names]
        if len({len(c) for c in cols}) != 1:
            raise ConnectorError(f"{where}: coordinate arrays have different lengths")
        pts = np.stack(cols, axis=1)
    else:
        pts = np.asarray(values, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] not in (1, 2, 3):
            raise UnsupportedSourceError(
                f"{where}: explicit `values` must be x/y/z arrays or an (N, 2|3) array",
                hint=_HINT_SPEC,
            )
    if pts.shape[1] < 3:
        pts = np.concatenate([pts, np.zeros((len(pts), 3 - pts.shape[1]))], axis=1)
    return np.ascontiguousarray(pts)


def _pad(conn: np.ndarray, sizes: np.ndarray, offsets: np.ndarray | None, where: str) -> np.ndarray:
    """General ``sizes``/``offsets`` connectivity -> ``(K, P)`` matrix padded with -1."""
    k = len(sizes)
    if k == 0:
        return np.full((0, 1), -1, dtype=np.int64)
    if offsets is None:
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    if len(offsets) != k:
        raise ConnectorError(f"{where}: {len(offsets)} offsets for {k} elements")
    if sizes.min() < 0 or offsets.min() < 0 or int((offsets + sizes).max()) > len(conn):
        raise ConnectorError(f"{where}: sizes/offsets run outside the connectivity array")
    width = max(int(sizes.max()), 1)
    cols = np.arange(width)[None, :]
    mask = cols < sizes[:, None]
    idx = offsets[:, None] + cols
    out = np.full((k, width), -1, dtype=np.int64)
    out[mask] = conn[idx[mask]]
    return out


def _polyhedra_points(
    face_ids: np.ndarray, sub: Mapping[str, Any], where: str
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the faces listed in ``face_ids``: ``(face position, vertex id)`` pairs."""
    sshape = text(_child(sub, "shape", where))
    sconn = _ints(_child(sub, "connectivity", where), f"{where}/connectivity")
    if sshape in ("polygonal", "mixed"):
        ssizes = _ints(_child(sub, "sizes", where), f"{where}/sizes")
        soffs = (
            _ints(sub["offsets"], f"{where}/offsets")
            if sub.get("offsets") is not None
            else np.concatenate([[0], np.cumsum(ssizes)[:-1]]).astype(np.int64)
        )
    elif sshape in ("tri", "quad"):
        n = SHAPES[sshape][1]
        ssizes = np.full(len(sconn) // n, n, dtype=np.int64)
        soffs = np.arange(len(ssizes), dtype=np.int64) * n
    else:
        raise UnsupportedSourceError(
            f"{where}: polyhedral faces of shape {sshape!r}", hint="use polygonal, tri or quad"
        )
    if len(face_ids) and (face_ids.min() < 0 or face_ids.max() >= len(ssizes)):
        raise ConnectorError(f"{where}: polyhedra reference faces outside the subelements")
    counts = ssizes[face_ids]
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(face_ids)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.repeat(soffs[face_ids], counts) + (np.arange(total) - start)
    return owner, sconn[idx]


def _element_block(
    elements: Mapping[str, Any], sub: Any, where: str
) -> tuple[np.ndarray, np.ndarray]:
    """One ``elements`` object -> padded ``cells`` and VTK ``cell_types``."""
    shape = text(_child(elements, "shape", where))
    conn = _ints(_child(elements, "connectivity", where), f"{where}/connectivity")
    if shape in SHAPES and SHAPES[shape][1]:
        vtk, n = SHAPES[shape]
        if len(conn) % n:
            raise ConnectorError(
                f"{where}: connectivity of {len(conn)} entries is not a multiple of {n} ({shape})"
            )
        cells = conn.reshape(-1, n)
        return cells, np.full(len(cells), vtk, dtype=np.uint8)
    if shape not in ("polygonal", "polyhedral", "mixed"):
        raise UnsupportedSourceError(
            f"{where}: unsupported element shape {shape!r}",
            hint="supported: " + ", ".join(SHAPES) + ", mixed",
        )
    sizes = _ints(_child(elements, "sizes", where), f"{where}/sizes")
    offsets = (
        _ints(elements["offsets"], f"{where}/offsets")
        if elements.get("offsets") is not None
        else np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    )
    k = len(sizes)
    if shape == "polygonal":
        return _pad(conn, sizes, offsets, where), np.full(k, meshutil.VTK_POLYGON, dtype=np.uint8)
    if shape == "polyhedral":
        types = np.full(k, meshutil.VTK_POLYHEDRON, dtype=np.uint8)
    else:
        smap = _child(elements, "shape_map", where)
        codes = _ints(_child(elements, "shapes", where), f"{where}/shapes")
        if len(codes) != k:
            raise ConnectorError(f"{where}: {len(codes)} shapes for {k} sizes")
        types = np.zeros(k, dtype=np.uint8)
        seen = np.zeros(k, dtype=bool)
        for name, code in smap.items():
            if name not in SHAPES:
                raise UnsupportedSourceError(
                    f"{where}/shape_map: unsupported shape {name!r}",
                    hint="supported: " + ", ".join(SHAPES),
                )
            sel = codes == int(_ints(code, f"{where}/shape_map/{name}")[0])
            vtk, n = SHAPES[name]
            if n and sel.any() and not (sizes[sel] == n).all():
                raise ConnectorError(f"{where}: a {name} element does not have {n} nodes")
            types[sel] = vtk
            seen |= sel
        if not seen.all():
            raise ConnectorError(
                f"{where}: shape code(s) {sorted(set(codes[~seen].tolist()))} are not in shape_map"
            )
    poly = types == meshutil.VTK_POLYHEDRON
    if not poly.any():
        return _pad(conn, sizes, offsets, where), types
    if not isinstance(sub, Mapping):
        raise UnsupportedSourceError(
            f"{where}: polyhedral elements need `subelements`", hint=_HINT_SPEC
        )
    cell_ids = np.arange(k)
    # (cell, vertex) pairs: fixed shapes keep their node order, polyhedra list unique points
    rep = np.repeat(cell_ids, sizes)
    pos = np.repeat(offsets, sizes) + (
        np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    )
    entries = conn[pos]
    is_poly_entry = poly[rep]
    owner, verts = _polyhedra_points(
        entries[is_poly_entry], sub, where.replace("elements", "subelements")
    )
    pcell = rep[is_poly_entry][owner]
    pairs = np.unique(np.stack([pcell, verts], axis=1), axis=0)
    cell_all = np.concatenate([rep[~is_poly_entry], pairs[:, 0]])
    vert_all = np.concatenate([entries[~is_poly_entry], pairs[:, 1]])
    order = np.argsort(cell_all, kind="stable")
    counts = np.bincount(cell_all, minlength=k)
    csr = np.concatenate([[0], np.cumsum(counts)])
    return meshutil.pad_cells(vert_all[order], csr), types


def _structured_cells(nodes: Sequence[int]) -> tuple[np.ndarray, int]:
    """Cells of a structured block with ``nodes`` points per axis (first axis fastest)."""
    n = [int(v) for v in nodes] + [1] * (3 - len(nodes))
    ni, nj, _nk = n
    active = [d for d in n if d > 1]
    ci, cj, ck = (max(d - 1, 1) for d in n)
    k, j, i = np.meshgrid(np.arange(ck), np.arange(cj), np.arange(ci), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()

    def pid(a: Any, b: Any, c: Any) -> Any:
        return a + ni * (b + nj * c)

    if len(active) == 3:
        cells = np.stack(
            [
                pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k),
                pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1),
                pid(i, j + 1, k + 1),
            ],
            axis=1,
        )  # fmt: skip
        return cells.astype(np.int64), meshutil.VTK_HEXAHEDRON
    if len(active) == 2:
        # the two axes that vary span the quads
        steps = [
            np.array(s) for s, d in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), n, strict=True) if d > 1
        ]
        a, b = steps
        base = (i, j, k)

        def at(off: np.ndarray) -> Any:
            return pid(base[0] + off[0], base[1] + off[1], base[2] + off[2])

        cells = np.stack([at(a * 0), at(a), at(a + b), at(b)], axis=1)
        return cells.astype(np.int64), meshutil.VTK_QUAD
    if len(active) == 1:
        total = int(np.prod(n))
        ids = np.arange(total - 1)
        return np.stack([ids, ids + 1], axis=1).astype(np.int64), meshutil.VTK_LINE
    return np.zeros((1, 1), dtype=np.int64), meshutil.VTK_VERTEX


def _finish_unstructured(
    points: np.ndarray, cells: np.ndarray, types: np.ndarray, where: str
) -> Geometry:
    if cells.size and (int(cells.max()) >= len(points) or int(cells.min()) < -1):
        raise ConnectorError(
            f"{where}: connectivity references points outside [0, {len(points)})",
            hint="Blueprint connectivity is 0-based into the coordset",
        )
    k = len(cells)
    dims = {meshutil.VTK_TYPE_DIM.get(int(t), 3) for t in np.unique(types)}
    geo = Geometry(kind="mesh", points=points, n_points=len(points), n_cells=k)
    geo.dim = max(dims) if dims else 0
    if k and dims == {2}:
        faces, parent = meshutil.triangulate(cells, types)
        geo.kind = "surface"
        geo.faces = faces.astype(np.int32)
        geo.parent = parent
        geo.metadata["source_cells"] = meshutil.census(types)
        geo.n_cells = len(faces)
        geo.metadata["n_source_cells"] = k
        return geo
    if k == 0 or dims == {0}:
        geo.kind = "point_cloud"
        geo.metadata["n_source_cells"] = k
        if k and not (cells.shape[1] == 1 and np.array_equal(cells[:, 0], np.arange(len(points)))):
            # a subset or a permutation of the points: keep exactly the listed ones
            geo.points = points[cells[:, 0]]
            geo.n_points = k
        geo.n_cells = 0
        return geo
    geo.cells = cells
    geo.cell_types = types
    if (types == meshutil.VTK_POLYHEDRON).any():
        geo.metadata["polyhedral"] = True
    return geo


def build_geometry(domain: Mapping[str, Any], topo_name: str, where: str = "") -> Geometry:
    """Geometry of one topology of one domain."""
    twhere = f"{where}topologies/{topo_name}"
    topo = _child(_child(domain, "topologies", where or "domain"), topo_name, twhere)
    if not isinstance(topo, Mapping):
        raise UnsupportedSourceError(f"{twhere} is not an object", hint=_HINT_SPEC)
    if topo.get("grid_function") is not None:
        raise UnsupportedSourceError(
            f"{twhere}: high-order meshes (`grid_function`) are not read",
            hint="refine to a linear mesh in the adaptor (e.g. MFEM/Ascent low-order refinement)",
        )
    ttype = text(_child(topo, "type", twhere))
    cname = text(_child(topo, "coordset", twhere))
    cwhere = f"{where}coordsets/{cname}"
    coordset = _child(_child(domain, "coordsets", where or "domain"), cname, cwhere)
    elements = topo.get("elements") if isinstance(topo.get("elements"), Mapping) else {}
    assert isinstance(elements, Mapping)
    dims_node = elements.get("dims")
    if isinstance(dims_node, Mapping) and ("offsets" in dims_node or "strides" in dims_node):
        raise UnsupportedSourceError(
            f"{twhere}: strided structured topologies are not read",
            hint="pass compact arrays (no elements/dims/offsets|strides)",
        )
    if ttype in ("uniform", "rectilinear"):
        axes = grid_axes(coordset, cwhere)
        if axes is None:
            raise UnsupportedSourceError(
                f"{twhere}: a {ttype} topology needs a uniform or rectilinear coordset",
                hint="use type 'structured' with elements/dims for explicit coordinates",
            )
        names = list(axes)
        meta_plane: dict[str, Any] = {}
        while len(names) > 2 and len(axes[names[-1]]) == 1:
            dropped = names.pop()
            meta_plane[f"plane_{dropped}"] = float(axes[dropped][0])
        coords = {a: axes[a] for a in names}
        shape = [len(coords[a]) for a in names]
        geo = Geometry(
            kind="grid",
            coords=coords,
            shape=shape,
            n_points=int(np.prod(shape)),
            n_cells=int(np.prod([max(s - 1, 1) for s in shape])),
            dim=len(shape),
            metadata=dict(meta_plane),
        )
        ctype = text(coordset["type"])
        geo.metadata["coordset_type"] = ctype
        if ctype == "uniform":
            geo.metadata["spacing"] = [
                float(c[1] - c[0]) if len(c) > 1 else 1.0 for c in coords.values()
            ]
            geo.metadata["origin"] = [float(c[0]) for c in coords.values()]
        return geo
    points = explicit_points(coordset, cwhere)
    if ttype == "points":
        geo = Geometry(kind="point_cloud", points=points, n_points=len(points), dim=0)
        return geo
    if ttype == "structured":
        axes = grid_axes(coordset, cwhere)
        if axes is not None:
            nodes = [len(axes[a]) for a in axes]
        else:
            if not isinstance(dims_node, Mapping):
                raise UnsupportedSourceError(
                    f"{twhere}: structured topology without elements/dims", hint=_HINT_SPEC
                )
            nodes = [
                int(_ints(dims_node[k], f"{twhere}/elements/dims/{k}")[0]) + 1
                for k in IJK
                if k in dims_node
            ]
        if int(np.prod(nodes)) != len(points):
            raise ConnectorError(
                f"{twhere}: elements/dims give {int(np.prod(nodes))} points, the coordset "
                f"has {len(points)}"
            )
        cells, vtk = _structured_cells(nodes)
        geo = _finish_unstructured(points, cells, np.full(len(cells), vtk, dtype=np.uint8), twhere)
        geo.shape = [int(v) for v in nodes]
        geo.metadata["structured_shape"] = [int(v) for v in nodes] + [1] * (3 - len(nodes))
        return geo
    if ttype == "unstructured":
        if not elements:
            raise UnsupportedSourceError(f"{twhere}: missing `elements`", hint=_HINT_SPEC)
        sub = topo.get("subelements")
        if "shape" in elements:
            cells, types = _element_block(elements, sub, f"{twhere}/elements")
        else:
            blocks = []
            for bname, block in elements.items():
                if not isinstance(block, Mapping) or "shape" not in block:
                    raise UnsupportedSourceError(
                        f"{twhere}/elements/{bname}: not an element block", hint=_HINT_SPEC
                    )
                bsub = sub.get(bname) if isinstance(sub, Mapping) and bname in sub else sub
                blocks.append(_element_block(block, bsub, f"{twhere}/elements/{bname}"))
            width = max(c.shape[1] for c, _ in blocks)
            cells = np.full((sum(len(c) for c, _ in blocks), width), -1, dtype=np.int64)
            row = 0
            for c, _ in blocks:
                cells[row : row + len(c), : c.shape[1]] = c
                row += len(c)
            types = np.concatenate([t for _, t in blocks])
        return _finish_unstructured(points, cells, types, twhere)
    raise UnsupportedSourceError(
        f"{twhere}: unknown topology type {ttype!r}",
        hint="supported: points, uniform, rectilinear, structured, unstructured",
    )


# -- fields ------------------------------------------------------------------------------------


def field_values(node: Mapping[str, Any], where: str) -> tuple[np.ndarray, list[str] | None]:
    """``values`` of a field as an ``(N, C)`` array plus component names (mcarray)."""
    values = _child(node, "values", where)
    if isinstance(values, Mapping):
        names = [str(k) for k in values]
        cols = []
        for k in names:
            if isinstance(values[k], Mapping):
                raise UnsupportedSourceError(
                    f"{where}/values/{k}: nested components are not read", hint=_HINT_SPEC
                )
            cols.append(np.asarray(values[k]).reshape(-1))
        if len({len(c) for c in cols}) != 1:
            raise ConnectorError(f"{where}: components have different lengths")
        return np.stack(cols, axis=1), names
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return arr[:, None], None
    return arr.reshape(arr.shape[0], -1), None


def _field_skip_reason(node: Any) -> str | None:
    if not isinstance(node, Mapping):
        return "not an object"
    if "values" not in node:
        if "matset_values" in node:
            return "material-dependent field (matset_values only)"
        return "no `values`"
    return None


def topology_fields(
    domain: Mapping[str, Any], topo_name: str, n_topologies: int
) -> list[tuple[str, Mapping[str, Any]]]:
    out = []
    fields = domain.get("fields")
    if not isinstance(fields, Mapping):
        return out
    for name, node in fields.items():
        if not isinstance(node, Mapping):
            continue
        t = node.get("topology")
        if t is None:
            if n_topologies != 1:
                continue
        elif text(t) != topo_name:
            continue
        out.append((str(name), node))
    return out


def _location(node: Mapping[str, Any], where: str) -> str:
    if node.get("basis") is not None and node.get("association") is None:
        raise UnsupportedSourceError(
            f"{where}: high-order field (`basis`) is not read",
            hint="project the field to vertex or element values in the adaptor",
        )
    assoc = text(_child(node, "association", where))
    if assoc not in ("vertex", "element"):
        raise UnsupportedSourceError(
            f"{where}: association {assoc!r}", hint="use 'vertex' or 'element'"
        )
    return "point" if assoc == "vertex" else "cell"


def build_fields(
    domain: Mapping[str, Any],
    topo_name: str,
    geo: Geometry,
    patterns: Sequence[str] | None,
    where: str = "",
    warnings: list[str] | None = None,
) -> dict[str, FieldData]:
    n_topo = len(domain.get("topologies") or {}) or 1
    out: dict[str, FieldData] = {}
    cell_grid: dict[str, tuple[np.ndarray, list[str] | None, str]] = {}
    for name, node in topology_fields(domain, topo_name, n_topo):
        fwhere = f"{where}fields/{name}"
        reason = _field_skip_reason(node)
        if reason:
            if warnings is not None:
                warnings.append(f"{fwhere}: skipped ({reason})")
            continue
        if not wants(name, patterns):
            continue
        if node.get("offsets") is not None or node.get("strides") is not None:
            raise UnsupportedSourceError(
                f"{fwhere}: strided field values are not read", hint="pass compact arrays"
            )
        loc = _location(node, fwhere)
        arr, comp = field_values(node, fwhere)
        units = text(node["units"]) if node.get("units") is not None else ""
        if geo.kind == "grid":
            assert geo.shape is not None
            shape = geo.shape if loc == "point" else [max(s - 1, 1) for s in geo.shape]
            if arr.shape[0] != int(np.prod(shape)):
                raise ConnectorError(
                    f"{fwhere}: {arr.shape[0]} values for a grid of {shape} "
                    f"{'points' if loc == 'point' else 'cells'}",
                    hint="grid fields are flat with i varying fastest",
                )
            full = shape + [1] * (3 - len(shape))
            a = arr.reshape(full[2], full[1], full[0], arr.shape[1]).transpose(2, 1, 0, 3)
            a = a.reshape(*shape, arr.shape[1])
            if loc == "cell":
                cell_grid[name] = (a, comp, units)
                continue
            data = a.astype(np.float32) if a.dtype.kind == "f" else np.ascontiguousarray(a)
            out[name] = FieldData(
                data=np.ascontiguousarray(data),
                dims=[*list(geo.coords), "component"],
                location="grid",
                units=units,
                component_names=comp,
            )
            continue
        if geo.kind == "point_cloud":
            n_src = int(geo.metadata.get("n_source_cells", 0))
            if loc == "cell" and n_src not in (0, geo.n_points):
                raise ConnectorError(f"{fwhere}: element field on a point topology of other size")
            want, loc_out = geo.n_points, "point"
        elif geo.kind == "surface" and loc == "cell":
            want, loc_out = int(geo.metadata.get("n_source_cells", geo.n_cells)), "cell"
        else:
            want, loc_out = (geo.n_points if loc == "point" else geo.n_cells), loc
        if arr.shape[0] != want:
            raise ConnectorError(
                f"{fwhere}: {arr.shape[0]} values but the topology {topo_name!r} has {want} "
                f"{'vertices' if loc == 'point' else 'elements'}",
                hint="check `association` and `topology` of the field",
            )
        if geo.kind == "surface" and loc == "cell" and geo.parent is not None:
            arr = arr[geo.parent]
        out[name] = as_field(arr, loc_out, units=units, component_names=comp)
    if cell_grid:
        assert geo.shape is not None
        axes = list(geo.coords)
        if out:
            # one set of grid axes per sample: element values are averaged onto the nodes
            for name, (a, comp, units) in cell_grid.items():
                out[name] = FieldData(
                    data=cells_to_nodes(a, geo.shape),
                    dims=[*axes, "component"],
                    location="grid",
                    units=units,
                    component_names=comp,
                    description="element-associated in the source, averaged to the grid nodes",
                )
            geo.metadata["cell_fields_averaged_to_nodes"] = sorted(cell_grid)
        else:
            geo.metadata["grid_location"] = "cell_centers"
            for name, (a, comp, units) in cell_grid.items():
                data = a.astype(np.float32) if a.dtype.kind == "f" else np.ascontiguousarray(a)
                out[name] = FieldData(
                    data=np.ascontiguousarray(data),
                    dims=[*axes, "component"],
                    location="grid",
                    units=units,
                    component_names=comp,
                )
    return out


def cell_center_coords(coords: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for ax, c in coords.items():
        c = np.asarray(c, dtype=np.float64)
        out[ax] = 0.5 * (c[:-1] + c[1:]) if len(c) > 1 else c
    return out


# -- zones -------------------------------------------------------------------------------------


@dataclass
class Zone:
    name: str
    domain_name: str | None
    topology: str
    domain: dict[str, Any]
    where: str = ""


def tree_zones(tree: Any) -> list[Zone]:
    """Every (domain, topology) pair of a normalised tree, named for the user."""
    domains = split_domains(tree)
    zones: list[Zone] = []
    for dname, dom in domains:
        topos = dom.get("topologies")
        if not isinstance(topos, dict) or not topos:
            raise UnsupportedSourceError(
                f"domain {dname or '<root>'} has no topologies",
                hint="a step that omits the mesh needs an earlier step that carried it "
                "(static_mesh streams)",
            )
        for tname in topos:
            if dname is None:
                name = str(tname)
            elif len(topos) == 1:
                name = dname
            else:
                name = f"{dname}/{tname}"
            zones.append(Zone(name, dname, str(tname), dom, f"{dname}/" if dname else ""))
    names = [z.name for z in zones]
    if len(set(names)) != len(names):
        raise ConnectorError(f"zone names are not unique: {names}")
    return zones


def tree_warnings(tree: Any) -> list[str]:
    out = []
    for dname, dom in split_domains(tree):
        present = [s for s in _IGNORED_SECTIONS if dom.get(s)]
        if present:
            out.append(
                f"{dname or 'mesh'}: {', '.join(present)} present and ignored (geometry and "
                "plain fields are read)"
            )
    return out


def state_of(domain: Mapping[str, Any]) -> dict[str, float]:
    state = domain.get("state")
    out: dict[str, float] = {}
    if isinstance(state, Mapping):
        for key in ("time", "cycle", "domain_id"):
            v = _scalar(state.get(key))
            if v is not None:
                out[key] = v
    return out


def tree_time(tree: Any) -> tuple[float | None, int | None]:
    """``(time, cycle)`` recorded in a tree: Catalyst state first, then the first domain."""
    cs = catalyst_state(tree) if isinstance(tree, Mapping) else {}
    time = cs.get("time")
    cycle = cs.get("cycle", cs.get("timestep"))
    if time is None or cycle is None:
        for _, dom in split_domains(tree):
            st = state_of(dom)
            time = st.get("time") if time is None else time
            cycle = st.get("cycle") if cycle is None else cycle
            break
    return time, (int(cycle) if cycle is not None else None)


def carry_mesh(tree: dict[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill ``coordsets`` / ``topologies`` that a step omitted from the previous step's tree
    (static-mesh streams). Returns ``tree`` (modified in place)."""
    if previous is None:
        return tree
    try:
        prev = dict(split_domains(dict(previous)))
    except UnsupportedSourceError:
        return tree
    for dname, dom in split_domains(tree):
        src = prev.get(dname)
        if src is None:
            continue
        for key in ("coordsets", "topologies"):
            if key not in dom and key in src:
                dom[key] = src[key]
    return tree


def describe_tree(tree: Any) -> tuple[list[ZoneInfo], list[FieldInfo], list[str]]:
    """Zones, fields and warnings of a normalised tree."""
    warnings = tree_warnings(tree)
    zones: list[ZoneInfo] = []
    merged: dict[tuple[str, str], FieldInfo] = {}
    for z in tree_zones(tree):
        geo = build_geometry(z.domain, z.topology, z.where)
        fields = build_fields(z.domain, z.topology, geo, None, z.where, warnings)
        zones.append(
            ZoneInfo(
                name=z.name,
                kind=geo.kind,
                role={"surface": "boundary", "grid": "grid", "point_cloud": "probes"}.get(
                    geo.kind, "volume"
                ),
                n_points=geo.n_points,
                n_cells=geo.n_cells,
                dim=geo.dim,
                cell_types=geo.census(),
                shape=geo.shape,
                bounds=geo.bounds(),
            )
        )
        for name, f in fields.items():
            key = (name, f.location)
            if key in merged:
                zl = merged[key].zones
                if zl is not None:
                    zl.append(z.name)
            else:
                merged[key] = FieldInfo(
                    name=name,
                    location=f.location,
                    components=f.components,
                    units=f.units,
                    component_names=f.component_names,
                    dtype=str(np.asarray(f.data[:1]).dtype),
                    zones=[z.name],
                )
    for fi in merged.values():
        if fi.zones is not None and len(fi.zones) == len(zones):
            fi.zones = None
    return zones, list(merged.values()), warnings


def _digest(arr: np.ndarray | None) -> bytes:
    if arr is None:
        return b""
    a = np.ascontiguousarray(arr)
    return (
        hashlib.blake2b(a.view(np.uint8).reshape(-1).data, digest_size=16).digest()
        + str(a.shape).encode()
    )


class StepConverter:
    """Turns the Blueprint tree of each step into step-form ``RawSample`` objects.

    Keeps what is needed between steps: the zone selection (resolved on the first step), the
    last mesh of every domain (for steps that omit it) and geometry digests, so that unchanged
    points / connectivity are passed as ``None`` ("unchanged") unless ``geometry="always"``
    (needed for ``time_as_samples`` conversions of streams whose topology changes).
    """

    def __init__(
        self,
        base_id: str,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        geometry: str = "changed",
    ) -> None:
        if geometry not in ("changed", "always"):
            raise ConnectorError(
                f"bad option geometry={geometry!r}", hint="use geometry=changed | always"
            )
        self.base_id = base_id
        self.zone_patterns = list(zones) if zones is not None else None
        self.fields = list(fields) if fields is not None else None
        self.metadata = dict(metadata or {})
        self.geometry = geometry
        self.selected: list[str] | None = None
        self.previous: dict[str, Any] | None = None
        self.digests: dict[str, tuple[bytes, bytes]] = {}
        self.warnings: list[str] = []
        self.n_steps = 0

    def convert(
        self, node: Any, *, time: float | None = None, time_index: int | None = None
    ) -> list[RawSample]:
        tree = to_tree(node)
        if not isinstance(tree, dict):
            raise UnsupportedSourceError("the step holds no Blueprint tree", hint=_HINT_SPEC)
        tree = carry_mesh(tree, self.previous)
        zones = tree_zones(tree)
        if self.selected is None:
            self.selected = match_names([z.name for z in zones], self.zone_patterns, what="zone")
            for w in tree_warnings(tree):
                self._warn(w)
        self.previous = tree
        out: list[RawSample] = []
        for z in zones:
            if z.name not in self.selected:
                continue
            geo = build_geometry(z.domain, z.topology, z.where)
            warns: list[str] = []
            fdata = build_fields(z.domain, z.topology, geo, self.fields, z.where, warns)
            for w in warns:
                self._warn(w)
            meta = dict(self.metadata)
            meta.update(geo.metadata)
            bp: dict[str, Any] = {"topology": z.topology}
            if z.domain_name:
                bp["domain"] = z.domain_name
            st = state_of(z.domain)
            if "domain_id" in st:
                bp["domain_id"] = int(st["domain_id"])
            meta["blueprint"] = bp
            s = RawSample(
                id=meshutil.zone_sample_id(self.base_id, z.name, len(self.selected)),
                kind=geo.kind,
                zone=z.name,
                fields=fdata,
                time=time,
                time_index=time_index,
                metadata=meta,
            )
            if geo.kind == "grid":
                s.coords = (
                    cell_center_coords(geo.coords)
                    if geo.metadata.get("grid_location") == "cell_centers"
                    else dict(geo.coords)
                )
            else:
                conn = geo.faces if geo.kind == "surface" else geo.cells
                dig = (_digest(geo.points), _digest(conn))
                old = self.digests.get(z.name)
                self.digests[z.name] = dig
                first = old is None or self.geometry == "always" or time is None
                if first or (old is not None and dig[1] != old[1]):
                    s.points = geo.points
                    s.cells, s.cell_types, s.faces = geo.cells, geo.cell_types, geo.faces
                    if geo.parent is not None and not np.array_equal(
                        geo.parent, np.arange(int(geo.metadata.get("n_source_cells", -1)))
                    ):
                        s.static["face_parent"] = as_field(geo.parent.astype(np.int32), "cell")
                elif old is not None and dig[0] != old[0]:
                    s.points = geo.points  # moving mesh, same connectivity
            out.append(s)
        self.n_steps += 1
        return out

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            log.warning("%s", message)


def mesh_to_samples(
    node: Any,
    sample_id: str = "mesh",
    *,
    time: float | None = None,
    time_index: int | None = None,
    zones: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[RawSample]:
    """Convert one Blueprint mesh (``conduit.Node`` or nested ``dict``) into ``RawSample``
    objects, one per (domain, topology). For a sequence of steps use :class:`StepConverter`."""
    conv = StepConverter(sample_id, zones=zones, fields=fields, metadata=metadata)
    return conv.convert(node, time=time, time_index=time_index)
