# SPDX-License-Identifier: Apache-2.0
"""Block-structured unstructured mesh (a ``meshio.Mesh`` or anything shaped like one) ->
zones -> ``RawSample``.

One rule set is shared by every connector built on it (Gmsh, MED, SU2, XDMF, the generic
meshio formats):

* **Split by topological dimension.** Volume cells, boundary faces and edges of one file are
  never dumped into one sample. Vertex (0-D) cells are ignored.
* **Named groups become zones.** Groups of the top dimension that are pairwise disjoint are
  the default zones (cells in no group form the zone ``ungrouped``). Overlapping top-level
  groups are offered as extra non-default zones next to one zone holding every top-level
  cell. Groups of lower dimension are boundary zones and are not read by default. Without
  groups there is one zone per dimension: ``volume`` (or ``surface`` / ``lines``) and
  ``boundary`` (plus ``edges`` for 1-D cells of a 3-D mesh).
* **2-D cells.** In 3-D space they are a ``surface`` (triangulated; cell data follows the
  parent map). A *planar* mesh (2-D points, or ``z == 0`` everywhere) whose top dimension is 2
  is a 2-D computational domain: by default (``planar_as="mesh"``) it stays a ``mesh`` with
  ``points (N, 2)``; ``planar_as="surface"`` turns it into a triangulated surface in z = 0.
* Points a zone does not reference are dropped and point data follows.
* Node ordering is VTK's. meshio already uses it except for the linear wedge. Element types
  the dataset vocabulary does not know (cubic and higher, quadratic wedges and pyramids) are
  reduced to their corner nodes.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError
from frontier_connectors.model import FieldData, FieldInfo, RawSample, ZoneInfo, as_field
from frontier_connectors.select import match_names, wants

PLANAR_CHOICES = ("mesh", "surface")

#: corner-node count and linear VTK type per element family (meshio type name without digits)
_FAMILY: dict[str, tuple[int, int, int]] = {
    # family: (topological dim, corner nodes, linear VTK type)
    "vertex": (0, 1, meshutil.VTK_VERTEX),
    "line": (1, 2, meshutil.VTK_LINE),
    "triangle": (2, 3, meshutil.VTK_TRIANGLE),
    "quad": (2, 4, meshutil.VTK_QUAD),
    "polygon": (2, 0, meshutil.VTK_POLYGON),
    "polyhedron": (3, 0, meshutil.VTK_POLYHEDRON),  # rows list the unique point ids
    "tetra": (3, 4, meshutil.VTK_TETRA),
    "hexahedron": (3, 8, meshutil.VTK_HEXAHEDRON),
    "wedge": (3, 6, meshutil.VTK_WEDGE),
    "pyramid": (3, 5, meshutil.VTK_PYRAMID),
}

#: higher-order types kept as they are (same node order in meshio and VTK)
_QUADRATIC: dict[str, int] = {
    "line3": meshutil.VTK_QUADRATIC_EDGE,
    "triangle6": meshutil.VTK_QUADRATIC_TRIANGLE,
    "quad8": meshutil.VTK_QUADRATIC_QUAD,
    "quad9": 28,
    "tetra10": meshutil.VTK_QUADRATIC_TETRA,
    "hexahedron20": meshutil.VTK_QUADRATIC_HEXAHEDRON,
    "hexahedron27": 29,
}

#: meshio (= Gmsh/Abaqus/Nastran) linear wedge -> VTK wedge (opposite base orientation)
_WEDGE_TO_VTK = np.array([0, 2, 1, 3, 5, 4])

_TOP_NAME = {3: "volume", 2: "surface", 1: "lines"}


def guess_domain(names: Sequence[str]) -> str:
    """Physics domain from result variable names (a hint only; formats do not record it)."""
    low = [n.lower() for n in names]

    def has(*keys: str) -> bool:
        return any(k in n for n in low for k in keys)

    structural = has("disp", "depl", "stress", "strain", "sief", "epsi", "eqps", "mises")
    # velocities exist in structural dynamics too: only pressure-like fields make it a flow
    flow = has("pres", "vort", "mach", "density", "momentum") or any(
        n in ("p", "rho", "u", "ux", "uy", "uz") for n in low
    )
    flow = flow or (has("vel") and not structural)
    thermal = has("temp") or any(n in ("t", "temp") for n in low)
    if structural and flow:
        return "multiphysics"
    if structural:
        return "structural"
    if flow:
        return "cfd"
    if thermal:
        return "thermal"
    return "other"


def planar_option(options: Mapping[str, Any] | None) -> str:
    """Validated ``options["planar_as"]`` (default ``"mesh"``)."""
    value = str((options or {}).get("planar_as", "mesh")).lower()
    if value not in PLANAR_CHOICES:
        raise ConnectorError(
            f"unknown planar_as option {value!r}",
            hint='use options={"planar_as": "mesh"} (2-D domain, the default) or "surface"',
        )
    return value


def is_planar(points: np.ndarray) -> bool:
    """2-D points, or a third coordinate that is exactly zero everywhere."""
    p = np.asarray(points)
    return p.ndim == 2 and (p.shape[1] < 3 or not np.any(p[:, 2]))


@dataclass
class Group:
    """A named set of cells (physical group, family group, marker, element set)."""

    name: str
    cells: np.ndarray  # global cell indices
    bc_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Zone:
    name: str
    kind: str
    role: str
    dim: int
    default: bool
    cells: np.ndarray  # global cell indices, sorted (point ids for a point_cloud zone)
    bc_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneGeometry:
    points: np.ndarray
    used: np.ndarray  # original point index of every kept point
    cells: np.ndarray | None = None
    cell_types: np.ndarray | None = None
    faces: np.ndarray | None = None
    #: per output cell/triangle: index into the zone's ``cells`` (None = identity)
    parent: np.ndarray | None = None


def _family(cell_type: str) -> str:
    return re.sub(r"\d+$", "", cell_type)


class ZonedMesh:
    """Points + typed cell blocks + groups, split into zones (see the module docstring).

    ``blocks`` are ``(meshio cell type, (k, p) connectivity)`` in meshio node order (or in
    VTK's with ``vtk_order=True``: the two differ for the linear wedge only); cell data handed
    to :meth:`fields` is indexed by the *global* cell index (blocks concatenated in order).
    """

    def __init__(
        self,
        points: np.ndarray,
        blocks: Sequence[tuple[str, np.ndarray]],
        *,
        groups: Sequence[Group] = (),
        point_sets: Mapping[str, np.ndarray] | None = None,
        planar_as: str = "mesh",
        vtk_order: bool = False,
        planar: bool | None = None,
    ) -> None:
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] not in (1, 2, 3):
            raise ConnectorError(
                f"points must be (N, 2|3), got {pts.shape}", hint="the mesh file is malformed"
            )
        self.warnings: list[str] = []
        self._vtk_order = vtk_order
        #: zone -> number of inverted cells that were mirrored to VTK's orientation
        self.reoriented: dict[str, int] = {}
        self._blocks: list[tuple[int, int, int, np.ndarray]] = []  # vtk, dim, corners, conn
        dims: list[np.ndarray] = []
        start = 0
        self._starts: list[int] = []
        for cell_type, data in blocks:
            vtk, dim, corners, conn = self._convert_block(str(cell_type), data, len(pts))
            self._blocks.append((vtk, dim, corners, conn))
            self._starts.append(start)
            start += len(conn)
            dims.append(np.full(len(conn), dim, dtype=np.int8))
        self.n_cells = start
        self.cell_dim = np.concatenate(dims) if dims else np.zeros((0,), dtype=np.int8)
        self.top_dim = int(self.cell_dim.max()) if self.n_cells else 0
        # ``planar`` overrides the test for sources that declare their dimension themselves
        self.planar = is_planar(pts) if planar is None else planar
        self.planar_as = planar_as
        #: 2-D domain kept as a mesh with (N, 2) points
        self.flat = self.planar and self.top_dim == 2 and planar_as == "mesh"
        if self.flat or (self.top_dim < 2 and pts.shape[1] < 3):
            self.points = np.ascontiguousarray(pts[:, :2]) if pts.shape[1] > 2 else pts
        elif pts.shape[1] < 3:
            pad = np.zeros((len(pts), 3 - pts.shape[1]), dtype=pts.dtype)
            self.points = np.concatenate([pts, pad], axis=1)
        else:
            self.points = pts
        self.zones: list[Zone] = []
        self._plan(list(groups), dict(point_sets or {}))
        self._cache: tuple[str, ZoneGeometry] | None = None

    # -- construction -------------------------------------------------------------------
    def _convert_block(
        self, cell_type: str, data: Any, n_points: int
    ) -> tuple[int, int, int, np.ndarray]:
        family = _family(cell_type)
        if family not in _FAMILY:
            self.warnings.append(f"cell type {cell_type!r} is not supported and was skipped")
            return meshutil.VTK_VERTEX, 0, 1, np.zeros((0, 1), dtype=np.int64)
        dim, corners, linear = _FAMILY[family]
        conn = np.asarray(data, dtype=np.int64)
        if conn.ndim == 1:
            conn = conn.reshape(-1, 1) if dim == 0 else conn.reshape(len(conn), -1)
        if conn.size and (conn.min() < 0 or conn.max() >= n_points):
            raise ConnectorError(
                f"{cell_type} connectivity references point {int(conn.max())} "
                f"but the mesh has {n_points} points",
                hint="the file is truncated or its node numbering is inconsistent",
            )
        if family in ("polygon", "polyhedron"):
            return linear, dim, conn.shape[1], conn
        if cell_type == family:
            if family == "wedge" and not self._vtk_order:
                conn = conn[:, _WEDGE_TO_VTK]
            return linear, dim, corners, conn
        if cell_type in _QUADRATIC:
            return _QUADRATIC[cell_type], dim, corners, conn
        self.warnings.append(
            f"{cell_type} elements were reduced to their {corners} corner nodes ({family})"
        )
        conn = conn[:, :corners]
        if family == "wedge" and not self._vtk_order:
            conn = conn[:, _WEDGE_TO_VTK]
        return linear, dim, corners, conn

    def _kind(self, dim: int) -> str:
        if dim == 2:
            return "mesh" if self.flat else "surface"
        return "mesh"

    def _plan(self, groups: list[Group], point_sets: dict[str, np.ndarray]) -> None:
        top = self.top_dim
        taken: set[str] = set()

        def add(name: str, dim: int, cells: np.ndarray, *, default: bool, g: Group | None) -> None:
            base, n, i = name or "zone", name or "zone", 2
            while n in taken:
                n, i = f"{base}_{i}", i + 1
            taken.add(n)
            kind = self._kind(dim)
            role = "boundary" if dim < top else ("volume" if kind == "mesh" and dim > 1 else "part")
            self.zones.append(
                Zone(
                    name=n,
                    kind=kind,
                    role=role,
                    dim=dim,
                    default=default,
                    cells=np.sort(np.asarray(cells, dtype=np.int64)),
                    bc_type=g.bc_type if g else None,
                    metadata=dict(g.metadata) if g else {},
                )
            )

        if top == 0:
            self.zones.append(
                Zone("points", "point_cloud", "probes", 0, True, np.arange(len(self.points)))
            )
            taken.add("points")
        by_dim: dict[int, list[Group]] = {}
        for g in groups:
            idx = np.unique(np.asarray(g.cells, dtype=np.int64))
            if len(idx) == 0:
                continue
            if idx[0] < 0 or idx[-1] >= self.n_cells:
                raise ConnectorError(
                    f"group {g.name!r} references cell {int(idx[-1])} of {self.n_cells}",
                    hint="the group table of the file is inconsistent with its elements",
                )
            gd = self.cell_dim[idx]
            present = [int(d) for d in np.unique(gd) if d > 0]
            for d in present:
                # a set spanning several dimensions keeps its name for the highest one
                name = g.name if d == max(present) else f"{g.name}_{d}d"
                by_dim.setdefault(d, []).append(
                    Group(name, idx[gd == d], bc_type=g.bc_type, metadata=g.metadata)
                )
        for dim in range(top, 0, -1):
            all_cells = np.nonzero(self.cell_dim == dim)[0]
            if len(all_cells) == 0:
                continue
            named = by_dim.get(dim, [])
            covered = np.zeros(self.n_cells, dtype=np.int32)
            for g in named:
                covered[g.cells] += 1
            rest = all_cells[covered[all_cells] == 0]
            if dim == top:
                if named and int(covered.max()) <= 1:
                    for g in named:
                        add(g.name, dim, g.cells, default=True, g=g)
                    if len(rest):
                        add("ungrouped", dim, rest, default=True, g=None)
                else:
                    top_name = "volume" if self._kind(dim) == "mesh" and dim > 1 else _TOP_NAME[dim]
                    add(top_name, dim, all_cells, default=True, g=None)
                    for g in named:
                        add(g.name, dim, g.cells, default=False, g=g)
            else:
                for g in named:
                    add(g.name, dim, g.cells, default=False, g=g)
                if len(rest):
                    add("boundary" if dim == top - 1 else "edges", dim, rest, default=False, g=None)
        for name, ids in point_sets.items():
            idx = np.unique(np.asarray(ids, dtype=np.int64))
            if len(idx) == 0 or idx[0] < 0 or idx[-1] >= len(self.points):
                continue
            n = name if name not in taken else f"{name}_nodes"
            taken.add(n)
            self.zones.append(Zone(n, "point_cloud", "probes", 0, False, idx))
        if not self.zones:
            raise ConnectorError(
                "the mesh has no cells and no points", hint="the file holds no geometry"
            )

    # -- queries ------------------------------------------------------------------------
    @property
    def zone_names(self) -> list[str]:
        return [z.name for z in self.zones]

    def zone(self, name: str) -> Zone:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)

    def select(self, zones: Sequence[str] | None) -> list[str]:
        """Zone names to read: the default zones, or the ones matching ``zones``."""
        if zones is None:
            return [z.name for z in self.zones if z.default] or self.zone_names[:1]
        return match_names(self.zone_names, zones, what="zone")

    def _zone_blocks(self, z: Zone, corners_only: bool) -> list[tuple[int, np.ndarray]]:
        out = []
        for (vtk, dim, corners, conn), start in zip(self._blocks, self._starts, strict=True):
            if dim != z.dim or len(conn) == 0:
                continue
            lo, hi = np.searchsorted(z.cells, [start, start + len(conn)])
            if hi > lo:
                rows = conn[z.cells[lo:hi] - start]
                out.append((vtk, rows[:, :corners] if corners_only else rows))
        return out

    def census(self, z: Zone) -> dict[str, int]:
        out: dict[str, int] = {}
        for vtk, rows in self._zone_blocks(z, False):
            name = meshutil.VTK_TYPE_NAMES.get(vtk, f"vtk_{vtk}")
            out[name] = out.get(name, 0) + len(rows)
        return out

    def geometry(self, name: str) -> ZoneGeometry:
        """Compact geometry of one zone (the last one asked for is cached)."""
        if self._cache is not None and self._cache[0] == name:
            return self._cache[1]
        z = self.zone(name)
        if z.kind == "point_cloud":
            geom = ZoneGeometry(points=self.points[z.cells], used=z.cells)
        elif z.kind == "surface":
            cells, _ = meshutil.pad_cell_blocks(self._zone_blocks(z, True))
            faces, parent = meshutil.triangulate(cells)
            pts, faces, used = meshutil.compact_points(self.points, faces)
            identity = len(parent) == len(cells) and np.array_equal(parent, np.arange(len(cells)))
            geom = ZoneGeometry(
                points=pts,
                used=used,
                faces=faces.astype(np.int32),
                parent=None if identity else parent,
            )
        else:
            cells, types = meshutil.pad_cell_blocks(self._zone_blocks(z, False))
            pts, cells, used = meshutil.compact_points(self.points, cells)
            n = orient_cells(pts, cells, types) if z.dim == 3 else 0
            if n and name not in self.reoriented:
                self.reoriented[name] = n
            geom = ZoneGeometry(points=pts, used=used, cells=cells, cell_types=types)
        self._cache = (name, geom)
        return geom

    def info(self, name: str) -> ZoneInfo:
        z = self.zone(name)
        g = self.geometry(name)
        n_cells = None if z.kind == "point_cloud" else len(z.cells)
        return ZoneInfo(
            name=z.name,
            kind=z.kind,
            role=z.role,
            n_points=len(g.points),
            n_cells=n_cells,
            dim=2 if self.flat else 3,
            cell_types=self.census(z),
            bounds=meshutil.bounds(g.points),
            bc_type=z.bc_type,
            default=z.default,
        )

    def infos(self) -> list[ZoneInfo]:
        return [self.info(z.name) for z in self.zones]

    def fields(
        self,
        name: str,
        point_data: Mapping[str, Any] | None,
        cell_data: Mapping[str, Any] | None,
        patterns: Sequence[str] | None = None,
        *,
        units: Mapping[str, str] | None = None,
        component_names: Mapping[str, list[str]] | None = None,
    ) -> tuple[dict[str, FieldData], dict[str, FieldData]]:
        """Slice whole-mesh arrays to one zone. Returns ``(fields, static)``: integer
        bookkeeping arrays (``gmsh:geometrical``, ``medit:ref``, ...) go to ``static``."""
        z = self.zone(name)
        g = self.geometry(name)
        out: dict[str, FieldData] = {}
        static: dict[str, FieldData] = {}
        units = units or {}
        component_names = component_names or {}

        def put(key: str, arr: np.ndarray, location: str) -> None:
            if arr.dtype.kind not in "fiub" or not wants(key, patterns):
                return
            if is_bookkeeping(key, arr):
                static[field_name(key)] = as_field(arr, location)
                return
            label = key if key not in out else f"{key}__{location}"
            out[label] = as_field(
                arr, location, units=units.get(key, ""), component_names=component_names.get(key)
            )

        for key, values in (point_data or {}).items():
            arr = np.asarray(values)
            if len(arr) != len(self.points):
                continue
            put(key, arr[g.used], "point")
        if z.kind != "point_cloud":
            for key, values in (cell_data or {}).items():
                arr = concat_cell_data(values)
                if arr is None or len(arr) != self.n_cells:
                    continue
                arr = arr[z.cells]
                if g.parent is not None:
                    arr = arr[g.parent]
                put(key, arr, "cell")
        return out, static


#: linear 3-D VTK type -> (three corners spanning a positive volume with corner 0, the node
#: permutation that mirrors the cell)
_ORIENT: dict[int, tuple[tuple[int, int, int], list[int]]] = {
    meshutil.VTK_TETRA: ((1, 2, 3), [0, 2, 1, 3]),
    meshutil.VTK_HEXAHEDRON: ((1, 3, 4), [0, 3, 2, 1, 4, 7, 6, 5]),
    meshutil.VTK_WEDGE: ((2, 1, 3), [0, 2, 1, 3, 5, 4]),
    meshutil.VTK_PYRAMID: ((1, 3, 4), [0, 3, 2, 1, 4]),
}


def orient_cells(points: np.ndarray, cells: np.ndarray, cell_types: np.ndarray) -> int:
    """Mirror, in place, the linear 3-D cells whose corner-0 Jacobian is negative so that every
    cell follows VTK's orientation. Returns how many cells were changed.

    Formats disagree on the orientation of the base face (Exodus, Gmsh and Abaqus wedges, MED
    3-D cells) and some writers (FEniCS) do not orient tetrahedra at all.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(cells) == 0:
        return 0
    flipped = 0
    for vtk_type, ((a, b, c), perm) in _ORIENT.items():
        rows = np.nonzero(cell_types == vtk_type)[0]
        if len(rows) == 0 or cells.shape[1] < len(perm):
            continue
        conn = cells[rows][:, : len(perm)]
        origin = p[conn[:, 0]]
        volume = np.einsum(
            "ij,ij->i",
            np.cross(p[conn[:, a]] - origin, p[conn[:, b]] - origin),
            p[conn[:, c]] - origin,
        )
        bad = rows[volume < 0]
        if len(bad):
            cells[bad[:, None], np.arange(len(perm))[None, :]] = cells[bad][:, perm]
            flipped += len(bad)
    return flipped


def field_name(key: str) -> str:
    """Field names are used as zarr array names: no ``:`` or ``/``."""
    return re.sub(r"[:/\\]+", "_", key).strip("_") or "field"


def is_bookkeeping(key: str, arr: np.ndarray) -> bool:
    """Integer tag arrays of the mesh generator (``gmsh:physical``, ``cell_tags``, ...)."""
    if ":" in key:  # meshio's "<format>:<tag>" arrays (tetgen:ref is stored as float)
        return True
    return arr.dtype.kind in "iub" and key in ("cell_tags", "point_tags", "cell_ids")


def concat_cell_data(values: Any) -> np.ndarray | None:
    """Per-block list (meshio ``cell_data`` entry) or whole-mesh array -> whole-mesh array."""
    if isinstance(values, np.ndarray):
        return values
    blocks = [np.asarray(v) for v in values if v is not None]
    if not blocks or len(blocks) != len(values):
        return None
    if len({b.shape[1:] for b in blocks}) != 1:
        return None
    return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]


def groups_from_labels(
    labels: np.ndarray,
    cell_dim: np.ndarray,
    names: Mapping[tuple[int, int], str] | None = None,
    *,
    prefix: str = "group",
) -> list[Group]:
    """One group per ``(dimension, label)`` pair of an integer label per cell; label 0 means
    "in no group". ``names`` maps ``(label, dim)`` to the group name."""
    labels = np.asarray(labels).astype(np.int64).ravel()
    names = names or {}
    out: list[Group] = []
    used_names: dict[str, int] = {}
    for dim in sorted({int(d) for d in np.unique(cell_dim)}, reverse=True):
        if dim == 0:
            continue
        sel = np.nonzero(cell_dim == dim)[0]
        for value in np.unique(labels[sel]):
            if value == 0:
                continue
            name = names.get((int(value), dim)) or f"{prefix}_{int(value)}"
            # the same label in two dimensions (Netgen, Medit) must stay two zones
            if name in used_names and used_names[name] != dim:
                name = f"{name}_{dim}d"
            used_names[name] = dim
            out.append(Group(name, sel[labels[sel] == value], metadata={"tag": int(value)}))
    return out


#: integer cell arrays that carry the region / material / marker id, by priority
_LABEL_KEYS = ("gmsh:physical", "cell_tags", "su2:tag")
_LABEL_SUFFIXES = (":ref", ":material", ":index", ":group", ":tag", ":physical")


def label_key(cell_data: Mapping[str, Any]) -> str | None:
    for key in _LABEL_KEYS:
        if key in cell_data:
            return key
    for key in cell_data:
        if key.endswith(_LABEL_SUFFIXES):
            return key
    return None


def block_dims(blocks: Sequence[tuple[str, Any]]) -> np.ndarray:
    """Topological dimension per global cell (0 for unknown types)."""
    parts = [
        np.full(len(data), _FAMILY.get(_family(str(t)), (0, 0, 0))[0], dtype=np.int8)
        for t, data in blocks
    ]
    return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.int8)


def zoned_from_meshio(
    mesh: Any,
    *,
    planar_as: str = "mesh",
    names: Mapping[tuple[int, int], str] | None = None,
    prefix: str = "group",
    extra_groups: Sequence[Group] = (),
    set_name: Any = None,
    global_sets: bool = False,
) -> tuple[ZonedMesh, dict[str, Any], dict[str, Any]]:
    """Build a :class:`ZonedMesh` from a ``meshio.Mesh``. Groups come from the format's
    integer label array (``gmsh:physical``, ``medit:ref``, ...) named through
    ``mesh.field_data`` (``{name: [tag, dim]}``), and from ``mesh.cell_sets`` (``set_name``
    may rewrite the set names).

    Returns ``(zoned, point_data, cell_data)``; the label array used for the zones is dropped
    from ``cell_data``.
    """
    blocks = [(c.type, c.data) for c in mesh.cells]
    blocks = [b for b in blocks if isinstance(b[1], np.ndarray) and b[1].dtype != object]
    if len(blocks) != len(mesh.cells):
        raise ConnectorError(
            "polyhedral cell blocks are not supported by this reader",
            hint="convert the mesh to VTU (meshio convert in.xx out.vtu) and use the vtk connector",
        )
    dims = block_dims(blocks)
    label_names: dict[tuple[int, int], str] = dict(names or {})
    for name, value in (getattr(mesh, "field_data", None) or {}).items():
        v = np.asarray(value).ravel()
        if v.dtype.kind in "iu" and len(v) == 2:
            label_names.setdefault((int(v[0]), int(v[1])), str(name))
    groups: list[Group] = list(extra_groups)
    cell_data = dict(mesh.cell_data)
    key = label_key(cell_data)
    if key is not None:
        labels = concat_cell_data(cell_data[key])
        if labels is not None and labels.ndim == 1 and len(labels) == len(dims):
            groups += groups_from_labels(labels, dims, label_names, prefix=prefix)
            cell_data.pop(key)
    starts = np.concatenate([[0], np.cumsum([len(d) for _, d in blocks])])
    for name, per_block in (getattr(mesh, "cell_sets", None) or {}).items():
        if ":" in name and name.split(":")[0] == "gmsh":
            continue  # gmsh:bounding_entities is topology bookkeeping, not a region
        if len(per_block) != len(blocks) or any(g.name == name for g in groups):
            continue
        # meshio's convention is block-local indices, but its FLAC3D reader stores global ones
        parts = [
            np.asarray(ids, dtype=np.int64) + (0 if global_sets else starts[i])
            for i, ids in enumerate(per_block)
            if ids is not None and len(ids)
        ]
        if parts:
            label = set_name(str(name)) if set_name is not None else str(name)
            groups.append(Group(label, np.concatenate(parts)))
    point_sets = {str(k): np.asarray(v) for k, v in (getattr(mesh, "point_sets", {}) or {}).items()}
    zoned = ZonedMesh(
        mesh.points, blocks, groups=groups, point_sets=point_sets, planar_as=planar_as
    )
    return zoned, dict(mesh.point_data), cell_data


def describe_fields(
    zoned: ZonedMesh,
    point_data: Mapping[str, Any] | None,
    cell_data: Mapping[str, Any] | None,
    *,
    units: Mapping[str, str] | None = None,
    component_names: Mapping[str, list[str]] | None = None,
    time_dependent: bool = False,
) -> list[FieldInfo]:
    """``FieldInfo`` rows for whole-mesh arrays (bookkeeping arrays are left out)."""
    out: list[FieldInfo] = []
    seen: set[str] = set()
    units = units or {}
    component_names = component_names or {}
    for location, data, n in (
        ("point", point_data or {}, len(zoned.points)),
        ("cell", cell_data or {}, zoned.n_cells),
    ):
        for key, values in data.items():
            arr = concat_cell_data(values) if location == "cell" else np.asarray(values)
            if arr is None or len(arr) != n or arr.dtype.kind not in "fiub":
                continue
            if is_bookkeeping(key, arr):
                continue
            name = key if key not in seen else f"{key}__{location}"
            seen.add(name)
            out.append(
                FieldInfo(
                    name=name,
                    location=location,
                    components=1 if arr.ndim == 1 else int(np.prod(arr.shape[1:])),
                    units=units.get(key, ""),
                    component_names=component_names.get(key),
                    dtype="float32" if arr.dtype.kind == "f" else str(arr.dtype),
                    time_dependent=time_dependent,
                )
            )
    return out


#: one time step of whole-mesh data: (time index, time, point_data, cell_data)
Step = tuple[int | None, float | None, Mapping[str, Any], Mapping[str, Any]]


def iter_samples(
    zoned: ZonedMesh,
    base_id: str,
    steps: Iterable[Step],
    *,
    zones: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    component_names: Mapping[str, list[str]] | None = None,
    static_point_data: Mapping[str, Any] | None = None,
    static_cell_data: Mapping[str, Any] | None = None,
) -> Iterator[RawSample]:
    """Yield step-form samples: ``steps`` is consumed lazily (one step in memory); the first
    step of every zone carries the geometry, later ones only the fields. ``static_*`` are the
    time-independent arrays of a transient source: they go to ``RawSample.static`` once."""
    selected = zoned.select(zones)
    first = True
    for time_index, time, point_data, cell_data in steps:
        for name in selected:
            z = zoned.zone(name)
            g = zoned.geometry(name)
            meta = dict(metadata or {})
            meta["zone_role"] = z.role
            if name in zoned.reoriented:
                meta["reoriented_cells"] = zoned.reoriented[name]
            if z.bc_type is not None:
                meta["bc_type"] = z.bc_type
            meta.update(z.metadata)
            sample = RawSample(
                id=meshutil.zone_sample_id(base_id, name, len(selected)),
                kind=z.kind,
                zone=name,
                time=time,
                time_index=time_index,
                metadata=meta,
            )
            if first:
                sample.points = g.points
                sample.cells = g.cells
                sample.cell_types = g.cell_types
                sample.faces = g.faces
            sample.fields, static = zoned.fields(
                name, point_data, cell_data, fields, units=units, component_names=component_names
            )
            if first:
                sample.static = static
                if static_point_data or static_cell_data:
                    extra, book = zoned.fields(
                        name,
                        static_point_data,
                        static_cell_data,
                        fields,
                        units=units,
                        component_names=component_names,
                    )
                    sample.static.update(book)
                    sample.static.update(extra)
                if g.parent is not None:
                    sample.static["face_parent"] = as_field(g.parent.astype(np.int32), "cell")
            yield sample
        first = False


def grid_cell_fields_to_nodes(sample: RawSample) -> list[str]:
    """Move the cell-centred fields of a ``grid`` sample onto its nodes (every node takes the
    mean of the cells around it). The dataset format stores grid fields on the grid axes only,
    so cell-centred arrays (dims ``x_cell`` ...) cannot be written as they are. Returns the
    names of the fields that were converted."""
    if sample.kind != "grid":
        return []
    axes = [a for a in ("x", "y", "z") if a in sample.coords]
    target = [len(sample.coords[a]) for a in axes]
    moved: list[str] = []
    for name, f in list(sample.fields.items()):
        if not f.dims or not f.dims[0].endswith("_cell") or len(f.dims) != len(axes) + 1:
            continue
        data = np.asarray(f.data, dtype=np.float64)
        total = np.zeros((*target, data.shape[-1]))
        count = np.zeros((*target, 1))
        spans = [(0, 1) if t == n + 1 else (0,) for t, n in zip(target, data.shape, strict=False)]
        for shift in itertools.product(*spans):
            where = tuple(
                slice(o, o + n) for o, n in zip(shift, data.shape[: len(axes)], strict=True)
            )
            total[where] += data
            count[where] += 1
        f.data = np.ascontiguousarray(total / np.maximum(count, 1), dtype=np.float32)
        f.dims = [*axes, "component"]
        moved.append(name)
    if moved:
        sample.metadata["cell_fields_averaged_to_nodes"] = moved
    return moved
