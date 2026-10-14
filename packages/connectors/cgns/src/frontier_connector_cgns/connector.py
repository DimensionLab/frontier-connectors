# SPDX-License-Identifier: Apache-2.0
"""CGNS (CFD General Notation System) files.

CGNS/HDF5 files are read with ``h5py`` alone, following the public SIDS and SIDS-to-HDF5
documents: every CGNS node is an HDF5 group with the attributes ``name``, ``label``
(``Zone_t``, ``FlowSolution_t``, ...) and ``type`` (``MT``, ``I4``, ``I8``, ``R4``, ``R8``,
``C1``, ``LK``); its value is the dataset ``" data"`` stored transposed (Fortran order), so a
structured array of logical shape ``(ni, nj, nk)`` appears as ``(nk, nj, ni)`` and ravels
i-fastest. A link node (``LK``) holds an HDF5 soft/external link called ``" link"``.

What becomes what:

* every ``Zone_t`` is one zone (``<Base>/<Zone>`` when the file has several bases). A structured
  zone becomes hexahedra (quads in 2-D) with ``metadata["structured_shape"]``; an unstructured
  zone keeps its element sections of the zone's dimension as VTK cells in VTK node order
  (``NGON_n``/``NFACE_n`` become polyhedra that list their unique point ids);
* ``BC_t`` patches and lower-dimensional element sections become surface zones
  ``<zone>/<name>`` (``role="boundary"``, not read unless asked for);
* ``FlowSolution_t`` arrays become point or cell fields, ``VelocityX/Y/Z`` style components
  are assembled into one vector field;
* ``BaseIterativeData_t`` + ``ZoneIterativeData_t`` give the time steps of a transient file,
  a numbered series of files (``sol_0001.cgns`` ...) gives one step per file.

Legacy ADF-based CGNS files are not HDF5: they are read through VTK's ``vtkCGNSReader``
(``options={"engine": "vtk"}`` forces that reader for any file).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import _vtk, meshutil
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
    match_extension,
)
from frontier_connectors.paths import h5_check_file, h5_open
from frontier_connectors.select import select_time_indices, wants
from frontier_connectors.units import normalize_unit

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
#: ADF files start with ``@(#)ADF Database Version``; real files set the high bit of the four
#: ``@(#)`` bytes, so only the text after them is compared
ADF_MAGIC = b"ADF Database Version"

CORRUPT_HINT = (
    "the file is truncated or not a conformant CGNS file; write it again (the CGNS tool "
    "`cgnscheck <file>` reports what is wrong), or try options={'engine': 'vtk'}"
)
ADF_HINT = (
    "convert it to CGNS/HDF5 with the CGNS tool `cgnsconvert -h <file>.cgns`, or read it "
    "through VTK with options={'engine': 'vtk'} (CLI: --option engine=vtk)"
)

# -- element types (SIDS ElementType_t) ---------------------------------------------------------

_ELEMENT_NAMES = (
    "ElementTypeNull", "ElementTypeUserDefined", "NODE", "BAR_2", "BAR_3", "TRI_3", "TRI_6",
    "QUAD_4", "QUAD_8", "QUAD_9", "TETRA_4", "TETRA_10", "PYRA_5", "PYRA_14", "PENTA_6",
    "PENTA_15", "PENTA_18", "HEXA_8", "HEXA_20", "HEXA_27", "MIXED", "PYRA_13", "NGON_n",
    "NFACE_n", "BAR_4", "TRI_9", "TRI_10", "QUAD_12", "QUAD_16", "TETRA_16", "TETRA_20",
    "PYRA_21", "PYRA_29", "PYRA_30", "PENTA_24", "PENTA_38", "PENTA_40", "HEXA_32", "HEXA_56",
    "HEXA_64", "BAR_5", "TRI_12", "TRI_15", "QUAD_P4_16", "QUAD_25", "TETRA_22", "TETRA_34",
    "TETRA_35", "PYRA_P4_29", "PYRA_50", "PYRA_55", "PENTA_33", "PENTA_66", "PENTA_75",
    "HEXA_44", "HEXA_98", "HEXA_125",
)  # fmt: skip
MIXED, NGON_N, NFACE_N = 20, 22, 23
_PREFIX_DIM = {
    "NODE": 0,
    "BAR": 1,
    "TRI": 2,
    "QUAD": 2,
    "TETRA": 3,
    "PYRA": 3,
    "PENTA": 3,
    "HEXA": 3,
}


def _element_table() -> tuple[dict[int, int], dict[int, int]]:
    npe: dict[int, int] = {}
    dim: dict[int, int] = {}
    for code, name in enumerate(_ELEMENT_NAMES):
        prefix = name.split("_")[0]
        if prefix not in _PREFIX_DIM:
            continue
        npe[code] = 1 if name == "NODE" else int(name.rsplit("_", 1)[1])
        dim[code] = _PREFIX_DIM[prefix]
    return npe, dim


#: nodes per element / topological dimension of the fixed-size element types
_NPE, _DIM = _element_table()

_HEX20 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15)
_WEDGE15 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11)
#: CGNS element code -> (VTK cell type, permutation). ``vtk_cell[j] = cgns_cell[perm[j]]``;
#: ``None`` = same node order. CGNS lists the mid-edge nodes of prisms and hexahedra as
#: bottom, vertical, top; VTK as bottom, top, vertical. PYRA_14 loses its base-centre node
#: (VTK has no 14-node pyramid).
_VTK_ELEMENTS: dict[int, tuple[int, tuple[int, ...] | None]] = {
    2: (1, None),
    3: (3, None),
    4: (21, None),
    5: (5, None),
    6: (22, None),
    7: (9, None),
    8: (23, None),
    9: (28, None),
    10: (10, None),
    11: (24, None),
    12: (14, None),
    13: (27, tuple(range(13))),
    14: (13, None),
    15: (26, _WEDGE15),
    16: (32, (*_WEDGE15, 15, 16, 17)),
    17: (12, None),
    18: (25, _HEX20),
    19: (29, (*_HEX20, 24, 22, 21, 23, 20, 25, 26)),
    21: (27, None),
}


def element_name(code: int) -> str:
    """SIDS name of an ``ElementType_t`` code (``17`` -> ``"HEXA_8"``)."""
    return _ELEMENT_NAMES[code] if 0 <= code < len(_ELEMENT_NAMES) else f"ElementType({code})"


def _unsupported_element(code: int, where: str) -> UnsupportedSourceError:
    return UnsupportedSourceError(
        f"{where}: element type {element_name(code)} is not supported by the h5py reader",
        hint="linear and quadratic elements, MIXED, NGON_n and NFACE_n are read natively; "
        "read this file through VTK with options={'engine': 'vtk'} (CLI: --option engine=vtk)",
    )


# -- units and boundary types -------------------------------------------------------------------

_UNIT_ALIASES = {"poundmass": "lbm", "nauticalmile": "nmi", "mile": "mi"}
_UNIT_KEYS = ("mass", "length", "time", "temperature", "angle")

#: dimensional exponents (mass, length, time, temperature, angle) of SIDS standard names, used
#: when a dimensional array carries no DimensionalExponents_t of its own
_STANDARD_EXPONENTS: dict[str, tuple[float, ...]] = {
    "Density": (1, -3, 0, 0, 0),
    "Pressure": (1, -1, -2, 0, 0),
    "PressureStagnation": (1, -1, -2, 0, 0),
    "PressureDynamic": (1, -1, -2, 0, 0),
    "Temperature": (0, 0, 0, 1, 0),
    "TemperatureStagnation": (0, 0, 0, 1, 0),
    "Velocity": (0, 1, -1, 0, 0),
    "VelocityMagnitude": (0, 1, -1, 0, 0),
    "VelocitySound": (0, 1, -1, 0, 0),
    "Momentum": (1, -2, -1, 0, 0),
    "MomentumMagnitude": (1, -2, -1, 0, 0),
    "EnergyStagnationDensity": (1, -1, -2, 0, 0),
    "ViscosityMolecular": (1, -1, -1, 0, 0),
    "ViscosityEddy": (1, -1, -1, 0, 0),
    "ViscosityKinematic": (0, 2, -1, 0, 0),
    "TurbulentDistance": (0, 1, 0, 0, 0),
    "TurbulentEnergyKinetic": (0, 2, -2, 0, 0),
}

_BC_PREFIXES = (
    ("BCWall", "wall"),
    ("BCInflow", "inlet"),
    ("BCTunnelInflow", "inlet"),
    ("BCOutflow", "outlet"),
    ("BCTunnelOutflow", "outlet"),
    ("BCSymmetry", "symmetry"),
    ("BCFarfield", "farfield"),
    ("BCAxisymmetricWedge", "wedge"),
    ("BCDegenerate", "degenerate"),
)


def bc_type_id(cgns_type: str) -> str:
    """``BCType_t`` name -> the lower-case vocabulary of the other CFD connectors
    (``"BCWallViscous"`` -> ``"wall"``, ``"BCInflowSubsonic"`` -> ``"inlet"``); names without
    a common counterpart lose their ``BC`` prefix (``"BCExtrapolate"`` -> ``"extrapolate"``)."""
    name = cgns_type.strip()
    for prefix, generic in _BC_PREFIXES:
        if name.startswith(prefix):
            return generic
    if name in ("", "Null", "UserDefined", "FamilySpecified", "BCTypeNull", "BCTypeUserDefined"):
        return "unknown"
    return (name[2:] if name.startswith("BC") else name).lower()


def _unit_from_exponents(exponents: Sequence[float], units: Mapping[str, str]) -> str:
    """``(1, -1, -2, 0, 0)`` with kg/m/s -> ``"kg/m/s^2"``; empty when a needed unit is unknown."""
    num, den = [], []
    for key, p in zip(_UNIT_KEYS, exponents, strict=False):
        if p == 0:
            continue
        unit = units.get(key)
        if not unit:
            return ""
        mag = abs(float(p))
        (num if p > 0 else den).append(unit if mag == 1 else f"{unit}^{mag:g}")
    if not num and not den:
        return ""
    return ("*".join(num) if num else "1") + "".join(f"/{d}" for d in den)


# -- HDF5 node access ---------------------------------------------------------------------------


def _attr(group: Any, key: str) -> str:
    try:
        value = group.attrs.get(key)
    except (OSError, RuntimeError, KeyError):
        return ""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("ascii", "replace").strip("\x00 ")
    return str(value).strip("\x00 ")


class _Node:
    """One CGNS node: an HDF5 group with its links resolved."""

    __slots__ = ("_kids", "_log", "group", "label", "name")

    def __init__(self, name: str, label: str, group: Any, log: list[str]) -> None:
        self.name = name
        self.label = label
        self.group = group
        self._log = log
        self._kids: list[_Node] | None = None

    @property
    def path(self) -> str:
        return str(getattr(self.group, "name", self.name))

    @property
    def children(self) -> list[_Node]:
        if self._kids is None:
            self._kids = self._load_children()
        return self._kids

    def _load_children(self) -> list[_Node]:
        out: list[_Node] = []
        for key in list(self.group.keys()):
            if key.startswith(" "):  # " data", " link", " path", " file", " format", ...
                continue
            try:
                child = self.group[key]
            except (KeyError, OSError, RuntimeError):
                self._log.append(f"node {self.path}/{key} cannot be opened (skipped)")
                continue
            if not hasattr(child, "keys"):
                continue
            name = _attr(child, "name") or key
            if _attr(child, "type") == "LK":
                try:
                    child = child[" link"]
                except (KeyError, OSError, RuntimeError):
                    self._log.append(
                        f"link {self.path}/{key} points at a file or node that is missing (skipped)"
                    )
                    continue
                if not hasattr(child, "keys"):
                    continue
            out.append(_Node(name, _attr(child, "label"), child, self._log))
        return out

    def all(self, label: str) -> list[_Node]:
        return [c for c in self.children if c.label == label]

    def first(self, label: str) -> _Node | None:
        return next((c for c in self.children if c.label == label), None)

    def child(self, name: str) -> _Node | None:
        return next((c for c in self.children if c.name == name), None)

    def has_data(self) -> bool:
        return " data" in self.group

    def shape(self) -> tuple[int, ...]:
        """Shape of the value as stored (the Fortran shape reversed); ``()`` without data."""
        if not self.has_data():
            return ()
        return tuple(int(s) for s in self.group[" data"].shape)

    def array(self) -> np.ndarray:
        """The node value in HDF5 order (C order of the stored, transposed array)."""
        try:
            return np.asarray(self.group[" data"][()])
        except KeyError as e:
            raise ConnectorError(
                f"CGNS node {self.path} ({self.label}) has no data", hint=CORRUPT_HINT
            ) from e
        except (OSError, RuntimeError, ValueError) as e:
            raise ConnectorError(
                f"cannot read the data of CGNS node {self.path}: {e}", hint=CORRUPT_HINT
            ) from e

    def texts(self) -> list[str]:
        """Character data as strings: one per row of a 2-D ``C1`` array."""
        if not self.has_data():
            return []
        arr = self.array()
        if arr.dtype.kind in "SU":
            return [_decode(v) for v in np.atleast_1d(arr).ravel()]
        if arr.dtype.kind not in "iu":
            return []
        rows = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
        return [_decode(row.astype(np.uint8).tobytes()) for row in rows]

    def text(self) -> str:
        return "".join(self.texts()[:1])


def _decode(value: Any) -> str:
    raw = value if isinstance(value, bytes) else str(value).encode("latin-1", "replace")
    return raw.decode("latin-1").strip("\x00 ")


# -- parsed tree --------------------------------------------------------------------------------


@dataclass
class _Solution:
    name: str
    location: str
    rind: np.ndarray | None
    arrays: list[_Node]
    node: _Node


@dataclass
class _Section:
    name: str
    etype: int
    start: int
    end: int
    node: _Node

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    def may_have_dim(self, dim: int) -> bool:
        if self.etype == MIXED:
            return True
        if self.etype == NGON_N:
            return dim == 2
        if self.etype == NFACE_N:
            return dim == 3
        return _DIM.get(self.etype) == dim


@dataclass
class _BC:
    name: str
    cgns_type: str
    family: str | None
    location: str
    point_range: np.ndarray | None
    list_node: _Node | None

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": bc_type_id(self.cgns_type), "cgns_type": self.cgns_type}
        if self.family:
            out["family"] = self.family
        out["location"] = self.location
        return out


@dataclass
class _Block:
    """The elements of one section: padded 0-based connectivity in VTK node order."""

    cells: np.ndarray
    types: np.ndarray
    dims: np.ndarray


@dataclass
class _Base:
    name: str
    cell_dim: int
    phys_dim: int
    node: _Node
    families: dict[str, str] = field(default_factory=dict)
    zones: list[_Zone] = field(default_factory=list)


class _Zone:
    """One ``Zone_t`` with lazily read (and releasable) geometry."""

    def __init__(self, base: _Base, node: _Node, name: str) -> None:
        self.base = base
        self.node = node
        self.name = name
        self.raw_name = node.name
        size = np.atleast_1d(node.array()).astype(np.int64)
        ztype = node.first("ZoneType_t")
        kind = ztype.text() if ztype is not None else ""
        if kind not in ("Structured", "Unstructured"):
            raise ConnectorError(
                f"zone {name!r} has no valid ZoneType_t (found {kind!r})", hint=CORRUPT_HINT
            )
        self.structured = kind == "Structured"
        if size.size < 3 or size.size % 3 != 0:
            raise ConnectorError(
                f"zone {name!r}: size array has {size.size} entries", hint=CORRUPT_HINT
            )
        size = size.reshape(3, -1)
        #: structured: vertices per index direction (ni, nj[, nk])
        self.shape: list[int] = [int(v) for v in size[0]] if self.structured else []
        self.n_points = int(np.prod(size[0]))
        self.declared_cells = int(np.prod(size[1]))
        self.grids: dict[str, _Node] = {g.name: g for g in node.all("GridCoordinates_t")}
        self.sections: list[_Section] = []
        if not self.structured:
            for sec in node.all("Elements_t"):
                self.sections.append(_parse_section(sec, name))
            self.sections.sort(key=lambda s: s.start)
        self.solutions = [_parse_solution(s) for s in node.all("FlowSolution_t")]
        self.bcs: list[_BC] = []
        for zbc in node.all("ZoneBC_t"):
            self.bcs.extend(_parse_bc(b, base.families) for b in zbc.all("BC_t"))
        family = node.first("FamilyName_t")
        self.family = family.text() if family is not None else None
        self.flow_pointers: list[str] = []
        self.grid_pointers: list[str] = []
        zid = node.first("ZoneIterativeData_t")
        if zid is not None:
            fp, gp = zid.child("FlowSolutionPointers"), zid.child("GridCoordinatesPointers")
            self.flow_pointers = fp.texts() if fp is not None else []
            self.grid_pointers = gp.texts() if gp is not None else []
        self._blocks: dict[str, _Block] = {}
        self._points: dict[str, np.ndarray] = {}
        self._faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._bc_faces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._census: dict[str, dict[int, dict[str, int]]] = {}

    def release(self) -> None:
        """Drop the cached geometry arrays (called after every time step)."""
        self._blocks.clear()
        self._points.clear()
        self._faces = None
        self._bc_faces.clear()

    # -- dimensions and counts ---------------------------------------------------------
    @property
    def dim(self) -> int:
        """Topological dimension of the zone's cells."""
        if self.structured:
            return len(self.shape)
        present = [d for d in (3, 2, 1) if d <= self.base.cell_dim and self.count(d) > 0]
        return present[0] if present else 0

    def section_census(self, sec: _Section) -> dict[int, dict[str, int]]:
        """``{dimension: {vtk type name: count}}`` of one section (reads MIXED sections)."""
        if sec.name in self._census:
            return self._census[sec.name]
        out: dict[int, dict[str, int]] = {}
        if sec.count <= 0:
            pass
        elif sec.etype == NGON_N:
            out[2] = {"polygon": sec.count}
        elif sec.etype == NFACE_N:
            out[3] = {"polyhedron": sec.count}
        elif sec.etype == MIXED:
            block = self.block(sec)
            for d in np.unique(block.dims):
                out[int(d)] = meshutil.census(block.types[block.dims == d])
        elif sec.etype in _VTK_ELEMENTS:
            vtk_type = _VTK_ELEMENTS[sec.etype][0]
            out[_DIM[sec.etype]] = {meshutil.VTK_TYPE_NAMES[vtk_type]: sec.count}
        else:
            raise _unsupported_element(sec.etype, f"zone {self.name!r}, section {sec.name!r}")
        self._census[sec.name] = out
        return out

    def census(self, dim: int) -> dict[str, int]:
        if self.structured:
            name = {3: "hexahedron", 2: "quad", 1: "line"}[len(self.shape)]
            return {name: int(np.prod([max(n - 1, 0) for n in self.shape]))}
        out: dict[str, int] = {}
        for sec in self.sections:
            for name, n in self.section_census(sec).get(dim, {}).items():
                out[name] = out.get(name, 0) + n
        return out

    def count(self, dim: int) -> int:
        return sum(self.census(dim).values())

    @property
    def has_nface(self) -> bool:
        return any(s.etype == NFACE_N for s in self.sections)

    # -- coordinates -------------------------------------------------------------------
    def grid_name(self, step: int | None) -> str:
        if step is not None and step < len(self.grid_pointers):
            name = self.grid_pointers[step]
            if name in self.grids:
                return name
        if "GridCoordinates" in self.grids or not self.grids:
            return "GridCoordinates"
        return next(iter(self.grids))

    @property
    def moving(self) -> bool:
        return len({p for p in self.grid_pointers if p in self.grids}) > 1

    def points(self, grid: str) -> np.ndarray:
        if grid in self._points:
            return self._points[grid]
        node = self.grids.get(grid)
        if node is None:
            raise ConnectorError(
                f"zone {self.name!r} has no GridCoordinates_t node", hint=CORRUPT_HINT
            )
        arrays = {a.name: a for a in node.all("DataArray_t")}
        if "CoordinateX" not in arrays:
            raise UnsupportedSourceError(
                f"zone {self.name!r} has no CoordinateX array (found {sorted(arrays)})",
                hint="only Cartesian coordinates (CoordinateX/Y/Z) are read; convert the grid "
                "to Cartesian in the writing tool, or try options={'engine': 'vtk'}",
            )
        rind_node = node.first("Rind_t")
        rind = rind_node.array().ravel() if rind_node is not None else None
        out: np.ndarray | None = None
        for col, axis in enumerate("XYZ"):
            a = arrays.get(f"Coordinate{axis}")
            if a is None:
                continue
            values = _core(a.array(), rind, self.structured).ravel()
            if values.size != self.n_points:
                raise ConnectorError(
                    f"zone {self.name!r}: {a.name} has {values.size} values, the zone has "
                    f"{self.n_points} vertices",
                    hint=CORRUPT_HINT,
                )
            if out is None:
                dtype = np.float32 if values.dtype == np.float32 else np.float64
                out = np.zeros((self.n_points, 3), dtype=dtype)
            out[:, col] = values
        assert out is not None
        self._points[grid] = out
        return out

    # -- unstructured elements ---------------------------------------------------------
    def block(self, sec: _Section) -> _Block:
        if sec.name not in self._blocks:
            self._blocks[sec.name] = self._read_block(sec)
        return self._blocks[sec.name]

    def _read_block(self, sec: _Section) -> _Block:
        where = f"zone {self.name!r}, section {sec.name!r}"
        conn_node = sec.node.child("ElementConnectivity")
        if conn_node is None:
            raise ConnectorError(f"{where}: no ElementConnectivity array", hint=CORRUPT_HINT)
        conn = conn_node.array().ravel().astype(np.int64, copy=False)
        off_node = sec.node.child("ElementStartOffset")
        offsets = None
        if off_node is not None:
            offsets = off_node.array().ravel().astype(np.int64, copy=False)
            offsets = offsets - offsets[0] if offsets.size else offsets
            if (
                offsets.size != sec.count + 1
                or (np.diff(offsets) < 0).any()
                or offsets[-1] > conn.size
            ):
                raise ConnectorError(
                    f"{where}: ElementStartOffset does not describe {sec.count} elements in "
                    f"{conn.size} connectivity entries",
                    hint=CORRUPT_HINT,
                )
        if sec.etype == MIXED:
            block = _mixed_block(sec, conn, offsets, where)
        elif sec.etype == NGON_N:
            nodes, offs = _poly_records(sec, conn, offsets, where)
            cells = meshutil.pad_cells(nodes - 1, offs)
            block = _Block(cells, np.full(sec.count, 7, np.uint8), np.full(sec.count, 2, np.int8))
        elif sec.etype == NFACE_N:
            faces, offs = _poly_records(sec, conn, offsets, where)
            cells = self._polyhedra(np.abs(faces), offs, where)
            block = _Block(cells, np.full(sec.count, 42, np.uint8), np.full(sec.count, 3, np.int8))
            return block  # node ids come from already checked face blocks
        else:
            block = _fixed_block(sec, conn, where)
        lowest = -1 if sec.etype in (MIXED, NGON_N) else 0  # -1 pads the shorter rows
        if block.cells.size and (
            int(block.cells.min()) < lowest or int(block.cells.max()) >= self.n_points
        ):
            raise ConnectorError(f"{where}: node ids outside 1..{self.n_points}", hint=CORRUPT_HINT)
        return block

    def face_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All 2-D elements of the zone: ``(element ids (sorted), padded cells, types)``."""
        if self._faces is None:
            ids, blocks = [], []
            for sec in self.sections:
                if sec.etype == NFACE_N or not sec.may_have_dim(2):
                    continue
                b = self.block(sec)
                rows = np.nonzero(b.dims == 2)[0]
                if len(rows):
                    ids.append(sec.start + rows)
                    blocks.append((b.cells[rows], b.types[rows]))
            cells, types = _stack_blocks(blocks)
            all_ids = np.concatenate(ids) if ids else np.zeros((0,), np.int64)
            self._faces = (all_ids.astype(np.int64), cells, types)
        return self._faces

    def _polyhedra(self, face_ids: np.ndarray, offsets: np.ndarray, where: str) -> np.ndarray:
        """NFACE_n cells -> padded lists of the unique point ids of their faces."""
        ids, table, _ = self.face_table()
        n = len(offsets) - 1
        pos = np.clip(np.searchsorted(ids, face_ids), 0, max(len(ids) - 1, 0))
        if len(face_ids) and (len(ids) == 0 or (ids[pos] != face_ids).any()):
            raise ConnectorError(
                f"{where}: NFACE_n cells reference face elements that are not in the zone",
                hint=CORRUPT_HINT,
            )
        nodes_out, counts_out = [], []
        chunk = 500_000  # cells per pass, to bound the (faces x nodes) work array
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            lo, hi = int(offsets[c0]), int(offsets[c1])
            nodes = table[pos[lo:hi]]
            owner = np.repeat(np.arange(c1 - c0), np.diff(offsets[c0 : c1 + 1]))
            key = owner[:, None] * np.int64(self.n_points) + nodes
            key = np.unique(key[nodes >= 0])
            counts_out.append(np.bincount(key // self.n_points, minlength=c1 - c0))
            nodes_out.append(key % self.n_points)
        if not nodes_out:
            return np.full((0, 1), -1, dtype=np.int64)
        counts = np.concatenate(counts_out)
        return meshutil.pad_cells(
            np.concatenate(nodes_out), np.concatenate([[0], np.cumsum(counts)])
        )

    def volume_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Cells of the zone's own dimension, in element-number order."""
        dim = self.dim
        blocks = []
        for sec in self.sections:
            if not sec.may_have_dim(dim) or not self.section_census(sec).get(dim):
                continue
            b = self.block(sec)
            keep = b.dims == dim
            blocks.append((b.cells, b.types) if keep.all() else (b.cells[keep], b.types[keep]))
        return _stack_blocks(blocks)

    def section_faces(self, sec: _Section) -> tuple[np.ndarray, np.ndarray]:
        b = self.block(sec)
        keep = b.dims == 2
        return b.cells[keep], b.types[keep]

    def bc_faces(self, bc: _BC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Faces of a boundary patch: ``(padded cells, types, cell map)``.

        Unstructured: the 2-D elements the BC lists (or, for a vertex list, the 2-D elements
        whose nodes all belong to it); the third item holds their element numbers.
        Structured: the quads of the index range; the third item holds the adjacent cells.
        """
        if bc.name not in self._bc_faces:
            if self.structured:
                found = _structured_patch(self.shape, bc)
            else:
                found = self._unstructured_patch(bc)
            empty = (np.full((0, 1), -1, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.int64))
            self._bc_faces[bc.name] = found if found is not None else empty
        return self._bc_faces[bc.name]

    def _unstructured_patch(self, bc: _BC) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if bc.point_range is not None:
            lo, hi = sorted(int(v) for v in bc.point_range.ravel()[:2])
            wanted = np.arange(lo, hi + 1, dtype=np.int64)
        elif bc.list_node is not None:
            wanted = bc.list_node.array().ravel().astype(np.int64)
        else:
            return None
        ids, cells, types = self.face_table()
        if len(ids) == 0 or len(wanted) == 0:
            return None
        if bc.location == "Vertex":
            mark = np.zeros(self.n_points + 1, dtype=bool)
            mark[wanted[(wanted >= 1) & (wanted <= self.n_points)]] = True
            rows = np.nonzero((mark[cells + 1] | (cells < 0)).all(axis=1))[0]
        else:
            pos = np.clip(np.searchsorted(ids, wanted), 0, len(ids) - 1)
            rows = pos[ids[pos] == wanted]
        if len(rows) == 0:
            return None
        return cells[rows], types[rows], ids[rows]

    def covered(self, sec: _Section) -> bool:
        """``True`` when one boundary patch consists of exactly the 2-D elements of ``sec``."""
        n_faces = sum(self.section_census(sec).get(2, {}).values())
        for bc in self.bcs:
            ids = self.bc_faces(bc)[2]
            if n_faces and len(ids) == n_faces:
                inside = (ids >= sec.start) & (ids <= sec.end)
                if inside.all() and len(np.unique(ids)) == n_faces:
                    return True
        return False


def _parse_section(node: _Node, zone: str) -> _Section:
    head = node.array().ravel()
    rng = node.child("ElementRange")
    if head.size < 1 or rng is None:
        raise ConnectorError(
            f"zone {zone!r}, section {node.name!r}: missing element type or ElementRange",
            hint=CORRUPT_HINT,
        )
    span = rng.array().ravel()
    if span.size < 2 or int(span[1]) < int(span[0]) - 1:
        raise ConnectorError(
            f"zone {zone!r}, section {node.name!r}: bad ElementRange {span.tolist()}",
            hint=CORRUPT_HINT,
        )
    return _Section(node.name, int(head[0]), int(span[0]), int(span[1]), node)


def _parse_solution(node: _Node) -> _Solution:
    loc = node.first("GridLocation_t")
    rind = node.first("Rind_t")
    return _Solution(
        name=node.name,
        location=loc.text() if loc is not None else "Vertex",
        rind=rind.array().ravel().astype(np.int64) if rind is not None else None,
        arrays=node.all("DataArray_t"),
        node=node,
    )


def _parse_bc(node: _Node, families: Mapping[str, str]) -> _BC:
    cgns_type = node.text()
    fam = node.first("FamilyName_t")
    family = fam.text() if fam is not None else None
    if family and cgns_type in ("FamilySpecified", "") and families.get(family):
        cgns_type = families[family]
    loc = node.first("GridLocation_t")
    rng = node.child("PointRange") or node.child("ElementRange")
    lst = node.child("PointList") or node.child("ElementList")
    location = loc.text() if loc is not None else "Vertex"
    if loc is None and (node.child("ElementRange") or node.child("ElementList")):
        location = "FaceCenter"  # CGNS 2.x patches list boundary elements
    return _BC(
        name=node.name,
        cgns_type=cgns_type,
        family=family,
        location=location,
        point_range=rng.array().astype(np.int64) if rng is not None else None,
        list_node=lst,
    )


def _core(arr: np.ndarray, rind: np.ndarray | None, structured: bool) -> np.ndarray:
    """Strip rind (ghost) planes. ``rind`` is ``[ilo, ihi, jlo, jhi, ...]`` in index order
    while the array axes are stored reversed."""
    if rind is None or not np.any(rind):
        return arr
    if not structured or arr.ndim == 1:
        flat = arr.ravel()
        return flat[int(rind[0]) : flat.size - int(rind[1])]
    cut = []
    for axis in range(arr.ndim):
        d = arr.ndim - 1 - axis
        lo, hi = (int(rind[2 * d]), int(rind[2 * d + 1])) if 2 * d + 1 < len(rind) else (0, 0)
        cut.append(slice(lo, arr.shape[axis] - hi))
    return arr[tuple(cut)]


def _core_size(shape: Sequence[int], rind: np.ndarray | None, structured: bool) -> int:
    if not shape:
        return 0
    if rind is None or not np.any(rind):
        return int(np.prod(shape))
    if not structured or len(shape) == 1:
        return int(np.prod(shape)) - int(rind[0]) - int(rind[1])
    total = 1
    for axis, n in enumerate(shape):
        d = len(shape) - 1 - axis
        lo, hi = (int(rind[2 * d]), int(rind[2 * d + 1])) if 2 * d + 1 < len(rind) else (0, 0)
        total *= max(n - lo - hi, 0)
    return total


def _stack_blocks(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    blocks = [(c, t) for c, t in blocks if len(c)]
    if not blocks:
        return np.full((0, 1), -1, dtype=np.int64), np.zeros((0,), dtype=np.uint8)
    if len(blocks) == 1:
        return blocks[0][0], blocks[0][1]
    width = max(c.shape[1] for c, _ in blocks)
    cells = np.full((sum(len(c) for c, _ in blocks), width), -1, dtype=np.int64)
    row = 0
    for c, _ in blocks:
        cells[row : row + len(c), : c.shape[1]] = c
        row += len(c)
    return cells, np.concatenate([t for _, t in blocks])


def _fixed_block(sec: _Section, conn: np.ndarray, where: str) -> _Block:
    spec = _VTK_ELEMENTS.get(sec.etype)
    if spec is None:
        raise _unsupported_element(sec.etype, where)
    npe = _NPE[sec.etype]
    if conn.size != sec.count * npe:
        raise ConnectorError(
            f"{where}: {conn.size} connectivity entries for {sec.count} "
            f"{element_name(sec.etype)} elements (expected {sec.count * npe})",
            hint=CORRUPT_HINT,
        )
    cells = conn.reshape(sec.count, npe) - 1
    vtk_type, perm = spec
    if perm is not None:
        cells = cells[:, perm]
    return _Block(
        cells,
        np.full(sec.count, vtk_type, np.uint8),
        np.full(sec.count, _DIM[sec.etype], np.int8),
    )


def _inline_offsets(conn: np.ndarray, n: int, *, mixed: bool, where: str) -> np.ndarray:
    """Record offsets of an inline (CGNS < 4) MIXED / NGON_n / NFACE_n array, in which every
    record starts with its element type (MIXED) or its item count."""
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    head = int(conn[0]) if conn.size else -1
    size = (_NPE.get(head, -1) if mixed else head) + 1
    if size > 0 and conn.size == n * size and (conn[::size] == head).all():
        return np.arange(n + 1, dtype=np.int64) * size  # all records alike: no scan needed
    values = conn.tolist()
    offsets = np.empty(n + 1, dtype=np.int64)
    pos = 0
    for i in range(n):
        offsets[i] = pos
        if pos >= len(values):
            raise ConnectorError(
                f"{where}: connectivity ends after {i} of {n} elements", hint=CORRUPT_HINT
            )
        step = _NPE.get(values[pos]) if mixed else values[pos]
        if step is None:
            raise _unsupported_element(int(values[pos]), where)
        if step < 0:
            raise ConnectorError(f"{where}: negative item count", hint=CORRUPT_HINT)
        pos += step + 1
    if pos > len(values):
        raise ConnectorError(
            f"{where}: connectivity holds {len(values)} entries, {pos} are needed",
            hint=CORRUPT_HINT,
        )
    offsets[n] = pos
    return offsets


def _poly_records(
    sec: _Section, conn: np.ndarray, offsets: np.ndarray | None, where: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(items, offsets)`` of an NGON_n/NFACE_n section in either storage form."""
    if offsets is not None:
        return conn[: int(offsets[-1])], offsets
    inline = _inline_offsets(conn, sec.count, mixed=False, where=where)
    keep = np.ones(int(inline[-1]), dtype=bool)
    keep[inline[:-1]] = False
    return conn[: int(inline[-1])][keep], inline - np.arange(sec.count + 1)


def _mixed_block(sec: _Section, conn: np.ndarray, offsets: np.ndarray | None, where: str) -> _Block:
    n = sec.count
    if offsets is None:
        offsets = _inline_offsets(conn, n, mixed=True, where=where)
    starts = offsets[:-1]
    if n and int(starts.max()) >= conn.size:
        raise ConnectorError(f"{where}: connectivity is too short", hint=CORRUPT_HINT)
    codes = conn[starts] if n else np.zeros(0, np.int64)
    sizes = np.diff(offsets) - 1
    plan = []
    width = 1
    for code in np.unique(codes):
        code = int(code)
        spec = _VTK_ELEMENTS.get(code)
        if spec is None:
            raise _unsupported_element(code, where)
        rows = np.nonzero(codes == code)[0]
        npe = _NPE[code]
        if (sizes[rows] != npe).any():
            raise ConnectorError(
                f"{where}: {element_name(code)} records do not have {npe} nodes",
                hint=CORRUPT_HINT,
            )
        plan.append((rows, npe, spec, code))
        width = max(width, len(spec[1]) if spec[1] is not None else npe)
    cells = np.full((n, width), -1, dtype=np.int64)
    types = np.zeros(n, dtype=np.uint8)
    dims = np.zeros(n, dtype=np.int8)
    for rows, npe, (vtk_type, perm), code in plan:
        block = conn[starts[rows][:, None] + 1 + np.arange(npe)[None, :]] - 1
        if perm is not None:
            block = block[:, perm]
        cells[rows, : block.shape[1]] = block
        types[rows] = vtk_type
        dims[rows] = _DIM[code]
    return _Block(cells, types, dims)


def _structured_cells(shape: Sequence[int]) -> tuple[np.ndarray, int]:
    """Connectivity of a structured block with ``shape`` vertices per direction (i fastest)."""
    ni = shape[0]
    nj = shape[1] if len(shape) > 1 else 1
    nk = shape[2] if len(shape) > 2 else 1
    k, j, i = np.meshgrid(
        np.arange(max(nk - 1, 1)), np.arange(max(nj - 1, 1)), np.arange(ni - 1), indexing="ij"
    )
    p = (i + ni * (j + nj * k)).ravel().astype(np.int64)
    if len(shape) == 1:
        return np.stack([p, p + 1], axis=1), meshutil.VTK_LINE
    quad = [p, p + 1, p + 1 + ni, p + ni]
    if len(shape) == 2:
        return np.stack(quad, axis=1), meshutil.VTK_QUAD
    return np.stack(quad + [q + ni * nj for q in quad], axis=1), meshutil.VTK_HEXAHEDRON


def _structured_patch(
    shape: Sequence[int], bc: _BC
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Quads of a BC index range on a 3-D structured zone, plus the adjacent cell of each."""
    if bc.point_range is None or len(shape) != 3 or bc.point_range.size != 6:
        return None
    rng = np.sort(bc.point_range.reshape(2, 3), axis=0) - 1
    lo, hi = rng[0].copy(), rng[1].copy()
    flat = [d for d in range(3) if lo[d] == hi[d]]
    if bc.location.endswith("FaceCenter"):
        normal = {"IFaceCenter": 0, "JFaceCenter": 1, "KFaceCenter": 2}.get(bc.location)
        if normal is None:
            if len(flat) != 1:
                return None
            normal = flat[0]
        for d in range(3):  # face-centred ranges count faces: one vertex more per direction
            if d != normal:
                hi[d] += 1
        hi[normal] = lo[normal]
    elif bc.location != "Vertex":
        return None
    flat = [d for d in range(3) if lo[d] == hi[d]]
    if len(flat) != 1 or (lo < 0).any() or (hi >= np.asarray(shape)).any():
        return None
    n = flat[0]
    a, b = (d for d in range(3) if d != n)
    bb, aa = np.meshgrid(np.arange(lo[b], hi[b]), np.arange(lo[a], hi[a]), indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    if aa.size == 0:
        return None
    ni, nj = shape[0], shape[1]

    def vertex(da: int, db: int) -> np.ndarray:
        idx: list[Any] = [0, 0, 0]
        idx[n], idx[a], idx[b] = lo[n], aa + da, bb + db
        return idx[0] + ni * (idx[1] + nj * idx[2])

    quads = np.stack([vertex(0, 0), vertex(1, 0), vertex(1, 1), vertex(0, 1)], axis=1)
    cidx: list[Any] = [0, 0, 0]
    cidx[n], cidx[a], cidx[b] = min(int(lo[n]), shape[n] - 2), aa, bb
    cell = cidx[0] + (ni - 1) * (cidx[1] + (nj - 1) * cidx[2])
    return quads.astype(np.int64), np.full(len(quads), 9, np.uint8), cell.astype(np.int64)


def _dimensional_units(node: _Node) -> dict[str, str]:
    """``DimensionalUnits_t`` of ``node`` -> ``{"mass": "kg", "length": "m", ...}``."""
    units = node.first("DimensionalUnits_t")
    if units is None:
        return {}
    out = {}
    for key, name in zip(_UNIT_KEYS, units.texts(), strict=False):
        if name and name not in ("Null", "UserDefined"):
            out[key] = _UNIT_ALIASES.get(name.lower(), normalize_unit(name))
    return out


def _data_class(node: _Node) -> str:
    dc = node.first("DataClass_t")
    return dc.text() if dc is not None else ""


@dataclass
class _FieldSpec:
    name: str
    location: str  # point | cell
    arrays: list[_Node]
    rind: np.ndarray | None
    units: str = ""
    component_names: list[str] | None = None


@dataclass
class _Entry:
    """One readable zone: a ``Zone_t``, a boundary patch or a boundary element section."""

    name: str
    zone: _Zone
    kind: str
    role: str
    default: bool
    bc: _BC | None = None
    section: _Section | None = None


@dataclass
class _State:
    """What later time steps of an entry need from its first step."""

    point_rows: np.ndarray | None = None
    cell_rows: np.ndarray | None = None
    cell_data: bool = False


class _Tree:
    """An open CGNS/HDF5 file parsed down to zones, sections, solutions and patches."""

    def __init__(self, path: Path) -> None:
        import h5py  # noqa: F401 - availability check; the file is opened by h5_open

        self.path = Path(path)
        self.warnings: list[str] = []
        try:
            self.file: Any = h5_open(self.path, what="CGNS link")
        except OSError as e:
            raise ConnectorError(
                f"{self.path.name} cannot be opened as HDF5: {str(e).splitlines()[0]}",
                hint=CORRUPT_HINT,
            ) from e
        try:
            self._parse()
        except ConnectorError:
            self.close()
            raise
        except (KeyError, OSError, RuntimeError, ValueError, IndexError, TypeError) as e:
            self.close()
            raise ConnectorError(
                f"{self.path.name}: malformed CGNS tree ({type(e).__name__}: {e})",
                hint=CORRUPT_HINT,
            ) from e

    def close(self) -> None:
        try:
            self.file.close()
        except (OSError, RuntimeError, ValueError):
            pass

    def _parse(self) -> None:
        root = _Node("/", "", self.file, self.warnings)
        version = root.first("CGNSLibraryVersion_t")
        self.version: float | None = None
        if version is not None and version.has_data():
            self.version = round(float(version.array().ravel()[0]), 3)
        base_nodes = root.all("CGNSBase_t")
        if not base_nodes:
            raise UnsupportedSourceError(
                f"{self.path.name} is an HDF5 file without a CGNSBase_t node: not CGNS",
                hint="generic HDF5 files are read by the 'hdf5' connector with a mapping",
            )
        self.bases: list[_Base] = []
        self.units: dict[str, str] = {}
        self.data_class = ""
        self.parameters: dict[str, float] = {}
        self.physics: dict[str, Any] = {}
        self.times: list[float] = []
        self.time_quantity = "time"
        self.iterations: list[int] = []
        self.descriptors: dict[str, str] = {}
        for node in base_nodes:
            dims = node.array().ravel()
            if dims.size < 2:
                raise ConnectorError(
                    f"base {node.name!r} has no cell/physical dimension", hint=CORRUPT_HINT
                )
            base = _Base(node.name, int(dims[0]), int(dims[1]), node)
            for fam in node.all("Family_t"):
                fbc = fam.first("FamilyBC_t")
                base.families[fam.name] = fbc.text() if fbc is not None else ""
            self._parse_base_metadata(node)
            for znode in node.all("Zone_t"):
                name = f"{base.name}/{znode.name}" if len(base_nodes) > 1 else znode.name
                zone = _Zone(base, znode, name)
                base.zones.append(zone)
                if not self.units:
                    grid = zone.grids.get("GridCoordinates")
                    for holder in (znode, grid):
                        if holder is not None and not self.units:
                            self.units = _dimensional_units(holder)
                            self.data_class = self.data_class or _data_class(holder)
            self.bases.append(base)
        self.zones = [z for b in self.bases for z in b.zones]
        if not self.zones:
            raise UnsupportedSourceError(
                f"{self.path.name} holds no Zone_t node (nothing to read)",
                hint="the file only carries a CGNS skeleton; export grid and solution",
            )

    def _parse_base_metadata(self, node: _Node) -> None:
        if not self.units:
            self.units = _dimensional_units(node)
        self.data_class = self.data_class or _data_class(node)
        sim = node.first("SimulationType_t")
        if sim is not None and sim.text():
            self.physics.setdefault("simulation_type", sim.text())
        for d in node.all("Descriptor_t"):
            self.descriptors.setdefault(d.name, d.text()[:2000])
        ref = node.first("ReferenceState_t")
        if ref is not None:
            for arr in ref.all("DataArray_t"):
                shape = arr.shape()
                if not shape or int(np.prod(shape)) != 1:
                    continue
                value = arr.array().ravel()[0]
                if np.asarray(value).dtype.kind in "fiu":
                    self.parameters.setdefault(arr.name, float(value))
            for src, alias in (("Mach", "Ma"), ("Reynolds", "Re")):
                if src in self.parameters:
                    self.parameters.setdefault(alias, self.parameters[src])
        eqs = node.first("FlowEquationSet_t")
        if eqs is not None:
            for child in eqs.children:
                key = _EQUATION_KEYS.get(child.label)
                if key and child.has_data() and child.text():
                    self.physics.setdefault(key, child.text())
                elif child.name == "EquationDimension" and child.has_data():
                    self.physics.setdefault("equation_dimension", int(child.array().ravel()[0]))
        it = node.first("BaseIterativeData_t")
        if it is not None and not self.times:
            self._parse_iterations(it)

    def _parse_iterations(self, it: _Node) -> None:
        n_steps = int(it.array().ravel()[0]) if it.has_data() else 0
        tv, iv = it.child("TimeValues"), it.child("IterationValues")
        times = [float(v) for v in tv.array().ravel()] if tv is not None else []
        iters = [int(v) for v in iv.array().ravel()] if iv is not None else []
        self.iterations = iters
        increasing = all(b > a for a, b in zip(times, times[1:], strict=False))
        if times and increasing:
            self.times = times
        elif iters and all(b > a for a, b in zip(iters, iters[1:], strict=False)):
            self.times = [float(v) for v in iters]
            self.time_quantity = "iteration"
            if times:
                self.warnings.append(
                    "TimeValues are not increasing: the iteration numbers are the time axis"
                )
        elif n_steps > 0 or times:
            self.times = [float(i) for i in range(max(n_steps, len(times)))]
            self.time_quantity = "iteration"

    @property
    def n_steps(self) -> int:
        return len(self.times) if len(self.times) > 1 else 0


_EQUATION_KEYS = {
    "GoverningEquations_t": "governing_equations",
    "TurbulenceModel_t": "turbulence_model",
    "TurbulenceClosure_t": "turbulence_closure",
    "GasModel_t": "gas_model",
    "ViscosityModel_t": "viscosity_model",
    "ThermalConductivityModel_t": "thermal_conductivity_model",
    "ThermalRelaxationModel_t": "thermal_relaxation_model",
    "ChemicalKineticsModel_t": "chemical_kinetics_model",
}


class _H5Reader:
    """Turns a parsed tree into a description and into samples."""

    def __init__(self, tree: _Tree, source: Path, options: Mapping[str, Any]) -> None:
        self.tree = tree
        self.source = source
        self.raw_names = bool(options.get("raw_names", False))
        self.skipped: list[str] = []
        self._entries: list[_Entry] | None = None

    # -- zones -------------------------------------------------------------------------
    def entries(self) -> list[_Entry]:
        if self._entries is None:
            entries = [e for zone in self.tree.zones for e in self._zone_entries(zone)]
            # zones of a surface base next to a volume base repeat its boundaries: on request
            top = max(z.dim for z in self.tree.zones)
            for e in entries:
                e.default = e.default and e.zone.dim == top
            self._entries = entries
        return self._entries

    def _zone_entries(self, zone: _Zone) -> list[_Entry]:
        dim = zone.dim
        kind = "surface" if dim == 2 else ("point_cloud" if dim == 0 else "mesh")
        planar = zone.base.phys_dim == 2
        role = "volume" if dim == 3 or (dim == 2 and planar) else "boundary"
        if dim == 0:
            role = "probes"
        out = [_Entry(zone.name, zone, kind, role, True)]
        if dim != 3:
            return out  # the boundaries of a 2-D zone are curves, not surfaces
        taken = {zone.name}
        for bc in zone.bcs:
            name = f"{zone.name}/{bc.name}"
            if name in taken or len(zone.bc_faces(bc)[0]) == 0:
                continue
            taken.add(name)
            out.append(_Entry(name, zone, "surface", "boundary", False, bc=bc))
        for sec in zone.sections:
            name = f"{zone.name}/{sec.name}"
            # the NGON_n faces of a polyhedral zone include every interior face
            if sec.etype == NFACE_N or (sec.etype == NGON_N and zone.has_nface):
                continue
            if name in taken or not zone.section_census(sec).get(2) or zone.covered(sec):
                continue
            taken.add(name)
            out.append(_Entry(name, zone, "surface", "boundary", False, section=sec))
        return out

    def _topology(self, entry: _Entry) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(padded cells, cell types, row of the zone's cell data per cell or None)``."""
        zone = entry.zone
        if entry.bc is not None:
            cells, types, extra = zone.bc_faces(entry.bc)
            return cells, types, extra if zone.structured else None
        if entry.section is not None:
            cells, types = zone.section_faces(entry.section)
            return cells, types, None
        if zone.structured:
            cells, vtk_type = _structured_cells(zone.shape)
            types = np.full(len(cells), vtk_type, dtype=np.uint8)
        else:
            cells, types = zone.volume_cells()
        return cells, types, np.arange(len(cells), dtype=np.int64)

    # -- fields ------------------------------------------------------------------------
    def step_solutions(
        self, zone: _Zone, step: int | None
    ) -> tuple[list[_Solution], list[_Solution]]:
        """``(solutions of this step, solutions that belong to no step)``."""
        n_steps = self.tree.n_steps
        if step is None or n_steps == 0:
            return zone.solutions, []
        by_name = {s.name: s for s in zone.solutions}
        pointers = zone.flow_pointers
        if len(pointers) >= n_steps:
            used = {p for p in pointers[:n_steps] if p in by_name}
            static = [s for s in zone.solutions if s.name not in used]
            current = by_name.get(pointers[step])
            return ([current] if current is not None else []), static
        # no FlowSolutionPointers: k solutions per location and k = number of steps is the
        # only layout that can be attributed to steps without guessing
        groups: dict[str, list[_Solution]] = {}
        for s in zone.solutions:
            groups.setdefault(s.location, []).append(s)
        if groups and all(len(g) == n_steps for g in groups.values()):
            return [g[step] for g in groups.values()], []
        return [], zone.solutions

    def layout(self, zone: _Zone, solutions: Sequence[_Solution]) -> list[_FieldSpec]:
        n_cells = zone.count(zone.dim) if zone.dim else 0
        out: list[_FieldSpec] = []
        taken: dict[str, str] = {}
        ordered = sorted(solutions, key=lambda s: s.location != "Vertex")
        for sol in ordered:
            location = {"Vertex": "point", "CellCenter": "cell"}.get(sol.location)
            if location is None:
                self._skip(f"solution {zone.name}/{sol.name}: GridLocation {sol.location}")
                continue
            expected = zone.n_points if location == "point" else n_cells
            arrays: dict[str, _Node] = {}
            for arr in sol.arrays:
                if _core_size(arr.shape(), sol.rind, zone.structured) != expected:
                    self._skip(
                        f"array {zone.name}/{sol.name}/{arr.name}: shape {arr.shape()} does "
                        f"not match {expected} {location}s"
                    )
                    continue
                arrays[arr.name] = arr
            for name, members in self._group(list(arrays), zone.base.phys_dim):
                out_name = name
                if out_name in taken:
                    suffix = "cell" if taken[out_name] != location else sol.name
                    out_name = f"{name}__{suffix}"
                taken[out_name] = location
                nodes = [arrays[m] for m in members]
                out.append(
                    _FieldSpec(
                        name=out_name,
                        location=location,
                        arrays=nodes,
                        rind=sol.rind,
                        units=self._units(name, nodes[0], sol),
                        component_names=list("xyz"[: len(nodes)]) if len(nodes) > 1 else None,
                    )
                )
        return out

    def _skip(self, message: str) -> None:
        if message not in self.skipped:
            self.skipped.append(message)

    def _group(self, names: list[str], phys_dim: int) -> list[tuple[str, list[str]]]:
        """Assemble ``<stem>X/Y/Z`` scalars into one vector named ``<stem>``."""
        if self.raw_names:
            return [(n, [n]) for n in names]
        present = set(names)
        groups: dict[str, tuple[str, list[str]]] = {}
        for n in names:
            if len(n) < 2 or n[-1] != "X":
                continue
            stem = n[:-1]
            members = [stem + c for c in "XYZ"]
            if members[2] not in present and phys_dim == 2:
                members = members[:2]
            target = stem.rstrip("_")
            if not all(m in present for m in members) or not target or target in present:
                continue
            if any(m in groups for m in members):
                continue
            for m in members:
                groups[m] = (target, members)
        out: list[tuple[str, list[str]]] = []
        done: set[str] = set()
        for n in names:
            if n in done:
                continue
            if n in groups:
                target, members = groups[n]
                out.append((target, members))
                done.update(members)
            else:
                out.append((n, [n]))
        return out

    def _units(self, name: str, array: _Node, sol: _Solution) -> str:
        data_class = _data_class(array) or _data_class(sol.node) or self.tree.data_class
        if data_class != "Dimensional":
            return ""
        units = {**self.tree.units, **_dimensional_units(sol.node), **_dimensional_units(array)}
        exp = array.first("DimensionalExponents_t")
        if exp is not None and exp.has_data():
            return _unit_from_exponents([float(v) for v in exp.array().ravel()], units)
        std = _STANDARD_EXPONENTS.get(name)
        return _unit_from_exponents(std, units) if std is not None else ""

    def _values(self, zone: _Zone, spec: _FieldSpec) -> np.ndarray:
        cols = [_core(a.array(), spec.rind, zone.structured).ravel() for a in spec.arrays]
        return cols[0] if len(cols) == 1 else np.stack(cols, axis=1)

    # -- describe ----------------------------------------------------------------------
    def describe(self, connector_id: str) -> SourceDescription:
        tree = self.tree
        desc = SourceDescription(
            connector_id=connector_id,
            path=self.source,
            format="cgns-hdf5",
            domain="cfd",
            physics=dict(tree.physics),
            units=dict(tree.units),
            parameters=dict(tree.parameters),
        )
        if tree.n_steps:
            units = tree.units.get("time", "s") if tree.time_quantity == "time" else ""
            desc.time = TimeInfo(values=list(tree.times), units=units, quantity=tree.time_quantity)
        elif len(tree.times) == 1:
            desc.metadata["time_value"] = tree.times[0]
        for entry in self.entries():
            desc.zones.append(self._zone_info(entry))
        merged: dict[tuple[str, str, bool], FieldInfo] = {}
        for zone in tree.zones:
            current, static = self.step_solutions(zone, 0 if tree.n_steps else None)
            for group, dependent in ((current, bool(tree.n_steps)), (static, False)):
                for spec in self.layout(zone, group):
                    key = (spec.name, spec.location, dependent)
                    if key in merged:
                        zones = merged[key].zones
                        if zones is not None and zone.name not in zones:
                            zones.append(zone.name)
                        continue
                    merged[key] = FieldInfo(
                        name=spec.name,
                        location=spec.location,
                        components=len(spec.arrays),
                        units=spec.units,
                        component_names=spec.component_names,
                        dtype=str(spec.arrays[0].group[" data"].dtype),
                        time_dependent=dependent,
                        zones=[zone.name],
                    )
            for bc in zone.bcs:
                desc.boundary_conditions[f"{zone.name}/{bc.name}"] = bc.info()
        for info in merged.values():
            if info.zones is not None and len(info.zones) == len(tree.zones):
                info.zones = None
            desc.fields.append(info)
        desc.metadata.update(
            {
                "cgns_version": tree.version,
                "bases": [
                    {"name": b.name, "cell_dim": b.cell_dim, "phys_dim": b.phys_dim}
                    for b in tree.bases
                ],
                "families": {k: v for b in tree.bases for k, v in b.families.items()},
            }
        )
        if tree.data_class:
            desc.metadata["data_class"] = tree.data_class
        if tree.iterations:
            desc.metadata["iterations"] = tree.iterations
        if tree.descriptors:
            desc.metadata["descriptors"] = tree.descriptors
        if any(z.moving for z in tree.zones):
            desc.metadata["moving_grid"] = True
        desc.warnings.extend(tree.warnings)
        desc.warnings.extend(f"not read: {m}" for m in self.skipped)
        for zone in tree.zones:
            zone.release()
        return desc

    def _zone_info(self, entry: _Entry) -> ZoneInfo:
        zone = entry.zone
        if entry.bc is not None or entry.section is not None:
            types = self._topology(entry)[1]
            return ZoneInfo(
                name=entry.name,
                kind="surface",
                role=entry.role,
                n_cells=len(types),
                dim=2,
                cell_types=meshutil.census(types),
                bc_type=_known_bc(entry.bc),
                default=False,
            )
        dim = zone.dim
        return ZoneInfo(
            name=entry.name,
            kind=entry.kind,
            role=entry.role,
            # a surface zone only keeps the points its cells use, which describe() does not read
            n_points=zone.n_points if entry.kind != "surface" else None,
            n_cells=zone.count(dim) if dim else 0,
            dim=dim,
            cell_types=zone.census(dim) if dim else {},
            shape=list(zone.shape) if zone.structured else None,
            default=entry.default,
        )

    # -- read --------------------------------------------------------------------------
    def read(
        self,
        base_id: str,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
    ) -> Iterator[RawSample]:
        entries = self.entries()
        names = [e.name for e in entries]
        selected = set(
            _vtk.select_zones(names, zones, [e.name for e in entries if e.default] or None)
        )
        chosen = [e for e in entries if e.name in selected]
        ids = _sample_ids(base_id, [e.name for e in chosen])
        tree = self.tree
        steps: list[int | None]
        if tree.n_steps:
            steps = list(select_time_indices(tree.times, times))
        else:
            select_time_indices([], times)  # validates the selection
            steps = [None]
        states: dict[str, _State] = {}
        for n, step in enumerate(steps):
            for entry in chosen:
                state = states.setdefault(entry.name, _State())
                yield self._sample(entry, ids[entry.name], step, n == 0, fields, state)
            for zone in tree.zones:
                zone.release()

    def _metadata(self, entry: _Entry) -> dict[str, Any]:
        tree, zone = self.tree, entry.zone
        meta: dict[str, Any] = {
            "source": str(self.source),
            "solver": "unknown",
            "format": "cgns-hdf5",
            "base": zone.base.name,
            "cgns_zone": zone.raw_name,
            "zone_type": "structured" if zone.structured else "unstructured",
            "role": entry.role,
        }
        if tree.version is not None:
            meta["cgns_version"] = tree.version
        if tree.physics:
            meta["physics"] = dict(tree.physics)
        if tree.parameters:
            meta["parameters"] = dict(tree.parameters)
        if tree.units:
            meta["units"] = dict(tree.units)
        if tree.n_steps:
            if tree.time_quantity == "time":
                meta["time_units"] = tree.units.get("time", "s")
            else:
                meta["time_quantity"] = tree.time_quantity
        elif len(tree.times) == 1:
            meta["time_value"] = tree.times[0]
        if zone.family:
            meta["family"] = zone.family
        if entry.bc is not None:
            info = entry.bc.info()
            if info["type"] != "unknown":
                meta["bc_type"] = info["type"]
            meta["boundary_conditions"] = {entry.name: info}
        elif entry.section is not None:
            meta["section"] = entry.section.name
        elif zone.bcs:
            meta["boundary_conditions"] = {f"{zone.name}/{b.name}": b.info() for b in zone.bcs}
        return meta

    def _sample(
        self,
        entry: _Entry,
        sample_id: str,
        step: int | None,
        first: bool,
        fields: Sequence[str] | None,
        state: _State,
    ) -> RawSample:
        zone = entry.zone
        sample = RawSample(
            id=sample_id, kind=entry.kind, zone=entry.name, metadata=self._metadata(entry)
        )
        if step is not None:
            sample.time = float(self.tree.times[step])
            sample.time_index = step
        grid = zone.grid_name(step)
        if first:
            self._geometry(entry, sample, zone.points(grid), state)
        elif zone.moving:
            points = zone.points(grid)
            sample.points = points if state.point_rows is None else points[state.point_rows]
        current, static = self.step_solutions(zone, step)
        groups = [(current, sample.fields)] + ([(static, sample.static)] if first else [])
        for solutions, target in groups:
            for spec in self.layout(zone, solutions):
                if not wants(spec.name, fields):
                    continue
                if spec.location == "cell" and not state.cell_data:
                    continue
                rows = state.point_rows if spec.location == "point" else state.cell_rows
                values = self._values(zone, spec)
                target[spec.name] = as_field(
                    values if rows is None else values[rows],
                    spec.location,
                    units=spec.units,
                    component_names=spec.component_names,
                )
        return sample

    def _geometry(
        self, entry: _Entry, sample: RawSample, points: np.ndarray, state: _State
    ) -> None:
        zone = entry.zone
        if entry.kind == "point_cloud":
            sample.points = points
            return
        cells, types, cell_map = self._topology(entry)
        if entry.kind == "surface":
            faces, parent = meshutil.triangulate(cells, types)
            pts, faces, used = meshutil.compact_points(points, faces)
            sample.points = pts
            sample.faces = faces.astype(np.int32)
            state.point_rows = used if len(used) != len(points) else None
            if cell_map is not None:
                state.cell_data = True
                state.cell_rows = cell_map[parent]
            if len(parent) and not np.array_equal(parent, np.arange(len(cells))):
                sample.static["face_parent"] = as_field(parent.astype(np.int32), "cell")
            return
        sample.points = points
        sample.cells = cells
        sample.cell_types = types
        state.cell_data = True
        if zone.structured:
            sample.metadata["structured_shape"] = list(zone.shape)
        if (types == meshutil.VTK_POLYHEDRON).any():
            sample.metadata["polyhedral"] = True


def _known_bc(bc: _BC | None) -> str | None:
    kind = bc_type_id(bc.cgns_type) if bc is not None else "unknown"
    return None if kind == "unknown" else kind


def _sample_ids(base: str, zones: Sequence[str]) -> dict[str, str]:
    """``zone -> sample id``; names that only differ in characters an id cannot hold get a
    numeric suffix."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for zone in zones:
        sid = candidate = meshutil.zone_sample_id(base, zone, len(zones))
        n = 1
        while candidate in taken:
            n += 1
            candidate = f"{sid}_{n}"
        taken.add(candidate)
        out[zone] = candidate
    return out


# -- file level helpers -------------------------------------------------------------------------


def file_format(path: Path) -> str | None:
    """``"hdf5"`` | ``"adf"`` | ``None`` from the first bytes of ``path``."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
    except OSError:
        return None
    if head.startswith(HDF5_MAGIC):
        return "hdf5"
    if head[4:].startswith(ADF_MAGIC) and bytes(b & 0x7F for b in head[:4]) == b"@(#)":
        return "adf"
    return None


def is_cgns_hdf5(path: Path) -> bool | None:
    """``True`` when the HDF5 file has a ``CGNSBase_t`` / ``CGNSLibraryVersion_t`` child at
    the root, ``False`` when it has not, ``None`` when it cannot be opened."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            for key in [str(k) for k in f]:
                if key.startswith(" "):
                    continue
                item = f.get(key)
                if item is not None and _attr(item, "label") in (
                    "CGNSBase_t",
                    "CGNSLibraryVersion_t",
                ):
                    return True
    except (OSError, RuntimeError, KeyError):
        return None
    return False


_NUMBERED = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


def _series_members(path: Path) -> list[tuple[int, Path]]:
    """``(number, file)`` of the numbered ``.cgns`` files that form a series with ``path``
    (a directory, or one member file), sorted by number."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".cgns")
        groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}
        for p in files:
            m = _NUMBERED.match(p.name)
            if m:
                groups.setdefault((m.group(1), m.group(3).lower()), []).append((int(m.group(2)), p))
        if len(files) == 1:
            return [(0, files[0])]
        if len(groups) == 1 and sum(len(g) for g in groups.values()) == len(files):
            return sorted(next(iter(groups.values())))
        if not files:
            raise UnsupportedSourceError(
                f"{path} holds no .cgns file", hint="pass a .cgns file or a directory of them"
            )
        raise UnsupportedSourceError(
            f"{path} holds {len(files)} .cgns files that are not one numbered series",
            hint="pass one file, or keep only the files of one series "
            "(name_0001.cgns, name_0002.cgns, ...) in the directory",
        )
    m = _NUMBERED.match(path.name)
    if m is None:
        return [(0, path)]
    prefix, ext = m.group(1), m.group(3).lower()
    out = []
    for p in path.parent.iterdir():
        mm = _NUMBERED.match(p.name)
        if mm and mm.group(1) == prefix and mm.group(3).lower() == ext and p.is_file():
            out.append((int(mm.group(2)), p))
    return sorted(out) if len(out) > 1 else [(0, path)]


def _peek_time(path: Path) -> float | None:
    """The single ``TimeValues`` entry of a CGNS/HDF5 file, read without parsing the zones."""
    import h5py

    if file_format(path) != "hdf5":
        return None
    try:
        with h5py.File(path, "r") as f:
            root = _Node("/", "", f, [])
            for base in root.all("CGNSBase_t"):
                it = base.first("BaseIterativeData_t")
                tv = it.child("TimeValues") if it is not None else None
                if tv is not None and tv.has_data():
                    values = tv.array().ravel()
                    return float(values[0]) if values.size == 1 else None
    except (OSError, RuntimeError, KeyError, ValueError, ConnectorError):
        return None
    return None


@dataclass
class _Source:
    """A resolved path: one file, or the files of a one-file-per-step series."""

    files: list[Path]
    times: list[float]
    quantity: str = "time"
    base_id: str = ""

    @property
    def series(self) -> bool:
        return len(self.files) > 1


def _resolve(path: Path, options: Mapping[str, Any]) -> _Source:
    if not path.exists():
        raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
    if path.is_file() and options.get("series", True) is False:
        return _Source([path], [], base_id=meshutil.default_sample_id(path))
    members = _series_members(path)
    files = [p for _, p in members]
    if len(files) == 1:
        base = path if path.is_dir() else files[0]
        return _Source(files, [], base_id=meshutil.default_sample_id(base))
    stamps = [t for t in (_peek_time(p) for p in files) if t is not None]
    if len(stamps) == len(files) and all(b > a for a, b in zip(stamps, stamps[1:], strict=False)):
        times, quantity = stamps, "time"
    else:
        times, quantity = [float(n) for n, _ in members], "iteration"
    if path.is_dir():
        base_id = meshutil.default_sample_id(path)
    else:
        m = _NUMBERED.match(path.name)
        stem = m.group(1).rstrip("._-") if m else ""
        base_id = meshutil.sanitize_id(stem) if stem else meshutil.default_sample_id(path)
    return _Source(files, times, quantity, base_id)


def _engine(path: Path, options: Mapping[str, Any]) -> str:
    """``"h5py"`` or ``"vtk"`` for one file."""
    engine = str(options.get("engine", "auto")).lower()
    if engine not in ("auto", "h5py", "vtk"):
        raise ConnectorError(
            f"unknown CGNS engine {engine!r}", hint="use engine=auto, engine=h5py or engine=vtk"
        )
    if not path.is_file():
        raise UnsupportedSourceError(f"{path} is not a file", hint="pass a .cgns file")
    fmt = file_format(path)
    if fmt is None:
        raise ConnectorError(
            f"{path.name} is not a CGNS file (neither an HDF5 nor an ADF header)",
            hint=CORRUPT_HINT,
        )
    if engine == "vtk":
        return "vtk"
    if fmt == "adf":
        if engine == "h5py":
            raise ConnectorError(
                f"{path.name} is an ADF-based CGNS file, which h5py cannot read", hint=ADF_HINT
            )
        return "vtk"
    return "h5py"


# -- VTK engine ---------------------------------------------------------------------------------


class _VtkErrors:
    """VTK ``ErrorEvent`` observer that keeps the messages."""

    CallDataType = "string0"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, obj: Any, event: str, message: str) -> None:
        lines = str(message).strip().splitlines() if message else []
        self.messages.append(lines[-1] if lines else "error")


class _VtkSession:
    """``vtkCGNSReader`` configured for one file, with its block paths mapped to zone names."""

    def __init__(self, path: Path, options: Mapping[str, Any]) -> None:
        from vtkmodules.vtkCommonCore import vtkCommand
        from vtkmodules.vtkIOCGNSReader import vtkCGNSReader

        self.path = path
        self.errors = _VtkErrors()
        self.patches: set[str] = set()
        reader: Any = vtkCGNSReader()
        reader.AddObserver(vtkCommand.ErrorEvent, self.errors)
        self.reader = reader
        with _quiet_vtk():
            if not reader.CanReadFile(str(path)):
                raise ConnectorError(
                    f"VTK cannot open {path.name} as CGNS"
                    + (f": {self.errors.messages[-1]}" if self.errors.messages else ""),
                    hint=CORRUPT_HINT,
                )
            h5_check_file(path, what="CGNS link")
            reader.SetFileName(str(path))
            reader.SetLoadBndPatch(bool(options.get("boundaries", True)))
            reader.UpdateInformation()
            reader.EnableAllBases()
            reader.EnableAllPointArrays()
            reader.EnableAllCellArrays()
            reader.UpdateInformation()
        self.n_bases = int(reader.GetNumberOfBaseArrays())
        self.check("reading the header of")

    def check(self, doing: str) -> None:
        if self.errors.messages:
            raise ConnectorError(
                f"VTK failed {doing} {self.path.name}: {self.errors.messages[0]}",
                hint=CORRUPT_HINT,
            )

    def rename(self, name: str) -> str:
        """``Base/Zone/Internal`` -> ``Zone``; ``Base/Zone/Patches/wall`` -> ``Zone/wall``."""
        parts = name.split("/")
        patch = len(parts) >= 2 and parts[-2] == "Patches"
        if patch:
            parts = parts[:-2] + parts[-1:]
        elif parts[-1] == "Internal":
            parts = parts[:-1]
        if self.n_bases <= 1 and len(parts) > 1:
            parts = parts[1:]
        out = "/".join(parts)
        if patch:
            self.patches.add(out)
        return out

    def is_default(self, name: str, dataset: Any) -> bool:
        return name not in self.patches

    def role(self, name: str, dataset: Any) -> str | None:
        return "boundary" if name in self.patches else "volume"


def _quiet_vtk() -> Any:
    from frontier_connector_ensight.connector import quiet_vtk

    return quiet_vtk()


def _vtk_metadata(path: Path) -> dict[str, Any]:
    fmt = "cgns-adf" if file_format(path) == "adf" else "cgns-hdf5"
    return {"source": str(path), "solver": "unknown", "format": fmt, "engine": "vtk"}


def _vtk_describe(connector_id: str, path: Path, options: Mapping[str, Any]) -> SourceDescription:
    session = _VtkSession(path, options)
    times = _vtk.reader_times(session.reader)
    with _quiet_vtk():
        obj = _vtk.update_to_time(session.reader, times[0] if times else None)
    session.check("reading")
    meta = _vtk_metadata(path)
    desc = SourceDescription(
        connector_id=connector_id, path=path, format=meta["format"], domain="cfd"
    )
    _vtk.describe_object(
        obj, desc, role=session.role, default=session.is_default, rename=session.rename
    )
    if not desc.zones:
        raise ConnectorError(f"VTK found no zone in {path.name}", hint=CORRUPT_HINT)
    if len(times) > 1:
        desc.time = TimeInfo(values=times)
    elif times and times[0] != 0.0:
        desc.metadata["time_value"] = times[0]
    desc.metadata["engine"] = "vtk"
    desc.warnings.append(
        "read through VTK's vtkCGNSReader: boundary-condition types, units, reference state "
        "and flow-equation metadata are not extracted"
        + (" (convert the ADF file with `cgnsconvert -h`)" if meta["format"] == "cgns-adf" else "")
    )
    return desc


def _vtk_read(
    path: Path,
    base_id: str,
    zones: Sequence[str] | None,
    fields: Sequence[str] | None,
    times: TimeSelection,
    options: Mapping[str, Any],
) -> Iterator[RawSample]:
    session = _VtkSession(path, options)
    with _quiet_vtk():
        stream = _vtk.read_reader_steps(
            session.reader,
            base_id,
            zones=zones,
            fields=fields,
            times=times,
            metadata=_vtk_metadata(path),
            default_zone=session.is_default,
            rename=session.rename,
        )
        count = 0
        for sample in stream:
            session.check("reading")
            if sample.zone in session.patches:
                sample.metadata["role"] = "boundary"
            count += 1
            yield sample
    session.check("reading")
    if count == 0:
        raise ConnectorError(f"VTK found no zone in {path.name}", hint=CORRUPT_HINT)


# -- connector ----------------------------------------------------------------------------------


class CgnsConnector(BaseConnector):
    id = "cgns"
    name = "CGNS (HDF5 natively with h5py; ADF through VTK)"
    formats = ["cgns-hdf5", "cgns-adf"]
    vendor = "CGNS Steering Committee (open standard)"
    license = "Apache-2.0 (reader: h5py, BSD-3; ADF fallback: VTK, BSD-3)"
    extensions = [".cgns"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["surface", "mesh", "point_cloud"]
    export_help = (
        "Export CGNS from the solver or post-processor (Ansys CFX: cfx5export -cgns; Fluent: "
        "File > Export > CGNS; STAR-CCM+: File > Export > CGNS; Fidelity/FINE, SU2, elsA, "
        "Pointwise write it natively) and point at the .cgns file or a directory holding a "
        "numbered series."
    )
    status_note = (
        "Reads CGNS/HDF5 with h5py: structured and unstructured zones, linear and quadratic "
        "elements, MIXED, NGON_n / NFACE_n, boundary patches as surface zones, vertex and "
        "cell-centred solutions, rind, links, FlowSolutionPointers, file-per-step series, units, "
        "reference state and flow equations. Cubic and higher elements, face-centred solutions and "
        "non-Cartesian coordinates are not supported. ADF files and options={'engine': 'vtk'} go "
        "through VTK's vtkCGNSReader, without the CGNS metadata."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            if path.is_dir():
                return self._detect_dir(path)
            if not path.is_file():
                return None
            ext = match_extension(path, [".cgns", ".h5", ".hdf", ".hdf5", ".adf"])
            if ext is None:
                return None
            fmt = file_format(path)
            if fmt == "adf":
                if ext not in (".cgns", ".adf"):
                    return None
                notes = [f"extension {ext}", "ADF header (read through VTK)"]
                return Detection(self.id, path, 0.9 if ext == ".cgns" else 0.5, "cgns-adf", notes)
            if fmt != "hdf5":
                return None
            cgns = is_cgns_hdf5(path)
        except OSError:
            return None
        if cgns is None:
            if ext != ".cgns":
                return None
            notes = ["extension .cgns", "HDF5 header, but the file cannot be opened"]
            return Detection(self.id, path, 0.5, "cgns-hdf5", notes)
        if not cgns:
            return None
        conf = 0.98 if ext == ".cgns" else 0.9
        return Detection(self.id, path, conf, "cgns-hdf5", [f"extension {ext}", "CGNSBase_t"])

    def _detect_dir(self, path: Path) -> Detection | None:
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".cgns")
        if len(files) < 2 or file_format(files[0]) is None:
            return None
        try:
            members = _series_members(path)
        except UnsupportedSourceError:
            return None
        notes = [f"directory with a numbered series of {len(members)} .cgns files"]
        return Detection(self.id, path, 0.55, "cgns-series", notes)

    # -- describe ----------------------------------------------------------------------
    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        opts = dict(options or {})
        source = _resolve(path, opts)
        first = source.files[0]
        if _engine(first, opts) == "vtk":
            desc = _vtk_describe(self.id, first, opts)
        else:
            desc = self._describe_h5(first, opts)
        desc.path = path
        if source.series:
            units = desc.units.get("time", "s") if source.quantity == "time" else ""
            desc.time = TimeInfo(values=source.times, units=units, quantity=source.quantity)
            desc.metadata["series_files"] = [p.name for p in source.files]
            desc.metadata.pop("time_value", None)
            for f in desc.fields:
                f.time_dependent = True
        return desc

    def _describe_h5(self, file: Path, options: Mapping[str, Any]) -> SourceDescription:
        tree = _Tree(file)
        try:
            return _H5Reader(tree, file, options).describe(self.id)
        except ConnectorError:
            raise
        except (KeyError, OSError, RuntimeError, ValueError, IndexError, TypeError) as e:
            raise _malformed(file, e) from e
        finally:
            tree.close()

    # -- read --------------------------------------------------------------------------
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
        source = _resolve(path, opts)
        if not source.series:
            yield from self._read_file(source.files[0], source.base_id, zones, fields, times, opts)
            return
        from frontier_connector_ensight.connector import drop_unchanged_geometry

        # every file of a series is complete: an unchanged mesh is passed on only once
        first_steps: dict[str, RawSample] = {}
        every_step = bool(opts.get("geometry_every_step", False))
        for ti in select_time_indices(source.times, times):
            for s in self._read_file(source.files[ti], source.base_id, zones, fields, "last", opts):
                s.time = float(source.times[ti])
                s.time_index = ti
                s.metadata.pop("time_value", None)
                if source.quantity == "time":
                    s.metadata.setdefault(
                        "time_units", s.metadata.get("units", {}).get("time", "s")
                    )
                else:
                    s.metadata["time_quantity"] = source.quantity
                if not every_step:
                    drop_unchanged_geometry(s, first_steps)
                yield s

    def _read_file(
        self,
        file: Path,
        base_id: str,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
        options: Mapping[str, Any],
    ) -> Iterator[RawSample]:
        if _engine(file, options) == "vtk":
            yield from _vtk_read(file, base_id, zones, fields, times, options)
            return
        tree = _Tree(file)
        try:
            yield from _H5Reader(tree, file, options).read(base_id, zones, fields, times)
        except ConnectorError:
            raise
        except (KeyError, OSError, RuntimeError, ValueError, IndexError, TypeError) as e:
            raise _malformed(file, e) from e
        finally:
            tree.close()


def _malformed(file: Path, error: Exception) -> ConnectorError:
    return ConnectorError(
        f"{file.name}: malformed CGNS data ({type(error).__name__}: {error})", hint=CORRUPT_HINT
    )
