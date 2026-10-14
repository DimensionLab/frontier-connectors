# SPDX-License-Identifier: Apache-2.0
"""Neutral in-memory model shared by every connector (numpy is the only dependency).

``RawSample`` is the form that gets written to ``frontier-dataset v1.1``. Two shapes are
allowed so that readers never have to hold a whole transient in memory:

* **step form** (preferred for meshes): one ``RawSample`` per time step with ``time`` and
  ``time_index`` set and field ``dims`` *without* a leading ``"time"``. All steps of one
  simulation share ``RawSample.id``; the first one carries the geometry, later ones may set
  ``points``/``cells``/``faces`` to ``None`` ("unchanged") or pass new ``points`` (moving mesh).
* **block form** (natural for grids and array stores): ``times`` set and every time-dependent
  field has ``dims[0] == "time"``. ``FieldData.data`` may then be a *lazy* array (an
  ``h5py.Dataset``, ``zarr.Array``, ``numpy.memmap``): the writer slices it step by step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

SampleKind = Literal["surface", "mesh", "grid", "point_cloud"]
Location = Literal["point", "cell", "grid", "global"]
Maturity = Literal[
    "verified-public-sample",
    "verified-synthetic",
    "exporter+reader",
    "implemented-unverified-live",
    "planned",
]

MATURITIES: tuple[str, ...] = (
    "verified-public-sample",
    "verified-synthetic",
    "exporter+reader",
    "implemented-unverified-live",
    "planned",
)

#: Anything ``Connector.read(times=...)`` accepts. ``None``/"all" = every step, "first"/"last",
#: an ``int`` index (negative allowed), a ``slice`` of indices, a sequence of ``int`` indices,
#: or a sequence of ``float`` physical times (the nearest stored step is used for each).
TimeSelection = None | str | int | slice | Sequence[int] | Sequence[float]


@dataclass
class Detection:
    """What a connector found at a path. ``confidence`` is 0..1; 1.0 = unambiguous magic bytes
    or directory layout, ~0.5 = extension only, <0.5 = generic container that might fit."""

    connector_id: str
    path: Path
    confidence: float
    format: str = ""
    notes: list[str] = field(default_factory=list)
    #: things the user must provide before ``read`` works (optional package, exporter run, ...)
    needs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["root"] = str(self.path)  # name used by the Rust `Detection`
        return d


@dataclass
class FieldInfo:
    name: str
    location: str = "point"  # Location
    components: int = 1
    units: str = ""
    component_names: list[str] | None = None
    dtype: str = "float32"
    description: str = ""
    time_dependent: bool = True
    #: zones that carry this field; ``None`` = all zones
    zones: list[str] | None = None


@dataclass
class ZoneInfo:
    """A part of the source that becomes one sample: a cell zone, boundary patch, block, part."""

    name: str
    kind: str = "mesh"  # SampleKind
    role: str = "volume"  # volume | boundary | interface | part | grid | probes | geometry
    n_points: int | None = None
    n_cells: int | None = None
    dim: int = 3
    #: element census, e.g. ``{"hexahedron": 1200, "wedge": 40}``
    cell_types: dict[str, int] = field(default_factory=dict)
    #: grid shape (X, Y[, Z]) for ``kind == "grid"`` / structured zones
    shape: list[int] | None = None
    bounds: list[float] | None = None  # xmin, xmax, ymin, ymax, zmin, zmax
    #: boundary-condition type when the format records one (``wall``, ``velocity-inlet``, ...)
    bc_type: str | None = None
    #: read by default? (boundary patches of a volume case usually are not)
    default: bool = True


@dataclass
class TimeInfo:
    values: list[float] = field(default_factory=list)
    units: str = "s"
    #: what the axis is: "time" | "iteration" | "load_step" | "frequency" | "mode"
    quantity: str = "time"

    @property
    def n_steps(self) -> int:
        return len(self.values)

    @property
    def steady(self) -> bool:
        return len(self.values) <= 1

    @property
    def dt(self) -> float | None:
        if len(self.values) < 2:
            return None
        return float(np.median(np.diff(np.asarray(self.values, dtype=np.float64))))

    @property
    def uniform(self) -> bool:
        if len(self.values) < 3:
            return True
        d = np.diff(np.asarray(self.values, dtype=np.float64))
        return bool(np.allclose(d, d[0], rtol=1e-6, atol=1e-12))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [float(v) for v in self.values],
            "n_steps": self.n_steps,
            "dt": self.dt,
            "units": self.units,
            "uniform": self.uniform,
            "quantity": self.quantity,
        }


@dataclass
class SourceDescription:
    """Everything ``describe`` can learn without reading the heavy arrays."""

    connector_id: str
    path: Path
    format: str = ""
    solver: str = "unknown"
    solver_version: str | None = None
    #: cfd | structural | thermal | multiphysics | materials | geophysical | other
    domain: str = "other"
    #: physics model: ``{"turbulence_model": "kOmegaSST", "compressible": False, ...}``
    physics: dict[str, Any] = field(default_factory=dict)
    zones: list[ZoneInfo] = field(default_factory=list)
    fields: list[FieldInfo] = field(default_factory=list)
    time: TimeInfo = field(default_factory=TimeInfo)
    #: per patch/zone: ``{"inlet": {"type": "fixedValue", "U": [10, 0, 0]}}``
    boundary_conditions: dict[str, Any] = field(default_factory=dict)
    #: unit system of the source: ``{"length": "mm", "system": "mm-t-s"}``
    units: dict[str, str] = field(default_factory=dict)
    #: scalar conditioning parameters (Re, Ma, AoA, material constants, ...)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def temporal(self) -> str:
        return "steady" if self.time.steady else "transient"

    def zone(self, name: str) -> ZoneInfo:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["time"] = self.time.to_dict()
        d["temporal"] = self.temporal
        return _jsonable(d)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, str | int | float | bool) or o is None:
        return o
    return str(o)


def jsonable(o: Any) -> Any:
    """Convert numpy scalars/arrays, paths and bytes into plain JSON types (NaN -> null)."""
    return _jsonable(o)


@dataclass
class FieldData:
    """One field array plus the labels a reader needs.

    ``dims`` names every axis of ``data``: ``["point","component"]``,
    ``["time","cell","component"]``, ``["time","x","y","component"]``, ``["component"]``
    (global scalar/vector) ... The last axis is always ``"component"``.
    """

    data: Any  # np.ndarray or a lazy array with .shape/.dtype/__getitem__
    dims: list[str]
    location: str = "point"  # Location
    units: str = ""
    component_names: list[str] | None = None
    description: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def components(self) -> int:
        return self.shape[-1] if self.dims and self.dims[-1] == "component" else 1

    @property
    def time_dependent(self) -> bool:
        return bool(self.dims) and self.dims[0] == "time"

    def validate(self, name: str = "") -> None:
        if len(self.dims) != len(self.shape):
            raise ValueError(
                f"field {name!r}: dims {self.dims} do not match array shape {self.shape}"
            )
        if self.dims[-1] != "component":
            raise ValueError(f"field {name!r}: last dim must be 'component', got {self.dims}")


def as_field(
    values: Any,
    location: str = "point",
    *,
    units: str = "",
    component_names: list[str] | None = None,
    description: str = "",
) -> FieldData:
    """Wrap a per-step ``(N,)`` or ``(N, C)`` array as ``FieldData`` with dims
    ``[location, "component"]`` (float32 unless integer)."""
    arr = np.asarray(values)
    if arr.dtype.kind in "fc" or arr.dtype.kind not in "iub":
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return FieldData(
        data=np.ascontiguousarray(arr),
        dims=[location, "component"],
        location=location,
        units=units,
        component_names=component_names,
        description=description,
    )


@dataclass
class RawSample:
    """Neutral sample (or one time step of it). See the module docstring for the two forms.

    Geometry by ``kind``:

    * ``surface``: ``points (N,3)``, ``faces (M,3)`` triangles.
    * ``mesh``: ``points (N,D)``, ``cells (K,P)`` point ids padded with -1, ``cell_types (K,)``
      VTK cell type ids. Polyhedra (type 42) list their unique point ids.
    * ``grid``: ``coords`` (``x``/``y``/``z`` 1-D axes); fields shaped ``([T,] X, Y[, Z], C)``.
    * ``point_cloud``: ``points (N,D)``.
    """

    id: str
    kind: str  # SampleKind
    zone: str | None = None
    points: np.ndarray | None = None
    cells: np.ndarray | None = None
    cell_types: np.ndarray | None = None
    faces: np.ndarray | None = None
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    fields: dict[str, FieldData] = field(default_factory=dict)
    #: time-independent inputs (geometry masks, material maps, wall distance, zone ids)
    static: dict[str, FieldData] = field(default_factory=dict)
    time: float | None = None
    time_index: int | None = None
    times: np.ndarray | None = None
    #: solver, solver_version, physics, boundary_conditions, parameters, units, source, zone
    #: role, ... (copied into ``meta.json``)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def block_form(self) -> bool:
        return self.times is not None

    @property
    def transient(self) -> bool:
        return self.times is not None or self.time is not None

    @property
    def n_points(self) -> int:
        return 0 if self.points is None else int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        if self.kind == "surface":
            return 0 if self.faces is None else int(self.faces.shape[0])
        return 0 if self.cells is None else int(self.cells.shape[0])

    def validate(self) -> None:
        """Raise ``ValueError`` when shapes and ``dims`` are inconsistent."""
        if self.kind not in ("surface", "mesh", "grid", "point_cloud"):
            raise ValueError(f"sample {self.id!r}: unknown kind {self.kind!r}")
        if self.points is not None and self.points.ndim != 2:
            raise ValueError(f"sample {self.id!r}: points must be (N, D), got {self.points.shape}")
        if self.kind == "surface" and self.faces is not None:
            if self.faces.ndim != 2 or self.faces.shape[1] != 3:
                raise ValueError(f"sample {self.id!r}: faces must be (M, 3)")
        if self.kind == "mesh" and self.cells is not None:
            if self.cell_types is None or len(self.cell_types) != len(self.cells):
                raise ValueError(f"sample {self.id!r}: cell_types must have one id per cell")
        sizes = {"point": self.n_points or None, "cell": self.n_cells or None}
        for group in (self.fields, self.static):
            for name, f in group.items():
                f.validate(name)
                for axis, dim in zip(f.shape, f.dims, strict=True):
                    want = sizes.get(dim)
                    if want is not None and axis != want:
                        raise ValueError(
                            f"sample {self.id!r} field {name!r}: axis '{dim}' has {axis} "
                            f"entries, geometry has {want}"
                        )
                    if dim == "time" and self.times is not None and axis != len(self.times):
                        raise ValueError(
                            f"sample {self.id!r} field {name!r}: {axis} steps but "
                            f"{len(self.times)} times"
                        )
                    if dim in self.coords and axis != len(self.coords[dim]):
                        raise ValueError(
                            f"sample {self.id!r} field {name!r}: axis '{dim}' has {axis} "
                            f"entries, coords['{dim}'] has {len(self.coords[dim])}"
                        )


@runtime_checkable
class Connector(Protocol):
    """The connector plugin contract.

    Register an instance factory under the entry-point group ``frontier.connectors``.
    Optional class attributes read by the registry (all have defaults): ``extensions``
    (``[".cgns"]``), ``maturity`` (one of :data:`MATURITIES`), ``temporal``
    (``["steady","transient"]``), ``kinds`` (``["surface","mesh"]``), ``requires`` (optional
    pip packages), ``export_help`` (how to get data out of the vendor tool), ``status_note``.
    ``describe``/``read`` may also accept ``options: Mapping[str, Any] | None``.
    """

    id: str
    name: str
    formats: list[str]
    vendor: str
    license: str

    def detect(self, path: Path) -> Detection | None: ...

    def describe(self, path: Path) -> SourceDescription: ...

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
    ) -> Iterator[RawSample]: ...


class BaseConnector:
    """Convenience base with the optional attributes filled in and extension-based detection.

    Subclasses set ``id``, ``name``, ``formats``, ``vendor``, ``license`` and implement
    ``describe`` and ``read``; override ``detect`` when magic bytes or layout are available.
    """

    id: str = ""
    name: str = ""
    formats: list[str] = []
    vendor: str = ""
    license: str = "Apache-2.0"
    extensions: list[str] = []
    maturity: str = "planned"
    temporal: list[str] = ["steady"]
    kinds: list[str] = ["mesh"]
    #: optional pip packages (extra engines or variants); `hard_requires` are the ones without
    #: which the connector cannot read anything at all
    requires: list[str] = []
    hard_requires: list[str] = []
    export_help: str = ""
    status_note: str = ""
    #: confidence reported for an extension-only match
    extension_confidence: float = 0.6

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        if ext is None or not path.is_file():
            return None
        return Detection(
            connector_id=self.id,
            path=path,
            confidence=self.extension_confidence,
            format=self.formats[0] if self.formats else self.id,
            notes=[f"extension {ext}"],
        )

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        raise NotImplementedError

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        raise NotImplementedError


def match_extension(path: Path, extensions: Sequence[str]) -> str | None:
    """Return the entry of ``extensions`` that ``path`` ends with (case-insensitive, compound
    suffixes like ``.cas.h5`` supported, longest first), else ``None``."""
    name = Path(path).name.lower()
    for ext in sorted(extensions, key=len, reverse=True):
        if name.endswith(ext.lower()):
            return ext
    return None
