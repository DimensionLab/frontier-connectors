# SPDX-License-Identifier: Apache-2.0
"""Open mesh / result formats: Exodus II, XDMF, Gmsh MSH, MED (Salome, Code_Aster,
Code_Saturne), SU2 native meshes and the remaining meshio formats (Medit, Netgen, TetGen,
UGRID, MOAB, FLAC3D, PERMAS, DOLFIN XML, AVS-UCD, Kratos MDPA, HMF).

How each format is read:

* **Exodus II** through VTK's ``vtkExodusIIReader`` (bundled netCDF), one time step at a time.
* **XDMF** through ``meshio`` (single unstructured grids, meshio-style time series) or VTK's
  ``vtkXdmfReader`` (everything else: structured topologies, spatial collections, dolfinx and
  ParaView time series).
* **Gmsh** mesh through ``meshio``; the ``$NodeData`` / ``$ElementData`` post-processing views
  (including multi-step views) with a small parser here.
* **MED** and **SU2** with readers written here on h5py / numpy (meshio's MED reader keeps one
  time step and MED's node order; its SU2 reader loses the marker names).

All of them share the zone rules of :mod:`frontier_connectors._meshio` and the
option ``planar_as`` (``"mesh"`` by default: a planar 2-D mesh is a 2-D domain with
``points (N, 2)``; ``"surface"`` makes it a triangulated surface in z = 0).
"""

from __future__ import annotations

import mmap
import re
import xml.etree.ElementTree as ET  # noqa: N817
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import _meshio as mz
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
    match_extension,
)
from frontier_connectors.paths import case_root_of, h5_check_file, h5_open, resolve_in_case
from frontier_connectors.select import select_time_indices, wants

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# -- shared helpers -----------------------------------------------------------------------------


def _head(path: Path, n: int = 4096) -> bytes:
    """First bytes of a file; empty for directories and unreadable paths (never raises)."""
    try:
        if not path.is_file():
            return b""
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise UnsupportedSourceError(f"{path} is not a file", hint="pass the mesh/result file")
    return path


def _clean(value: Any) -> str:
    """HDF5 / NetCDF fixed-width string attribute -> stripped ``str``."""
    if isinstance(value, bytes | np.bytes_):
        value = bytes(value).decode("utf-8", "replace")
    return str(value).replace("\x00", " ").strip()


def read_meshio(path: Path, file_format: str | None = None) -> Any:
    """``meshio.read`` with its failure modes (``ReadError``, ``SystemExit`` from the XDMF
    reader, stray ``KeyError``/``ValueError`` on corrupt files) turned into ``ConnectorError``."""
    import meshio

    try:
        return meshio.read(str(path), file_format=file_format)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - meshio raises anything on bad input
        detail = str(e).strip() or type(e).__name__
        raise ConnectorError(
            f"meshio could not read {path} as {file_format or 'auto'}: {detail[:300]}",
            hint="the file is truncated, corrupt or uses a variant meshio does not support; "
            "re-export it or convert it with the tool that wrote it",
        ) from e


@dataclass
class _Loaded:
    """A parsed source: the zoned mesh plus whole-mesh data, steady or per time step."""

    zoned: mz.ZonedMesh
    point_data: dict[str, Any] = field(default_factory=dict)
    cell_data: dict[str, Any] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    #: step index -> (point_data, cell_data) of the time-dependent fields
    load_step: Callable[[int], tuple[dict[str, Any], dict[str, Any]]] | None = None
    transient_fields: list[FieldInfo] = field(default_factory=list)
    format: str = ""
    solver: str = "unknown"
    solver_version: str | None = None
    domain: str | None = None
    physics: dict[str, Any] = field(default_factory=dict)
    boundary_conditions: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    field_units: dict[str, str] = field(default_factory=dict)
    component_names: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    time_units: str = "s"
    close: Callable[[], None] | None = None

    @property
    def transient(self) -> bool:
        return self.load_step is not None and len(self.times) > 1


class _ZonedConnector(BaseConnector):
    """Shared ``describe``/``read`` for connectors that parse into a :class:`_Loaded`."""

    kinds = ["mesh", "surface", "point_cloud"]

    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        raise NotImplementedError

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = _require_file(Path(path))
        return self._describe_loaded(self._load(path, options or {}), path)

    def _describe_loaded(self, ld: _Loaded, path: Path) -> SourceDescription:
        try:
            steady = mz.describe_fields(
                ld.zoned,
                ld.point_data,
                ld.cell_data,
                units=ld.field_units,
                component_names=ld.component_names,
                time_dependent=not ld.transient,
            )
            names = [f.name for f in steady + ld.transient_fields]
            return SourceDescription(
                connector_id=self.id,
                path=path,
                format=ld.format or self.formats[0],
                solver=ld.solver,
                solver_version=ld.solver_version,
                domain=ld.domain or mz.guess_domain(names),
                physics=ld.physics,
                zones=ld.zoned.infos(),
                fields=steady + ld.transient_fields,
                time=TimeInfo(values=list(ld.times) if ld.transient else [], units=ld.time_units),
                boundary_conditions=ld.boundary_conditions,
                units=ld.units,
                parameters=ld.parameters,
                metadata=ld.metadata,
                warnings=ld.warnings + ld.zoned.warnings,
            )
        finally:
            if ld.close is not None:
                ld.close()

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        path = _require_file(Path(path))
        ld = self._load(path, options or {})
        yield from self._read_loaded(ld, path, zones, fields, times)

    def _read_loaded(
        self,
        ld: _Loaded,
        path: Path,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
    ) -> Iterator[RawSample]:
        try:
            metadata: dict[str, Any] = {
                "solver": ld.solver,
                "source": str(path),
                "format": ld.format or self.formats[0],
            }
            if ld.solver_version:
                metadata["solver_version"] = ld.solver_version
            for key in ("physics", "boundary_conditions", "parameters", "units"):
                if getattr(ld, key):
                    metadata[key] = getattr(ld, key)
            if ld.transient:
                metadata["time_units"] = ld.time_units
            yield from mz.iter_samples(
                ld.zoned,
                meshutil.default_sample_id(path),
                self._steps(ld, times),
                zones=zones,
                fields=fields,
                metadata=metadata,
                units=ld.field_units,
                component_names=ld.component_names,
                static_point_data=ld.point_data if ld.transient else None,
                static_cell_data=ld.cell_data if ld.transient else None,
            )
        finally:
            if ld.close is not None:
                ld.close()

    @staticmethod
    def _steps(ld: _Loaded, times: TimeSelection) -> Iterator[mz.Step]:
        if not ld.transient or ld.load_step is None:
            pd, cd = dict(ld.point_data), dict(ld.cell_data)
            if ld.load_step is not None and ld.times:
                # a single stored step is a steady result
                spd, scd = ld.load_step(0)
                pd.update(spd)
                cd.update(scd)
            yield None, None, pd, cd
            return
        for ti in select_time_indices(ld.times, times):
            pd, cd = ld.load_step(ti)
            yield ti, float(ld.times[ti]), pd, cd


# =============================================================================================
# Exodus II
# =============================================================================================

_EXODUS_NAME = re.compile(r"\.(e|exo|ex2|exii|g|gen)(-s\.?\d+)?(\.\d+\.\d+)?$", re.IGNORECASE)
_EXODUS_KEYS = (b"num_el_blk", b"num_elem", b"num_nodes", b"num_dim", b"coor_names")


def _exodus_signature(path: Path) -> str | None:
    """``"netcdf3"`` / ``"netcdf4"`` when the file is NetCDF *and* carries Exodus dimension
    names, else ``None`` (so plain ``.nc`` files and BYU ``.g`` files are left alone)."""
    head = _head(path, 1 << 16)
    if head[:3] == b"CDF" and head[3:4] in (b"\x01", b"\x02", b"\x05"):
        # NetCDF classic lists the dimension names first
        if sum(k in head for k in _EXODUS_KEYS) >= 2:
            return "netcdf3"
        return None
    if head.startswith(_HDF5_MAGIC):
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if sum(k.decode() in f for k in _EXODUS_KEYS) >= 2:
                    return "netcdf4"
        except Exception:  # noqa: BLE001 - detection must never raise
            return None
    return None


def _points_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    return a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)


def _leaf_is_flat(ds: Any) -> bool:
    b = ds.GetBounds()
    return b[4] == 0.0 and b[5] == 0.0


def _vtk_flat(leaves: Sequence[tuple[str, Any]], planar_as: str) -> bool:
    """The shared planar rule for sources read through VTK: every block lies in z = 0 and the
    highest cell dimension is 2."""
    if planar_as != "mesh" or not leaves:
        return False
    kinds = {_vtk.dataset_kind(ds) for _, ds in leaves}
    return "surface" in kinds and all(_leaf_is_flat(ds) for _, ds in leaves)


def _only_vertices(ds: Any) -> bool:
    """Unstructured grid made of vertex / poly-vertex cells only (particles, node sets)."""
    if not ds.IsA("vtkUnstructuredGrid") or ds.GetNumberOfCells() == 0:
        return False
    return bool(np.isin(_vtk.cell_types(ds), (1, 2)).all())


def _vtk_zone_info(name: str, ds: Any, *, flat: bool, role: str | None, default: bool) -> ZoneInfo:
    info = _vtk.zone_info(name, ds, role=role, default=default)
    if _only_vertices(ds):
        info.kind, info.role = "point_cloud", role or "probes"
    if flat:
        info.dim = 2
        if info.kind == "surface":
            info.kind, info.role = "mesh", role or "volume"
    return info


def _vtk_sample(
    ds: Any,
    sample_id: str,
    *,
    zone: str,
    fields: Sequence[str] | None,
    time: float | None,
    time_index: int | None,
    metadata: dict[str, Any],
    flat: bool,
) -> RawSample:
    force = "mesh" if flat and _vtk.dataset_kind(ds) == "surface" else None
    if _only_vertices(ds):
        force = "point_cloud"
    s = _vtk.dataset_to_sample(
        ds,
        sample_id,
        zone=zone,
        fields=fields,
        time=time,
        time_index=time_index,
        metadata=metadata,
        force_kind=force,
    )
    if flat and s.points is not None and s.kind != "grid":
        s.points = np.ascontiguousarray(s.points[:, :2])
    if s.kind == "mesh" and s.cells is not None and s.cell_types is not None and not flat:
        cells = np.array(s.cells, dtype=np.int64)
        n = mz.orient_cells(np.asarray(s.points), cells, np.asarray(s.cell_types))
        if n:
            s.cells = cells
            s.metadata["reoriented_cells"] = n
    return s


def _vtk_steps(
    reader: Any,
    base_id: str,
    names: Callable[[str], str | None],
    *,
    selected: Callable[[list[str]], list[str]],
    fields: Sequence[str] | None,
    times: TimeSelection,
    metadata: dict[str, Any],
    planar_as: str,
    step_metadata: Callable[[Any, int], dict[str, Any]] | None = None,
    zone_metadata: Callable[[str], dict[str, Any]] | None = None,
    positional: bool = False,
    prepare: Callable[[Any], None] | None = None,
    finish: Callable[[RawSample], None] | None = None,
) -> Iterator[RawSample]:
    """Stream a time-aware VTK reader, one step in memory. ``names`` maps a block path to the
    zone name (``None`` drops the block); ``selected`` picks the zones from those present.
    Geometry is emitted once per zone and again only when the points really change.

    ``positional``: blocks of later steps take the zone names of the first step by position
    (XDMF names every step's grids differently). ``prepare`` may repair a dataset in place
    before conversion and ``finish`` the sample after it."""
    all_times = _vtk.reader_times(reader)
    transient = len(all_times) > 1
    chosen: list[str] | None = None
    flat = False
    reference: dict[str, np.ndarray] = {}
    first_names: list[str] = []
    for ti in select_time_indices(all_times, times):
        t = all_times[ti] if all_times else None
        obj = _vtk.update_to_time(reader, t)
        leaves = [(n, ds) for n, ds in ((names(raw), ds) for raw, ds in _vtk.iter_leaves(obj)) if n]
        if not leaves:
            raise ConnectorError(
                "the reader produced no geometry",
                hint="the file is truncated, or its heavy-data files are missing",
            )
        if positional and first_names and len(first_names) == len(leaves):
            leaves = [(n, ds) for n, (_, ds) in zip(first_names, leaves, strict=True)]
        if prepare is not None:
            for _, ds in leaves:
                prepare(ds)
        if chosen is None:
            first_names = [n for n, _ in leaves]
            chosen = selected(first_names)
            flat = _vtk_flat(leaves, planar_as)
        for name, ds in leaves:
            if name not in chosen:
                continue
            meta = dict(metadata)
            if zone_metadata is not None:
                meta.update(zone_metadata(name))
            if step_metadata is not None:
                meta.update(step_metadata(ds, ti))
            sid = meshutil.zone_sample_id(base_id, name, len(chosen))
            s = _vtk_sample(
                ds,
                sid,
                zone=name,
                fields=fields,
                time=float(t) if transient and t is not None else None,
                time_index=ti if transient else None,
                metadata=meta,
                flat=flat,
            )
            if sid in reference:
                if s.kind == "grid" or _points_equal(s.points, reference[sid]):
                    s.points = s.cells = s.cell_types = s.faces = None
                    s.static.pop("face_parent", None)
            elif s.points is not None:
                reference[sid] = s.points
            if finish is not None:
                finish(s)
            yield s


class ExodusConnector(BaseConnector):
    id = "exodus"
    name = "Exodus II (Sandia SEACAS, MOOSE, libMesh, Cubit, Sierra)"
    formats = ["exodus-ii"]
    vendor = "Sandia National Laboratories (open format)"
    license = "Apache-2.0 (reader: VTK vtkExodusIIReader, BSD-3)"
    extensions = [".e", ".exo", ".ex2", ".exii", ".g", ".gen"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["mesh", "surface", "point_cloud"]
    export_help = (
        "MOOSE: [Outputs] exodus = true. Sierra/SEACAS/Cubit/libMesh write .e/.exo/.g "
        "directly; join parallel pieces (file.e.N.i) with SEACAS epu first."
    )
    status_note = (
        "Element blocks are the default zones; side sets and node sets are extra zones. Reads "
        "nodal, element and global variables at all time steps. One piece of a parallel "
        "decomposition is read as it is, without merging."
    )

    _CATEGORIES = (
        ("Element Blocks", "ELEM_BLOCK", "block", "volume"),
        ("Side Sets", "SIDE_SET", "sideset", "boundary"),
        ("Node Sets", "NODE_SET", "nodeset", "probes"),
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        named = _EXODUS_NAME.search(path.name) is not None
        head = _head(path, 8)
        if not (head[:3] == b"CDF" or head.startswith(_HDF5_MAGIC)):
            return None
        if not named and path.suffix.lower() in (".h5", ".hdf5", ".h5m", ".med", ".cgns"):
            return None  # other HDF5 dialects with their own connectors
        sig = _exodus_signature(path)
        if sig is None:
            return None
        notes = [f"{sig} container with Exodus II dimensions"]
        piece = re.search(r"\.(\d+)\.(\d+)$", path.name)
        if piece:
            notes.append(f"piece {piece.group(2)} of a {piece.group(1)}-way parallel decomposition")
        return Detection(self.id, path, 0.97 if named else 0.9, "exodus-ii", notes)

    # -- reader plumbing ----------------------------------------------------------------
    def _reader(self, path: Path) -> Any:
        from vtkmodules.vtkIOExodus import vtkExodusIIReader

        path = _require_file(Path(path))
        if _exodus_signature(path) is None:
            raise UnsupportedSourceError(
                f"{path} is not an Exodus II (NetCDF) file",
                hint="Exodus files are NetCDF with dimensions such as num_nodes / num_el_blk",
            )
        reader = vtkExodusIIReader()
        reader.SetFileName(str(path))
        reader.UpdateInformation()
        if reader.GetNumberOfNodes() <= 0:
            raise ConnectorError(
                f"VTK could not read the Exodus header of {path}",
                hint="the file is truncated or was written with an incompatible NetCDF build",
            )
        reader.GenerateGlobalNodeIdArrayOff()
        reader.GenerateGlobalElementIdArrayOff()
        reader.GenerateObjectIdCellArrayOff()
        return reader

    def _plan(self, reader: Any) -> list[tuple[str, str, int, int, str]]:
        """``(zone name, block path, object type, object index, role)`` for every element
        block, side set and node set. Unnamed objects are called ``block_<id>``,
        ``sideset_<id>``, ``nodeset_<id>``; a set that shares its name with an element block
        gets the category prefix."""
        plan: list[tuple[str, str, int, int, str]] = []
        taken: set[str] = set()
        for category, attr, prefix, role in self._CATEGORIES:
            otype = getattr(reader, attr)
            for i in range(reader.GetNumberOfObjects(otype)):
                raw = reader.GetObjectName(otype, i) or ""
                oid = reader.GetObjectId(otype, i)
                if not raw or raw.startswith("Unnamed "):
                    name = f"{prefix}_{oid}"
                else:
                    name = raw.strip()
                    if prefix == "nodeset" or name in taken:
                        name = f"{prefix}_{name}"
                while name in taken:
                    name += "_"
                taken.add(name)
                plan.append((name, f"{category}/{raw}", otype, i, role))
        return plan

    @staticmethod
    def _enable_arrays(reader: Any, on: bool, fields: Sequence[str] | None = None) -> None:
        for attr in ("NODAL", "ELEM_BLOCK", "SIDE_SET", "NODE_SET", "GLOBAL"):
            otype = getattr(reader, attr)
            for i in range(reader.GetNumberOfObjectArrays(otype)):
                name = reader.GetObjectArrayName(otype, i)
                keep = on and (attr == "GLOBAL" or wants(name, fields))
                reader.SetObjectArrayStatus(otype, i, 1 if keep else 0)

    @staticmethod
    def _char_rows(ds: Any, name: str, limit: int = 4096) -> list[str]:
        """Rows of a fixed-width character field-data array (QA and info records)."""
        from vtkmodules.util.numpy_support import vtk_to_numpy

        arr = ds.GetFieldData().GetAbstractArray(name)
        if arr is None or arr.GetNumberOfTuples() == 0:
            return []
        if arr.IsA("vtkStringArray"):
            n = min(arr.GetNumberOfValues(), limit)
            return [str(arr.GetValue(i)).replace("\x00", "").strip() for i in range(n)]
        rows = np.asarray(vtk_to_numpy(arr)).reshape(arr.GetNumberOfTuples(), -1)[:limit]
        return [bytes(r.astype(np.uint8)).split(b"\x00")[0].decode("latin-1").strip() for r in rows]

    def _qa(self, ds: Any) -> tuple[str | None, str | None, str | None]:
        """``(code name, code version, title)``. The last QA record names the code that wrote
        the results; MOOSE writes no QA record but signs its info records."""
        title = next(iter(self._char_rows(ds, "Title")), "") or None
        qa = self._char_rows(ds, "QA_Records")
        if len(qa) >= 4:
            rec = qa[4 * (len(qa) // 4 - 1) :][:4]
            return rec[0] or None, rec[1] or None, title
        if any("Created by MOOSE" in row for row in self._char_rows(ds, "Info_Records", 16)):
            return "MOOSE", None, title
        return None, None, title

    @staticmethod
    def _globals(ds: Any, n_times: int) -> dict[str, np.ndarray]:
        from vtkmodules.util.numpy_support import vtk_to_numpy

        out: dict[str, np.ndarray] = {}
        fd = ds.GetFieldData()
        for i in range(fd.GetNumberOfArrays()):
            arr = fd.GetArray(i)
            if arr is None or arr.GetName() in ("ElementBlockIds", None):
                continue
            if n_times and arr.GetNumberOfTuples() == n_times and arr.GetNumberOfComponents() == 1:
                out[arr.GetName()] = np.array(vtk_to_numpy(arr), dtype=np.float64)
        return out

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        reader = self._reader(path)
        plan = self._plan(reader)
        for _, _, otype, i, _ in plan:
            reader.SetObjectStatus(otype, i, 1)
        self._enable_arrays(reader, False)
        for i in range(reader.GetNumberOfObjectArrays(reader.GLOBAL)):
            reader.SetObjectArrayStatus(reader.GLOBAL, i, 1)
        reader.ApplyDisplacementsOff()
        times = _vtk.reader_times(reader)
        obj = _vtk.update_to_time(reader, times[0] if times else None)
        by_path = {p: (name, role) for name, p, _, _, role in plan}
        leaves = [(by_path[raw], ds) for raw, ds in _vtk.iter_leaves(obj) if raw in by_path]
        if not leaves:
            raise ConnectorError(
                f"{path} holds no element blocks", hint="the Exodus file has no mesh"
            )
        flat = _vtk_flat([(n, ds) for (n, _), ds in leaves], mz.planar_option(options))
        desc = SourceDescription(connector_id=self.id, path=path, format="exodus-ii")
        for (name, role), ds in leaves:
            zrole = None if role == "volume" else role
            info = _vtk_zone_info(name, ds, flat=flat, role=zrole, default=role == "volume")
            if role == "volume" and info.kind == "surface":
                info.role = "part"  # shell element block
            desc.zones.append(info)
        blocks = [z.name for z in desc.zones if z.default]
        sets = [n for n, _, _, _, role in plan if role == "boundary"]
        for attr, location, zone_names in (
            ("NODAL", "point", None),
            ("ELEM_BLOCK", "cell", blocks),
            ("SIDE_SET", "cell", sets),
        ):
            otype = getattr(reader, attr)
            for i in range(reader.GetNumberOfObjectArrays(otype)):
                desc.fields.append(
                    FieldInfo(
                        name=reader.GetObjectArrayName(otype, i),
                        location=location,
                        components=reader.GetNumberOfObjectArrayComponents(otype, i),
                        zones=zone_names,
                    )
                )
        first = leaves[0][1]
        code, version, title = self._qa(first)
        desc.solver = code or "unknown"
        desc.solver_version = version
        desc.domain = mz.guess_domain([f.name for f in desc.fields])
        desc.time = TimeInfo(values=times if len(times) > 1 else [])
        if reader.GetHasModeShapes():
            desc.time.quantity = "mode"
        if title:
            desc.metadata["title"] = title
        glob = self._globals(first, len(times))
        if glob:
            desc.metadata["global_variables"] = {k: v.tolist() for k, v in glob.items()}
        if any(f.name.upper().startswith("DISP") and f.components > 1 for f in desc.fields):
            desc.metadata["displacement_applied"] = False
            desc.warnings.append(
                'points are the reference mesh; pass options={"apply_displacements": True} '
                "to move them with the displacement field"
            )
        if re.search(r"\.\d+\.\d+$", path.name):
            desc.warnings.append(
                "this is one piece of a parallel decomposition; join the pieces with SEACAS "
                "epu to read the whole mesh"
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
        opts = options or {}
        reader = self._reader(path)
        plan = self._plan(reader)
        names = [p[0] for p in plan]
        chosen = _vtk.select_zones(names, zones, [p[0] for p in plan if p[4] == "volume"])
        for name, _, otype, i, _ in plan:
            reader.SetObjectStatus(otype, i, 1 if name in chosen else 0)
        self._enable_arrays(reader, True, fields)
        reader.SetApplyDisplacements(1 if opts.get("apply_displacements") else 0)
        by_path = {p: name for name, p, _, _, _ in plan}
        roles = {name: role for name, _, _, _, role in plan}
        n_times = len(_vtk.reader_times(reader))
        info: dict[str, Any] = {}

        def step_metadata(ds: Any, ti: int) -> dict[str, Any]:
            if not info:
                code, version, title = self._qa(ds)
                info.update({"solver": code or "unknown"})
                if version:
                    info["solver_version"] = version
                if title:
                    info["title"] = title
            out = dict(info)
            glob = self._globals(ds, n_times)
            if glob:
                out["globals"] = {k: float(v[ti]) for k, v in glob.items()}
            return out

        yield from _vtk_steps(
            reader,
            meshutil.default_sample_id(path),
            lambda raw: by_path.get(raw),
            selected=lambda present: [n for n in chosen if n in present],
            fields=fields,
            times=times,
            metadata={"source": str(path), "format": "exodus-ii"},
            planar_as=mz.planar_option(opts),
            step_metadata=step_metadata,
            zone_metadata=lambda name: {"zone_role": roles[name]},
        )


# =============================================================================================
# XDMF
# =============================================================================================

_XI = ("{http://www.w3.org/2001/XInclude}include", "{http://www.w3.org/2003/XInclude}include")
_STRUCTURED = ("2dcorectmesh", "3dcorectmesh", "2drectmesh", "3drectmesh", "2dsmesh", "3dsmesh")


@dataclass
class _XdmfScan:
    version: str = ""
    grids: int = 0
    temporal: bool = False
    spatial: bool = False
    meshio_series: bool = False
    times: list[float] = field(default_factory=list)
    topologies: list[str] = field(default_factory=list)
    constructs: list[str] = field(default_factory=list)
    information: dict[str, str] = field(default_factory=dict)
    #: heavy-data files (HDF5 / raw binary) the light data points at, resolved
    heavy: list[Path] = field(default_factory=list)


_XDMF_EXTERNAL_ENTITY = re.compile(rb"<!ENTITY\s+(?:%\s*)?[^\s>]+\s+(?:SYSTEM|PUBLIC)\b")
#: light-data files are small; the DOCTYPE (where entities are declared) precedes the root
_XDMF_DOCTYPE_SCAN_BYTES = 1 << 20
_XDMF_INCLUDE_DEPTH = 8


def _xdmf_refuse_external_entities(path: Path) -> None:
    """VTK's Xdmf reader (libxml2) substitutes entities; an external one reads another file."""
    try:
        with open(path, "rb") as f:
            head = f.read(_XDMF_DOCTYPE_SCAN_BYTES)
    except OSError:
        return
    if _XDMF_EXTERNAL_ENTITY.search(head):
        raise ConnectorError(
            f"{path} declares an external XML entity (<!ENTITY ... SYSTEM/PUBLIC>), which "
            "would make the reader load another file",
            hint="remove the external entity; name heavy-data files in the DataItem text",
        )


def _xdmf_heavy(holder: Path, case_root: Path, text: str) -> list[Path]:
    """Heavy-data file a ``DataItem`` body names, confined to the case directory."""
    # "file.h5:/group/data" or "file.h5:group/data"; raw binary items name the file only
    body = text.strip()
    m = re.match(r"^(.+?\.\w+):", body)
    name = m.group(1) if m else body
    if not name:
        return []
    heavy = resolve_in_case(case_root, name, what="XDMF DataItem", referrer=holder)
    if heavy.is_file():
        h5_check_file(heavy, case_root=case_root, what="XDMF heavy-data link")
    return [heavy]


def _xdmf_includes(
    holder: Path, case_root: Path, root: ET.Element, scan: _XdmfScan, *, depth: int
) -> None:
    for el in root.iter():
        if el.tag not in _XI:
            continue
        href = el.get("href")
        if not href:
            continue  # an xpointer into the same document
        target = resolve_in_case(case_root, href, what="XDMF xi:include", referrer=holder)
        if depth >= _XDMF_INCLUDE_DEPTH:
            raise ConnectorError(
                f"{holder}: xi:include nesting deeper than {_XDMF_INCLUDE_DEPTH} levels",
                hint="flatten the includes",
            )
        if not target.is_file():
            continue
        _xdmf_refuse_external_entities(target)
        try:
            sub = ET.parse(target).getroot()
        except (ET.ParseError, OSError):
            continue  # not XML (xi:include parse="text"): nothing further to follow
        for item in sub.iter("DataItem"):
            if (item.get("Format") or "XML").upper() in ("HDF", "HDF5", "BINARY") and item.text:
                for heavy in _xdmf_heavy(target, case_root, item.text):
                    if heavy not in scan.heavy:
                        scan.heavy.append(heavy)
        _xdmf_includes(target, case_root, sub, scan, depth=depth + 1)


def scan_xdmf(path: Path) -> _XdmfScan:
    """Parse the light XML only: what kind of file is it, and which engine can read it.

    Every file the light data refers to (heavy-data files, XInclude targets) is confined to
    the case directory, i.e. the directory of ``path`` (see ``frontier_connectors.paths``)."""
    case_root = case_root_of(path)
    _xdmf_refuse_external_entities(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        raise ConnectorError(
            f"{path} is not valid XDMF XML: {e}", hint="the light-data file is truncated"
        ) from e
    if root.tag != "Xdmf":
        raise UnsupportedSourceError(
            f"{path} has root element <{root.tag}>, not <Xdmf>", hint="not an XDMF file"
        )
    scan = _XdmfScan(version=root.get("Version", ""))
    domain = root.find("Domain")
    if domain is None:
        raise ConnectorError(f"{path} has no <Domain>", hint="the XDMF file is empty")
    top = domain.findall("Grid")
    scan.grids = len(top)
    for g in root.iter("Grid"):
        if (g.get("GridType") or "").lower() == "collection":
            ctype = (g.get("CollectionType") or "spatial").lower()
            if ctype == "temporal":
                scan.temporal = True
                children = g.findall("Grid")
                if any(c.find(tag) is not None for c in children for tag in _XI):
                    scan.meshio_series = any((t.get("Name") or "") == "mesh" for t in top)
                if not scan.times:
                    for c in children:
                        t = c.find("Time")
                        if t is not None and t.get("Value") is not None:
                            try:
                                scan.times.append(float(t.get("Value", "nan")))
                            except ValueError:
                                pass
            else:
                scan.spatial = True
    for t in root.iter("Topology"):
        name = (t.get("TopologyType") or t.get("Type") or "").strip()
        if name and name not in scan.topologies:
            scan.topologies.append(name)
    for item in root.iter("DataItem"):
        kind = (item.get("ItemType") or "").lower()
        label = f"DataItem ItemType={item.get('ItemType')}"
        if kind in ("function", "hyperslab", "coordinates") and label not in scan.constructs:
            scan.constructs.append(label)
        if (item.get("Format") or "XML").upper() in ("HDF", "HDF5", "BINARY") and item.text:
            # "file.h5:/group/data" or "file.h5:group/data"; raw binary items name the file only
            for heavy in _xdmf_heavy(path, case_root, item.text):
                if heavy not in scan.heavy:
                    scan.heavy.append(heavy)
    # the VTK reader expands XInclude itself: confine each include and what it refers to
    _xdmf_includes(path, case_root, root, scan, depth=0)
    for a in root.iter("Attribute"):
        if (a.get("ItemType") or "") == "FiniteElementFunction":
            label = "Attribute ItemType=FiniteElementFunction"
            if label not in scan.constructs:
                scan.constructs.append(label)
    for i in domain.iter("Information"):
        if i.get("Name") and i.get("Value") is not None:
            scan.information[str(i.get("Name"))] = str(i.get("Value"))
    return scan


#: run in a child process: VTK's Xdmf2 reader crashes (SIGSEGV) on files it cannot parse
_XDMF_PROBE = """
import sys
from vtkmodules.vtkIOXdmf2 import vtkXdmfReader
r = vtkXdmfReader()
r.SetFileName(sys.argv[1])
r.UpdateInformation()
r.Update()
out = r.GetOutputDataObject(0)
sys.exit(0 if out is not None else 3)
"""
_xdmf_probed: dict[tuple[str, int, int], str | None] = {}


def probe_vtk_xdmf(path: Path) -> str | None:
    """Open ``path`` with VTK's Xdmf2 reader in a child process. Returns ``None`` when the
    first time step loads, else what the reader complained about. The result is cached per
    file version. Skipped (``None``) when no plain Python interpreter is available to run the
    probe, e.g. inside an embedding application."""
    import os
    import subprocess
    import sys

    exe = sys.executable or ""
    if "python" not in os.path.basename(exe).lower():
        return None
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _xdmf_probed:
        try:
            proc = subprocess.run(  # noqa: S603 - our own interpreter and a fixed script
                [exe, "-c", _XDMF_PROBE, str(path)],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            failed = proc.returncode != 0
            output = proc.stderr + proc.stdout
        except (OSError, subprocess.SubprocessError) as e:
            failed, output = True, str(e)
        messages = re.findall(r"XDMF Error in .*?\((.*)\)\s*$", output, flags=re.MULTILINE)
        reason = "; ".join(dict.fromkeys(messages)) or "the reader crashed"
        _xdmf_probed[key] = reason[:400] if failed else None
    return _xdmf_probed[key]


def _fix_tuple_counts(ds: Any) -> None:
    """VTK's Xdmf2 reader returns vectors with 2 or 4 components (and mis-declared ones) as a
    single long tuple: reshape them to one tuple per point/cell, drop what cannot be fixed."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    for attrs, n in (
        (ds.GetPointData(), ds.GetNumberOfPoints()),
        (ds.GetCellData(), ds.GetNumberOfCells()),
    ):
        arrays = [attrs.GetArray(i) for i in range(attrs.GetNumberOfArrays())]
        for arr in arrays:
            if arr is None or n == 0 or arr.GetNumberOfTuples() == n:
                continue
            name = arr.GetName()
            total = arr.GetNumberOfTuples() * arr.GetNumberOfComponents()
            values = np.array(vtk_to_numpy(arr)).reshape(-1)
            attrs.RemoveArray(name)
            if total % n == 0:
                fixed = numpy_to_vtk(np.ascontiguousarray(values.reshape(n, total // n)), deep=True)
                fixed.SetName(name)
                attrs.AddArray(fixed)


def _fix_plane_grid(s: RawSample) -> None:
    """VTK puts a 2-D XDMF CoRect/Rect mesh in the y-z plane (XDMF axes shifted by one): turn
    the single-layer grid back into an x-y grid."""
    if s.kind != "grid" or len(s.coords.get("x", [0, 0])) != 1 or "z" not in s.coords:
        return
    s.coords = {"x": s.coords["y"], "y": s.coords["z"]}
    for f in list(s.fields.values()) + list(s.static.values()):
        if len(f.dims) == 4 and f.shape[0] == 1:
            f.data = np.ascontiguousarray(np.asarray(f.data)[0])
            cell = f.dims[0].endswith("_cell")
            f.dims = ["x_cell", "y_cell", "component"] if cell else ["x", "y", "component"]
    for key in ("spacing", "origin"):
        if isinstance(s.metadata.get(key), list) and len(s.metadata[key]) == 3:
            s.metadata[key] = s.metadata[key][1:]


def _finish_xdmf_sample(s: RawSample) -> None:
    _fix_plane_grid(s)
    mz.grid_cell_fields_to_nodes(s)


class XdmfConnector(_ZonedConnector):
    id = "xdmf"
    name = "XDMF (FEniCS/dolfinx, Firedrake, PETSc, MOOSE, ParaView)"
    formats = ["xdmf3", "xdmf2"]
    vendor = "open format (Kitware / ARL)"
    license = "Apache-2.0 (readers: meshio MIT, VTK BSD-3)"
    extensions = [".xdmf", ".xmf"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["mesh", "surface", "grid", "point_cloud"]
    export_help = (
        "dolfinx: XDMFFile.write_mesh / write_function(u, t); legacy FEniCS: XDMFFile.write; "
        "ParaView: Save Data as Xdmf. Keep the .h5 heavy-data file next to the .xdmf."
    )
    status_note = (
        "Unstructured grids and meshio time series are read through meshio; structured topologies "
        "(CoRect, Rect, SMesh), spatial collections and dolfinx / ParaView temporal collections "
        "through VTK's Xdmf2 reader. FiniteElementFunction (dolfinx checkpoint) attributes are not "
        "evaluated: only the mesh of such files is read. Cell-centred data of regular grids is "
        "averaged to the grid nodes."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        head = _head(path, 2048)
        if b"<Xdmf" not in head:
            return None
        if ext is None and path.suffix.lower() != ".xml":
            return None
        return Detection(self.id, path, 0.97 if ext else 0.8, "xdmf", ["<Xdmf> root element"])

    def _engine(self, scan: _XdmfScan, options: Mapping[str, Any]) -> str:
        engine = str(options.get("engine", "auto")).lower()
        if engine not in ("auto", "meshio", "vtk"):
            raise ConnectorError(
                f"unknown XDMF engine {engine!r}", hint='use options={"engine": "meshio"|"vtk"}'
            )
        if engine != "auto":
            return "meshio-series" if engine == "meshio" and scan.temporal else engine
        if scan.meshio_series:
            return "meshio-series"
        structured = any(t.lower() in _STRUCTURED for t in scan.topologies)
        if scan.grids == 1 and not (scan.temporal or scan.spatial or structured):
            return "meshio"
        return "vtk"

    # -- VTK engine ---------------------------------------------------------------------
    @staticmethod
    def _check_heavy(path: Path, scan: _XdmfScan) -> None:
        missing = [h.name for h in scan.heavy if not h.is_file()]
        if missing:
            raise ConnectorError(
                f"{path}: heavy-data file(s) not found: {', '.join(missing[:5])}",
                hint="keep the .h5 / .bin files the XDMF refers to in the same directory",
            )

    def _vtk_reader(self, path: Path, scan: _XdmfScan, options: Mapping[str, Any]) -> Any:
        try:
            from vtkmodules.vtkIOXdmf2 import vtkXdmfReader
        except ImportError as e:
            raise ConnectorError(
                f"{path} needs VTK's Xdmf reader, which this VTK build lacks "
                f"(topologies: {', '.join(scan.topologies) or 'unknown'})",
                hint="install a VTK wheel with vtkIOXdmf2, or convert the file to .vtu/.pvd",
            ) from e
        if options.get("probe", True):
            reason = probe_vtk_xdmf(path)
            if reason is not None:
                raise self._vtk_fail(path, scan, reason)
        reader = vtkXdmfReader()
        reader.SetFileName(str(path))
        reader.UpdateInformation()
        return reader

    def _vtk_fail(self, path: Path, scan: _XdmfScan, reason: str = "") -> ConnectorError:
        what = ", ".join(scan.topologies + scan.constructs) or "unknown constructs"
        detail = f": {reason}" if reason else ""
        return ConnectorError(
            f"neither meshio nor VTK's Xdmf2 reader could read {path} (file uses {what}){detail}",
            hint="XDMF3-only features (Function items, some polyhedral or mixed layouts) and "
            "malformed light data are not supported: convert the file to .vtu/.pvd in ParaView",
        )

    def _vtk_describe(
        self, path: Path, scan: _XdmfScan, options: Mapping[str, Any]
    ) -> SourceDescription:
        reader = self._vtk_reader(path, scan, options)
        times = _vtk.reader_times(reader)
        obj = _vtk.update_to_time(reader, times[0] if times else None)
        leaves = list(_vtk.iter_leaves(obj))
        if not leaves:
            raise self._vtk_fail(path, scan)
        for _, ds in leaves:
            _fix_tuple_counts(ds)
        desc = SourceDescription(
            connector_id=self.id, path=path, format=f"xdmf{scan.version[:1] or '3'}"
        )
        flat = _vtk_flat(leaves, mz.planar_option(options))
        sub = SourceDescription(connector_id=self.id, path=path)
        _vtk.describe_object(obj, sub)
        desc.fields = sub.fields
        for name, ds in leaves:
            info = _vtk_zone_info(name, ds, flat=flat, role=None, default=True)
            if info.kind == "grid" and info.shape and info.shape[0] == 1:
                info.shape, info.dim = info.shape[1:], 2
            desc.zones.append(info)
        desc.time = TimeInfo(values=times if len(times) > 1 else [])
        desc.domain = mz.guess_domain([f.name for f in desc.fields])
        desc.metadata["engine"] = "vtk"
        return desc

    # -- meshio engines -----------------------------------------------------------------
    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        scan = scan_xdmf(path)
        engine = self._engine(scan, options)
        planar_as = mz.planar_option(options)
        fmt = f"xdmf{scan.version[:1] or '3'}"
        if engine == "meshio":
            mesh = read_meshio(path, "xdmf")
            zoned, pd, cd = mz.zoned_from_meshio(mesh, planar_as=planar_as)
            return _Loaded(zoned, pd, cd, format=fmt, metadata={"engine": "meshio"})
        import meshio

        try:
            reader = meshio.xdmf.TimeSeriesReader(str(path))
            reader.__enter__()
        except (Exception, SystemExit) as e:  # noqa: BLE001
            raise ConnectorError(
                f"meshio could not open the XDMF time series {path}: {str(e)[:200]}",
                hint='try options={"engine": "vtk"}',
            ) from e
        try:
            points, cells = reader.read_points_cells()
            n = int(reader.num_steps)
            mesh = meshio.Mesh(points, cells)
            zoned, _, _ = mz.zoned_from_meshio(mesh, planar_as=planar_as)
            times = scan.times if len(scan.times) == n else []
            first: tuple[float, dict[str, Any], dict[str, Any]] | None = None
            if n:
                first = reader.read_data(0)
            if not times:
                times = [float(reader.read_data(k)[0]) for k in range(n)]
        except (Exception, SystemExit) as e:  # noqa: BLE001
            reader.__exit__(None, None, None)
            raise ConnectorError(
                f"meshio could not read the XDMF time series {path}: {str(e)[:200]}",
                hint="check that the .h5 heavy-data file sits next to the .xdmf",
            ) from e

        def load_step(k: int) -> tuple[dict[str, Any], dict[str, Any]]:
            _, pd, cd = reader.read_data(k)
            return dict(pd), dict(cd)

        ld = _Loaded(
            zoned,
            times=[float(t) for t in times],
            load_step=load_step,
            format=fmt,
            metadata={"engine": "meshio"},
            close=lambda: reader.__exit__(None, None, None),
        )
        if first is not None:
            ld.transient_fields = mz.describe_fields(zoned, first[1], first[2], time_dependent=True)
            if n == 1:
                ld.point_data, ld.cell_data = dict(first[1]), dict(first[2])
                ld.load_step, ld.transient_fields = None, []
        return ld

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = _require_file(Path(path))
        opts = options or {}
        scan = scan_xdmf(path)
        self._check_heavy(path, scan)
        engine = self._engine(scan, opts)
        desc: SourceDescription | None = None
        if engine != "vtk":
            try:
                desc = self._describe_loaded(self._load(path, opts), path)
            except ConnectorError:
                if str(opts.get("engine", "auto")).lower() != "auto":
                    raise
        if desc is None:
            desc = self._vtk_describe(path, scan, opts)
        if "Attribute ItemType=FiniteElementFunction" in scan.constructs:
            desc.warnings.append(
                "FiniteElementFunction attributes (dolfinx checkpoints) are not read; write "
                "the function with XDMFFile.write_function for point/cell data"
            )
        desc.metadata["xdmf_version"] = scan.version
        desc.metadata["topologies"] = scan.topologies
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
        path = _require_file(Path(path))
        opts = options or {}
        scan = scan_xdmf(path)
        self._check_heavy(path, scan)
        engine = self._engine(scan, opts)
        if engine != "vtk":
            ld: _Loaded | None = None
            try:
                ld = self._load(path, opts)
            except ConnectorError:
                if str(opts.get("engine", "auto")).lower() != "auto":
                    raise
            if ld is not None:
                yield from self._read_loaded(ld, path, zones, fields, times)
                return
        reader = self._vtk_reader(path, scan, opts)
        try:
            yield from _vtk_steps(
                reader,
                meshutil.default_sample_id(path),
                lambda raw: raw,
                selected=lambda present: _vtk.select_zones(present, zones),
                fields=fields,
                times=times,
                metadata={"solver": "unknown", "source": str(path), "format": "xdmf"},
                planar_as=mz.planar_option(opts),
                positional=True,
                prepare=_fix_tuple_counts,
                finish=_finish_xdmf_sample,
            )
        except ConnectorError as e:
            if "no geometry" in e.message:
                raise self._vtk_fail(path, scan) from e
            raise


# =============================================================================================
# Gmsh
# =============================================================================================


@dataclass
class _GmshView:
    kind: str  # "NodeData" | "ElementData"
    name: str
    time: float
    step: int
    components: int
    items: int
    offset: int
    end: int
    #: byte range of the whole ``$...Data`` block, markers included
    block_start: int = 0
    block_end: int = 0


def _gmsh_header(path: Path) -> tuple[str, bool, int]:
    """``(version, binary, data size)`` from the ``$MeshFormat`` block."""
    head = _head(path, 256)
    lines = head.split(b"\n")
    if not head.startswith(b"$MeshFormat") or len(lines) < 2:
        raise UnsupportedSourceError(
            f"{path} does not start with $MeshFormat",
            hint="Fluent .msh files go through the fluent-legacy connector; Gmsh MSH1 files "
            "must be re-saved as MSH 2.2 or 4.1",
        )
    try:
        version, ftype, size = lines[1].split()[:3]
        return version.decode(), int(ftype) == 1, int(size)
    except ValueError as e:
        raise ConnectorError(
            f"{path}: malformed $MeshFormat line {lines[1][:40]!r}",
            hint="expected '<version> <file-type> <data-size>'",
        ) from e


def _tag_list(mm: Any, cursor: int, cast: Callable[[bytes], Any]) -> tuple[list[Any], int]:
    """Read ``<count>`` then ``count`` lines from ``cursor``; returns ``(values, new cursor)``.
    Raises ``ValueError`` when the block ends early or a line does not parse."""

    def line() -> bytes:
        nonlocal cursor
        end = mm.find(b"\n", cursor)
        if end < 0:
            raise ValueError("truncated")
        text = mm[cursor:end].strip()
        cursor = end + 1
        return text

    values = [cast(line()) for _ in range(int(line()))]
    return values, cursor


def scan_gmsh_views(path: Path) -> list[_GmshView]:
    """Index the ``$NodeData`` / ``$ElementData`` blocks (header only, no values read)."""
    views: list[_GmshView] = []
    if path.stat().st_size == 0:
        return views
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for kind in ("NodeData", "ElementData"):
            marker = b"\n$" + kind.encode()
            pos = 0
            while True:
                at = mm.find(marker, pos)
                if at < 0:
                    break
                pos = at + len(marker)
                cursor = mm.find(b"\n", at + 1) + 1
                try:
                    strings, cursor = _tag_list(mm, cursor, lambda b: b.decode().strip('"'))
                    reals, cursor = _tag_list(mm, cursor, float)
                    ints, cursor = _tag_list(mm, cursor, int)
                except ValueError as e:
                    raise ConnectorError(
                        f"{path}: malformed or truncated ${kind} header near byte {cursor}",
                        hint="expected string, real and integer tag lists; re-export the view",
                    ) from e
                if len(ints) < 3:
                    raise ConnectorError(
                        f"{path}: ${kind} view has {len(ints)} integer tags, need 3",
                        hint="time step, component count and item count are required",
                    )
                end = mm.find(b"$End" + kind.encode(), cursor)
                after = len(mm) if end < 0 else mm.find(b"\n", end)
                views.append(
                    _GmshView(
                        kind=kind,
                        name=strings[0] if strings and strings[0] else kind,
                        time=reals[0] if reals else float(ints[0]),
                        step=ints[0],
                        components=ints[1],
                        items=ints[2],
                        offset=cursor,
                        end=end if end >= 0 else len(mm),
                        block_start=at + 1,
                        block_end=len(mm) if after < 0 else after + 1,
                    )
                )
    return views


def read_gmsh_view(path: Path, view: _GmshView, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(tags (n,), values (n, components))`` of one view block."""
    with open(path, "rb") as f:
        f.seek(view.offset)
        if binary:
            dtype = np.dtype([("tag", "<i4"), ("values", "<f8", (view.components,))])
            raw = np.fromfile(f, dtype=dtype, count=view.items)
            if len(raw) != view.items:
                raise ConnectorError(
                    f"{path}: view {view.name!r} step {view.step} is truncated "
                    f"({len(raw)} of {view.items} items)",
                    hint="re-export the post-processing view",
                )
            return raw["tag"].astype(np.int64), np.array(raw["values"], dtype=np.float64)
        text = f.read(max(view.end - view.offset, 0))
    try:
        flat = np.array(text.split(), dtype=np.float64)
    except ValueError as e:
        raise ConnectorError(
            f"{path}: non-numeric value in view {view.name!r} step {view.step}",
            hint="the ASCII view block is corrupt",
        ) from e
    width = 1 + view.components
    if len(flat) != view.items * width:
        raise ConnectorError(
            f"{path}: view {view.name!r} step {view.step} has {len(flat)} numbers, expected "
            f"{view.items} x {width}",
            hint="the view block is truncated or its component count is wrong",
        )
    table = flat.reshape(view.items, width)
    return table[:, 0].astype(np.int64), table[:, 1:]


def read_gmsh_mesh(path: Path, views: Sequence[_GmshView]) -> Any:
    """Read the mesh of an MSH file with meshio. The post-processing views are cut out first
    (into a temporary copy): meshio keeps one step of them at best and fails on several
    layouts Gmsh writes, and they are parsed here anyway."""
    if not views:
        return read_meshio(path, "gmsh")
    import os
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".msh", prefix="frontier-gmsh-")
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            pos = 0
            for v in sorted(views, key=lambda v: v.block_start):
                if v.block_start > pos:
                    src.seek(pos)
                    remaining = v.block_start - pos
                    while remaining > 0:
                        chunk = src.read(min(remaining, 1 << 24))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                pos = max(pos, v.block_end)
            src.seek(pos)
            while chunk := src.read(1 << 24):
                out.write(chunk)
        return read_meshio(Path(name), "gmsh")
    except ConnectorError as e:
        raise ConnectorError(e.message.replace(name, str(path)), hint=e.hint) from e
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def _gmsh_node_tags(path: Path, version: str, binary: bool, size: int) -> np.ndarray | None:
    """0-based node tag per meshio point (file order), or ``None`` when they cannot be read.
    Needed to place view values, which are keyed by node tag, on the right points."""
    try:
        from meshio.gmsh import _gmsh22, _gmsh41

        with open(path, "rb") as f:
            while True:
                text = f.readline()
                if not text:
                    return None
                if text.strip() == b"$Nodes":
                    break
            if version.startswith("4"):
                return np.asarray(_gmsh41._read_nodes(f, not binary, size)[1], dtype=np.int64)
            return np.asarray(_gmsh22._read_nodes(f, not binary)[1], dtype=np.int64) - 1
    except Exception:  # noqa: BLE001 - private meshio API: fall back to sequential tags
        return None


class GmshConnector(_ZonedConnector):
    id = "gmsh"
    name = "Gmsh MSH (2.2 / 4.1, ASCII and binary)"
    formats = ["gmsh-msh"]
    vendor = "Gmsh (C. Geuzaine, J.-F. Remacle; open format)"
    license = "Apache-2.0 (reader: meshio, MIT)"
    extensions = [".msh"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    export_help = (
        "Gmsh: File > Export > .msh (version 4.1 or 2.2; tick 'Save all elements' only if you "
        "have no physical groups). Post-processing views: Save 'Mesh-based' views into the "
        ".msh; parsed-view .pos files must be re-saved that way."
    )
    status_note = (
        "Physical groups become zones. $NodeData and $ElementData views are read, including "
        "several time steps of one view; element views assume the consecutive element tags Gmsh "
        "writes. $ElementNodeData and .pos parsed views are not read. MSH 4 files need every saved "
        "entity in a physical group. Views and binary files are validated on generated files only."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        head = _head(path, 64)
        if not head.startswith(b"$MeshFormat"):
            return None  # Fluent .msh, MSH1, anything else
        ext = match_extension(path, self.extensions)
        try:
            version = head.split(b"\n")[1].split()[0].decode()
        except (IndexError, UnicodeDecodeError):
            version = "?"
        conf = 0.97 if ext else 0.9
        return Detection(self.id, path, conf, f"gmsh-msh{version}", ["$MeshFormat header"])

    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        version, binary, size = _gmsh_header(path)
        views = scan_gmsh_views(path)
        try:
            mesh = read_gmsh_mesh(path, views)
        except ConnectorError as e:
            if version.startswith("4"):
                raise ConnectorError(
                    e.message,
                    hint="meshio reads MSH 4 files only when every saved entity is in a "
                    "physical group: export without 'Save all elements', or as MSH 2.2",
                ) from e
            raise
        zoned, pd, cd = mz.zoned_from_meshio(
            mesh, planar_as=mz.planar_option(options), prefix="physical"
        )
        ld = _Loaded(zoned, pd, cd, format=f"gmsh-msh{version}", solver="unknown")
        ld.metadata["generator"] = "Gmsh"
        if self._has_marker(path, b"\n$ElementNodeData"):
            ld.warnings.append("$ElementNodeData views are not read")
        if not views:
            return ld
        by_name: dict[tuple[str, str], list[_GmshView]] = {}
        for v in views:
            by_name.setdefault((v.kind, v.name), []).append(v)
        for group in by_name.values():
            group.sort(key=lambda v: (v.step, v.time))
        longest = max(len(g) for g in by_name.values())
        tags = _gmsh_node_tags(path, version, binary, size)
        n_points, n_cells = len(mesh.points), zoned.n_cells
        if tags is not None and len(tags) == n_points and int(tags.min()) >= 0:
            lookup = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
            lookup[tags] = np.arange(n_points)
        else:
            lookup = np.arange(n_points)

        def load(view: _GmshView) -> np.ndarray | None:
            ids, values = read_gmsh_view(path, view, binary)
            if view.kind == "NodeData":
                idx = ids - 1  # 0-based node tag
                ok = (idx >= 0) & (idx < len(lookup))
                target = np.where(ok, lookup[np.clip(idx, 0, len(lookup) - 1)], -1)
                n = n_points
            else:
                # element tags are sequential in file order in files Gmsh writes
                target = ids - 1
                n = n_cells
            good = (target >= 0) & (target < n)
            if not good.any():
                return None
            out = np.full((n, view.components), np.nan, dtype=np.float64)
            out[target[good]] = values[good]
            return out

        transient: dict[tuple[str, str], list[_GmshView]] = {}
        for key, group in by_name.items():
            target = pd if key[0] == "NodeData" else cd
            if longest > 1 and len(group) == longest:
                transient[key] = group
                continue
            if len(group) > 1:
                ld.warnings.append(
                    f"view {key[1]!r} has {len(group)} of {longest} steps; its last step is "
                    "read as a time-independent field"
                )
            arr = load(group[-1])
            if arr is None:
                ld.warnings.append(f"view {key[1]!r}: tags do not match the mesh; skipped")
            else:
                target[key[1]] = arr
        ld.domain = mz.guess_domain([k[1] for k in by_name])
        if not transient:
            return ld
        ref = next(iter(transient.values()))
        ld.times = [v.time for v in ref]
        if any(b <= a for a, b in zip(ld.times, ld.times[1:], strict=False)):
            ld.times = [float(v.step) for v in ref]
            ld.warnings.append("view times are not increasing; step numbers are used as time")

        def load_step(k: int) -> tuple[dict[str, Any], dict[str, Any]]:
            spd: dict[str, Any] = {}
            scd: dict[str, Any] = {}
            for (kind, name), group in transient.items():
                arr = load(group[k])
                if arr is not None:
                    (spd if kind == "NodeData" else scd)[name] = arr
            return spd, scd

        ld.load_step = load_step
        ld.transient_fields = [
            FieldInfo(
                name=name,
                location="point" if kind == "NodeData" else "cell",
                components=group[0].components,
            )
            for (kind, name), group in transient.items()
        ]
        return ld

    @staticmethod
    def _has_marker(path: Path, marker: bytes) -> bool:
        if path.stat().st_size == 0:
            return False
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(marker) >= 0


# =============================================================================================
# MED
# =============================================================================================

#: MED geometry name -> (meshio type, permutation MED node order -> VTK node order)
_MED_TYPES: dict[str, tuple[str, list[int] | None]] = {
    "PO1": ("vertex", None),
    "SE2": ("line", None),
    "SE3": ("line3", None),
    "TR3": ("triangle", None),
    "TR6": ("triangle6", None),
    "QU4": ("quad", None),
    "QU8": ("quad8", None),
    "QU9": ("quad9", None),
    # MED numbers the base face of 3-D cells with the opposite orientation to VTK
    "TE4": ("tetra", [0, 2, 1, 3]),
    "T10": ("tetra10", [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]),
    "PY5": ("pyramid", [0, 3, 2, 1, 4]),
    "P13": ("pyramid13", [0, 3, 2, 1, 4]),
    "PE6": ("wedge", [0, 2, 1, 3, 5, 4]),
    "P15": ("wedge15", [0, 2, 1, 3, 5, 4]),
    "P18": ("wedge18", [0, 2, 1, 3, 5, 4]),
    "HE8": ("hexahedron", [0, 3, 2, 1, 4, 7, 6, 5]),
    "H20": (
        "hexahedron20",
        [0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17],
    ),
}
#: element types whose mid-side nodes are dropped by the zone bridge anyway
_MED_CORNERS_ONLY = {"P13": 5, "P15": 6, "P18": 6}
_NO_PROFILE = "MED_NO_PROFILE_INTERNAL"


def _med_group_names(family: Any) -> list[str]:
    if "GRO" not in family or "NOM" not in family["GRO"]:
        return []
    raw = np.asarray(family["GRO"]["NOM"][()])
    return [bytes(row.astype(np.uint8)).split(b"\x00")[0].decode("latin-1").strip() for row in raw]


class _MedFile:
    """Structure of a MED 3.x/4.x file (HDF5): one mesh, its families/groups, the fields."""

    def __init__(self, path: Path, mesh_name: str | None = None) -> None:
        import h5py  # noqa: F401 - availability check; the file is opened by h5_open

        self.path = path
        try:
            self.h5: Any = h5_open(path)
        except OSError as e:
            raise ConnectorError(
                f"{path} is not a readable HDF5 file: {e}",
                hint="MED 2.1 (non-HDF5) files must be converted with Salome's medimport",
            ) from e
        try:
            self._parse(mesh_name)
        except ConnectorError:
            self.h5.close()
            raise
        except (KeyError, ValueError, IndexError, TypeError) as e:
            self.h5.close()
            raise ConnectorError(
                f"{path}: unexpected MED layout ({type(e).__name__}: {e})",
                hint="the file is corrupt or uses a MED feature this reader does not know",
            ) from e

    def _parse(self, mesh_name: str | None) -> None:
        f = self.h5
        self.warnings: list[str] = []
        if "ENS_MAA" not in f or len(f["ENS_MAA"]) == 0:
            raise UnsupportedSourceError(
                f"{self.path} has no /ENS_MAA mesh group", hint="not a MED mesh file"
            )
        names = list(f["ENS_MAA"])
        if mesh_name is not None and mesh_name not in names:
            raise ConnectorError(
                f"mesh {mesh_name!r} not found in {self.path}",
                hint=f"available meshes: {', '.join(names)}",
            )
        self.mesh_name = mesh_name or names[0]
        if len(names) > 1 and mesh_name is None:
            self.warnings.append(
                f"the file holds {len(names)} meshes ({', '.join(names)}); reading "
                f'{self.mesh_name!r} (pick another with options={{"mesh": name}})'
            )
        top = f["ENS_MAA"][self.mesh_name]
        if int(top.attrs.get("TYP", 0)) != 0:
            raise ConnectorError(
                f"mesh {self.mesh_name!r} is a structured MED mesh",
                hint="convert it to an unstructured mesh in Salome (MEDCoupling "
                "buildUnstructured) and export again",
            )
        self.description = _clean(top.attrs.get("DES", ""))
        dim = int(top.attrs.get("ESP", top.attrs.get("DIM", 3)))
        mesh = top
        if "NOE" not in mesh:
            steps = sorted(mesh)
            if not steps:
                raise ConnectorError(f"mesh {self.mesh_name!r} is empty", hint="no nodes stored")
            if len(steps) > 1:
                self.warnings.append(
                    f"the mesh has {len(steps)} computation steps; the first one is read"
                )
            mesh = mesh[steps[0]]
        coo = mesh["NOE"]["COO"]
        n = int(coo.attrs["NBR"])
        self.points = np.asarray(coo[()], dtype=np.float64).reshape((n, dim), order="F")
        unit = _clean(top.attrs.get("UNI", ""))[:16].strip()
        self.length_unit = "" if unit.upper() in ("", "INCONNUE", "NO_UNIT") else unit
        self.blocks: list[tuple[str, np.ndarray]] = []
        self.med_types: list[str] = []
        fam: list[np.ndarray] = []
        for med_type, grp in mesh["MAI"].items() if "MAI" in mesh else []:
            if med_type not in _MED_TYPES or "NOD" not in grp:
                self.warnings.append(f"MED cell type {med_type} is not supported and was skipped")
                continue
            nod = grp["NOD"]
            k = int(nod.attrs["NBR"])
            conn = np.asarray(nod[()], dtype=np.int64).reshape((k, -1), order="F") - 1
            cell_type, perm = _MED_TYPES[med_type]
            if med_type in _MED_CORNERS_ONLY:
                conn = conn[:, : _MED_CORNERS_ONLY[med_type]]
                cell_type = re.sub(r"\d+$", "", cell_type)
                self.warnings.append(f"{med_type} elements were reduced to their corner nodes")
            if perm is not None:
                conn = conn[:, perm]
            self.blocks.append((cell_type, conn))
            self.med_types.append(med_type)
            fam.append(
                np.asarray(grp["FAM"][()], dtype=np.int64)
                if "FAM" in grp
                else np.zeros(k, dtype=np.int64)
            )
        self.cell_family = np.concatenate(fam) if fam else np.zeros((0,), dtype=np.int64)
        self.node_family = (
            np.asarray(mesh["NOE"]["FAM"][()], dtype=np.int64) if "FAM" in mesh["NOE"] else None
        )
        self.starts = np.concatenate([[0], np.cumsum([len(c) for _, c in self.blocks])])
        fas = None
        if "FAS" in f and self.mesh_name in f["FAS"]:
            fas = f["FAS"][self.mesh_name]
        self.groups: list[mz.Group] = []
        self.point_sets: dict[str, np.ndarray] = {}
        if fas is not None:
            self._families(fas)
        self.fields = self._field_index()

    def _families(self, fas: Any) -> None:
        cell_groups: dict[str, list[int]] = {}
        node_groups: dict[str, list[int]] = {}
        for section, target in (("ELEME", cell_groups), ("NOEUD", node_groups)):
            if section not in fas:
                continue
            for family in fas[section].values():
                num = int(family.attrs.get("NUM", 0))
                for name in _med_group_names(family):
                    if name:
                        target.setdefault(name, []).append(num)
        for name, nums in cell_groups.items():
            idx = np.nonzero(np.isin(self.cell_family, nums))[0]
            if len(idx):
                self.groups.append(mz.Group(name, idx))
        if self.node_family is not None:
            for name, nums in node_groups.items():
                idx = np.nonzero(np.isin(self.node_family, nums))[0]
                if len(idx):
                    self.point_sets[name] = idx

    def _field_index(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if "CHA" not in self.h5:
            return out
        for name, grp in self.h5["CHA"].items():
            mesh = _clean(grp.attrs.get("MAI", self.mesh_name))
            if mesh and mesh != self.mesh_name:
                continue
            nco = int(grp.attrs.get("NCO", 1))
            nom = _clean(grp.attrs.get("NOM", "")).ljust(16 * nco)
            uni = _clean(grp.attrs.get("UNI", "")).ljust(16 * nco)
            raw_nom = grp.attrs.get("NOM", b"")
            raw_uni = grp.attrs.get("UNI", b"")
            if isinstance(raw_nom, bytes | np.bytes_):
                nom = bytes(raw_nom).decode("latin-1").ljust(16 * nco)
            if isinstance(raw_uni, bytes | np.bytes_):
                uni = bytes(raw_uni).decode("latin-1").ljust(16 * nco)
            comps = [nom[16 * i : 16 * i + 16].strip() for i in range(nco)]
            units = {uni[16 * i : 16 * i + 16].strip() for i in range(nco)} - {"", "INCONNUE"}
            steps = sorted(
                grp,
                key=lambda k: (int(grp[k].attrs.get("NDT", 0)), int(grp[k].attrs.get("NOR", 0))),
            )
            if not steps:
                continue
            supports = sorted({s for k in steps for s in grp[k]})
            out[name] = {
                "components": comps if nco > 1 and all(comps) else None,
                "n_components": nco,
                "units": units.pop() if len(units) == 1 else "",
                "steps": steps,
                "times": [float(grp[k].attrs.get("PDT", 0.0)) for k in steps],
                "nodal": "NOE" in supports,
                "cell": any(s.startswith(("MAI.", "NOE.")) for s in supports),
            }
        return out

    def _values(self, support: Any, n_entities: int, nco: int) -> np.ndarray:
        """``(n_entities, nco)`` float array of one support (mean over Gauss / element nodes,
        NaN outside a profile)."""
        profile = _clean(support.attrs.get("PFL", _NO_PROFILE)) or _NO_PROFILE
        data = support[profile] if profile in support else next(iter(support.values()))
        nga = max(int(data.attrs.get("NGA", 1)), 1)
        raw = np.asarray(data["CO"][()], dtype=np.float64)
        n_stored = raw.size // (nco * nga)
        # MED stores component by component; within a component entity by entity
        vals = raw[: n_stored * nga * nco].reshape(nco, n_stored, nga).mean(axis=2).T
        if profile == _NO_PROFILE:
            if n_stored != n_entities:
                raise ConnectorError(
                    f"{self.path}: field support holds {n_stored} entities, mesh has {n_entities}",
                    hint="the field belongs to another mesh or the file is inconsistent",
                )
            return vals
        prof = self.h5["PROFILS"][profile] if "PROFILS" in self.h5 else None
        if prof is None or "PFL" not in prof:
            raise ConnectorError(
                f"{self.path}: profile {profile!r} is referenced but not stored",
                hint="re-export the field on the whole mesh",
            )
        index = np.asarray(prof["PFL"][()], dtype=np.int64) - 1
        out = np.full((n_entities, nco), np.nan)
        out[index[: len(vals)]] = vals
        return out

    def read_field(self, name: str, step: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(nodal values, whole-mesh cell values)`` of one field at one stored step."""
        info = self.fields[name]
        grp = self.h5["CHA"][name][info["steps"][step]]
        nco = info["n_components"]
        nodal = cell = None
        try:
            for support_name, support in grp.items():
                if support_name == "NOE":
                    nodal = self._values(support, len(self.points), nco)
                    continue
                med_type = support_name.partition(".")[2]
                if med_type not in self.med_types:
                    continue
                b = self.med_types.index(med_type)
                lo, hi = int(self.starts[b]), int(self.starts[b + 1])
                if cell is None:
                    cell = np.full((int(self.starts[-1]), nco), np.nan)
                cell[lo:hi] = self._values(support, hi - lo, nco)
        except (KeyError, ValueError, IndexError) as e:
            raise ConnectorError(
                f"{self.path}: field {name!r} step {step} is malformed ({e})",
                hint="the MED field group is truncated or uses an unsupported layout",
            ) from e
        return nodal, cell

    def close(self) -> None:
        self.h5.close()


class MedConnector(_ZonedConnector):
    id = "med"
    name = "MED (Salome, Code_Aster, Code_Saturne)"
    formats = ["med"]
    vendor = "EDF / CEA (open format, Salome platform)"
    license = "Apache-2.0 (reader: own, on h5py BSD-3)"
    extensions = [".med", ".rmed", ".mmed"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    export_help = (
        "Code_Aster: IMPR_RESU(FORMAT='MED', RESU=_F(RESULTAT=...)); Salome: File > Export > "
        "MED; Code_Saturne: writer format 'MED'. MED 2.1 files need Salome's medimport."
    )
    status_note = (
        "Groups (families) become zones; node groups are point-cloud zones. Reads fields on nodes "
        "and cells for every stored time step. Gauss-point (ELGA) and element-node (ELNO) fields "
        "are averaged per cell; their storage order for more than one cell is not validated on a "
        "real file. Fields on a profile are NaN outside it. Structured MED meshes and polyhedra "
        "are not read. Several time steps are validated on generated files only."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        if not _head(path, 8).startswith(_HDF5_MAGIC):
            return None
        try:
            import h5py

            with h5py.File(path, "r") as f:
                has_mesh = "ENS_MAA" in f
                has_info = "INFOS_GENERALES" in f
                version = ""
                if has_info:
                    a: Any = f["INFOS_GENERALES"].attrs
                    version = ".".join(str(int(a[k])) for k in ("MAJ", "MIN", "REL") if k in a)
        except Exception:  # noqa: BLE001 - detection must never raise
            return None
        if not has_mesh:
            return None
        conf = 0.97 if ext and has_info else 0.9 if has_info else 0.75
        return Detection(self.id, path, conf, "med", [f"HDF5 with /ENS_MAA (MED {version})"])

    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        med = _MedFile(path, options.get("mesh"))
        try:
            zoned = mz.ZonedMesh(
                med.points,
                med.blocks,
                groups=med.groups,
                point_sets=med.point_sets,
                planar_as=mz.planar_option(options),
                vtk_order=True,  # the MED -> VTK permutation was applied while reading
            )
        except ConnectorError:
            med.close()
            raise
        aster = "ASTER" in med.description.upper() or any("____" in n for n in med.fields)
        ld = _Loaded(
            zoned,
            format="med",
            solver="Code_Aster" if aster else "Salome MED",
            warnings=list(med.warnings),
            close=med.close,
        )
        try:
            a = med.h5["INFOS_GENERALES"].attrs
            ld.metadata["med_version"] = ".".join(str(int(a[k])) for k in ("MAJ", "MIN", "REL"))
        except (KeyError, ValueError):
            pass
        ld.metadata["mesh"] = med.mesh_name
        if med.description:
            ld.metadata["description"] = med.description
        if med.length_unit:
            ld.units["length"] = med.length_unit
        n_steps = max((len(i["steps"]) for i in med.fields.values()), default=0)
        axis: list[float] = []
        moving: list[str] = []
        for name, info in med.fields.items():
            ld.field_units[name] = info["units"]
            if info["components"]:
                ld.component_names[name] = info["components"]
            if n_steps > 1 and len(info["steps"]) == n_steps:
                if not axis:
                    axis = info["times"]
                moving.append(name)
            elif len(info["steps"]) > 1:
                ld.warnings.append(
                    f"field {name!r} has {len(info['steps'])} of {n_steps} steps; its last "
                    "step is read as a time-independent field"
                )

        def collect(
            names: Sequence[str], step: int | None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            pd: dict[str, Any] = {}
            cd: dict[str, Any] = {}
            for name in names:
                k = len(med.fields[name]["steps"]) - 1 if step is None else step
                nodal, cell = med.read_field(name, k)
                if nodal is not None:
                    pd[name] = nodal
                if cell is not None:
                    key = name if nodal is None else f"{name}__cell"
                    cd[key] = cell
                    ld.field_units[key] = ld.field_units.get(name, "")
                    if name in ld.component_names:
                        ld.component_names[key] = ld.component_names[name]
            return pd, cd

        steady = [n for n in med.fields if n not in moving]
        ld.point_data, ld.cell_data = collect(steady, None)
        ld.domain = mz.guess_domain(list(med.fields))
        if moving:
            if any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
                ld.warnings.append("field times (PDT) do not increase; step numbers are used")
                axis = [float(i) for i in range(n_steps)]
            ld.times = axis
            ld.load_step = lambda k: collect(moving, k)
            for name in moving:
                info = med.fields[name]
                for location, present in (("point", info["nodal"]), ("cell", info["cell"])):
                    if present:
                        both = location == "cell" and info["nodal"]
                        ld.transient_fields.append(
                            FieldInfo(
                                name=f"{name}__cell" if both else name,
                                location=location,
                                components=info["n_components"],
                                units=info["units"],
                                component_names=info["components"],
                            )
                        )
        if "PROFILS" in med.h5 and len(med.h5["PROFILS"]) > 0:
            ld.warnings.append("fields stored on a profile are NaN outside their support")
        return ld


# =============================================================================================
# SU2
# =============================================================================================

#: SU2 element code (= VTK type) -> (meshio type, node count)
_SU2_TYPES: dict[int, tuple[str, int]] = {
    3: ("line", 2),
    5: ("triangle", 3),
    9: ("quad", 4),
    10: ("tetra", 4),
    12: ("hexahedron", 8),
    13: ("wedge", 6),
    14: ("pyramid", 5),
}

#: SU2 config marker option -> normalised boundary-condition type
_SU2_BC: dict[str, str] = {
    "MARKER_EULER": "wall",
    "MARKER_HEATFLUX": "wall",
    "MARKER_ISOTHERMAL": "wall",
    "MARKER_HEATTRANSFER": "wall",
    "MARKER_WALL_FUNCTIONS": "wall",
    "MARKER_FAR": "farfield",
    "MARKER_SYM": "symmetry",
    "MARKER_INLET": "inlet",
    "MARKER_SUPERSONIC_INLET": "inlet",
    "MARKER_OUTLET": "outlet",
    "MARKER_SUPERSONIC_OUTLET": "outlet",
    "MARKER_PERIODIC": "periodic",
    "MARKER_PRESSURE": "pressure",
    "MARKER_RIEMANN": "riemann",
    "MARKER_GILES": "giles",
    "MARKER_CLAMPED": "clamped",
    "MARKER_LOAD": "load",
    "MARKER_INTERNAL": "internal",
    "MARKER_FLUID_INTERFACE": "interface",
    "MARKER_ZONE_INTERFACE": "interface",
    "MARKER_CHT_INTERFACE": "interface",
}
_SU2_PARAMETERS = (
    "MACH_NUMBER",
    "AOA",
    "SIDESLIP_ANGLE",
    "REYNOLDS_NUMBER",
    "REYNOLDS_LENGTH",
    "FREESTREAM_TEMPERATURE",
    "FREESTREAM_PRESSURE",
    "FREESTREAM_DENSITY",
    "REF_AREA",
    "REF_LENGTH",
    "TIME_STEP",
)
_SU2_PHYSICS = {
    "SOLVER": "solver",
    "PHYSICAL_PROBLEM": "solver",
    "KIND_TURB_MODEL": "turbulence_model",
    "KIND_TRANS_MODEL": "transition_model",
    "MATH_PROBLEM": "math_problem",
    "TIME_MARCHING": "time_marching",
    "UNSTEADY_SIMULATION": "time_marching",
    "FLUID_MODEL": "fluid_model",
    "VISCOSITY_MODEL": "viscosity_model",
}


def parse_su2(path: Path) -> dict[str, Any]:
    """Parse a native SU2 mesh. Returns ``points``, ``blocks`` (meshio type, connectivity in
    VTK node order), ``groups`` (zones of a multi-zone mesh and markers) and ``markers``."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
    except OSError as e:
        raise ConnectorError(f"cannot read {path}: {e}", hint="check the path") from e

    def fail(i: int, what: str, hint: str) -> ConnectorError:
        return ConnectorError(f"{path}: line {i + 1}: {what}", hint=hint)

    def table(start: int, count: int, what: str) -> list[str]:
        rows = lines[start : start + count]
        if len(rows) < count or any("=" in r or not r.strip() for r in rows[:1] + rows[-1:]):
            raise fail(
                start,
                f"expected {count} {what} rows",
                "the file is truncated or the count in the header is wrong",
            )
        return rows

    def elements(start: int, count: int, what: str) -> list[tuple[int, np.ndarray]]:
        rows = table(start, count, what)
        try:
            codes = np.fromiter(
                (int(r.split(None, 1)[0]) for r in rows), dtype=np.int64, count=count
            )
        except (ValueError, IndexError) as e:
            raise fail(start, f"malformed {what} row", "each row is '<type> <node ids...>'") from e
        out = []
        for code in np.unique(codes):
            if int(code) not in _SU2_TYPES:
                raise fail(
                    start, f"unknown SU2 element type {code}", "supported: 3 5 9 10 12 13 14"
                )
            sel = np.nonzero(codes == code)[0]
            n = _SU2_TYPES[int(code)][1]
            text = " ".join(rows[i] for i in sel)
            try:
                flat = np.array(text.split(), dtype=np.int64)
                conn = flat.reshape(len(sel), -1)[:, 1 : 1 + n]
            except ValueError as e:
                first = start + int(sel[0])
                raise fail(first, f"{what} rows of type {code} have uneven length", "") from e
            if conn.shape[1] != n:
                raise fail(start + int(sel[0]), f"{what} row has too few node ids", "")
            out.append((int(code), conn))
        return out

    dim = 0
    points_parts: list[np.ndarray] = []
    blocks: list[tuple[str, np.ndarray]] = []
    groups: list[mz.Group] = []
    markers: list[str] = []
    n_zone = 1
    zone = 0
    offset = 0  # point offset of the current zone
    n_cells = 0
    marker = ""
    i = 0
    seen_points = False
    while i < len(lines):
        text = lines[i].split("%", 1)[0].strip()
        i += 1
        if not text or "=" not in text:
            continue
        key, _, value = text.partition("=")
        key, value = key.strip().upper(), value.strip()
        try:
            if key == "NZONE":
                n_zone = int(value)
            elif key == "IZONE":
                zone = int(value)
                offset = sum(len(p) for p in points_parts)
            elif key == "NDIME":
                dim = int(value)
                if dim not in (2, 3):
                    raise fail(i - 1, f"NDIME= {dim}", "SU2 meshes are 2-D or 3-D")
            elif key == "NPOIN":
                count = int(value.split()[0])
                if dim == 0:
                    raise fail(i - 1, "NPOIN before NDIME", "NDIME= must come first")
                rows = table(i, count, "point")
                try:
                    flat = np.array(" ".join(rows).split(), dtype=np.float64)
                except ValueError as e:
                    raise fail(i, f"point rows are not numeric ({e})", "expected 'x y [z]'") from e
                if len(flat) % count or len(flat) // count < dim:
                    raise fail(i, "point rows have uneven length", "expected 'x y [z] [index]'")
                points_parts.append(flat.reshape(count, -1)[:, :dim])
                seen_points = True
                i += count
            elif key in ("NELEM", "MARKER_ELEMS"):
                count = int(value.split()[0])
                start_cell = n_cells
                for code, conn in elements(i, count, "element") if count else []:
                    blocks.append((_SU2_TYPES[code][0], conn + offset))
                    n_cells += len(conn)
                i += count
                idx = np.arange(start_cell, n_cells)
                prefix = f"zone{zone}/" if n_zone > 1 else ""
                if key == "NELEM" and n_zone > 1:
                    groups.append(mz.Group(f"zone{zone}", idx))
                elif key == "MARKER_ELEMS" and len(idx):
                    groups.append(mz.Group(prefix + marker, idx))
                    markers.append(prefix + marker)
            elif key == "MARKER_TAG":
                marker = value
        except ValueError as e:
            raise fail(i - 1, f"cannot parse {text[:60]!r}", "expected 'KEY= <integer>'") from e
    if dim == 0 or not seen_points or not blocks:
        raise ConnectorError(
            f"{path}: not a complete SU2 mesh (NDIME/NPOIN/NELEM missing)",
            hint="a native .su2 mesh needs NDIME=, NELEM= and NPOIN= sections",
        )
    return {
        "dim": dim,
        "points": np.concatenate(points_parts),
        "blocks": blocks,
        "groups": groups,
        "markers": markers,
        "n_zone": n_zone,
    }


def parse_su2_config(path: Path) -> dict[str, str]:
    """``KEY -> value`` of an SU2 ``.cfg`` (comments stripped, continuation lines joined)."""
    out: dict[str, str] = {}
    pending = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.split("%", 1)[0].strip()
        if not text:
            continue
        text = pending + text
        if text.endswith("\\"):
            pending = text[:-1] + " "
            continue
        pending = ""
        if "=" in text:
            key, _, value = text.partition("=")
            out[key.strip().upper()] = value.strip()
    return out


def su2_boundary_conditions(config: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Per marker: ``{"type": ..., "su2_option": "MARKER_...", "values": [...]}``. A marker
    option lists marker names, each optionally followed by its numeric values."""
    out: dict[str, dict[str, Any]] = {}
    for key, value in config.items():
        if key not in _SU2_BC:
            continue
        current: dict[str, Any] | None = None
        for token in re.split(r"[\s,()]+", value):
            if not token or token.upper() == "NONE":
                continue
            try:
                number = float(token)
            except ValueError:
                current = {"type": _SU2_BC[key], "su2_option": key, "values": []}
                out[token] = current
                continue
            if current is not None:
                current["values"].append(number)
    return out


class Su2Connector(_ZonedConnector):
    id = "su2"
    name = "SU2 native mesh (.su2) with its .cfg"
    formats = ["su2-mesh"]
    vendor = "SU2 Foundation (open source)"
    license = "Apache-2.0 (reader: own)"
    extensions = [".su2"]
    maturity = "verified-public-sample"
    temporal = ["steady"]
    kinds = ["mesh", "surface"]
    export_help = (
        "The .su2 file is the mesh only. SU2 results are written as VTU/PVD (OUTPUT_FILES= "
        "PARAVIEW), Tecplot, CSV (surface_flow.csv, restart_flow.csv) or CGNS: read those "
        "with the vtk, tecplot, tabular or cgns connectors."
    )
    status_note = (
        "Reads the mesh, markers as boundary zones and multi-zone meshes; solver, turbulence "
        "model, free-stream parameters and the boundary condition of each marker come from a .cfg "
        "next to the mesh. No solution fields."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        if ext is None or not path.is_file():
            return None
        head = _head(path, 8192)
        if b"NDIME" in head and b"=" in head:
            return Detection(self.id, path, 0.95, "su2-mesh", ["NDIME= keyword"])
        if b"NELEM" in head or b"NPOIN" in head or b"NZONE" in head:
            return Detection(self.id, path, 0.8, "su2-mesh", ["SU2 mesh keywords"])
        return None

    @staticmethod
    def find_config(path: Path) -> Path | None:
        """The ``.cfg`` that belongs to a mesh: same stem, or the only one in the directory
        whose ``MESH_FILENAME`` names the mesh."""
        same = path.with_suffix(".cfg")
        if same.is_file():
            return same
        try:
            candidates = sorted(p for p in path.parent.glob("*.cfg") if p.is_file())[:50]
        except OSError:
            return None
        matches = []
        for cfg in candidates:
            try:
                name = parse_su2_config(cfg).get("MESH_FILENAME", "")
            except OSError:
                continue
            if Path(name).name == path.name:
                matches.append(cfg)
        return matches[0] if len(matches) == 1 else None

    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        data = parse_su2(path)
        cfg_path = options.get("config")
        cfg_file = Path(cfg_path) if cfg_path else self.find_config(path)
        config: dict[str, str] = {}
        if cfg_file is not None:
            if not cfg_file.is_file():
                raise ConnectorError(
                    f"SU2 config {cfg_file} does not exist", hint="check options['config']"
                )
            config = parse_su2_config(cfg_file)
        bcs = su2_boundary_conditions(config)
        for g in data["groups"]:
            bc = bcs.get(g.name.split("/")[-1])
            if bc is not None:
                g.bc_type = bc["type"]
        zoned = mz.ZonedMesh(
            data["points"],
            data["blocks"],
            groups=data["groups"],
            planar_as=mz.planar_option(options),
            vtk_order=True,  # SU2 uses VTK's element codes and node order
        )
        ld = _Loaded(zoned, format="su2-mesh", solver="SU2", domain="cfd")
        ld.metadata["markers"] = data["markers"]
        ld.metadata["dimension"] = data["dim"]
        if data["n_zone"] > 1:
            ld.metadata["n_zone"] = data["n_zone"]
        if config:
            ld.metadata["config"] = str(cfg_file)
            ld.boundary_conditions = {k: v for k, v in bcs.items()}
            for key, label in _SU2_PHYSICS.items():
                if key in config and label not in ld.physics:
                    ld.physics[label] = config[key]
            solver = str(ld.physics.get("solver", "")).upper()
            if solver:
                ld.physics["compressible"] = not solver.startswith("INC")
            for key in _SU2_PARAMETERS:
                if key in config:
                    try:
                        ld.parameters[key.lower()] = float(config[key])
                    except ValueError:
                        pass
            missing = [m for m in data["markers"] if m.split("/")[-1] not in bcs]
            if missing:
                ld.warnings.append(
                    "no boundary condition in the .cfg for markers: " + ", ".join(missing[:10])
                )
        else:
            ld.warnings.append(
                "no .cfg found next to the mesh: boundary-condition types are unknown "
                '(pass options={"config": path})'
            )
        return ld


# =============================================================================================
# generic meshio formats
# =============================================================================================

#: extension -> (meshio format, tool label, domain)
_GENERIC: dict[str, tuple[str, str, str]] = {
    ".mesh": ("medit", "Medit (INRIA: MMG, FreeFEM, Gmsh export)", "other"),
    ".meshb": ("medit", "Medit (INRIA: MMG, FreeFEM, Gmsh export)", "other"),
    ".vol": ("netgen", "Netgen/NGSolve", "other"),
    ".vol.gz": ("netgen", "Netgen/NGSolve", "other"),
    ".node": ("tetgen", "TetGen", "other"),
    ".ele": ("tetgen", "TetGen", "other"),
    ".ugrid": ("ugrid", "AFLR3 / FUN3D UGRID", "cfd"),
    ".h5m": ("h5m", "MOAB", "other"),
    ".f3grid": ("flac3d", "Itasca FLAC3D", "geophysical"),
    ".post": ("permas", "PERMAS", "structural"),
    ".post.gz": ("permas", "PERMAS", "structural"),
    ".dato": ("permas", "PERMAS", "structural"),
    ".dato.gz": ("permas", "PERMAS", "structural"),
    ".xml": ("dolfin-xml", "FEniCS (legacy DOLFIN XML)", "other"),
    ".avs": ("avsucd", "AVS-UCD", "other"),
    ".mdpa": ("mdpa", "Kratos Multiphysics", "other"),
    ".hmf": ("hmf", "HMF (meshio)", "other"),
}


def _flac3d_set_name(name: str) -> str:
    """meshio names FLAC3D groups ``zone:<name>:<slot>`` / ``face:<name>:<slot>``."""
    parts = [p.strip().strip('"') for p in name.split(":")]
    if len(parts) == 3:
        # slot 1 is FLAC3D's default slot
        return parts[1] if parts[2] in ("Default", "", "1") else f"{parts[1]}_{parts[2]}"
    return name.replace('"', "")


class MeshioConnector(_ZonedConnector):
    id = "meshio"
    name = "Other meshio formats (Medit, Netgen, TetGen, UGRID, MOAB, FLAC3D, PERMAS, ...)"
    formats = [
        "medit", "netgen", "tetgen", "ugrid", "h5m", "flac3d", "permas", "dolfin-xml", "avsucd",
        "mdpa", "hmf",
    ]  # fmt: skip
    vendor = "various open formats (via meshio)"
    license = "Apache-2.0 (reader: meshio, MIT)"
    extensions = list(_GENERIC)
    maturity = "verified-public-sample"
    temporal = ["steady"]
    extension_confidence = 0.55
    export_help = (
        "These are mesh formats: results usually live elsewhere (FUN3D: Tecplot/VTK, "
        "NGSolve: VTK, FEniCS: XDMF/PVD). Any other meshio format can be converted with "
        "`meshio convert in.xxx out.vtu` and read by the vtk connector."
    )
    status_note = (
        "Reads Medit .mesh / .meshb, Netgen .vol, TetGen .node / .ele, UGRID and FLAC3D .f3grid. "
        "Region and material ids become zones. MOAB .h5m, DOLFIN XML, AVS-UCD, PERMAS, Kratos "
        ".mdpa and HMF are validated on meshio-written files only."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        if ext is None or not path.is_file():
            return None
        fmt = _GENERIC[ext][0]
        head = _head(path, 1024)
        text = head.lstrip()
        conf = self.extension_confidence
        notes = [f"extension {ext}"]
        if ext == ".xml":
            if b"<dolfin" not in head:
                return None
            conf, notes = 0.9, ["<dolfin> root element"]
        elif ext == ".mesh":
            if b"MeshVersionFormatted" not in head:
                return None
            conf, notes = 0.9, ["MeshVersionFormatted keyword"]
        elif ext == ".meshb":
            if head[:4] not in (b"\x01\x00\x00\x00", b"\x00\x00\x00\x01"):
                return None
            conf, notes = 0.8, ["Medit binary byte-order mark"]
        elif ext == ".vol":
            if b"mesh3d" not in head and b"dimension" not in head:
                return None
            conf, notes = 0.85, ["Netgen 'mesh3d' keyword"]
        elif ext in (".node", ".ele"):
            other = path.with_suffix(".ele" if ext == ".node" else ".node")
            if not other.is_file() or head.startswith(_HDF5_MAGIC):
                return None
            conf, notes = 0.7, ["TetGen .node/.ele pair"]
        elif ext in (".h5m", ".hmf"):
            if not head.startswith(_HDF5_MAGIC):
                return None
            conf, notes = 0.8, ["HDF5 container"]
        elif ext == ".f3grid":
            if text.startswith(b"*") or b"FLAC3D" in head.upper():
                conf, notes = 0.85, ["FLAC3D grid header"]
        elif ext == ".mdpa":
            if b"Begin" not in head:
                return None
            conf, notes = 0.85, ["Kratos 'Begin' blocks"]
        elif ext.endswith(".gz") and not head.startswith(b"\x1f\x8b"):
            return None
        return Detection(self.id, path, conf, fmt, notes)

    def _load(self, path: Path, options: Mapping[str, Any]) -> _Loaded:
        ext = match_extension(path, self.extensions)
        fmt = str(options.get("format") or (_GENERIC[ext][0] if ext else ""))
        if not fmt:
            raise UnsupportedSourceError(
                f"{path} has no extension the meshio connector knows",
                hint='pass options={"format": "<meshio format name>"} '
                f"(known here: {', '.join(sorted(set(self.formats)))})",
            )
        source = path
        if ext == ".ele":
            source = path.with_suffix(".node")
        mesh = read_meshio(source, fmt)
        zoned, pd, cd = mz.zoned_from_meshio(
            mesh,
            planar_as=mz.planar_option(options),
            prefix="region",
            set_name=_flac3d_set_name if fmt == "flac3d" else None,
            global_sets=fmt == "flac3d",
        )
        label, domain = (_GENERIC[ext][1], _GENERIC[ext][2]) if ext else (fmt, "other")
        ld = _Loaded(zoned, pd, cd, format=fmt, solver="unknown", domain=domain)
        ld.metadata["generator"] = label
        return ld
