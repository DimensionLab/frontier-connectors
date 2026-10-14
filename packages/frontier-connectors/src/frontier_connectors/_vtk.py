# SPDX-License-Identifier: Apache-2.0
"""VTK data-object -> ``RawSample`` conversion shared by every connector that reads through
VTK (VTK XML, EnSight, OpenFOAM, Fluent, CGNS cross-checks, Exodus, LS-DYNA, ...).

Only ``vtkmodules`` is used (no pyvista) so the package stays light.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.model import FieldData, FieldInfo, RawSample, ZoneInfo, as_field
from frontier_connectors.select import wants

#: point/cell arrays VTK adds for bookkeeping; never exported as fields
_INTERNAL_ARRAYS = (
    "vtkGhostType",
    "vtkOriginalPointIds",
    "vtkOriginalCellIds",
    "vtkValidPointMask",
)


def _np(arr: Any) -> np.ndarray:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(arr)


def grid_dims(ds: Any) -> tuple[int, int, int]:
    """Point dimensions ``(nx, ny, nz)`` of a structured VTK dataset."""
    d = [0, 0, 0]
    ds.GetDimensions(d)
    return int(d[0]), int(d[1]), int(d[2])


def cell_types(ds: Any) -> np.ndarray:
    """VTK cell type id per cell of an unstructured grid (uint8), across VTK 9.3 .. 9.7 APIs."""
    import warnings

    n = int(ds.GetNumberOfCells())
    if n == 0:
        return np.zeros((0,), dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        arr = ds.GetCellTypesArray()
    try:
        return np.array(_np(arr), dtype=np.uint8)
    except Exception:  # noqa: BLE001 - implicit (constant) arrays have no raw buffer
        return np.fromiter((ds.GetCellType(i) for i in range(n)), dtype=np.uint8, count=n)


def iter_leaves(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(name, dataset)`` for every non-empty leaf of a composite VTK object.

    Names come from the block metadata (``vtkCompositeDataSet.NAME``) joined with ``/``;
    unnamed blocks are called ``block<i>``. A plain dataset yields ``(prefix or "main", obj)``.
    """
    from vtkmodules.vtkCommonDataModel import (
        vtkCompositeDataSet,
        vtkDataSet,
        vtkMultiBlockDataSet,
        vtkPartitionedDataSet,
        vtkPartitionedDataSetCollection,
    )

    if obj is None:
        return
    if isinstance(obj, vtkMultiBlockDataSet):
        for i in range(obj.GetNumberOfBlocks()):
            child = obj.GetBlock(i)
            name = None
            if obj.HasMetaData(i):
                name = obj.GetMetaData(i).Get(vtkCompositeDataSet.NAME())
            name = name or f"block{i}"
            yield from iter_leaves(child, f"{prefix}/{name}" if prefix else name)
    elif isinstance(obj, vtkPartitionedDataSetCollection):
        for i in range(obj.GetNumberOfPartitionedDataSets()):
            child = obj.GetPartitionedDataSet(i)
            name = None
            if obj.HasMetaData(i):
                name = obj.GetMetaData(i).Get(vtkCompositeDataSet.NAME())
            name = name or f"block{i}"
            yield from iter_leaves(child, f"{prefix}/{name}" if prefix else name)
    elif isinstance(obj, vtkPartitionedDataSet):
        parts = [obj.GetPartition(i) for i in range(obj.GetNumberOfPartitions())]
        parts = [p for p in parts if p is not None and p.GetNumberOfPoints() > 0]
        if len(parts) == 1:
            yield prefix or "main", parts[0]
        elif parts:
            yield prefix or "main", merge_datasets(parts)
    elif isinstance(obj, vtkDataSet):
        if obj.GetNumberOfPoints() > 0:
            yield prefix or "main", obj


def merge_datasets(parts: Sequence[Any]) -> Any:
    """Append several datasets (e.g. the partitions of a decomposed zone) into one
    ``vtkUnstructuredGrid``, merging coincident points."""
    from vtkmodules.vtkFiltersCore import vtkAppendFilter

    app = vtkAppendFilter()
    app.MergePointsOn()
    for p in parts:
        app.AddInputData(p)
    app.Update()
    return app.GetOutput()


def dataset_kind(ds: Any) -> str:
    """Classify a VTK dataset as ``surface`` | ``mesh`` | ``grid`` | ``point_cloud``."""
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkPolyData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkUnstructuredGrid,
    )

    if isinstance(ds, vtkImageData | vtkRectilinearGrid):
        return "grid"
    if ds.GetNumberOfCells() == 0:
        return "point_cloud"
    if isinstance(ds, vtkPolyData):
        if ds.GetNumberOfPolys() + ds.GetNumberOfStrips() > 0:
            return "surface"
        return "point_cloud" if ds.GetNumberOfLines() == 0 else "mesh"
    if isinstance(ds, vtkStructuredGrid):
        dims = [d for d in grid_dims(ds) if d > 1]
        return "surface" if len(dims) == 2 else "mesh"
    if isinstance(ds, vtkUnstructuredGrid):
        types = cell_types(ds)
        if len(types) and meshutil.all_surface(types):
            return "surface"
        if len(types) and np.isin(types, (1, 2)).all():
            return "point_cloud"
    return "mesh"


def _attributes(attrs: Any, patterns: Sequence[str] | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i in range(attrs.GetNumberOfArrays()):
        arr = attrs.GetArray(i)  # None for string arrays
        if arr is None:
            continue
        name = arr.GetName() or f"array{i}"
        if name in _INTERNAL_ARRAYS or not wants(name, patterns):
            continue
        # copy: the VTK reader reuses its buffers on the next UpdateTimeStep
        out[name] = np.array(_np(arr), copy=True)
    return out


def field_infos(ds: Any, zone: str | None = None) -> list[FieldInfo]:
    """Describe the numeric point/cell arrays of one dataset without copying them."""
    out = []
    for loc, attrs in (("point", ds.GetPointData()), ("cell", ds.GetCellData())):
        for i in range(attrs.GetNumberOfArrays()):
            arr = attrs.GetArray(i)
            if arr is None or (arr.GetName() or "") in _INTERNAL_ARRAYS:
                continue
            names = None
            nc = arr.GetNumberOfComponents()
            if nc > 1 and arr.HasAComponentName():
                names = [arr.GetComponentName(c) or str(c) for c in range(nc)]
            out.append(
                FieldInfo(
                    name=arr.GetName() or f"array{i}",
                    location=loc,
                    components=nc,
                    dtype=str(np.dtype(_np_dtype(arr))),
                    component_names=names,
                    zones=[zone] if zone else None,
                )
            )
    return out


def _np_dtype(arr: Any) -> Any:
    from vtkmodules.util.numpy_support import get_numpy_array_type

    return get_numpy_array_type(arr.GetDataType())


def zone_info(name: str, ds: Any, *, role: str | None = None, default: bool = True) -> ZoneInfo:
    """``ZoneInfo`` of one dataset (counts, element census, bounds)."""
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
        vtkUnstructuredGrid,
    )

    kind = dataset_kind(ds)
    cell_census: dict[str, int] = {}
    if isinstance(ds, vtkUnstructuredGrid):
        cell_census = meshutil.census(cell_types(ds))
    elif isinstance(ds, vtkStructuredGrid):
        n = len([d for d in grid_dims(ds) if d > 1])
        cell_name = {3: "hexahedron", 2: "quad", 1: "line"}.get(n, "vertex")
        cell_census = {cell_name: int(ds.GetNumberOfCells())}
    elif kind == "surface":
        cell_census = {"polygon": int(ds.GetNumberOfCells())}
    shape = None
    if isinstance(ds, vtkImageData | vtkRectilinearGrid | vtkStructuredGrid):
        shape = [int(d) for d in grid_dims(ds)]
    return ZoneInfo(
        name=name,
        kind=kind,
        role=role
        or {"surface": "boundary", "grid": "grid", "point_cloud": "probes"}.get(kind, "volume"),
        n_points=int(ds.GetNumberOfPoints()),
        n_cells=int(ds.GetNumberOfCells()),
        cell_types=cell_census,
        shape=shape,
        bounds=[float(b) for b in ds.GetBounds()],
        default=default,
    )


def _points(ds: Any) -> np.ndarray:
    pts = ds.GetPoints()
    if pts is None or pts.GetData() is None:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(_np(pts.GetData()), copy=True)


def _ugrid_cells(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    ca = ds.GetCells()
    conn = _np(ca.GetConnectivityArray())
    offs = _np(ca.GetOffsetsArray())
    return meshutil.pad_cells(conn, offs), cell_types(ds)


def _polydata_polys(ds: Any) -> np.ndarray:
    polys = ds.GetPolys()
    return meshutil.pad_cells(_np(polys.GetConnectivityArray()), _np(polys.GetOffsetsArray()))


def to_unstructured(ds: Any) -> Any:
    """Convert any dataset (structured grid, polydata with strips, ...) to an unstructured grid."""
    from vtkmodules.vtkFiltersCore import vtkAppendFilter

    app = vtkAppendFilter()
    app.AddInputData(ds)
    app.Update()
    return app.GetOutput()


def dataset_to_sample(
    ds: Any,
    sample_id: str,
    *,
    zone: str | None = None,
    fields: Sequence[str] | None = None,
    time: float | None = None,
    time_index: int | None = None,
    geometry: bool = True,
    metadata: dict[str, Any] | None = None,
    force_kind: str | None = None,
) -> RawSample:
    """Convert one VTK dataset into a step-form ``RawSample``.

    * surfaces are triangulated (fan); cell data follows through the parent map,
    * unstructured volumes keep their VTK cell types with -1 padded connectivity,
    * structured (curvilinear) grids become hexahedral/quad unstructured meshes with
      ``metadata["structured_shape"]``,
    * image / rectilinear data become ``grid`` samples with fields shaped ``(X, Y, Z, C)``.

    With ``geometry=False`` the points/cells are omitted (later time steps of a static mesh).
    """
    from vtkmodules.vtkCommonDataModel import (
        vtkImageData,
        vtkPolyData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
    )

    meta = dict(metadata or {})
    kind = force_kind or dataset_kind(ds)
    if kind == "grid" and isinstance(ds, vtkImageData | vtkRectilinearGrid):
        return _grid_sample(ds, sample_id, zone, fields, time, time_index, meta)

    if isinstance(ds, vtkStructuredGrid):
        meta["structured_shape"] = [int(d) for d in grid_dims(ds)]
        ds = to_unstructured(ds)
    elif isinstance(ds, vtkPolyData) and (ds.GetNumberOfStrips() > 0 or kind == "mesh"):
        ds = to_unstructured(ds)
    elif isinstance(ds, vtkImageData | vtkRectilinearGrid):
        ds = to_unstructured(ds)

    pdata = _attributes(ds.GetPointData(), fields)
    cdata = _attributes(ds.GetCellData(), fields)
    sample = RawSample(
        id=sample_id, kind=kind, zone=zone, time=time, time_index=time_index, metadata=meta
    )
    parent: np.ndarray | None = None
    if kind == "surface":
        if isinstance(ds, vtkPolyData):
            cells = _polydata_polys(ds)
            types = None
            # cell data is ordered verts, lines, polys, strips: keep the polygon rows only
            skip = ds.GetNumberOfVerts() + ds.GetNumberOfLines()
            if skip:
                cdata = {k: v[skip : skip + len(cells)] for k, v in cdata.items()}
        else:
            cells, types = _ugrid_cells(ds)
        faces, parent = meshutil.triangulate(cells, types)
        if geometry:
            sample.points = _points(ds)
            sample.faces = faces.astype(np.int32)
        if len(parent) and not np.array_equal(parent, np.arange(len(cells))):
            sample.static["face_parent"] = as_field(parent.astype(np.int32), "cell")
    elif kind == "mesh":
        if geometry:
            cells, types = _ugrid_cells(ds)
            sample.points = _points(ds)
            sample.cells = cells
            sample.cell_types = types
            if (types == meshutil.VTK_POLYHEDRON).any():
                meta["polyhedral"] = True
    else:  # point cloud
        if geometry:
            sample.points = _points(ds)
        cdata = {}
    for name, arr in pdata.items():
        sample.fields[name] = as_field(arr, "point")
    for name, arr in cdata.items():
        if parent is not None:
            arr = arr[parent]
        key = name if name not in sample.fields else f"{name}__cell"
        sample.fields[key] = as_field(arr, "cell")
    if not geometry and kind == "surface":
        sample.static.pop("face_parent", None)
    return sample


def _grid_sample(
    ds: Any,
    sample_id: str,
    zone: str | None,
    fields: Sequence[str] | None,
    time: float | None,
    time_index: int | None,
    meta: dict[str, Any],
) -> RawSample:
    from vtkmodules.vtkCommonDataModel import vtkImageData

    nx, ny, nz = (int(d) for d in grid_dims(ds))
    if isinstance(ds, vtkImageData):
        ox, oy, oz = ds.GetOrigin()
        sx, sy, sz = ds.GetSpacing()
        coords = {
            "x": ox + sx * np.arange(nx),
            "y": oy + sy * np.arange(ny),
            "z": oz + sz * np.arange(nz),
        }
        meta.setdefault("spacing", [sx, sy, sz])
        meta.setdefault("origin", [ox, oy, oz])
    else:
        coords = {
            "x": np.array(_np(ds.GetXCoordinates()), dtype=np.float64),
            "y": np.array(_np(ds.GetYCoordinates()), dtype=np.float64),
            "z": np.array(_np(ds.GetZCoordinates()), dtype=np.float64),
        }
    axes = ["x", "y", "z"]
    shape = [nx, ny, nz]
    if nz == 1:
        axes, shape = ["x", "y"], [nx, ny]
        coords.pop("z")
    sample = RawSample(
        id=sample_id,
        kind="grid",
        zone=zone,
        coords=coords,
        time=time,
        time_index=time_index,
        metadata=meta,
    )
    for name, arr in _attributes(ds.GetPointData(), fields).items():
        a = np.asarray(arr)
        c = 1 if a.ndim == 1 else a.shape[1]
        # VTK orders points x-fastest: reshape as (Z, Y, X, C) then move to (X, Y, Z, C)
        a = a.reshape(nz, ny, nx, c).transpose(2, 1, 0, 3).reshape(*shape, c)
        data = a.astype(np.float32) if a.dtype.kind == "f" else np.ascontiguousarray(a)
        sample.fields[name] = FieldData(data=data, dims=[*axes, "component"], location="grid")
    cdata = _attributes(ds.GetCellData(), fields)
    if not cdata:
        return sample
    cshape = [max(n - 1, 1) for n in (nx, ny, nz)]
    cell_arrays: dict[str, np.ndarray] = {}
    for name, arr in cdata.items():
        a = np.asarray(arr)
        c = 1 if a.ndim == 1 else a.shape[1]
        a = a.reshape(cshape[2], cshape[1], cshape[0], c).transpose(2, 1, 0, 3)
        cell_arrays[name] = a.reshape(*[cshape[i] for i in range(len(axes))], c)
    if not sample.fields:
        # cell data only (finite-volume output): store it losslessly on the grid of cell
        # centres instead of smearing it onto the nodes
        for i, ax in enumerate(axes):
            c = np.asarray(coords[ax], dtype=np.float64)
            sample.coords[ax] = 0.5 * (c[:-1] + c[1:]) if len(c) > 1 else c
            del i
        sample.metadata["grid_location"] = "cell_centers"
        for name, a in cell_arrays.items():
            data = a.astype(np.float32) if a.dtype.kind == "f" else np.ascontiguousarray(a)
            sample.fields[name] = FieldData(data=data, dims=[*axes, "component"], location="grid")
        return sample
    # point and cell data on one grid: the format has a single set of grid axes, so cell
    # values are averaged onto the nodes (each node takes the mean of the cells around it)
    averaged = []
    for name, a in cell_arrays.items():
        key = name if name not in sample.fields else f"{name}__cell"
        sample.fields[key] = FieldData(
            data=cells_to_nodes(a, shape),
            dims=[*axes, "component"],
            location="grid",
            description="cell-centred in the source, averaged to the grid nodes",
        )
        averaged.append(key)
    sample.metadata["cell_fields_averaged_to_nodes"] = averaged
    return sample


def cells_to_nodes(cell_data: np.ndarray, node_shape: Sequence[int]) -> np.ndarray:
    """Average a cell-centred grid array ``(X-1, Y-1[, Z-1], C)`` onto the ``node_shape`` grid:
    every node takes the mean of the cells that touch it."""
    import itertools

    data = np.asarray(cell_data, dtype=np.float64)
    nd = len(node_shape)
    total = np.zeros((*node_shape, data.shape[-1]))
    count = np.zeros((*node_shape, 1))
    spans = [(0, 1) if t == n + 1 else (0,) for t, n in zip(node_shape, data.shape, strict=False)]
    for shift in itertools.product(*spans):
        where = tuple(slice(o, o + n) for o, n in zip(shift, data.shape[:nd], strict=True))
        total[where] += data
        count[where] += 1
    return np.ascontiguousarray(total / np.maximum(count, 1), dtype=np.float32)


def reader_times(reader: Any) -> list[float]:
    """Time values advertised by an (already ``UpdateInformation``-ed) VTK reader."""
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline

    info = reader.GetOutputInformation(0)
    key = vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    if info is None or not info.Has(key):
        return []
    return [float(info.Get(key, i)) for i in range(info.Length(key))]


def update_to_time(reader: Any, time: float | None) -> Any:
    """Update a VTK reader for one time value and return its output data object."""
    if time is not None:
        reader.UpdateTimeStep(float(time))
    else:
        reader.Update()
    return reader.GetOutputDataObject(0)


def describe_object(
    obj: Any,
    desc: Any,
    *,
    role: Any = None,
    default: Any = None,
    rename: Any = None,
) -> None:
    """Fill ``desc.zones`` and ``desc.fields`` (a ``SourceDescription``) from a VTK data object.

    ``role(name, dataset) -> str | None`` and ``default(name, dataset) -> bool`` customise the
    zone rows; ``rename(name) -> str`` maps block paths to zone names. Fields that occur in
    several zones are merged into one row with the list of zones.
    """
    merged: dict[tuple[str, str], FieldInfo] = {}
    for raw_name, ds in iter_leaves(obj):
        name = rename(raw_name) if rename else raw_name
        desc.zones.append(
            zone_info(
                name,
                ds,
                role=role(name, ds) if role else None,
                default=default(name, ds) if default else True,
            )
        )
        for fi in field_infos(ds, name):
            key = (fi.name, fi.location)
            if key in merged:
                zones = merged[key].zones
                if zones is not None and name not in zones:
                    zones.append(name)
            else:
                merged[key] = fi
    n_zones = len(desc.zones)
    for fi in merged.values():
        if fi.zones is not None and len(fi.zones) == n_zones:
            fi.zones = None
        desc.fields.append(fi)


def select_zones(
    names: Sequence[str], zones: Sequence[str] | None, defaults: Sequence[str] | None = None
) -> list[str]:
    """Resolve a zone selection: explicit patterns via ``match_names``, else ``defaults`` (or
    everything)."""
    from frontier_connectors.select import match_names

    if zones is None:
        return list(defaults) if defaults is not None else list(names)
    return match_names(names, zones, what="zone")


def read_reader_steps(
    reader: Any,
    base_id: str,
    *,
    zones: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    times: Any = None,
    metadata: dict[str, Any] | None = None,
    default_zone: Any = None,
    rename: Any = None,
    zone_metadata: Any = None,
    time_values: Sequence[float] | None = None,
) -> Iterator[RawSample]:
    """Stream a time-aware VTK reader as step-form ``RawSample`` objects, one time step in
    memory at a time.

    ``reader`` must already be configured (file name, arrays enabled) and
    ``UpdateInformation()``-ed. ``default_zone(name, dataset) -> bool`` picks the zones read
    when ``zones`` is ``None``; ``zone_metadata(name) -> dict`` adds per-zone metadata (BCs).
    ``time_values`` overrides the reader's advertised times (e.g. when they are step numbers).
    """
    from frontier_connectors.select import select_time_indices

    all_times = reader_times(reader)
    idxs = select_time_indices(all_times, times)
    transient = len(all_times) > 1
    selected: list[str] | None = None
    for ti in idxs:
        t = all_times[ti] if all_times else None
        obj = update_to_time(reader, t)
        leaves = [(rename(n) if rename else n, ds) for n, ds in iter_leaves(obj)]
        names = [n for n, _ in leaves]
        if selected is None:
            defaults = (
                [n for n, ds in leaves if default_zone(n, ds)] if default_zone is not None else None
            )
            selected = select_zones(names, zones, defaults or None)
        t_out = None
        if transient and t is not None:
            t_out = float(time_values[ti]) if time_values is not None else float(t)
        for name, ds in leaves:
            if name not in selected:
                continue
            meta = dict(metadata or {})
            if zone_metadata is not None:
                meta.update(zone_metadata(name) or {})
            yield dataset_to_sample(
                ds,
                meshutil.zone_sample_id(base_id, name, len(selected)),
                zone=name,
                fields=fields,
                time=t_out,
                time_index=ti if transient else None,
                metadata=meta,
            )


#: public names for helpers other connector modules use
points = _points
attributes = _attributes
