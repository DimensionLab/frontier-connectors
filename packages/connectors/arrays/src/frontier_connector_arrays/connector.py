# SPDX-License-Identifier: Apache-2.0
"""Generic array stores: NetCDF (3 and 4), HDF5 and Zarr (v2 and v3), read lazily.

Self-describing stores are detected automatically:

* **NetCDF / CF**: coordinate variables, ``axis`` / ``standard_name`` / ``units`` attributes
  (``time``, ``latitude``, ``longitude``, ``projection_x_coordinate``, height / pressure / level
  coordinates), record dimension, ``_FillValue`` / ``missing_value`` (-> NaN) and
  ``scale_factor`` / ``add_offset``. Every data variable on a grid becomes a field
  ``([time,] x, y[, z], 1)``; variables on different grids become separate zones. Time values
  stay numeric relative to the reference date (``metadata["time_reference"]``) and are converted
  to seconds when the unit is one of seconds/minutes/hours/days. UGRID 2-D face meshes
  (``cf_role = "mesh_topology"``) become ``mesh`` zones.
* **HDF5**: datasets with attached dimension scales (same logic as NetCDF-4, which *is* HDF5
  with dimension scales); The Well (``dimensions/``, ``t0_fields/``, ``t1_fields/``,
  ``t2_fields/``, ``scalars/``, ``boundary_conditions/``); PDEBench / FNO-style files (1-D
  coordinate datasets ``x-coordinate``, ``t-coordinate``, ``x``, ``t``, ``grid/x`` ... plus one
  dataset per field shaped ``([batch,] [time,] x[, y[, z]][, C])``, or one group per sample);
  mesh layouts (``points``/``coordinates``/``nodes`` + ``cells``/``connectivity``/``elements``
  + per-node or per-cell datasets ``([T,] N[, C])``).
* **Zarr**: xarray-style stores (``_ARRAY_DIMENSIONS`` attributes in v2, ``dimension_names``
  in v3), the HDF5 layouts above, and ``frontier-dataset`` directories / sample stores
  (``surface.zarr``, ``mesh.zarr``, ``grid.zarr``, ``points.zarr``) as a round-trip path.

Anything else needs a **mapping config**. The error raised for an unmapped file prints the
dataset tree and a mapping skeleton to start from.

Mapping config
--------------

Passed as ``options={"mapping": {...}}`` (a dict or the path of a JSON file) or on the command
line as ``--mapping mapping.json``. The same schema serves the three connectors. Array paths
are absolute (``/group/array``) or relative to ``group``; ``path@name`` addresses an attribute
(``@name`` = root attribute).

.. code-block:: json

    {
      "kind": "grid",
      "group": "/results",
      "time": "time",
      "coords": {"x": "lon", "y": "lat", "z": "level"},
      "points": "/mesh/coordinates",
      "cells": "/mesh/topology",
      "cell_type": "tetra",
      "cell_types": "/mesh/vtk_types",
      "index_base": 0,
      "fields": {
        "u": {"path": "/fields/u", "dims": ["time", "x", "y", "component"], "units": "m/s"},
        "velocity": {"paths": ["/u", "/v"]},
        "p": "/fields/p"
      },
      "static": {"mask": "/mask"},
      "samples": {"axis": "batch", "ids": "/names"},
      "parameters": {"Re": "/params@Re", "nu": "/nu", "Ma": 0.3},
      "units": {"length": "m"},
      "domain": "cfd",
      "solver": "my-solver",
      "solver_version": "1.2",
      "physics": {"equations": "incompressible-ns"},
      "boundary_conditions": {"x": "periodic"},
      "metadata": {"anything": "copied into meta.json"},
      "zones": {"fluid": {"...": "a complete mapping per zone"}}
    }

``kind``
    ``grid`` (needs ``coords``), ``mesh`` or ``surface`` (need ``points`` and ``cells``),
    ``point_cloud`` (needs ``points``). Default: ``grid`` when ``coords`` is given, ``mesh`` when
    ``cells`` is given, else ``point_cloud``.
``time``
    Path of the time coordinate (CF ``units`` such as ``hours since 2000-01-01`` are decoded),
    ``{"path": ..., "units": "ms"}``, ``{"start": 0, "dt": 0.01}``, ``{"attr": "/sim@dt"}``
    (``dt`` taken from an attribute, optional ``start``) or ``{"values": [...]}``. Omitted for a
    steady store; when fields have a time axis but no ``time`` is mapped, step indices are used.
``coords``
    1-D axes of a grid, always named ``x``, then ``y``, then ``z`` whatever they mean physically
    (the on-disk names are kept in ``metadata["axis_names"]``). Each entry is an array path,
    ``{"start": 0, "step": 0.1}``, ``{"linspace": [0, 1]}`` (length taken from the fields) or
    ``{"values": [...]}``.
``points`` / ``cells`` / ``cell_type`` / ``cell_types`` / ``index_base``
    ``points`` is an ``(N, 2|3)`` array or a list of 1-D coordinate arrays; a ``(T, N, D)``
    array contributes its first step. ``cells`` is ``(K, P)`` point indices in **VTK node
    order**, padded with a negative value or ``_FillValue`` for mixed meshes. ``cell_type`` is a
    name (``triangle``, ``quad``, ``tetra``, ``hexahedron``, ``wedge``, ``pyramid``, ``line``,
    ``polygon`` = triangles/quads/polygons by node count) or a VTK type id; ``cell_types`` is
    the path of per-cell VTK ids. Without either the type follows from the row width.
    ``index_base`` is 0 or 1 (default: the ``start_index`` attribute, else 1 when indices run
    ``1..N``, else 0). ``kind: surface`` triangulates 2-D cells and carries cell data along.
``fields``
    ``name -> spec``. ``path`` is one array, ``paths`` stacks several equally shaped arrays into
    a vector. ``dims`` names **every on-disk axis in on-disk order** using ``time``,
    ``x``/``y``/``z`` (grids), ``point`` or ``cell`` (meshes, point clouds), ``component`` (may
    repeat: tensors are flattened) and the sample axis name; the reader transposes lazily to
    the contract order ``([time,] x, y[, z], component)`` / ``([time,] point|cell, component)``.
    Any other axis name needs ``"select": {"name": index}`` (or must have length 1). Without
    ``dims`` the stored dimension names are used, then the conventional order
    ``([sample,] [time,] space..., [component])`` checked against the axis lengths. Optional:
    ``units``, ``description``, ``component_names``, ``location``. Without ``fields`` every
    numeric array that fits the mapped axes is taken.
``static``
    Same as ``fields`` for time-independent inputs (masks, material maps, wall distance).
``samples``
    ``{"axis": "batch"}``: fields carry a leading sample axis named ``batch`` in ``dims`` (the
    names ``sample``, ``batch``, ``trajectory``, ``member``, ``realization`` are always
    recognised) and every index becomes one sample ``<file>_<id>``; ``ids`` is an array of
    names. ``{"groups": "traj_*"}``: every matching group is one sample and all other paths are
    relative to it. ``options={"samples": 10 | "0:100:10" | [3, 7] | ["traj_0*"]}`` reads a
    subset.
``parameters``
    Per-sample scalar conditioning: a number, ``path@attribute``, or an array path (a scalar,
    or one value per sample).
``units`` / ``domain`` / ``solver`` / ``solver_version`` / ``physics`` /
``boundary_conditions`` / ``metadata``
    Copied into the dataset metadata.
``zones``
    Several zones in one file: ``{"zone name": {<mapping>}}``; the other top-level keys are
    defaults for every zone.

A *partial* mapping (only ``fields`` / ``static`` / ``parameters`` / metadata keys) is applied
on top of the detected layout, e.g. ``{"fields": {"wind": {"paths": ["u10", "v10"]}}}`` groups
two detected scalars into a vector. ``describe`` returns the mapping in effect under
``metadata["mapping"]``.

Reads are lazy: time-dependent fields are yielded as block-form samples whose ``FieldData.data``
is an :class:`~frontier_connectors._arraymap.LazyArray`; the dataset writer slices it
a few steps at a time. ``times=`` narrows that view without reading anything.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import _arraymap as am
from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError, SelectionError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldData,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    ZoneInfo,
    match_extension,
)
from frontier_connectors.paths import resolve_in_case
from frontier_connectors.select import match_names, select_time_indices, wants

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
NETCDF_EXTENSIONS = [".nc", ".nc4", ".nc3", ".netcdf", ".cdf", ".cdf5"]
HDF5_EXTENSIONS = [".h5", ".hdf5", ".hdf", ".he5"]
FRONTIER_STORES = {
    "surface.zarr": "surface",
    "mesh.zarr": "mesh",
    "grid.zarr": "grid",
    "points.zarr": "point_cloud",
}

#: file-name endings of HDF5/NetCDF based formats that have their own connector
_FOREIGN_SUFFIXES = {
    ".cas.h5": "fluent-cff", ".dat.h5": "fluent-cff", ".msh.h5": "fluent-cff",
    ".cgns": "cgns", ".mat": "matlab", ".vtkhdf": "vtk", ".med": "med", ".rmed": "med",
    ".pyfrs": "pyfr", ".pyfrm": "pyfr", ".exo": "exodus", ".e": "exodus", ".ex2": "exodus",
    ".exii": "exodus", ".g": "exodus", ".xdmf": "xdmf", ".xmf": "xdmf",
}  # fmt: skip


def _head(path: Path, n: int = 4096) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _is_hdf5(head: bytes) -> bool:
    # the signature may sit after a 512-byte user block (MATLAB v7.3 does that)
    return head.startswith(HDF5_MAGIC) or head[512:520] == HDF5_MAGIC


def foreign_hdf5(path: Path, head: bytes) -> str | None:
    """Connector id that owns this HDF5/NetCDF file (Fluent CFF, CGNS, MATLAB v7.3, VTKHDF,
    Exodus, XDMF heavy data, MED, PyFR), else ``None``. Reads the root group only."""
    owner = match_extension(path, list(_FOREIGN_SUFFIXES))
    if owner is not None:
        return _FOREIGN_SUFFIXES[owner]
    if head.startswith(b"MATLAB 7.3 MAT-file"):
        return "matlab"
    if any(path.with_suffix(s).is_file() for s in (".xdmf", ".xmf")):
        return "xdmf"
    if head.startswith(b"CDF"):
        return "exodus" if (b"connect1" in head or b"num_el_blk" in head) else None
    try:
        import h5py

        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            attrs = set(f.attrs.keys())
    except Exception:  # noqa: BLE001 - unreadable here means "not ours to judge"
        return None
    if "VTKHDF" in keys:
        return "vtk"
    if keys & {"CGNSLibraryVersion", " format", " hdf5version"}:
        return "cgns"
    if "meshes" in keys and keys & {"results", "settings"}:
        return "fluent-cff"
    if keys & {"INFOS_GENERALES", "ENS_MAA", "FAS", "CHA"}:
        return "med"
    if "#refs#" in keys or "#subsystem#" in keys:
        return "matlab"
    if "mesh_uuid" in keys or any(k.startswith(("spt_", "soln_", "con_p")) for k in keys):
        return "pyfr"
    if {"api_version", "floating_point_word_size"} <= attrs or keys & {"connect1", "coordx"}:
        return "exodus"
    return None


def _apply_overrides(desc: SourceDescription, options: Mapping[str, Any] | None) -> None:
    for key in ("domain", "solver"):
        if options and options.get(key):
            setattr(desc, key, str(options[key]))


class _ArrayStoreConnector(BaseConnector):
    """Describe/read shared by the three array-store connectors."""

    vendor = "open format"
    temporal = ["steady", "transient"]
    kinds = ["grid", "mesh", "surface", "point_cloud"]
    detectors: tuple[Callable[[am.ArrayStore], dict[str, Any] | None], ...] = (
        am.cf_mapping,
        am.generic_mapping,
    )

    def _open(self, path: Path, options: Mapping[str, Any] | None) -> am.ArrayStore:
        raise NotImplementedError

    def _plan(self, path: Path, options: Mapping[str, Any] | None) -> am.Plan:
        if not Path(path).exists() and not (options or {}).get("url"):
            raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
        store = self._open(Path(path), options)
        return am.make_plan(store, options, self.detectors)

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        plan = self._plan(path, options)
        store = plan.layouts[0].store
        desc = SourceDescription(connector_id=self.id, path=path, format=store.format)
        am.describe_layouts(plan, desc)
        _apply_overrides(desc, options)
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
        plan = self._plan(path, options)
        extra = {"format": plan.layouts[0].store.format}
        yield from am.read_layouts(
            plan,
            path,
            zones=zones,
            fields=fields,
            times=times,
            options=options,
            extra_metadata=extra,
        )


class NetcdfConnector(_ArrayStoreConnector):
    id = "netcdf"
    name = "NetCDF 3/4 (CF conventions, UGRID 2-D), optional xarray engine"
    formats = ["netcdf4", "netcdf3", "cf", "ugrid"]
    vendor = "Unidata (open format)"
    license = "Apache-2.0 (readers: h5py BSD-3, scipy BSD-3; optional xarray Apache-2.0)"
    extensions = NETCDF_EXTENSIONS
    maturity = "verified-public-sample"
    requires = ["xarray"]
    detectors = (am.cf_mapping,)
    export_help = (
        "Most climate/ocean/weather models and many CFD codes write NetCDF directly; "
        "from Python use xarray.Dataset.to_netcdf, from ParaView/VisIt use the NetCDF writer. "
        "OPeNDAP URLs, GRIB and multi-file datasets go through options={'engine': 'xarray'}."
    )
    status_note = (
        "NetCDF-4 is read with h5py, NetCDF-3 classic and 64-bit-offset with scipy "
        "(memory-mapped). CF grids with 1-D coordinates and UGRID 2-D face meshes are detected "
        "automatically. Curvilinear grids, CDF-5, UGRID 1-D / 3-D-layered and DSG feature types "
        "need a mapping or options={'engine': 'xarray'}."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        if not path.is_file():
            return None
        ext = match_extension(path, self.extensions)
        head = _head(path)
        is_cdf = head[:3] == b"CDF" and head[3:4] in (b"\x01", b"\x02", b"\x05")
        if not is_cdf and not _is_hdf5(head):
            if ext is None or match_extension(path, list(_FOREIGN_SUFFIXES)):
                return None
            return Detection(
                self.id, path, 0.3, "netcdf", [f"extension {ext} but no NetCDF/HDF5 signature"]
            )
        owner = foreign_hdf5(path, head)
        if owner is not None:
            return None
        if is_cdf:
            fmt = {b"\x01": "classic", b"\x02": "64-bit offset", b"\x05": "CDF-5"}[head[3:4]]
            det = Detection(self.id, path, 0.95, "netcdf3", [f"NetCDF-3 {fmt} signature"])
            if head[3:4] == b"\x05":
                det.needs.append("xarray + netCDF4 (CDF-5 is not read by the built-in reader)")
            return det
        if ext is not None:
            return Detection(self.id, path, 0.9, "netcdf4", [f"HDF5 signature, extension {ext}"])
        try:
            import h5py

            with h5py.File(path, "r") as f:
                nc = "_NCProperties" in f.attrs
        except Exception:  # noqa: BLE001
            return None
        if not nc:
            return None
        return Detection(self.id, path, 0.6, "netcdf4", ["HDF5 file with _NCProperties"])

    def _open(self, path: Path, options: Mapping[str, Any] | None) -> am.ArrayStore:
        opts = options or {}
        engine = str(opts.get("engine") or "").lower()
        if engine == "xarray":
            return am.open_xarray(path, opts)
        if engine not in ("", "auto", "h5py", "scipy"):
            raise ConnectorError(
                f"unknown NetCDF engine {engine!r}", hint="use auto (default) or xarray"
            )
        head = _head(path, 520)
        if head[:3] == b"CDF":
            return am.open_nc3(path)
        if _is_hdf5(head):
            store = am.open_h5(path)
            store.format = "netcdf4"
            return store
        raise ConnectorError(
            f"{path} is neither NetCDF-3 (CDF magic) nor NetCDF-4 (HDF5 magic)",
            hint="the file is truncated or another format; GRIB / OPeNDAP need "
            "options={'engine': 'xarray'}",
        )


class Hdf5Connector(_ArrayStoreConnector):
    id = "hdf5"
    name = "Generic HDF5 (dimension scales, PDEBench / The Well / mesh layouts, mapping config)"
    formats = ["hdf5"]
    vendor = "The HDF Group (open format)"
    license = "Apache-2.0 (reader: h5py, BSD-3)"
    extensions = HDF5_EXTENSIONS
    maturity = "verified-public-sample"
    #: generic container: solver-specific HDF5 connectors must win detection
    extension_confidence = 0.3
    export_help = (
        "Write the arrays with h5py / HDF5 from any solver or script, then describe the layout "
        "with a mapping config (--mapping file.json) unless it is one of the auto-detected "
        "layouts (dimension scales, PDEBench, The Well, points+cells+fields)."
    )
    status_note = (
        "Detects datasets with dimension scales, The Well, PDEBench / FNO-style grids and points + "
        "cells mesh layouts. Any other layout needs a mapping config: the error prints the dataset "
        "tree and a mapping skeleton. The Well, dimension-scale and mesh layouts are validated on "
        "generated files only. Fluent CFF, CGNS, MATLAB v7.3, VTKHDF, Exodus, XDMF heavy data, MED "
        "and PyFR files belong to their own connectors."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        ext = match_extension(path, self.extensions)
        if ext is None or not path.is_file():
            return None
        head = _head(path)
        if not _is_hdf5(head):
            return None
        if foreign_hdf5(path, head) is not None:
            return None
        return Detection(
            self.id,
            path,
            self.extension_confidence,
            "hdf5",
            [f"HDF5 signature, extension {ext} (generic container)"],
            ["a mapping config unless the layout is self-describing"],
        )

    def _open(self, path: Path, options: Mapping[str, Any] | None) -> am.ArrayStore:
        if not path.is_file():
            raise UnsupportedSourceError(f"{path} is not a file", hint="pass one HDF5 file")
        return am.open_h5(path)


# -- frontier-dataset round trip ---------------------------------------------------------------


def _json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ConnectorError(f"{path} is not readable JSON ({e})") from e
    return doc if isinstance(doc, dict) else {}


def _frontier_form(path: Path) -> str | None:
    """``"dataset"`` (manifest.json), ``"sample"`` (a ``samples/<id>`` directory) or ``"store"``
    (one ``surface.zarr``/``mesh.zarr``/``grid.zarr``/``points.zarr``)."""
    if not path.is_dir():
        return None
    manifest = path / "manifest.json"
    if manifest.is_file():
        try:
            if json.loads(manifest.read_text()).get("format") == "frontier-dataset":
                return "dataset"
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
    if any((path / s).is_dir() for s in FRONTIER_STORES):
        return "sample"
    if path.name in FRONTIER_STORES and (path / "zarr.json").is_file():
        try:
            attrs = json.loads((path / "zarr.json").read_text()).get("attributes", {})
        except (OSError, json.JSONDecodeError):
            return None
        return "store" if str(attrs.get("format", "")).startswith("frontier-dataset/") else None
    return None


def _frontier_entries(path: Path, form: str) -> list[tuple[str, str, Path]]:
    """``[(sample id, kind, store path)]`` of a dataset / sample directory / store."""
    if form == "store":
        sid = path.parent.name if (path.parent / "meta.json").is_file() else path.stem
        return [(meshutil.sanitize_id(sid), FRONTIER_STORES[path.name], path)]
    if form == "sample":
        store = next(path / s for s in FRONTIER_STORES if (path / s).is_dir())
        return [(meshutil.sanitize_id(path.name), FRONTIER_STORES[store.name], store)]
    out = []
    for entry in _json(path / "manifest.json").get("samples", []):
        sdir = resolve_in_case(
            path,
            str(entry.get("path") or f"samples/{entry.get('id')}"),
            what="manifest sample path",
            referrer=path / "manifest.json",
        )
        store = next((sdir / s for s in FRONTIER_STORES if (sdir / s).is_dir()), None)
        if store is None:
            raise ConnectorError(
                f"sample {entry.get('id')!r} has no zarr store under {sdir}",
                hint="the dataset directory is incomplete; re-run the conversion",
            )
        out.append((str(entry["id"]), FRONTIER_STORES[store.name], store))
    if not out:
        raise ConnectorError(f"{path}/manifest.json lists no samples")
    return out


def _zopen(store: Path) -> Any:
    import zarr

    try:
        return zarr.open_group(str(store), mode="r")
    except Exception as e:  # noqa: BLE001
        raise ConnectorError(
            f"{store} is not a readable zarr group ({type(e).__name__}: {e})",
            hint="the sample store is incomplete",
        ) from e


def _frontier_arrays(g: Any) -> list[tuple[str, str, Any]]:
    """``[(group, field name, zarr array)]`` for fields, cell_fields and static."""
    out = []
    for gname in ("fields", "cell_fields", "static"):
        if gname not in g:
            continue
        for name, obj in g[gname].members(max_depth=None):
            if hasattr(obj, "shape"):
                out.append((gname, str(obj.attrs.get("source_name") or name), obj))
    return out


def _frontier_dims(gname: str, arr: Any, kind: str, transient: bool) -> list[str]:
    dims = arr.attrs.get("dims")
    if dims and len(dims) == len(arr.shape):
        return [str(d) for d in dims]
    if kind == "grid":
        space = list(am.GRID_AXES[: arr.ndim - 1 - int(transient and gname != "static")])
    else:
        space = ["cell" if gname == "cell_fields" else "point"]
    lead = ["time"] if transient and gname != "static" and arr.ndim == len(space) + 2 else []
    return (lead + space + ["component"])[: arr.ndim]


class ZarrConnector(_ArrayStoreConnector):
    id = "zarr"
    name = "Zarr v2/v3 (xarray-style stores, mapping config, frontier-dataset round trip)"
    formats = ["zarr", "zarr-v2", "zarr-v3", "frontier-dataset"]
    vendor = "Zarr community (open format)"
    license = "Apache-2.0 (reader: zarr-python, MIT)"
    extensions = [".zarr"]
    maturity = "verified-synthetic"
    export_help = (
        "xarray: Dataset.to_zarr(path). Any zarr group written by a solver or script can be "
        "described with a mapping config. A frontier-dataset directory (or one of its sample "
        "stores) is read back as-is."
    )
    status_note = (
        "Reads directory and .zip stores in Zarr v2 and v3. Dimension names come from "
        "_ARRAY_DIMENSIONS (v2) or dimension_names (v3); CF attributes are honoured as for NetCDF. "
        "Remote stores (s3://, gs://) are not opened. Validated on generated files only."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        if path.is_file():
            if path.name.lower().endswith(".zarr.zip"):
                return Detection(self.id, path, 0.6, "zarr-zip", ["extension .zarr.zip"])
            return None
        if not path.is_dir():
            return None
        form = _frontier_form(path)
        if form is not None:
            return Detection(self.id, path, 0.9, "frontier-dataset", [f"frontier-dataset {form}"])
        v3 = (path / "zarr.json").is_file()
        v2 = any((path / n).is_file() for n in (".zgroup", ".zarray", ".zmetadata"))
        if v3 or v2:
            conf = 0.95 if path.suffix.lower() == ".zarr" else 0.9
            fmt = "zarr-v3" if v3 else "zarr-v2"
            return Detection(self.id, path, conf, fmt, [f"{fmt} metadata file"])
        if path.suffix.lower() == ".zarr":
            return Detection(self.id, path, 0.4, "zarr", ["directory named *.zarr, no metadata"])
        return None

    def _open(self, path: Path, options: Mapping[str, Any] | None) -> am.ArrayStore:
        return am.open_zarr(path)

    # frontier-dataset --------------------------------------------------------------------
    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        form = _frontier_form(path)
        if form is None:
            return super().describe(path, options=options)
        entries = _frontier_entries(path, form)
        desc = SourceDescription(connector_id=self.id, path=path, format="frontier-dataset")
        manifest = _json(path / "manifest.json") if form == "dataset" else {}
        desc.domain = str(manifest.get("domain") or "other")
        desc.solver = str((manifest.get("source") or {}).get("solver") or "unknown")
        counts: dict[str, int] = {}
        for _sid, kind, _store in entries:
            counts[kind] = counts.get(kind, 0) + 1
        seen: set[tuple[str, str]] = set()
        for kind in counts:
            sid, _, store = next(e for e in entries if e[1] == kind)
            g = _zopen(store)
            attrs = dict(g.attrs)
            transient = "times" in g
            role = {"grid": "grid", "point_cloud": "probes", "surface": "boundary"}.get(
                kind, "volume"
            )
            z = ZoneInfo(name=kind, kind=kind, role=role)
            if kind == "grid":
                z.shape = [int(s) for s in attrs.get("shape") or []]
                z.n_points = int(np.prod(z.shape)) if z.shape else None
                z.dim = len(z.shape)
            else:
                z.n_points = int(g["points"].shape[0])
                z.dim = int(g["points"].shape[1])
                z.bounds = [float(b) for b in attrs.get("bounds") or []] or None
                if kind == "surface":
                    z.n_cells = int(g["faces"].shape[0])
                    z.cell_types = {"triangle": z.n_cells}
                elif kind == "mesh":
                    z.n_cells = int(g["cells"].shape[0])
            desc.zones.append(z)
            for gname, name, arr in _frontier_arrays(g):
                dims = _frontier_dims(gname, arr, kind, transient)
                loc = "grid" if kind == "grid" else ("cell" if "cell" in dims else "point")
                if (name, loc) in seen:
                    continue
                seen.add((name, loc))
                desc.fields.append(
                    FieldInfo(
                        name=name,
                        location=loc,
                        components=int(arr.shape[-1]) if dims[-1] == "component" else 1,
                        units=str(arr.attrs.get("units") or ""),
                        time_dependent=dims[0] == "time",
                        zones=[kind] if len(counts) > 1 else None,
                    )
                )
            if transient and desc.time.steady:
                meta = _json(store.parent / "meta.json") if form != "store" else {}
                units = str((meta.get("time") or {}).get("units") or "s")
                desc.time = TimeInfo(values=[float(v) for v in g["times"][...]], units=units)
            if not desc.parameters and (store.parent / "meta.json").is_file():
                desc.parameters = dict(_json(store.parent / "meta.json").get("parameters") or {})
            del sid
        desc.metadata["n_samples"] = len(entries)
        desc.metadata["sample_ids"] = [e[0] for e in entries[:20]]
        if manifest.get("name"):
            desc.metadata["dataset"] = manifest["name"]
        if len(entries) > 1:
            desc.warnings.append(
                "zones are the sample kinds; pick samples with options={'samples': [ids|globs]}"
            )
        _apply_overrides(desc, options)
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
        form = _frontier_form(path)
        if form is None:
            yield from super().read(path, zones=zones, fields=fields, times=times, options=options)
            return
        entries = _frontier_entries(path, form)
        kinds = match_names(list(dict.fromkeys(e[1] for e in entries)), zones)
        ids = [e[0] for e in entries]
        picked = am.select_samples((options or {}).get("samples"), len(entries), ids)
        for i in picked:
            sid, kind, store = entries[i]
            if kind in kinds:
                yield from _read_frontier_sample(sid, kind, store, fields, times)


def _frontier_metadata(store: Path) -> dict[str, Any]:
    meta_path = store.parent / "meta.json"
    md: dict[str, Any] = {"source": str(store), "format": "frontier-dataset"}
    if not meta_path.is_file():
        return {"solver": "unknown", **md}
    meta = _json(meta_path)
    extra = dict(meta.get("extra") or {})
    md["solver"] = str(meta.get("solver") or "unknown")
    for key in ("solver_version", "group", "fidelity", "application", "turbulence_model"):
        if meta.get(key) is not None:
            md[key] = meta[key]
    for key in ("parameters", "boundary_conditions", "conditions", "geometry_params"):
        if meta.get(key):
            md[key] = meta[key]
    if isinstance(extra.get("physics"), dict):
        md["physics"] = extra["physics"]
    if isinstance(extra.get("units"), dict):
        md["units"] = extra["units"]
    if (meta.get("provenance") or {}).get("source"):
        md["original_source"] = meta["provenance"]["source"]
    if (meta.get("time") or {}).get("units"):
        md["time_units"] = meta["time"]["units"]
    return md


def _read_frontier_sample(
    sid: str, kind: str, store: Path, fields: Sequence[str] | None, times: TimeSelection
) -> Iterator[RawSample]:
    g = _zopen(store)
    try:
        geo: dict[str, Any] = {}
        if kind == "grid":
            geo["coords"] = {
                a: np.asarray(g["coords"][a][...], dtype=np.float64)
                for a in am.GRID_AXES
                if a in g["coords"]
            }
        else:
            geo["points"] = np.asarray(g["points"][...])
            if kind == "surface":
                geo["faces"] = np.asarray(g["faces"][...], dtype=np.int64)
            elif kind == "mesh":
                geo["cells"] = np.asarray(g["cells"][...], dtype=np.int64)
                geo["cell_types"] = np.asarray(g["cell_types"][...], dtype=np.uint8)
        tvals = np.asarray(g["times"][...], dtype=np.float64) if "times" in g else None
    except KeyError as e:
        raise ConnectorError(
            f"{store} is missing array {e} required for a {kind} sample",
            hint="the store does not follow frontier-dataset v1.1",
        ) from e
    transient = tvals is not None
    sel = select_time_indices(tvals, times) if tvals is not None else [0]
    full = tvals is not None and sel == list(range(len(tvals)))
    sample = RawSample(id=sid, kind=kind, zone=kind, metadata=_frontier_metadata(store), **geo)
    timed: dict[str, tuple[Any, list[str], FieldData]] = {}
    for gname, name, arr in _frontier_arrays(g):
        dims = _frontier_dims(gname, arr, kind, transient)
        loc = "grid" if kind == "grid" else ("cell" if "cell" in dims else "point")
        fd = FieldData(
            data=None,
            dims=dims,
            location=loc,
            units=str(arr.attrs.get("units") or ""),
            component_names=arr.attrs.get("component_names"),
            description=str(arr.attrs.get("description") or ""),
        )
        if gname == "static":
            fd.data = np.asarray(arr[...])
            sample.static[name] = fd
        elif not wants(name, fields):
            continue
        elif dims[0] == "time" and transient:
            node = am.ArrayNode(name, arr.shape, arr.dtype, arr)
            fd.data = am.LazyArray(
                [node], dims, dims[:-1], time_index=None if full else sel, decode=False
            )
            timed[name] = (arr, dims, fd)
            sample.fields[name] = fd
        elif transient:
            fd.data = np.asarray(arr[...])
            sample.static[name] = fd
        else:
            fd.data = np.asarray(arr[...])
            sample.fields[name] = fd
    if fields is not None and not sample.fields and not sample.static:
        names = sorted({n for _, n, _ in _frontier_arrays(g)})
        raise SelectionError(
            f"no field of sample {sid!r} matches {list(fields)}",
            hint="available fields: " + ", ".join(names),
        )
    if not (transient and timed):
        yield sample
        return
    assert tvals is not None
    if "points_t" not in g:
        sample.times = tvals[sel]
        yield sample
        return
    # a moving mesh cannot be expressed in block form: yield one step at a time
    points_t = g["points_t"]
    for n, ti in enumerate(sel):
        step = RawSample(
            id=sid,
            kind=kind,
            zone=kind,
            points=np.asarray(points_t[ti]),
            faces=sample.faces if n == 0 else None,
            cells=sample.cells if n == 0 else None,
            cell_types=sample.cell_types if n == 0 else None,
            static=sample.static if n == 0 else {},
            time=float(tvals[ti]),
            time_index=int(ti),
            metadata=sample.metadata,
        )
        for name, (arr, dims, fd) in timed.items():
            step.fields[name] = FieldData(
                data=np.asarray(arr[ti]),
                dims=dims[1:],
                location=fd.location,
                units=fd.units,
                component_names=fd.component_names,
                description=fd.description,
            )
        yield step
