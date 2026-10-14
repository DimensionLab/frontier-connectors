# SPDX-License-Identifier: Apache-2.0
"""Shared machinery of the array-store connectors (NetCDF / HDF5 / Zarr).

Three layers, none of which loads heavy data before a sample is written:

* :class:`ArrayStore`: a flat view of every array in a container (path, shape, dtype,
  attributes, dimension names and coordinate arrays when the container records them), with one
  opener per container type (:func:`open_h5`, :func:`open_nc3`, :func:`open_zarr`,
  :func:`open_xarray`).
* the *mapping config* (documented in :mod:`frontier_connector_arrays.connector`) resolved by
  :func:`resolve_mapping` into :class:`Layout` objects. Auto-detection never builds layouts
  directly: every detector (:func:`cf_mapping`, :func:`well_mapping`, :func:`mlpde_mapping`,
  :func:`mesh_mapping`) writes a mapping config, so what was guessed can be printed, edited and
  passed back with ``--mapping``.
* :class:`LazyArray`: the object placed in ``FieldData.data``. It reads from disk only when it
  is sliced along its leading (time) axis and applies axis transposition, component stacking,
  fill-value masking and ``scale_factor``/``add_offset`` to that slice only.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import (
    ConnectorError,
    MappingError,
    MissingDependencyError,
    SelectionError,
)
from frontier_connectors.model import (
    FieldData,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    ZoneInfo,
    jsonable,
)
from frontier_connectors.paths import h5_open
from frontier_connectors.select import match_names, select_time_indices, wants

GRID_AXES = ("x", "y", "z")
SAMPLE_NAMES = ("sample", "batch", "trajectory", "realization", "member", "ensemble", "run")
#: a trailing unnamed axis longer than this is not taken for a component axis
MAX_COMPONENTS = 16

CELL_TYPES: dict[str, int] = {
    "vertex": 1, "line": 3, "triangle": 5, "tri": 5, "polygon": 7, "quad": 9, "tetra": 10,
    "tet": 10, "tetrahedron": 10, "hexahedron": 12, "hex": 12, "hexa": 12, "wedge": 13,
    "prism": 13, "pyramid": 14,
}  # fmt: skip
#: nodes per cell -> VTK type, for 3-D and 2-D point sets (4 nodes are tets in 3-D, quads in 2-D)
_WIDTH_TYPES_3D = {2: 3, 3: 5, 4: 10, 5: 14, 6: 13, 8: 12}
_WIDTH_TYPES_2D = {2: 3, 3: 5, 4: 9}


# -- attribute helpers -----------------------------------------------------------------------


def clean_attr(v: Any) -> Any:
    """HDF5/NetCDF/Zarr attribute value -> plain Python (bytes decoded, 1-element arrays
    unwrapped, other arrays as lists)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").rstrip("\x00")
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "SUO":
            items = [clean_attr(x) for x in v.ravel().tolist()]
            return items[0] if v.size == 1 else items
        if v.dtype.kind not in "fiub":
            return str(v)
        if v.size == 1:
            return v.ravel()[0].item()
        return v.tolist()
    if isinstance(v, np.generic):
        return clean_attr(v.item())
    return v


def _text(v: Any) -> str:
    v = clean_attr(v)
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _number(v: Any) -> float | None:
    v = clean_attr(v)
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def norm_path(path: str) -> str:
    return str(path).strip().strip("/")


def basename(path: str) -> str:
    return norm_path(path).rsplit("/", 1)[-1]


def parent(path: str) -> str:
    p = norm_path(path)
    return p.rsplit("/", 1)[0] if "/" in p else ""


# -- array store -----------------------------------------------------------------------------


class ArrayNode:
    """One array of a container. ``handle[...]`` returns numpy; nothing is read before that."""

    def __init__(
        self,
        path: str,
        shape: Sequence[int],
        dtype: Any,
        handle: Any,
        attrs: Callable[[], dict[str, Any]] | dict[str, Any] | None = None,
        dims: list[str] | None = None,
        scales: list[str | None] | None = None,
    ) -> None:
        self.path = norm_path(path)
        self.shape = tuple(int(s) for s in shape)
        try:
            self.dtype = np.dtype(dtype)
        except TypeError:
            self.dtype = np.dtype("O")
        self.handle = handle
        self._attrs = attrs
        #: dimension name per axis when the container records them
        self.dims = dims
        #: path of the coordinate array of every axis (``None`` = no coordinate variable)
        self.scales = scales
        #: a NetCDF-4 dimension without a coordinate variable (carries a size, no data)
        self.phantom = False

    @property
    def attrs(self) -> dict[str, Any]:
        if callable(self._attrs):
            try:
                self._attrs = self._attrs()
            except Exception:  # noqa: BLE001 - unreadable attributes must not stop a read
                self._attrs = {}
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def name(self) -> str:
        return basename(self.path)

    @property
    def numeric(self) -> bool:
        return self.dtype.kind in "fiub" and not self.phantom

    def read(self) -> np.ndarray:
        if self.ndim == 0:
            return np.asarray(self.handle[()])
        return np.asarray(self.handle[...])

    def __repr__(self) -> str:
        return f"ArrayNode({self.path!r}, {self.shape}, {self.dtype})"


class ArrayStore:
    """Flat index of the arrays in one container (insertion order = on-disk order)."""

    def __init__(self, path: Path | str, fmt: str) -> None:
        self.path = Path(path)
        self.format = fmt
        self.nodes: dict[str, ArrayNode] = {}
        #: group path ("" = root) -> attributes
        self.group_attrs: dict[str, dict[str, Any]] = {"": {}}
        #: objects that must stay alive while lazy arrays reference the container
        self.keep: list[Any] = []
        self.warnings: list[str] = []

    def add(self, node: ArrayNode) -> None:
        self.nodes[node.path] = node

    def has(self, path: str) -> bool:
        return norm_path(path) in self.nodes

    def node(self, path: str, prefix: str = "", *, what: str = "array") -> ArrayNode:
        """Array at ``path``: absolute when it starts with ``/``, else relative to ``prefix``."""
        raw = str(path)
        p = norm_path(raw)
        if prefix and not raw.startswith("/"):
            rel = f"{norm_path(prefix)}/{p}"
            if rel in self.nodes:
                return self.nodes[rel]
        if p in self.nodes:
            return self.nodes[p]
        near = [n for n in self.nodes if basename(n).lower() == basename(p).lower()]
        shown = ", ".join(("/" + n) for n in (near or list(self.nodes))[:20])
        raise MappingError(
            f"mapping {what} {raw!r} does not exist in {self.path.name}",
            hint=f"{'similar' if near else 'available'} arrays: {shown or '(none)'}",
        )

    def attr(self, spec: str, prefix: str = "") -> Any:
        """Attribute addressed as ``"path@name"`` (``"@name"`` = root attribute)."""
        target, _, name = str(spec).partition("@")
        p = norm_path(target)
        candidates = [p] if (target.startswith("/") or not prefix) else [f"{prefix}/{p}", p]
        for c in candidates:
            c = norm_path(c)
            if c in self.nodes and name in self.nodes[c].attrs:
                return clean_attr(self.nodes[c].attrs[name])
            if c in self.group_attrs and name in self.group_attrs[c]:
                return clean_attr(self.group_attrs[c][name])
        raise MappingError(
            f"attribute {spec!r} not found in {self.path.name}",
            hint="address attributes as '/group/or/array@name' ('@name' for a root attribute)",
        )

    def groups(self) -> list[str]:
        out = dict.fromkeys(self.group_attrs)
        for p in self.nodes:
            g = parent(p)
            while g and g not in out:
                out[g] = None
                g = parent(g)
        return [g for g in out if g]

    def children(self, group: str) -> list[ArrayNode]:
        g = norm_path(group)
        return [n for n in self.nodes.values() if parent(n.path) == g]

    def below(self, group: str) -> list[ArrayNode]:
        g = norm_path(group)
        if not g:
            return list(self.nodes.values())
        return [n for n in self.nodes.values() if n.path.startswith(g + "/")]

    def scale(self, node: ArrayNode, axis: int) -> ArrayNode | None:
        """Coordinate array of ``node``'s ``axis`` (dimension scale / coordinate variable)."""
        if node.scales is not None and axis < len(node.scales):
            p = node.scales[axis]
            return self.nodes.get(p) if p else None
        if node.dims is None or axis >= len(node.dims):
            return None
        dim = node.dims[axis]
        g = parent(node.path)
        while True:
            cand = self.nodes.get(f"{g}/{dim}" if g else dim)
            if cand is not None and cand.ndim == 1 and cand.shape[0] == node.shape[axis]:
                return None if cand.phantom else cand
            if not g:
                return None
            g = parent(g)

    def tree(self, limit: int = 40) -> str:
        """Compact listing ``/path  shape  dtype  [dims]`` used in :class:`MappingError` hints."""
        rows = []
        for n in self.nodes.values():
            if n.phantom:
                continue
            dims = f"  [{', '.join(n.dims)}]" if n.dims else ""
            rows.append(f"  /{n.path}  {n.shape}  {n.dtype}{dims}")
        extra = f"\n  ... and {len(rows) - limit} more arrays" if len(rows) > limit else ""
        root = ", ".join(list(self.group_attrs.get("", {}))[:12])
        head = f"  / attributes: {root}\n" if root else ""
        return head + "\n".join(rows[:limit]) + extra


# -- openers ---------------------------------------------------------------------------------

_NC4_PHANTOM = "This is a netCDF dimension but not a netCDF variable"


def _h5_attrs(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in obj.attrs:
        if k in ("DIMENSION_LIST", "REFERENCE_LIST"):
            continue
        try:
            out[k] = clean_attr(obj.attrs[k])
        except Exception:  # noqa: BLE001 - e.g. attribute types h5py cannot convert
            continue
    return out


def open_h5(path: Path | str) -> ArrayStore:
    """Index an HDF5 file (NetCDF-4 files included: dimension scales give ``dims``)."""
    import h5py

    path = Path(path)
    try:
        f: Any = h5_open(path)
    except OSError as e:
        raise ConnectorError(
            f"{path} is not a readable HDF5 file ({e})",
            hint="the file is truncated or not HDF5; `h5ls -r <file>` shows whether it opens",
        ) from e
    store = ArrayStore(path, "hdf5")
    store.keep.append(f)
    try:
        store.group_attrs[""] = _h5_attrs(f)
        if "_NCProperties" in f.attrs or any(k.startswith("_NC") for k in f.attrs):
            store.format = "netcdf4"

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                store.group_attrs[norm_path(name)] = _h5_attrs(obj)
                return
            if not isinstance(obj, h5py.Dataset):
                return
            node = ArrayNode(name, obj.shape or (), obj.dtype, obj, lambda o=obj: _h5_attrs(o))
            cls = obj.attrs.get("CLASS")
            if cls is not None and _text(cls) == "DIMENSION_SCALE":
                node.phantom = _text(obj.attrs.get("NAME", b"")).startswith(_NC4_PHANTOM)
                if obj.ndim == 1:
                    node.dims = [basename(name)]
                    node.scales = [None if node.phantom else norm_path(name)]
            if "DIMENSION_LIST" in obj.attrs:
                node.dims, node.scales = _h5_dimension_list(f, obj)
            store.add(node)

        f.visititems(visit)
    except (OSError, RuntimeError, KeyError) as e:
        raise ConnectorError(
            f"{path}: HDF5 structure could not be read ({e})",
            hint="the file is corrupt or was truncated while being written",
        ) from e
    return store


def _h5_dimension_list(f: Any, ds: Any) -> tuple[list[str], list[str | None]]:
    names: list[str] = []
    scales: list[str | None] = []
    refs = ds.attrs["DIMENSION_LIST"]
    labels = [_text(x) for x in np.atleast_1d(ds.attrs.get("DIMENSION_LABELS", []))]
    for i in range(ds.ndim):
        name, scale = (labels[i] if i < len(labels) and labels[i] else f"dim_{i}"), None
        try:
            target = f[refs[i][0]]
            p = norm_path(target.name)
            name = basename(p)
            if not _text(target.attrs.get("NAME", b"")).startswith(_NC4_PHANTOM):
                scale = p
        except Exception:  # noqa: BLE001 - axis without an attached scale
            pass
        names.append(name)
        scales.append(scale)
    return names, scales


def _first_bytes(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def open_nc3(path: Path | str) -> ArrayStore:
    """Index a NetCDF-3 classic / 64-bit-offset file. ``scipy.io.netcdf_file`` parses the
    header; the variables are then exposed as views on a read-only memory map, so slicing a
    record variable touches only the pages of that record."""
    from scipy.io import netcdf_file

    path = Path(path)
    if path.is_file() and _first_bytes(path, 4) == b"CDF\x05":
        raise ConnectorError(
            f"{path} is NetCDF CDF-5 (64-bit data), which the built-in reader does not support",
            hint="convert with `nccopy -k nc4 in.nc out.nc` or pass options={'engine': 'xarray'}",
        )
    try:
        nc: Any = netcdf_file(str(path), "r", mmap=True, maskandscale=False)
    except Exception as e:  # noqa: BLE001 - scipy raises TypeError/ValueError/OSError
        raise ConnectorError(
            f"{path} could not be read as NetCDF-3 ({e})",
            hint="the file is truncated or not NetCDF-3 classic",
        ) from e
    store = ArrayStore(path, "netcdf3")
    try:
        store.group_attrs[""] = {k: clean_attr(v) for k, v in nc._attributes.items()}
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        buf = getattr(nc, "_mm_buf", None)
        base = buf.__array_interface__["data"][0] if buf is not None else None
        for name, var in nc.variables.items():
            data = var.data
            attrs = {k: clean_attr(v) for k, v in var._attributes.items()}
            dims = [str(d) for d in var.dimensions]
            if base is not None and data.size:
                # re-create the view on our own map so scipy's file can be closed cleanly
                off = data.__array_interface__["data"][0] - base
                view = np.ndarray(
                    shape=data.shape,
                    dtype=data.dtype,
                    buffer=mm,
                    offset=off,
                    strides=data.strides,
                )
            else:
                view = np.array(data)
            store.add(ArrayNode(name, view.shape, view.dtype, view, attrs, dims))
        store.keep.append(mm)
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        raise ConnectorError(
            f"{path}: NetCDF-3 variables could not be mapped ({e})",
            hint="the file is truncated; re-download or re-export it",
        ) from e
    finally:
        # scipy refuses to unmap while anything still references its buffer
        var = data = buf = None  # noqa: F841
        nc.variables = {}
        try:
            nc.close()
        except Exception:  # noqa: BLE001
            pass
    return store


def open_zarr(path: Path | str) -> ArrayStore:
    """Index a Zarr v2 or v3 store (directory, or ``.zip``)."""
    import zarr

    path = Path(path)
    try:
        if path.is_file() and path.suffix.lower() == ".zip":
            from zarr.storage import ZipStore

            root: Any = zarr.open_group(ZipStore(str(path), mode="r"), mode="r")
        else:
            root = zarr.open_group(str(path), mode="r")
    except Exception as e:  # noqa: BLE001 - zarr raises many unrelated exception types
        try:
            arr: Any = zarr.open_array(str(path), mode="r")
        except Exception:  # noqa: BLE001
            raise ConnectorError(
                f"{path} is not a readable Zarr store ({type(e).__name__}: {e})",
                hint="expected a directory with zarr.json (v3) or .zgroup/.zarray (v2)",
            ) from e
        store = ArrayStore(path, "zarr")
        store.add(_zarr_node(path.stem or "data", arr))
        return store
    store = ArrayStore(path, "zarr")
    store.keep.append(root)
    try:
        store.group_attrs[""] = {k: clean_attr(v) for k, v in dict(root.attrs).items()}
        for name, obj in root.members(max_depth=None):
            if hasattr(obj, "shape"):
                store.add(_zarr_node(name, obj))
            else:
                store.group_attrs[norm_path(name)] = {
                    k: clean_attr(v) for k, v in dict(obj.attrs).items()
                }
    except Exception as e:  # noqa: BLE001
        raise ConnectorError(
            f"{path}: Zarr metadata could not be read ({type(e).__name__}: {e})",
            hint="the store is incomplete (a zarr.json / .zarray file is missing or corrupt)",
        ) from e
    return store


def _zarr_node(name: str, arr: Any) -> ArrayNode:
    attrs = {k: clean_attr(v) for k, v in dict(arr.attrs).items()}
    dims: list[str] | None = None
    raw = attrs.get("_ARRAY_DIMENSIONS")
    if raw is None:
        raw = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if raw is not None and len(raw) == len(arr.shape) and all(d for d in raw):
        dims = [str(d) for d in raw]
    return ArrayNode(name, arr.shape, arr.dtype, arr, attrs, dims)


class _XarrayHandle:
    def __init__(self, var: Any) -> None:
        self.var = var

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self.var[key].values)


def open_xarray(path: Path | str, options: Mapping[str, Any]) -> ArrayStore:
    """Index anything xarray can open (OPeNDAP URLs, GRIB via cfgrib, multi-file datasets).
    Decoding is switched off so fill values / scaling / time units go through the same code
    as the built-in readers."""
    try:
        import importlib

        xr: Any = importlib.import_module("xarray")
    except ImportError as e:
        raise MissingDependencyError(
            "xarray",
            extra="frontier-connector-arrays[xarray]",
            purpose="use options={'engine': 'xarray'}",
            license="Apache-2.0",
        ) from e
    kw: dict[str, Any] = {"decode_cf": False, "mask_and_scale": False, "decode_times": False}
    kw.update(dict(options.get("engine_kwargs") or {}))
    target: Any = options.get("url") or options.get("files") or str(path)
    try:
        if isinstance(target, list | tuple) or any(c in str(target) for c in "*?["):
            ds: Any = xr.open_mfdataset(target, **kw)
        else:
            ds = xr.open_dataset(target, **kw)
    except Exception as e:  # noqa: BLE001 - backend specific errors
        raise ConnectorError(
            f"xarray could not open {target} ({type(e).__name__}: {e})",
            hint="check engine_kwargs (e.g. {'engine': 'cfgrib'}) and that the backend package "
            "is installed; multi-file globs need dask",
        ) from e
    store = ArrayStore(path, "xarray")
    store.keep.append(ds)
    store.group_attrs[""] = {str(k): clean_attr(v) for k, v in ds.attrs.items()}
    for name, var in ds.variables.items():
        attrs = {str(k): clean_attr(v) for k, v in var.attrs.items()}
        dims = [str(d) for d in var.dims]
        store.add(ArrayNode(str(name), var.shape, var.dtype, _XarrayHandle(var), attrs, dims))
    return store


# -- lazy field array ------------------------------------------------------------------------


def _progression(idx: np.ndarray) -> slice | None:
    """``idx`` as a slice when it is an increasing arithmetic progression."""
    if idx.size == 0:
        return slice(0, 0)
    if idx.size == 1:
        return slice(int(idx[0]), int(idx[0]) + 1)
    d = np.diff(idx)
    if d[0] > 0 and np.all(d == d[0]):
        return slice(int(idx[0]), int(idx[-1]) + 1, int(d[0]))
    return None


class _Decoder:
    """CF packing of one source array: fill values -> NaN, ``* scale_factor + add_offset``."""

    def __init__(self, node: ArrayNode) -> None:
        a = node.attrs
        self.fills = [
            v for v in (_number(a.get("_FillValue")), _number(a.get("missing_value")))
            if v is not None and not np.isnan(v)
        ]  # fmt: skip
        self.scale = _number(a.get("scale_factor"))
        self.offset = _number(a.get("add_offset"))
        self.active = bool(self.fills) or self.scale is not None or self.offset is not None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if not self.active:
            return raw
        mask = None
        for fv in self.fills:
            m = raw == np.asarray(fv).astype(raw.dtype) if raw.dtype.kind in "iub" else raw == fv
            mask = m if mask is None else (mask | m)
        packed = self.scale is not None or self.offset is not None
        out = raw.astype(np.float64 if packed else np.float32)
        if self.scale is not None:
            out *= self.scale
        if self.offset is not None:
            out += self.offset
        out = out.astype(np.float32, copy=False)
        if mask is not None and mask.any():
            out[mask] = np.nan
        return out


class LazyArray:
    """Array-like view of one field in contract axis order; reads on ``__getitem__`` only.

    ``sources`` are the on-disk arrays (several = scalars stacked into a vector), ``roles``
    names every on-disk axis (``time``, ``sample``, ``x``/``y``/``z``, ``point``/``cell``,
    ``component`` or ``drop``), ``out`` is the contract order of the non-component axes.
    """

    def __init__(
        self,
        sources: Sequence[ArrayNode],
        roles: Sequence[str],
        out: Sequence[str],
        *,
        fixed: Mapping[int, int] | None = None,
        time_index: Sequence[int] | np.ndarray | None = None,
        entity_index: np.ndarray | None = None,
        decode: bool = True,
    ) -> None:
        self.sources = list(sources)
        self.roles = list(roles)
        self.out = list(out)
        self.fixed = {int(k): int(v) for k, v in (fixed or {}).items()}
        self.time_index = None if time_index is None else np.asarray(time_index, dtype=np.int64)
        self.entity_index = entity_index
        self.decoders = [_Decoder(n) if decode else None for n in self.sources]
        first = self.sources[0]
        self._time_axis = self.roles.index("time") if "time" in self.roles else None
        self.time_first = bool(self.out) and self.out[0] == "time"
        sizes = []
        for r in self.out:
            n = first.shape[self.roles.index(r)]
            if r == "time" and self.time_index is not None:
                n = len(self.time_index)
            if r in ("point", "cell") and entity_index is not None:
                n = len(entity_index)
            sizes.append(int(n))
        comps = 0
        for s in self.sources:
            c = [s.shape[i] for i, r in enumerate(self.roles) if r == "component"]
            comps += int(np.prod(c)) if c else 1
        self.shape: tuple[int, ...] = (*sizes, comps)
        packed = any(d is not None and d.active for d in self.decoders)
        kinds = {s.dtype.kind for s in self.sources}
        if packed or "f" in kinds or len({s.dtype for s in self.sources}) > 1:
            self.dtype = np.dtype(np.float32)
        elif kinds == {"b"}:
            self.dtype = np.dtype(np.uint8)
        else:
            self.dtype = first.dtype.newbyteorder("=")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        names = ", ".join("/" + s.path for s in self.sources)
        return f"LazyArray({names} -> {[*self.out, 'component']} {self.shape})"

    def _read_source(self, i: int, lead: np.ndarray | None) -> np.ndarray:
        node = self.sources[i]
        key: list[Any] = []
        loop: tuple[int, np.ndarray] | None = None
        kept: list[str] = []
        for ax, role in enumerate(self.roles):
            if ax in self.fixed:
                key.append(self.fixed[ax])
                continue
            if ax == self._time_axis and (lead is not None or self.time_index is not None):
                idx = self.time_index if self.time_index is not None else None
                if lead is not None:
                    idx = lead if idx is None else idx[lead]
                assert idx is not None
                sl = _progression(np.asarray(idx))
                if sl is None:
                    loop = (len(kept), np.asarray(idx))
                    key.append(None)
                else:
                    key.append(sl)
            else:
                key.append(slice(None))
            kept.append(role)
        if loop is None:
            raw = np.asarray(node.handle[tuple(key)])
        else:
            # arbitrary step lists are read one step at a time (h5py/zarr basic indexing only)
            pos, idx = loop
            ax = key.index(None)
            parts = []
            for j in idx:
                key[ax] = int(j)
                parts.append(np.asarray(node.handle[tuple(key)]))
            raw = np.stack(parts, axis=pos)
        if not raw.dtype.isnative:
            raw = raw.astype(raw.dtype.newbyteorder("="))
        perm = [kept.index(r) for r in self.out] + [
            j for j, r in enumerate(kept) if r == "component"
        ]
        raw = raw.transpose(perm)
        raw = raw.reshape(*raw.shape[: len(self.out)], -1)
        dec = self.decoders[i]
        return dec(raw) if dec is not None else raw

    def _read(self, lead: np.ndarray | None) -> np.ndarray:
        """Read the steps ``lead`` (indices on the leading output axis; ``None`` = all)."""
        parts = [self._read_source(i, lead) for i in range(len(self.sources))]
        out = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        if self.entity_index is not None:
            out = np.take(out, self.entity_index, axis=1 if self.time_first else 0)
        if out.dtype != self.dtype:
            out = out.astype(self.dtype)
        return np.ascontiguousarray(out)

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        if not self.time_first or not key or key[0] is Ellipsis:
            return self._read(None)[key]
        k0, rest = key[0], key[1:]
        n = self.shape[0]
        squeeze = False
        if isinstance(k0, slice):
            idx = np.arange(n)[k0]
        elif isinstance(k0, int | np.integer):
            if not -n <= int(k0) < n:
                raise IndexError(f"index {k0} is out of bounds for axis 0 with size {n}")
            idx = np.array([int(k0) % n])
            squeeze = True
        else:
            idx = np.asarray(k0)
            idx = np.nonzero(idx)[0] if idx.dtype.kind == "b" else idx.astype(np.int64) % n
        out = self._read(idx)
        if squeeze:
            out = out[0]
            return out[rest] if rest else out
        return out[(slice(None), *rest)] if rest else out

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = self._read(None)
        return out.astype(dtype) if dtype is not None else out


# -- layouts ---------------------------------------------------------------------------------


@dataclass
class FieldPlan:
    """How one output field is assembled from on-disk arrays."""

    name: str
    sources: list[ArrayNode]
    #: role of every on-disk axis: time | sample | x | y | z | point | cell | component | drop
    roles: list[str]
    #: on-disk axis -> fixed index (axes that are not part of the contract)
    fixed: dict[int, int] = field(default_factory=dict)
    location: str = "grid"
    units: str = ""
    description: str = ""
    component_names: list[str] | None = None

    @property
    def time_dependent(self) -> bool:
        return "time" in self.roles

    @property
    def sample_axis(self) -> int | None:
        return self.roles.index("sample") if "sample" in self.roles else None

    def size(self, role: str) -> int | None:
        return self.sources[0].shape[self.roles.index(role)] if role in self.roles else None

    @property
    def components(self) -> int:
        total = 0
        for s in self.sources:
            c = [s.shape[i] for i, r in enumerate(self.roles) if r == "component"]
            total += int(np.prod(c)) if c else 1
        return total

    def space(self) -> list[str]:
        return [r for r in (*GRID_AXES, "point", "cell") if r in self.roles]

    def lazy(
        self,
        *,
        sample: int | None = None,
        time_index: Sequence[int] | None = None,
        entity_index: np.ndarray | None = None,
    ) -> LazyArray:
        fixed = dict(self.fixed)
        if self.sample_axis is not None:
            fixed[self.sample_axis] = int(sample or 0)
        out = (["time"] if self.time_dependent else []) + self.space()
        return LazyArray(
            self.sources,
            self.roles,
            out,
            fixed=fixed,
            time_index=time_index if self.time_dependent else None,
            entity_index=entity_index if self.location == "cell" else None,
        )


@dataclass
class Layout:
    """One zone of a store, fully resolved: geometry sources, time axis and field plans."""

    zone: str
    kind: str
    store: ArrayStore
    axes: list[str] = field(default_factory=list)
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    axis_names: dict[str, str] = field(default_factory=dict)
    points: list[ArrayNode] = field(default_factory=list)
    cells: ArrayNode | None = None
    cell_type: int | str | None = None
    cell_types: ArrayNode | None = None
    index_base: int | None = None
    #: ``(T,)`` or ``(S, T)`` (per-sample time axes)
    times: np.ndarray | None = None
    time_units: str = "s"
    time_meta: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, FieldPlan] = field(default_factory=dict)
    static: dict[str, FieldPlan] = field(default_factory=dict)
    n_samples: int | None = None
    sample_ids: list[str] | None = None
    #: group-per-sample stores: one group path per sample, resolved with ``mapping``
    groups: list[str] | None = None
    mapping: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    domain: str = "other"
    warnings: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int | None:
        if not self.points:
            return None
        p = self.points[0]
        return p.shape[0] if (len(self.points) > 1 or p.ndim <= 2) else p.shape[-2]

    @property
    def point_dim(self) -> int:
        if not self.points:
            return len(self.axes)
        return len(self.points) if len(self.points) > 1 else int(self.points[0].shape[-1])

    @property
    def n_cells(self) -> int | None:
        return None if self.cells is None else self.cells.shape[0]

    def times_for(self, sample: int | None) -> np.ndarray | None:
        if self.times is None:
            return None
        if self.times.ndim == 2:
            return self.times[int(sample or 0)]
        return self.times

    def uniform_cell_type(self) -> int | None:
        """VTK type shared by every cell when that is known without reading connectivity."""
        if self.cells is None or self.cell_types is not None:
            return None
        if isinstance(self.cell_type, int):
            return self.cell_type
        if isinstance(self.cell_type, str):
            return CELL_TYPES.get(self.cell_type.lower()) if self.cell_type != "polygon" else None
        if self.cells.ndim != 2:
            return None
        table = _WIDTH_TYPES_2D if self.point_dim == 2 else _WIDTH_TYPES_3D
        return table.get(self.cells.shape[1])

    def load_geometry(self) -> dict[str, Any]:
        """Read points and connectivity (one copy in memory, like one time step)."""
        out: dict[str, Any] = {"points": None, "cells": None, "cell_types": None, "faces": None}
        out["parent"] = None
        if not self.points:
            return out
        if len(self.points) > 1:
            pts = np.stack([np.asarray(_decoded(n), dtype=np.float64) for n in self.points], 1)
        else:
            node = self.points[0]
            if node.ndim == 3:
                pts = np.asarray(node.handle[0], dtype=np.float64)
            else:
                pts = np.asarray(_decoded(node), dtype=np.float64)
            if pts.ndim == 1:
                pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] > 3:
            raise MappingError(
                f"points /{self.points[0].path} must be (N, 1..3), got {pts.shape}",
                hint="map 'points' to an (N, D) array or to a list of 1-D coordinate arrays",
            )
        out["points"] = pts
        if self.cells is None:
            return out
        raw = np.asarray(self.cells.read())
        if raw.dtype.kind not in "iu":
            if raw.dtype.kind != "f" or not np.allclose(raw, np.round(raw), equal_nan=True):
                raise MappingError(
                    f"cells /{self.cells.path} has dtype {raw.dtype}; integer indices expected",
                    hint="map 'cells' to the connectivity array (point indices per cell)",
                )
            raw = np.where(np.isnan(raw), -1, raw)
        fill = _number(self.cells.attrs.get("_FillValue"))
        cells = raw.astype(np.int64)
        ut = self.uniform_cell_type()
        if cells.ndim == 1:
            width = {v: k for k, v in _WIDTH_TYPES_3D.items()} | {9: 4}
            if ut is None or ut not in width or cells.size % width[ut]:
                raise MappingError(
                    f"cells /{self.cells.path} is 1-D; give 'cell_type' so it can be reshaped",
                    hint='e.g. "cell_type": "tetra" for a flat list of 4 point ids per cell',
                )
            cells = cells.reshape(-1, width[ut])
        if cells.ndim != 2:
            raise MappingError(
                f"cells /{self.cells.path} must be (K, P), got {cells.shape}",
                hint="polyhedral / ragged connectivity is not supported by the generic mapping",
            )
        pad = cells < 0
        if fill is not None:
            pad |= cells == int(fill)
        base = self.index_base
        if base is None:
            start = _number(self.cells.attrs.get("start_index"))
            valid = cells[~pad]
            if start is not None:
                base = int(start)
            elif valid.size and int(valid.min()) >= 1 and int(valid.max()) == len(pts):
                base = 1
                self.warnings.append(
                    f"/{self.cells.path}: indices run 1..{len(pts)}, read as 1-based "
                    '(set "index_base" in the mapping to override)'
                )
            else:
                base = 0
        cells = np.where(pad, -1, cells - base)
        if cells.size and (int(cells.max()) >= len(pts) or int(cells.min()) < -1):
            raise MappingError(
                f"cells /{self.cells.path} reference points outside 0..{len(pts) - 1} "
                f"(index_base {base})",
                hint='set "index_base" (0 or 1) and check that "points" is the matching array',
            )
        counts = (cells >= 0).sum(axis=1)
        if self.cell_types is not None:
            ctypes = np.asarray(self.cell_types.read()).reshape(-1).astype(np.uint8)
        elif ut is not None:
            ctypes = np.full(len(cells), ut, dtype=np.uint8)
        else:
            flat = self.cell_type == "polygon" or pts.shape[1] == 2
            table = _WIDTH_TYPES_2D if flat else _WIDTH_TYPES_3D
            ctypes = np.full(len(cells), 7 if flat else 0, dtype=np.uint8)
            for n_nodes, t in table.items():
                ctypes[counts == n_nodes] = t
            if (ctypes == 0).any():
                raise MappingError(
                    f"cannot infer the cell type of /{self.cells.path} rows with "
                    f"{sorted(set(counts[ctypes == 0].tolist()))} nodes",
                    hint='give "cell_type" (e.g. "hexahedron") or "cell_types" (path of VTK ids)',
                )
        if len(ctypes) != len(cells):
            raise MappingError("'cell_types' must have one VTK type id per cell")
        if self.kind == "surface":
            if not meshutil.all_surface(ctypes):
                raise MappingError(
                    f"kind 'surface' needs 2-D cells; /{self.cells.path} holds "
                    f"{meshutil.census(ctypes)}",
                    hint='use "kind": "mesh" for volume cells',
                )
            faces, par = meshutil.triangulate(cells, ctypes)
            if pts.shape[1] < 3:
                pts = np.concatenate([pts, np.zeros((len(pts), 3 - pts.shape[1]))], axis=1)
                out["points"] = pts
            out["faces"], out["parent"] = faces, par
        else:
            out["cells"], out["cell_types"] = cells, ctypes
        return out


def _decoded(node: ArrayNode) -> np.ndarray:
    raw = np.asarray(node.read())
    if not raw.dtype.isnative:
        raw = raw.astype(raw.dtype.newbyteorder("="))
    return _Decoder(node)(raw)


# -- time axes -------------------------------------------------------------------------------

_SECONDS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}  # fmt: skip
_SINCE = re.compile(r"^\s*([A-Za-z_]+)\s+since\s+(.+?)\s*$", re.IGNORECASE)


def decode_time(values: np.ndarray, attrs: Mapping[str, Any]) -> tuple[np.ndarray, str, dict]:
    """CF time coordinate -> ``(numeric values, units, metadata)``. Values stay relative to the
    reference date (kept in ``metadata["time_reference"]``); known units become seconds."""
    vals = np.asarray(values, dtype=np.float64)
    units = _text(attrs.get("units", "")).strip()
    meta: dict[str, Any] = {}
    if attrs.get("calendar"):
        meta["calendar"] = _text(attrs["calendar"])
    if not units:
        return vals, "s", meta
    m = _SINCE.match(units)
    word = (m.group(1) if m else units).lower()
    if m:
        meta["time_reference"] = m.group(2)
        meta["time_source_units"] = units
    if word in _SECONDS:
        return vals * _SECONDS[word], "s", meta
    return vals, (m.group(1) if m else units), meta


# -- mapping resolution ----------------------------------------------------------------------

MAPPING_KEYS = (
    "kind", "group", "time", "coords", "points", "cells", "cell_type", "cell_types",
    "index_base", "fields", "static", "samples", "parameters", "units", "domain", "solver",
    "solver_version", "physics", "boundary_conditions", "metadata", "zones",
)  # fmt: skip
_POINT_WORDS = ("point", "points", "node", "nodes", "vertex", "vertices")
_CELL_WORDS = ("cell", "cells", "element", "elements", "face", "faces")
_COMP_WORDS = ("component", "components", "comp", "c", "channel", "channels")


def load_mapping(value: Any) -> dict[str, Any] | None:
    """``options["mapping"]``: a dict, or a path to a JSON file."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    p = Path(str(value))
    if not p.is_file():
        raise MappingError(f"mapping file {p} not found", hint="pass a dict or a JSON file path")
    try:
        out = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise MappingError(f"mapping file {p} is not valid JSON: {e}") from e
    if not isinstance(out, dict):
        raise MappingError(f"mapping file {p} must contain a JSON object")
    return out


class _Resolver:
    def __init__(self, store: ArrayStore, m: Mapping[str, Any], zone: str, prefix: str) -> None:
        self.store = store
        self.m = m
        self.prefix = norm_path(prefix)
        unknown = [k for k in m if k not in MAPPING_KEYS and not str(k).startswith("_")]
        if unknown:
            raise MappingError(
                f"unknown mapping key(s): {', '.join(map(str, unknown))}",
                hint="valid keys: " + ", ".join(MAPPING_KEYS),
            )
        samples = m.get("samples") or {}
        self.samples: dict[str, Any] = (
            {"axis": samples} if isinstance(samples, str) else dict(samples)
        )
        self.sample_names = {s.lower() for s in SAMPLE_NAMES}
        if self.samples.get("axis"):
            self.sample_names.add(str(self.samples["axis"]).lower())
        self.used: set[str] = set()
        self.alias: dict[str, str] = {}
        self.time_names: set[str] = {"time", "t"}
        self.pending_coords: dict[str, Mapping[str, Any]] = {}
        self.pending_time: tuple[float, float] | None = None
        self.lay = Layout(zone=zone, kind="grid", store=store)

    def node(self, path: Any, what: str) -> ArrayNode:
        if not isinstance(path, str):
            raise MappingError(
                f"mapping '{what}' must be an array path, got {path!r}",
                hint="paths look like '/group/array'",
            )
        n = self.store.node(path, self.prefix, what=what)
        self.used.add(n.path)
        return n

    # geometry ------------------------------------------------------------------------------
    def geometry(self) -> None:
        m, lay = self.m, self.lay
        spec = m.get("coords") or {}
        if not isinstance(spec, Mapping):
            raise MappingError("mapping 'coords' must be an object {x: ..., y: ..., z: ...}")
        bad = [k for k in spec if k not in GRID_AXES]
        if bad:
            raise MappingError(
                f"mapping 'coords' has unknown axis {bad[0]!r}", hint="use x, then y, then z"
            )
        lay.axes = [a for a in GRID_AXES if a in spec]
        if lay.axes != list(GRID_AXES[: len(lay.axes)]):
            raise MappingError(
                f"grid axes must be x, then y, then z; got {lay.axes}",
                hint="name the first axis x even when it is a latitude or a level",
            )
        for ax in lay.axes:
            c = spec[ax]
            if isinstance(c, str):
                node = self.node(c, f"coords.{ax}")
                arr = np.squeeze(_decoded(node).astype(np.float64))
                if arr.ndim != 1:
                    raise MappingError(
                        f"coords.{ax} /{node.path} has shape {node.shape}; a 1-D axis is needed",
                        hint="curvilinear grids: use kind 'point_cloud'/'mesh' with 'points', or "
                        'an index axis {"start": 0, "step": 1} and map the 2-D coordinate arrays '
                        "under 'static'",
                    )
                lay.coords[ax] = arr
                lay.axis_names[ax] = node.name
                self.alias[node.name.lower()] = ax
                if node.dims:
                    self.alias[node.dims[0].lower()] = ax
            elif isinstance(c, Mapping):
                self.pending_coords[ax] = c
                if c.get("name"):
                    self.alias[str(c["name"]).lower()] = ax
                    lay.axis_names[ax] = str(c["name"])
            else:
                lay.coords[ax] = np.asarray(c, dtype=np.float64).reshape(-1)
        pts = m.get("points")
        if pts is not None:
            lay.points = [self.node(p, "points") for p in ([pts] if isinstance(pts, str) else pts)]
            if len(lay.points) > 1 and any(p.ndim != 1 for p in lay.points):
                raise MappingError(
                    "a list under 'points' must name 1-D coordinate arrays (x, y[, z])"
                )
        if m.get("cells") is not None:
            lay.cells = self.node(m["cells"], "cells")
        if m.get("cell_types") is not None:
            lay.cell_types = self.node(m["cell_types"], "cell_types")
        ct = m.get("cell_type")
        if isinstance(ct, str) and ct.lower() != "polygon" and ct.lower() not in CELL_TYPES:
            raise MappingError(
                f"unknown cell_type {ct!r}",
                hint="one of: polygon, " + ", ".join(sorted(CELL_TYPES)) + ", or a VTK type id",
            )
        lay.cell_type = ct.lower() if isinstance(ct, str) else ct
        if m.get("index_base") is not None:
            lay.index_base = int(m["index_base"])
        kind = m.get("kind")
        if kind is None:
            kind = "grid" if lay.axes else ("mesh" if lay.cells is not None else "point_cloud")
        if kind not in ("grid", "mesh", "surface", "point_cloud"):
            raise MappingError(
                f"unknown kind {kind!r}", hint="one of grid | mesh | surface | point_cloud"
            )
        lay.kind = kind
        if kind == "grid" and not lay.axes:
            raise MappingError(
                "a grid mapping needs 'coords'",
                hint='e.g. "coords": {"x": "/x", "y": "/y"} or {"x": {"start": 0, "step": 0.1}}',
            )
        if kind != "grid" and not lay.points:
            raise MappingError(
                f"kind {kind!r} needs 'points'", hint='e.g. "points": "/mesh/coordinates"'
            )
        if kind in ("mesh", "surface") and lay.cells is None:
            raise MappingError(f"kind {kind!r} needs 'cells'", hint='e.g. "cells": "/mesh/cells"')

    # time ----------------------------------------------------------------------------------
    def time(self) -> None:
        spec, lay = self.m.get("time"), self.lay
        if spec is None:
            return
        units = None
        if isinstance(spec, Mapping) and "path" not in spec:
            units = spec.get("units")
            if "values" in spec:
                lay.times = np.asarray(spec["values"], dtype=np.float64).reshape(-1)
            elif "attr" in spec:
                dt = _number(self.store.attr(str(spec["attr"]), self.prefix))
                if dt is None:
                    raise MappingError(f"time attribute {spec['attr']!r} is not a number")
                self.pending_time = (float(spec.get("start", 0.0)), dt)
            elif "dt" in spec or "start" in spec:
                self.pending_time = (float(spec.get("start", 0.0)), float(spec.get("dt", 1.0)))
            else:
                raise MappingError(
                    "mapping 'time' object needs 'path', 'values', 'attr' or 'start'/'dt'",
                    hint='e.g. "time": "/time" or {"start": 0, "dt": 0.01}',
                )
            for key in ("name", "dim"):
                if spec.get(key):
                    self.time_names.add(str(spec[key]).lower())
        elif isinstance(spec, str | Mapping):
            path = spec if isinstance(spec, str) else spec["path"]
            node = self.node(path, "time")
            if node.ndim not in (1, 2) or not node.numeric:
                raise MappingError(
                    f"time /{node.path} has shape {node.shape} dtype {node.dtype}",
                    hint='map \'time\' to a 1-D numeric array, or use {"start": 0, "dt": ...}',
                )
            vals, lay.time_units, lay.time_meta = decode_time(node.read(), node.attrs)
            lay.times = vals
            self.time_names.add(node.name.lower())
            if node.dims:
                self.time_names.add(node.dims[-1].lower())
            if isinstance(spec, Mapping):
                units = spec.get("units")
        else:
            lay.times = np.asarray(spec, dtype=np.float64).reshape(-1)
        if units:
            lay.time_units = str(units)

    # fields --------------------------------------------------------------------------------
    def role_of(self, name: str) -> str:
        n = str(name).lower()
        if n in self.time_names:
            return "time"
        if n in self.sample_names:
            return "sample"
        if n in GRID_AXES:
            return n
        if n in self.alias:
            return self.alias[n]
        if n in _POINT_WORDS:
            return "point"
        if n in _CELL_WORDS:
            return "cell"
        if n in _COMP_WORDS:
            return "component"
        return "drop"

    def _space_ok(self, roles: list[str]) -> str | None:
        """``None`` when ``roles`` covers the zone's space axes exactly once, else the reason."""
        lay = self.lay
        want = lay.axes if lay.kind == "grid" else None
        if want is not None:
            for ax in want:
                if roles.count(ax) != 1:
                    return f"axis '{ax}' appears {roles.count(ax)} times"
            extra = [r for r in roles if r in GRID_AXES and r not in want]
            return f"axis '{extra[0]}' has no coords entry" if extra else None
        ent = [r for r in roles if r in ("point", "cell")]
        if len(ent) != 1:
            return f"needs exactly one 'point' or 'cell' axis, has {len(ent)}"
        if ent[0] == "cell" and lay.cells is None:
            return "a 'cell' axis needs 'cells' in the mapping"
        if any(r in GRID_AXES for r in roles):
            return "x/y/z axes are only valid for kind 'grid'"
        return None

    def _sizes_ok(self, shape: tuple[int, ...], roles: list[str]) -> str | None:
        lay = self.lay
        for size, role in zip(shape, roles, strict=True):
            if role in lay.coords and size != len(lay.coords[role]):
                return (
                    f"axis '{role}' has {size} entries, coords.{role} has {len(lay.coords[role])}"
                )
            if role == "point" and lay.n_points is not None and size != lay.n_points:
                return f"'point' axis has {size} entries, points has {lay.n_points}"
            if role == "cell" and lay.n_cells is not None and size != lay.n_cells:
                return f"'cell' axis has {size} entries, cells has {lay.n_cells}"
            if role == "time" and lay.times is not None:
                t = lay.times.shape[-1]
                if size != t and size + 1 != t:
                    return f"'time' axis has {size} entries, time has {t}"
        for role in ("time", "sample"):
            if roles.count(role) > 1:
                return f"axis '{role}' appears {roles.count(role)} times"
        return None

    def _from_names(
        self, node: ArrayNode, names: Sequence[Any], select: Mapping[str, Any], strict: bool
    ) -> tuple[list[str], dict[int, int]] | str:
        if len(names) != node.ndim:
            return f"{len(names)} dims named for an array of shape {node.shape}"
        roles: list[str] = []
        fixed: dict[int, int] = {}
        for i, (name, size) in enumerate(zip(names, node.shape, strict=True)):
            role = self.role_of(str(name))
            if role == "drop":
                if str(name) in select:
                    fixed[i] = int(select[str(name)]) % max(size, 1)
                elif size == 1:
                    fixed[i] = 0
                elif not strict and i == node.ndim - 1 and size <= MAX_COMPONENTS:
                    role = "component"
                else:
                    return (
                        f"axis {name!r} (size {size}) is not a contract axis; add "
                        f'"select": {{"{name}": <index>}} or name it time/x/y/z/point/cell/'
                        "component/sample"
                    )
            roles.append(role)
        live = [r for i, r in enumerate(roles) if i not in fixed]
        why = self._space_ok(live) or self._sizes_ok(
            tuple(s for i, s in enumerate(node.shape) if i not in fixed), live
        )
        return why if why else (roles, fixed)

    def _positional(self, node: ArrayNode, location: str | None) -> tuple[list[str], str] | str:
        lay = self.lay
        if lay.kind == "grid":
            spaces = [list(lay.axes)]
        else:
            spaces = [[loc] for loc in ("point", "cell") if location in (None, loc)]
            if lay.cells is None:
                spaces = [["point"]]
        declared_t = self.m.get("time") is not None
        declared_s = bool(self.samples.get("axis"))
        why = "its rank fits no ([sample,] [time,] space..., [component]) layout"
        for s in [True, False] if declared_s else [False]:
            for t in [True, False] if declared_t else [False]:
                for c in (False, True):
                    for space in spaces:
                        roles = ["sample"] * s + ["time"] * t + space + ["component"] * c
                        if len(roles) != node.ndim:
                            continue
                        if c and node.shape[-1] > MAX_COMPONENTS and lay.kind != "grid":
                            continue
                        bad = self._sizes_ok(node.shape, roles)
                        if bad is None:
                            return roles, space[0]
                        why = bad
        return why

    def plan(self, name: str, spec: Any, *, strict: bool = True) -> FieldPlan | None:
        """Field plan for one mapping entry. ``strict=False`` (auto-collected arrays) returns
        ``None`` instead of raising when the array does not fit the zone."""
        if isinstance(spec, str):
            spec = {"path": spec}
        if not isinstance(spec, Mapping) or not (spec.get("path") or spec.get("paths")):
            raise MappingError(
                f"mapping field {name!r} needs 'path' or 'paths'",
                hint='e.g. {"path": "/fields/u", "dims": ["time", "x", "y", "component"]}',
            )
        paths = [spec["path"]] if spec.get("path") else list(spec["paths"])
        nodes = [self.node(p, f"field {name!r}") for p in paths]
        first = nodes[0]
        for n in nodes:
            if not n.numeric or n.ndim == 0:
                raise MappingError(
                    f"field {name!r}: /{n.path} (shape {n.shape}, dtype {n.dtype}) is not a "
                    "numeric array"
                )
            if n.shape != first.shape:
                raise MappingError(
                    f"field {name!r}: stacked arrays differ in shape "
                    f"(/{first.path} {first.shape} vs /{n.path} {n.shape})"
                )
        select = spec.get("select") or {}
        location = spec.get("location")
        got: Any
        if spec.get("dims"):
            got = self._from_names(first, list(spec["dims"]), select, True)
        else:
            got = "no dims"
            if first.dims:
                got = self._from_names(first, first.dims, select, False)
            if isinstance(got, str):
                pos = self._positional(first, location)
                got = pos if isinstance(pos, str) else (pos[0], {})
        if isinstance(got, str):
            if not strict:
                return None
            raise MappingError(
                f"field {name!r} (/{first.path}, shape {first.shape}): {got}",
                hint='name every axis of the array, e.g. "dims": '
                + json.dumps(self._example_dims(first.ndim)),
            )
        roles, fixed = got
        live = [r for i, r in enumerate(roles) if i not in fixed]
        if self.lay.kind == "grid":
            loc = "grid"
        else:
            loc = "cell" if "cell" in live else "point"
        a = first.attrs
        return FieldPlan(
            name=name,
            sources=nodes,
            roles=["drop" if i in fixed else r for i, r in enumerate(roles)],
            fixed=fixed,
            location=loc,
            units=str(spec.get("units") or _text(a.get("units", ""))),
            description=str(
                spec.get("description")
                or _text(a.get("long_name") or a.get("description") or a.get("standard_name"))
            ),
            component_names=list(spec["component_names"]) if spec.get("component_names") else None,
        )

    def _example_dims(self, ndim: int) -> list[str]:
        space = list(self.lay.axes) if self.lay.kind == "grid" else ["point"]
        dims = [*space, "component"]
        for lead in ("time", "sample"):
            if len(dims) < ndim:
                dims.insert(0, lead)
        return dims[:ndim] if len(dims) >= ndim else ["?"] * (ndim - len(dims)) + dims

    def fields(self) -> None:
        lay, m = self.lay, self.m
        for name, spec in (m.get("static") or {}).items():
            p = self.plan(str(name), spec)
            assert p is not None
            if p.time_dependent:
                raise MappingError(
                    f"static array {name!r} has a time axis", hint="list it under 'fields'"
                )
            lay.static[str(name)] = p
        spec = m.get("fields")
        if isinstance(spec, Mapping):
            for name, s in spec.items():
                p = self.plan(str(name), s)
                assert p is not None
                lay.fields[str(name)] = p
        elif spec is None or spec in ("auto", "*"):
            reserved = set(self.used)
            for node in self.store.below(self.prefix):
                if node.path in reserved or not node.numeric or node.ndim == 0:
                    continue
                p = self.plan(node.name, {"path": "/" + node.path}, strict=False)
                if p is not None and node.name not in lay.fields:
                    lay.fields[node.name] = p
            if not lay.fields:
                raise MappingError(
                    f"no array in {self.store.path.name} fits the mapped axes",
                    hint="list the arrays under 'fields' with explicit 'dims':\n"
                    + self.store.tree(),
                )
        else:
            raise MappingError("mapping 'fields' must be an object {name: spec}")

    # finish --------------------------------------------------------------------------------
    def finish(self) -> Layout:
        lay, m = self.lay, self.m
        plans = [*lay.fields.values(), *lay.static.values()]
        for ax, c in self.pending_coords.items():
            n = next((p.size(ax) for p in plans if p.size(ax) is not None), None)
            if n is None:
                raise MappingError(f"coords.{ax}: no field has an '{ax}' axis to size it")
            if "values" in c:
                lay.coords[ax] = np.asarray(c["values"], dtype=np.float64).reshape(-1)
            elif "linspace" in c:
                lo, hi = (float(v) for v in c["linspace"])
                lay.coords[ax] = np.linspace(lo, hi, n)
            else:
                step = float(c.get("step", 1.0))
                lay.coords[ax] = float(c.get("start", 0.0)) + step * np.arange(n)
        for p in plans:
            for ax in lay.axes:
                if p.size(ax) != len(lay.coords[ax]):
                    raise MappingError(
                        f"field {p.name!r}: axis '{ax}' has {p.size(ax)} entries, coords.{ax} "
                        f"has {len(lay.coords[ax])}"
                    )
        steps = {p.size("time") for p in lay.fields.values() if p.time_dependent}
        if len(steps) > 1:
            raise MappingError(
                f"fields disagree on the number of time steps: {sorted(s or 0 for s in steps)}",
                hint="map fields with different time axes as separate 'zones'",
            )
        n_t = next(iter(steps)) if steps else None
        if n_t is None:
            lay.times = None
        elif self.pending_time is not None:
            lay.times = self.pending_time[0] + self.pending_time[1] * np.arange(n_t)
        elif lay.times is None:
            lay.times = np.arange(n_t, dtype=np.float64)
            lay.time_meta["time_quantity"] = "iteration"
            lay.warnings.append(
                f"zone {lay.zone!r}: no time coordinate, step indices are used as times "
                '(set "time": {"start": ..., "dt": ...} in the mapping)'
            )
        elif lay.times.shape[-1] == n_t + 1:
            # PDEBench stores the T+1 interval edges next to T saved states
            lay.times = lay.times[..., :n_t]
            lay.warnings.append(
                f"zone {lay.zone!r}: time coordinate has {n_t + 1} entries for {n_t} steps; "
                "the last one was dropped"
            )
        if lay.times is not None:
            t = lay.times_for(0)
            assert t is not None
            if t.size > 1 and not np.all(np.diff(t) > 0):
                lay.warnings.append(
                    f"zone {lay.zone!r}: time coordinate is not strictly increasing; step "
                    "indices are used as times"
                )
                lay.times = np.arange(n_t or 0, dtype=np.float64)
                lay.time_meta["time_quantity"] = "iteration"
        counts = {p.size("sample") for p in plans if p.sample_axis is not None}
        if len(counts) > 1:
            raise MappingError(f"fields disagree on the number of samples: {sorted(counts)}")  # type: ignore[type-var]
        if counts:
            lay.n_samples = next(iter(counts))
        elif lay.times is not None and lay.times.ndim == 2:
            lay.times = lay.times[0]
        ids = self.samples.get("ids")
        if ids is not None and lay.n_samples is not None:
            raw = self.node(ids, "samples.ids").read().reshape(-1) if isinstance(ids, str) else ids
            lay.sample_ids = [meshutil.sanitize_id(_text(v)) for v in list(raw)]
            if len(lay.sample_ids) != lay.n_samples or len(set(lay.sample_ids)) != lay.n_samples:
                raise MappingError(
                    f"samples.ids must give {lay.n_samples} unique names, got {len(lay.sample_ids)}"
                )
        for key, v in (m.get("parameters") or {}).items():
            lay.parameters[str(key)] = self._parameter(str(key), v)
        for key in ("solver", "solver_version", "physics", "boundary_conditions", "units"):
            if m.get(key) is not None:
                lay.metadata[key] = m[key]
        lay.metadata.update(dict(m.get("metadata") or {}))
        lay.domain = str(m.get("domain") or "other")
        lay.warnings.extend(str(w) for w in m.get("_warnings", []))
        return lay

    def _parameter(self, key: str, v: Any) -> Any:
        if isinstance(v, int | float) and not isinstance(v, bool):
            return float(v)
        if not isinstance(v, str):
            raise MappingError(f"parameter {key!r} must be a number, an array path or 'path@attr'")
        if "@" in v:
            val = self.store.attr(v, self.prefix)
            num = _number(val)
            return num if num is not None else val
        arr = np.asarray(self.store.node(v, self.prefix, what=f"parameter {key!r}").read())
        arr = np.squeeze(arr)
        if arr.dtype.kind not in "fiub":
            raise MappingError(f"parameter {key!r}: /{norm_path(v)} is not numeric")
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1 and (self.lay.n_samples in (None, arr.shape[0])):
            return arr.astype(np.float64)
        raise MappingError(
            f"parameter {key!r}: /{norm_path(v)} has shape {arr.shape}; a scalar or one value "
            "per sample is needed"
        )


def resolve_mapping(
    store: ArrayStore, mapping: Mapping[str, Any], *, zone: str = "main", prefix: str = ""
) -> Layout:
    """Resolve one zone of a mapping config against ``store``."""
    if not isinstance(mapping, Mapping):
        raise MappingError("the mapping must be a JSON object", hint="see the hdf5 connector docs")
    if mapping.get("group"):
        prefix = str(mapping["group"])
    samples = mapping.get("samples")
    if isinstance(samples, Mapping) and samples.get("groups") is not None:
        pats = samples["groups"]
        pats = [pats] if isinstance(pats, str) else [str(p) for p in pats]
        base = norm_path(prefix)
        groups = [
            g
            for g in store.groups()
            if parent(g) == base
            and any(
                fnmatch.fnmatchcase(basename(g), norm_path(p)) or g == norm_path(p) for p in pats
            )
        ]
        if not groups:
            raise MappingError(
                f"samples.groups {pats} matched no group under /{base}",
                hint="groups present: " + ", ".join("/" + g for g in store.groups()[:20]),
            )
        sub = {k: v for k, v in mapping.items() if k not in ("samples", "group")}
        lay = resolve_mapping(store, sub, zone=zone, prefix=groups[0])
        lay.groups, lay.mapping = groups, sub
        lay.sample_ids = [meshutil.sanitize_id(basename(g)) for g in groups]
        return lay
    r = _Resolver(store, mapping, zone, prefix)
    r.geometry()
    r.time()
    r.fields()
    return r.finish()


def resolve_all(store: ArrayStore, mapping: Mapping[str, Any]) -> list[Layout]:
    """Resolve a mapping with or without a ``zones`` table into layouts."""
    zones = mapping.get("zones")
    if not zones:
        return [resolve_mapping(store, mapping, zone=str(mapping.get("_zone", "main")))]
    if not isinstance(zones, Mapping):
        raise MappingError("mapping 'zones' must be an object {zone name: mapping}")
    shared = {k: v for k, v in mapping.items() if k != "zones"}
    out = []
    for name, zm in zones.items():
        merged = {**shared, **dict(zm)}
        if shared.get("parameters") or zm.get("parameters"):
            merged["parameters"] = {**(shared.get("parameters") or {}), **zm.get("parameters", {})}
        out.append(resolve_mapping(store, merged, zone=str(name)))
    return out


# -- auto-detection: CF / dimension scales / xarray-style dims ---------------------------------

_BOUNDS_DIMS = ("bnds", "bounds", "nv", "nbnd", "nbnds", "bnd", "vertices", "nvertices", "d2")
_X_STD = ("longitude", "grid_longitude", "projection_x_coordinate")
_Y_STD = ("latitude", "grid_latitude", "projection_y_coordinate")
_Z_STD = (
    "height", "altitude", "depth", "air_pressure", "model_level_number", "geopotential_height",
    "height_above_reference_ellipsoid", "atmosphere_sigma_coordinate",
    "atmosphere_hybrid_sigma_pressure_coordinate", "ocean_sigma_coordinate",
)  # fmt: skip
_X_UNITS = ("degrees_east", "degree_east", "degrees_e", "degree_e", "degreese", "degreee")
_Y_UNITS = ("degrees_north", "degree_north", "degrees_n", "degree_n", "degreesn", "degreen")
_NAME_ROLES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("T", ("time", "t", "times", "time_counter", "valid_time", "month", "year"), ("time",)),
    ("X", ("lon", "longitude", "x", "xc", "xi", "nx", "west_east", "rlon", "nlon", "i"),
     ("lon", "xi_")),
    ("Y", ("lat", "latitude", "y", "yc", "eta", "ny", "south_north", "rlat", "nlat", "j"),
     ("lat", "eta_")),
    ("Z", ("lev", "level", "levels", "z", "zc", "depth", "height", "plev", "alt", "altitude",
           "bottom_top", "nz", "sigma", "layer", "k", "isobaric", "pressure"),
     ("lev", "plev", "depth", "height", "isobaric", "s_rho", "s_w")),
)  # fmt: skip


def cf_axis(dim: str, coord: ArrayNode | None) -> tuple[str | None, bool]:
    """CF role of a dimension: ``("T"|"X"|"Y"|"Z"|"S"|None, geographic?)``. Attributes of the
    coordinate variable win over the dimension name."""
    a = coord.attrs if coord is not None else {}
    axis = _text(a.get("axis", "")).upper()
    std = _text(a.get("standard_name", "")).lower()
    units = _text(a.get("units", "")).lower()
    if std == "time" or " since " in units or axis == "T":
        return "T", False
    if std in _X_STD or units in _X_UNITS:
        return "X", std != "projection_x_coordinate"
    if std in _Y_STD or units in _Y_UNITS:
        return "Y", std != "projection_y_coordinate"
    if axis in ("X", "Y", "Z"):
        return axis, False
    vertical = std.endswith("_coordinate") and std.startswith(("ocean_", "atmosphere_"))
    if std in _Z_STD or vertical or a.get("positive") is not None:
        return "Z", False
    n = dim.lower()
    if n in SAMPLE_NAMES:
        return "S", False
    for role, exact, prefixes in _NAME_ROLES:
        if n in exact or any(n.startswith(p) for p in prefixes):
            return role, n.startswith(("lon", "lat", "rlon", "rlat"))
    return None, False


def _is_coordinate(node: ArrayNode) -> bool:
    if node.ndim != 1 or not node.dims:
        return False
    if node.scales is not None and node.scales[0] == node.path:
        return True
    return node.dims[0] == node.name


def _referenced(store: ArrayStore, keys: Sequence[str]) -> set[str]:
    """Paths of arrays named by the attributes ``keys`` (``bounds``, ``climatology``...)."""
    out: set[str] = set()
    for n in store.nodes.values():
        if n.phantom:
            continue
        for key in keys:
            v = n.attrs.get(key)
            if not isinstance(v, str):
                continue
            for name in v.split():
                g = parent(n.path)
                out.add(f"{g}/{name}" if g else name)
                out.add(norm_path(name))
    return out


def cf_mapping(store: ArrayStore) -> dict[str, Any] | None:
    """Mapping for a self-describing store: arrays with named dimensions and coordinate
    variables (NetCDF, HDF5 dimension scales, xarray-style Zarr), CF attributes, UGRID 2-D."""
    data = [n for n in store.nodes.values() if n.dims and n.numeric and n.ndim >= 1]
    if not data:
        return None
    warnings: list[str] = []
    skip = _referenced(store, ("bounds", "climatology"))
    zones: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for zname, zm, used in _ugrid_zones(store, warnings):
        zones[zname] = zm
        consumed |= used
    groups: dict[tuple[tuple[str, str, str | None], ...], list[tuple[ArrayNode, list[Any]]]] = {}
    geographic = False
    skipped: list[str] = []
    for n in data:
        if n.path in consumed or n.path in skip or _is_coordinate(n):
            continue
        assert n.dims is not None
        axes: list[Any] = []
        ok = True
        for i, dim in enumerate(n.dims):
            coord = store.scale(n, i)
            role, geo = cf_axis(dim, coord)
            if role is None:
                if n.shape[i] == 1:
                    role = "drop"
                elif (
                    i == n.ndim - 1
                    and n.shape[i] <= MAX_COMPONENTS
                    and dim.lower() not in _BOUNDS_DIMS
                ):
                    role = "C"
                else:
                    ok = False
            axes.append((role, dim, coord.path if coord is not None else None, geo))
        roles = [a[0] for a in axes]
        space = [a for a in axes if a[0] in ("X", "Y", "Z")]
        if not ok or not space or any(roles.count(r) > 1 for r in "TXYZS"):
            if n.ndim > 1 or not ok:
                skipped.append(f"{n.name}({', '.join(n.dims)})")
            continue
        geographic = geographic or any(a[3] for a in space)
        key = tuple(sorted(((a[0], a[1], a[2]) for a in space), key=lambda a: "XYZ".index(a[0])))
        groups.setdefault(key, []).append((n, axes))
    # 1-D variables along one axis of a larger grid (latitude weights, level bounds) are auxiliary
    keys = list(groups)
    for key in keys:
        mine = {k[1] for k in key}
        bigger = any(mine < {k[1] for k in other} for other in keys if other != key)
        timeless = all("T" not in [a[0] for a in axes] for _, axes in groups[key])
        if bigger and timeless and len(key) == 1:
            skipped.extend(f"{n.name}({', '.join(n.dims or [])})" for n, _ in groups[key])
            del groups[key]
    if skipped:
        warnings.append(
            "variables not on a recognised grid were skipped: "
            + ", ".join(skipped[:12])
            + (" ..." if len(skipped) > 12 else "")
            + " (map them explicitly with a mapping config if needed)"
        )
    for key, members in groups.items():
        zm: dict[str, Any] = {"kind": "grid", "coords": {}, "fields": {}}
        contract: dict[str, str] = {}
        for ax, (role, dim, scale) in zip(GRID_AXES, key, strict=False):
            contract[role] = ax
            zm["coords"][ax] = "/" + scale if scale else {"start": 0, "step": 1, "name": dim}
        unlabelled = [k[1] for k in key if k[2] is None]
        if unlabelled:
            warnings.append(
                f"dimensions {unlabelled} have no 1-D coordinate variable (curvilinear or "
                "unlabelled grid): index coordinates are used; 2-D coordinate arrays on the "
                "same grid are kept as static fields"
            )
        tdim: tuple[str, str | None] | None = None
        for n, axes in members:
            t = next(((a[1], a[2]) for a in axes if a[0] == "T"), None)
            if t is not None and tdim is None:
                tdim = t
            if t is not None and t != tdim:
                warnings.append(f"{n.name}: uses a second time axis {t[0]!r}; skipped")
                continue
            dims: list[str] = []
            select: dict[str, int] = {}
            for i, (role, _dim, _scale, _geo) in enumerate(axes):
                if role == "drop":
                    dims.append(f"_drop{i}")
                    select[f"_drop{i}"] = 0
                else:
                    dims.append(
                        {"T": "time", "S": "sample", "C": "component"}.get(role) or contract[role]
                    )
            name = n.name.removeprefix("_nc4_non_coord_")
            if name in zm["fields"]:
                name = n.path.replace("/", "_")
            spec: dict[str, Any] = {"path": "/" + n.path, "dims": dims}
            if select:
                spec["select"] = select
            zm["fields"][name] = spec
        if tdim is not None:
            tnode = store.nodes[tdim[1]] if tdim[1] is not None else None
            if tnode is None or not tnode.numeric:
                zm["time"] = {"start": 0, "dt": 1, "name": tdim[0]}
            elif tnode.attrs.get("units") or tdim[0].lower() not in ("month", "year"):
                zm["time"] = "/" + tnode.path
            else:
                zm["time"] = {"path": "/" + tnode.path, "units": tdim[0].lower()}
        if not zm["fields"]:
            continue
        zones["_".join(k[1] for k in key)] = zm
    if not zones:
        return None
    if len(zones) == 1:
        ((zname, only),) = zones.items()
        out: dict[str, Any] = {**only}
        out["_zone"] = "grid" if only.get("kind") == "grid" else zname
    else:
        out = {"zones": zones}
    if not geographic:
        # curvilinear grids carry longitude/latitude as 2-D arrays, not as axes
        geographic = any(
            _text(n.attrs.get("units", "")).lower() in (*_X_UNITS, *_Y_UNITS)
            for n in store.nodes.values()
            if not n.phantom
        )
    if geographic:
        out["domain"] = "geophysical"
    out["_warnings"] = warnings
    return out


_UGRID_AUX = (
    "face_coordinates",
    "edge_coordinates",
    "edge_node_connectivity",
    "face_edge_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
)


def _ugrid_zones(
    store: ArrayStore, warnings: list[str]
) -> Iterator[tuple[str, dict[str, Any], set[str]]]:
    """UGRID 1.0 two-dimensional meshes (``cf_role = mesh_topology``)."""
    for topo in list(store.nodes.values()):
        a = topo.attrs
        if _text(a.get("cf_role", "")) != "mesh_topology":
            continue
        g = parent(topo.path)

        def rel(name: str, g: str = g) -> str:
            return f"{g}/{name}" if g else name

        coords = [rel(c) for c in _text(a.get("node_coordinates", "")).split()]
        conn = _text(a.get("face_node_connectivity", ""))
        if _number(a.get("topology_dimension")) != 2 or len(coords) < 2 or not conn:
            warnings.append(
                f"UGRID topology {topo.name!r}: only 2-D face meshes are read (planned: 1-D "
                "networks and 3-D layered meshes); pass a mapping for other layouts"
            )
            continue
        if not all(store.has(p) for p in [*coords, rel(conn)]):
            warnings.append(f"UGRID topology {topo.name!r} references missing variables")
            continue
        n_nodes = store.nodes[coords[0]].shape[0]
        n_faces = store.nodes[rel(conn)].shape[0]
        used = {topo.path, rel(conn), *coords}
        for key in _UGRID_AUX:
            used |= {rel(v) for v in _text(a.get(key, "")).split()}
        zm: dict[str, Any] = {
            "kind": "mesh",
            "points": ["/" + c for c in coords],
            "cells": "/" + rel(conn),
            "cell_type": "polygon",
            "fields": {},
        }
        tdim: str | None = None
        ndim_name = (store.nodes[coords[0]].dims or [None])[0]
        fdim_name = (store.nodes[rel(conn)].dims or [None])[0]
        for n in store.nodes.values():
            if n.path in used or not n.numeric or not n.dims or _is_coordinate(n):
                continue
            owner = _text(n.attrs.get("mesh", ""))
            loc = {"node": "point", "face": "cell"}.get(_text(n.attrs.get("location", "")))
            if not owner and loc is None:
                # files that omit the mesh/location attributes: go by the dimension names
                if ndim_name in n.dims:
                    owner, loc = topo.name, "point"
                elif fdim_name in n.dims:
                    owner, loc = topo.name, "cell"
            if owner != topo.name:
                continue
            dims: list[str] = []
            for i, dim in enumerate(n.dims):
                coord = store.scale(n, i)
                if cf_axis(dim, coord)[0] == "T":
                    dims.append("time")
                    tdim = "/" + coord.path if coord is not None else tdim
                elif loc and n.shape[i] == (n_nodes if loc == "point" else n_faces):
                    dims.append(loc)
                else:
                    dims.append("?")
            used.add(n.path)
            if loc is None or "?" in dims or dims.count(loc) != 1:
                warnings.append(f"UGRID variable {n.name!r} is not node/face data; skipped")
                continue
            zm["fields"][n.name] = {"path": "/" + n.path, "dims": dims}
        if tdim:
            zm["time"] = tdim
        elif any("time" in f["dims"] for f in zm["fields"].values()):
            zm["time"] = {"start": 0, "dt": 1}
        yield topo.name, zm, used


# -- auto-detection: HDF5 layouts without dimension names --------------------------------------

_COORD_NAMES = {
    "x": ("x", "x-coordinate", "x_coordinate", "xcoord", "x_coord", "xs", "grid_x", "x_grid"),
    "y": ("y", "y-coordinate", "y_coordinate", "ycoord", "y_coord", "ys", "grid_y", "y_grid"),
    "z": ("z", "z-coordinate", "z_coordinate", "zcoord", "z_coord", "zs", "grid_z", "z_grid"),
}
_TIME_NAMES = ("t", "t-coordinate", "t_coordinate", "time", "times", "tcoord", "grid_t", "ts")
_POINT_NAMES = (
    "points", "coordinates", "nodes", "pos", "mesh_pos", "node_coordinates", "vertices",
    "coords", "xyz", "node_coords", "geometry",
)  # fmt: skip
_CELL_NAMES = (
    "cells", "connectivity", "elements", "topology", "triangles", "tetrahedra", "faces", "conn",
    "elems", "tets", "hexahedra", "quads",
)  # fmt: skip


def _relpath(node: ArrayNode, prefix: str) -> str:
    """Path as written into a generated mapping: relative to ``prefix`` (so one mapping can be
    replayed on every per-sample group), absolute for the root."""
    if prefix:
        return node.path[len(prefix) + 1 :]
    return "/" + node.path


def _pick(nodes: Sequence[ArrayNode], names: Sequence[str], ok: Callable[[ArrayNode], bool]):
    found = [n for n in nodes if n.name.lower() in names and ok(n)]
    found.sort(key=lambda n: (n.path.count("/"), names.index(n.name.lower())))
    return found[0] if found else None


def _truthy(v: Any) -> bool:
    v = clean_attr(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def well_mapping(store: ArrayStore, prefix: str = "") -> dict[str, Any] | None:
    """The Well layout (``dimensions/``, ``t0_fields/``, ``t1_fields/``, ``t2_fields/``,
    ``scalars/``, ``boundary_conditions/``), as documented by PolymathicAI/the_well."""
    pre = f"{prefix}/" if prefix else ""
    tensor = [(k, store.children(f"{pre}t{k}_fields")) for k in (0, 1, 2)]
    if not any(nodes for _, nodes in tensor) or not store.children(f"{pre}dimensions"):
        return None
    dnodes = {n.name: n for n in store.children(f"{pre}dimensions")}
    dattrs = store.group_attrs.get(f"{pre}dimensions", {})
    spatial = dattrs.get("spatial_dims")
    spatial = [spatial] if isinstance(spatial, str) else list(spatial or [])
    spatial = [s for s in spatial if s in dnodes] or [d for d in ("x", "y", "z") if d in dnodes]
    if not spatial or len(spatial) > 3:
        return None
    warnings: list[str] = []
    m: dict[str, Any] = {"kind": "grid", "coords": {}, "fields": {}, "parameters": {}}
    for ax, name in zip(GRID_AXES, spatial, strict=False):
        if dnodes[name].ndim != 1:
            warnings.append(f"dimensions/{name} varies per sample or in time; first row used")
            return None
        m["coords"][ax] = _relpath(dnodes[name], prefix)
    if "time" in dnodes:
        m["time"] = _relpath(dnodes["time"], prefix)
    has_sample = False
    for order, nodes in tensor:
        for n in nodes:
            a = n.attrs
            sample = _truthy(a.get("sample_varying", True))
            timed = _truthy(a.get("time_varying", True))
            vary = a.get("dim_varying")
            vary = [True] * len(spatial) if vary is None else [bool(v) for v in np.atleast_1d(vary)]
            if not all(vary):
                warnings.append(
                    f"t{order}_fields/{n.name} is constant along "
                    f"{[s for s, v in zip(spatial, vary, strict=False) if not v]}; skipped "
                    "(broadcast fields are not supported yet)"
                )
                continue
            dims = (
                (["sample"] if sample else [])
                + (["time"] if timed else [])
                + list(GRID_AXES[: len(spatial)])
                + ["component"] * order
            )
            if len(dims) != n.ndim:
                warnings.append(f"t{order}_fields/{n.name}: shape {n.shape} != flags; skipped")
                continue
            has_sample = has_sample or sample
            m["fields"][n.name] = {"path": _relpath(n, prefix), "dims": dims}
    if not m["fields"]:
        return None
    if has_sample:
        m["samples"] = {"axis": "sample"}
    n_traj = next((f for f in m["fields"].values() if "sample" in f["dims"]), None)
    n_s = store.node(n_traj["path"], prefix).shape[0] if n_traj else None
    for n in store.children(f"{pre}scalars"):
        if _truthy(n.attrs.get("time_varying", False)):
            continue  # a per-step signal, not a conditioning parameter
        if n.numeric and (n.ndim == 0 or n.shape in ((1,), (n_s,))):
            m["parameters"][n.name] = _relpath(n, prefix)
    root = store.group_attrs.get(prefix, {})
    for name in np.atleast_1d(root.get("simulation_parameters", [])).tolist():
        if _number(root.get(str(name))) is not None:
            m["parameters"][str(name)] = f"/{prefix}@{name}" if prefix else f"@{name}"
    bcs: dict[str, Any] = {}
    for g in store.groups():
        if parent(g) == f"{pre}boundary_conditions":
            ga = store.group_attrs.get(g, {})
            bcs[basename(g)] = {
                k: ga[k] for k in ("bc_type", "associated_dims", "associated_fields") if k in ga
            }
    if bcs:
        m["boundary_conditions"] = bcs
    meta = {k: root[k] for k in ("dataset_name", "grid_type", "n_trajectories") if k in root}
    m["metadata"] = {
        **meta,
        "layout": "the-well",
        "axis_names": dict(zip(GRID_AXES, spatial, strict=False)),
    }
    m["_zone"] = "grid"
    m["_warnings"] = warnings
    return m


def mlpde_mapping(store: ArrayStore, prefix: str = "") -> dict[str, Any] | None:
    """PDEBench / FNO-style layout: 1-D coordinate arrays (``x-coordinate``, ``t-coordinate``,
    ``x``, ``t``, ``grid/x`` ...) and one array per field shaped
    ``([batch,] [time,] x[, y[, z]][, C])``."""
    nodes = [n for n in store.below(prefix) if n.numeric]

    def axis_ok(n: ArrayNode) -> bool:
        return n.ndim == 1 and n.shape[0] > 1

    coords = []
    for ax in GRID_AXES:
        c = _pick(nodes, _COORD_NAMES[ax], axis_ok)
        if c is None:
            break
        coords.append(c)
    if not coords:
        return None
    t = _pick(nodes, _TIME_NAMES, lambda n: n.ndim == 1)
    block = tuple(c.shape[0] for c in coords)
    nsp = len(block)
    reserved = {c.path for c in coords} | ({t.path} if t is not None else set())
    m: dict[str, Any] = {"kind": "grid", "coords": {}, "fields": {}}
    for ax, c in zip(GRID_AXES, coords, strict=False):
        m["coords"][ax] = _relpath(c, prefix)
    n_t = t.shape[0] if t is not None else None
    any_time = any_sample = False
    for n in nodes:
        if n.path in reserved or n.ndim < nsp or n.name.lower() in _TIME_NAMES:
            continue
        for k in range(0, min(2, n.ndim - nsp) + 1):
            post = n.ndim - nsp - k
            if n.shape[k : k + nsp] != block or post not in (0, 1):
                continue
            if post and n.shape[-1] > MAX_COMPONENTS:
                continue
            if k == 2:
                lead = ["sample", "time"]
            elif k == 1:
                timed = n_t is not None and n.shape[0] in (n_t, n_t - 1)
                lead = ["time"] if timed else ["sample"]
            else:
                lead = []
            dims = lead + list(GRID_AXES[:nsp]) + ["component"] * post
            m["fields"][n.name if n.name not in m["fields"] else n.path.replace("/", "_")] = {
                "path": _relpath(n, prefix),
                "dims": dims,
            }
            any_time = any_time or "time" in lead
            any_sample = any_sample or "sample" in lead
            break
    if not m["fields"]:
        return None
    if any_time:
        m["time"] = _relpath(t, prefix) if t is not None else {"start": 0, "dt": 1}
    if any_sample:
        m["samples"] = {"axis": "sample"}
    params = {}
    for key, v in store.group_attrs.get(prefix, {}).items():
        if _number(v) is not None and not isinstance(v, bool) and not str(key).startswith("_"):
            params[str(key)] = f"/{prefix}@{key}" if prefix else f"@{key}"
    if params:
        m["parameters"] = params
    m["_zone"] = "grid"
    m["_warnings"] = [
        "axis order ([sample,] [time,] x[, y[, z]][, component]) was inferred from array shapes "
        "and coordinate array names; pass a mapping config to override"
    ]
    return m


def mesh_mapping(store: ArrayStore, prefix: str = "") -> dict[str, Any] | None:
    """Mesh layout: ``points``/``coordinates``/``nodes`` ``(N, 2|3)`` + ``cells``/
    ``connectivity``/``elements`` ``(K, P)`` + per-node/per-cell arrays ``([T,] N|K[, C])``."""
    nodes = [n for n in store.below(prefix) if n.numeric]
    pts = _pick(
        nodes,
        _POINT_NAMES,
        lambda n: n.dtype.kind == "f" and n.ndim in (2, 3) and n.shape[-1] in (2, 3),
    )
    cells = _pick(nodes, _CELL_NAMES, lambda n: n.dtype.kind in "iu" and n.ndim == 2)
    if pts is None or cells is None:
        return None
    n_p, n_c = pts.shape[-2], cells.shape[0]
    t = _pick(nodes, _TIME_NAMES, lambda n: n.ndim == 1)
    n_t = t.shape[0] if t is not None else None
    surface = cells.shape[1] == 3 and pts.shape[-1] == 3
    m: dict[str, Any] = {
        "kind": "surface" if surface else "mesh",
        "points": _relpath(pts, prefix),
        "cells": _relpath(cells, prefix),
        "fields": {},
    }
    sizes = {n_p: "point", n_c: "cell"} if n_p != n_c else {n_p: "point"}
    any_time = False
    for n in nodes:
        if n.path in (pts.path, cells.path) or (t is not None and n.path == t.path):
            continue
        s = n.shape
        dims: list[str] | None = None
        timed_2d = n.ndim == 2 and n_t is not None and s[0] == n_t and s[1] in sizes
        if n.ndim in (1, 2) and s[0] in sizes and not timed_2d:
            if n.ndim == 1 or s[1] <= MAX_COMPONENTS:
                dims = [sizes[s[0]]] + ["component"] * (n.ndim - 1)
        elif n.ndim in (2, 3) and s[1] in sizes and (n_t is None or s[0] == n_t):
            if n.ndim == 2 or s[2] <= MAX_COMPONENTS:
                dims = ["time", sizes[s[1]]] + ["component"] * (n.ndim - 2)
                any_time = True
        if dims is not None:
            m["fields"][n.name if n.name not in m["fields"] else n.path.replace("/", "_")] = {
                "path": _relpath(n, prefix),
                "dims": dims,
            }
    if any_time:
        m["time"] = _relpath(t, prefix) if t is not None else {"start": 0, "dt": 1}
    m["_zone"] = "mesh"
    m["_warnings"] = [
        f"mesh layout inferred from array names ({pts.name}, {cells.name}); cells are assumed to "
        "use VTK node ordering"
    ]
    if pts.ndim == 3:
        m["_warnings"].append(f"/{pts.path} is time dependent; the first step is the geometry")
    return m


GENERIC_DETECTORS: tuple[Callable[[ArrayStore, str], dict[str, Any] | None], ...] = (
    well_mapping,
    mlpde_mapping,
    mesh_mapping,
)


def generic_mapping(store: ArrayStore) -> dict[str, Any] | None:
    """Run the name/shape based detectors on the root, then on per-sample groups (every
    top-level group holding the same arrays is one sample, e.g. PDEBench ``0000/data``)."""
    top = [g for g in store.groups() if "/" not in g]
    rel = [frozenset(n.path[len(g) + 1 :] for n in store.below(g)) for g in top]
    per_group = len(top) > 1 and not store.children("") and len(set(rel)) == 1 and bool(rel[0])
    if not per_group:
        for det in GENERIC_DETECTORS:
            m = det(store, "")
            if m is not None:
                return m
    if not top:
        return None
    for det in GENERIC_DETECTORS:
        m = det(store, top[0])
        if m is None:
            continue
        need = {n.path[len(top[0]) + 1 :] for n in store.below(top[0])}
        same = [g for g in top if need <= {n.path[len(g) + 1 :] for n in store.below(g)}]
        m["samples"] = {"groups": "*" if len(same) == len(top) else same}
        if len(same) != len(top):
            m.setdefault("_warnings", []).append(
                f"{len(top) - len(same)} top-level groups do not hold the same arrays as "
                f"/{top[0]} and were skipped"
            )
        return m
    return None


def _named_skeleton(store: ArrayStore, top: ArrayNode) -> dict[str, Any]:
    """Skeleton for arrays with named dimensions: recognised axes are filled in, the others
    are left under their own name with a ``select`` index for the user to decide on."""
    assert top.dims is not None
    roles = [cf_axis(d, store.scale(top, i))[0] for i, d in enumerate(top.dims)]
    out: dict[str, Any] = {"kind": "grid", "coords": {}, "fields": {}}
    contract: dict[int, str] = {}
    spatial = [i for i, r in enumerate(roles) if r in ("X", "Y", "Z")]
    spatial.sort(key=lambda i: "XYZ".index(str(roles[i])))
    for ax, i in zip(GRID_AXES, spatial, strict=False):
        scale = store.scale(top, i)
        contract[i] = ax
        out["coords"][ax] = "/" + scale.path if scale is not None else {"start": 0.0, "step": 1.0}
    dims: list[str] = []
    select: dict[str, int] = {}
    for i, (dim, role) in enumerate(zip(top.dims, roles, strict=True)):
        if i in contract:
            dims.append(contract[i])
        elif role == "T":
            scale = store.scale(top, i)
            out["time"] = "/" + scale.path if scale is not None else {"start": 0.0, "dt": 1.0}
            dims.append("time")
        else:
            dims.append(dim)
            select[dim] = 0
    if not out["coords"]:
        out["coords"] = {"x": {"start": 0.0, "step": 1.0}}
    for n in store.nodes.values():
        if n.numeric and n.dims == top.dims and len(out["fields"]) < 4:
            spec: dict[str, Any] = {"path": "/" + n.path, "dims": dims}
            if select:
                spec["select"] = select
            out["fields"][n.name] = spec
    return out


def skeleton(store: ArrayStore) -> dict[str, Any]:
    """Mapping skeleton guessed from dimension names or ranks, for the user to correct."""
    nodes = [n for n in store.nodes.values() if n.numeric and n.ndim >= 1]
    named = [n for n in nodes if n.dims and n.ndim >= 2 and not _is_coordinate(n)]
    if named:
        return _named_skeleton(store, max(named, key=lambda n: (n.ndim, int(np.prod(n.shape)))))
    big = sorted(nodes, key=lambda n: (-n.ndim, -int(np.prod(n.shape))))
    one_d = [n for n in nodes if n.ndim == 1]
    top = big[0] if big else None
    rank = top.ndim if top is not None else 3
    n_space = max(1, min(3, rank - 2))
    coords: dict[str, Any] = {}
    for i, ax in enumerate(GRID_AXES[:n_space]):
        size = top.shape[min(1 + i, rank - 1)] if top is not None else 0
        match = next((n for n in one_d if n.shape[0] == size), None)
        coords[ax] = "/" + match.path if match else {"start": 0.0, "step": 1.0}
    dims = ["time", *GRID_AXES[:n_space], "component"][:rank]
    dims = ["?"] * (rank - len(dims)) + dims
    fields = {
        n.name: {"path": "/" + n.path, "dims": dims}
        for n in big[:3]
        if top is not None and n.ndim == top.ndim
    }
    return {
        "kind": "grid",
        "time": {"start": 0.0, "dt": 1.0},
        "coords": coords,
        "fields": fields or {"u": {"path": "/<array>", "dims": dims}},
    }


def needs_mapping(store: ArrayStore, why: str = "") -> MappingError:
    """The error raised when a store is not self-describing: dataset tree + mapping skeleton."""
    skel = json.dumps(skeleton(store), indent=1)
    return MappingError(
        f"{store.path.name}: the layout is not self-describing{why}; a mapping config is needed",
        hint="arrays in the file:\n"
        + store.tree()
        + "\n  suggested mapping skeleton (fix the axis names, save as mapping.json and pass "
        "--mapping mapping.json or options={'mapping': {...}}):\n"
        + skel
        + "\n  axis names: time, x, y, z (grids), point, cell (meshes), component, sample; any "
        'other name needs a "select" index (or name it under "samples": {"axis": ...} to get '
        'one sample per index); for meshes use "points"/"cells" instead of "coords"',
    )


# -- plan = explicit mapping, or auto-detected mapping (+ user overlay) ------------------------

_STRUCTURAL = ("kind", "coords", "points", "cells", "time", "samples", "zones", "group")


def _overlay(auto: dict[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a partial user mapping (vector grouping, units, metadata) on a detected one."""
    out = dict(auto)
    zmaps = list(out["zones"].values()) if out.get("zones") else [out]
    zone_keys = ("fields", "static", "parameters")
    for key, v in user.items():
        if key not in zone_keys:
            out[key] = v
    matched = False
    for zm in zmaps:
        if user.get("parameters"):
            zm["parameters"] = {**zm.get("parameters", {}), **dict(user["parameters"])}
        if user.get("static"):
            zm["static"] = {**zm.get("static", {}), **dict(user["static"])}
        zfields: dict[str, Any] = dict(zm.get("fields") or {})
        by_path = {norm_path(f["path"]): k for k, f in zfields.items() if "path" in f}
        for name, spec in (user.get("fields") or {}).items():
            spec = {"path": spec} if isinstance(spec, str) else dict(spec)
            paths = [spec["path"]] if spec.get("path") else list(spec.get("paths") or [])
            found = [by_path.get(norm_path(p)) or (p if p in zfields else None) for p in paths]
            keys = [str(k) for k in found if k is not None]
            if not paths or len(keys) != len(paths):
                continue
            matched = True
            base = zfields[keys[0]]
            merged = {**{k: v for k, v in base.items() if k != "path"}, **spec}
            merged.pop("path", None)
            merged["paths"] = [zfields[k]["path"] for k in keys]
            for k in keys:
                if k != name:
                    zfields.pop(k, None)
            zfields[str(name)] = merged
        zm["fields"] = zfields
    if user.get("fields") and not matched:
        raise MappingError(
            "none of the mapped fields refers to an array of the detected layout",
            hint="a partial mapping may only regroup detected fields; add 'kind'/'coords'/"
            "'points' to describe the layout yourself",
        )
    return out


@dataclass
class Plan:
    layouts: list[Layout]
    mapping: dict[str, Any]
    auto: bool = False


def make_plan(
    store: ArrayStore,
    options: Mapping[str, Any] | None,
    detectors: Sequence[Callable[[ArrayStore], dict[str, Any] | None]],
) -> Plan:
    user = load_mapping((options or {}).get("mapping"))
    if user and any(k in user for k in _STRUCTURAL):
        return Plan(resolve_all(store, user), dict(user))
    auto = None
    for det in detectors:
        auto = det(store)
        if auto is not None:
            break
    if auto is None:
        raise needs_mapping(store)
    if user:
        auto = _overlay(auto, user)
    return Plan(resolve_all(store, auto), auto, auto=True)


def public_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Mapping without the private ``_*`` keys, as shown in ``describe`` metadata."""
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        if str(k).startswith("_"):
            continue
        if k == "zones" and isinstance(v, Mapping):
            out[k] = {zn: public_mapping(zm) for zn, zm in v.items()}
        else:
            out[k] = v
    return jsonable(out)


# -- describe / read -------------------------------------------------------------------------


def select_samples(spec: Any, n: int, ids: Sequence[str]) -> list[int]:
    """``options["samples"]``: ``None`` (all), an int (first N), ``"a:b[:c]"``, or a list of
    indices / ids / globs."""
    if spec is None:
        return list(range(n))
    if isinstance(spec, bool):
        raise SelectionError("options['samples'] cannot be a bool")
    if isinstance(spec, int):
        return list(range(min(max(spec, 0), n)))
    if isinstance(spec, str) and re.fullmatch(r"-?\d*:-?\d*(:-?\d*)?", spec.strip()):
        parts = [int(p) if p else None for p in spec.strip().split(":")]
        return list(range(n))[slice(*parts)]
    items = [spec] if isinstance(spec, str) else list(spec)
    out: list[int] = []
    for it in items:
        if isinstance(it, int | np.integer) and not isinstance(it, bool):
            if not -n <= int(it) < n:
                raise SelectionError(
                    f"sample index {it} out of range", hint=f"the source has {n} samples"
                )
            out.append(int(it) % n)
        else:
            hit = [i for i, s in enumerate(ids) if s == str(it) or fnmatch.fnmatchcase(s, str(it))]
            if not hit:
                raise SelectionError(
                    f"sample {it!r} not found",
                    hint="available samples: " + ", ".join(ids[:20]) + (" ..." if n > 20 else ""),
                )
            out.extend(hit)
    return sorted(set(out))


def _sample_labels(lay: Layout) -> list[str]:
    n = len(lay.groups) if lay.groups else (lay.n_samples or 0)
    if lay.sample_ids:
        return lay.sample_ids
    width = max(4, len(str(max(n - 1, 0))))
    return [f"{i:0{width}d}" for i in range(n)]


def describe_layouts(plan: Plan, desc: SourceDescription) -> SourceDescription:
    layouts = plan.layouts
    multi = len(layouts) > 1
    seen: dict[tuple[str, str], FieldInfo] = {}
    for lay in layouts:
        role = {"grid": "grid", "point_cloud": "probes", "surface": "boundary"}.get(
            lay.kind, "volume"
        )
        z = ZoneInfo(name=lay.zone, kind=lay.kind, role=role, dim=lay.point_dim)
        if lay.kind == "grid":
            z.shape = [len(lay.coords[a]) for a in lay.axes]
            z.n_points = int(np.prod(z.shape))
            z.dim = len(lay.axes)
            b = [0.0] * 6
            for i, a in enumerate(lay.axes):
                c = lay.coords[a]
                if c.size:
                    b[2 * i], b[2 * i + 1] = float(np.nanmin(c)), float(np.nanmax(c))
            z.bounds = b
        else:
            z.n_points, z.n_cells = lay.n_points, lay.n_cells
            ut = lay.uniform_cell_type()
            if ut is not None and lay.n_cells is not None:
                z.cell_types = {meshutil.VTK_TYPE_NAMES.get(ut, f"vtk_{ut}"): lay.n_cells}
        desc.zones.append(z)
        transient = lay.times is not None
        for group, static in ((lay.fields, False), (lay.static, True)):
            for name, p in group.items():
                key = (name, p.location)
                if key in seen:
                    if seen[key].zones is not None:
                        seen[key].zones.append(lay.zone)  # type: ignore[union-attr]
                    continue
                fi = FieldInfo(
                    name=name,
                    location=p.location,
                    components=p.components,
                    units=p.units,
                    component_names=p.component_names,
                    dtype="float32",
                    description=p.description,
                    time_dependent=transient and p.time_dependent and not static,
                    zones=[lay.zone] if multi else None,
                )
                seen[key] = fi
                desc.fields.append(fi)
        desc.warnings.extend(w for w in lay.warnings if w not in desc.warnings)
    desc.warnings.extend(w for w in layouts[0].store.warnings if w not in desc.warnings)
    timed = next((lay for lay in layouts if lay.times is not None), None)
    if timed is not None:
        t = timed.times_for(0)
        assert t is not None
        desc.time = TimeInfo(
            values=[float(v) for v in t],
            units=timed.time_units,
            quantity=str(timed.time_meta.get("time_quantity", "time")),
        )
        desc.metadata.update({k: v for k, v in timed.time_meta.items() if k != "time_quantity"})
    first = layouts[0]
    desc.domain = next((lay.domain for lay in layouts if lay.domain != "other"), "other")
    desc.solver = str(first.metadata.get("solver") or "unknown")
    if first.metadata.get("solver_version"):
        desc.solver_version = str(first.metadata["solver_version"])
    desc.physics = dict(first.metadata.get("physics") or {})
    desc.boundary_conditions = dict(first.metadata.get("boundary_conditions") or {})
    desc.units = {str(k): str(v) for k, v in (first.metadata.get("units") or {}).items()}
    for key, v in first.parameters.items():
        if isinstance(v, np.ndarray):
            desc.parameters[key] = (
                v.tolist()
                if v.size <= 32
                else {"min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
            )
        else:
            desc.parameters[key] = v
    n_samples = [len(lay.groups) if lay.groups else lay.n_samples for lay in layouts]
    if any(n is not None for n in n_samples):
        desc.metadata["n_samples"] = int(sum(n or 1 for n in n_samples))
        desc.metadata["sample_ids"] = _sample_labels(first)[:20]
    axis_names = {lay.zone: lay.axis_names for lay in layouts if lay.axis_names}
    if axis_names:
        desc.metadata["axis_names"] = axis_names if multi else next(iter(axis_names.values()))
    desc.metadata["mapping"] = public_mapping(plan.mapping)
    desc.metadata["mapping_source"] = "auto-detected" if plan.auto else "user"
    attrs = {
        k: v
        for k, v in first.store.group_attrs.get("", {}).items()
        if isinstance(v, str | int | float) and len(str(v)) <= 400
    }
    if attrs:
        desc.metadata["attributes"] = attrs
    return desc


def read_layouts(
    plan: Plan,
    path: Path,
    *,
    zones: Sequence[str] | None,
    fields: Sequence[str] | None,
    times: TimeSelection,
    options: Mapping[str, Any] | None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Iterator[RawSample]:
    layouts = plan.layouts
    selected = match_names([lay.zone for lay in layouts], zones)
    if fields is not None:
        names = dict.fromkeys(n for lay in layouts for n in lay.fields)
        match_names(names, [f for f in fields if not f.startswith("!")], what="field")
    base = meshutil.default_sample_id(path)
    opts = options or {}
    for lay in layouts:
        if lay.zone not in selected:
            continue
        zid = meshutil.zone_sample_id(base, lay.zone, len(selected))
        md = {"source": str(path), **dict(extra_metadata or {})}
        if lay.groups:
            labels = _sample_labels(lay)
            for i in select_samples(opts.get("samples"), len(lay.groups), labels):
                sub = lay
                if i:
                    sub = resolve_mapping(
                        lay.store, lay.mapping, zone=lay.zone, prefix=lay.groups[i]
                    )
                yield _sample(sub, f"{zid}_{labels[i]}", None, fields, times, md, labels[i])
        elif lay.n_samples is not None:
            labels = _sample_labels(lay)
            for i in select_samples(opts.get("samples"), lay.n_samples, labels):
                yield _sample(lay, f"{zid}_{labels[i]}", i, fields, times, md, labels[i])
        else:
            yield _sample(lay, zid, None, fields, times, md, None)


def _sample(
    lay: Layout,
    sid: str,
    index: int | None,
    fields: Sequence[str] | None,
    times: TimeSelection,
    metadata: Mapping[str, Any],
    label: str | None,
) -> RawSample:
    names = [n for n in lay.fields if wants(n, fields)]
    tvals = lay.times_for(index)
    transient = tvals is not None and any(lay.fields[n].time_dependent for n in names)
    geo = lay.load_geometry()
    md: dict[str, Any] = {"solver": "unknown", **dict(metadata)}
    md.update({k: v for k, v in lay.metadata.items() if v is not None})
    params: dict[str, float] = {}
    for key, v in lay.parameters.items():
        val = v[index or 0] if isinstance(v, np.ndarray) else v
        if isinstance(val, int | float | np.integer | np.floating):
            params[key] = float(val)
        else:
            md.setdefault("attributes", {})[key] = val
    if params:
        md["parameters"] = params
    if lay.axis_names:
        md["axis_names"] = dict(lay.axis_names)
    if label is not None:
        md["sample"] = label
    sample = RawSample(
        id=sid,
        kind=lay.kind,
        zone=lay.zone,
        points=geo["points"],
        cells=geo["cells"],
        cell_types=geo["cell_types"],
        faces=geo["faces"],
        coords={a: lay.coords[a] for a in lay.axes},
        metadata=md,
    )
    sel: list[int] | None = None
    if transient:
        assert tvals is not None
        picked = select_time_indices(tvals, times)
        sample.times = np.asarray(tvals, dtype=np.float64)[picked]
        sel = None if picked == list(range(len(tvals))) else picked
        md["time_units"] = lay.time_units
        md.update({k: v for k, v in lay.time_meta.items() if k != "time_quantity"})

    def data(p: FieldPlan, name: str, lazy: bool) -> FieldData:
        arr: Any = p.lazy(sample=index, time_index=sel, entity_index=geo["parent"])
        dims = (["time"] if p.time_dependent else []) + p.space() + ["component"]
        return FieldData(
            data=arr if lazy else np.asarray(arr),
            dims=dims,
            location=p.location,
            units=p.units,
            component_names=p.component_names,
            description=p.description,
        )

    for name in names:
        p = lay.fields[name]
        if transient and p.time_dependent:
            sample.fields[name] = data(p, name, True)
        elif transient:
            sample.static[name] = data(p, name, False)
        else:
            sample.fields[name] = data(p, name, False)
    for name, p in lay.static.items():
        sample.static[name] = data(p, name, False)
    return sample
