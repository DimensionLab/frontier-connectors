# SPDX-License-Identifier: Apache-2.0
"""``RawSample`` stream -> ``frontier-dataset v1.1`` directory.

Two writers produce the same on-disk layout:

* ``frontier_data`` (``DatasetWriter.write_{surface,mesh,grid,points}_sample``) is used when it
  is importable and the sample is complete in memory (steady, or an in-memory time block).
* a built-in **streaming** zarr v3 writer is used for step-form transients and lazy blocks: it
  appends one time step at a time, so a 10 000-step case never sits in RAM. It is also the
  fallback when ``frontier_data`` is not installed (the connectors package is usable alone).

``ConvertResult.backends`` records which one wrote each sample.
"""

from __future__ import annotations

import json
import shutil
import time as _time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError
from frontier_connectors.model import FieldData, RawSample, SourceDescription, jsonable

FORMAT_NAME = "frontier-dataset"
FORMAT_VERSION = 1
FORMAT_MINOR = 1
STORE_NAMES = {
    "surface": "surface.zarr",
    "mesh": "mesh.zarr",
    "grid": "grid.zarr",
    "point_cloud": "points.zarr",
}
DOMAINS = ("cfd", "structural", "thermal", "multiphysics", "materials", "geophysical", "other")
CHUNK_ROWS = 1 << 18
STEP_CHUNK_BYTES = 64 << 20
#: steps smaller than this are grouped several per chunk and buffered before writing, so a
#: 10 000-step probe series does not become 10 000 tiny chunk files written one by one
SMALL_STEP_BYTES = 64 << 10
SMALL_STEPS_PER_CHUNK = 16
FLUSH_BYTES = 4 << 20
GRID_AXES = ("x", "y", "z")


@dataclass
class ConvertResult:
    out: Path
    name: str
    sample_ids: list[str] = field(default_factory=list)
    #: sample id -> "frontier_data" | "zarr-stream"
    backends: dict[str, str] = field(default_factory=dict)
    n_steps: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    #: a multi-case conversion (`frontier_connectors.cases`): the cases with their parameters,
    #: where the parameters came from, and the splits assigned by case
    cases: list[dict[str, Any]] = field(default_factory=list)
    parameters_from: str | None = None
    splits: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        multi: dict[str, Any] = {}
        if self.cases:
            multi = {
                "n_cases": len(self.cases),
                "cases": self.cases,
                "parameters": sorted({k for c in self.cases for k in c["parameters"]}),
                "parameters_from": self.parameters_from,
                "splits": self.splits,
            }
        return {
            **multi,
            "out": str(self.out),
            "name": self.name,
            "n_samples": len(self.sample_ids),
            "sample_ids": self.sample_ids,
            "backends": self.backends,
            "n_steps": self.n_steps,
            "warnings": self.warnings,
            "seconds": round(self.seconds, 3),
        }


def frontier_data_available() -> bool:
    """``True`` when ``frontier_data`` with the v1.1 writers is importable."""
    try:
        from frontier_data.dataset import DatasetWriter  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - any import problem means "use the built-in writer"
        return False
    return all(
        hasattr(DatasetWriter, n)
        for n in ("write_mesh_sample", "write_grid_sample", "write_points_sample")
    )


# -- zarr helpers ---------------------------------------------------------------------------


def _compressors() -> list[Any]:
    from zarr.codecs import BloscCodec

    return [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")]


def _safe(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").strip() or "field"


def _case_unique(
    s: RawSample, taken: dict[str, str], renames: dict[str, str], warnings: list[str]
) -> None:
    """Rename fields whose stored name differs from another one only in case.

    Field names become directory names of the zarr store, and macOS and Windows file systems
    are case-insensitive: ``Extra`` and ``extra`` would silently share one array. The later
    name gets a numeric suffix (``extra_2``). ``taken`` (folded name -> stored name) and
    ``renames`` (source name -> stored name) belong to one sample id and persist across its
    time steps, so every step of a transient uses the same names."""
    for group in (s.fields, s.static):
        for name in list(group):
            if name in renames:
                new = renames[name]
            else:
                low = _safe(name).casefold()
                if taken.setdefault(low, name) == name:
                    continue
                k = 2
                while f"{low}_{k}" in taken:
                    k += 1
                new = f"{_safe(name)}_{k}"
                taken[f"{low}_{k}"] = new
                renames[name] = new
                warnings.append(
                    f"{s.id}: field {name!r} differs from {taken[low]!r} only in case; "
                    f"stored as {new!r}"
                )
            f = group.pop(name)
            if not f.description:
                f.description = f"source name: {name}"
            group[new] = f


def _row_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return (1,)
    return (min(CHUNK_ROWS, max(shape[0], 1)), *(max(s, 1) for s in shape[1:]))


def _step_chunks(shape: tuple[int, ...], itemsize: int, steps: int) -> tuple[int, ...]:
    rest = shape[1:]
    if not rest:
        return (max(steps, 1),)
    step_bytes = int(np.prod(rest, dtype=np.int64)) * itemsize
    rows = rest[0]
    if step_bytes > STEP_CHUNK_BYTES and rows > 1:
        rows = max(1, int(rows * STEP_CHUNK_BYTES / step_bytes))
    return (max(steps, 1), max(min(rows, rest[0]), 1), *(max(s, 1) for s in rest[1:]))


def _put(group: Any, name: str, data: np.ndarray, dims: list[str]) -> Any:
    data = np.ascontiguousarray(data)
    arr = group.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=_row_chunks(data.shape),
        compressors=_compressors(),
    )
    if data.size:
        arr[...] = data
    arr.attrs["dims"] = list(dims)
    return arr


def _digest(conn: np.ndarray) -> bytes:
    import hashlib

    return hashlib.blake2b(np.ascontiguousarray(conn).tobytes(), digest_size=16).digest()


def _as_f32(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype.kind == "c":
        raise ConnectorError(
            "complex-valued field cannot be stored",
            hint="split it into real and imaginary parts in the connector (name_re / name_im)",
        )
    return arr.astype(np.float32, copy=False)


def _field_group(kind: str, f: FieldData) -> str:
    return "cell_fields" if f.location == "cell" and kind in ("surface", "mesh") else "fields"


def _field_attrs(arr: Any, f: FieldData, source_name: str, dims: list[str]) -> None:
    attrs: dict[str, Any] = {"dims": dims, "location": f.location}
    if f.units:
        attrs["units"] = f.units
    if f.component_names:
        attrs["component_names"] = list(f.component_names)
    if f.description:
        attrs["description"] = f.description
    if source_name != arr.basename:
        attrs["source_name"] = source_name
    arr.attrs.update(attrs)


# -- streaming writer -----------------------------------------------------------------------


class _StreamWriter:
    """Writes one dataset sample incrementally (append along time)."""

    def __init__(self, root: Path, first: RawSample, dataset: str, steps_per_chunk: int) -> None:
        import zarr

        self.root = root
        self.id = first.id
        self.kind = first.kind
        self.dataset = dataset
        self.spc = max(int(steps_per_chunk), 1)
        self.t0 = _time.perf_counter()
        self.sdir = root / "samples" / first.id
        self.sdir.mkdir(parents=True, exist_ok=True)
        for other in STORE_NAMES.values():
            if (self.sdir / other).exists():
                shutil.rmtree(self.sdir / other)
        self.store: Any = zarr.open_group(
            str(self.sdir / STORE_NAMES[self.kind]), mode="w", zarr_format=3
        )
        self.times: list[float] = []
        self.transient = first.transient
        self.metadata = dict(first.metadata)
        self.zone = first.zone
        self.infos: dict[str, dict[str, Any]] = {}
        self.static_names: list[str] = []
        self.arrays: dict[tuple[str, str], Any] = {}
        self.pending: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
        self.ref_points: np.ndarray | None = None
        self.last_points: np.ndarray | None = None
        self.points_t: Any = None
        self.n_cells = 0
        self.shape: list[int] | None = None
        self.area = 0.0
        self.bounds = [0.0] * 6
        self.coords: dict[str, np.ndarray] = {}
        self.conn_digest: bytes | None = None
        self._geometry(first)

    # geometry ----------------------------------------------------------------------------
    def _geometry(self, s: RawSample) -> None:
        kind = self.kind
        if kind == "grid":
            if not s.coords:
                raise ConnectorError(
                    f"grid sample {s.id!r} has no coords",
                    hint="connector bug: set RawSample.coords['x'|'y'|'z'] (1-D axes)",
                )
            axes = [a for a in GRID_AXES if a in s.coords]
            cg = self.store.create_group("coords")
            for ax in axes:
                c = np.asarray(s.coords[ax], dtype=np.float64).reshape(-1)
                self.coords[ax] = c
                _put(cg, ax, c, [ax])
            self.shape = [len(self.coords[a]) for a in axes]
            for i, ax in enumerate(axes):
                self.bounds[2 * i] = float(self.coords[ax].min())
                self.bounds[2 * i + 1] = float(self.coords[ax].max())
            return
        if s.points is None:
            raise ConnectorError(
                f"first sample of {s.id!r} carries no points",
                hint="connector bug: the first RawSample of an id must include the geometry",
            )
        pts = np.ascontiguousarray(np.asarray(s.points, dtype=np.float32))
        if pts.ndim != 2:
            raise ConnectorError(f"sample {s.id!r}: points must be (N, D), got {pts.shape}")
        self.ref_points = pts
        self.last_points = pts
        self.bounds = meshutil.bounds(pts)
        _put(self.store, "points", pts, ["point", "space"])
        if kind == "surface":
            if s.faces is None:
                raise ConnectorError(f"surface sample {s.id!r} has no faces")
            faces = np.ascontiguousarray(np.asarray(s.faces, dtype=np.int32))
            self._check_conn(faces, len(pts), s.id)
            self.conn_digest = _digest(faces)
            area = meshutil.triangle_areas(pts, faces)
            self.area = float(area.sum())
            self.n_cells = len(faces)
            _put(self.store, "faces", faces, ["cell", "cell_node"])
            _put(self.store, "cell_area", area.astype(np.float32), ["cell"])
            self.store.create_group("lod")
        elif kind == "mesh":
            if s.cells is None or s.cell_types is None:
                raise ConnectorError(f"mesh sample {s.id!r} has no cells/cell_types")
            cells = np.ascontiguousarray(np.asarray(s.cells, dtype=np.int32))
            self._check_conn(cells, len(pts), s.id)
            self.conn_digest = _digest(cells)
            self.n_cells = len(cells)
            _put(self.store, "cells", cells, ["cell", "cell_node"])
            _put(
                self.store,
                "cell_types",
                np.asarray(s.cell_types, dtype=np.uint8).reshape(-1),
                ["cell"],
            )

    @staticmethod
    def _check_conn(conn: np.ndarray, n_points: int, sid: str) -> None:
        if conn.size and (int(conn.max()) >= n_points or int(conn.min()) < -1):
            raise ConnectorError(
                f"sample {sid!r}: connectivity references points outside [0, {n_points})",
                hint="connector bug: cells/faces must be 0-based indices into points",
            )

    @property
    def n_points(self) -> int:
        if self.kind == "grid":
            return int(np.prod(self.shape or [0]))
        return 0 if self.ref_points is None else len(self.ref_points)

    # fields ------------------------------------------------------------------------------
    def _entity_len(self, f: FieldData) -> int | None:
        if self.kind == "grid":
            return None
        return self.n_cells if f.location == "cell" else self.n_points

    def _record(self, name: str, f: FieldData, dims: list[str], channels: int) -> None:
        self.infos[name] = {
            "name": name,
            "units": f.units or "-",
            "description": f.description,
            "channels": int(channels),
            "location": f.location,
            "dims": dims,
            "temporal": bool(dims) and dims[0] == "time",
        }

    def _write_static(self, name: str, f: FieldData) -> None:
        key = _safe(name)
        if key in self.static_names:
            return
        a = np.asarray(f.data)
        if a.dtype.kind == "f":
            a = a.astype(np.float32, copy=False)
        elif a.dtype.kind == "b":
            a = a.astype(np.uint8)
        group = (
            self.store["static"] if "static" in self.store else self.store.create_group("static")
        )
        arr = _put(group, key, a, [d for d in f.dims if d != "time"])
        _field_attrs(arr, f, name, [d for d in f.dims if d != "time"])
        self.static_names.append(key)

    def _time_array(self, name: str, f: FieldData, step_shape: tuple[int, ...], dims: list[str]):
        gname = _field_group(self.kind, f)
        key = (gname, _safe(name))
        if key in self.arrays:
            return self.arrays[key]
        group = self.store[gname] if gname in self.store else self.store.create_group(gname)
        shape = (0, *step_shape)
        step_bytes = int(np.prod(step_shape, dtype=np.int64)) * 4
        spc = self.spc
        if spc == 1 and step_bytes < SMALL_STEP_BYTES:
            spc = SMALL_STEPS_PER_CHUNK
        arr = group.create_array(
            key[1],
            shape=shape,
            dtype=np.float32,
            chunks=_step_chunks((1, *step_shape), 4, spc),
            compressors=_compressors(),
            fill_value=float("nan"),
        )
        _field_attrs(arr, f, name, dims)
        self.arrays[key] = arr
        self._record(key[1], f, dims, step_shape[-1])
        return arr

    def _check_len(self, name: str, f: FieldData, n: int) -> None:
        want = self._entity_len(f)
        if want is not None and n != want:
            raise ConnectorError(
                f"sample {self.id!r} field {name!r}: {n} {f.location} values but the mesh has "
                f"{want} {f.location}s",
                hint="the topology changed between time steps; convert with time_as_samples=True "
                "(CLI: --time-as-samples) to store every step as its own sample",
            )

    def add(self, s: RawSample) -> None:
        """Append one step (step form), a block of steps (block form) or steady data."""
        for name, f in s.static.items():
            self._write_static(name, f)
        if s.block_form:
            self._add_block(s)
            return
        if s.time is None:
            if self.transient and self.times:
                raise ConnectorError(
                    f"sample {self.id!r}: steady data arrived after transient steps",
                    hint="connector bug: do not mix steady and step-form samples under one id",
                )
            self._add_steady(s)
            return
        t = len(self.times)
        if self.times and s.time <= self.times[-1]:
            raise ConnectorError(
                f"sample {self.id!r}: time {s.time} is not after {self.times[-1]}",
                hint="connector bug: yield the steps of one sample id in increasing time order",
            )
        self.times.append(float(s.time))
        self._moving_points(s, t)
        for name, f in s.fields.items():
            f.validate(name)
            data = _as_f32(f.data)
            if self.kind != "grid":
                self._check_len(name, f, data.shape[0])
            dims = ["time", *f.dims]
            arr = self._time_array(name, f, tuple(data.shape), dims)
            if tuple(arr.shape[1:]) != tuple(data.shape):
                raise ConnectorError(
                    f"sample {self.id!r} field {name!r}: step shape {data.shape} differs from "
                    f"the first step {tuple(arr.shape[1:])}",
                    hint="convert with time_as_samples=True (CLI: --time-as-samples)",
                )
            self._buffer((_field_group(self.kind, f), _safe(name)), arr, t, data)

    def _buffer(self, key: tuple[str, str], arr: Any, t: int, data: np.ndarray) -> None:
        """Queue one step; write when a few megabytes (or a whole number of chunks) are ready."""
        queue = self.pending.setdefault(key, [])
        # the reader may reuse its buffer for the next step, so keep a private copy
        queue.append((t, np.array(data, dtype=np.float32, copy=True)))
        step_bytes = max(data.nbytes, 1)
        per_chunk = int(arr.chunks[0])
        limit = max(per_chunk, (FLUSH_BYTES // step_bytes) // per_chunk * per_chunk)
        if len(queue) >= limit:
            self._flush(key)

    def _flush(self, key: tuple[str, str] | None = None) -> None:
        for k in [key] if key is not None else list(self.pending):
            queue = self.pending.pop(k, [])
            if not queue:
                continue
            arr = self.arrays[k]
            first, last = queue[0][0], queue[-1][0]
            if arr.shape[0] < last + 1:
                arr.resize((last + 1, *arr.shape[1:]))
            if last - first + 1 == len(queue):
                arr[first : last + 1] = np.stack([d for _, d in queue])
            else:  # the field was absent at some steps: those stay NaN
                for t, d in queue:
                    arr[t] = d

    def _moving_points(self, s: RawSample, t: int) -> None:
        if self.kind == "grid" or self.ref_points is None:
            return
        pts = None if s.points is None else np.asarray(s.points, dtype=np.float32)
        if pts is not None and pts.shape != self.ref_points.shape:
            raise ConnectorError(
                f"sample {self.id!r}: step {t} has {pts.shape[0]} points, the first step has "
                f"{self.ref_points.shape[0]}",
                hint="the mesh topology changes in time (remeshing/adaptation); convert with "
                "time_as_samples=True (CLI: --time-as-samples)",
            )
        conn = s.faces if self.kind == "surface" else s.cells
        if t > 0 and conn is not None and self.conn_digest is not None:
            if _digest(np.ascontiguousarray(np.asarray(conn, dtype=np.int32))) != self.conn_digest:
                raise ConnectorError(
                    f"sample {self.id!r}: step {t} has a different connectivity than the first "
                    "step (same counts, different cells)",
                    hint="the mesh is rebuilt every step; convert with time_as_samples=True "
                    "(CLI: --time-as-samples)",
                )
        if self.points_t is None:
            if pts is None or t == 0 or np.array_equal(pts, self.ref_points):
                return
            self.points_t = self.store.create_array(
                "points_t",
                shape=(t, *self.ref_points.shape),
                dtype=np.float32,
                chunks=_step_chunks((1, *self.ref_points.shape), 4, self.spc),
                compressors=_compressors(),
            )
            self.points_t.attrs["dims"] = ["time", "point", "space"]
            for i in range(t):
                self.points_t[i] = self.ref_points
        if pts is not None:
            self.last_points = pts
        self.points_t.resize((t + 1, *self.ref_points.shape))
        self.points_t[t] = self.last_points

    def _add_steady(self, s: RawSample) -> None:
        for name, f in s.fields.items():
            f.validate(name)
            data = _as_f32(f.data)
            if self.kind != "grid":
                self._check_len(name, f, data.shape[0])
            gname = _field_group(self.kind, f)
            key = _safe(name)
            group = self.store[gname] if gname in self.store else self.store.create_group(gname)
            if key in group:
                continue
            arr = _put(group, key, data, list(f.dims))
            _field_attrs(arr, f, name, list(f.dims))
            self._record(key, f, list(f.dims), data.shape[-1])

    def _add_block(self, s: RawSample) -> None:
        assert s.times is not None
        times = np.asarray(s.times, dtype=np.float64).reshape(-1)
        base = len(self.times)
        if base and times.size and times[0] <= self.times[-1]:
            raise ConnectorError(f"sample {self.id!r}: time blocks overlap or are out of order")
        self._flush()
        self.times.extend(float(v) for v in times)
        self.transient = True
        for name, f in s.fields.items():
            f.validate(name)
            if not f.time_dependent:
                self._add_steady(RawSample(id=s.id, kind=s.kind, fields={name: f}))
                continue
            shape = f.shape
            if shape[0] != times.size:
                raise ConnectorError(
                    f"sample {self.id!r} field {name!r}: {shape[0]} steps but {times.size} times"
                )
            if self.kind != "grid":
                self._check_len(name, f, shape[1])
            arr = self._time_array(name, f, tuple(shape[1:]), list(f.dims))
            arr.resize((base + shape[0], *arr.shape[1:]))
            step_bytes = max(int(np.prod(shape[1:], dtype=np.int64)) * 4, 1)
            batch = max(1, min(shape[0], STEP_CHUNK_BYTES // step_bytes))
            for i in range(0, shape[0], batch):
                arr[base + i : base + i + batch] = _as_f32(f.data[i : i + batch])

    # finish ------------------------------------------------------------------------------
    def close(self) -> dict[str, Any]:
        self._flush()
        n_steps = len(self.times) if self.transient else 1
        if self.transient:
            for arr in self.arrays.values():
                if arr.shape[0] != n_steps:
                    arr.resize((n_steps, *arr.shape[1:]))
            if self.points_t is not None and self.points_t.shape[0] != n_steps:
                self.points_t.resize((n_steps, *self.points_t.shape[1:]))
            tarr = self.store.create_array(
                "times", shape=(n_steps,), dtype=np.float64, chunks=(max(n_steps, 1),)
            )
            tarr[...] = np.asarray(self.times, dtype=np.float64)
            tarr.attrs["dims"] = ["time"]
        attrs: dict[str, Any] = {
            "format": f"{FORMAT_NAME}/{self.kind}",
            "version": FORMAT_VERSION,
            "minor": FORMAT_MINOR,
            "temporal": "transient" if self.transient else "steady",
            "n_steps": n_steps,
            "static": self.static_names,
            "writer": "frontier-connectors/zarr-stream",
        }
        if self.kind == "grid":
            axes = list(self.coords)
            steps = [np.diff(self.coords[a]) for a in axes]
            uniform = all(d.size == 0 or np.allclose(d, d[0], rtol=1e-6, atol=0) for d in steps)
            attrs.update(
                {
                    "shape": self.shape,
                    "axes": axes,
                    "spacing": [float(d[0]) if d.size else 1.0 for d in steps] if uniform else None,
                    "origin": [float(self.coords[a][0]) for a in axes],
                    "boundary_conditions": jsonable(self.metadata.get("boundary_conditions", {})),
                }
            )
        else:
            attrs.update(
                {
                    "n_points": self.n_points,
                    "bounds": self.bounds,
                    "point_dim": int(self.ref_points.shape[1])
                    if self.ref_points is not None
                    else 3,
                }
            )
            if self.kind == "surface":
                attrs.update({"n_faces": self.n_cells, "area": self.area, "lod_levels": [0]})
            elif self.kind == "mesh":
                attrs["n_cells"] = self.n_cells
        self.store.attrs.update(attrs)
        time_units = str(self.metadata.get("time_units", "s"))
        meta = build_meta(
            self.id,
            self.dataset,
            self.kind,
            self.metadata,
            zone=self.zone,
            n_points=self.n_points,
            n_cells=self.n_cells,
            bounds=self.bounds,
            area=self.area,
            times=np.asarray(self.times) if self.transient else None,
            time_units=time_units,
            fields=list(self.infos.values()),
        )
        meta["extra"]["timings"] = {"write_s": _time.perf_counter() - self.t0}
        _write_json(self.sdir / "meta.json", meta)
        return entry_from_meta(meta, shape=self.shape, n_cells=self.n_cells)


# -- meta / manifest ------------------------------------------------------------------------


def _time_info(times: np.ndarray, units: str) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return {
            "n_steps": int(t.size),
            "dt": None,
            "units": units,
            "uniform": True,
            "t0": float(t[0]) if t.size else 0.0,
            "averaging_window": None,
        }
    d = np.diff(t)
    dt = float(np.median(d))
    uniform = bool(np.all(np.abs(d - dt) <= 1e-4 * max(abs(dt), 1e-300)))
    return {
        "n_steps": int(t.size),
        "dt": dt if uniform else None,
        "units": units,
        "uniform": uniform,
        "t0": float(t[0]),
        "averaging_window": None,
    }


def _float_params(d: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float | np.integer | np.floating) and np.isfinite(v):
            out[str(k)] = float(v)
    return out


def build_meta(
    sample_id: str,
    dataset: str,
    kind: str,
    metadata: Mapping[str, Any],
    *,
    zone: str | None,
    n_points: int,
    n_cells: int,
    bounds: list[float],
    area: float,
    times: np.ndarray | None,
    time_units: str,
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    """``meta.json`` content (keys of ``frontier_data.dataset.SampleMeta``) from the free-form
    ``RawSample.metadata``. Unknown metadata keys are kept under ``extra``."""
    md = dict(metadata)
    physics = md.pop("physics", None)
    phys = physics if isinstance(physics, dict) else {}
    known = {
        "solver": str(md.pop("solver", "unknown") or "unknown"),
        "solver_version": md.pop("solver_version", None),
        "application": md.pop("application", None) or phys.get("application"),
        "turbulence_model": md.pop("turbulence_model", None) or phys.get("turbulence_model"),
        "turbulence_family": md.pop("turbulence_family", None) or phys.get("turbulence_family"),
        "physics": physics if isinstance(physics, str) else phys.get("model"),
        "fidelity": md.pop("fidelity", None),
        "fidelity_tier": md.pop("fidelity_tier", None),
    }
    for k in ("solver_version", "application", "turbulence_model", "turbulence_family", "physics"):
        if known[k] is not None:
            known[k] = str(known[k])
    source = str(md.pop("source", "") or dataset or "unknown")
    params = _float_params(md.pop("parameters", None))
    group = md.pop("group", None)
    time_meta = md.pop("time", None)
    md.pop("time_units", None)
    tinfo = _time_info(times, time_units) if times is not None else None
    if tinfo is None and isinstance(time_meta, Mapping):
        tinfo = {**_time_info(np.zeros(0), time_units), **dict(time_meta)}
        tinfo["n_steps"] = int(tinfo.get("n_steps") or 1)
    extra = jsonable(md)
    if phys:
        extra["physics"] = jsonable(phys)
    if zone is not None:
        extra["zone"] = zone
    return {
        "id": sample_id,
        "dataset": dataset,
        **known,
        "mesh": {
            "n_points": int(n_points),
            "n_faces": int(n_cells),
            "bounds": [float(b) for b in bounds],
            "area": float(area),
        },
        "geometry_params": _float_params(extra.pop("geometry_params", None)),
        "conditions": jsonable(extra.pop("conditions", {}) or {}),
        "boundary_conditions": jsonable(extra.pop("boundary_conditions", {}) or {}),
        "forces": jsonable(extra.pop("forces", {}) or {}),
        "fields": fields,
        "lod_levels": [0],
        "provenance": {
            "source": source,
            "license": extra.pop("license", None),
            "citation": extra.pop("citation", None),
            "files": [],
            "ingested_at": datetime.now(UTC).isoformat(),
            "ingest_tool": "frontier-connectors",
        },
        "sample_kind": kind,
        "temporal": "transient" if times is not None else "steady",
        "time": tinfo,
        "parameters": params,
        "group": group,
        "extra": extra,
    }


def entry_from_meta(
    meta: Mapping[str, Any], *, shape: list[int] | None, n_cells: int
) -> dict[str, Any]:
    t = meta.get("time") or {}
    return {
        "id": meta["id"],
        "path": f"samples/{meta['id']}",
        "n_points": meta["mesh"]["n_points"],
        "n_faces": meta["mesh"]["n_faces"] if meta["sample_kind"] == "surface" else 0,
        "lod_levels": [0],
        "geometry_params": dict(meta.get("geometry_params") or {}),
        "forces": {},
        "sample_kind": meta["sample_kind"],
        "temporal": meta["temporal"],
        "n_steps": int(t.get("n_steps") or 1) if meta["temporal"] == "transient" else 1,
        "n_cells": int(n_cells),
        "shape": shape,
        "parameters": dict(meta.get("parameters") or {}),
        "group": meta.get("group"),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2))
    tmp.replace(path)


# -- frontier_data backend -------------------------------------------------------------------


def _in_memory(s: RawSample) -> bool:
    arrays = [f.data for f in s.fields.values()] + [f.data for f in s.static.values()]
    return all(isinstance(a, np.ndarray) and not isinstance(a, np.memmap) for a in arrays)


def _write_with_frontier_data(dw: Any, s: RawSample, dataset: str, steps_per_chunk: int) -> None:
    """Hand a complete in-memory sample to the ``frontier_data`` v1.1 writers."""
    s.validate()
    times = None if s.times is None else np.asarray(s.times, dtype=np.float64)
    point_f: dict[str, np.ndarray] = {}
    cell_f: dict[str, np.ndarray] = {}
    static: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    dims: dict[str, list[str]] = {}
    for name, f in s.fields.items():
        key = _safe(name)
        data = _as_f32(f.data)
        if times is not None and not f.time_dependent:
            static[key] = data
            dims[key] = list(f.dims)
            continue
        (cell_f if _field_group(s.kind, f) == "cell_fields" else point_f)[key] = data
        if f.units:
            units[key] = f.units
        if s.kind == "grid":
            dims[key] = list(f.dims)
    for name, f in s.static.items():
        static[_safe(name)] = np.asarray(f.data)
        dims[_safe(name)] = list(f.dims)
    n_cells = s.n_cells
    meta = build_meta(
        s.id,
        dataset,
        s.kind,
        s.metadata,
        zone=s.zone,
        n_points=s.n_points,
        n_cells=n_cells,
        bounds=meshutil.bounds(s.points),
        area=0.0,
        times=times,
        time_units=str(s.metadata.get("time_units", "s")),
        fields=[],
    )
    for k in ("id", "dataset", "mesh", "fields", "lod_levels", "sample_kind", "temporal", "time"):
        meta.pop(k, None)
    params = meta.pop("parameters")
    group = meta.pop("group")
    common: dict[str, Any] = {
        "times": times,
        "static": static or None,
        "meta": meta,
        "parameters": params or None,
        "group": group,
        "field_units": units or None,
        "dims": dims or None,
        "steps_per_chunk": steps_per_chunk,
        "time_units": str(s.metadata.get("time_units", "s")),
    }
    if s.kind == "surface":
        from frontier_data.mesh import SurfaceMesh

        assert s.points is not None and s.faces is not None
        pts = np.asarray(s.points, dtype=np.float32)
        if pts.shape[1] == 2:
            pts = np.concatenate([pts, np.zeros((len(pts), 1), np.float32)], axis=1)
        mesh = SurfaceMesh(points=pts, faces=np.asarray(s.faces, dtype=np.int32))
        dw.write_surface_sample(
            s.id, mesh, fields=point_f or None, cell_fields=cell_f or None, **common
        )
    elif s.kind == "mesh":
        dw.write_mesh_sample(
            s.id,
            points=s.points,
            cells=s.cells,
            cell_types=s.cell_types,
            fields=point_f or None,
            cell_fields=cell_f or None,
            **common,
        )
    elif s.kind == "grid":
        axes = [a for a in GRID_AXES if a in s.coords]
        dw.write_grid_sample(
            s.id,
            fields=point_f,
            coords={a: np.asarray(s.coords[a], dtype=np.float64) for a in axes},
            ndim=len(axes),
            boundary_conditions=jsonable(s.metadata.get("boundary_conditions") or {}) or None,
            **common,
        )
    else:
        dw.write_points_sample(s.id, points=s.points, fields=point_f or None, **common)


# -- public API ------------------------------------------------------------------------------


def to_frontier_dataset(
    samples: Iterable[RawSample],
    out: str | Path,
    *,
    name: str | None = None,
    description: SourceDescription | None = None,
    domain: str | None = None,
    license: str | None = None,  # noqa: A002 - contract field name
    citation: str | None = None,
    source: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    time_as_samples: bool = False,
    steps_per_chunk: int = 1,
    backend: str = "auto",
) -> ConvertResult:
    """Write a ``RawSample`` stream as a ``frontier-dataset v1.1`` directory.

    Samples are consumed lazily; samples that share an ``id`` are appended along time, so the
    iterator may yield one time step at a time. ``backend`` is ``"auto"`` (``frontier_data``
    writers for complete in-memory samples when installed, streaming zarr otherwise),
    ``"frontier_data"`` or ``"zarr"``. ``time_as_samples`` stores every step as its own steady
    sample (``<id>_t00012``, ``group = <id>``) for sources whose mesh changes in time.
    """
    t_start = _time.perf_counter()
    out = Path(out)
    if backend not in ("auto", "frontier_data", "zarr"):
        raise ConnectorError(f"unknown backend {backend!r}", hint="use auto | frontier_data | zarr")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise ConnectorError(
                f"{out} exists and is not empty", hint="pass overwrite=True (CLI: --overwrite)"
            )
        if not (out / "manifest.json").exists():
            raise ConnectorError(
                f"refusing to overwrite {out}: it is not a frontier-dataset directory",
                hint="choose an empty or new --out directory",
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    ds_name = name or out.name
    domain = domain or (description.domain if description else None) or "other"
    if domain not in DOMAINS:
        raise ConnectorError(f"unknown domain {domain!r}", hint="one of: " + ", ".join(DOMAINS))
    src: dict[str, Any] = dict(source or {})
    if description is not None:
        src.setdefault("connector", description.connector_id)
        src.setdefault("path", str(description.path))
        src.setdefault("format", description.format)
        src.setdefault("solver", description.solver)
    result = ConvertResult(out=out, name=ds_name)

    use_fd = backend in ("auto", "frontier_data") and frontier_data_available()
    if backend == "frontier_data" and not use_fd:
        raise ConnectorError(
            "frontier_data with v1.1 writers is not importable",
            hint="install frontier-data or use backend='zarr'",
        )
    dw: Any = None
    if use_fd:
        from frontier_data.dataset import DatasetWriter

        dw = DatasetWriter(out, ds_name, src, domain=domain, license=license, citation=citation)
    else:
        result.warnings.append(
            "frontier_data not importable: wrote zarr v3 directly per the v1.1 contract"
        )

    writers: dict[str, _StreamWriter] = {}
    entries: dict[str, dict[str, Any]] = {}
    metas: dict[str, dict[str, Any]] = {}
    done: set[str] = set()
    names: dict[str, tuple[dict[str, str], dict[str, str]]] = {}

    def finish(sid: str) -> None:
        w = writers.pop(sid)
        entry = w.close()
        entries[sid] = entry
        metas[sid] = json.loads((w.sdir / "meta.json").read_text())
        result.n_steps[sid] = entry["n_steps"]
        done.add(sid)

    try:
        for raw in samples:
            s = raw
            if time_as_samples and s.time is not None:
                idx = s.time_index if s.time_index is not None else 0
                md = dict(s.metadata)
                md["group"] = md.get("group") or s.id
                md["time"] = {"t0": float(s.time), "n_steps": 1}
                s = RawSample(
                    id=f"{s.id}_t{idx:05d}",
                    kind=s.kind,
                    zone=s.zone,
                    points=s.points,
                    cells=s.cells,
                    cell_types=s.cell_types,
                    faces=s.faces,
                    coords=s.coords,
                    fields=s.fields,
                    static=s.static,
                    metadata=md,
                )
                if s.kind != "grid" and s.points is None:
                    raise ConnectorError(
                        f"sample {raw.id!r} step {idx} carries no geometry",
                        hint="time_as_samples needs a connector that yields the mesh every step",
                    )
            sid = meshutil.sanitize_id(s.id)
            if sid != s.id:
                s.id = sid
            taken, renames = names.setdefault(sid, ({}, {}))
            _case_unique(s, taken, renames, result.warnings)
            if sid in done:
                raise ConnectorError(
                    f"sample id {sid!r} was already written",
                    hint="connector bug: all steps of one id must be yielded before it is "
                    "closed, and ids must be unique per zone",
                )
            if sid in writers:
                writers[sid].add(s)
                continue
            complete = not s.transient or (s.block_form and _in_memory(s))
            if dw is not None and complete and _in_memory(s):
                try:
                    _write_with_frontier_data(dw, s, ds_name, steps_per_chunk)
                except (TypeError, ValueError, AttributeError) as e:
                    if backend == "frontier_data":
                        raise
                    result.warnings.append(
                        f"{sid}: frontier_data writer rejected the sample ({e}); "
                        "used the streaming zarr writer"
                    )
                else:
                    result.backends[sid] = "frontier_data"
                    result.n_steps[sid] = len(s.times) if s.times is not None else 1
                    result.sample_ids.append(sid)
                    done.add(sid)
                    continue
            # close steady stream writers eagerly: only transient ids stay open
            for other in [k for k, w in writers.items() if not w.transient]:
                finish(other)
            w = _StreamWriter(out, s, ds_name, steps_per_chunk)
            writers[sid] = w
            w.add(s)
            result.backends[sid] = "zarr-stream"
            result.sample_ids.append(sid)
        for sid in list(writers):
            finish(sid)
        if not result.sample_ids:
            raise ConnectorError(
                "the source produced no samples",
                hint="check --zones/--fields/--times; `describe` lists what the source contains",
            )
    except BaseException:
        # a failed conversion must not leave a half-written directory that blocks the retry
        shutil.rmtree(out, ignore_errors=True)
        raise
    _finalize(out, ds_name, src, domain, license, citation, dw, entries, metas)
    result.seconds = _time.perf_counter() - t_start
    return result


def _finalize(
    out: Path,
    name: str,
    source: dict[str, Any],
    domain: str,
    license_: str | None,
    citation: str | None,
    dw: Any,
    entries: dict[str, dict[str, Any]],
    metas: dict[str, dict[str, Any]],
) -> None:
    if dw is not None:
        from frontier_data.dataset import SampleEntry, SampleMeta

        for sid, entry in entries.items():
            dw.add_entry(SampleEntry.model_validate(entry), SampleMeta.model_validate(metas[sid]))
        dw.finalize()
        return
    samples = sorted(entries.values(), key=lambda e: e["id"])
    fields: dict[str, dict[str, Any]] = {}
    params: list[str] = []
    for m in metas.values():
        for f in m["fields"]:
            fields.setdefault(f["name"], f)
        for p in m["parameters"]:
            if p not in params:
                params.append(p)
    kinds = [e["sample_kind"] for e in samples]
    transient = [e for e in samples if e["temporal"] == "transient"]
    time = None
    if transient:
        tms = [metas[e["id"]]["time"] for e in transient if metas[e["id"]].get("time")]
        dts = {round(t["dt"], 12) if t["dt"] is not None else None for t in tms}
        dt = next(iter(dts)) if len(dts) == 1 else None
        time = {
            "n_steps": max(e["n_steps"] for e in transient),
            "dt": dt,
            "units": tms[0]["units"] if tms else "s",
            "uniform": dt is not None and all(t["uniform"] for t in tms),
            "t0": min((t["t0"] for t in tms), default=0.0),
            "averaging_window": None,
        }
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "minor": FORMAT_MINOR,
        "sample_kind": max(set(kinds), key=kinds.count),
        "temporal": "transient" if transient else "steady",
        "time": time,
        "domain": domain,
        "license": license_,
        "citation": citation,
        "hub_id": None,
        "parameters": params,
        "name": name,
        "source": source,
        "created_at": datetime.now(UTC).isoformat(),
        "samples": samples,
        "splits": {},
        "split_strategy": None,
        "split_families": {},
        "fields": list(fields.values()),
        "conditions": [],
        "geometry_params": [],
        "pipeline": None,
        "quality": None,
        "stats": {
            "n_samples": len(samples),
            "n_points_total": int(sum(e["n_points"] for e in samples)),
            "n_faces_total": int(sum(e["n_faces"] for e in samples)),
            "n_steps_total": int(sum(e["n_steps"] for e in samples)),
        },
    }
    _write_json(out / "manifest.json", manifest)


def convert(
    path: str | Path,
    out: str | Path,
    *,
    connector: str | None = None,
    zones: list[str] | None = None,
    fields: list[str] | None = None,
    times: Any = None,
    options: Mapping[str, Any] | None = None,
    name: str | None = None,
    length_unit: str | None = None,
    source_length: str | None = None,
    **kwargs: Any,
) -> ConvertResult:
    """Detect, describe and convert ``path`` into a dataset directory in one call.

    ``length_unit`` rescales coordinates to that unit (``"m"``); the source unit comes from
    the connector (``metadata["units"]["length"]``) or from ``source_length`` when the format
    does not record one. Field values are never rescaled.
    """
    from frontier_connectors import registry
    from frontier_connectors.units import convert_sample_units

    c = registry.resolve(path, connector)
    desc = registry.describe(path, c.id, options=options)
    samples = registry.read(path, c.id, zones=zones, fields=fields, times=times, options=options)
    if length_unit:

        def rescaled(stream: Any) -> Any:
            for s in stream:
                try:
                    yield convert_sample_units(s, length=length_unit, source_length=source_length)
                except KeyError as e:
                    raise ConnectorError(
                        f"cannot convert lengths to {length_unit!r}: {e.args[0]}",
                        hint="this format does not record its length unit: pass source_length "
                        "(CLI: --source-length mm)",
                    ) from e

        samples = rescaled(samples)
    kwargs.setdefault("description", desc)
    result = to_frontier_dataset(samples, out, name=name, **kwargs)
    result.warnings = list(desc.warnings) + result.warnings
    return result
