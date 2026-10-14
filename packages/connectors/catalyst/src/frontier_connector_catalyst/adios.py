# SPDX-License-Identifier: Apache-2.0
"""ADIOS2 step sources: BP4 / BP5 files (also while they are being written) and SST streams.

``adios2`` (Apache-2.0) is optional and imported lazily. API used, read from the ``adios2``
2.12.1 wheel on 2026-10-10 (https://adios2.readthedocs.io/en/latest/api_python/api_python.html,
https://adios2.readthedocs.io/en/latest/engines/engines.html): ``adios2.Adios().declare_io``,
``IO.set_engine`` / ``set_parameters`` (``OpenTimeoutSecs``), ``adios2.Stream(io, path, "r")``,
``Stream.begin_step(timeout=...)`` returning ``bindings.StepStatus`` (``OK``, ``NotReady``,
``EndOfStream``, ``OtherError``), ``available_variables()``, ``available_attributes()``,
``read(name)``, ``read_attribute(name)``, ``end_step()``, ``close()``.

Variable conventions
--------------------

**Blueprint paths (preferred).** Every array of the Conduit Mesh Blueprint tree is an ADIOS
variable named by its path, every string leaf an ADIOS attribute of the same name::

    variables   coordsets/coords/values/x   topologies/mesh/elements/connectivity
                fields/T/values             fields/U/values (N, 3)    state/time   state/cycle
    attributes  coordsets/coords/type="explicit"   topologies/mesh/type="unstructured"
                topologies/mesh/coordset="coords"  topologies/mesh/elements/shape="tet"
                fields/T/association="vertex"      fields/T/topology="mesh"   fields/T/units="K"

Several domains use a common prefix (``fluid/coordsets/...``, ``wall/coordsets/...``). The mesh
may be written in the first step only; variables that are absent in a later step keep their
last value. Stream attributes ``frontier/solver``, ``frontier/solver_version``,
``frontier/domain`` and ``frontier/time_units`` are copied into the dataset metadata.

**Fides predefined data models** (the attribute convention ParaView's Fides reader uses,
https://fides.readthedocs.io/en/latest/components/predefined_models.html, read 2026-10-10):
``Fides_Data_Model`` = ``uniform`` (``Fides_Origin``, ``Fides_Spacing``,
``Fides_Dimension_Variable``), ``rectilinear`` (``Fides_X_Variable`` ...), ``unstructured_single``
(``Fides_Cell_Type``, ``Fides_Coordinates_Variable``, ``Fides_Connectivity_Variable``) and
``unstructured`` (``Fides_Cell_Types_Variable``, ``Fides_Num_Vertices_Variable``), plus
``Fides_Time_Variable``, ``Fides_Variable_List`` and ``Fides_Variable_Associations``. They are
translated to a Blueprint tree. Grid arrays are taken in ADIOS row-major order with the *last*
index varying fastest being x, i.e. a shape ``(nz, ny, nx)`` (option ``fides_order=xyz`` for
arrays shaped ``(nx, ny, nz)``); fields without a listed association are matched by size. A
Fides JSON data-model file and the XGC / GTC models are not read. This translation was tested
against files written with these attributes by this package's tests, not against Fides itself.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connector_catalyst.stream import FollowOptions, Step, StreamTimeoutError
from frontier_connectors.errors import (
    ConnectorError,
    MissingDependencyError,
    UnsupportedSourceError,
)

log = logging.getLogger("frontier_connectors.catalyst")

ENGINES = ("BPFile", "BP5", "BP4", "BP3", "SST", "FileStream")
_FIDES_CELLS = {
    "vertex": "point", "line": "line", "triangle": "tri", "quad": "quad",
    "tetrahedron": "tet", "hexahedron": "hex", "wedge": "wedge", "pyramid": "pyramid",
}  # fmt: skip
_VTK_SHAPES = {
    1: "point", 3: "line", 5: "tri", 9: "quad", 10: "tet", 12: "hex", 13: "wedge", 14: "pyramid",
}  # fmt: skip
_TIME_NAMES = ("state/time", "time", "Time", "TIME")


def import_adios2() -> Any:
    try:
        import adios2  # type: ignore[import-not-found]
    except ImportError as e:
        raise MissingDependencyError(
            "adios2", purpose="read ADIOS2 BP files and SST streams", license="Apache-2.0"
        ) from e
    if not hasattr(adios2, "Stream") or not hasattr(adios2, "Adios"):
        raise ConnectorError(
            f"adios2 {getattr(adios2, '__version__', '?')} is too old",
            hint="the reader needs the Python API of ADIOS2 >= 2.10: pip install -U adios2",
        )
    return adios2


def stream_name(path: Path) -> tuple[str, str | None]:
    """``(name to open, engine implied by the path)``: ``run.sst`` is the contact file of the
    SST stream ``run``; a ``.bp`` path is a BP file."""
    p = Path(path)
    if p.suffix == ".sst":
        return str(p.with_suffix("")), "SST"
    if p.suffix in (".bp", ".bp4", ".bp5"):
        return str(p), None
    return str(p), None


def looks_like_adios(path: Path) -> tuple[float, str] | None:
    p = Path(path)
    if p.is_dir() and p.suffix in (".bp", ".bp4", ".bp5") and (p / "md.idx").exists():
        return 0.8, "adios2-bp"
    if p.is_dir() and (p / "md.idx").exists() and (p / "md.0").exists():
        return 0.6, "adios2-bp"
    if p.is_file() and p.suffix == ".sst":
        return 0.8, "adios2-sst"
    return None


def _attr_value(info: Mapping[str, Any], stream: Any, name: str) -> Any:
    if info.get("Type") == "string":
        v = stream.read_attribute(name)
        if isinstance(v, list | tuple):
            return [str(x) for x in v] if len(v) != 1 else str(v[0])
        return str(v)
    return np.asarray(stream.read_attribute(name))


def _set(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return
    cur = tree
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    if not isinstance(cur.get(parts[-1]), dict):
        cur[parts[-1]] = value


class AdiosSource:
    """One ADIOS2 file or stream. A source is read once, front to back."""

    def __init__(
        self,
        path: Path,
        *,
        engine: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        fides_order: str = "zyx",
    ) -> None:
        self.path = Path(path)
        self.name, implied = stream_name(self.path)
        self.engine = engine or implied or "BPFile"
        if self.engine not in ENGINES:
            raise ConnectorError(
                f"unknown ADIOS2 engine {self.engine!r}", hint="use one of: " + ", ".join(ENGINES)
            )
        if fides_order not in ("zyx", "xyz"):
            raise ConnectorError(f"bad option fides_order={fides_order!r}", hint="zyx | xyz")
        self.parameters = {str(k): str(v) for k, v in (parameters or {}).items()}
        self.fides_order = fides_order
        self.streaming = self.engine == "SST"
        self.header: dict[str, Any] = {}
        self.events: list[str] = []
        self.end: dict[str, Any] = {}
        self._stream: Any = None
        self._adios: Any = None
        self._static: dict[str, Any] = {}
        self._count = 0
        #: a step read ahead by ``peek`` and not consumed yet
        self._peeked: Step | None = None
        self._finished = False

    # -- low level -------------------------------------------------------------------------
    def _open(self, open_timeout: float) -> Any:
        if self._stream is not None:
            return self._stream
        adios2 = import_adios2()
        if not self.streaming and open_timeout > 0:
            # the BP file may not exist yet (reader started before the writer)
            import time as _time

            index = Path(self.name) / "md.idx"
            start = _time.monotonic()
            while not index.exists() and _time.monotonic() - start <= open_timeout:
                _time.sleep(0.02)
            if not index.exists():
                raise StreamTimeoutError(
                    f"{self.name} did not appear within {open_timeout:g} s",
                    hint="is the writer running? raise --option timeout=<seconds>",
                )
        self._adios = adios2.Adios()
        io = self._adios.declare_io(f"frontier:{self.name}:{id(self)}")
        io.set_engine(self.engine)
        params = {"OpenTimeoutSecs": f"{max(open_timeout, 0.0):g}"}
        params.update(self.parameters)
        io.set_parameters(params)
        try:
            self._stream = adios2.Stream(io, self.name, "r")
        except Exception as e:  # noqa: BLE001 - adios2 raises RuntimeError / ValueError variants
            msg = str(e).replace("\x1b", "")
            if "imeout" in msg or "could not be found" in msg or "did not find" in msg:
                raise StreamTimeoutError(
                    f"ADIOS2 could not open {self.name} ({self.engine}) within {open_timeout:g} s",
                    hint="is the writer running? raise --option timeout=<seconds>; for SST pass "
                    "the stream name (or its .sst contact file) and --option engine=SST",
                ) from e
            raise UnsupportedSourceError(
                f"ADIOS2 cannot open {self.name} with engine {self.engine}: {msg[-300:]}",
                hint="check the path and --option engine=BP5|BP4|SST",
            ) from e
        return self._stream

    def close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001 - closing a dead stream must not mask errors
                log.debug("closing %s: %r", self.name, e)
        self._adios = None
        self._finished = True

    def _read_current(self, stream: Any) -> Step:
        """Build the Blueprint tree of the step the stream is positioned on."""
        variables = stream.available_variables()
        attributes = stream.available_attributes()
        attrs = {name: _attr_value(info, stream, name) for name, info in attributes.items()}
        values: dict[str, Any] = {}
        for name, info in variables.items():
            if info.get("Type") == "string":
                values[name] = str(stream.read(name))
                continue
            arr = np.asarray(stream.read(name))
            if str(info.get("SingleValue", "false")).lower() == "true":
                arr = arr.reshape(-1)[:1]
            values[name] = arr
        index = int(stream.current_step())
        for key, target in (
            ("frontier/solver", "solver"),
            ("frontier/solver_version", "solver_version"),
            ("frontier/domain", "domain"),
            ("frontier/time_units", "time_units"),
        ):
            if isinstance(attrs.get(key), str):
                self.header[target] = attrs[key]
        if "Fides_Data_Model" in attrs:
            tree, time = self._fides_tree(values, attrs)
        else:
            tree = {}
            for name, value in {**self._static, **values}.items():
                _set(tree, name, value)
            for name, value in attrs.items():
                if not name.startswith(("frontier/", "Fides_")):
                    _set(tree, name, value if isinstance(value, str) else np.asarray(value))
            self._static.update(
                {
                    k: v
                    for k, v in values.items()
                    if "/coordsets/" in f"/{k}" or "/topologies/" in f"/{k}"
                }
            )
            time = None
            for key in _TIME_NAMES:
                if key in values and not isinstance(values[key], str):
                    time = float(np.asarray(values[key]).reshape(-1)[0])
                    break
        cycle = values.get("state/cycle")
        self._count += 1
        return Step(
            index=index,
            tree=tree,
            time=time if time is not None else float(index),
            cycle=int(np.asarray(cycle).reshape(-1)[0]) if cycle is not None else index,
            has_mesh=True,
        )

    def _fides_tree(
        self, values: dict[str, Any], attrs: dict[str, Any]
    ) -> tuple[dict[str, Any], float | None]:
        model = str(attrs["Fides_Data_Model"])
        merged = {**self._static, **values}

        def var(attr: str, default: str | None) -> str:
            name = attrs.get(attr, default)
            if not isinstance(name, str) or name not in merged:
                raise UnsupportedSourceError(
                    f"Fides model {model!r}: variable {name!r} ({attr}) is not in the step",
                    hint="see the variable conventions in the catalyst connector README",
                )
            return name

        used: set[str] = set()
        elements: dict[str, Any]
        if model in ("uniform", "rectilinear"):
            if model == "uniform":
                dim_var = var("Fides_Dimension_Variable", None)
                shape = list(np.asarray(merged[dim_var]).shape)
                dims = shape[::-1] if self.fides_order == "zyx" else shape
                origin = np.asarray(attrs.get("Fides_Origin", [0, 0, 0]), dtype=float).reshape(-1)
                spacing = np.asarray(attrs.get("Fides_Spacing", [1, 1, 1]), dtype=float).reshape(-1)
                coordset: dict[str, Any] = {
                    "type": "uniform",
                    "dims": dict(zip("ijk", dims, strict=False)),
                    "origin": dict(zip("xyz", origin[: len(dims)].tolist(), strict=False)),
                    "spacing": dict(
                        zip(("dx", "dy", "dz"), spacing[: len(dims)].tolist(), strict=False)
                    ),
                }
            else:
                names = [
                    var("Fides_X_Variable", "x"),
                    var("Fides_Y_Variable", "y"),
                    var("Fides_Z_Variable", "z"),
                ]
                used.update(names)
                axes = [np.asarray(merged[n], dtype=float).reshape(-1) for n in names]
                dims = [len(a) for a in axes]
                coordset = {"type": "rectilinear", "values": dict(zip("xyz", axes, strict=True))}
            topo: dict[str, Any] = {"type": model, "coordset": "coords"}
            n_points = int(np.prod(dims))
            n_cells = int(np.prod([max(d - 1, 1) for d in dims]))
            grid_dims: list[int] | None = dims
        elif model in ("unstructured_single", "unstructured"):
            pname = var("Fides_Coordinates_Variable", "points")
            cname = var("Fides_Connectivity_Variable", "connectivity")
            used.update((pname, cname))
            pts = np.asarray(merged[pname], dtype=float)
            if pts.ndim != 2:
                raise UnsupportedSourceError(f"Fides coordinates {pname!r} must be (N, 3)")
            conn = np.asarray(merged[cname]).reshape(-1)
            if model == "unstructured_single":
                ctype = str(attrs.get("Fides_Cell_Type", ""))
                if ctype not in _FIDES_CELLS:
                    raise UnsupportedSourceError(
                        f"Fides_Cell_Type {ctype!r}", hint="one of: " + ", ".join(_FIDES_CELLS)
                    )
                elements = {"shape": _FIDES_CELLS[ctype], "connectivity": conn}
                from frontier_connector_catalyst.blueprint import SHAPES

                n_cells = len(conn) // SHAPES[_FIDES_CELLS[ctype]][1]
            else:
                tname = var("Fides_Cell_Types_Variable", "cell_types")
                nname = var("Fides_Num_Vertices_Variable", "num_verts")
                used.update((tname, nname))
                types = np.asarray(merged[tname]).reshape(-1).astype(np.int64)
                sizes = np.asarray(merged[nname]).reshape(-1).astype(np.int64)
                unknown = sorted(set(types.tolist()) - set(_VTK_SHAPES))
                if unknown:
                    raise UnsupportedSourceError(
                        f"Fides cell types {unknown} are not supported",
                        hint="linear VTK cell types 1, 3, 5, 9, 10, 12, 13, 14",
                    )
                elements = {
                    "shape": "mixed",
                    "shape_map": {_VTK_SHAPES[t]: t for t in sorted(set(types.tolist()))},
                    "shapes": types,
                    "sizes": sizes,
                    "connectivity": conn,
                }
                n_cells = len(types)
            coordset = {"type": "explicit", "values": pts}
            topo = {"type": "unstructured", "coordset": "coords", "elements": elements}
            n_points = len(pts)
            grid_dims = None
            self._static.update({k: merged[k] for k in used})
        else:
            raise UnsupportedSourceError(
                f"Fides data model {model!r} is not supported",
                hint="supported: uniform, rectilinear, unstructured_single, unstructured",
            )
        time_var = attrs.get("Fides_Time_Variable")
        time = None
        if isinstance(time_var, str) and time_var in values:
            time = float(np.asarray(values[time_var]).reshape(-1)[0])
            used.add(time_var)
        listed = attrs.get("Fides_Variable_List")
        assoc_list = attrs.get("Fides_Variable_Associations")
        listed = [listed] if isinstance(listed, str) else listed
        assoc_list = [assoc_list] if isinstance(assoc_list, str) else assoc_list
        assoc: dict[str, str] = {}
        if listed and assoc_list and len(listed) == len(assoc_list):
            assoc = dict(zip(listed, assoc_list, strict=True))
        fields: dict[str, Any] = {}
        for name, value in values.items():
            if name in used or isinstance(value, str) or (listed and name not in listed):
                continue
            arr = np.asarray(value)
            if arr.size <= 1:
                continue
            flat, how = self._fides_field(arr, grid_dims, n_points, n_cells, assoc.get(name))
            if flat is None:
                if listed:
                    raise UnsupportedSourceError(
                        f"Fides variable {name!r} of shape {arr.shape} fits neither the "
                        f"{n_points} points nor the {n_cells} cells"
                    )
                continue
            fields[name] = {"association": how, "topology": "mesh", "values": flat}
        tree = {"coordsets": {"coords": coordset}, "topologies": {"mesh": topo}, "fields": fields}
        return tree, time

    def _fides_field(
        self,
        arr: np.ndarray,
        grid_dims: list[int] | None,
        n_points: int,
        n_cells: int,
        declared: str | None,
    ) -> tuple[np.ndarray | None, str]:
        want = {"points": "vertex", "cell_set": "element", "cells": "element"}.get(declared or "")
        for how, n in (("vertex", n_points), ("element", n_cells)):
            if want is not None and how != want:
                continue
            if grid_dims is not None:
                dims = grid_dims if how == "vertex" else [max(d - 1, 1) for d in grid_dims]
                shape = tuple(dims[::-1] if self.fides_order == "zyx" else dims)
                if arr.shape == shape or (arr.ndim == len(shape) + 1 and arr.shape[:-1] == shape):
                    scalar = arr.ndim == len(shape)
                    full = arr.reshape(*shape, -1)
                    if self.fides_order == "xyz":  # bring x to the fastest (last) position
                        nd = len(shape)
                        full = full.transpose(*range(nd - 1, -1, -1), nd)
                    flat = np.ascontiguousarray(full).reshape(n, -1)
                    return (flat[:, 0] if scalar else flat), how
                if arr.ndim == 1 and arr.shape[0] == n:
                    return arr, how
            elif arr.shape[0] == n and arr.ndim in (1, 2):
                return arr, how
        return None, ""

    # -- stepping --------------------------------------------------------------------------
    def _next(self, opts: FollowOptions, follow: bool) -> Step | None:
        """The next step, or ``None`` at the end of the stream."""
        if self._peeked is not None:
            step, self._peeked = self._peeked, None
            return step
        if self._finished:
            return None
        adios2 = import_adios2()
        status_t = adios2.bindings.StepStatus
        stream = self._open(opts.timeout if follow or self.streaming else 0.0)
        waited_from = opts.clock()
        while True:
            wait = min(max(opts.poll, 0.05), 1.0) if (follow or self.streaming) else 0.0
            status = stream.begin_step(timeout=wait)
            if status == status_t.OK:
                try:
                    step = self._read_current(stream)
                finally:
                    stream.end_step()
                return step
            if status == status_t.EndOfStream:
                self.end = {"reason": "done", "status": "complete", "n_steps": self._count}
                self.close()
                return None
            if status == status_t.NotReady:
                if not (follow or self.streaming):
                    self.end = {"reason": "end-of-file", "n_steps": self._count}
                    self.close()
                    return None
                if opts.clock() - waited_from > opts.timeout:
                    msg = (
                        f"no new step from {self.name} for {opts.timeout:g} s ({self._count} read)"
                    )
                    self.close()
                    if self._count == 0 or opts.on_timeout == "error":
                        raise StreamTimeoutError(
                            msg,
                            hint="the writer stopped without closing the stream; use "
                            "on_timeout=stop to keep what arrived or raise --option timeout",
                        )
                    self.end = {"reason": "timeout", "n_steps": self._count}
                    self.events.append(msg + "; stopping (on_timeout=stop)")
                    log.warning("%s", self.events[-1])
                    return None
                continue
            self.close()
            raise ConnectorError(
                f"ADIOS2 reported an error while waiting for a step of {self.name}",
                hint="check the writer; SST streams end when the writer dies",
            )

    def peek(self, opts: FollowOptions, follow: bool) -> Step:
        """The first step without consuming it (used by ``describe``)."""
        if self._peeked is None:
            if self._count:
                raise ConnectorError(f"{self.name}: the stream was already read")
            step = self._next(opts, follow)
            if step is None:
                raise ConnectorError(
                    f"{self.name} holds no steps", hint="the writer closed it without a step"
                )
            self._peeked = step
        return self._peeked

    def steps(self, opts: FollowOptions, follow: bool) -> Iterator[Step]:
        count = 0
        try:
            while True:
                step = self._next(opts, follow)
                if step is None:
                    return
                count += 1
                yield step
                if opts.max_steps is not None and count >= opts.max_steps:
                    self.end = {"reason": "max_steps", "n_steps": count}
                    return
        finally:
            self.close()

    def scan_times(self) -> list[float]:
        """Time of every step currently in a BP *file* (a second, metadata-only pass)."""
        if self.streaming:
            raise ConnectorError("an SST stream cannot be scanned ahead")
        probe = AdiosSource(self.path, engine=self.engine, parameters=self.parameters)
        adios2 = import_adios2()
        status_t = adios2.bindings.StepStatus
        stream = probe._open(0.0)
        times: list[float] = []
        try:
            while stream.begin_step(timeout=0.0) == status_t.OK:
                variables = stream.available_variables()
                attributes = stream.available_attributes()
                names: list[str] = list(_TIME_NAMES)
                if "Fides_Time_Variable" in attributes:
                    names.insert(
                        0,
                        str(
                            _attr_value(
                                attributes["Fides_Time_Variable"], stream, "Fides_Time_Variable"
                            )
                        ),
                    )
                t = float(stream.current_step())
                for n in names:
                    if n in variables:
                        t = float(np.asarray(stream.read(n)).reshape(-1)[0])
                        break
                times.append(t)
                stream.end_step()
        finally:
            probe.close()
        return times
