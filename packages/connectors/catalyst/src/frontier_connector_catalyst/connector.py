# SPDX-License-Identifier: Apache-2.0
"""ParaView Catalyst 2 / in-situ connector: Conduit Mesh Blueprint steps from a spool
directory, Conduit relay files or an ADIOS2 file / stream, finite or followed live.

``read`` on a path reads what is there. With ``options={"follow": True, ...}`` it tails the
source while the solver runs and returns when the producer's end marker arrives (or after
``timeout`` seconds without a new step); see
:class:`frontier_connector_catalyst.stream.FollowOptions`
for every option. Because ``to_frontier_dataset`` consumes the samples lazily, the dataset
grows along time as the steps arrive.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from frontier_connector_catalyst import adios, blueprint, spool
from frontier_connector_catalyst.stream import FollowOptions, Step, StreamTimeoutError
from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError, SelectionError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
)
from frontier_connectors.select import select_time_indices

log = logging.getLogger("frontier_connectors.catalyst")

#: written by the ``catalyst`` importer into its work directory: where the live source is
POINTER_NAME = "frontier-follow.json"
_DOMAINS = ("cfd", "structural", "thermal", "multiphysics", "materials", "geophysical", "other")

EXPORT_HELP = (
    "Get the solver-side scripts with `python -m frontier_connectors plugins install "
    "exporter-catalyst`, then use one of: (1) copy frontier_emit.py (numpy only) next to the "
    "solver and call Emitter.write_step(step, time, blueprint_mesh) at each output step; (2) run "
    "the Catalyst 2 pipeline frontier_catalyst_pipeline.py (CATALYST_IMPLEMENTATION_NAME=paraview),"
    " which writes the same spool; (3) write Conduit relay Blueprint files (relay.io.blueprint."
    "save_mesh, Ascent) or ADIOS2 BP5 / SST with Blueprint paths as variable names. Follow the run "
    "with `python -m frontier_connectors import catalyst <spool|stream> --out ds`. Files a "
    "Catalyst pipeline writes with plain VTK extractors (.vtpd / .vtm series, VTKHDF) are read by "
    "the `vtk` connector."
)


def _base_id(path: Path, header: Mapping[str, Any]) -> str:
    sid = header.get("stream_id")
    if isinstance(sid, str) and sid:
        return meshutil.sanitize_id(sid)
    p = Path(path)
    name = p.name
    m = spool.relay.ROOT_SERIES.match(name)
    if m:
        return meshutil.sanitize_id(m.group("base"))
    for suffix in (".root", ".bp", ".bp4", ".bp5", ".sst", ".npz"):
        if name.endswith(suffix):
            return meshutil.sanitize_id(name[: -len(suffix)])
    return meshutil.sanitize_id(name)


class CatalystConnector(BaseConnector):
    id = "catalyst"
    name = "ParaView Catalyst 2 / in-situ streams (Conduit Blueprint spool, relay, ADIOS2)"
    formats = ["frontier-spool", "conduit-blueprint", "conduit-relay", "adios2-bp", "adios2-sst"]
    vendor = "Kitware / LLNL / ORNL (open source)"
    license = "Apache-2.0"
    extensions = [".root", ".bp", ".sst"]
    maturity = "verified-synthetic"
    temporal = ["steady", "transient"]
    kinds = ["grid", "mesh", "surface", "point_cloud"]
    requires = ["adios2", "h5py"]
    export_help = EXPORT_HELP
    status_note = (
        "Reads frontier-spool directories, Conduit relay Blueprint files (hdf5, json, yaml, "
        "conduit_*) and ADIOS2 BP4 / BP5 files and BP5 / SST streams, during or after the run. "
        "Validated with Conduit 0.9.5 and ADIOS2 2.12.1. Not validated against ParaView Catalyst "
        "(the pipeline script), Fides or MPI-parallel writers."
    )

    def __init__(self) -> None:
        #: live ADIOS2 sources opened by ``describe`` and handed over to ``read`` (an SST
        #: stream can be opened only once)
        self._sessions: dict[str, adios.AdiosSource] = {}
        #: descriptions handed out by ``describe``: ``convert`` reads their ``warnings`` after
        #: the stream ended, so what happened while following (timeout, restart, lost step,
        #: failed run) is appended there and ends up in the conversion result
        self._described: dict[str, SourceDescription] = {}

    # -- detection -------------------------------------------------------------------------
    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        found: tuple[float, str] | None = None
        notes: list[str] = []
        needs: list[str] = []
        if path.is_dir() and (path / POINTER_NAME).is_file():
            found = (1.0, "frontier-follow")
            notes.append("work directory of the catalyst importer (points at a live stream)")
        elif path.is_dir():
            found = spool.looks_like_spool(path) or adios.looks_like_adios(path)
        elif path.is_file():
            if spool.STEP_PATTERN.match(path.name):
                found = (0.9, "frontier-spool")
            elif path.suffix == ".root" and spool.relay.is_blueprint_root(path):
                found = (0.85, "conduit-relay")
            else:
                found = adios.looks_like_adios(path)
        if found is None:
            return None
        if found[1].startswith("adios2"):
            try:
                adios.import_adios2()
            except ConnectorError:
                needs.append("pip install adios2 (Apache-2.0)")
        return Detection(
            connector_id=self.id,
            path=path,
            confidence=found[0],
            format=found[1],
            notes=notes,
            needs=needs,
        )

    # -- plumbing --------------------------------------------------------------------------
    @staticmethod
    def _resolve(path: Path, options: Mapping[str, Any] | None) -> tuple[Path, dict[str, Any]]:
        """Follow the importer's pointer file and merge its options under the caller's."""
        path = Path(path)
        opts = dict(options or {})
        pointer = path / POINTER_NAME
        if path.is_dir() and pointer.is_file():
            try:
                doc = json.loads(pointer.read_text())
                source = Path(doc["source"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise ConnectorError(
                    f"{pointer} is not a valid follow pointer: {e}",
                    hint="re-run `python -m frontier_connectors import catalyst <source> ...`",
                ) from e
            opts = {**(doc.get("options") or {}), **opts}
            path = source
        return path, opts

    def _is_adios(self, path: Path, opts: Mapping[str, Any]) -> bool:
        if opts.get("engine"):
            return True
        if path.suffix in (".bp", ".bp4", ".bp5", ".sst"):
            return True
        return path.is_dir() and adios.looks_like_adios(path) is not None

    def _adios_source(self, path: Path, opts: Mapping[str, Any]) -> adios.AdiosSource:
        key = str(path.resolve())
        src = self._sessions.pop(key, None)
        if src is not None:
            return src
        params = opts.get("adios_parameters") or {}
        if not isinstance(params, Mapping):
            raise ConnectorError("option adios_parameters must be a JSON object")
        return adios.AdiosSource(
            path,
            engine=str(opts["engine"]) if opts.get("engine") else None,
            parameters=params,
            fides_order=str(opts.get("fides_order", "zyx")),
        )

    @staticmethod
    def _spool_source(
        path: Path, opts: Mapping[str, Any], fo: FollowOptions | None = None
    ) -> spool.SpoolSource:
        if fo is not None and fo.follow:
            start = fo.clock()  # the reader may be started before the solver
            while not path.exists() and fo.clock() - start <= fo.timeout:
                fo.sleep(fo.poll)
            if not path.exists():
                raise StreamTimeoutError(
                    f"{path} did not appear within {fo.timeout:g} s",
                    hint="start the solver (it creates the spool directory), or raise "
                    "--option timeout=<seconds>",
                )
        if not path.exists():
            raise UnsupportedSourceError(
                f"{path} does not exist",
                hint="pass a spool directory, a step/.root file or an ADIOS2 .bp / .sst path; "
                "use the `catalyst` importer to wait for a stream that has not started",
            )
        return spool.SpoolSource(path, name=str(opts["name"]) if opts.get("name") else None)

    @staticmethod
    def _first_step(src: spool.SpoolSource, fo: FollowOptions) -> Step:
        """The first step of a spool; in follow mode wait until it is complete."""
        if not fo.follow:
            entries = src.entries()
            if not entries:
                raise UnsupportedSourceError(
                    f"{src.directory} holds no steps yet",
                    hint="follow the live stream with --option follow=true (or the `catalyst` "
                    "importer), which waits for the solver",
                )
            return src.load(entries[0])
        start = fo.clock()
        while True:
            entry = src.wait_for_first(fo)
            try:
                return spool.load_entry(entry)
            except spool.PartialStepError as err:
                if fo.clock() - start > max(fo.timeout, fo.partial_grace):
                    raise src._corrupt(entry, err) from err
                fo.sleep(fo.poll)

    def _metadata(self, header: Mapping[str, Any], path: Path, fmt: str) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "solver": str(header.get("solver") or "unknown"),
            "source": str(header.get("stream_id") or _base_id(path, header)),
            "in_situ": {"transport": fmt, "path": str(path)},
        }
        if header.get("solver_version"):
            meta["solver_version"] = str(header["solver_version"])
        for key in ("physics", "parameters", "units"):
            if isinstance(header.get(key), Mapping) and header[key]:
                meta[key] = dict(header[key])
        if header.get("time_units"):
            meta["time_units"] = str(header["time_units"])
        if header.get("run_id"):
            meta["in_situ"]["run_id"] = header["run_id"]
        return meta

    # -- describe --------------------------------------------------------------------------
    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        given = Path(path)
        path, opts = self._resolve(given, options)
        fo = FollowOptions.from_options(opts)
        warnings: list[str] = []
        metadata: dict[str, Any] = {"live": fo.follow}
        if self._is_adios(path, opts):
            src = self._adios_source(path, opts)
            fmt = "adios2-sst" if src.streaming else "adios2-bp"
            first = src.peek(fo, fo.follow)
            times = [] if src.streaming else src.scan_times()
            if src.streaming or fo.follow:
                self._sessions[str(path.resolve())] = src
            else:
                src.close()
            header: Mapping[str, Any] = dict(src.header)
            metadata["engine"] = src.engine
            if src.streaming:
                times = [first.time if first.time is not None else 0.0]
                warnings.append("SST stream: the number of steps is unknown until it ends")
        else:
            ssrc = self._spool_source(path, opts, fo)
            first = self._first_step(ssrc, fo)
            header = ssrc.header
            fmt = "conduit-relay" if ssrc.flavor() == "relay" else "frontier-spool"
            times = [
                t if t is not None else float(e.index) for e, t in self._safe_times(ssrc, fo.follow)
            ]
            done = ssrc.done()
            metadata["done"] = done is not None
            if done:
                metadata["end"] = {
                    k: done[k] for k in ("status", "message", "n_steps") if k in done
                }
            if header.get("restarts"):
                metadata["restarts"] = header["restarts"]
            for key in ("stream_id", "run_id", "static_mesh"):
                if key in header:
                    metadata[key] = header[key]
        if fo.follow:
            warnings.append(
                f"live stream: {len(times)} step(s) present when described; more may follow"
            )
        zones, fields, tree_warn = blueprint.describe_tree(first.tree)
        metadata["n_steps_present"] = len(times)
        domain = str(header.get("domain") or "other")
        desc = SourceDescription(
            connector_id=self.id,
            path=given,
            format=fmt,
            solver=str(header.get("solver") or "unknown"),
            solver_version=str(header["solver_version"]) if header.get("solver_version") else None,
            domain=domain if domain in _DOMAINS else "other",
            physics=dict(header.get("physics") or {}),
            zones=zones,
            fields=fields,
            time=TimeInfo(
                values=list(times) if len(times) > 1 or fo.follow else [],
                units=str(header.get("time_units") or "s"),
            ),
            units=dict(header.get("units") or {}),
            parameters=dict(header.get("parameters") or {}),
            metadata=metadata,
            warnings=warnings + tree_warn,
        )
        self._described[str(given.resolve())] = desc
        return desc

    def _report(self, given: Path, events: Sequence[str]) -> None:
        desc = self._described.pop(str(given.resolve()), None)
        if desc is not None:
            desc.warnings.extend(e for e in events if e not in desc.warnings)

    @staticmethod
    def _safe_times(src: spool.SpoolSource, live: bool) -> list[tuple[spool.Entry, float | None]]:
        """Step times of what is present; while a run is live the newest file may still be
        settling, which must not fail ``describe``."""
        entries = src.entries()
        out: list[tuple[spool.Entry, float | None]] = []
        for e in entries:
            try:
                out.extend(src.step_times([e]))
            except ConnectorError:
                if not live:
                    raise
                break
        return out

    # -- read ------------------------------------------------------------------------------
    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        given = Path(path)
        path, opts = self._resolve(given, options)
        fo = FollowOptions.from_options(opts)
        geometry = str(opts.get("geometry", "changed"))
        if self._is_adios(path, opts):
            return self._read_adios(given, path, opts, fo, zones, fields, times, geometry)
        return self._read_spool(given, path, opts, fo, zones, fields, times, geometry)

    @staticmethod
    def _live_window(times: TimeSelection) -> tuple[int, int | None, int]:
        """A time selection that can be applied to a stream of unknown length."""
        if times is None or (isinstance(times, str) and times.strip().lower() in ("all", "*", "")):
            return 0, None, 1
        if isinstance(times, slice):
            start, stop, step = times.start or 0, times.stop, times.step or 1
            if start >= 0 and (stop is None or stop >= 0) and step > 0:
                return start, stop, step
        raise SelectionError(
            f"time selection {times!r} cannot be applied to a live stream",
            hint="while following use all or a forward range start:stop:step; select freely "
            "once the run has ended (without follow)",
        )

    def _emit(
        self,
        steps: Iterator[tuple[int, Step]],
        conv: blueprint.StepConverter,
        transient: bool,
    ) -> Iterator[RawSample]:
        last_time: float | None = None
        for ordinal, step in steps:
            t: float | None = None
            if transient:
                t = step.time if step.time is not None else float(step.index)
                if last_time is not None and t <= last_time:
                    raise ConnectorError(
                        f"step {step.index} has time {t:g}, not after the previous {last_time:g}",
                        hint="the producer must emit increasing times (a restarted solver "
                        "should open the Emitter with restart_from=<checkpoint step>)",
                    )
                last_time = t
            yield from conv.convert(step.tree, time=t, time_index=ordinal if transient else None)

    def _read_spool(
        self,
        given: Path,
        path: Path,
        opts: Mapping[str, Any],
        fo: FollowOptions,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
        geometry: str,
    ) -> Iterator[RawSample]:
        src = self._spool_source(path, opts, fo)
        if fo.follow:
            src.wait_for_first(fo)
        header = src.header
        fmt = "conduit-relay" if src.flavor() == "relay" else "frontier-spool"
        entries = src.entries()
        named = entries[0].path if fmt == "conduit-relay" and entries else path
        conv = blueprint.StepConverter(
            _base_id(named, header),
            zones=zones,
            fields=fields,
            metadata=self._metadata(header, path, fmt),
            geometry=geometry,
        )
        if fo.follow:
            start, stop, stride = self._live_window(times)
            live = itertools.islice(enumerate(src.follow(fo)), start, stop, stride)
            try:
                yield from self._emit(self._primed(live, src, conv), conv, True)
            finally:
                self._report(given, [*src.events, *conv.warnings])
            return
        pairs = src.step_times()
        if not pairs:
            raise UnsupportedSourceError(
                f"{src.directory} holds no steps yet",
                hint="follow the live stream with --option follow=true",
            )
        all_times = [t if t is not None else float(e.index) for e, t in pairs]
        idxs = select_time_indices(all_times if len(pairs) > 1 else [], times)
        transient = len(pairs) > 1

        def finite() -> Iterator[tuple[int, Step]]:
            for i in idxs:
                step = src.load(pairs[i][0])
                if step.time is None:
                    step.time = all_times[i]
                yield i, step

        yield from self._emit(self._primed(finite(), src, conv), conv, transient)

    @staticmethod
    def _primed(
        steps: Iterator[tuple[int, Step]], src: spool.SpoolSource, conv: blueprint.StepConverter
    ) -> Iterator[tuple[int, Step]]:
        """Static-mesh streams store the mesh once: when the first step that is read omits it
        (a time selection, or a follower that starts late), load the latest earlier step that
        has it."""
        for ordinal, step in steps:
            if not step.has_mesh and conv.previous is None:
                donor = None
                for e in src.entries():
                    if e.index >= step.index:
                        break
                    if e.flavor != "npz" or spool.read_step_meta(e.path).get("has_mesh", True):
                        donor = e
                if donor is None:
                    raise ConnectorError(
                        f"step {step.index} carries no mesh and no earlier step in "
                        f"{src.directory} has one",
                        hint="static_mesh streams need their first step; do not delete it",
                    )
                conv.previous = blueprint.to_tree(src.load(donor).tree)
            yield ordinal, step

    def _read_adios(
        self,
        given: Path,
        path: Path,
        opts: Mapping[str, Any],
        fo: FollowOptions,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
        geometry: str,
    ) -> Iterator[RawSample]:
        src = self._adios_source(path, opts)
        live = fo.follow or src.streaming
        if live:
            start, stop, stride = self._live_window(times)
            wanted: set[int] | None = None
            transient = True
        else:
            all_times = src.scan_times()
            if not all_times:
                raise UnsupportedSourceError(
                    f"{path} holds no steps", hint="follow a live stream with --option follow=true"
                )
            transient = len(all_times) > 1
            wanted = set(select_time_indices(all_times if transient else [], times))
            start, stop, stride = 0, None, 1
        first = src.peek(fo, live)
        fmt = "adios2-sst" if src.streaming else "adios2-bp"
        conv = blueprint.StepConverter(
            _base_id(path, src.header),
            zones=zones,
            fields=fields,
            metadata=self._metadata(src.header, path, fmt),
            geometry=geometry,
        )
        del first

        def selected() -> Iterator[tuple[int, Step]]:
            for ordinal, step in enumerate(src.steps(fo, live)):
                if wanted is not None and ordinal not in wanted:
                    continue
                yield ordinal, step

        try:
            yield from self._emit(
                itertools.islice(selected(), start, stop, stride), conv, transient
            )
        finally:
            src.close()
            self._report(given, [*src.events, *conv.warnings])


__all__ = ["POINTER_NAME", "CatalystConnector", "StreamTimeoutError"]
