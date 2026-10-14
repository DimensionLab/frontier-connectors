# SPDX-License-Identifier: Apache-2.0
"""File-based step transport: a *spool directory* that a running solver fills one complete step
at a time, and the follower that tails it.

Two layouts are read:

``frontier-spool`` version 1 (written by ``frontier_emit.py`` of the ``exporter-catalyst``
plugin; numpy only on both sides)
    ``frontier-stream.json`` (header), ``step_<8 digits>.npz`` (one step each) and
    ``frontier-stream.done`` (end marker). Every array of the step's Blueprint tree is one
    ``.npz`` member named by its path; the member ``__frontier_step__`` is a JSON document with
    ``step``, ``time``, ``cycle``, ``prev``, ``run_id``, ``has_mesh`` and the string leaves.
    Files appear under their final name by atomic rename, hidden ``.<name>.tmp-<pid>`` files
    are work in progress and are never opened.

Conduit relay Blueprint series (``<name>.cycle_<n>.root`` plus domain files, see
:mod:`frontier_connector_catalyst.relay`)
    Written by ``conduit.relay.io.blueprint.save_mesh`` / Ascent. These writers do not rename
    atomically, so a root is only read once it and the files it refers to have been unmodified
    for ``settle`` seconds, and a file that fails to decode is retried. There is no end marker
    in this layout: touch ``frontier-stream.done`` when the run ends or follow with
    ``until=timeout``.

The follower yields steps in increasing step number, exactly once, and only when complete.
"""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connector_catalyst import relay
from frontier_connector_catalyst.stream import FollowOptions, Step, StreamTimeoutError
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError

log = logging.getLogger("frontier_connectors.catalyst")

PROTOCOL = "frontier-spool"
VERSION = 1
HEADER_NAME = "frontier-stream.json"
DONE_NAME = "frontier-stream.done"
STEP_META = "__frontier_step__"
STEP_PATTERN = re.compile(r"^step_(\d{1,12})\.npz$")


class PartialStepError(Exception):
    """A visible step file cannot be read (yet)."""


@dataclass
class Entry:
    index: int
    path: Path
    flavor: str  # "npz" | "relay"


def read_json(path: Path) -> dict[str, Any] | None:
    """A small JSON file, ``None`` when absent; ``{}`` when empty or not (yet) valid."""
    try:
        txt = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        doc = json.loads(txt) if txt.strip() else {}
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def scan(directory: Path, name: str | None = None) -> list[Entry]:
    """Complete-looking step files of a spool directory, sorted by step number."""
    npz: list[Entry] = []
    roots: dict[str, list[Entry]] = {}
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return []
    for fname in names:
        if fname.startswith("."):
            continue
        m = STEP_PATTERN.match(fname)
        if m:
            npz.append(Entry(int(m.group(1)), directory / fname, "npz"))
            continue
        r = relay.ROOT_SERIES.match(fname)
        if r:
            roots.setdefault(r.group("base"), []).append(
                Entry(int(r.group("cycle")), directory / fname, "relay")
            )
        elif fname.endswith(".root"):
            roots.setdefault(fname[: -len(".root")], []).append(
                Entry(0, directory / fname, "relay")
            )
    if npz and roots and name is None:
        raise UnsupportedSourceError(
            f"{directory} mixes frontier-spool steps and Conduit relay roots",
            hint="keep one layout per directory, or select a relay series with "
            "--option name=<base>",
        )
    if npz and name is None:
        return sorted(npz, key=lambda e: e.index)
    if not roots:
        return []
    if name is None:
        if len(roots) > 1:
            raise UnsupportedSourceError(
                f"{directory} holds several relay series: {', '.join(sorted(roots))}",
                hint="select one with --option name=<base>",
            )
        name = next(iter(roots))
    if name not in roots:
        raise UnsupportedSourceError(
            f"no relay series {name!r} in {directory}",
            hint="available: " + ", ".join(sorted(roots)),
        )
    return sorted(roots[name], key=lambda e: e.index)


def read_step_meta(path: Path) -> dict[str, Any]:
    """The ``__frontier_step__`` document of a step file (arrays are not read)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if STEP_META not in z.files:
                raise UnsupportedSourceError(
                    f"{path} is not a frontier-spool step (no {STEP_META} member)",
                    hint="write steps with frontier_emit.Emitter (exporter-catalyst plugin)",
                )
            meta = json.loads(str(z[STEP_META][()]))
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, KeyError) as e:
        raise PartialStepError(f"{path.name}: {e}") from e
    if not isinstance(meta, dict) or meta.get("protocol") != PROTOCOL:
        raise UnsupportedSourceError(f"{path}: unknown step protocol {meta!r:.80}")
    if int(meta.get("version", 0)) > VERSION:
        raise UnsupportedSourceError(
            f"{path}: frontier-spool version {meta.get('version')} is newer than this reader",
            hint="update the catalyst connector plugin",
        )
    return meta


def load_npz_step(path: Path, index: int) -> Step:
    """Read one ``step_*.npz`` completely. Raises :class:`PartialStepError` if it is truncated."""
    meta = read_step_meta(path)
    tree: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key == STEP_META:
                    continue
                _set(tree, key, np.asarray(z[key]))
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, KeyError) as e:
        raise PartialStepError(f"{path.name}: {e}") from e
    for key, value in (meta.get("strings") or {}).items():
        _set(tree, key, str(value))
    prev = meta.get("prev")
    return Step(
        index=int(meta.get("step", index)),
        tree=tree,
        time=float(meta["time"]) if meta.get("time") is not None else None,
        cycle=int(meta["cycle"]) if meta.get("cycle") is not None else None,
        prev=int(prev) if prev is not None else None,
        run_id=meta.get("run_id"),
        has_mesh=bool(meta.get("has_mesh", True)),
    )


def _set(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split("/") if p]
    cur = tree
    for p in parts[:-1]:
        nxt = cur.setdefault(p, {})
        if not isinstance(nxt, dict):
            raise ConnectorError(f"step member {path!r} collides with a leaf")
        cur = nxt
    cur[parts[-1]] = value


def load_relay_step(path: Path, index: int) -> Step:
    try:
        doc = relay.load_file(path)
        tree = relay.load_root(path)
    except relay.IncompleteFileError as e:
        raise PartialStepError(str(e)) from e
    time, cycle = relay.root_state(doc) if isinstance(doc, dict) else (None, None)
    return Step(index=index, tree=tree, time=time, cycle=cycle if cycle is not None else index)


def load_entry(entry: Entry) -> Step:
    if entry.flavor == "npz":
        return load_npz_step(entry.path, entry.index)
    return load_relay_step(entry.path, entry.index)


class SpoolSource:
    """A spool directory (or a single step / root file) as a source of steps."""

    def __init__(self, path: Path, *, name: str | None = None) -> None:
        self.path = Path(path)
        self.name = name
        self.single: Entry | None = None
        if self.path.is_file():
            m = STEP_PATTERN.match(self.path.name)
            r = relay.ROOT_SERIES.match(self.path.name)
            if m:
                self.single = Entry(int(m.group(1)), self.path, "npz")
            elif self.path.suffix == ".npz":
                self.single = Entry(0, self.path, "npz")
            else:
                self.single = Entry(int(r.group("cycle")) if r else 0, self.path, "relay")
            self.directory = self.path.parent
        else:
            self.directory = self.path
        #: what happened while following: restarts, gaps, how the stream ended
        self.events: list[str] = []
        self.end: dict[str, Any] = {}

    # -- finite view -----------------------------------------------------------------------
    @property
    def header(self) -> dict[str, Any]:
        if self.single is not None and self.single.flavor == "relay":
            return {}
        return read_json(self.directory / HEADER_NAME) or {}

    def done(self) -> dict[str, Any] | None:
        """The end marker (``{}`` for an empty marker file), ``None`` while the run goes on."""
        doc = read_json(self.directory / DONE_NAME)
        if doc is None:
            return None
        doc.setdefault("status", "complete")
        return doc

    def entries(self) -> list[Entry]:
        if self.single is not None:
            return [self.single]
        return scan(self.directory, self.name)

    def flavor(self) -> str:
        e = self.entries()
        return e[0].flavor if e else "npz"

    def step_times(self, entries: list[Entry] | None = None) -> list[tuple[Entry, float | None]]:
        """``(entry, time)`` of every step present, reading only the small metadata."""
        out: list[tuple[Entry, float | None]] = []
        for e in self.entries() if entries is None else entries:
            if e.flavor == "npz":
                meta = self._meta(e)
                t = meta.get("time")
                out.append((e, float(t) if t is not None else None))
            else:
                try:
                    doc = relay.load_file(e.path)
                except relay.IncompleteFileError as err:
                    raise self._corrupt(e, err) from err
                out.append((e, relay.root_state(doc)[0] if isinstance(doc, dict) else None))
        return out

    def _meta(self, e: Entry) -> dict[str, Any]:
        try:
            return read_step_meta(e.path)
        except PartialStepError as err:
            raise self._corrupt(e, err) from err

    @staticmethod
    def _corrupt(e: Entry, err: BaseException) -> ConnectorError:
        return ConnectorError(
            f"step file {e.path} is truncated or corrupt: {err}",
            hint="the producer must write to a temporary name and rename (frontier_emit does); "
            "remove the file to convert the remaining steps, or follow the live stream with "
            "--option follow=true",
        )

    def load(self, entry: Entry) -> Step:
        try:
            return load_entry(entry)
        except PartialStepError as err:
            raise self._corrupt(entry, err) from err

    def read_steps(self, entries: list[Entry] | None = None) -> Iterator[Step]:
        for e in self.entries() if entries is None else entries:
            yield self.load(e)

    # -- live view -------------------------------------------------------------------------
    def wait_for_first(self, opts: FollowOptions) -> Entry:
        """Block until the first step is visible (the solver may not have started yet)."""
        start = opts.clock()
        while True:
            entries = self.entries()
            if entries:
                return entries[0]
            if self.done() is not None and opts.until == "done":
                raise ConnectorError(
                    f"the stream in {self.directory} ended without a single step",
                    hint="check the solver log; the end marker is " + DONE_NAME,
                )
            if opts.clock() - start > opts.timeout:
                raise StreamTimeoutError(
                    f"no step appeared in {self.directory} within {opts.timeout:g} s",
                    hint="is the solver running and writing to this directory? raise "
                    "--option timeout=<seconds> for slow first steps",
                )
            opts.sleep(opts.poll)

    def _settled(self, entry: Entry, opts: FollowOptions) -> bool:
        """Relay files have no atomic rename: wait until they stopped changing."""
        if entry.flavor != "relay" or opts.settle <= 0:
            return True
        import time as _time

        try:
            files = relay.root_files(entry.path)
            newest = max(os.stat(f).st_mtime for f in [entry.path, *files])
        except (relay.IncompleteFileError, OSError):
            return False
        return _time.time() - newest >= opts.settle

    def follow(self, opts: FollowOptions, *, start_after: int | None = None) -> Iterator[Step]:
        """Yield steps as they complete, in increasing step number, each exactly once."""
        last = start_after
        last_progress = opts.clock()
        first_seen: dict[int, float] = {}
        bad: dict[int, tuple[float, int]] = {}
        run_id = self.header.get("run_id")
        count = 0
        while True:
            # the end marker is read BEFORE the directory: the producer writes it after the
            # last step, so a scan that follows a seen marker sees every step
            done = self.done() if opts.until == "done" else None
            now = opts.clock()
            header_run = self.header.get("run_id")
            if header_run and run_id and header_run != run_id:
                msg = (
                    f"the producer restarted (run {run_id[:8]} -> {header_run[:8]}); steps up to "
                    f"{last} are already consumed and are not read again"
                )
                if opts.on_restart == "error":
                    raise ConnectorError(
                        msg, hint="use on_restart=skip to keep following after a restart"
                    )
                self._event(msg)
            run_id = header_run or run_id
            pending = [e for e in self.entries() if last is None or e.index > last]
            waiting = False
            for entry in pending:
                first_seen.setdefault(entry.index, now)
                if done is None and not self._settled(entry, opts):
                    waiting = True
                    break
                try:
                    step = load_entry(entry)
                except PartialStepError as err:
                    try:
                        size = entry.path.stat().st_size
                    except OSError:
                        size = -1
                    since, old_size = bad.get(entry.index, (now, size))
                    if size != old_size:
                        since = now
                    bad[entry.index] = (since, size)
                    if done is not None or now - since > opts.partial_grace:
                        raise self._corrupt(entry, err) from err
                    waiting = True
                    break
                expected_prev = step.prev
                if (
                    expected_prev is not None
                    and expected_prev != last
                    and (last is None or expected_prev > last)
                ):
                    # the producer emitted a step between `last` and this one that is not
                    # visible yet (out-of-order arrival on a network file system)
                    if done is None and now - first_seen[entry.index] <= opts.order_grace:
                        waiting = True
                        break
                    self._event(
                        f"step {expected_prev} (before {entry.index}) never arrived; "
                        "continuing without it"
                    )
                bad.pop(entry.index, None)
                last = entry.index
                last_progress = opts.clock()
                count += 1
                yield step
                if opts.max_steps is not None and count >= opts.max_steps:
                    self.end = {"reason": "max_steps", "n_steps": count}
                    return
            if done is not None and not waiting:
                status = str(done.get("status", "complete"))
                self.end = {"reason": "done", "status": status, "n_steps": count, **done}
                if status != "complete":
                    self._event(
                        f"the producer ended the stream with status {status!r}: "
                        f"{done.get('message', '')}"
                    )
                return
            if opts.clock() - last_progress > opts.timeout:
                msg = (
                    f"no new complete step in {self.directory} for {opts.timeout:g} s "
                    f"(last step: {last}, {count} read)"
                )
                if count == 0 or opts.on_timeout == "error":
                    raise StreamTimeoutError(
                        msg,
                        hint="the solver stopped or died without writing the end marker "
                        f"{DONE_NAME}; use on_timeout=stop to keep what arrived, or raise "
                        "--option timeout=<seconds>",
                    )
                self.end = {"reason": "timeout", "n_steps": count, "last_step": last}
                if opts.until == "done":
                    self._event(msg + "; stopping (on_timeout=stop)")
                return
            opts.sleep(opts.poll)

    def _event(self, message: str) -> None:
        if message not in self.events:
            self.events.append(message)
            log.warning("%s", message)


def looks_like_spool(directory: Path) -> tuple[float, str] | None:
    """``(confidence, format)`` when a directory is a spool."""
    if not directory.is_dir():
        return None
    header = read_json(directory / HEADER_NAME)
    if header is not None and header.get("protocol") == PROTOCOL:
        return 1.0, "frontier-spool"
    try:
        names = [n for n in os.listdir(directory) if not n.startswith(".")]
    except OSError:
        return None
    if any(STEP_PATTERN.match(n) for n in names):
        return 0.9, "frontier-spool"
    roots = [n for n in names if n.endswith(".root")]
    if roots and relay.is_blueprint_root(directory / sorted(roots)[0]):
        return 0.85, "conduit-relay"
    return None
