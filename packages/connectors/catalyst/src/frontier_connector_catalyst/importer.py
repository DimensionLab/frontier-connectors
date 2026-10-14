# SPDX-License-Identifier: Apache-2.0
"""The ``catalyst`` importer: follow a live in-situ stream and build the dataset while the
solver runs.

``python -m frontier_connectors import catalyst <spool-dir | stream.bp | stream.sst> --out ds``

The framework's import is "fetch, then convert". A live stream has nothing to fetch, so
``fetch`` only waits until the stream exists (the solver may start later than the reader) and
leaves a small pointer file in the work directory. The pointer is recognised by the ``catalyst``
connector, whose ``read`` then follows the stream with the options recorded here; since the
dataset writer consumes samples lazily, steps are appended to the dataset as they arrive.

Options (``--option key=value``): ``follow`` (default ``true`` here), ``timeout``, ``poll``,
``until`` (``done-marker`` | ``timeout``), ``on_timeout`` (``stop`` | ``error``),
``on_restart``, ``order_grace``, ``partial_grace``, ``settle``, ``max_steps``, ``geometry``
(``changed`` | ``always``), ``name`` (relay series), ``engine`` / ``adios_parameters`` /
``fides_order`` (ADIOS2). They are described in
:class:`frontier_connector_catalyst.stream.FollowOptions` and the README.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from frontier_connector_catalyst import adios, spool
from frontier_connector_catalyst.connector import POINTER_NAME
from frontier_connector_catalyst.stream import FollowOptions, StreamTimeoutError
from frontier_connectors.importers import BaseImporter, PlannedCall


def _options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = dict(options or {})
    opts.setdefault("follow", True)
    return opts


def _is_adios(source: Path, opts: Mapping[str, Any]) -> bool:
    return bool(opts.get("engine")) or source.suffix in (".bp", ".bp4", ".bp5", ".sst")


class CatalystImporter(BaseImporter):
    id = "catalyst"
    name = "ParaView Catalyst 2 / in-situ stream follower"
    vendor = "Kitware / LLNL / ORNL (open source)"
    license = "Apache-2.0"
    maturity = "verified-synthetic"
    credentials_env: list[str] = []
    requires = ["adios2"]
    #: connector that reads what ``fetch`` leaves behind
    connector = "catalyst"
    source_help = (
        "<spool directory written by frontier_emit / the Catalyst pipeline / Conduit relay> | "
        "<ADIOS2 file.bp> | <SST stream name or .sst contact file>"
    )
    status_note = (
        "Follows a live stream and appends to the dataset step by step. Needs no network and no "
        "credentials."
    )
    api_reference = (
        "https://catalyst-in-situ.readthedocs.io/en/latest/ ; "
        "https://docs.paraview.org/en/latest/Catalyst/blueprints.html ; "
        "https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html ; "
        "https://adios2.readthedocs.io/en/latest/ (read 2026-10-10)"
    )

    def plan(self, source: str, *, options: Mapping[str, Any] | None = None) -> list[PlannedCall]:
        opts = _options(options)
        fo = FollowOptions.from_options(opts)
        src = Path(source).expanduser()
        shown = {
            "follow": fo.follow,
            "timeout_s": fo.timeout,
            "poll_s": fo.poll,
            "until": "done-marker" if fo.until == "done" else "timeout",
            "on_timeout": fo.on_timeout,
        }
        if _is_adios(src, opts):
            name, implied = adios.stream_name(src)
            engine = str(opts.get("engine") or implied or "BPFile")
            return [
                PlannedCall(
                    "SDK",
                    "adios2.Stream(io, path, 'r')",
                    "open the ADIOS2 file / stream (waits for the writer)",
                    {"path": name, "engine": engine, "OpenTimeoutSecs": fo.timeout},
                ),
                PlannedCall(
                    "SDK",
                    "adios2.Stream.begin_step(timeout=...)",
                    "read every step as it is published and append it to the dataset",
                    shown,
                ),
            ]
        calls = [
            PlannedCall(
                "FILE",
                str(src / spool.HEADER_NAME),
                "stream header: solver, units, run id (optional for Conduit relay series)",
            ),
            PlannedCall(
                "FILE",
                str(src / "step_<n>.npz") + " | " + str(src / "<name>.cycle_<n>.root"),
                "read each complete step in order while the solver runs; hidden .tmp files "
                "are never opened",
                shown,
            ),
        ]
        if fo.until == "done":
            calls.append(
                PlannedCall("FILE", str(src / spool.DONE_NAME), "end marker: stop following")
            )
        return calls

    def fetch(self, source: str, dest: Path, *, options: Mapping[str, Any] | None = None) -> Path:
        opts = _options(options)
        fo = FollowOptions.from_options(opts)
        src = Path(source).expanduser().resolve()
        if fo.follow:
            self._wait_for(src, opts, fo)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        pointer = {"source": str(src), "options": opts, "importer": self.id}
        tmp = dest / f".{POINTER_NAME}.tmp"
        tmp.write_text(json.dumps(pointer, indent=1))
        tmp.replace(dest / POINTER_NAME)
        return dest

    @staticmethod
    def _wait_for(src: Path, opts: Mapping[str, Any], fo: FollowOptions) -> None:
        """Block until the stream exists: the reader is often started before the solver."""
        if _is_adios(src, opts):
            name, implied = adios.stream_name(src)
            engine = str(opts.get("engine") or implied or "BPFile")
            if engine != "SST":
                return  # ADIOS2 itself waits for a BP file (OpenTimeoutSecs)
            contact = Path(name + ".sst")
            start = fo.clock()
            while not contact.exists():
                if fo.clock() - start > fo.timeout:
                    raise StreamTimeoutError(
                        f"the SST contact file {contact} did not appear within {fo.timeout:g} s",
                        hint="start the writer, or raise --option timeout=<seconds>",
                    )
                fo.sleep(fo.poll)
            return
        start = fo.clock()
        while not src.exists():
            if fo.clock() - start > fo.timeout:
                raise StreamTimeoutError(
                    f"{src} did not appear within {fo.timeout:g} s",
                    hint="start the solver (it creates the spool directory), or raise "
                    "--option timeout=<seconds>",
                )
            fo.sleep(fo.poll)
        spool.SpoolSource(src, name=str(opts["name"]) if opts.get("name") else None).wait_for_first(
            fo
        )
