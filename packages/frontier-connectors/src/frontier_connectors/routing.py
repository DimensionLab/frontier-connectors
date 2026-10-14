# SPDX-License-Identifier: Apache-2.0
"""Vendor routes: tools whose native result files are not openly readable (or that simply write
an interchange format).

Each connector here is an honest registry row: it tells the user how to get data out of the
tool, raises :class:`NeedsExportError` with tool-specific instructions for the native files it
recognises, and, when it is pointed at an export in an interchange format that another
connector already reads, delegates ``describe``/``read`` to that *route* connector and stamps
the solver name on the result.

``detect`` only fires on tool-specific markers (a native extension, a documented directory
layout, the tool's name in an export header). A generic ``.case``/``.cgns``/``.vti`` file is
never claimed: those stay with the ``ensight``/``cgns``/``vtk`` connectors.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from frontier_connectors.errors import ConnectorError, NeedsExportError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    RawSample,
    SourceDescription,
    TimeSelection,
    match_extension,
)

#: a routed target: (route connector id, path handed to it, why it was recognised)
Target = tuple[str, Path, str]


def _listdir(path: Path, limit: int = 4000) -> list[Path]:
    try:
        return sorted(itertools.islice(path.iterdir(), limit))
    except OSError:
        return []


def _ensight_cases(directory: Path, depth: int = 1) -> list[Path]:
    """EnSight case files in ``directory`` and (``depth`` levels of) its sub-directories."""
    from frontier_connector_ensight.connector import find_cases

    found = list(find_cases(directory))
    if depth > 0:
        for sub in _listdir(directory, 200):
            if sub.is_dir() and not sub.name.startswith("."):
                found.extend(_ensight_cases(sub, depth - 1))
    return found


def _ensight_header(case: Path) -> str:
    """Lower-cased text that identifies the writer of an EnSight export: the comment lines of
    the case file and the two description lines of its geometry file (part names are not
    looked at: a part called ``converge_nozzle`` must not select a vendor)."""
    from frontier_connector_ensight.connector import (
        case_version,
        geometry_description,
        parse_case,
    )

    if case_version(case) is None:
        return ""
    try:
        parsed = parse_case(case)
        times = parsed.times()
        geo = parsed.geometry_file(times[0] if times else None)
    except ConnectorError:
        return ""
    return " | ".join([*parsed.comments, *geometry_description(geo)]).lower()


class VendorRouteConnector(BaseConnector):
    """Base of the vendor-route connectors (see the module docstring)."""

    solver: str = ""
    domain: str = "cfd"
    #: ids of the connectors that read this tool's exports, preferred first
    routes: list[str] = []
    #: proprietary native files, recognised by extension
    native_extensions: list[str] = []
    #: what to do with a native file (the hint of the ``NeedsExportError``)
    native_hint: str = ""
    maturity = "planned"
    license = "Apache-2.0 (no vendor code; delegates to open-format readers)"
    native_confidence: float = 0.6

    # -- hooks --------------------------------------------------------------------------
    def find_export(self, path: Path) -> Target | None:
        """Tool-specific marker for an export this connector can route. Must be cheap and must
        not raise."""
        return None

    def native_reason(self, path: Path) -> str | None:
        """Why ``path`` is a native (unreadable) file of this tool, else ``None``."""
        if path.is_file():
            ext = match_extension(path, self.native_extensions)
            if ext is not None:
                return f"{path.name} is a native {self.solver} file ({ext})"
        return None

    def enrich(self, desc: SourceDescription, target: Target) -> None:
        """Add tool-specific metadata found next to the export."""

    # -- contract -----------------------------------------------------------------------
    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            target = self.find_export(path)
            if target is not None:
                route, _, why = target
                return Detection(self.id, path, 0.98, f"{self.id}+{route}", [why])
            reason = self.native_reason(path)
        except OSError:
            return None
        if reason is None:
            return None
        return Detection(
            self.id,
            path,
            self.native_confidence,
            f"{self.id}-native",
            [reason],
            needs=[self.export_help],
        )

    def _target(self, path: Path) -> Target:
        path = Path(path)
        if not path.exists():
            raise UnsupportedSourceError(f"{path} does not exist", hint=self.export_help)
        target = self.find_export(path)
        if target is not None:
            return target
        reason = self.native_reason(path)
        if reason is not None:
            raise NeedsExportError(
                f"{reason}: proprietary, not readable without {self.solver}",
                hint=self.native_hint or self.export_help,
            )
        # forced with --connector on a plain interchange file: let the first route that
        # recognises it read it
        from frontier_connectors import registry

        for route in self.routes:
            try:
                if registry.get(route).detect(path) is not None:
                    return route, path, f"read as {route}"
            except Exception:  # noqa: BLE001, S112 - a broken route connector is not this file's problem
                continue
        raise UnsupportedSourceError(
            f"{path} is not a {self.solver} export this connector can route",
            hint=self.export_help,
        )

    @staticmethod
    def _route_connector(route: str) -> Any:
        from frontier_connectors import registry

        return registry.get(route)

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        target = self._target(Path(path))
        route, routed, why = target
        desc: SourceDescription = self._route_connector(route).describe(routed)
        desc.connector_id = self.id
        desc.path = Path(path)
        desc.solver = self.solver
        desc.domain = self.domain
        desc.metadata.update({"route": route, "routed_path": str(routed), "recognised_by": why})
        self.enrich(desc, target)
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
        route, routed, _ = self._target(Path(path))
        inner = self._route_connector(route)
        kw: dict[str, Any] = {"options": options} if options else {}
        for s in inner.read(routed, zones=zones, fields=fields, times=times, **kw):
            s.metadata["solver"] = self.solver
            s.metadata["route"] = route
            s.metadata.setdefault("source", str(routed))
            yield s


def _case_with_marker(path: Path, needles: Sequence[str], depth: int = 0) -> Target | None:
    """An EnSight case (the file itself, or one inside a directory) whose header names the
    tool."""
    if path.is_file():
        cases = [path] if match_extension(path, [".case", ".encas"]) else []
    elif path.is_dir():
        cases = _ensight_cases(path, depth)
    else:
        cases = []
    for case in cases[:20]:
        header = _ensight_header(case)
        for needle in needles:
            if needle in header:
                return "ensight", case, f"EnSight export header mentions '{needle}'"
    return None
