# SPDX-License-Identifier: Apache-2.0
"""Connector discovery and dispatch.

The framework ships no connector. Every connector comes from an installed plugin, found through
the entry-point group ``frontier.connectors``:

.. code-block:: toml

    [project.entry-points."frontier.connectors"]
    myformat = "my_pkg.connector:MyConnector"

First-party plugins (``frontier-connector-*``) are loaded first, then third-party ones, so a
third party can replace a connector by registering the same id. Plugins that are disabled in
``~/.frontier/plugins.toml``, incompatible with this framework's data-model version or carry an
invalid manifest are skipped (see :mod:`frontier_connectors.plugins`). A connector that fails
to import is kept as a :class:`BrokenConnector` so ``list`` can show the reason instead of
hiding it: one broken plugin never takes the registry down.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
from frontier_connectors.model import (
    MATURITIES,
    Connector,
    Detection,
    RawSample,
    SourceDescription,
    TimeSelection,
)
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as plugin_state
from frontier_connectors.plugins.manifest import GROUP_CONNECTORS

log = logging.getLogger("frontier_connectors")

ENTRY_POINT_GROUP = GROUP_CONNECTORS


@dataclass
class ConnectorInfo:
    """One row of the registry table (``list --json``)."""

    id: str
    name: str
    vendor: str
    formats: list[str]
    extensions: list[str]
    license: str
    maturity: str
    temporal: list[str]
    kinds: list[str]
    #: optional pip packages (extra engines, optional variants of the format)
    requires: list[str] = field(default_factory=list)
    #: entries of ``requires`` that are not installed
    missing: list[str] = field(default_factory=list)
    #: available | missing-deps | planned | broken
    status: str = "available"
    source: str = "runtime"  # plugin:<plugin id> | runtime
    #: id of the plugin providing the connector ("" for runtime registrations)
    plugin: str = ""
    export_help: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrokenConnector:
    """Placeholder for a connector whose module failed to import."""

    formats: list[str] = []
    vendor = ""
    license = ""
    maturity = "planned"

    def __init__(self, cid: str, target: str, error: BaseException) -> None:
        self.id = cid
        self.name = cid
        self.target = target
        self.error = error

    def detect(self, path: Path) -> Detection | None:
        return None

    def describe(self, path: Path) -> SourceDescription:
        raise ConnectorError(
            f"connector {self.id!r} failed to load: {self.error!r}",
            hint=f"fix or reinstall the package providing {self.target}",
        )

    def read(self, path: Path, **_: Any) -> Iterator[RawSample]:
        self.describe(path)
        yield from ()


_runtime: dict[str, Any] = {}
_cache: dict[str, Any] | None = None
_sources: dict[str, str] = {}


def register(connector: Any) -> None:
    """Register a connector instance at runtime (tests, notebooks). Overrides same-id entries."""
    _runtime[connector.id] = connector
    _sources[connector.id] = "runtime"
    if _cache is not None:
        _cache[connector.id] = connector


def unregister(connector_id: str) -> None:
    _runtime.pop(connector_id, None)
    reload()


def reload() -> None:
    """Forget the cached connector instances (call after installing a plugin in-process)."""
    global _cache
    _cache = None


def connectors() -> dict[str, Any]:
    """All loaded connectors by id: those of every enabled plugin (first-party plugins first, so
    third parties can override an id), then runtime registrations."""
    global _cache
    if _cache is not None:
        return _cache
    found: dict[str, Any] = {}
    try:
        state = plugin_state.load()
        records = discovery.discover(state)
    except Exception as e:  # noqa: BLE001 - never let plugin discovery take the registry down
        log.warning("plugin discovery failed: %r", e)
        state, records = plugin_state.PluginState(), []
    for rec in records:
        if not rec.enabled:
            continue
        for name, target in rec.connectors.items():
            if state.is_disabled("connectors", name):
                continue
            try:
                inst = discovery.load_target(target)
                cid = getattr(inst, "id", None) or name
            except Exception as e:  # noqa: BLE001 - a broken connector must not take the registry down
                log.debug("connector %s of plugin %s failed to load: %r", name, rec.id, e)
                inst, cid = BrokenConnector(name, target, e), name
            if state.is_disabled("connectors", cid):
                continue
            found[cid] = inst
            _sources[cid] = f"plugin:{rec.id}"
    found.update(_runtime)
    _cache = found
    return found


def get(connector_id: str) -> Any:
    """Connector by id; raises :class:`UnsupportedSourceError` with what to do about it (the
    plugin that provides it when it is known but not installed or disabled)."""
    reg = connectors()
    if connector_id not in reg:
        raise UnsupportedSourceError(
            f"unknown connector {connector_id!r}",
            hint=discovery.missing_hint("connector", connector_id)
            or "known ids: " + ", ".join(sorted(reg)),
        )
    return reg[connector_id]


def _plugin_of(connector_id: str) -> str:
    source = _sources.get(connector_id, "")
    return source.removeprefix("plugin:") if source.startswith("plugin:") else ""


def _missing(requires: Sequence[str]) -> list[str]:
    missing = []
    for req in requires:
        module = req.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        module = _IMPORT_NAMES.get(module.lower(), module.replace("-", "_"))
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(req)
        except (ImportError, ValueError):
            missing.append(req)
    return missing


#: pip name -> import name for the optional readers
_IMPORT_NAMES = {
    "pynastran": "pyNastran",
    "lasso-python": "lasso",
    "ansys-mapdl-reader": "ansys.mapdl.reader",
    "ansys-dpf-core": "ansys.dpf.core",
    "ansys-fluent-core": "ansys.fluent.core",
    "matlabengine": "matlab.engine",
    "pytecplot": "tecplot",
    "simscale-sdk": "simscale_sdk",
    "netcdf4": "netCDF4",
    "pyarrow": "pyarrow",
    "cadquery-ocp": "OCP",
}


def info(connector: Any) -> ConnectorInfo:
    """Registry row for one connector (optional attributes get defaults)."""
    cid = getattr(connector, "id", "?")
    if isinstance(connector, BrokenConnector):
        return ConnectorInfo(
            id=cid,
            name=cid,
            vendor="",
            formats=[],
            extensions=[],
            license="",
            maturity="planned",
            temporal=[],
            kinds=[],
            status="broken",
            source=_sources.get(cid, "runtime"),
            plugin=_plugin_of(cid),
            note=f"{type(connector.error).__name__}: {connector.error}",
        )
    maturity = getattr(connector, "maturity", "verified-synthetic")
    if maturity not in MATURITIES:
        maturity = "planned"
    requires = list(getattr(connector, "requires", []) or [])
    missing = _missing(requires)
    # `requires` lists optional readers/engines; only `hard_requires` (packages without which
    # the connector cannot read anything) turn the status into missing-deps
    hard_missing = _missing(list(getattr(connector, "hard_requires", []) or []))
    status = (
        "planned" if maturity == "planned" else ("missing-deps" if hard_missing else "available")
    )
    return ConnectorInfo(
        id=cid,
        name=getattr(connector, "name", cid),
        vendor=getattr(connector, "vendor", ""),
        formats=list(getattr(connector, "formats", []) or []),
        extensions=list(getattr(connector, "extensions", []) or []),
        license=getattr(connector, "license", ""),
        maturity=maturity,
        temporal=list(getattr(connector, "temporal", ["steady"]) or []),
        kinds=list(getattr(connector, "kinds", ["mesh"]) or []),
        requires=requires,
        missing=missing,
        status=status,
        source=_sources.get(cid, "runtime"),
        plugin=_plugin_of(cid),
        export_help=getattr(connector, "export_help", ""),
        note=getattr(connector, "status_note", ""),
    )


def list_connectors() -> list[ConnectorInfo]:
    return [info(c) for c in connectors().values()]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|") or "-"


def support_matrix_markdown(rows: Sequence[ConnectorInfo] | None = None) -> str:
    """The support matrix of the README, generated from the registry so it cannot drift."""
    rows = list(rows) if rows is not None else list_connectors()
    lines = [
        "| Connector | Format / tool | Vendor | Files | Time | Kinds | Maturity "
        "| Reader and licence | Getting the data out |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        temporal = "steady + transient" if "transient" in r.temporal else "steady"
        files = ", ".join(f"`{e}`" for e in r.extensions) or "-"
        deps = r.license + (f"; optional: {', '.join(r.requires)}" if r.requires else "")
        how = r.export_help + (f" *{r.note}*" if r.note else "")
        lines.append(
            f"| `{r.id}` | {_cell(r.name)} | {_cell(r.vendor)} | {files} | {temporal} "
            f"| {_cell(', '.join(r.kinds))} | {r.maturity} | {_cell(deps)} | {_cell(how)} |"
        )
    return "\n".join(lines)


def detect(path: str | Path) -> list[Detection]:
    """Run every connector's ``detect`` on ``path``; best match first. A connector that raises
    is skipped (logged at debug level): detection must never fail because one plugin is buggy."""
    p = Path(path)
    if not p.exists():
        raise UnsupportedSourceError(
            f"{p} does not exist", hint="pass a file or a case directory that exists"
        )
    found: list[Detection] = []
    for cid, c in connectors().items():
        try:
            d = c.detect(p)
        except Exception as e:  # noqa: BLE001
            log.debug("connector %s detect() raised %r", cid, e)
            continue
        if d is not None:
            found.append(d)
    found.sort(key=lambda d: -d.confidence)
    return found


def resolve(path: str | Path, connector: str | None = None) -> Any:
    """Pick the connector for ``path``: the named one, else the best detection."""
    if connector:
        return get(connector)
    found = detect(path)
    if not found:
        raise UnsupportedSourceError(
            f"no connector recognises {path}",
            hint="run `python -m frontier_connectors list` for the loaded connectors and "
            "`python -m frontier_connectors plugins search <format>` for installable ones, or "
            "force one with --connector <id> (generic containers: hdf5, netcdf, matlab, tabular "
            "take a --mapping config)",
        )
    return get(found[0].connector_id)


def _accepts(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def describe(
    path: str | Path,
    connector: str | None = None,
    *,
    options: Mapping[str, Any] | None = None,
) -> SourceDescription:
    """Describe ``path`` with the named or best-matching connector."""
    c = resolve(path, connector)
    if options and _accepts(c.describe, "options"):
        return c.describe(Path(path), options=options)
    return c.describe(Path(path))


def read(
    path: str | Path,
    connector: str | None = None,
    *,
    zones: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    times: TimeSelection = None,
    options: Mapping[str, Any] | None = None,
) -> Iterator[RawSample]:
    """Lazily read ``path`` as ``RawSample`` objects (see :mod:`frontier_connectors.model`)."""
    c = resolve(path, connector)
    kwargs: dict[str, Any] = {"zones": zones, "fields": fields, "times": times}
    if options and _accepts(c.read, "options"):
        kwargs["options"] = options
    return c.read(Path(path), **kwargs)


def is_connector(obj: Any) -> bool:
    """Structural check against the :class:`Connector` protocol."""
    return isinstance(obj, Connector)
