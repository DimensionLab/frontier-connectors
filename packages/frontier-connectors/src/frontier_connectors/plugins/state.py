# SPDX-License-Identifier: Apache-2.0
"""Plugin state in ``~/.frontier/plugins.toml``: what is disabled, where exporters were copied.

.. code-block:: toml

    version = 1

    [disabled]                       # everything not listed is enabled
    plugins = ["luminary"]           # whole plugins, by plugin id
    connectors = ["fluent-live"]     # single connectors of an otherwise enabled plugin
    importers = []
    exporters = []

    [sources]                        # defaults for `plugins install` / `plugins search`
    find_links = ["/path/to/wheels"]
    index_url = ""

    [[deployed]]                     # written by `plugins install-exporter`
    vendor = "matlab"
    plugin = "exporter-matlab"
    version = "0.2.0"
    path = "/home/me/case/matlab"
    installed_at = "2026-10-10T12:00:00+00:00"

Enabling and disabling never touches the installed distributions, so it works without network
access and without write access to the environment. The file location follows
``FRONTIER_HOME`` (default ``~/.frontier``); the Rust crate reads the same file.
"""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 1
STATE_FILE = "plugins.toml"
HOME_ENV = "FRONTIER_HOME"

#: scopes of `enable`/`disable`: plugin ids, or one connector / importer / exporter
SCOPES: tuple[str, ...] = ("plugins", "connectors", "importers", "exporters")


def frontier_home() -> Path:
    env = os.environ.get(HOME_ENV)
    return Path(env).expanduser() if env else Path.home() / ".frontier"


def state_path() -> Path:
    return frontier_home() / STATE_FILE


@dataclass
class Deployed:
    """One copy of an exporter's scripts made by ``plugins install-exporter``."""

    vendor: str
    plugin: str
    version: str
    path: str
    installed_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "vendor": self.vendor,
            "plugin": self.plugin,
            "version": self.version,
            "path": self.path,
            "installed_at": self.installed_at,
        }


@dataclass
class PluginState:
    disabled: dict[str, list[str]] = field(default_factory=lambda: {s: [] for s in SCOPES})
    find_links: list[str] = field(default_factory=list)
    index_url: str = ""
    deployed: list[Deployed] = field(default_factory=list)
    #: set when the file exists but could not be parsed (state then behaves as empty)
    error: str = ""
    path: Path = field(default_factory=state_path)

    def is_disabled(self, scope: str, name: str) -> bool:
        return name in self.disabled.get(scope, [])

    def set_enabled(self, scope: str, name: str, enabled: bool) -> bool:
        """Returns whether anything changed."""
        names = self.disabled.setdefault(scope, [])
        if enabled and name in names:
            names.remove(name)
            return True
        if not enabled and name not in names:
            names.append(name)
            names.sort()
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "disabled": {s: list(self.disabled.get(s, [])) for s in SCOPES},
            "sources": {"find_links": list(self.find_links), "index_url": self.index_url},
            "deployed": [d.to_dict() for d in self.deployed],
            "error": self.error,
        }


def _strings(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def load(path: Path | None = None) -> PluginState:
    """Read the state file. A missing file is the empty state; an unreadable one is the empty
    state with ``error`` set (a damaged state file must not disable the whole registry)."""
    p = Path(path) if path is not None else state_path()
    state = PluginState(path=p)
    if not p.is_file():
        return state
    try:
        doc = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        state.error = f"{p} could not be read ({e}); every plugin is treated as enabled"
        return state
    disabled = doc.get("disabled", {})
    if isinstance(disabled, dict):
        for scope in SCOPES:
            state.disabled[scope] = sorted(set(_strings(disabled.get(scope))))
    sources = doc.get("sources", {})
    if isinstance(sources, dict):
        state.find_links = _strings(sources.get("find_links"))
        state.index_url = str(sources.get("index_url", "") or "")
    for item in doc.get("deployed", []) if isinstance(doc.get("deployed"), list) else []:
        if isinstance(item, dict) and item.get("vendor") and item.get("path"):
            state.deployed.append(
                Deployed(
                    vendor=str(item["vendor"]),
                    plugin=str(item.get("plugin", "")),
                    version=str(item.get("version", "")),
                    path=str(item["path"]),
                    installed_at=str(item.get("installed_at", "")),
                )
            )
    return state


def _q(text: str) -> str:
    """A TOML basic string (JSON string escaping is a subset of TOML's)."""
    return json.dumps(str(text), ensure_ascii=False)


def _array(items: list[str]) -> str:
    return "[" + ", ".join(_q(i) for i in items) + "]"


def dumps(state: PluginState) -> str:
    lines = [
        "# Frontier plugin state. Written by `python -m frontier_connectors plugins ...`;",
        "# safe to edit by hand. Everything not listed under [disabled] is enabled.",
        f"version = {STATE_VERSION}",
        "",
        "[disabled]",
    ]
    lines += [f"{scope} = {_array(state.disabled.get(scope, []))}" for scope in SCOPES]
    lines += [
        "",
        "[sources]",
        f"find_links = {_array(state.find_links)}",
        f"index_url = {_q(state.index_url)}",
    ]
    for d in state.deployed:
        lines += ["", "[[deployed]]"] + [f"{k} = {_q(v)}" for k, v in d.to_dict().items()]
    return "\n".join(lines) + "\n"


def save(state: PluginState) -> Path:
    """Write the state atomically (temp file in the same directory, then rename)."""
    p = state.path
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".plugins-", suffix=".toml", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(state))
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p
