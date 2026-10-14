# SPDX-License-Identifier: Apache-2.0
"""Find the installed plugins and decide which of them are loaded.

A plugin is an installed distribution with an entry point in the group ``frontier.plugins``
whose value is the import package holding ``plugin.toml``. Its connectors, importers and
exporter scripts are ordinary entry points in ``frontier.connectors``, ``frontier.importers``
and ``frontier.exporters``, so a distribution that only declares ``frontier.connectors``
(the form documented before manifests existed) still works: it is listed as a plugin without
a manifest.

Nothing here imports plugin code. A plugin is given one of these states:

``enabled``       loaded by the registry
``disabled``      installed, switched off in ``~/.frontier/plugins.toml``
``incompatible``  written against a frontier-dataset version this framework cannot serve
``invalid``       ``plugin.toml`` is missing or does not validate

Only ``enabled`` plugins reach the registry, and a failure in one plugin (bad manifest, import
error, an exception in ``detect``) is recorded on that plugin and never propagates.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata as importlib_metadata
from importlib import resources
from pathlib import Path
from typing import Any

from frontier_connectors.plugins import state as state_mod
from frontier_connectors.plugins.manifest import (
    DATA_MODEL_VERSION,
    GROUP_CONNECTORS,
    GROUP_EXPORTERS,
    GROUP_IMPORTERS,
    GROUP_PLUGINS,
    ManifestError,
    PluginManifest,
    data_model_compatible,
    load_from_package,
    package_dir,
)

log = logging.getLogger("frontier_connectors.plugins")

FRAMEWORK_DIST = "frontier-connectors"
#: first-party distribution name prefixes (loaded first, so third parties can override)
FIRST_PARTY_PREFIXES = ("frontier-connector-", "frontier-exporter-", "frontier-importer-")


def norm_dist(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def norm_vendor(name: str) -> str:
    return name.strip().lower().replace("_", "-")


@dataclass
class PluginRecord:
    """One installed plugin and what the registry decided about it."""

    id: str
    dist: str
    dist_version: str
    module: str = ""
    kind: str = "connector"
    manifest: PluginManifest | None = None
    #: enabled | disabled | incompatible | invalid
    status: str = "enabled"
    problems: list[str] = field(default_factory=list)
    #: entry points declared by the distribution: id -> "module:attribute"
    connectors: dict[str, str] = field(default_factory=dict)
    importers: dict[str, str] = field(default_factory=dict)
    #: vendor key -> import package holding the scripts
    exporters: dict[str, str] = field(default_factory=dict)
    location: str = ""
    first_party: bool = False

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"

    @property
    def version(self) -> str:
        return self.manifest.version if self.manifest else self.dist_version

    def to_dict(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "id": self.id,
            "kind": self.kind,
            "name": m.name if m else self.id,
            "summary": m.summary if m else "",
            "version": self.version,
            "dist": self.dist,
            "module": self.module,
            "status": self.status,
            "enabled": self.enabled,
            "installed": True,
            "problems": list(self.problems),
            "data_model_version": m.data_model_version if m else "",
            "vendor": m.vendor if m else "",
            "formats": list(m.formats) if m else [],
            "maturity": m.maturity if m else "",
            "licence": m.licence if m else "",
            "requires": list(m.requires) if m else [],
            "plugins": list(m.plugins) if m else [],
            "suggests": list(m.suggests) if m else [],
            "optional": [vars(o) for o in m.optional] if m else [],
            "auth": [a.to_provider() for a in m.auth] if m else [],
            "connectors": sorted(self.connectors),
            "importers": sorted(self.importers),
            "exporters": sorted(self.exporters),
            "location": self.location,
            "first_party": self.first_party,
            "has_manifest": m is not None,
        }


def _entry_points(dist: Any) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {
        GROUP_PLUGINS: {},
        GROUP_CONNECTORS: {},
        GROUP_IMPORTERS: {},
        GROUP_EXPORTERS: {},
    }
    try:
        eps = dist.entry_points
    except Exception as e:  # noqa: BLE001 - damaged metadata of an unrelated distribution
        log.debug("entry points of %r unreadable: %r", getattr(dist, "name", "?"), e)
        return groups
    for ep in eps:
        if ep.group in groups:
            groups[ep.group][ep.name] = ep.value
    return groups


def _record_with_manifest(
    pid: str, module: str, name: str, version: str, eps: dict[str, dict[str, str]]
) -> PluginRecord:
    rec = PluginRecord(
        id=pid,
        dist=name,
        dist_version=version,
        module=module,
        connectors=dict(eps[GROUP_CONNECTORS]),
        importers=dict(eps[GROUP_IMPORTERS]),
        exporters=dict(eps[GROUP_EXPORTERS]),
        location=str(package_dir(module) or ""),
        first_party=name.startswith(FIRST_PARTY_PREFIXES),
    )
    try:
        m = load_from_package(module)
    except ManifestError as e:
        rec.status = "invalid"
        rec.problems = list(e.problems)
        return rec
    rec.manifest = m
    rec.kind = m.kind
    problems: list[str] = []
    if m.id != pid:
        problems.append(f"plugin.toml id {m.id!r} differs from the entry-point name {pid!r}")
    if m.version != version:
        problems.append(
            f"plugin.toml version {m.version} differs from the distribution version {version}"
        )
    if m.connectors != rec.connectors:
        problems.append(
            "[connectors] of plugin.toml and the frontier.connectors entry points differ"
        )
    if m.importers != rec.importers:
        problems.append("[importers] of plugin.toml and the frontier.importers entry points differ")
    if m.exporter is not None and norm_vendor(m.exporter.vendor) not in {
        norm_vendor(v) for v in rec.exporters
    }:
        problems.append("[exporter] vendor has no frontier.exporters entry point")
    if problems:
        rec.status = "invalid"
        rec.problems = problems
        return rec
    reason = data_model_compatible(m.data_model_version, DATA_MODEL_VERSION)
    if reason:
        rec.status = "incompatible"
        rec.problems = [reason]
    return rec


def discover(state: state_mod.PluginState | None = None) -> list[PluginRecord]:
    """Every installed plugin, first-party ones first, each with its status."""
    st = state if state is not None else state_mod.load()
    records: dict[str, PluginRecord] = {}
    seen: set[str] = set()
    try:
        dists = list(importlib_metadata.distributions())
    except Exception as e:  # noqa: BLE001
        log.warning("could not enumerate installed distributions: %r", e)
        dists = []
    for dist in dists:
        try:
            raw_name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: BLE001 - a distribution without usable metadata
            continue
        if not raw_name:
            continue
        name = norm_dist(raw_name)
        if name in seen:  # the first one on sys.path is the one that is imported
            continue
        seen.add(name)
        eps = _entry_points(dist)
        if eps[GROUP_PLUGINS]:
            for pid, module in eps[GROUP_PLUGINS].items():
                records[pid] = _record_with_manifest(pid, module, name, version, eps)
        elif name != FRAMEWORK_DIST and (
            eps[GROUP_CONNECTORS] or eps[GROUP_IMPORTERS] or eps[GROUP_EXPORTERS]
        ):
            kind = (
                "connector"
                if eps[GROUP_CONNECTORS]
                else ("importer" if eps[GROUP_IMPORTERS] else "exporter")
            )
            records[name] = PluginRecord(
                id=name,
                dist=name,
                dist_version=version,
                kind=kind,
                connectors=dict(eps[GROUP_CONNECTORS]),
                importers=dict(eps[GROUP_IMPORTERS]),
                exporters=dict(eps[GROUP_EXPORTERS]),
                first_party=False,
                problems=["no plugin.toml (declares entry points only)"],
            )
    for rec in records.values():
        if rec.status == "enabled" and st.is_disabled("plugins", rec.id):
            rec.status = "disabled"
    return sorted(records.values(), key=lambda r: (not r.first_party, r.id))


def find(plugin_id: str, records: list[PluginRecord] | None = None) -> PluginRecord | None:
    """Installed plugin by id, distribution name, or the id of something it provides."""
    recs = records if records is not None else discover()
    wanted = plugin_id.strip()
    for rec in recs:
        if rec.id == wanted or rec.dist == norm_dist(wanted):
            return rec
    return None


def load_target(target: str) -> Any:
    """Import ``module:attribute`` and instantiate it when it is a class."""
    module, _, attr = target.partition(":")
    obj = getattr(importlib.import_module(module), attr)
    return obj() if inspect.isclass(obj) else obj


def exporter_files(rec: PluginRecord, vendor: str) -> Path | None:
    """Directory holding the scripts of ``vendor`` inside plugin ``rec``."""
    module = next((m for v, m in rec.exporters.items() if norm_vendor(v) == vendor), None)
    if module is None:
        return None
    base = package_dir(module.partition(":")[0])
    if base is None:
        return None
    sub = rec.manifest.exporter.files if rec.manifest and rec.manifest.exporter else "files"
    return base / sub


@lru_cache(maxsize=1)
def catalog() -> list[dict[str, Any]]:
    """The first-party plugins this framework release knows about (installed or not)."""
    try:
        text = (resources.files("frontier_connectors.plugins") / "catalog.json").read_text("utf-8")
        doc = json.loads(text)
    except (OSError, ValueError) as e:
        log.debug("plugin catalog unreadable: %r", e)
        return []
    return list(doc.get("plugins", []))


def catalog_provider(kind: str, name: str) -> dict[str, Any] | None:
    """Catalog entry of the plugin providing connector / importer / exporter ``name``."""
    key = {"connector": "connectors", "importer": "importers", "exporter": "exporters"}[kind]
    wanted = norm_vendor(name) if kind == "exporter" else name
    for entry in catalog():
        if wanted in entry.get(key, []):
            return entry
    return None


def missing_hint(kind: str, name: str) -> str | None:
    """What to tell a user who asked for something that is not loaded."""
    records = discover()
    key = {"connector": "connectors", "importer": "importers", "exporter": "exporters"}[kind]
    wanted = norm_vendor(name) if kind == "exporter" else name
    for rec in records:
        provided = (
            {norm_vendor(v) for v in rec.exporters} if kind == "exporter" else getattr(rec, key)
        )
        if wanted not in provided:
            continue
        if rec.status == "disabled":
            return (
                f"plugin {rec.id!r} is disabled: "
                f"python -m frontier_connectors plugins enable {rec.id}"
            )
        if rec.status in ("invalid", "incompatible"):
            return f"plugin {rec.id!r} is {rec.status}: " + "; ".join(rec.problems)
        if kind != "exporter" and state_mod.load().is_disabled(key, name):
            return (
                f"{kind} {name!r} is disabled: "
                f"python -m frontier_connectors plugins enable {kind}:{name}"
            )
    entry = catalog_provider(kind, name)
    if entry is not None:
        return (
            f"{kind} {name!r} is provided by the plugin {entry['id']!r}, which is not installed: "
            f"python -m frontier_connectors plugins install {entry['id']}"
        )
    return None
