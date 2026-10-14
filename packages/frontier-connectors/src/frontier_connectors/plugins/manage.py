# SPDX-License-Identifier: Apache-2.0
"""Enable and disable plugins (or single connectors / importers / exporters) without
uninstalling anything."""

from __future__ import annotations

from typing import Any

from frontier_connectors.errors import ConnectorError
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as state_mod

_SCOPE_OF = {"connector": "connectors", "importer": "importers", "exporter": "exporters"}


def resolve_target(name: str) -> tuple[str, str]:
    """``(scope, id)`` for ``fluent`` / ``connector:fluent-live`` / ``exporter:matlab``.

    A bare name is a plugin id when such a plugin is installed, otherwise the single connector,
    importer or exporter of that name."""
    records = discovery.discover()
    if ":" in name:
        kind, _, ident = name.partition(":")
        if kind == "plugin":
            scope = "plugins"
        elif kind in _SCOPE_OF:
            scope = _SCOPE_OF[kind]
        else:
            raise ConnectorError(
                f"unknown scope {kind!r}", hint="use plugin:, connector:, importer: or exporter:"
            )
        if scope == "exporters":
            ident = discovery.norm_vendor(ident)
        known: set[str] = set()
        for r in records:
            if scope == "plugins":
                known.add(r.id)
            elif scope == "exporters":
                known |= {discovery.norm_vendor(v) for v in r.exporters}
            else:
                known |= set(getattr(r, scope))
        if ident not in known:
            raise ConnectorError(
                f"no installed {kind} {ident!r}",
                hint=discovery.missing_hint(kind, ident) if kind in _SCOPE_OF else None,
            )
        return scope, ident
    rec = discovery.find(name, records)
    if rec is not None:
        return "plugins", rec.id
    for r in records:
        if name in r.connectors:
            return "connectors", name
        if name in r.importers:
            return "importers", name
        if discovery.norm_vendor(name) in {discovery.norm_vendor(v) for v in r.exporters}:
            return "exporters", discovery.norm_vendor(name)
    raise ConnectorError(
        f"nothing installed is called {name!r}",
        hint="see `python -m frontier_connectors plugins list`",
    )


def set_enabled(name: str, enabled: bool) -> dict[str, Any]:
    """Switch a plugin (or one connector / importer / exporter of it) on or off."""
    from frontier_connectors.plugins.installer import check_change

    check_change("enable" if enabled else "disable", name)
    scope, ident = resolve_target(name)
    st = state_mod.load()
    if st.error:
        raise ConnectorError(st.error, hint=f"fix or delete {st.path}")
    changed = st.set_enabled(scope, ident, enabled)
    if changed:
        state_mod.save(st)
    from frontier_connectors import registry

    registry.reload()
    from frontier_connectors.plugins import auth as auth_mod

    auth_mod.refresh()  # the credentials of a disabled plugin are not offered
    return {
        "scope": scope,
        "id": ident,
        "enabled": enabled,
        "changed": changed,
        "state": str(st.path),
    }
