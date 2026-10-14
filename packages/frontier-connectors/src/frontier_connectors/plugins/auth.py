# SPDX-License-Identifier: Apache-2.0
"""The credentials plugins declare (``[[auth]]`` in ``plugin.toml``) and whether they are
configured.

:func:`declarations` merges the declarations of the enabled plugins by provider id (several
plugins may share one credential; each is listed in ``declared_by``). :func:`write_providers`
records them in ``$FRONTIER_HOME/auth-providers.json``, a list of provider tables:

.. code-block:: json

    [{"id": "flow360", "title": "Flexcompute Flow360", "kind": "api_key",
      "fields": [{"name": "api_key", "title": "api key", "secret": true, "required": true,
                  "env": ["FLOW360_APIKEY"]}],
      "env": ["FLOW360_APIKEY"], "help_url": "https://...", "scopes_hint": "...",
      "declared_by": ["flow360"]}]

That file is how ``frontier auth``, the control plane and the app learn about plugin
credentials without importing Python. It is rewritten whenever plugins are installed,
uninstalled, enabled or disabled, and by ``python -m frontier_connectors plugins auth``.

:func:`status` says, per declaration, whether the credential is configured and where it comes
from. It never returns a value.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from frontier_connectors import credentials
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as state_mod

PROVIDERS_FILE = "auth-providers.json"


def providers_path() -> Path:
    """``$FRONTIER_HOME/auth-providers.json`` (the SDK's home when it is installed, so both
    sides agree on the location)."""
    cfg = credentials.config()
    home = Path(cfg.paths.home) if cfg is not None else state_mod.frontier_home()
    return home / PROVIDERS_FILE


def declarations(
    records: list[discovery.PluginRecord] | None = None,
) -> list[dict[str, Any]]:
    """Provider tables of every enabled plugin, merged by id, in plugin order. A provider
    declared by several plugins keeps the first definition and gains the other declarers,
    their extra environment names and any field it did not have."""
    merged: dict[str, dict[str, Any]] = {}
    for rec in records if records is not None else discovery.discover():
        if not rec.enabled or rec.manifest is None:
            continue
        for decl in rec.manifest.auth:
            p = decl.to_provider()
            have = merged.get(p["id"])
            if have is None:
                merged[p["id"]] = p | {"declared_by": [rec.id]}
                continue
            if rec.id not in have["declared_by"]:
                have["declared_by"].append(rec.id)
            have["env"] += [e for e in p["env"] if e not in have["env"]]
            for f in p["fields"]:
                mine = next((x for x in have["fields"] if x["name"] == f["name"]), None)
                if mine is None:
                    have["fields"].append(f)
                else:
                    mine["env"] += [e for e in f["env"] if e not in mine["env"]]
            for key in ("help_url", "test", "scopes_hint", "oauth"):
                if key not in have and key in p:
                    have[key] = p[key]
    return list(merged.values())


def write_providers(
    records: list[discovery.PluginRecord] | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Record :func:`declarations` in ``auth-providers.json`` (atomically). Returns
    ``{"path", "providers", "written", "error"}``; a home that cannot be written is reported,
    never raised: managing plugins must not fail because of it."""
    target = Path(path) if path is not None else providers_path()
    providers = declarations(records)
    result: dict[str, Any] = {
        "path": str(target),
        "providers": providers,
        "written": False,
        "error": "",
    }
    text = json.dumps(providers, indent=2, ensure_ascii=False) + "\n"
    try:
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            return result
        if not providers and not target.exists():
            return result  # nothing declared and nothing to correct: leave the home alone
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".auth-providers-", suffix=".json", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        result["written"] = True
    except OSError as e:
        result["error"] = f"{target} could not be written: {e.strerror or e}"
    return result


def refresh() -> dict[str, Any]:
    """:func:`write_providers` after a change to the installed or enabled plugins."""
    try:
        return write_providers()
    except Exception as e:  # noqa: BLE001 - bookkeeping must never fail the operation
        return {"path": "", "providers": [], "written": False, "error": f"{type(e).__name__}"}


def status(provider: dict[str, Any], *, keychain: bool = True) -> dict[str, Any]:
    """Whether one declared credential is configured: ``{"id", "title", "kind", "login",
    "configured", "source", "missing"}``. ``missing`` names the required fields without a
    value; ``source`` says where the primary value comes from, as in the credentials API:
    ``None`` or ``{"store": "keychain" | "file" | "env" | "job", "detail": <variable>}``.
    No value is ever returned.

    ``keychain=False`` is for listings: the OS keychain is never touched (no prompt) and
    nothing goes over the network. When the answer would need the keychain, ``configured``,
    ``source`` and ``missing`` are left out."""
    pid = provider["id"]
    out: dict[str, Any] = {
        "id": pid,
        "title": provider.get("title", pid),
        "kind": provider.get("kind", "token"),
        "login": f"frontier auth login {pid}",
    }
    auth = credentials.sdk("auth")
    if auth is not None:
        try:
            st = auth.status(pid) if keychain else auth.quiet_status(pid)
            known = True
        except Exception:  # noqa: BLE001 - unknown to the SDK yet (file not written): below
            st, known = None, False
        if known:
            if st is not None:
                out["configured"] = bool(st["configured"])
                out["source"] = st["source"]
                out["missing"] = [
                    f["name"] for f in st["fields"] if f["required"] and not f["configured"]
                ]
            return out
    fields = provider.get("fields", [])
    missing: list[str] = []
    source: str | None = None
    for f in fields:
        names = f.get("env", [])
        var = next((e for e in names if os.environ.get(e, "").strip()), None)
        if var is None and keychain:
            var = next((e for e in names if credentials.secret(e)), None)
            if var is not None and source is None and f.get("secret", True):
                source = "keychain:legacy"
        elif var is not None and source is None and f.get("secret", True):
            source = f"env:{var}"
        if var is None and f.get("required", True):
            missing.append(f["name"])
    if missing and not keychain and credentials.keyring_installed():
        return out  # a keyring entry may hold it: not looked at, so nothing is claimed
    return out | {
        "configured": bool(fields) and not missing,
        "source": _source_json(source),
        "missing": missing,
    }


def _source_json(source: str | None) -> dict[str, Any] | None:
    if source is None:
        return None
    store, _, detail = source.partition(":")
    return {"store": store, "detail": detail or None}


def source_label(source: Any) -> str:
    """``env HF_TOKEN`` / ``keychain`` for a status row's ``source``; ``""`` for none."""
    if not source:
        return ""
    if isinstance(source, str):
        return source.replace(":", " ", 1)
    return " ".join(str(v) for v in (source.get("store"), source.get("detail")) if v)


def statuses(
    records: list[discovery.PluginRecord] | None = None, *, keychain: bool = True
) -> list[dict[str, Any]]:
    """:func:`status` of every declared provider, each with ``declared_by``."""
    return [
        status(p, keychain=keychain) | {"declared_by": p["declared_by"]}
        for p in declarations(records)
    ]


def plugin_status(rec: discovery.PluginRecord, *, keychain: bool = True) -> list[dict[str, Any]]:
    """:func:`status` of the credentials one plugin declares."""
    if rec.manifest is None:
        return []
    return [status(a.to_provider(), keychain=keychain) for a in rec.manifest.auth]
