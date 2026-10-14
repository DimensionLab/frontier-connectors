# SPDX-License-Identifier: Apache-2.0
"""``plugins doctor``: why a plugin is not working, and what to install to fix it."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from importlib import metadata as importlib_metadata
from typing import Any

from frontier_connectors.errors import ConnectorError
from frontier_connectors.plugins import auth as auth_mod
from frontier_connectors.plugins import discovery, exporters, installer
from frontier_connectors.plugins import state as state_mod
from frontier_connectors.plugins.manifest import DATA_MODEL_VERSION

_REQ_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def requirement_name(requirement: str) -> str:
    m = _REQ_NAME.match(requirement)
    return m.group(1) if m else requirement


def installed_version(package: str, module: str = "") -> str | None:
    """Version of an installed distribution, ``"installed"`` when only the module is
    importable, ``None`` when it is absent. Nothing is imported."""
    try:
        return importlib_metadata.version(requirement_name(package))
    except importlib_metadata.PackageNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - damaged metadata
        return None
    name = module or requirement_name(package).replace("-", "_")
    try:
        return "installed" if importlib.util.find_spec(name) is not None else None
    except (ImportError, ValueError, AttributeError):
        return None


def check_plugin(
    rec: discovery.PluginRecord, st: state_mod.PluginState, load: bool
) -> dict[str, Any]:
    row = rec.to_dict()
    m = rec.manifest
    by_id = {r.id: r for r in discovery.discover(st)}
    requires = []
    for req in m.requires if m else []:
        version = installed_version(req)
        requires.append({"requirement": req, "installed": version})
    needs_plugins = []
    for pid in m.plugins if m else []:
        other = by_id.get(pid)
        needs_plugins.append(
            {
                "plugin": pid,
                "status": other.status if other else "not-installed",
                "fix": None
                if other and other.enabled
                else (
                    f"python -m frontier_connectors plugins enable {pid}"
                    if other and other.status == "disabled"
                    else f"python -m frontier_connectors plugins install {pid}"
                ),
            }
        )
    optional = []
    for opt in m.optional if m else []:
        version = installed_version(opt.package, opt.module)
        optional.append(
            {
                "package": opt.package,
                "installed": version,
                "licence": opt.licence,
                "purpose": opt.purpose,
                "flag": opt.flag,
                "fix": None if version else f"pip install {opt.package}",
            }
        )
    loads: list[dict[str, Any]] = []
    if load and rec.enabled:
        for kind, table in (("connector", rec.connectors), ("importer", rec.importers)):
            for name, target in sorted(table.items()):
                scope = kind + "s"
                if st.is_disabled(scope, name):
                    loads.append({"kind": kind, "id": name, "ok": True, "disabled": True})
                    continue
                try:
                    discovery.load_target(target)
                    loads.append({"kind": kind, "id": name, "ok": True})
                except Exception as e:  # noqa: BLE001 - report, never raise
                    loads.append(
                        {
                            "kind": kind,
                            "id": name,
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
        for vendor in rec.exporters:
            path = discovery.exporter_files(rec, discovery.norm_vendor(vendor))
            ok = path is not None and path.is_dir()
            loads.append(
                {"kind": "exporter", "id": vendor, "ok": ok}
                | ({} if ok else {"error": "script directory is missing"})
            )
    problems = list(rec.problems) if rec.status in ("invalid", "incompatible") else []
    problems += [f"missing requirement {r['requirement']}" for r in requires if not r["installed"]]
    problems += [
        f"needs plugin {p['plugin']} ({p['status']})" for p in needs_plugins if p["fix"] is not None
    ]
    problems += [f"{x['kind']} {x['id']}: {x['error']}" for x in loads if not x["ok"]]
    row.update(
        {
            "auth_status": auth_mod.plugin_status(rec, keychain=False),
            "requires_status": requires,
            "plugins_status": needs_plugins,
            "optional_status": optional,
            "loads": loads,
            "problems": problems,
            "notes": list(rec.problems) if rec.status not in ("invalid", "incompatible") else [],
            "ok": not problems,
        }
    )
    return row


def doctor(load: bool = True) -> dict[str, Any]:
    st = state_mod.load()
    records = discovery.discover(st)
    rows = [check_plugin(r, st, load) for r in records]
    try:
        cmd: list[str] | None = installer.installer_command()
        installer_error = ""
    except ConnectorError as e:
        cmd, installer_error = None, str(e)
    try:
        framework = importlib_metadata.version(discovery.FRAMEWORK_DIST)
    except importlib_metadata.PackageNotFoundError:
        framework = "unknown"
    stale = exporters.stale_copies()
    return {
        "framework": {
            "dist": discovery.FRAMEWORK_DIST,
            "version": framework,
            "data_model_version": DATA_MODEL_VERSION,
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "state": st.to_dict(),
        "installer": {"command": cmd, "error": installer_error},
        "plugins": rows,
        "deployed_exporters": stale,
        "ok": all(r["ok"] for r in rows if r["status"] != "disabled") and not st.error,
        "frontier_home": str(state_mod.frontier_home()),
        "env": {
            k: os.environ[k]
            for k in (state_mod.HOME_ENV, installer.INSTALLER_ENV)
            if k in os.environ
        },
    }


def doctor_text(report: dict[str, Any]) -> str:
    fw = report["framework"]
    lines = [
        f"frontier-connectors {fw['version']} (frontier-dataset v{fw['data_model_version']}), "
        f"python {fw['python']}",
        f"state: {report['state']['path']}"
        + (f"  ERROR: {report['state']['error']}" if report["state"]["error"] else ""),
        "installer: "
        + (" ".join(report["installer"]["command"]) if report["installer"]["command"] else "none")
        + (f"  ({report['installer']['error']})" if report["installer"]["error"] else ""),
        "",
    ]
    for row in report["plugins"]:
        mark = "ok" if row["ok"] else "PROBLEM"
        if row["status"] == "disabled":
            mark = "disabled"
        lines.append(f"{row['id']:<22} {row['version']:<8} {row['status']:<12} {mark}")
        for p in row["problems"]:
            lines.append(f"    problem: {p}")
        for p in row["plugins_status"]:
            if p["fix"]:
                lines.append(f"    fix: {p['fix']}")
        for r in row["requires_status"]:
            if not r["installed"]:
                lines.append(f"    fix: pip install '{r['requirement']}'")
        for o in row["optional_status"]:
            if o["installed"]:
                continue
            flag = f"; {o['flag']}" if o["flag"] else ""
            lines.append(
                f"    optional, not installed: {o['package']} ({o['licence']}{flag}), "
                f"used to {o['purpose']}: {o['fix']}"
            )
        for a in row.get("auth_status", []):
            if a.get("configured") is False:
                lines.append(
                    f"    credential {a['id']} is not configured "
                    f"(needed for live access only): {a['login']}"
                )
        for n in row["notes"]:
            lines.append(f"    note: {n}")
    for d in report["deployed_exporters"]:
        lines.append(f"exporter copy {d['path']}: {d['problem']}")
    lines.append("")
    lines.append("all plugins healthy" if report["ok"] else "problems found (see above)")
    return "\n".join(lines)
