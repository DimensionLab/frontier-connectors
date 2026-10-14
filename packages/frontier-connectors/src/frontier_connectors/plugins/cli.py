# SPDX-License-Identifier: Apache-2.0
"""``python -m frontier_connectors plugins ...`` (see :mod:`frontier_connectors.plugins`)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from frontier_connectors.errors import ConnectorError
from frontier_connectors.plugins import auth as auth_mod
from frontier_connectors.plugins import discovery, exporters, installer, manage
from frontier_connectors.plugins import health as doctor_mod
from frontier_connectors.plugins import state as state_mod
from frontier_connectors.plugins.manifest import DATA_MODEL_VERSION

Emit = Callable[[argparse.Namespace, Any, str], None]


def plugin_rows(include_available: bool = False, kind: str | None = None) -> list[dict[str, Any]]:
    """Installed plugins (and, on request, catalog plugins that are not installed)."""
    st = state_mod.load()
    rows = []
    for rec in discovery.discover(st):
        row = rec.to_dict()
        # a listing never touches the OS keychain: status only where it is known without it
        row["auth_status"] = auth_mod.plugin_status(rec, keychain=False)
        row["disabled_connectors"] = [c for c in rec.connectors if st.is_disabled("connectors", c)]
        row["disabled_importers"] = [c for c in rec.importers if st.is_disabled("importers", c)]
        row["disabled_exporters"] = [
            discovery.norm_vendor(v)
            for v in rec.exporters
            if st.is_disabled("exporters", discovery.norm_vendor(v))
        ]
        rows.append(row)
    if include_available:
        have = {r["id"] for r in rows}
        for entry in discovery.catalog():
            if entry["id"] in have:
                continue
            rows.append(
                dict(entry)
                | {
                    "status": "not-installed",
                    "enabled": False,
                    "installed": False,
                    "problems": [],
                    "module": "",
                    "location": "",
                    "first_party": True,
                    "has_manifest": True,
                    "auth_status": [
                        auth_mod.status(a, keychain=False) for a in entry.get("auth", [])
                    ],
                    "disabled_connectors": [],
                    "disabled_importers": [],
                    "disabled_exporters": [],
                }
            )
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return rows


def _provides(row: dict[str, Any]) -> str:
    parts = []
    if row.get("connectors"):
        parts.append("connectors: " + ", ".join(row["connectors"]))
    if row.get("importers"):
        parts.append("importers: " + ", ".join(row["importers"]))
    if row.get("exporters"):
        parts.append("exporter scripts: " + ", ".join(row["exporters"]))
    return "; ".join(parts)


def _auth_word(a: dict[str, Any]) -> str:
    if "configured" not in a:
        return f"see `plugins auth` ({a['login']})"
    if a["configured"]:
        label = auth_mod.source_label(a.get("source"))
        return f"configured ({label})" if label else "configured"
    return f"missing: {a['login']}"


def _auth_table(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'credential':<16} {'kind':<10} {'status':<28} declared by"]
    for a in rows:
        lines.append(
            f"{a['id']:<16} {a['kind']:<10} {_auth_word(a):<28} {', '.join(a['declared_by'])}"
        )
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'id':<24} {'kind':<10} {'version':<8} {'status':<14} provides"]
    for r in rows:
        lines.append(
            f"{r['id']:<24} {r['kind']:<10} {r.get('version', ''):<8} {r['status']:<14} "
            f"{_provides(r) or r.get('name', '')}"
        )
        off = (
            r.get("disabled_connectors", [])
            + r.get("disabled_importers", [])
            + r.get("disabled_exporters", [])
        )
        if off:
            lines.append(f"{'':<24} disabled inside: {', '.join(off)}")
        for a in r.get("auth_status", []):
            lines.append(f"{'':<24} credential {a['id']}: {_auth_word(a)}")
        if r["status"] in ("invalid", "incompatible"):
            lines += [f"{'':<24} {p}" for p in r.get("problems", [])]
    return "\n".join(lines)


def make_commands(emit: Emit) -> dict[str, Callable[[argparse.Namespace], int]]:
    def cmd_list(args: argparse.Namespace) -> int:
        rows = plugin_rows(include_available=args.available, kind=args.kind)
        payload = {
            "data_model_version": DATA_MODEL_VERSION,
            "state_file": str(state_mod.state_path()),
            "plugins": rows,
        }
        emit(args, payload, _table(rows) if rows else "(no plugins installed)")
        return 0

    def cmd_search(args: argparse.Namespace) -> int:
        rows = installer.search(
            " ".join(args.query), find_links=args.find_links or [], catalog=args.catalog
        )
        lines = [f"{'id':<24} {'kind':<10} {'version':<8} {'status':<14} summary"]
        for r in rows:
            lines.append(
                f"{r['id']:<24} {r['kind']:<10} {r.get('version', ''):<8} {r['status']:<14} "
                f"{r.get('summary') or r.get('name', '')}"
            )
        emit(args, {"query": " ".join(args.query), "plugins": rows}, "\n".join(lines))
        return 0

    def cmd_info(args: argparse.Namespace) -> int:
        rec = discovery.find(args.id)
        if rec is None:
            rows = installer.available(args.find_links or [], args.catalog)
            if args.id not in rows:
                raise ConnectorError(
                    f"no plugin {args.id!r}",
                    hint="see `python -m frontier_connectors plugins search`",
                )
            row = dict(rows[args.id]) | {"installed": False, "status": "not-installed"}
        else:
            row = doctor_mod.check_plugin(rec, state_mod.load(), load=True)
        lines = [
            f"{row['id']}  ({row['kind']} plugin, {row.get('dist', '')} {row.get('version', '')})",
            f"  name:      {row.get('name', '')}",
            f"  status:    {row['status']}",
            f"  vendor:    {row.get('vendor', '')}",
            f"  maturity:  {row.get('maturity', '')}",
            f"  licence:   {row.get('licence', '')}",
            f"  data model: frontier-dataset v{row.get('data_model_version', '')}",
            f"  formats:   {', '.join(row.get('formats', []))}",
            f"  provides:  {_provides(row)}",
        ]
        if row.get("summary"):
            lines.insert(2, f"  summary:   {row['summary']}")
        if row.get("requires"):
            lines.append(f"  requires:  {', '.join(row['requires'])}")
        if row.get("plugins"):
            lines.append(f"  plugins:   {', '.join(row['plugins'])}")
        for o in row.get("optional", []):
            flag = f"; {o.get('flag')}" if o.get("flag") else ""
            lines.append(
                f"  optional:  {o['package']} ({o.get('licence', '')}{flag}): "
                f"{o.get('purpose', '')}"
            )
        auth_rows = row.get("auth_status")
        if auth_rows is None:
            auth_rows = [auth_mod.status(a) for a in row.get("auth", [])]
            row["auth_status"] = auth_rows
        for a in auth_rows:
            lines.append(f"  credential: {a['id']} ({a['kind']}): {_auth_word(a)}")
        for p in row.get("problems", []):
            lines.append(f"  problem:   {p}")
        if row.get("location"):
            lines.append(f"  location:  {row['location']}")
        emit(args, row, "\n".join(lines))
        return 0

    def cmd_install(args: argparse.Namespace) -> int:
        result = installer.install(
            args.spec,
            find_links=args.find_links or [],
            index_url=args.index_url,
            extra_index_url=args.extra_index_url,
            no_index=args.no_index,
            with_suggested=not args.no_suggested,
            upgrade=args.upgrade,
            editable=args.editable,
            catalog=args.catalog,
            dry_run=args.dry_run,
        )
        lines = [("would run: " if args.dry_run else "ran: ") + " ".join(result["command"])]
        if result.get("policy_excluded"):
            lines.append(
                "  not used (policy.plugin_sources): " + ", ".join(result["policy_excluded"])
            )
        for p in result["plugins"]:
            lines.append(f"  {p['id']} {p['version']}: {p['status']}  {_provides(p)}")
            lines += [f"    {x}" for x in p["problems"] if p["status"] != "enabled"]
        bad = [p for p in result["plugins"] if p["status"] in ("invalid", "incompatible")]
        emit(args, result, "\n".join(lines))
        return 1 if bad else 0

    def cmd_uninstall(args: argparse.Namespace) -> int:
        result = installer.uninstall(args.id, purge=args.purge, dry_run=args.dry_run)
        text = ("would run: " if args.dry_run else "ran: ") + " ".join(result["command"])
        emit(args, result, text + "\nuninstalled: " + ", ".join(result["uninstalled"]))
        return 0

    def cmd_enable(args: argparse.Namespace) -> int:
        results = [manage.set_enabled(name, True) for name in args.id]
        emit(
            args,
            {"results": results},
            "\n".join(f"enabled {r['scope']}:{r['id']}" for r in results),
        )
        return 0

    def cmd_disable(args: argparse.Namespace) -> int:
        results = [manage.set_enabled(name, False) for name in args.id]
        text = "\n".join(
            f"disabled {r['scope']}:{r['id']} (still installed; state in {r['state']})"
            for r in results
        )
        emit(args, {"results": results}, text)
        return 0

    def cmd_doctor(args: argparse.Namespace) -> int:
        report = doctor_mod.doctor(load=not args.no_load)
        emit(args, report, doctor_mod.doctor_text(report))
        return 0 if report["ok"] else 1

    def cmd_auth(args: argparse.Namespace) -> int:
        written = auth_mod.write_providers()
        rows = auth_mod.statuses()
        payload = {
            "path": written["path"],
            "written": written["written"],
            "error": written["error"],
            "sdk": auth_mod.credentials.sdk("auth") is not None,
            "providers": written["providers"],
            "status": rows,
        }
        text = _auth_table(rows) if rows else "(no installed plugin declares a credential)"
        text += f"\ndeclarations: {written['path']}"
        if written["error"]:
            text += f"\n  warning: {written['error']}"
        emit(args, payload, text)
        return 0

    def cmd_exporters(args: argparse.Namespace) -> int:
        rows = exporters.list_exporters()
        lines = []
        for r in rows:
            off = "" if r["enabled"] else "  (disabled)"
            lines.append(f"{r['vendor']:<18} {r['version']:<8} {r['tool']}{off}")
            lines += [f"    {f}" for f in r["files"]]
        emit(args, {"exporters": rows}, "\n".join(lines) or "(no exporter plugins installed)")
        return 0

    def cmd_install_exporter(args: argparse.Namespace) -> int:
        result = exporters.install_exporter(args.vendor, args.to, force=args.force)
        lines = [
            f"copied {len(result['files'])} file(s) of exporter {result['vendor']} "
            f"{result['version']} to {result['path']}"
        ]
        if result["replaced_version"] and result["replaced_version"] != result["version"]:
            lines.append(f"  replaced version {result['replaced_version']}")
        if result["usage"]:
            lines.append(f"  run: {result['usage']}")
        emit(args, result, "\n".join(lines))
        return 0

    def cmd_uninstall_exporter(args: argparse.Namespace) -> int:
        result = exporters.uninstall_exporter(args.vendor, args.from_dir, force=args.force)
        lines = [f"removed {len(result['removed'])} file(s) of exporter {result['vendor']}"]
        lines += [f"  kept (modified since install): {f}" for f in result["kept_modified"]]
        emit(args, result, "\n".join(lines))
        return 0

    return {
        "list": cmd_list,
        "search": cmd_search,
        "info": cmd_info,
        "install": cmd_install,
        "uninstall": cmd_uninstall,
        "enable": cmd_enable,
        "disable": cmd_disable,
        "doctor": cmd_doctor,
        "auth": cmd_auth,
        "exporters": cmd_exporters,
        "install-exporter": cmd_install_exporter,
        "uninstall-exporter": cmd_uninstall_exporter,
    }


def add_parser(sub: Any, common: argparse.ArgumentParser, emit: Emit) -> None:
    """Attach the ``plugins`` command group to the main parser."""
    commands = make_commands(emit)
    top = sub.add_parser(
        "plugins",
        parents=[common],
        help="Manage connector, exporter and importer plugins",
    )
    psub = top.add_subparsers(dest="plugins_command", required=True)

    def sources(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--find-links",
            action="append",
            metavar="DIR",
            help="Look for wheels in this directory; repeatable",
        )
        p.add_argument("--catalog", metavar="URL|FILE", help="Read this plugins-index.json")

    p = psub.add_parser("list", parents=[common], help="List installed plugins and their state")
    p.add_argument(
        "--available", action="store_true", help="Include plugins that are not installed"
    )
    p.add_argument(
        "--kind", choices=["connector", "exporter", "importer"], help="Show only this kind"
    )
    p.set_defaults(func=commands["list"])

    p = psub.add_parser("search", parents=[common], help="Search known plugins")
    p.add_argument("query", nargs="*", help="Words to match")
    sources(p)
    p.set_defaults(func=commands["search"])

    p = psub.add_parser("info", parents=[common], help="Show the manifest and health of one plugin")
    p.add_argument("id", help="Plugin id")
    sources(p)
    p.set_defaults(func=commands["info"])

    p = psub.add_parser(
        "install",
        parents=[common],
        help="Install plugins by id, or from a path, a wheel or a git URL",
    )
    p.add_argument("spec", nargs="+", help="Plugin id, distribution, path, wheel or git+URL")
    sources(p)
    p.add_argument(
        "--index-url",
        help="Use this package index instead of PyPI (default: plugins.index_url)",
    )
    p.add_argument("--extra-index-url", help="Also search this package index")
    p.add_argument("--no-index", action="store_true", help="Use only --find-links directories")
    p.add_argument("--no-suggested", action="store_true", help="Skip suggested exporter plugins")
    p.add_argument("--upgrade", action="store_true", help="Upgrade installed plugins")
    p.add_argument(
        "-e", "--editable", action="store_true", help="Install a local path in editable mode"
    )
    p.add_argument("--dry-run", action="store_true", help="Print the installer command and exit")
    p.set_defaults(func=commands["install"])

    p = psub.add_parser("uninstall", parents=[common], help="Uninstall plugins")
    p.add_argument("id", nargs="+", help="Plugin id")
    p.add_argument(
        "--purge", action="store_true", help="Also forget their enabled or disabled state"
    )
    p.add_argument("--dry-run", action="store_true", help="Print the uninstall command and exit")
    p.set_defaults(func=commands["uninstall"])

    scope_help = "Plugin id, or connector:<id>, importer:<id> or exporter:<vendor>"
    p = psub.add_parser("enable", parents=[common], help="Enable a disabled plugin")
    p.add_argument("id", nargs="+", help=scope_help)
    p.set_defaults(func=commands["enable"])

    p = psub.add_parser("disable", parents=[common], help="Disable a plugin and keep it installed")
    p.add_argument("id", nargs="+", help=scope_help)
    p.set_defaults(func=commands["disable"])

    p = psub.add_parser(
        "doctor", parents=[common], help="Report what is broken or missing, with the fix"
    )
    p.add_argument("--no-load", action="store_true", help="Check without importing plugin code")
    p.set_defaults(func=commands["doctor"])

    p = psub.add_parser(
        "auth",
        parents=[common],
        help="List the credentials enabled plugins need and whether they are set",
    )
    p.set_defaults(func=commands["auth"])

    p = psub.add_parser(
        "exporters", parents=[common], help="List the installed exporter script bundles"
    )
    p.set_defaults(func=commands["exporters"])

    p = psub.add_parser(
        "install-exporter",
        parents=[common],
        help="Copy an exporter script bundle to a directory",
    )
    p.add_argument("vendor", help="Vendor key; see `plugins exporters`")
    p.add_argument("--to", required=True, metavar="DIR", help="Copy the scripts to DIR/<vendor>/")
    p.add_argument("--force", action="store_true", help="Overwrite an existing copy")
    p.set_defaults(func=commands["install-exporter"])

    p = psub.add_parser(
        "uninstall-exporter", parents=[common], help="Remove copies made by install-exporter"
    )
    p.add_argument("vendor", help="Vendor key; see `plugins exporters`")
    p.add_argument(
        "--from",
        dest="from_dir",
        metavar="DIR",
        help="Remove from this directory (default: every recorded copy)",
    )
    p.add_argument("--force", action="store_true", help="Also remove files edited since install")
    p.set_defaults(func=commands["uninstall-exporter"])
