# SPDX-License-Identifier: Apache-2.0
"""``python -m frontier_connectors <command>``: list, detect, describe, convert, import,
importers, exporters, doctor and the ``plugins`` group.

Every command takes ``--json`` and then prints exactly one JSON document on stdout (errors as
``{"error": ..., "message": ..., "hint": ...}`` with exit code 2), which is what the Rust
``frontier-connectors`` crate parses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from frontier_connectors import registry
from frontier_connectors.errors import ConnectorError
from frontier_connectors.model import jsonable
from frontier_connectors.plugins import cli as plugins_cli
from frontier_connectors.select import parse_time_selection


def _csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [t.strip() for t in text.split(",") if t.strip()]
    return items or None


def _load_mapping(path: str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise ConnectorError(f"mapping file {p} not found", hint="pass --mapping <file.json>")
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ConnectorError(
                "YAML mapping needs PyYAML", hint="pip install pyyaml, or use a .json mapping"
            ) from e
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ConnectorError(f"mapping file {p} is not valid JSON: {e}") from e


def _options(args: argparse.Namespace) -> dict[str, Any] | None:
    opts: dict[str, Any] = {}
    for item in getattr(args, "option", None) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConnectorError(f"bad --option {item!r}", hint="use --option key=value")
        try:
            opts[key] = json.loads(value)
        except json.JSONDecodeError:
            opts[key] = value
    if getattr(args, "mapping", None):
        opts["mapping"] = _load_mapping(args.mapping)
    return opts or None


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if args.json:
        sys.stdout.write(json.dumps(jsonable(payload), indent=None if args.compact else 2) + "\n")
    else:
        sys.stdout.write(text.rstrip("\n") + "\n")


def _cmd_list(args: argparse.Namespace) -> int:
    infos = registry.list_connectors()
    if args.maturity:
        infos = [i for i in infos if i.maturity == args.maturity]
    if args.markdown:
        sys.stdout.write(registry.support_matrix_markdown(infos) + "\n")
        return 0
    lines = [f"{'id':<14} {'maturity':<28} {'status':<13} {'time':<9} formats"]
    for i in infos:
        temporal = "transient" if "transient" in i.temporal else "steady"
        lines.append(
            f"{i.id:<14} {i.maturity:<28} {i.status:<13} {temporal:<9} "
            f"{i.name}: {', '.join(i.extensions or i.formats)}"
        )
        if i.missing:
            what = "needs" if i.status == "missing-deps" else "optional, not installed"
            lines.append(f"{'':<14} {what}: {', '.join(i.missing)}")
        if i.status == "broken":
            lines.append(f"{'':<14} error: {i.note}")
    _emit(args, [i.to_dict() for i in infos], "\n".join(lines))
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    found = registry.detect(args.path)
    if not found:
        raise ConnectorError(
            f"no connector recognises {args.path}",
            hint="see `python -m frontier_connectors list`; generic containers (hdf5, netcdf, "
            "matlab, tabular) can be forced with --connector and a --mapping config",
        )
    lines = []
    for d in found:
        lines.append(f"{d.connector_id:<16} confidence={d.confidence:.2f}  {'; '.join(d.notes)}")
        if d.needs:
            lines.append(f"{'':<16} needs: {'; '.join(d.needs)}")
    _emit(args, [d.to_dict() for d in found], "\n".join(lines))
    return 0


def _describe_text(d: dict[str, Any]) -> str:
    t = d["time"]
    lines = [
        f"{d['path']}",
        f"  connector: {d['connector_id']}  format: {d['format']}  domain: {d['domain']}",
        f"  solver:    {d['solver']} {d.get('solver_version') or ''}".rstrip(),
        f"  temporal:  {d['temporal']}"
        + (
            f" ({t['n_steps']} steps, {t['values'][0]:g} .. {t['values'][-1]:g} {t['units']})"
            if t["n_steps"] > 1
            else ""
        ),
    ]
    if d["physics"]:
        lines.append("  physics:   " + json.dumps(d["physics"]))
    if d["units"]:
        lines.append("  units:     " + json.dumps(d["units"]))
    lines.append(f"  zones ({len(d['zones'])}):")
    for z in d["zones"][:60]:
        n_pts = "?" if z.get("n_points") is None else z["n_points"]
        n_cells = "?" if z.get("n_cells") is None else z["n_cells"]
        size = f"{n_pts} pts, {n_cells} cells"
        if z.get("shape"):
            size += f", shape {z['shape']}"
        bc = f" bc={z['bc_type']}" if z.get("bc_type") else ""
        default = "" if z.get("default", True) else " (not read by default)"
        lines.append(f"    {z['name']:<28} {z['kind']:<11} {z['role']:<9} {size}{bc}{default}")
    if len(d["zones"]) > 60:
        lines.append(f"    ... {len(d['zones']) - 60} more")
    lines.append(f"  fields ({len(d['fields'])}):")
    for f in d["fields"][:80]:
        units = f" [{f['units']}]" if f.get("units") else ""
        lines.append(f"    {f['name']:<28} {f['location']:<6} x{f['components']}{units}")
    if len(d["fields"]) > 80:
        lines.append(f"    ... {len(d['fields']) - 80} more")
    if d["boundary_conditions"]:
        lines.append(f"  boundary conditions: {len(d['boundary_conditions'])} patches")
    for w in d["warnings"]:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)


def _cmd_describe(args: argparse.Namespace) -> int:
    desc = registry.describe(args.path, args.connector, options=_options(args))
    d = desc.to_dict()
    _emit(args, d, _describe_text(d))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from frontier_connectors.convert import convert

    run: Any = convert
    extra: dict[str, Any] = {}
    if args.cases or args.case_params:
        from frontier_connectors.cases import convert_cases

        run = convert_cases
        extra = {
            "cases": args.cases or "auto",
            "params": args.case_params,
            "split_seed": args.split_seed,
        }
    result = run(
        args.path,
        args.out,
        **extra,
        connector=args.connector,
        zones=_csv(args.zones),
        fields=_csv(args.fields),
        times=parse_time_selection(args.times),
        options=_options(args),
        name=args.name,
        domain=args.domain,
        license=args.license,
        citation=args.citation,
        overwrite=args.overwrite,
        time_as_samples=args.time_as_samples,
        steps_per_chunk=args.steps_per_chunk,
        backend=args.backend,
        length_unit=args.length_unit,
        source_length=args.source_length,
    )
    lines = [f"wrote {len(result.sample_ids)} sample(s) to {result.out} in {result.seconds:.2f}s"]
    if result.cases:
        names = sorted({k for c in result.cases for k in c["parameters"]})
        sizes = ", ".join(f"{k} {len(v)}" for k, v in result.splits.items())
        lines.append(
            f"  {len(result.cases)} cases; parameters ({result.parameters_from}): "
            f"{', '.join(names) or 'none'}; splits: {sizes}"
        )
    for sid in result.sample_ids[:40]:
        lines.append(f"  {sid}: {result.n_steps.get(sid, 1)} step(s) via {result.backends[sid]}")
    lines += [f"  warning: {w}" for w in result.warnings]
    _emit(args, result.to_dict(), "\n".join(lines))
    return 0


def _cmd_exporters(args: argparse.Namespace) -> int:
    from frontier_connectors.plugins import exporters

    if args.name is None:
        payload = {r["vendor"]: r["files"] for r in exporters.list_exporters() if r["enabled"]}
        text = "\n".join(f"{n}:\n  " + "\n  ".join(files) for n, files in payload.items())
        _emit(args, payload, text or "(no exporter plugins installed)")
        return 0
    if args.copy_to:
        result = exporters.install_exporter(args.name, args.copy_to, force=True)
        _emit(args, {"copied": result["path"]} | result, f"copied exporter to {result['path']}")
    else:
        src = exporters.exporter_dir(args.name)
        _emit(args, {"path": str(src)}, str(src))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from frontier_connectors.convert import frontier_data_available
    from frontier_connectors.plugins import health as doctor_mod

    report = doctor_mod.doctor(load=True)
    seen: dict[str, dict[str, Any]] = {}
    for mod, lic in (("numpy", "BSD-3-Clause"), ("zarr", "MIT"), ("numcodecs", "MIT")):
        seen[mod] = {
            "module": mod,
            "version": doctor_mod.installed_version(mod),
            "license": lic,
            "required": True,
            "used_by": ["frontier-connectors"],
        }
    for row in report["plugins"]:
        for r in row["requires_status"]:
            name = doctor_mod.requirement_name(r["requirement"])
            entry = seen.setdefault(
                name,
                {"module": name, "version": r["installed"], "license": "", "required": True},
            )
            entry.setdefault("used_by", []).append(row["id"])
        for o in row["optional_status"]:
            entry = seen.setdefault(
                o["package"],
                {
                    "module": o["package"],
                    "version": o["installed"],
                    "license": o["licence"] + (f" ({o['flag']})" if o["flag"] else ""),
                    "required": False,
                },
            )
            entry.setdefault("used_by", []).append(row["id"])
    fd = doctor_mod.installed_version("frontier-data", "frontier_data")
    seen["frontier_data"] = {
        "module": "frontier_data",
        "version": fd,
        "license": "Apache-2.0",
        "required": False,
        "used_by": ["frontier-connectors"],
    }
    rows = list(seen.values())
    payload = {
        "modules": rows,
        "frontier_data_v11_writers": frontier_data_available(),
        "plugins": [
            {"id": r["id"], "status": r["status"], "ok": r["ok"], "problems": r["problems"]}
            for r in report["plugins"]
        ],
        "ok": report["ok"],
    }
    lines = [
        f"{r['module']:<22} {r['version'] or 'missing':<14} {r['license']}"
        + (" (required)" if r["required"] else "")
        for r in rows
    ]
    lines.append(f"frontier_data v1.1 writers: {payload['frontier_data_v11_writers']}")
    bad = [r for r in report["plugins"] if not r["ok"] and r["status"] != "disabled"]
    lines += [f"plugin {r['id']}: {'; '.join(r['problems'])}" for r in bad]
    lines.append("details per plugin: python -m frontier_connectors plugins doctor")
    _emit(args, payload, "\n".join(lines))
    return 0 if all(r["version"] for r in rows if r["required"]) and not bad else 1


def _cmd_importers(args: argparse.Namespace) -> int:
    from frontier_connectors import importers

    rows = importers.list_importers()
    lines = [f"{'id':<14} {'maturity':<28} {'credentials':<28} source"]
    for r in rows:
        env = ", ".join(
            f"{k}{'' if k in r['credentials_present'] else ' (unset)'}"
            for k in r["credentials_env"]
        )
        lines.append(f"{r['id']:<14} {r['maturity']:<28} {env or '-':<28} {r['source']}")
        if r["status"] == "broken":
            lines.append(f"{'':<14} error: {r['note']}")
    _emit(args, rows, "\n".join(lines) if rows else "(no importer plugins installed)")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from frontier_connectors import importers

    result = importers.run_import(
        args.importer,
        args.source,
        args.out,
        download_dir=args.download_dir,
        dry_run=args.dry_run,
        options=_options(args),
        zones=_csv(args.zones),
        fields=_csv(args.fields),
        times=parse_time_selection(args.times),
        connector=args.connector,
        overwrite=args.overwrite,
    )
    if result["dry_run"]:
        lines = [f"dry run: {args.importer} would make {len(result['calls'])} call(s)"]
        lines += ["  " + importers.PlannedCall(**c).line() for c in result["calls"]]
    else:
        lines = [f"fetched {args.source} into {result['downloaded']}"]
        listing = result.get("listing")
        if listing is not None:
            rows = listing.get("items", listing) if isinstance(listing, dict) else listing
            if isinstance(rows, dict):
                rows = next((v for v in rows.values() if isinstance(v, list)), [rows])
            lines = [f"{args.source}: {len(rows)} item(s)"]
            for row in rows[:200]:
                if isinstance(row, dict):
                    ident = row.get("id") or row.get("name") or ""
                    rest = ", ".join(
                        f"{k}={v}"
                        for k, v in row.items()
                        if k != "id" and isinstance(v, str | int | float | bool)
                    )
                    lines.append(f"  {ident}  {rest}"[:200])
                else:
                    lines.append(f"  {row}")
        if "dataset" in result:
            ds = result["dataset"]
            lines.append(f"wrote {len(ds['sample_ids'])} sample(s) to {ds['out']}")
    _emit(args, result, "\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Print one JSON document on stdout")
    common.add_argument("--compact", action="store_true", help="Print the JSON without indentation")
    parser = argparse.ArgumentParser(
        prog="python -m frontier_connectors",
        description="Read CFD and CAE solver output into a Frontier dataset.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", parents=[common], help="List connectors with their maturity")
    p.add_argument("--maturity", help="Show only connectors with this maturity")
    p.add_argument("--markdown", action="store_true", help="Print the support matrix as Markdown")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("detect", parents=[common], help="Show which connectors recognise a path")
    p.add_argument("path")
    p.set_defaults(func=_cmd_detect)

    def source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("path")
        p.add_argument("--connector", help="Use this connector instead of detection")
        p.add_argument("--mapping", help="Read the layout from this JSON or YAML mapping config")
        p.add_argument(
            "--option",
            action="append",
            metavar="KEY=VALUE",
            help="Set a connector option; repeatable",
        )

    p = sub.add_parser(
        "describe", parents=[common], help="Show zones, fields, time steps and metadata"
    )
    source_args(p)
    p.set_defaults(func=_cmd_describe)

    p = sub.add_parser("convert", parents=[common], help="Convert a source to a Frontier dataset")
    source_args(p)
    p.add_argument("--out", required=True, help="Write the dataset to this directory")
    p.add_argument("--zones", help="Select zones by name or glob, comma-separated; !name excludes")
    p.add_argument("--fields", help="Select fields by name or glob, comma-separated")
    p.add_argument(
        "--times",
        help="Select time steps: all, first, last, <index>, start:stop:step, i,j,k or t=<time>,...",
    )
    p.add_argument(
        "--cases",
        help="Convert many cases under the path, one sample each: a glob such as '*/' or "
        "'run_*/case.foam', or auto for every child a connector recognises",
    )
    p.add_argument(
        "--case-params",
        help="Read case parameters from this CSV or JSON table, from the case names (names), "
        "or not at all (none); default: cases.csv or parameters.csv under the path, else names",
    )
    p.add_argument("--split-seed", type=int, default=0, help="Seed of the split by case")
    p.add_argument("--name", help="Set the dataset name (default: the output directory name)")
    p.add_argument("--domain", help="Set the domain: cfd, structural, thermal, multiphysics, ...")
    p.add_argument("--license", help="Record the dataset licence")
    p.add_argument("--citation", help="Record how to cite the dataset")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing dataset")
    p.add_argument(
        "--time-as-samples",
        action="store_true",
        help="Store every time step as its own sample, for meshes that change in time",
    )
    p.add_argument(
        "--steps-per-chunk", type=int, default=1, help="Set the time steps per chunk (default: 1)"
    )
    p.add_argument("--length-unit", help="Rescale coordinates to this unit, for example m")
    p.add_argument("--source-length", help="Set the length unit of a source that records none")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "frontier_data", "zarr"],
        help="Choose the dataset writer (default: auto)",
    )
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser(
        "exporters", parents=[common], help="List the installed exporter script bundles"
    )
    p.add_argument("name", nargs="?", help="Show the bundle of this tool")
    p.add_argument("--copy-to", help="Copy the bundle to this directory")
    p.set_defaults(func=_cmd_exporters)

    p = sub.add_parser(
        "doctor", parents=[common], help="List optional readers, their licences and what is missing"
    )
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser(
        "importers",
        parents=[common],
        help="List importers: cloud services, live sessions, in-situ streams",
    )
    p.set_defaults(func=_cmd_importers)

    p = sub.add_parser(
        "import",
        parents=[common],
        help="Fetch from a cloud service, live session or stream, then convert",
    )
    p.add_argument("importer", help="Importer id; see `importers`")
    p.add_argument("source", help="Case id, job id, session spec or stream to fetch")
    p.add_argument("--out", help="Write the dataset to this directory; omit to download only")
    p.add_argument(
        "--download-dir", help="Download raw files to this directory (default: the user cache)"
    )
    p.add_argument("--dry-run", action="store_true", help="Print the calls and exit")
    p.add_argument("--connector", help="Use this connector on the downloaded files")
    p.add_argument("--mapping", help="Read the layout from this JSON or YAML mapping config")
    p.add_argument(
        "--option", action="append", metavar="KEY=VALUE", help="Set an importer option; repeatable"
    )
    p.add_argument("--zones", help="Select zones by name or glob, comma-separated")
    p.add_argument("--fields", help="Select fields by name or glob, comma-separated")
    p.add_argument("--times", help="Select time steps; same syntax as convert --times")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing dataset")
    p.set_defaults(func=_cmd_import)

    plugins_cli.add_parser(sub, common, _emit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConnectorError as e:
        if args.json:
            sys.stdout.write(json.dumps(e.to_dict()) + "\n")
        else:
            sys.stderr.write(f"error: {e}\n")
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
