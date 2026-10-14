# SPDX-License-Identifier: Apache-2.0
"""Vendor-side exporter scripts: locate them, copy them next to a case, remove them again.

An exporter plugin carries scripts that run *inside* the vendor tool (a STAR-CCM+ macro, an
Abaqus Python script, the MATLAB ``+frontier`` package, a Fluent journal ...). They are copied
out of the Python environment with ``plugins install-exporter <vendor> --to <dir>``; the copy
gets a stamp file recording the plugin version and a hash per file, so that
``uninstall-exporter`` removes exactly what was installed and ``plugins doctor`` can tell
when a copy is older than the installed plugin.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frontier_connectors.errors import ConnectorError
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as state_mod

STAMP = ".frontier-exporter.json"
STAMP_SCHEMA = "frontier-exporter-stamp/1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def installed_exporters(include_disabled: bool = False) -> dict[str, discovery.PluginRecord]:
    """Vendor key -> the plugin that ships its scripts."""
    st = state_mod.load()
    out: dict[str, discovery.PluginRecord] = {}
    for rec in discovery.discover(st):
        if not rec.exporters or (not rec.enabled and not include_disabled):
            continue
        for vendor in rec.exporters:
            key = discovery.norm_vendor(vendor)
            if st.is_disabled("exporters", key) and not include_disabled:
                continue
            out[key] = rec
    return out


def exporter_dir(vendor: str) -> Path:
    """Directory holding the scripts for ``vendor`` (``matlab``, ``starccm``, ``abaqus`` ...)."""
    key = discovery.norm_vendor(vendor)
    rec = installed_exporters().get(key)
    if rec is None:
        known = ", ".join(sorted(installed_exporters())) or "none installed"
        raise ConnectorError(
            f"no exporter scripts for {vendor!r}",
            hint=discovery.missing_hint("exporter", key) or f"installed exporters: {known}",
        )
    path = discovery.exporter_files(rec, key)
    if path is None or not path.is_dir():
        raise ConnectorError(
            f"plugin {rec.id!r} declares exporter {vendor!r} but ships no files",
            hint=f"reinstall it: python -m frontier_connectors plugins install {rec.id}",
        )
    return path


def exporter_files(vendor: str) -> list[str]:
    root = exporter_dir(vendor)
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def list_exporters() -> list[dict[str, Any]]:
    rows = []
    for vendor, rec in sorted(installed_exporters(include_disabled=True).items()):
        spec = rec.manifest.exporter if rec.manifest else None
        enabled = rec.enabled and not state_mod.load().is_disabled("exporters", vendor)
        rows.append(
            {
                "vendor": vendor,
                "plugin": rec.id,
                "version": rec.version,
                "tool": spec.tool if spec else "",
                "usage": spec.usage if spec else "",
                "enabled": enabled,
                "path": str(discovery.exporter_files(rec, vendor) or ""),
                "files": exporter_files(vendor) if enabled else [],
            }
        )
    return rows


def read_stamp(directory: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((directory / STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == STAMP_SCHEMA else None


def install_exporter(vendor: str, to: Path | str, *, force: bool = False) -> dict[str, Any]:
    """Copy the scripts of ``vendor`` into ``<to>/<vendor>/`` and stamp the copy."""
    from frontier_connectors.plugins.installer import check_change

    check_change("install-exporter", vendor)
    key = discovery.norm_vendor(vendor)
    src = exporter_dir(key)
    rec = installed_exporters()[key]
    dest = Path(to).expanduser().resolve() / key
    old = read_stamp(dest)
    if dest.exists() and old is None and any(dest.iterdir()) and not force:
        raise ConnectorError(
            f"{dest} exists and was not created by install-exporter",
            hint="choose another --to directory or pass --force to copy into it",
        )
    files = exporter_files(key)
    dest.mkdir(parents=True, exist_ok=True)
    stamped = []
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        stamped.append({"path": rel, "sha256": _sha256(target)})
    if old:  # files of an older version that no longer exist in this one
        current = set(files)
        for item in old.get("files", []):
            stale = dest / item["path"]
            if item["path"] not in current and stale.is_file():
                if _sha256(stale) == item.get("sha256"):
                    stale.unlink()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    spec = rec.manifest.exporter if rec.manifest else None
    stamp = {
        "schema": STAMP_SCHEMA,
        "vendor": key,
        "plugin": rec.id,
        "dist": rec.dist,
        "version": rec.version,
        "data_model_version": rec.manifest.data_model_version if rec.manifest else "",
        "installed_at": now,
        "files": stamped,
    }
    (dest / STAMP).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    st = state_mod.load()
    st.deployed = [d for d in st.deployed if not (d.vendor == key and d.path == str(dest))]
    st.deployed.append(
        state_mod.Deployed(
            vendor=key, plugin=rec.id, version=rec.version, path=str(dest), installed_at=now
        )
    )
    state_mod.save(st)
    return {
        "vendor": key,
        "plugin": rec.id,
        "version": rec.version,
        "path": str(dest),
        "files": files,
        "replaced_version": old.get("version") if old else None,
        "usage": spec.usage if spec else "",
    }


def uninstall_exporter(
    vendor: str, from_dir: Path | str | None = None, *, force: bool = False
) -> dict[str, Any]:
    """Remove copies made by :func:`install_exporter`: every recorded copy of ``vendor``, or
    the one under ``from_dir``. Files changed since they were copied are kept unless ``force``."""
    from frontier_connectors.plugins.installer import check_change

    check_change("uninstall-exporter", vendor)
    key = discovery.norm_vendor(vendor)
    st = state_mod.load()
    targets: list[Path]
    if from_dir is not None:
        base = Path(from_dir).expanduser().resolve()
        targets = [base if read_stamp(base) else base / key]
    else:
        targets = [Path(d.path) for d in st.deployed if d.vendor == key]
    if not targets:
        raise ConnectorError(
            f"no recorded copy of the {vendor!r} exporter",
            hint="pass --from <dir> (the directory given to install-exporter --to)",
        )
    removed: list[str] = []
    kept: list[str] = []
    for dest in targets:
        stamp = read_stamp(dest)
        if stamp is None or stamp.get("vendor") != key:
            if dest.exists():
                raise ConnectorError(
                    f"{dest} has no {STAMP} for {vendor!r}; not touching it",
                    hint="delete the directory by hand if it is no longer needed",
                )
            st.deployed = [d for d in st.deployed if not (d.vendor == key and d.path == str(dest))]
            continue
        for item in stamp.get("files", []):
            f = dest / item["path"]
            if not f.is_file():
                continue
            if force or _sha256(f) == item.get("sha256"):
                f.unlink()
                removed.append(str(f))
            else:
                kept.append(str(f))
        if not kept:
            (dest / STAMP).unlink(missing_ok=True)
        for d in sorted((p for p in dest.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
        if dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        st.deployed = [d for d in st.deployed if not (d.vendor == key and d.path == str(dest))]
    state_mod.save(st)
    return {"vendor": key, "removed": removed, "kept_modified": kept}


def stale_copies() -> list[dict[str, str]]:
    """Recorded copies whose version differs from the installed plugin (for ``doctor``)."""
    current = {v: r.version for v, r in installed_exporters(include_disabled=True).items()}
    out = []
    for d in state_mod.load().deployed:
        stamp = read_stamp(Path(d.path))
        have = stamp.get("version") if stamp else None
        want = current.get(d.vendor)
        if stamp is None:
            out.append({"vendor": d.vendor, "path": d.path, "problem": "copy is gone"})
        elif want is None:
            out.append({"vendor": d.vendor, "path": d.path, "problem": "plugin is not installed"})
        elif have != want:
            out.append(
                {
                    "vendor": d.vendor,
                    "path": d.path,
                    "problem": f"copy is version {have}, installed plugin is {want}: "
                    f"rerun install-exporter {d.vendor} --to {Path(d.path).parent}",
                }
            )
    return out
