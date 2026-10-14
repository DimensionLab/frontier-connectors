# SPDX-License-Identifier: Apache-2.0
"""Install, uninstall and search plugins. Thin, explicit wrappers around ``pip`` / ``uv pip``.

Three sources, all through the same installer so dependency resolution stays standard:

* an index (PyPI by default, ``--index-url`` for a private one),
* a directory of wheels (``--find-links DIR``, add ``--no-index`` on an air-gapped machine),
* a path, wheel file, or ``git+https://...`` URL of a third-party plugin.

A plugin id (``fluent``, ``exporter-matlab``) is translated to its distribution name through the
catalog shipped with the framework and the ``plugins-index.json`` of a wheel directory.

With the Frontier SDK installed (``frontier-connectors[config]``) the defaults come from the
configuration and the organisation policy is enforced before anything is installed:

* ``plugins.installer`` (``auto`` | ``uv`` | ``pip``), ``plugins.wheel_dir`` (a wheel directory
  consulted in addition to ``--find-links``) and ``plugins.index_url``. Command-line flags win,
  then ``[sources]`` of ``plugins.toml``, then the settings;
* ``policy.plugin_sources`` lists where plugins may come from: ``any``, ``bundled`` (the wheel
  directory shipped inside the product, :data:`BUNDLED_WHEEL_DIR`), ``pypi``, ``wheel_dir``
  (the configured ``plugins.wheel_dir``), ``index`` (the configured ``plugins.index_url``), or a
  URL or path prefix. A source named on the command line that is not allowed is refused
  (:class:`~frontier_connectors.errors.PolicyError`); a default source that is not allowed
  (PyPI, a directory from ``plugins.toml``) is left out, and the install is refused when no
  allowed source remains.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from frontier_connectors import credentials
from frontier_connectors.errors import ConnectorError, PolicyError
from frontier_connectors.network import offline, require_online
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as state_mod
from frontier_connectors.plugins.manifest import (
    MANIFEST_NAME,
    ManifestError,
    auth_providers,
    validate,
)

INSTALLER_ENV = "FRONTIER_PLUGIN_INSTALLER"
#: Pass installer output on to stderr line by line as it is produced (set by the server).
STREAM_ENV = "FRONTIER_PLUGIN_STREAM"
INDEX_FILE = "plugins-index.json"
#: Written into a wheel directory when it is built (the air-gapped bundle): distribution,
#: version, file name and sha256 of every first-party wheel. When it is present, the wheels
#: of that directory are verified against it before the installer runs.
LOCK_FILE = "plugins-lock.json"
_URL_PREFIXES = ("git+", "http://", "https://", "file://", "hg+", "svn+", "bzr+")
#: wheels shipped inside the product; consulted automatically when the directory exists and
#: the source that ``policy.plugin_sources`` calls ``bundled``
BUNDLED_WHEEL_DIR = Path(sys.prefix) / "share" / "frontier" / "plugins"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_wheel_dir(directory: str | Path) -> dict[str, Any]:
    """Check a wheel directory against its ``plugins-lock.json``.

    A directory without the file is not verified (``{"locked": False}``): it is a plain
    ``--find-links`` directory, trusted as much as whoever put it in the plugin sources.
    With the file, every ``frontier*`` wheel in the directory must be listed with the same
    sha256, and every listed wheel that is present must match; a wheel that was swapped,
    altered or added after the directory was built raises :class:`ConnectorError` and
    nothing is installed. The lock itself is covered by the bundle's ``SHA256SUMS``, which
    the release signs.
    """
    d = Path(directory).expanduser()
    lock = d / LOCK_FILE
    if not lock.is_file():
        return {"locked": False, "directory": str(d)}
    try:
        entries = {e["file"]: e for e in _read_json(str(lock)).get("wheels", [])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ConnectorError(
            f"{lock} cannot be read ({e}); refusing to install from {d}",
            hint="restore the directory from the release bundle and check SHA256SUMS",
        ) from e
    problems: list[str] = []
    for wheel in sorted(d.glob("frontier*.whl")):
        entry = entries.get(wheel.name)
        if entry is None:
            problems.append(f"{wheel.name}: not listed in {LOCK_FILE}")
        elif _sha256(wheel) != str(entry.get("sha256", "")).lower():
            problems.append(f"{wheel.name}: content differs from {LOCK_FILE}")
    if problems:
        raise ConnectorError(
            f"wheel directory {d} failed verification:\n  " + "\n  ".join(problems),
            hint="the wheels were changed after the directory was built; restore it from the "
            "release bundle (sha256sum -c SHA256SUMS)",
        )
    return {"locked": True, "directory": str(d), "wheels": len(entries)}


def installer_choice() -> str:
    """``"pip"``, ``"uv"`` or ``""`` (automatic): the setting ``plugins.installer`` when the
    Frontier SDK is installed, else ``FRONTIER_PLUGIN_INSTALLER`` (the setting's legacy
    variable)."""
    choice = credentials.setting("plugins.installer")
    if choice is None:
        choice = os.environ.get(INSTALLER_ENV, "")
    choice = str(choice).strip().lower()
    return "" if choice == "auto" else choice


def source_defaults() -> dict[str, str]:
    """``{"wheel_dir", "index_url"}`` from the settings ``plugins.wheel_dir`` and
    ``plugins.index_url`` (empty strings without the SDK or when unset)."""
    wheel_dir = credentials.setting("plugins.wheel_dir")
    index_url = credentials.setting("plugins.index_url")
    return {
        "wheel_dir": str(Path(str(wheel_dir)).expanduser().resolve()) if wheel_dir else "",
        "index_url": str(index_url or ""),
    }


def installer_command() -> list[str]:
    """``[python, -m, pip]`` or ``[uv, pip]`` for the running interpreter.

    The setting ``plugins.installer`` (``FRONTIER_PLUGIN_INSTALLER=pip|uv``) forces one.
    Environments created by ``uv venv`` have no pip, so uv is the fallback (and the first
    choice when pip is absent)."""
    choice = installer_choice()
    has_pip = importlib.util.find_spec("pip") is not None
    uv = shutil.which("uv")
    if choice == "pip" or (not choice and has_pip):
        if not has_pip:
            raise ConnectorError(
                "pip is not available in this environment",
                hint=f"unset {INSTALLER_ENV} or run: {sys.executable} -m ensurepip",
            )
        return [sys.executable, "-m", "pip"]
    if choice in ("", "uv") and uv:
        return [uv, "pip"]
    raise ConnectorError(
        "no package installer found (neither pip in this environment nor uv on PATH)",
        hint=f"run: {sys.executable} -m ensurepip, or install uv (https://docs.astral.sh/uv/)",
    )


def _is_uv(cmd: Sequence[str]) -> bool:
    return len(cmd) >= 2 and cmd[1] == "pip" and Path(cmd[0]).name.startswith("uv")


def _python_args(cmd: Sequence[str]) -> list[str]:
    return ["--python", sys.executable] if _is_uv(cmd) else []


def wheel_manifest(wheel: Path) -> dict[str, Any] | None:
    """Catalog entry for a plugin wheel, read from the ``plugin.toml`` inside the archive."""
    try:
        with zipfile.ZipFile(wheel) as zf:
            names = [
                n for n in zf.namelist() if n.endswith("/" + MANIFEST_NAME) and n.count("/") == 1
            ]
            if not names:
                return None
            doc = tomllib.loads(zf.read(names[0]).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None
    try:
        m = validate(doc, source=str(wheel))
    except ManifestError:
        return None
    dist = wheel.name.split("-")[0].replace("_", "-").lower()
    return catalog_entry(m.to_dict(), dist) | {"wheel": wheel.name}


def catalog_entry(manifest: dict[str, Any], dist: str) -> dict[str, Any]:
    """The catalog / index row for one manifest (the JSON shape of ``plugins search``)."""
    exporter = manifest.get("exporter") or {}
    return {
        "id": manifest["id"],
        "dist": dist,
        "kind": manifest["kind"],
        "name": manifest["name"],
        "summary": manifest.get("summary", ""),
        "version": manifest["version"],
        "data_model_version": manifest["data_model_version"],
        "vendor": manifest.get("vendor", ""),
        "formats": list(manifest.get("formats", [])),
        "maturity": manifest.get("maturity", ""),
        "licence": manifest.get("licence", ""),
        "requires": list(manifest.get("requires", [])),
        "plugins": list(manifest.get("plugins", [])),
        "suggests": list(manifest.get("suggests", [])),
        "optional": [dict(o) for o in manifest.get("optional", [])],
        "connectors": sorted(manifest.get("connectors", {})),
        "importers": sorted(manifest.get("importers", {})),
        "exporters": [exporter["vendor"]] if exporter.get("vendor") else [],
        "auth": auth_providers(manifest.get("auth", [])),
    }


def _read_json(source: str) -> Any:
    if source.startswith(("http://", "https://")):
        require_online("fetching the plugin index", url=source)
        with urllib.request.urlopen(source, timeout=20) as resp:  # noqa: S310 - scheme checked
            return json.loads(resp.read().decode("utf-8"))
    return json.loads(Path(source.removeprefix("file://")).read_text(encoding="utf-8"))


def available(
    find_links: Sequence[str] = (), catalog: str | None = None
) -> dict[str, dict[str, Any]]:
    """Known plugins by id: the built-in catalog, overlaid by ``plugins-index.json`` (or the
    wheels themselves) of every ``find_links`` directory and by an explicit catalog URL/file."""
    rows: dict[str, dict[str, Any]] = {
        e["id"]: dict(e) | {"origin": "catalog"} for e in discovery.catalog()
    }
    for directory in find_links:
        d = Path(directory).expanduser()
        if not d.is_dir():
            continue
        index = d / INDEX_FILE
        entries: list[dict[str, Any]] = []
        if index.is_file():
            try:
                entries = list(_read_json(str(index)).get("plugins", []))
            except (OSError, ValueError, AttributeError):
                entries = []
        if not entries:
            entries = [e for e in map(wheel_manifest, sorted(d.glob("*.whl"))) if e]
        for e in entries:
            rows[e["id"]] = dict(e) | {"origin": str(d)}
    if catalog:
        try:
            for e in _read_json(catalog).get("plugins", []):
                rows[e["id"]] = dict(e) | {"origin": catalog}
        except (OSError, ValueError, AttributeError) as e:
            raise ConnectorError(
                f"plugin catalog {catalog} could not be read: {e}",
                hint="pass the URL or path of a plugins-index.json",
            ) from e
    return rows


def search(
    query: str = "", *, find_links: Sequence[str] = (), catalog: str | None = None
) -> list[dict[str, Any]]:
    """Catalog rows matching ``query`` (id, name, vendor, formats, connector ids, summary),
    each with ``installed`` / ``status`` of the local environment."""
    installed = {r.id: r for r in discovery.discover()}
    rows = available(find_links, catalog)
    for rec in installed.values():
        if rec.id not in rows:  # third-party plugins that no catalog knows
            rows[rec.id] = catalog_entry(rec.to_dict() | {"exporter": None}, rec.dist) | {
                "origin": "installed",
                "exporters": sorted(rec.exporters),
            }
    words = [w for w in query.lower().split() if w]
    out = []
    for pid in sorted(rows):
        row = rows[pid]
        hay = " ".join(
            [row["id"], row.get("dist", ""), row.get("name", ""), row.get("vendor", "")]
            + [row.get("summary", ""), row.get("kind", "")]
            + list(row.get("formats", []))
            + list(row.get("connectors", []))
            + list(row.get("importers", []))
            + list(row.get("exporters", []))
        ).lower()
        if all(w in hay for w in words):
            rec = installed.get(pid)
            row["installed"] = rec is not None
            row["installed_version"] = rec.version if rec else None
            row["status"] = rec.status if rec else "not-installed"
            out.append(row)
    return out


def _requirement(spec: str, rows: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    """(what to hand to the installer, plugin id when known)."""
    if spec.startswith(_URL_PREFIXES) or Path(spec).expanduser().exists():
        p = Path(spec).expanduser()
        return (str(p.resolve()) if p.exists() else spec), None
    bare = spec.split("=")[0].split("<")[0].split(">")[0].split("[")[0].strip()
    if bare in rows:
        return rows[bare]["dist"] + spec[len(bare) :], bare
    rec = discovery.find(bare)
    if rec is not None:
        return rec.dist + spec[len(bare) :], rec.id
    return spec, None


#: exported to the installer when offline mode is on: neither pip nor uv contacts an index
OFFLINE_INSTALLER_ENV = {
    "PIP_NO_INDEX": "1",
    "UV_OFFLINE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
}


def _run(cmd: list[str], dry_run: bool, *, no_network: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"command": cmd, "returncode": None, "output": "", "dry_run": True}
    # One pipe for both streams, read line by line: with FRONTIER_PLUGIN_STREAM=1 (the
    # server's operations set it) every line is passed on to stderr the moment the
    # installer prints it, so a page can show progress instead of waiting for the end.
    stream = os.environ.get(STREAM_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    if stream:
        print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    lines: list[str] = []
    env = {**os.environ, **OFFLINE_INSTALLER_ENV} if no_network else None
    with subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if stream:
                print(line.rstrip("\n"), file=sys.stderr, flush=True)
        returncode = proc.wait()
    return {
        "command": cmd,
        "returncode": returncode,
        "output": "".join(lines)[-4000:],
        "dry_run": False,
        "streamed": stream,
    }


def _resolve_dir(directory: str) -> str:
    return str(Path(directory).expanduser().resolve())


def _auth_refresh() -> dict[str, Any]:
    from frontier_connectors.plugins import auth as auth_mod

    info = auth_mod.refresh()
    return {"path": info["path"], "written": info["written"], "error": info["error"]}


class _Sources:
    """Where one install may take packages from, checked against ``policy.plugin_sources``."""

    def __init__(self, defaults: dict[str, str]) -> None:
        self.wheel_dir = defaults["wheel_dir"]
        self.index_url = defaults["index_url"]
        self.bundled = str(BUNDLED_WHEEL_DIR.resolve()) if BUNDLED_WHEEL_DIR.is_dir() else ""
        #: source labels the installer will consult, in order
        self.used: list[str] = []
        #: default sources left out because the policy does not allow them
        self.excluded: list[str] = []
        self._first_refusal: Any = None
        module = credentials.sdk("config")
        self._violation: Any = getattr(module, "PolicyViolation", ()) if module else ()
        try:
            self._policy = module.policy() if module is not None else None
        except Exception:  # noqa: BLE001 - no usable configuration: nothing to enforce
            self._policy = None

    def allow(self, label: str, *, explicit: bool) -> bool:
        """Record ``label`` as used when the policy allows it. A refused source raises when
        the user named it (``explicit``) and is left out otherwise."""
        if self._policy is not None:
            try:
                self._policy.check_plugin_source(label)
            except self._violation as v:
                if explicit:
                    raise PolicyError(v) from None
                self._first_refusal = self._first_refusal or v
                self.excluded.append(label)
                return False
        if label not in self.used:
            self.used.append(label)
        return True

    def allow_link(self, directory: str, *, explicit: bool) -> bool:
        if directory == self.bundled:
            label = "bundled"
        elif directory == self.wheel_dir:
            label = "wheel_dir"
        else:
            label = directory
        return self.allow(label, explicit=explicit)

    def allow_index(self, url: str, *, explicit: bool) -> bool:
        same = self.index_url and url.rstrip("/") == self.index_url.rstrip("/")
        return self.allow("index" if same else url, explicit=explicit)

    def nothing_left(self) -> None:
        """Every default source was excluded by the policy: say so instead of letting the
        installer fail with "no matching distribution"."""
        if self._first_refusal is not None:
            raise PolicyError(self._first_refusal)


def check_change(action: str, plugin: str) -> None:
    """Raise :class:`PolicyError` when ``policy.allow_plugin_changes`` freezes the plugin set
    (nothing to enforce without the SDK's configuration)."""
    module = credentials.sdk("config")
    if module is None:
        return
    try:
        policy = module.policy()
    except Exception:  # noqa: BLE001 - no usable configuration: nothing to enforce
        return
    check = getattr(policy, "check_plugin_change", None)
    if check is None:
        return
    try:
        check(action, plugin)
    except module.PolicyViolation as v:
        raise PolicyError(v) from None


def install(
    specs: Sequence[str],
    *,
    find_links: Sequence[str] = (),
    index_url: str | None = None,
    extra_index_url: str | None = None,
    no_index: bool = False,
    with_suggested: bool = True,
    upgrade: bool = False,
    editable: bool = False,
    catalog: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install plugins by id, distribution name, path, wheel or VCS URL."""
    if not specs:
        raise ConnectorError("nothing to install", hint="pass a plugin id, path or git URL")
    if not dry_run:
        check_change("install", " ".join(specs))
    # offline mode: no index, no URL; wheels come from the wheel directories only
    is_offline = offline()
    if is_offline and not dry_run:
        if index_url or extra_index_url:
            require_online("installing plugins from a package index")
        for spec in specs:
            if spec.startswith(_URL_PREFIXES) and "file://" not in spec:
                require_online(f"installing a plugin from {spec.split('://')[0]}://...")
    st = state_mod.load()
    sources = _Sources(source_defaults())
    links = [d for d in map(_resolve_dir, find_links) if sources.allow_link(d, explicit=True)]
    defaults = [*st.find_links, sources.wheel_dir, sources.bundled]
    for d in map(_resolve_dir, filter(None, defaults)):
        if d not in links and sources.allow_link(d, explicit=False):
            links.append(d)
    if index_url:
        sources.allow_index(index_url, explicit=True)
    else:
        default_index = st.index_url or sources.index_url or None
        if default_index and sources.allow_index(default_index, explicit=False):
            index_url = default_index
    if extra_index_url:
        sources.allow_index(extra_index_url, explicit=True)
    if is_offline:
        no_index, index_url, extra_index_url = True, None, None
    if not no_index and not index_url and not sources.allow("pypi", explicit=False):
        no_index = True  # PyPI is not an allowed source here: resolve from what is
    # a wheel directory that carries a lock is verified before anything is resolved from it
    verified = [verify_wheel_dir(d) for d in links]
    rows = available(links, catalog)
    requirements: list[str] = []
    ids: list[str] = []
    for spec in specs:
        req, pid = _requirement(spec, rows)
        if pid is None and (spec.startswith(_URL_PREFIXES) or Path(spec).expanduser().exists()):
            sources.allow(req, explicit=True)
        elif no_index and not links and not extra_index_url:
            sources.nothing_left()
        requirements.append(req)
        if pid:
            ids.append(pid)
            if with_suggested:
                for sug in rows.get(pid, {}).get("suggests", []):
                    if sug in rows and rows[sug]["dist"] not in requirements:
                        requirements.append(rows[sug]["dist"])
                        ids.append(sug)
    base = installer_command()
    cmd = [*base, "install", *_python_args(base)]
    for d in links:
        cmd += ["--find-links", d]
    if no_index:
        cmd.append("--no-index")
    if index_url:
        cmd += ["--index-url", index_url]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    if upgrade:
        cmd.append("--upgrade")
    if editable:
        if len(requirements) != 1:
            raise ConnectorError("--editable takes exactly one path")
        cmd.append("-e")
    cmd += requirements
    result = _run(cmd, dry_run, no_network=is_offline)
    result["requested"] = list(specs)
    result["plugins"] = []
    result["sources"] = sources.used
    result["verified_wheel_dirs"] = [v["directory"] for v in verified if v["locked"]]
    result["policy_excluded"] = sources.excluded
    if not dry_run:
        if result["returncode"] != 0:
            raise ConnectorError(
                "installing " + ", ".join(specs) + " failed:\n" + result["output"].strip(),
                hint="offline? add --find-links <wheel dir> --no-index; "
                "see `plugins search` for known ids",
            )
        importlib.invalidate_caches()
        records = discovery.discover()
        wanted = set(ids)
        result["plugins"] = [
            r.to_dict() for r in records if r.id in wanted or (not ids and not r.first_party)
        ]
        result["auth_providers"] = _auth_refresh()
    return result


def uninstall(ids: Sequence[str], *, purge: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Uninstall plugins by id or distribution name. The framework itself is refused."""
    if not dry_run:
        check_change("uninstall", " ".join(ids))
    records = discovery.discover()
    dists: list[str] = []
    removed: list[str] = []
    for pid in ids:
        rec = discovery.find(pid, records)
        dist = rec.dist if rec else discovery.norm_dist(pid)
        if dist == discovery.FRAMEWORK_DIST:
            raise ConnectorError(
                "refusing to uninstall the framework itself",
                hint="uninstall frontier-connectors with pip directly if that is what you want",
            )
        if rec is None:
            raise ConnectorError(
                f"no installed plugin {pid!r}",
                hint="see `python -m frontier_connectors plugins list`",
            )
        dists.append(dist)
        removed.append(rec.id)
    base = installer_command()
    cmd = [*base, "uninstall", *_python_args(base)]
    if not _is_uv(base):
        cmd.append("-y")
    cmd += dists
    result = _run(cmd, dry_run)
    result["uninstalled"] = removed
    if not dry_run:
        if result["returncode"] != 0:
            raise ConnectorError("uninstall failed:\n" + result["output"].strip())
        importlib.invalidate_caches()
        if purge:
            st = state_mod.load()
            changed = False
            for pid in removed:
                changed |= st.set_enabled("plugins", pid, True)
            if changed:
                state_mod.save(st)
        result["auth_providers"] = _auth_refresh()
    return result
