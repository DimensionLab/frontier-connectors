# SPDX-License-Identifier: Apache-2.0
"""Importers: plugins that *acquire* data which is not a local file yet.

A connector reads a path. An importer gets the data to a path first: it downloads the results
of a cloud solver run, drives a live solver session to export them, or consumes an in-situ
stream. What it leaves behind is an ordinary directory that connectors read, so the two halves
stay independently testable and an air-gapped site can run the fetch on one machine and the
conversion on another.

.. code-block:: python

    class MyCloudImporter(BaseImporter):
        id = "mycloud"
        name = "My Cloud CFD"
        credentials_env = ["MYCLOUD_API_KEY"]

        def plan(self, source, *, options=None):
            return [PlannedCall("GET", f"https://api.example.com/cases/{source}", "case metadata")]

        def fetch(self, source, dest, *, options=None):
            ...  # download into dest
            return dest

Register it under the entry-point group ``frontier.importers`` (``[importers]`` in
``plugin.toml``). ``python -m frontier_connectors import mycloud <source> --out ds`` runs
``fetch`` and then the normal conversion; ``--dry-run`` prints ``plan`` and touches nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from frontier_connectors import credentials
from frontier_connectors.errors import ConnectorError, NeedsCredentialsError, UnsupportedSourceError
from frontier_connectors.plugins import discovery
from frontier_connectors.plugins import state as state_mod

log = logging.getLogger("frontier_connectors")


@dataclass
class PlannedCall:
    """One thing an importer would do: an HTTP request, an SDK call or a command line.

    Secrets never appear here: headers carrying credentials are listed by name with the value
    replaced by the name of the environment variable they come from."""

    method: str  # GET | POST | SDK | CMD | FILE
    target: str  # URL, "package.function", command line or path
    purpose: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        extra = f"  {self.params}" if self.params else ""
        why = f"   # {self.purpose}" if self.purpose else ""
        return f"{self.method:<5} {self.target}{extra}{why}"


@runtime_checkable
class Importer(Protocol):
    """The importer plugin contract."""

    id: str
    name: str
    vendor: str
    license: str

    def plan(
        self, source: str, *, options: Mapping[str, Any] | None = None
    ) -> list[PlannedCall]: ...

    def fetch(
        self, source: str, dest: Path, *, options: Mapping[str, Any] | None = None
    ) -> Path: ...


class BaseImporter:
    """Convenience base with the optional attributes filled in."""

    id: str = ""
    name: str = ""
    vendor: str = ""
    license: str = "Apache-2.0"
    maturity: str = "planned"
    #: environment variables naming the importer's credentials (names only); each is resolved
    #: through :func:`frontier_connectors.credentials.secret`
    credentials_env: list[str] = []
    #: optional pip packages (SDKs), imported lazily
    requires: list[str] = []
    #: what ``source`` means, one line (``"<case-id>"``, ``"<project>/<simulation>/<run>"``)
    source_help: str = ""
    status_note: str = ""
    #: where the API shape was taken from: documentation URL and the date it was read
    api_reference: str = ""

    def credentials(self) -> dict[str, str]:
        """Values of ``credentials_env`` that are set. Raises when none is."""
        found = {k: v for k in self.credentials_env if (v := credentials.secret(k))}
        if self.credentials_env and not found:
            raise NeedsCredentialsError(
                f"{self.name} needs credentials",
                hint=" or ".join(credentials.login_hint(k) for k in self.credentials_env)
                + " (never in a file that is committed); use --dry-run to see the calls "
                "without them",
            )
        return found

    def plan(self, source: str, *, options: Mapping[str, Any] | None = None) -> list[PlannedCall]:
        raise NotImplementedError

    def fetch(self, source: str, dest: Path, *, options: Mapping[str, Any] | None = None) -> Path:
        raise NotImplementedError


class BrokenImporter(BaseImporter):
    """Placeholder for an importer whose module failed to import."""

    def __init__(self, iid: str, target: str, error: BaseException) -> None:
        self.id = iid
        self.name = iid
        self.target = target
        self.error = error

    def plan(self, source: str, *, options: Mapping[str, Any] | None = None) -> list[PlannedCall]:
        raise ConnectorError(
            f"importer {self.id!r} failed to load: {self.error!r}",
            hint="run `python -m frontier_connectors plugins doctor`",
        )

    def fetch(self, source: str, dest: Path, *, options: Mapping[str, Any] | None = None) -> Path:
        self.plan(source)
        return dest


_runtime: dict[str, Any] = {}


def register(importer: Any) -> None:
    """Register an importer instance at runtime (tests, notebooks)."""
    _runtime[importer.id] = importer


def unregister(importer_id: str) -> None:
    _runtime.pop(importer_id, None)


def importers() -> dict[str, Any]:
    """Importers of every enabled plugin, by id. A failing one becomes a BrokenImporter."""
    st = state_mod.load()
    found: dict[str, Any] = {}
    for rec in discovery.discover(st):
        if not rec.enabled:
            continue
        for iid, target in rec.importers.items():
            if st.is_disabled("importers", iid):
                continue
            try:
                inst = discovery.load_target(target)
                iid = getattr(inst, "id", None) or iid
            except Exception as e:  # noqa: BLE001 - one broken plugin must not hide the others
                log.debug("importer %s (%s) failed to load: %r", iid, rec.id, e)
                inst = BrokenImporter(iid, target, e)
            found[iid] = inst
    found.update(_runtime)
    return found


def get(importer_id: str) -> Any:
    reg = importers()
    if importer_id not in reg:
        raise UnsupportedSourceError(
            f"unknown importer {importer_id!r}",
            hint=discovery.missing_hint("importer", importer_id)
            or "known importers: " + (", ".join(sorted(reg)) or "none installed"),
        )
    return reg[importer_id]


def list_importers() -> list[dict[str, Any]]:
    rows = []
    for iid, imp in sorted(importers().items()):
        broken = isinstance(imp, BrokenImporter)
        env = list(getattr(imp, "credentials_env", []) or [])
        rows.append(
            {
                "id": iid,
                "name": getattr(imp, "name", iid),
                "vendor": getattr(imp, "vendor", ""),
                "license": getattr(imp, "license", ""),
                "maturity": getattr(imp, "maturity", "planned"),
                "source": getattr(imp, "source_help", ""),
                "credentials_env": env,
                # a listing: the keychain is never asked, so this is a lower bound
                "credentials_present": [k for k in env if credentials.present(k)],
                "requires": list(getattr(imp, "requires", []) or []),
                "api_reference": getattr(imp, "api_reference", ""),
                "status": "broken" if broken else "available",
                "note": repr(imp.error) if broken else getattr(imp, "status_note", ""),
            }
        )
    return rows


def run_import(
    importer_id: str,
    source: str,
    out: Path | str | None = None,
    *,
    download_dir: Path | str | None = None,
    dry_run: bool = False,
    options: Mapping[str, Any] | None = None,
    zones: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    times: Any = None,
    connector: str | None = None,
    **convert_kwargs: Any,
) -> dict[str, Any]:
    """Plan, fetch and convert. With ``dry_run`` only the plan is returned and nothing is
    contacted; without ``out`` the data is fetched and left in ``download_dir``."""
    imp = get(importer_id)
    if dry_run:
        calls = imp.plan(source, options=options)
        return {
            "importer": importer_id,
            "source": source,
            "dry_run": True,
            "calls": [c.to_dict() for c in calls],
        }
    if download_dir is None:
        from frontier_connectors.cloud import cache_root

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in source)[:120]
        download_dir = cache_root() / importer_id / (safe or "source")
    dest = Path(download_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    fetched = Path(imp.fetch(source, dest, options=options))
    result: dict[str, Any] = {
        "importer": importer_id,
        "source": source,
        "dry_run": False,
        "downloaded": str(fetched),
    }
    listing = fetched / "listing.json" if fetched.is_dir() else fetched
    if source.lower().startswith("list:") and listing.name == "listing.json" and listing.is_file():
        # `list:<kind>` sources of the cloud importers: a listing, nothing to convert
        import json

        try:
            result["listing"] = json.loads(listing.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ConnectorError(f"{fetched} is not a readable listing: {e}") from e
        return result
    if out is not None:
        from frontier_connectors.convert import convert

        res = convert(
            fetched,
            out,
            connector=connector,
            zones=list(zones) if zones else None,
            fields=list(fields) if fields else None,
            times=times,
            options=dict(options) if options else None,
            **convert_kwargs,
        )
        result["dataset"] = res.to_dict()
    return result
