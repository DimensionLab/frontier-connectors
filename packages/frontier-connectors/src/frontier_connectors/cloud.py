# SPDX-License-Identifier: Apache-2.0
"""Shared machinery of the cloud-solver plugins (Flow360, Luminary Cloud, SimScale, Rescale).

A cloud connector does not parse a file format of its own. It locates the results of a cloud
run, brings them into a local directory and hands that directory to the format connectors
(``vtk``, ``tecplot``, ``cgns``, ``ensight``, ...) through the registry, stamping the provider
metadata (ids, solver version, simulation settings, boundary conditions, mesh statistics) on
what those return. The provider code lives in the plugins (``frontier_connector_flow360`` ...);
this module holds what they share:

* one injectable HTTP transport (standard library only, https only, credentials never follow a
  redirect to another host) and :class:`ApiClient` (JSON, downloads, retry on 429/5xx, error
  hints);
* credentials: :func:`secret` resolves a key through the Frontier SDK when it is installed
  (``frontier auth login <provider>``), else from the environment or the OS keyring;
* reference files (``wing.flow360.json``), the ``frontier_cloud.json`` sidecar written after a
  download, safe archive extraction and the dispatch of downloaded files to format connectors;
* :class:`_CloudConnector` (reference file or downloaded directory -> RawSamples) and
  :class:`CloudImporter` (``python -m frontier_connectors import <id> <source>``), both driven
  by the same three provider hooks: ``_plan``, ``_list`` and ``_download``.

Three ways in:

* the importer: ``python -m frontier_connectors import flow360 <case id> --out ds``
  (``--dry-run`` prints the exact calls without credentials or network; the source
  ``list:projects`` / ``list:cases`` ... lists what the account holds);
* a small **reference file** naming the run, e.g. ``wing.flow360.json``::

      {"provider": "flow360", "case_id": "case-...", "outputs": ["surface", "volume"]}

  (``.toml`` works too, and so does a one-line ``flow360://<case id>`` file). ``describe`` and
  ``read`` then need credentials from the environment and download once into the directory
  the *caller* chose (``options={"download_dir": ...}``; default:
  ``~/.cache/frontier-connectors/cloud/<provider>/<id>``);
* an **already downloaded directory** (with the ``frontier_cloud.json`` sidecar, or a provider
  layout such as Flow360's ``simulation.json`` + ``surfaces.tar.gz``): no credentials, no
  network, pure delegation.

API keys are read at call time, are never stored on objects, written to disk or put in messages.

Trust boundary. A reference file is *data*: it may have been received from someone
else. It names what to fetch, never where from or where to:

* the API endpoint is the provider's built-in default, a configured value
  (``RESCALE_API_URL`` ...) whose host is on the provider's allow-list
  (:data:`ALLOWED_API_HOSTS`), or an explicit ``api_url`` argument / ``--option api_url=``
  of the caller. ``api_url`` in a reference file is ignored with a
  :class:`CloudReferenceWarning` when its host is an official one and refused when it is not;
* the download directory is the caller's (``download_dir`` option, the importer's ``dest``)
  or the cache; ``download_dir`` in a reference file is ignored with a warning;
* every file name a provider API returns is confined to the download directory with
  ``frontier_connectors.paths.confine`` before anything is written.

Every outbound call is preceded by ``frontier_connectors.network.require_online``.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from frontier_connectors import archives, credentials, meshutil
from frontier_connectors.errors import (
    ConnectorError,
    MissingDependencyError,
    NeedsCredentialsError,
    UnsupportedSourceError,
)
from frontier_connectors.importers import BaseImporter, PlannedCall
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeSelection,
)
from frontier_connectors.network import require_online
from frontier_connectors.paths import ReferenceEscapeError, confine
from frontier_connectors.select import match_names

SIDECAR = "frontier_cloud.json"
SIDECAR_SCHEMA = "frontier-cloud-download"
LISTING = "listing.json"
LISTING_SCHEMA = "frontier-cloud-listing"
CLOUD_IDS = ("flow360", "luminary", "simscale", "rescale")
EXTRACT_DIR = "_extracted"
#: service name under which :func:`secret` looks a variable up in the OS keyring
KEYRING_SERVICE = credentials.KEYRING_SERVICE
MATURITY = "implemented-unverified-live"

#: files that stand for a whole data set (their pieces must not be read a second time)
COLLECTION_EXT = (".pvd", ".vtm", ".vtmb", ".pvtu", ".pvtp", ".case", ".encas", ".foam", ".xdmf")
#: single-file results worth dispatching when no collection file is present
RESULT_EXT = (
    ".vtu", ".vtp", ".vts", ".vtr", ".vti", ".vtk", ".vtkhdf", ".cgns", ".szplt", ".plt",
    ".exo", ".e", ".med", ".rst", ".rth", ".op2", ".d3plot", ".h5", ".hdf5", ".nc",
)  # fmt: skip
TABLE_EXT = (".csv", ".tsv")
ARCHIVE_EXT = (".tar.gz", ".tgz", ".tar", ".zip")
#: statuses worth a second attempt (rate limit, gateway and availability errors)
RETRY_STATUS = (429, 502, 503, 504)
#: request headers that carry credentials for one of the supported services
SECRET_HEADERS = ("authorization", "x-api-key", "simcloud-api-key")

#: official API hosts per provider. The endpoint of a provider is its built-in default or a
#: configured value on this list; anything else needs an explicit argument of the caller
#: (``options["api_url"]`` / ``--option api_url=``). A reference file never sets it.
ALLOWED_API_HOSTS: dict[str, tuple[str, ...]] = {
    "flow360": ("flow360-api.simulation.cloud",),
    "luminary": ("apis.luminarycloud.com",),
    "simscale": ("api.simscale.com",),
    "rescale": (
        "platform.rescale.com",
        "eu.rescale.com",
        "platform.rescale.jp",
        "kr.rescale.com",
        "itar.rescale.com",
        "itar.rescale-gov.com",
    ),
}
#: keys a reference file must not set: endpoints and output locations are the caller's
FILE_IGNORED_KEYS = ("api_url", "download_dir")


class CloudReferenceWarning(UserWarning):
    """A reference file set a key that only the caller may set; the key was ignored."""


def api_host_allowed(provider: str, url: str) -> bool:
    """``url`` is https and its host is an official host of ``provider`` (default port)."""
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and port in (None, 443)
        and not parts.username
        and (parts.hostname or "").lower() in ALLOWED_API_HOSTS.get(provider, ())
    )


def api_base_url(
    provider: str,
    ref: Mapping[str, Any],
    default: str,
    *,
    service: str,
    configured: str | None = None,
    configured_from: str = "configuration",
) -> str:
    """The API endpoint of ``provider`` for this call (no trailing slash).

    Order: the caller's explicit ``api_url`` (``ref`` only carries one that came from an
    argument or ``--option``: :func:`load_reference` removes it from file content), then
    ``configured`` (a setting or environment variable, which must name an official host),
    then ``default``. Always https."""
    explicit = ref.get("api_url")
    if explicit:
        return require_https(str(explicit).rstrip("/"), f"{service} API URL")
    if configured:
        url = require_https(str(configured).rstrip("/"), f"{service} API URL")
        if not api_host_allowed(provider, url):
            raise ConnectorError(
                f"{configured_from} names a {service} API host that is not an official one "
                f"({urllib.parse.urlsplit(url).hostname})",
                hint="official hosts: "
                + ", ".join(ALLOWED_API_HOSTS.get(provider, ()))
                + "; for a private deployment pass it explicitly: --option api_url=https://...",
            )
        return url
    return require_https(default.rstrip("/"), f"{service} API URL")


# -- credentials --------------------------------------------------------------------------------


def secret(var: str) -> str | None:
    """Value of the credential ``var`` (named by its environment variable). With the Frontier
    SDK installed it is resolved through ``frontier.auth`` (job binding, the secret store
    filled by ``frontier auth login``, the environment, the legacy keyring entry); without it,
    the environment variable, else the keyring entry ``frontier-connectors`` / ``var``."""
    return credentials.secret(var)


def redact(text: str, *secrets: str | None) -> str:
    """Remove secrets from a message (defence in depth: keys are never formatted on purpose)."""
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


@contextlib.contextmanager
def redacted_errors(*secrets: str | None) -> Iterator[None]:
    """Scrub ``secrets`` from every :class:`ConnectorError` leaving the block."""
    try:
        yield
    except ConnectorError as e:
        e.message = redact(e.message, *secrets)
        e.hint = redact(e.hint, *secrets) if e.hint else e.hint
        e.args = (e.message,)
        raise


@contextlib.contextmanager
def sdk_environment(creds: Mapping[str, str]) -> Iterator[None]:
    """Vendor SDKs read their key from the environment themselves. A key that came from the
    keyring is exported for the duration of the SDK calls and removed again afterwards."""
    added = [k for k, v in creds.items() if v and not os.environ.get(k)]
    for k in added:
        os.environ[k] = creds[k]
    try:
        yield
    finally:
        for k in added:
            os.environ.pop(k, None)


# -- transport ----------------------------------------------------------------------------------


@dataclass
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    #: when set the body is streamed into this file instead of being returned
    sink: Path | None = None
    timeout: float = 120.0


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


Transport = Callable[[HttpRequest], HttpResponse]


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def require_https(url: str, what: str = "API URL") -> str:
    """``url`` unchanged when it is an https URL with a host; raises otherwise."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise ConnectorError(
            f"the {what} must be an https URL (got {parts.scheme or 'no scheme'}://...)",
            hint="cloud APIs are only contacted over https, so that the API key is never sent "
            "in clear text",
        )
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to https, and drop credential headers when the host changes
    (file downloads are often redirected to pre-signed object-storage URLs)."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "redirect to a non-https URL", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(newurl) != _origin(req.full_url):
            for store in (new.headers, new.unredirected_hdrs):
                for name in [k for k in store if k.lower() in SECRET_HEADERS]:
                    del store[name]
        return new


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """The only function of this module that touches the network (standard library only)."""
    require_https(request.url, "request URL")
    require_online(f"contacting {_origin(request.url)}", url=request.url)
    req = urllib.request.Request(  # noqa: S310 - scheme checked above
        request.url, method=request.method, headers=request.headers
    )
    opener = urllib.request.build_opener(_SafeRedirect())
    try:
        with opener.open(req, timeout=request.timeout) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            if request.sink is None:
                return HttpResponse(r.status, headers, r.read())
            tmp = request.sink.with_name(request.sink.name + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(request.sink)
            return HttpResponse(r.status, headers)
    except urllib.error.HTTPError as e:
        headers = {k.lower(): v for k, v in (e.headers or {}).items()}
        try:
            body = e.read()[:2000]
        except (OSError, ValueError):
            body = b""
        return HttpResponse(e.code, headers, body)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ConnectorError(
            f"could not reach {_origin(request.url)}: {getattr(e, 'reason', e)}",
            hint="check the network connection and the API URL; already downloaded results can "
            "be converted offline by pointing at the download directory",
        ) from e


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class ApiClient:
    """Authenticated JSON/download helper on top of a transport. The credential headers are
    only sent to the configured API origin, never to other hosts a response may point at."""

    def __init__(
        self,
        base_url: str,
        auth: Mapping[str, str],
        transport: Transport,
        service: str,
        *,
        key_env: str = "",
        retries: int = 2,
    ) -> None:
        self.base_url = require_https(base_url.rstrip("/"), f"{service} API URL")
        self._auth = dict(auth)
        self.transport = transport
        self.service = service
        self.key_env = key_env
        self.retries = retries

    def _headers(self, url: str) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "frontier-connectors"}
        if _origin(url) == _origin(self.base_url):
            headers.update(self._auth)
        return headers

    def url(self, path: str, query: Mapping[str, Any] | None = None) -> str:
        if "://" in path:
            url = require_https(path, f"{self.service} URL")
        else:
            url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query, doseq=True)
        return url

    def _send(self, request: HttpRequest) -> HttpResponse:
        attempt = 0
        # also with an injected transport: no request leaves in offline mode
        require_online(f"contacting the {self.service} API", url=request.url)
        while True:
            response = self.transport(request)
            if response.status not in RETRY_STATUS or attempt >= self.retries:
                return response
            attempt += 1
            try:
                wait = float(response.headers.get("retry-after", ""))
            except ValueError:
                wait = float(2 ** (attempt - 1))
            _sleep(max(0.0, min(wait, 30.0)))

    def _check(self, response: HttpResponse, url: str) -> None:
        status = response.status
        if status < 400:
            return
        path = urllib.parse.urlsplit(url).path
        said = redact(response.body.decode("utf-8", "replace")[:300], *self._auth.values())
        said = f"; server said: {said}" if said.strip() else ""
        key = self.key_env or "the API key"
        if status == 401:
            raise NeedsCredentialsError(
                f"{self.service} rejected the API key (HTTP 401) for {path}",
                hint=f"{key} is wrong, expired or revoked; create a new key and export it{said}",
            )
        if status == 403:
            raise NeedsCredentialsError(
                f"{self.service} denied access (HTTP 403) to {path}",
                hint=f"{key} is valid but may not read this resource: it belongs to another "
                f"account, workspace or region, or the plan has no API access{said}",
            )
        if status == 404:
            raise ConnectorError(
                f"{self.service} found nothing (HTTP 404) at {path}",
                hint="check the ids (the importer source `list:...` shows what the account "
                f"holds) and the API URL / region{said}",
            )
        if status == 429:
            raise ConnectorError(
                f"{self.service} rate limit reached (HTTP 429) for {path}",
                hint=f"wait a minute and run the command again; finished downloads are kept{said}",
            )
        if status >= 500:
            raise ConnectorError(
                f"{self.service} failed (HTTP {status}) for {path}",
                hint=f"the service has a problem, retry later; finished downloads are kept{said}",
            )
        raise ConnectorError(
            f"{self.service} returned HTTP {status} for {path}",
            hint=f"check the ids and options of the request{said}",
        )

    def get_json(self, path: str, query: Mapping[str, Any] | None = None) -> Any:
        url = self.url(path, query)
        response = self._send(HttpRequest("GET", url, self._headers(url)))
        self._check(response, url)
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ConnectorError(
                f"{self.service} sent a non-JSON answer for {urllib.parse.urlsplit(url).path}",
                hint="the API URL may be wrong (regional platform?) or the service is down",
            ) from e

    def download(self, url: str, dest: Path) -> Path:
        url = self.url(url)
        headers = self._headers(url)
        headers["Accept"] = "*/*"
        response = self._send(HttpRequest("GET", url, headers, sink=dest, timeout=3600.0))
        self._check(response, url)
        if not dest.is_file():
            raise ConnectorError(
                f"download of {dest.name} from {self.service} produced no file",
                hint="retry; if it persists download the results in the web UI and point the "
                "connector at that directory",
            )
        return dest


def planned_get(
    url: str,
    purpose: str,
    *,
    params: Mapping[str, Any] | None = None,
    auth: Mapping[str, str] | None = None,
) -> PlannedCall:
    """A planned GET. ``auth`` maps header names to the environment variable they are filled
    from (or a template such as ``"Token $VARIABLE"``); the plan shows the name, never a value."""
    headers = {n: (v if "$" in v else f"${v}") for n, v in (auth or {}).items()}
    return PlannedCall("GET", url, purpose, dict(params or {}), headers)


def safe_id(value: Any, key: str, hint: str = "") -> str:
    """``value`` when it is a plain identifier that can be put into a URL path."""
    if not isinstance(value, str) or not value:
        raise ConnectorError(f"no '{key}' given", hint=hint or f"set '{key}'")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", value) or set(value) <= {"."}:
        raise ConnectorError(f"'{key}' contains unexpected characters", hint=hint or None)
    return value


def as_list(value: Any) -> list[str]:
    """``"a,b"``, ``["a", "b"]`` or ``None`` as a list of strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


# -- reference files and sidecar ----------------------------------------------------------------


def _reference_suffixes(provider: str) -> tuple[str, ...]:
    return (f".{provider}.json", f".{provider}.toml", f".{provider}")


def load_reference(path: Path, provider: str | None = None) -> dict[str, Any] | None:
    """Parse a reference file. ``None`` when ``path`` is not one (never raises for arbitrary
    files; a malformed file with a reference name raises :class:`ConnectorError`)."""
    name = path.name.lower()
    named = [p for p in CLOUD_IDS if name.endswith(_reference_suffixes(p))]
    generic = name.endswith((".json", ".toml")) and not named
    if not named and not generic:
        return None
    try:
        if path.stat().st_size > 1 << 16:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    doc: Any
    try:
        if name.endswith(".toml"):
            import tomllib

            doc = tomllib.loads(text)
        elif name.endswith(".json"):
            doc = json.loads(text)
        else:
            m = re.match(r"^\s*([a-z0-9]+)://([^\s?#]+)(?:\?(\S*))?\s*$", text)
            if not m:
                raise ValueError("expected one line '<provider>://<id>'")
            doc = {"provider": m.group(1), "id": m.group(2).strip("/")}
            for k, v in urllib.parse.parse_qsl(m.group(3) or ""):
                doc[k] = v.split(",") if k in ("outputs", "include", "exclude") else v
    except ValueError as e:  # json.JSONDecodeError and tomllib.TOMLDecodeError are ValueErrors
        if named:
            raise ConnectorError(
                f"{path} is not a valid cloud reference file: {e}",
                hint='expected e.g. {"provider": "flow360", "case_id": "...", '
                '"outputs": ["surface", "volume"]}',
            ) from e
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("schema") == SIDECAR_SCHEMA:
        return None
    declared = str(doc.get("provider") or (named[0] if named else "")).lower()
    if declared not in CLOUD_IDS or (provider is not None and declared != provider):
        return None
    doc["provider"] = declared
    return _strip_caller_keys(path, doc, declared)


def _strip_caller_keys(path: Path, doc: dict[str, Any], provider: str) -> dict[str, Any]:
    """Drop the keys only a caller may set (:data:`FILE_IGNORED_KEYS`) from a reference file.

    An ``api_url`` on a host that is not an official one of the provider is an attempt to
    send the API key elsewhere: it raises instead of being skipped quietly."""
    url = doc.get("api_url")
    if url and not api_host_allowed(provider, str(url)):
        host = urllib.parse.urlsplit(str(url)).hostname or "?"
        raise ConnectorError(
            f"{path}: the reference file sets 'api_url' to a host that is not an official "
            f"{provider} API host ({host}); it was not contacted",
            hint="a reference file never chooses the API endpoint. Remove 'api_url' from the "
            "file; for a regional or private platform pass --option api_url=https://... "
            "yourself. Do not convert reference files from sources you do not trust.",
        )
    for key in FILE_IGNORED_KEYS:
        if key in doc:
            doc.pop(key)
            warnings.warn(
                f"{path}: '{key}' in a cloud reference file is ignored (the caller chooses "
                f"it: --option {key}=...)",
                CloudReferenceWarning,
                stacklevel=3,
            )
    return doc


def load_sidecar(directory: Path) -> dict[str, Any] | None:
    p = directory / SIDECAR
    try:
        doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None
    if isinstance(doc, dict) and doc.get("schema") == SIDECAR_SCHEMA:
        return doc
    return None


def write_sidecar(directory: Path, doc: Mapping[str, Any]) -> None:
    out = {"schema": SIDECAR_SCHEMA, "schema_version": 1, **doc}
    (directory / SIDECAR).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def cache_root() -> Path:
    """Where downloaded cloud results are cached: ``<storage.cache_dir>/cloud`` when the
    Frontier SDK is installed and the setting is set (``FRONTIER_CONNECTORS_CACHE`` is its
    legacy variable), else ``$FRONTIER_CONNECTORS_CACHE/cloud`` or
    ``~/.cache/frontier-connectors/cloud``."""
    root = credentials.setting("storage.cache_dir") or os.environ.get("FRONTIER_CONNECTORS_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "frontier-connectors"
    return base / "cloud"


# -- archives -----------------------------------------------------------------------------------


def extract_tar(tar: Any, dest: Path) -> int:
    """Extract an open tar (also a streaming one) into ``dest`` with
    :func:`frontier_connectors.archives.safe_extract_tar` (default caps). Returns the number
    of files written; a member that is a link, a device or leaves ``dest`` refuses the
    whole archive."""
    return archives.safe_extract_tar(tar, dest, name=str(getattr(tar, "name", "") or "archive"))


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract ``archive`` once into ``dest`` (a marker file records completion)."""
    marker = dest / ".extracted"
    if marker.is_file():
        return dest
    archives.safe_extract(archive, dest)
    marker.write_text("ok")
    return dest


def download_target(dest: Path, name: str, *, service: str) -> Path:
    """Local path for a file name a provider API returned, confined to the download
    directory ``dest`` (no absolute names, no ``..``, no symlink out of it). Backslashes are
    path separators too: the name may come from a Windows job."""
    text = str(name).replace("\\", "/")
    if any(part == ".." for part in text.split("/")):
        raise ReferenceEscapeError(
            case_file=dest,
            reference=str(name),
            case_root=dest,
            what=f"{service} file name",
            reason="contains a '..' component",
        )
    return confine(dest, text, what=f"{service} file name")


def _archive_stem(path: Path) -> str:
    name = path.name
    for ext in ARCHIVE_EXT:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem


# -- delegation ---------------------------------------------------------------------------------


@dataclass
class Entry:
    """One result data set inside a downloaded directory and the connector that reads it."""

    label: str
    path: Path
    connector: Any


def _is_archive(p: Path) -> bool:
    return p.is_file() and p.name.lower().endswith(ARCHIVE_EXT)


def _resolve_format(path: Path) -> Any | None:
    """Best non-cloud connector for ``path`` (``None`` when nothing recognises it)."""
    from frontier_connectors import registry

    try:
        found = registry.detect(path)
    except ConnectorError:
        return None
    for det in found:
        if det.connector_id not in CLOUD_IDS and det.confidence >= 0.5 and not det.needs:
            return registry.get(det.connector_id)
    return None


def _scan(directory: Path, tables: bool) -> list[Path]:
    """Result entry points below ``directory``: collection files win over their pieces."""
    found: list[Path] = []
    try:
        items = sorted(directory.iterdir())
    except OSError:
        return found
    files = [p for p in items if p.is_file() and not p.name.startswith(".")]
    dirs = [p for p in items if p.is_dir() and not p.name.startswith(".")]
    collections = [p for p in files if p.name.lower().endswith(COLLECTION_EXT)]
    skip_dirs: set[str] = set()
    if collections:
        found += collections
        # a collection keeps its pieces next to it or in a folder named after it
        skip_dirs = {p.stem for p in collections}
    else:
        singles = [p for p in files if p.name.lower().endswith(RESULT_EXT)]
        numbered = [p for p in singles if re.search(r"\d+$", p.stem)]
        if len(singles) > 1 and len(numbered) == len(singles):
            found.append(directory)  # a numbered family: let the format connector see the series
        else:
            found += singles
    if tables:
        found += [p for p in files if p.name.lower().endswith(TABLE_EXT)]
    for d in dirs:
        if d.name not in skip_dirs:
            found += _scan(d, tables)
    return found


def find_entries(directory: Path, *, tables: bool = False) -> list[Entry]:
    """Unpack result archives and map every data set under ``directory`` to a connector."""
    if not directory.is_dir():
        raise UnsupportedSourceError(
            f"{directory} is not a directory", hint="point at a downloaded result directory"
        )
    for archive in sorted(p for p in directory.rglob("*") if _is_archive(p)):
        if EXTRACT_DIR in archive.relative_to(directory).parts:
            continue
        extract_archive(archive, directory / EXTRACT_DIR / _archive_stem(archive))
    whole = _resolve_format(directory)
    if whole is not None:
        return [Entry("results", directory, whole)]
    entries: list[Entry] = []
    seen: set[str] = set()
    for path in _scan(directory, tables):
        connector = _resolve_format(path)
        if connector is None:
            continue
        rel = path.relative_to(directory)
        parts = [p for p in rel.parts if p != EXTRACT_DIR]
        label = meshutil.sanitize_id(
            Path(*parts).with_suffix("").as_posix() if parts else "results"
        )
        # an archive "surfaces.tar.gz" holding "surfaces.pvtu" should just be called "surfaces"
        halves = label.split("_")
        if len(halves) == 2 and halves[0] == halves[1]:
            label = halves[0]
        while label in seen:
            label += "_"
        seen.add(label)
        entries.append(Entry(label, path, connector))
    return entries


def _value(v: Any) -> Any:
    """Unwrap ``{"value": x, "units": ...}`` objects used by simulation parameter schemas."""
    if isinstance(v, Mapping) and "value" in v:
        return v["value"]
    return v


def _num(v: Any) -> float | None:
    v = _value(v)
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def safe_join(dest: Path, name: str) -> Path | None:
    """Kept for plugins written against 0.2: :func:`download_target` that returns ``None``
    instead of raising."""
    try:
        return download_target(dest, name, service="provider")
    except ConnectorError:
        return None


def jsonable(obj: Any, *, limit: int = 2 << 20) -> Any:
    """``obj`` as plain JSON data, or ``None`` when it is not serialisable or too large."""
    try:
        text = json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return None
    return json.loads(text) if len(text) <= limit else None


# -- base connector -----------------------------------------------------------------------------

#: option names accepted with or without the ``_id`` suffix (``--option project=...``)
_ID_ALIASES = {
    "project": "project_id",
    "simulation": "simulation_id",
    "run": "run_id",
    "job": "job_id",
    "case": "case_id",
    "mesh": "mesh_id",
}
_LIST_KEYS = ("outputs", "include", "exclude")
_CLOUD_KEYS = (
    "provider", "case_id", "simulation_id", "run_id", "project_id", "job_id", "mesh_id",
    "solution_id", "name", "status", "analysis", "api_url", "downloaded_at",
)  # fmt: skip


class _CloudConnector(BaseConnector):
    """Shared behaviour: reference file -> download -> delegation; directory -> delegation.

    A provider implements ``_plan`` (the calls, for ``--dry-run``), ``_list`` (projects, cases,
    runs ...) and ``_download`` (fetch into a directory, return the sidecar document)."""

    vendor = ""
    maturity = MATURITY
    temporal = ["steady", "transient"]
    kinds = ["surface", "mesh"]
    solver_name = "unknown"
    #: environment variables that must all be set for the live path
    env_vars: tuple[str, ...] = ()
    key_help = ""
    id_key = "id"
    reference_help = ""
    #: what the importer source means, and what ``list:<kind>`` accepts
    source_help = ""
    list_kinds: tuple[str, ...] = ()
    #: documentation the API shape was taken from, with the date it was read
    api_reference = ""

    def __init__(self, transport: Transport | None = None) -> None:
        #: every REST call goes through this function; tests replace it with a fake
        self.transport: Transport = transport or urllib_transport

    # -- detection ----------------------------------------------------------------------------
    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            if path.is_dir():
                side = load_sidecar(path)
                if side is not None and side.get("provider") == self.id:
                    return Detection(
                        self.id, path, 0.97, f"{self.id}-download", [f"{SIDECAR} sidecar"]
                    )
                note = self._native_layout(path)
                if note:
                    return Detection(self.id, path, 0.8, f"{self.id}-download", [note])
                return None
            if not path.is_file():
                return None
            try:
                ref = load_reference(path, self.id)
            except ConnectorError:
                ref = {"provider": self.id}  # named like a reference file but malformed
            if ref is None:
                return None
            needs = [] if self._cached(path, ref) else [self._needs()]
            return Detection(self.id, path, 0.95, f"{self.id}-reference", ["reference file"], needs)
        except OSError:
            return None

    def _native_layout(self, directory: Path) -> str | None:
        return None

    def _needs(self) -> str:
        return f"{' + '.join(self.env_vars)} in the environment (the results are downloaded)"

    # -- credentials --------------------------------------------------------------------------
    def credentials(self) -> dict[str, str]:
        """Values of :attr:`env_vars`; raises naming the first missing variable."""
        out: dict[str, str] = {}
        for var in self.env_vars:
            value = secret(var)
            if not value:
                raise NeedsCredentialsError(
                    f"{self.name}: environment variable {var} is not set",
                    hint=f"{self.key_help} Then {credentials.login_hint(var)} (never put the "
                    "key in a file of the repository). Already downloaded results need no key: "
                    "point the connector at the result directory instead; `--dry-run` shows the "
                    "calls without a key.",
                )
            out[var] = value
        return out

    # -- reference / cache --------------------------------------------------------------------
    def _run_id(self, ref: Mapping[str, Any]) -> str:
        value = ref.get(self.id_key) or ref.get("id")
        if not value or not isinstance(value, str):
            raise ConnectorError(
                f"the {self.id} reference file has no '{self.id_key}'",
                hint=self.reference_help,
            )
        return safe_id(value, self.id_key, self.reference_help)

    def _download_dir(
        self, ref_path: Path, ref: Mapping[str, Any], options: Mapping[str, Any] | None = None
    ) -> Path:
        """Where the results of a reference file are downloaded: the caller's
        ``options["download_dir"]`` (relative to the working directory), else the cache.
        Never a value from the reference file."""
        custom = (options or {}).get("download_dir")
        if custom:
            return Path(str(custom)).expanduser()
        return cache_root() / self.id / meshutil.sanitize_id(self._run_id(ref))

    def _cached(
        self, ref_path: Path, ref: Mapping[str, Any], options: Mapping[str, Any] | None = None
    ) -> bool:
        try:
            side = load_sidecar(self._download_dir(ref_path, ref, options))
        except ConnectorError:
            return False
        return side is not None and bool(side.get("complete"))

    def _resolve(self, path: Path, options: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
        """Local result directory and its sidecar document for a reference file or directory."""
        path = Path(path)
        if path.is_dir():
            side = load_sidecar(path) or self._native_sidecar(path)
            return path, side
        if not path.is_file():
            raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
        ref = load_reference(path, self.id)
        if ref is None:
            raise UnsupportedSourceError(
                f"{path} is not a {self.name} reference file or result directory",
                hint=self.reference_help,
            )
        if options.get("api_url"):  # the caller's explicit endpoint (never the file's)
            ref = {**ref, "api_url": str(options["api_url"])}
        dest = self._download_dir(path, ref, options)
        side = load_sidecar(dest)
        if side is not None and side.get("complete") and not options.get("refresh"):
            return dest, side
        return dest, self.fetch(ref, dest, options)

    def _native_sidecar(self, directory: Path) -> dict[str, Any]:
        return {"provider": self.id}

    # -- the live path: plan, list, fetch -------------------------------------------------------
    def reference(self, source: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The reference document for an importer ``source``: ``list:<kind>``, the path of a
        reference file, or the provider's id syntax (:attr:`source_help`). ``options`` are
        merged in (``outputs=surface,volume``, ``project=<id>`` ...)."""
        src = str(source).strip()
        if src.lower().startswith("list:"):
            ref: dict[str, Any] = {"provider": self.id, "list": src[5:].strip().lower()}
        elif len(src) < 1024 and Path(src).is_file():
            loaded = load_reference(Path(src), self.id)
            if loaded is None:
                raise UnsupportedSourceError(
                    f"{src} is not a {self.name} reference file", hint=self.reference_help
                )
            ref = loaded
        else:
            prefix = f"{self.id}://"
            src = src[len(prefix) :] if src.lower().startswith(prefix) else src
            ref = {"provider": self.id, **self.parse_source(src.strip("/"))}
        for key, value in (options or {}).items():
            key = _ID_ALIASES.get(key, key)
            if value is None or key == "mapping":
                continue
            ref[key] = str(value) if key.endswith("_id") else value
        for key in _LIST_KEYS:
            if key in ref:
                ref[key] = as_list(ref[key])
        return ref

    def parse_source(self, source: str) -> dict[str, Any]:
        return {self.id_key: safe_id(source, self.id_key, f"expected {self.source_help}")}

    def _list_kind(self, ref: Mapping[str, Any]) -> str:
        what = str(ref.get("list") or "")
        if what not in self.list_kinds:
            raise ConnectorError(
                f"{self.name} cannot list {what!r}",
                hint="use one of: " + ", ".join(f"list:{k}" for k in self.list_kinds),
            )
        return what

    def plan_calls(self, ref: Mapping[str, Any]) -> list[PlannedCall]:
        """The calls ``fetch`` (or a listing) would make. Needs no credentials, contacts
        nothing; credentials appear as ``$VARIABLE``."""
        return self._plan(ref, self._list_kind(ref) if "list" in ref else None)

    def list_remote(self, what: str, ref: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Rows (``id``, ``name``, ...) of what the account holds: see :attr:`list_kinds`."""
        ref = {**(ref or {}), "list": what}
        kind = self._list_kind(ref)
        require_online(f"listing {what} of the {self.name} account")
        creds = self.credentials()
        with redacted_errors(*creds.values()):
            return self._list(kind, ref, creds)

    def fetch(
        self, ref: Mapping[str, Any], dest: Path, options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Download the run named by ``ref`` into ``dest`` and write the sidecar."""
        run_id = self._run_id(ref)
        require_online(f"downloading {self.name} results")
        creds = self.credentials()
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with redacted_errors(*creds.values()):
            doc = self._download(ref, dest, creds, dict(options or {}))
        doc = {
            "provider": self.id,
            self.id_key: run_id,
            **doc,
            "api_reference": self.api_reference,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": True,
        }
        text = json.dumps(doc, default=str)
        if any(v in text for v in creds.values()):  # never happens; checked all the same
            raise ConnectorError(
                f"{self.name}: refusing to write a sidecar that contains a credential",
                hint="this is a bug in the connector, please report it",
            )
        write_sidecar(dest, doc)
        return doc

    def _plan(self, ref: Mapping[str, Any], listing: str | None) -> list[PlannedCall]:
        raise NotImplementedError

    def _list(
        self, what: str, ref: Mapping[str, Any], creds: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _download(
        self,
        ref: Mapping[str, Any],
        dest: Path,
        creds: Mapping[str, str],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    # -- metadata -----------------------------------------------------------------------------
    def _metadata(self, side: Mapping[str, Any]) -> dict[str, Any]:
        """Provider metadata for samples: ids, solver version and the settings the provider
        mapped at download time (``physics``, ``parameters``, ``boundary_conditions``,
        ``mesh``)."""
        cloud = {k: side[k] for k in _CLOUD_KEYS if side.get(k) not in (None, "", [])}
        meta: dict[str, Any] = {"cloud": cloud}
        if side.get("solver_version"):
            meta["solver_version"] = str(side["solver_version"])
        for key in ("physics", "parameters", "boundary_conditions", "mesh"):
            value = side.get(key)
            if isinstance(value, Mapping) and value:
                meta[key] = dict(value)
        return meta

    def _entry_metadata(self, entry: Entry, directory: Path, side: Mapping[str, Any]) -> dict:
        """Extra metadata for the samples of one data set (e.g. the variables of its run)."""
        return {}

    def _entries(self, directory: Path, side: Mapping[str, Any]) -> list[Entry]:
        outputs = [str(o).lower() for o in side.get("outputs") or []]
        entries = find_entries(directory, tables=any(o in ("tables", "csv") for o in outputs))
        if not entries:
            raise ConnectorError(
                f"no readable result files found in {directory}",
                hint="expected VTK/Tecplot/CGNS/EnSight output (possibly inside .tar.gz/.zip); "
                "check which output formats the run wrote, or download them in the web UI",
            )
        return entries

    @staticmethod
    def _outer(entry: Entry, zone: str, n_entries: int) -> str:
        if n_entries == 1:
            return zone
        return entry.label if zone in ("main", "", entry.label) else f"{entry.label}/{zone}"

    # -- describe / read ----------------------------------------------------------------------
    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        opts = dict(options or {})
        directory, side = self._resolve(Path(path), opts)
        entries = self._entries(directory, side)
        meta = self._metadata(side)
        desc = SourceDescription(
            connector_id=self.id,
            path=Path(path),
            format=f"{self.id}-download",
            solver=self.solver_name,
            solver_version=meta.get("solver_version"),
            domain="cfd",
        )
        fields: dict[tuple[str, str], FieldInfo] = {}
        delegated: dict[str, str] = {}
        for entry in entries:
            sub = entry.connector.describe(entry.path)
            delegated[entry.label] = sub.connector_id
            names = {}
            for z in sub.zones:
                names[z.name] = self._outer(entry, z.name, len(entries))
                z.name = names[z.name]
                desc.zones.append(z)
            for f in sub.fields:
                key = (f.name, f.location)
                zones = [names.get(z, z) for z in f.zones] if f.zones else list(names.values())
                if key in fields:
                    fields[key].zones = sorted(set((fields[key].zones or []) + zones))
                else:
                    f.zones = zones
                    fields[key] = f
            if len(sub.time.values) > len(desc.time.values):
                desc.time = sub.time
            desc.warnings += sub.warnings
            if sub.solver not in ("", "unknown") and self.solver_name == "unknown":
                desc.solver, desc.solver_version = sub.solver, sub.solver_version
            if sub.domain != "other":
                desc.domain = sub.domain
            for k, v in sub.units.items():
                desc.units.setdefault(k, v)
            desc.physics = {**sub.physics, **desc.physics}
            desc.boundary_conditions = {**sub.boundary_conditions, **desc.boundary_conditions}
            desc.parameters = {**sub.parameters, **desc.parameters}
        all_zones = {z.name for z in desc.zones}
        for f in fields.values():
            if f.zones is not None and set(f.zones) == all_zones:
                f.zones = None
        desc.fields = list(fields.values())
        desc.physics = {**desc.physics, **meta.get("physics", {})}
        desc.boundary_conditions = {
            **desc.boundary_conditions,
            **meta.get("boundary_conditions", {}),
        }
        desc.parameters = {**desc.parameters, **meta.get("parameters", {})}
        desc.warnings += [str(w) for w in side.get("warnings") or []]
        desc.metadata = {
            "cloud": meta["cloud"],
            "delegated_to": delegated,
            "directory": str(directory),
        }
        if meta.get("mesh"):
            desc.metadata["mesh"] = meta["mesh"]
        return desc

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        opts = dict(options or {})
        directory, side = self._resolve(Path(path), opts)
        entries = self._entries(directory, side)
        meta = self._metadata(side)
        base = meshutil.default_sample_id(Path(path))
        for suffix in (f".{self.id}",):
            base = base[: -len(suffix)] if base.endswith(suffix) else base
        plan: list[tuple[Entry, dict[str, str]]] = []
        for entry in entries:
            sub = entry.connector.describe(entry.path)
            plan.append(
                (entry, {z.name: self._outer(entry, z.name, len(entries)) for z in sub.zones})
            )
        wanted: set[str] | None = None
        if zones is not None:
            wanted = set(match_names([o for _, m in plan for o in m.values()], zones, what="zone"))
        for entry, names in plan:
            inner = None if wanted is None else [i for i, o in names.items() if o in wanted]
            if inner is not None and not inner:
                continue
            own_base = meshutil.default_sample_id(entry.path)
            prefix = base if len(entries) == 1 else f"{base}__{entry.label}"
            own = {**meta, **self._entry_metadata(entry, directory, side)}
            for s in entry.connector.read(entry.path, zones=inner, fields=fields, times=times):
                tail = s.id[len(own_base) :] if s.id.startswith(own_base) else f"__{s.id}"
                s.id = prefix + tail
                if s.zone is not None:
                    s.zone = names.get(s.zone, self._outer(entry, s.zone, len(entries)))
                stamped = dict(s.metadata)
                if self.solver_name != "unknown" or stamped.get("solver") in (None, "", "unknown"):
                    stamped["solver"] = self.solver_name
                for key in ("solver_version", "cloud", "mesh"):
                    if own.get(key):
                        stamped[key] = own[key]
                for key in ("physics", "parameters", "boundary_conditions"):
                    if own.get(key):
                        stamped[key] = {**(stamped.get(key) or {}), **own[key]}
                stamped["source"] = str(path)
                s.metadata = stamped
                yield s


# -- importer -----------------------------------------------------------------------------------


class CloudImporter(BaseImporter):
    """The importer of a cloud provider: a thin shell around its connector, so that
    ``import <id> <source>`` and a reference file run the same code.

    ``source`` is the provider's id syntax (``source_help``), the path of a reference file, or
    ``list:<kind>``: a listing is written to ``listing.json`` in the download directory (use
    it without ``--out``; ``python -m frontier_connector_<id> list <kind>`` prints it)."""

    connector_class: ClassVar[type[_CloudConnector]]
    maturity = MATURITY

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        c = cls.connector_class
        cls.id = c.id
        cls.name = c.name
        cls.vendor = c.vendor
        cls.license = c.license
        cls.credentials_env = list(c.env_vars)
        cls.requires = list(c.requires)
        cls.source_help = c.source_help
        cls.status_note = c.status_note
        cls.api_reference = c.api_reference

    def __init__(self, transport: Transport | None = None) -> None:
        self.connector = self.connector_class(transport)

    def credentials(self) -> dict[str, str]:
        return self.connector.credentials()

    def plan(self, source: str, *, options: Mapping[str, Any] | None = None) -> list[PlannedCall]:
        return self.connector.plan_calls(self.connector.reference(source, options))

    def list_remote(self, what: str, **ids: Any) -> list[dict[str, Any]]:
        """``list_remote("cases", project_id="prj-...")``: what the account holds."""
        return self.connector.list_remote(what, self.connector.reference(f"list:{what}", ids))

    def fetch(self, source: str, dest: Path, *, options: Mapping[str, Any] | None = None) -> Path:
        ref = self.connector.reference(source, options)
        dest = Path(dest)
        if "list" in ref:
            rows = self.connector.list_remote(str(ref["list"]), ref)
            dest.mkdir(parents=True, exist_ok=True)
            doc = {
                "schema": LISTING_SCHEMA,
                "provider": self.id,
                "list": ref["list"],
                "count": len(rows),
                "items": rows,
            }
            (dest / LISTING).write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            return dest
        self.connector.fetch(ref, dest, options)
        return dest


def main(importer_class: type[CloudImporter], argv: Sequence[str] | None = None) -> int:
    """``python -m frontier_connector_<id> list <kind> | plan <source>`` for one provider."""
    import argparse

    imp = importer_class()
    parser = argparse.ArgumentParser(
        prog=f"python -m frontier_connector_{imp.id}",
        description=f"{imp.name}: list what the account holds, or show the calls of an import. "
        f"Credentials: {', '.join(imp.credentials_env)} (environment).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    kinds = ", ".join(imp.connector.list_kinds)
    p = sub.add_parser("list", help=f"list {kinds}")
    p.add_argument("what", choices=imp.connector.list_kinds)
    p.add_argument("--dry-run", action="store_true", help="print the calls, contact nothing")
    q = sub.add_parser("plan", help="print the calls an import of <source> would make")
    q.add_argument("source", help=imp.source_help)
    for s in (p, q):
        s.add_argument("--option", action="append", metavar="KEY=VALUE", default=[])
        s.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    options = dict(item.partition("=")[::2] for item in args.option)
    try:
        if args.command == "plan" or args.dry_run:
            source = args.source if args.command == "plan" else f"list:{args.what}"
            calls = imp.plan(source, options=options)
            if args.json:
                print(json.dumps([c.to_dict() for c in calls], indent=2))
            else:
                print("\n".join(c.line() for c in calls))
            return 0
        rows = imp.list_remote(args.what, **options)
    except ConnectorError as e:
        print(f"error: {e}")
        return 2
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for r in rows:
            rest = "  ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "name") and v)
            print(f"{r.get('id', ''):<44} {r.get('name', '')!s:<32} {rest}")
        print(f"{len(rows)} {args.what}")
    return 0


def _import_sdk(module: str, package: str, purpose: str, licence: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingDependencyError(package, purpose=purpose, license=licence) from e


import_sdk = _import_sdk


def _accepted(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep the keyword arguments ``fn`` accepts (SDK signatures differ between releases)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


accepted = _accepted


# -- test support ---------------------------------------------------------------------------------


class ReplayTransport:
    """A transport for tests: answers from a table of exact URLs (query matched separately) and
    records every request. Unknown URLs answer 404, so a test fails on any call it did not
    describe; nothing ever reaches the network.

    .. code-block:: python

        t = ReplayTransport()
        t.json("https://api.example.com/v0/projects", {"_embedded": []}, query={"page": "1"})
        t.file("https://api.example.com/v0/files/1/contents/", b"...")
        connector = MyConnector(transport=t)
    """

    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []
        self._routes: list[tuple[str, dict[str, str], Callable[[HttpRequest], HttpResponse]]] = []

    def on(
        self,
        url: str,
        handler: Callable[[HttpRequest], HttpResponse],
        *,
        query: Mapping[str, Any] | None = None,
    ) -> None:
        self._routes.append((url, {k: str(v) for k, v in (query or {}).items()}, handler))

    def json(
        self, url: str, doc: Any, *, query: Mapping[str, Any] | None = None, status: int = 200
    ) -> None:
        body = json.dumps(doc).encode("utf-8")
        self.on(url, lambda req: HttpResponse(status, {}, body), query=query)

    def status(
        self,
        url: str,
        status: int,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
        *,
        query: Mapping[str, Any] | None = None,
    ) -> None:
        self.on(url, lambda req: HttpResponse(status, dict(headers or {}), body), query=query)

    def file(self, url: str, data: bytes) -> None:
        def handler(req: HttpRequest) -> HttpResponse:
            if req.sink is None:
                raise AssertionError("downloads must be streamed to disk")
            req.sink.parent.mkdir(parents=True, exist_ok=True)
            req.sink.write_bytes(data)
            return HttpResponse(200, {})

        self.on(url, handler)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        parts = urllib.parse.urlsplit(request.url)
        bare = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        got = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        best: Callable[[HttpRequest], HttpResponse] | None = None
        best_n = -1
        for url, query, handler in self._routes:
            if url == bare and all(got.get(k) == v for k, v in query.items()):
                if len(query) >= best_n:  # the most specific, then the latest, route wins
                    best, best_n = handler, len(query)
        if best is None:
            return HttpResponse(404, {}, b'{"message": "no such route in the test"}')
        return best(request)

    def paths(self) -> list[str]:
        """``"GET /path?query"`` of every request, in order."""
        out = []
        for r in self.requests:
            p = urllib.parse.urlsplit(r.url)
            out.append(f"{r.method} {p.path}" + (f"?{p.query}" if p.query else ""))
        return out
