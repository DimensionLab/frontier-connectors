# SPDX-License-Identifier: Apache-2.0
"""The plugin manifest (``plugin.toml``) and its validation.

Every plugin distribution ships one ``plugin.toml`` inside its import package and points at
that package from the entry-point group ``frontier.plugins``:

.. code-block:: toml

    schema = 1
    id = "fluent"                    # unique, lower-case, [a-z0-9-]
    kind = "connector"               # connector | exporter | importer
    name = "Ansys Fluent"
    summary = "CFF and legacy case/data files, live sessions through PyFluent"
    version = "0.2.0"                # must equal the distribution version
    data_model_version = "1.1"       # frontier-dataset version the plugin was written against
    vendor = "Ansys"
    formats = ["fluent-cff", "fluent-legacy"]
    maturity = "verified-public-sample"
    licence = "Apache-2.0"           # SPDX expression (`license` is accepted too)
    description = "Read Ansys Fluent case and data files"   # one line, shown on the card
    category = "solver"              # see CATEGORIES below
    tags = ["cfd", "ansys"]          # optional, free words for filtering
    keywords = ["cas.h5", "dat.h5"]  # optional, extra search terms
    icon = "icon.svg"                # optional, path inside the plugin folder
    homepage = "https://example.org"
    repository = "https://example.org/source"   # optional; leave out when there is none
    readme = "README.md"             # in the plugin folder; status and caveats go there
    updated = "2026-10-10"           # optional; the catalog takes the date from git history
    requires = ["h5py>=3.10"]        # pip requirements installed with the plugin
    plugins = ["ensight"]            # other Frontier plugins it delegates to
    suggests = ["exporter-fluent"]   # installed alongside by `plugins install` unless told not to

    [[optional]]                     # lazily imported extras; `plugins doctor` explains them
    package = "ansys-fluent-core"
    module = "ansys.fluent.core"
    licence = "MIT"
    purpose = "drive a live Fluent session"
    flag = "needs a licensed Ansys Fluent"

    [connectors]                     # mirrored into the entry-point group frontier.connectors
    fluent-cff = "frontier_connector_fluent.connector:FluentCffConnector"

    [importers]                      # -> frontier.importers
    fluent-live = "frontier_connector_fluent.live:FluentLiveImporter"

    [exporter]                       # kind = "exporter" only; -> frontier.exporters
    vendor = "fluent"
    tool = "Ansys Fluent"
    files = "files"                  # directory inside the package holding the scripts

    [author]                         # who publishes it; third parties use the same table
    name = "DimensionLab"
    company = "DimensionLab s.r.o."  # plugins of this company are listed as official
    email = "hello@dimensionlab.org"
    url = "https://dimensionlab.org"

    [[auth]]                         # a credential the plugin needs; several plugins may
    id = "flow360"                   # share one id ([a-z0-9_-], at most 64 characters)
    title = "Flexcompute Flow360"
    kind = "api_key"                 # see AUTH_KINDS below
    env = ["FLOW360_APIKEY"]         # accepted fallbacks of the first secret field
    help_url = "https://flow360.simulation.cloud/account"
    scopes_hint = "read access to the projects and cases to import"
    test = "frontier_connector_flow360.api:test_credential"   # optional "module:function"
    fields = [                       # default: one secret field, `api_key` or `token`
      { name = "api_key", title = "API key", secret = true },
      { name = "api_url", title = "API URL", secret = false, required = false,
        env = ["FLOW360_API_URL"] },
    ]

    [[auth]]                         # an OAuth sign-in (the flows run in `frontier auth login`)
    id = "acme-cloud"
    title = "Acme Cloud"
    kind = "oauth_device"            # or oauth_pkce
    [auth.oauth]
    token_url = "https://id.example.com/oauth/token"                 # always required
    device_code_url = "https://id.example.com/oauth/device/code"     # oauth_device
    authorize_url = "https://id.example.com/oauth/authorize"         # oauth_pkce
    client_id = "frontier-desktop"   # a public client: no secret
    scopes = ["read"]
    redirect_port = 8765             # optional, oauth_pkce loopback on 127.0.0.1
    audience = "https://api.example.com"                             # optional

An OAuth provider stores the fields ``access_token``, ``refresh_token`` and ``expires_at``
unless it declares its own.

``[[auth]]`` declares *what* a plugin needs, never a value. The plugin manager collects the
declarations of the enabled plugins into ``$FRONTIER_HOME/auth-providers.json`` (see
:mod:`frontier_connectors.plugins.auth`), where ``frontier auth``, the control plane and the
app find them; the values are provided with ``frontier auth login <id>`` and read through
:func:`frontier_connectors.credentials.secret`.

The manifest is read with :mod:`tomllib` straight from the package directory, *without
importing the plugin*, so a plugin with a broken import or a missing dependency can still be
listed, explained and uninstalled.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from frontier_connectors.model import MATURITIES

#: frontier-dataset version this framework writes (``manifest.json``: version 1, minor 1)
DATA_MODEL_VERSION = "1.1"
MANIFEST_SCHEMA = 1
MANIFEST_NAME = "plugin.toml"
KINDS: tuple[str, ...] = ("connector", "exporter", "importer")

#: entry-point groups
GROUP_PLUGINS = "frontier.plugins"
GROUP_CONNECTORS = "frontier.connectors"
GROUP_EXPORTERS = "frontier.exporters"
GROUP_IMPORTERS = "frontier.importers"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_AUTH_ID = re.compile(r"^[a-z0-9_-]{1,64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AUTH_KINDS: tuple[str, ...] = (
    "token",
    "api_key",
    "basic",
    "oauth_device",
    "oauth_pkce",
    "file",
    "none",
)
OAUTH_KINDS: tuple[str, ...] = ("oauth_device", "oauth_pkce")
_OAUTH_KEYS = (
    "authorize_url",
    "token_url",
    "device_code_url",
    "client_id",
    "scopes",
    "redirect_port",
    "audience",
)
_TARGET = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
#: marketplace categories (``category``); anything else is listed under ``other``
CATEGORIES: tuple[str, ...] = (
    "solver",
    "file-format",
    "cloud-service",
    "exporter",
    "importer",
    "other",
)
#: the company whose plugins Frontier lists as official. Derived from ``[author] company``,
#: never declared: a manifest cannot call itself official.
OFFICIAL_COMPANY = "DimensionLab s.r.o."
_AUTHOR_KEYS: tuple[str, ...] = ("name", "company", "email", "url")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


class ManifestError(ValueError):
    """``plugin.toml`` is missing, unreadable or invalid. ``problems`` lists every finding."""

    def __init__(self, problems: list[str], *, source: str = "") -> None:
        self.problems = problems
        self.source = source
        where = f"{source}: " if source else ""
        super().__init__(where + "; ".join(problems))


@dataclass
class OptionalDependency:
    """An optional package a plugin imports lazily (never needed at import time)."""

    package: str
    licence: str = ""
    purpose: str = ""
    #: import name when it differs from the pip name (``pyNastran``, ``ansys.fluent.core``)
    module: str = ""
    #: copyleft / proprietary / needs-vendor-licence note shown by ``plugins doctor``
    flag: str = ""


@dataclass
class ExporterSpec:
    """Vendor-side scripts shipped by an exporter plugin."""

    vendor: str
    tool: str = ""
    #: directory inside the plugin package that is copied by ``install-exporter``
    files: str = "files"
    #: how to run it, one line (shown after ``install-exporter``)
    usage: str = ""


@dataclass
class AuthField:
    """One value a credential needs. Non-secret fields (a user name, an API URL) are kept
    with the secret so one login is one unit."""

    name: str
    title: str = ""
    secret: bool = True
    required: bool = True
    #: environment variables accepted as a fallback for this field
    env: list[str] = field(default_factory=list)


@dataclass
class AuthDeclaration:
    """A credential a plugin needs (``[[auth]]``): which one and how to obtain it, never a
    value."""

    id: str
    title: str
    kind: str = "token"
    fields: list[AuthField] = field(default_factory=list)
    #: fallback variables of the first secret field (the common single-key case)
    env: list[str] = field(default_factory=list)
    help_url: str = ""
    #: ``module:function`` returning ``{"ok", "identity", "error"}``; optional
    test: str = ""
    scopes_hint: str = ""
    #: ``[auth.oauth]`` of an ``oauth_device`` / ``oauth_pkce`` provider, passed through
    #: unchanged: authorize_url, token_url, device_code_url, client_id, scopes, redirect_port,
    #: audience
    oauth: dict[str, Any] = field(default_factory=dict)

    def to_provider(self) -> dict[str, Any]:
        """The declaration as a provider table (the shape of ``auth-providers.json``), with
        the defaults filled in: a default field, field titles, and the provider-level ``env``
        on the first secret field."""
        fields = [asdict(f) for f in self.fields]
        if not fields and self.kind in OAUTH_KINDS:
            # the stored token set of an OAuth sign-in
            fields = [
                asdict(AuthField("access_token", "Access token")),
                asdict(AuthField("refresh_token", "Refresh token", required=False)),
                asdict(AuthField("expires_at", "Expires at", secret=False, required=False)),
            ]
        elif not fields and self.kind != "none":
            fields.append(asdict(AuthField("api_key" if self.kind == "api_key" else "token")))
        primary = next((i for i, f in enumerate(fields) if f["secret"]), 0)
        for i, f in enumerate(fields):
            f["title"] = f["title"] or f["name"].replace("_", " ")
            if i == primary:
                f["env"] += [e for e in self.env if e not in f["env"]]
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "fields": fields,
            "env": list(self.env),
        }
        for key in ("help_url", "test", "scopes_hint"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.oauth:
            out["oauth"] = dict(self.oauth)
        return out


def auth_providers(declarations: Any) -> list[dict[str, Any]]:
    """Provider tables for a list of ``[[auth]]`` declarations given as dictionaries (raw
    TOML tables, ``asdict`` output or provider tables: the result is the same)."""
    out = []
    for d in declarations if isinstance(declarations, list) else []:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        fields = [
            AuthField(
                name=str(f["name"]),
                title=str(f.get("title", "")),
                secret=f.get("secret", True) is True,
                required=f.get("required", True) is True,
                env=[str(e) for e in f.get("env", [])],
            )
            for f in d.get("fields", [])
            if isinstance(f, dict) and f.get("name")
        ]
        decl = AuthDeclaration(
            id=str(d["id"]),
            title=str(d.get("title", "")),
            kind=str(d.get("kind", "token")),
            fields=fields,
            env=[str(e) for e in d.get("env", [])],
            help_url=str(d.get("help_url", "")),
            test=str(d.get("test", "")),
            scopes_hint=str(d.get("scopes_hint", "")),
            oauth=dict(d["oauth"]) if isinstance(d.get("oauth"), dict) else {},
        )
        out.append(decl.to_provider())
    return out


@dataclass
class PluginManifest:
    id: str
    kind: str
    name: str
    version: str
    data_model_version: str
    vendor: str = ""
    summary: str = ""
    formats: list[str] = field(default_factory=list)
    maturity: str = "planned"
    licence: str = "Apache-2.0"
    requires: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    suggests: list[str] = field(default_factory=list)
    optional: list[OptionalDependency] = field(default_factory=list)
    connectors: dict[str, str] = field(default_factory=dict)
    importers: dict[str, str] = field(default_factory=dict)
    exporter: ExporterSpec | None = None
    auth: list[AuthDeclaration] = field(default_factory=list)
    homepage: str = ""
    #: one line for the marketplace card (falls back to ``summary``)
    description: str = ""
    #: ``name``, ``company``, ``email``, ``url``; all optional
    author: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    category: str = ""
    #: path of an image inside the plugin folder
    icon: str = ""
    repository: str = ""
    readme: str = "README.md"
    #: ISO date of the last change
    updated: str = ""
    schema: int = MANIFEST_SCHEMA

    @property
    def official(self) -> bool:
        """Published by DimensionLab (derived from the author's company)."""
        return self.author.get("company", "") == OFFICIAL_COMPANY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_version(text: str) -> tuple[int, int]:
    m = _VERSION.match(str(text).strip())
    if not m:
        raise ValueError(f"not a version: {text!r}")
    return int(m.group(1)), int(m.group(2))


def data_model_compatible(plugin: str, framework: str = DATA_MODEL_VERSION) -> str | None:
    """``None`` when a plugin written against ``plugin`` can run on ``framework``, else the
    reason. Same major version and a plugin minor that is not newer than the framework's:
    minor versions only add (``docs`` of frontier-dataset v1.1 are backward compatible with v1),
    so an older plugin keeps working and a newer one may emit things this writer cannot store.
    """
    try:
        p_major, p_minor = parse_version(plugin)
        f_major, f_minor = parse_version(framework)
    except ValueError as e:
        return str(e)
    if p_major != f_major:
        return (
            f"plugin targets frontier-dataset v{plugin}, this framework writes v{framework} "
            "(different major version)"
        )
    if p_minor > f_minor:
        return (
            f"plugin targets frontier-dataset v{plugin}, newer than this framework's "
            f"v{framework}: upgrade frontier-connectors"
        )
    return None


def _str_list(doc: dict[str, Any], key: str, problems: list[str]) -> list[str]:
    value = doc.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.append(f"`{key}` must be a list of strings")
        return []
    return list(value)


def _targets(doc: dict[str, Any], key: str, problems: list[str]) -> dict[str, str]:
    table = doc.get(key, {})
    if not isinstance(table, dict):
        problems.append(f'`[{key}]` must be a table of id = "module:attribute"')
        return {}
    out: dict[str, str] = {}
    for name, target in table.items():
        if not _ID.match(str(name)):
            problems.append(f"[{key}] id {name!r} must match [a-z0-9][a-z0-9-]*")
        if not isinstance(target, str) or not _TARGET.match(target):
            problems.append(f'[{key}] {name} = {target!r} is not "module:attribute"')
            continue
        out[str(name)] = target
    return out


def _auth(doc: dict[str, Any], problems: list[str]) -> list[AuthDeclaration]:
    raw = doc.get("auth", [])
    if not isinstance(raw, list):
        problems.append("`[[auth]]` must be an array of tables")
        return []
    out: list[AuthDeclaration] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        where = f"[[auth]] #{i + 1}"
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            problems.append(f"{where} needs an `id`")
            continue
        aid = item["id"]
        where = f"[[auth]] {aid}"
        before = len(problems)
        if not _AUTH_ID.match(aid):
            problems.append(f"{where}: id must match [a-z0-9_-]{{1,64}}")
        if aid in seen:
            problems.append(f"{where}: declared twice")
        seen.add(aid)
        unknown = sorted(set(item) - set(AuthDeclaration.__dataclass_fields__))
        if unknown:
            problems.append(f"{where}: unknown key(s) {', '.join(unknown)}")
        title = item.get("title")
        if not isinstance(title, str) or not title:
            problems.append(f"{where}: `title` is required")
        kind = item.get("kind", "token")
        if kind not in AUTH_KINDS:
            problems.append(f"{where}: kind {kind!r} must be one of {', '.join(AUTH_KINDS)}")
        texts = {k: item.get(k, "") for k in ("help_url", "test", "scopes_hint")}
        for key, value in texts.items():
            if not isinstance(value, str):
                problems.append(f"{where}: `{key}` must be a string")
        url = texts["help_url"]
        if isinstance(url, str) and url and not url.startswith("https://"):
            problems.append(f"{where}: help_url must be an https:// URL")
        target = texts["test"]
        if isinstance(target, str) and target and not _TARGET.match(target):
            problems.append(f'{where}: test = {target!r} is not "module:function"')
        env = _env_names(item.get("env", []), f"{where}: `env`", problems)
        fields: list[AuthField] = []
        raw_fields = item.get("fields", [])
        if not isinstance(raw_fields, list):
            problems.append(f"{where}: `fields` must be an array of tables")
            raw_fields = []
        for f in raw_fields:
            if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
                problems.append(f"{where}: every field needs a `name`")
                continue
            what = f"{where} field {f['name']}"
            extra = sorted(set(f) - set(AuthField.__dataclass_fields__))
            if extra:
                problems.append(f"{what}: unknown key(s) {', '.join(extra)}")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", f["name"]):
                problems.append(f"{what}: name must match [a-z][a-z0-9_]*")
            if f["name"] in {x.name for x in fields}:
                problems.append(f"{what}: declared twice")
            for flag in ("secret", "required"):
                if not isinstance(f.get(flag, True), bool):
                    problems.append(f"{what}: `{flag}` must be true or false")
            if not isinstance(f.get("title", ""), str):
                problems.append(f"{what}: `title` must be a string")
            fields.append(
                AuthField(
                    name=f["name"],
                    title=str(f.get("title", "")),
                    secret=f.get("secret", True) is True,
                    required=f.get("required", True) is True,
                    env=_env_names(f.get("env", []), f"{what}: `env`", problems),
                )
            )
        oauth = _oauth(item.get("oauth"), str(kind), where, problems)
        if kind == "none" and (fields or env):
            problems.append(f'{where}: kind "none" takes no fields and no env')
        if len(problems) == before:
            out.append(
                AuthDeclaration(
                    id=aid,
                    title=str(title),
                    kind=str(kind),
                    fields=fields,
                    env=env,
                    help_url=str(url),
                    test=str(target),
                    scopes_hint=str(texts["scopes_hint"]),
                    oauth=oauth,
                )
            )
    return out


def _oauth(raw: Any, kind: str, where: str, problems: list[str]) -> dict[str, Any]:
    """Validate ``[auth.oauth]``; the table is returned as written."""
    if raw is None:
        if kind in OAUTH_KINDS:
            problems.append(f"{where}: kind {kind!r} needs an `[auth.oauth]` table")
        return {}
    if not isinstance(raw, dict):
        problems.append(f"{where}: `oauth` must be a table")
        return {}
    if kind not in OAUTH_KINDS:
        problems.append(f"{where}: `[auth.oauth]` needs kind oauth_device or oauth_pkce")
    unknown = sorted(set(raw) - set(_OAUTH_KEYS))
    if unknown:
        problems.append(f"{where} oauth: unknown key(s) {', '.join(unknown)}")
    for key in ("authorize_url", "token_url", "device_code_url"):
        value = raw.get(key)
        if value is not None and not (isinstance(value, str) and value.startswith("https://")):
            problems.append(f"{where} oauth: {key} must be an https:// URL")
    required = ["token_url"]
    required += ["device_code_url"] if kind == "oauth_device" else []
    required += ["authorize_url"] if kind == "oauth_pkce" else []
    for key in required:
        if not raw.get(key):
            problems.append(f"{where} oauth: {key} is required for kind {kind!r}")
    client_id = raw.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        problems.append(f"{where} oauth: client_id is required (a public client id)")
    scopes = raw.get("scopes", [])
    if not isinstance(scopes, list) or not all(isinstance(x, str) for x in scopes):
        problems.append(f"{where} oauth: scopes must be a list of strings")
    port = raw.get("redirect_port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)):
        problems.append(f"{where} oauth: redirect_port must be an integer")
    elif isinstance(port, int) and not 0 < port < 65536:
        problems.append(f"{where} oauth: redirect_port must be between 1 and 65535")
    if not isinstance(raw.get("audience", ""), str):
        problems.append(f"{where} oauth: audience must be a string")
    return dict(raw)


def _env_names(value: Any, what: str, problems: list[str]) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        problems.append(f"{what} must be a list of environment variable names")
        return []
    bad = [v for v in value if not _ENV_NAME.match(v)]
    if bad:
        problems.append(f"{what}: {', '.join(map(repr, bad))} is not a variable name")
    return list(value)


def validate(doc: dict[str, Any], *, source: str = "") -> PluginManifest:
    """Build a :class:`PluginManifest` from a parsed ``plugin.toml``; every problem found is
    reported at once in a :class:`ManifestError`."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        raise ManifestError(["manifest is not a TOML table"], source=source)
    schema = doc.get("schema", MANIFEST_SCHEMA)
    if schema != MANIFEST_SCHEMA:
        problems.append(f"schema {schema!r} is not supported (this framework reads schema 1)")

    def text(key: str, *, required: bool = False, default: str = "") -> str:
        value = doc.get(key, default)
        if value in (None, "") and required:
            problems.append(f"`{key}` is required")
            return default
        if not isinstance(value, str):
            problems.append(f"`{key}` must be a string")
            return default
        return value

    pid = text("id", required=True)
    if pid and not _ID.match(pid):
        problems.append(f"id {pid!r} must match [a-z0-9][a-z0-9-]*")
    kind = text("kind", required=True)
    if kind and kind not in KINDS:
        problems.append(f"kind {kind!r} must be one of {', '.join(KINDS)}")
    name = text("name", required=True)
    version = text("version", required=True)
    if version and not _VERSION.match(version):
        problems.append(f"version {version!r} is not a version number")
    dmv = text("data_model_version", required=True)
    if dmv and not _VERSION.match(dmv):
        problems.append(f"data_model_version {dmv!r} is not a version number")
    maturity = text("maturity", default="planned")
    if maturity not in MATURITIES:
        problems.append(f"maturity {maturity!r} must be one of {', '.join(MATURITIES)}")
    licence = doc.get("licence", doc.get("license", ""))
    if not isinstance(licence, str) or not licence:
        problems.append("`licence` is required (an SPDX expression)")
        licence = ""

    optional: list[OptionalDependency] = []
    raw_optional = doc.get("optional", [])
    if not isinstance(raw_optional, list):
        problems.append("`[[optional]]` must be an array of tables")
        raw_optional = []
    for i, item in enumerate(raw_optional):
        if not isinstance(item, dict) or not isinstance(item.get("package"), str):
            problems.append(f"[[optional]] #{i + 1} needs a `package`")
            continue
        optional.append(
            OptionalDependency(
                package=item["package"],
                licence=str(item.get("licence", item.get("license", ""))),
                purpose=str(item.get("purpose", "")),
                module=str(item.get("module", "")),
                flag=str(item.get("flag", "")),
            )
        )

    connectors = _targets(doc, "connectors", problems)
    importers = _targets(doc, "importers", problems)
    exporter: ExporterSpec | None = None
    raw_exporter = doc.get("exporter")
    if raw_exporter is not None:
        if not isinstance(raw_exporter, dict) or not isinstance(raw_exporter.get("vendor"), str):
            problems.append("`[exporter]` needs a `vendor`")
        else:
            exporter = ExporterSpec(
                vendor=raw_exporter["vendor"],
                tool=str(raw_exporter.get("tool", "")),
                files=str(raw_exporter.get("files", "files")),
                usage=str(raw_exporter.get("usage", "")),
            )
            if not _ID.match(exporter.vendor):
                problems.append(f"[exporter] vendor {exporter.vendor!r} must match [a-z0-9-]+")
            if Path(exporter.files).is_absolute() or ".." in Path(exporter.files).parts:
                problems.append("[exporter] files must be a relative path inside the package")
    if kind == "exporter" and exporter is None:
        problems.append("an exporter plugin needs an `[exporter]` table")
    if kind == "connector" and not connectors:
        problems.append("a connector plugin needs at least one entry in `[connectors]`")
    if kind == "importer" and not importers:
        problems.append("an importer plugin needs at least one entry in `[importers]`")

    author: dict[str, str] = {}
    raw_author = doc.get("author", {})
    if not isinstance(raw_author, dict):
        problems.append("`[author]` must be a table (name, company, email, url)")
    else:
        unknown = sorted(set(raw_author) - set(_AUTHOR_KEYS))
        if unknown:
            problems.append(f"[author]: unknown key(s) {', '.join(unknown)}")
        for key in _AUTHOR_KEYS:
            value = raw_author.get(key, "")
            if not isinstance(value, str):
                problems.append(f"[author] `{key}` must be a string")
            elif value:
                author[key] = value
    category = text("category")
    if category and category not in CATEGORIES:
        problems.append(f"category {category!r} must be one of {', '.join(CATEGORIES)}")
    for key in ("homepage", "repository"):
        url = text(key)
        if url and not url.startswith(("https://", "http://")):
            problems.append(f"`{key}` must be an http(s) URL")
    for key in ("icon", "readme"):
        rel = text(key)
        if rel and (Path(rel).is_absolute() or ".." in Path(rel).parts):
            problems.append(f"`{key}` must be a relative path inside the plugin folder")
    updated = text("updated")
    if updated and not _DATE.match(updated):
        problems.append(f"updated {updated!r} must be a date (YYYY-MM-DD)")

    manifest = PluginManifest(
        id=pid,
        kind=kind,
        name=name,
        version=version,
        data_model_version=dmv,
        vendor=text("vendor"),
        summary=text("summary"),
        formats=_str_list(doc, "formats", problems),
        maturity=maturity if maturity in MATURITIES else "planned",
        licence=licence,
        requires=_str_list(doc, "requires", problems),
        plugins=_str_list(doc, "plugins", problems),
        suggests=_str_list(doc, "suggests", problems),
        optional=optional,
        connectors=connectors,
        importers=importers,
        exporter=exporter,
        auth=_auth(doc, problems),
        homepage=text("homepage"),
        description=text("description"),
        author=author,
        tags=_str_list(doc, "tags", problems),
        keywords=_str_list(doc, "keywords", problems),
        category=category,
        icon=text("icon"),
        repository=text("repository"),
        readme=text("readme", default="README.md"),
        updated=updated,
        schema=schema if isinstance(schema, int) else MANIFEST_SCHEMA,
    )
    if problems:
        raise ManifestError(problems, source=source)
    return manifest


def load_file(path: Path) -> PluginManifest:
    """Read and validate one ``plugin.toml``."""
    try:
        doc = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ManifestError([f"cannot read manifest: {e}"], source=str(path)) from e
    except tomllib.TOMLDecodeError as e:
        raise ManifestError([f"not valid TOML: {e}"], source=str(path)) from e
    return validate(doc, source=str(path))


def package_dir(module: str) -> Path | None:
    """Directory of an import package, found without executing it."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def load_from_package(module: str) -> PluginManifest:
    """The manifest shipped inside import package ``module`` (the plugin is not imported)."""
    directory = package_dir(module)
    if directory is None:
        raise ManifestError([f"package {module!r} cannot be located"], source=module)
    return load_file(directory / MANIFEST_NAME)
