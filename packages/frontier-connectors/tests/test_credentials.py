# SPDX-License-Identifier: Apache-2.0
"""Credentials, settings and policy of the connector framework: ``[[auth]]`` declarations,
``auth-providers.json``, resolution through the Frontier SDK (and without it), keychain-free
listings, installer defaults and ``policy.plugin_sources``. No network, no real keychain."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tomllib
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from frontier_connectors import cloud, credentials, importers, registry
from frontier_connectors.cli import main
from frontier_connectors.errors import PolicyError
from frontier_connectors.plugins import auth as plugin_auth
from frontier_connectors.plugins import installer, manage, manifest

PKG_ROOT = Path(__file__).parents[1]
PLUGINS_DIR = PKG_ROOT.parent / "connectors"
KEY = "acme-key-000001"

BASE: dict[str, Any] = {
    "schema": 1,
    "kind": "connector",
    "version": "1.0.0",
    "data_model_version": "1.1",
    "licence": "Apache-2.0",
}
ACME_AUTH = """
[[auth]]
id = "acme-cloud"
title = "Acme Cloud"
kind = "api_key"
env = ["ACME_API_KEY"]
help_url = "https://acme.example.com/keys"
scopes_hint = "read access"
fields = [
  { name = "api_key", title = "API key" },
  { name = "region", secret = false, required = false, env = ["ACME_REGION"] },
]
"""
OAUTH_AUTH = """
[[auth]]
id = "acme-sso"
title = "Acme SSO"
kind = "oauth_pkce"

[auth.oauth]
authorize_url = "https://id.example.com/authorize"
token_url = "https://id.example.com/token"
client_id = "frontier-desktop"
scopes = ["read", "offline_access"]
redirect_port = 8765
audience = "https://api.example.com"
"""


def fake_plugin(site: Path, pid: str, auth_toml: str) -> None:
    """A fake installed plugin `pid` whose manifest carries `auth_toml`."""
    module = pid.replace("-", "_") + "_plugin"
    pkg = site / module
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "connector.py").write_text(
        "import frontier_connectors as fc\n\n"
        f"class C(fc.BaseConnector):\n    id = {pid!r}\n    name = {pid!r}\n"
        "    formats = ['x']\n    extensions = ['.acmex']\n"
    )
    head = "".join(f"{k} = {json.dumps(v)}\n" for k, v in BASE.items())
    head += f'id = "{pid}"\nname = "{pid}"\n'
    (pkg / "plugin.toml").write_text(
        f'{head}{auth_toml}\n[connectors]\n{pid} = "{module}.connector:C"\n'
    )
    info = site / f"{module}-1.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0.0\n")
    (info / "entry_points.txt").write_text(
        f"[frontier.plugins]\n{pid} = {module}\n\n"
        f"[frontier.connectors]\n{pid} = {module}.connector:C\n"
    )


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / "site"
    d.mkdir()
    monkeypatch.syspath_prepend(str(d))
    monkeypatch.setitem(sys.modules, "keyring", None)  # not installed
    yield d
    for name in [m for m in sys.modules if m.endswith("_plugin") or "_plugin." in m]:
        del sys.modules[name]
    registry.reload()


@pytest.fixture
def no_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The framework on its own: the Frontier SDK is not installed."""
    monkeypatch.setattr(credentials, "sdk", lambda name: None)


def run_json(capsys: pytest.CaptureFixture[str], *argv: str, expect: int = 0) -> Any:
    capsys.readouterr()
    assert main([*argv, "--json"]) == expect
    return json.loads(capsys.readouterr().out)


def doc(auth_toml: str) -> dict[str, Any]:
    return (
        BASE
        | {"id": "acme", "name": "Acme"}
        | tomllib.loads(auth_toml + '\n[connectors]\nacme = "acme.connector:C"\n')
    )


# -- manifest -----------------------------------------------------------------------------------


def test_auth_declarations_are_parsed_and_normalised() -> None:
    m = manifest.validate(doc(ACME_AUTH + OAUTH_AUTH))
    key, sso = m.auth
    assert (key.id, key.kind, key.env, key.help_url) == (
        "acme-cloud",
        "api_key",
        ["ACME_API_KEY"],
        "https://acme.example.com/keys",
    )
    provider = key.to_provider()
    assert provider["fields"] == [
        {"name": "api_key", "title": "API key", "secret": True, "required": True,
         "env": ["ACME_API_KEY"]},
        {"name": "region", "title": "region", "secret": False, "required": False,
         "env": ["ACME_REGION"]},
    ]  # fmt: skip
    assert "oauth" not in provider and "test" not in provider
    # an OAuth provider: the table passes through unchanged, the fields are the token set
    assert sso.kind == "oauth_pkce"
    passed = sso.to_provider()
    assert passed["oauth"] == tomllib.loads(OAUTH_AUTH)["auth"][0]["oauth"]
    assert [f["name"] for f in passed["fields"]] == ["access_token", "refresh_token", "expires_at"]
    assert [f["secret"] for f in passed["fields"]] == [True, True, False]
    # every shape of the same declaration gives the same provider table
    assert manifest.auth_providers(m.to_dict()["auth"]) == [provider, passed]
    assert manifest.auth_providers([provider, passed]) == [provider, passed]
    default = manifest.validate(doc('[[auth]]\nid = "x"\ntitle = "X"\nenv = ["X_TOKEN"]\n'))
    assert default.auth[0].to_provider()["fields"] == [
        {"name": "token", "title": "token", "secret": True, "required": True, "env": ["X_TOKEN"]}
    ]


def test_auth_declaration_problems_are_all_reported() -> None:
    bad = """
[[auth]]
id = "Bad Id"
kind = "password"
env = ["not a name"]
help_url = "http://insecure.example.com"
test = "no target"
surprise = 1
fields = [{ name = "Key", secret = "yes" }, { title = "nameless" }]

[[auth]]
id = "dev"
title = "Device"
kind = "oauth_device"
[auth.oauth]
token_url = "http://id.example.com/token"
client_id = ""
scopes = "read"
redirect_port = 99999
extra = true

[[auth]]
id = "pkce"
title = "PKCE"
kind = "oauth_pkce"

[[auth]]
id = "plain"
title = "Plain"
kind = "token"
[auth.oauth]
token_url = "https://id.example.com/token"
client_id = "x"
"""
    with pytest.raises(manifest.ManifestError) as exc:
        manifest.validate(doc(bad))
    text = "\n".join(exc.value.problems)
    for expected in (
        "[[auth]] Bad Id: id must match",
        "`title` is required",
        "kind 'password' must be one of",
        "'not a name' is not a variable name",
        "help_url must be an https:// URL",
        'is not "module:function"',
        "unknown key(s) surprise",
        "field Key: name must match",
        "`secret` must be true or false",
        "every field needs a `name`",
        "[[auth]] dev oauth: token_url must be an https:// URL",
        "device_code_url is required for kind 'oauth_device'",
        "client_id is required",
        "scopes must be a list of strings",
        "redirect_port must be between 1 and 65535",
        "oauth: unknown key(s) extra",
        "[[auth]] pkce: kind 'oauth_pkce' needs an `[auth.oauth]` table",
        "[[auth]] plain: `[auth.oauth]` needs kind oauth_device or oauth_pkce",
    ):
        assert expected in text, (expected, text)
    with pytest.raises(manifest.ManifestError, match="authorize_url is required"):
        manifest.validate(
            doc(
                '[[auth]]\nid = "p"\ntitle = "P"\nkind = "oauth_pkce"\n[auth.oauth]\n'
                'token_url = "https://id.example.com/t"\nclient_id = "c"\n'
            )
        )




def test_sync_round_trips_auth_and_marketplace_keys(tmp_path: Path) -> None:
    """`tools/sync_plugins.py` never rewrites a `plugin.toml`, and what it generates carries
    both the `[[auth]]` tables (with `[auth.oauth]`) and the keys it does not know."""
    sys.path.insert(0, str(PKG_ROOT / "tools"))
    try:
        sync_plugins: Any = importlib.import_module("sync_plugins")
    finally:
        sys.path.remove(str(PKG_ROOT / "tools"))
    before = {p: p.read_bytes() for p in PLUGINS_DIR.glob("*/src/*/plugin.toml")}
    out = subprocess.run(
        [sys.executable, str(PKG_ROOT / "tools" / "sync_plugins.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if (PKG_ROOT.parents[1] / "nx.json").is_file():
        assert out.returncode == 0, out.stdout + out.stderr
    assert before == {p: p.read_bytes() for p in before}, "a manifest was rewritten"
    # a manifest with marketplace keys, unknown keys and an OAuth declaration
    raw = doc(ACME_AUTH + OAUTH_AUTH) | {
        "description": "Acme results",
        "tags": ["cfd"],
        "category": "cloud-service",
        "icon": "icon.svg",
        "repository": "https://example.com/acme",
        "readme": "README.md",
        "author": {"name": "Acme", "company": "Acme Inc."},
        "x_future_key": {"kept": True},
    }
    manifest.validate(raw)  # unknown and marketplace keys do not invalidate a manifest
    text = sync_plugins.catalog([(tmp_path, "acme", raw)])
    (row,) = json.loads(text)["plugins"]
    assert row["auth"] == manifest.auth_providers(raw["auth"])
    assert row["auth"][1]["oauth"] == raw["auth"][1]["oauth"]
    assert sync_plugins.catalog([(tmp_path, "acme", raw)]) == text, "generation is stable"
    assert "x_future_key" in raw and raw["auth"][0]["env"] == ["ACME_API_KEY"], "input untouched"


# -- auth-providers.json ------------------------------------------------------------------------


def test_declarations_reach_the_providers_file_and_follow_enable_disable(
    site: Path, _isolated_frontier_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_plugin(site, "acme", ACME_AUTH + OAUTH_AUTH)
    fake_plugin(site, "acme-extra", ACME_AUTH.replace("ACME_API_KEY", "ACME_KEY"))
    path = _isolated_frontier_home / plugin_auth.PROVIDERS_FILE
    assert plugin_auth.providers_path() == path and not path.exists()
    registry.reload()
    out = run_json(capsys, "plugins", "auth")
    assert out["path"] == str(path) and out["written"] is True and out["error"] == ""
    written = json.loads(path.read_text())
    assert written == out["providers"]
    by_id = {p["id"]: p for p in written}
    assert {"acme-cloud", "acme-sso"} <= set(by_id)
    acme = by_id["acme-cloud"]
    assert acme["declared_by"] == ["acme", "acme-extra"]
    assert acme["fields"][0]["env"] == ["ACME_API_KEY", "ACME_KEY"], "merged by id"
    assert by_id["acme-sso"]["oauth"]["client_id"] == "frontier-desktop"
    assert {s["id"] for s in out["status"]} == set(by_id)
    # the SDK (and the command line) read the same file
    from frontier import auth

    sdk_view = {p.id: p for p in auth.providers()}
    assert sdk_view["acme-cloud"].declared_by == ["acme", "acme-extra"]
    assert sdk_view["acme-sso"].kind == "oauth_pkce" and sdk_view["acme-sso"].oauth
    assert auth.provider_for_env("ACME_KEY") == ("acme-cloud", "api_key")
    # a second run changes nothing
    assert run_json(capsys, "plugins", "auth")["written"] is False
    # disabling a plugin withdraws what only it declared
    manage.set_enabled("acme", False)
    by_id = {p["id"]: p for p in json.loads(path.read_text())}
    assert "acme-sso" not in by_id and by_id["acme-cloud"]["declared_by"] == ["acme-extra"]
    manage.set_enabled("acme", True)
    assert "acme-sso" in {p["id"] for p in json.loads(path.read_text())}
    # a home that cannot be written is reported, not raised
    blocked = site / "file-not-dir"
    blocked.write_text("")
    info = plugin_auth.write_providers(path=blocked / "auth-providers.json")
    assert info["written"] is False and "could not be written" in info["error"]


# -- resolving ----------------------------------------------------------------------------------


def test_secret_resolves_through_the_sdk_store_then_env(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from frontier import auth

    monkeypatch.setenv("FRONTIER_SECRETS__BACKEND", "file")
    assert cloud.secret("FLOW360_APIKEY") is None and credentials.secret("FLOW360_APIKEY") is None
    assert credentials.provider_of("FLOW360_APIKEY") == ("flow360", "api_key")
    assert "frontier auth login flow360" in credentials.login_hint("FLOW360_APIKEY")
    monkeypatch.setenv("FLOW360_APIKEY", "from-env-000001")
    assert cloud.secret("FLOW360_APIKEY") == "from-env-000001"
    auth.login("flow360", {"api_key": KEY})
    assert cloud.secret("FLOW360_APIKEY") == KEY, "the store wins over the environment"
    monkeypatch.setenv(auth.BOUND_ENV, "flow360")
    assert cloud.secret("FLOW360_APIKEY") == "from-env-000001", "a job binding wins"
    monkeypatch.delenv(auth.BOUND_ENV)
    # a non-secret field of the same login
    auth.login("simscale", {"api_key": KEY, "api_url": "https://api.eu.example.com"})
    assert credentials.secret("SIMSCALE_API_URL") == "https://api.eu.example.com"
    # a variable no provider declares is read as before
    monkeypatch.setenv("SOME_PRIVATE_TOKEN", "v")
    assert credentials.provider_of("SOME_PRIVATE_TOKEN") is None
    assert credentials.secret("SOME_PRIVATE_TOKEN") == "v"
    assert credentials.login_hint("SOME_PRIVATE_TOKEN") == "`export SOME_PRIVATE_TOKEN=...`"

    class Imp(importers.BaseImporter):
        id = "acme-imp"
        name = "Acme"
        credentials_env = ["RESCALE_API_KEY"]

    with pytest.raises(Exception, match="needs credentials") as exc:
        Imp().credentials()
    assert "frontier auth login rescale" in str(exc.value)
    auth.login("rescale", {"api_key": KEY})
    assert Imp().credentials() == {"RESCALE_API_KEY": KEY}


def test_without_the_sdk_the_old_lookup_applies(
    no_sdk: None, monkeypatch: pytest.MonkeyPatch, _isolated_frontier_home: Path
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    assert credentials.config() is None and credentials.setting("plugins.installer") is None
    assert credentials.provider_of("FLOW360_APIKEY") is None
    assert cloud.secret("FLOW360_APIKEY") is None and not credentials.present("FLOW360_APIKEY")
    monkeypatch.setenv("FLOW360_APIKEY", "from-env-000001")
    assert cloud.secret("FLOW360_APIKEY") == "from-env-000001"
    assert credentials.present("FLOW360_APIKEY")
    monkeypatch.delenv("FLOW360_APIKEY")
    ring: Any = types.ModuleType("keyring")
    ring.get_password = lambda service, user: (
        KEY
        if (service, user)
        == (
            "frontier-connectors",
            "FLOW360_APIKEY",
        )
        else None
    )
    monkeypatch.setitem(sys.modules, "keyring", ring)
    assert cloud.secret("FLOW360_APIKEY") == KEY
    # settings and policy do not exist without the SDK
    monkeypatch.setenv("FRONTIER_CONNECTORS_CACHE", "/data/cache")
    assert cloud.cache_root() == Path("/data/cache/cloud")
    monkeypatch.setenv("FRONTIER_PLUGIN_INSTALLER", "UV")
    assert installer.installer_choice() == "uv"
    assert installer.source_defaults() == {"wheel_dir": "", "index_url": ""}
    assert plugin_auth.providers_path() == _isolated_frontier_home / "auth-providers.json"


def test_cache_root_follows_the_setting(
    monkeypatch: pytest.MonkeyPatch, _isolated_frontier_home: Path, tmp_path: Path
) -> None:
    monkeypatch.delenv("FRONTIER_CONNECTORS_CACHE", raising=False)
    assert cloud.cache_root() == Path.home() / ".cache" / "frontier-connectors" / "cloud"
    monkeypatch.setenv("FRONTIER_CONNECTORS_CACHE", str(tmp_path / "legacy"))
    assert cloud.cache_root() == tmp_path / "legacy" / "cloud"
    monkeypatch.delenv("FRONTIER_CONNECTORS_CACHE")
    (_isolated_frontier_home / "config.toml").write_text(
        f'[storage]\ncache_dir = "{(tmp_path / "configured").as_posix()}"\n'
    )
    assert cloud.cache_root() == tmp_path / "configured" / "cloud"


# -- listings never touch the keychain ----------------------------------------------------------




# -- installer: settings and policy ---------------------------------------------------------------


#: the system file of an administrator who allows any plugin source (the built-in default
#: is the official index plus the configured index and wheel directory)
ANY_SOURCE = '[policy]\nplugin_sources = ["any"]\n'


def configure(home: Path, user: str = "", system: str = ANY_SOURCE) -> None:
    (home / "config.toml").write_text(user)
    (home / "system.toml").write_text(system)


def test_installer_defaults_come_from_the_settings(
    monkeypatch: pytest.MonkeyPatch, _isolated_frontier_home: Path, tmp_path: Path
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    monkeypatch.delenv(installer.INSTALLER_ENV, raising=False)
    assert installer.installer_choice() == ""
    configure(
        _isolated_frontier_home,
        f'[plugins]\ninstaller = "uv"\nwheel_dir = "{wheels.as_posix()}"\n'
        'index_url = "https://pkgs.example.com/simple/"\n',
    )
    assert installer.installer_choice() == "uv"
    assert installer.source_defaults() == {
        "wheel_dir": str(wheels.resolve()),
        "index_url": "https://pkgs.example.com/simple",
    }
    monkeypatch.setattr(installer, "installer_command", lambda: [sys.executable, "-m", "pip"])
    result = installer.install(["fluent"], with_suggested=False, dry_run=True)
    cmd = result["command"]
    assert cmd[cmd.index("--find-links") + 1] == str(wheels.resolve())
    assert cmd[cmd.index("--index-url") + 1] == "https://pkgs.example.com/simple"
    assert result["sources"] == ["wheel_dir", "index"] and result["policy_excluded"] == []
    # flags win over the settings
    other = installer.install(
        ["fluent"], index_url="https://other.example.com/simple", dry_run=True
    )
    cmd = other["command"]
    assert cmd[cmd.index("--index-url") + 1] == "https://other.example.com/simple"
    assert other["sources"] == ["wheel_dir", "https://other.example.com/simple"]
    # the legacy variable is the setting's alias
    configure(_isolated_frontier_home)
    monkeypatch.setenv("FRONTIER_PLUGIN_INSTALLER", "pip")
    assert installer.installer_choice() == "pip"
    assert installer.install(["fluent"], dry_run=True)["sources"] == ["pypi"]


def test_policy_plugin_sources_is_enforced_before_any_install(
    monkeypatch: pytest.MonkeyPatch,
    _isolated_frontier_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheels, elsewhere = tmp_path / "wheels", tmp_path / "elsewhere"
    wheels.mkdir()
    elsewhere.mkdir()
    ran: list[list[str]] = []
    monkeypatch.setattr(installer, "installer_command", lambda: [sys.executable, "-m", "pip"])
    monkeypatch.setattr(
        installer,
        "_run",
        lambda cmd, dry, **kw: ran.append(cmd) or {"command": cmd, "returncode": None},
    )
    configure(
        _isolated_frontier_home,
        user=f'[plugins]\nwheel_dir = "{wheels.as_posix()}"\n',
        system='[policy]\nplugin_sources = ["wheel_dir", "https://pkgs.example.com/"]\n'
        '[lock]\n"policy.plugin_sources" = "Approved builds only"\n',
    )
    # PyPI is not allowed: it is left out, and the configured wheel directory is used alone
    result = installer.install(["fluent"], with_suggested=False, dry_run=True)
    cmd = result["command"]
    assert "--no-index" in cmd and cmd[cmd.index("--find-links") + 1] == str(wheels.resolve())
    assert result["sources"] == ["wheel_dir"] and result["policy_excluded"] == ["pypi"]
    # an allowed URL prefix
    ok = installer.install(["fluent"], index_url="https://pkgs.example.com/simple", dry_run=True)
    assert "--no-index" not in ok["command"]
    # sources the user names and the policy forbids are refused, before the installer runs
    ran.clear()
    for kwargs, spec in (
        ({"find_links": [str(elsewhere)]}, "fluent"),
        ({"index_url": "https://pypi.org/simple"}, "fluent"),
        ({"extra_index_url": "https://mirror.example.net/simple"}, "fluent"),
        ({}, "git+https://example.invalid/acme/plugin.git"),
        ({}, str(elsewhere)),
    ):
        with pytest.raises(PolicyError) as exc:
            installer.install([spec], **kwargs)  # type: ignore[arg-type]
        assert exc.value.policy == "policy.plugin_sources" and exc.value.locked
        assert "is not allowed here; allowed sources: wheel_dir" in exc.value.message
        assert "Approved builds only" in exc.value.message
    assert ran == []
    # nothing allowed remains: the policy is named instead of a failed resolution
    configure(_isolated_frontier_home, system='[policy]\nplugin_sources = ["bundled"]\n')
    with pytest.raises(PolicyError, match="installing plugins from `pypi` is not allowed"):
        installer.install(["fluent"], dry_run=True)
    assert ran == []
    # the command line: one JSON object, exit code 2
    out = run_json(capsys, "plugins", "install", "fluent", "--dry-run", expect=2)
    assert out["error"] == "PolicyError" and out["status"] == "blocked_by_policy"
    assert out["policy"] == "policy.plugin_sources" and out["locked"] is False
    assert "set by your administrator" in out["origin"] and "administrator" in out["hint"]
    assert main(["plugins", "install", "fluent", "--dry-run"]) == 2
    assert "policy policy.plugin_sources" in capsys.readouterr().err
    # the bundled wheel directory of the product is its own source
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(installer, "BUNDLED_WHEEL_DIR", bundled)
    result = installer.install(["fluent"], with_suggested=False, dry_run=True)
    assert result["sources"] == ["bundled"] and "--no-index" in result["command"]
    # the default policy allows everything
    configure(_isolated_frontier_home)
    assert installer.install([str(elsewhere)], dry_run=True)["policy_excluded"] == []
