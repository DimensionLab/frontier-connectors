# SPDX-License-Identifier: Apache-2.0
"""The plugin system: manifests, discovery, isolation, enable/disable, exporters, installer.

Fake plugin distributions are built on disk (a ``*.dist-info`` with ``entry_points.txt`` next to
an import package) and put on ``sys.path``, which is exactly what an installed wheel looks like
to :mod:`importlib.metadata`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import frontier_connectors as fc
from frontier_connectors import importers, plugins, registry
from frontier_connectors.cli import main
from frontier_connectors.errors import ConnectorError
from frontier_connectors.plugins import discovery, exporters, installer, manifest
from frontier_connectors.plugins import state as state_mod

PKG_ROOT = Path(__file__).parents[1]
REPO = PKG_ROOT.parents[1]

GOOD = {
    "schema": 1,
    "id": "toyfmt",
    "kind": "connector",
    "name": "Toy format",
    "version": "1.2.3",
    "data_model_version": "1.1",
    "vendor": "ACME",
    "formats": ["toy"],
    "maturity": "verified-synthetic",
    "licence": "Apache-2.0",
    "requires": [],
    "connectors": {"toyfmt": "toy_plugin.connector:ToyConnector"},
}

CONNECTOR_SOURCE = """
from pathlib import Path
import numpy as np
import frontier_connectors as fc

class ToyConnector(fc.BaseConnector):
    id = "toyfmt"
    name = "Toy format"
    formats = ["toy"]
    vendor = "ACME"
    extensions = [".toy"]
    maturity = "verified-synthetic"

    def describe(self, path, *, options=None):
        return fc.SourceDescription(connector_id=self.id, path=Path(path), format="toy")

    def read(self, path, *, zones=None, fields=None, times=None, options=None):
        yield fc.RawSample(
            id="toy", kind="point_cloud", points=np.zeros((2, 3), np.float32),
            fields={"p": fc.as_field(np.ones(2), "point")},
        )
"""


def _toml(doc: dict[str, Any]) -> str:
    lines = []
    tables = {k: v for k, v in doc.items() if isinstance(v, dict)}
    for k, v in doc.items():
        if k not in tables:
            lines.append(f"{k} = {json.dumps(v)}")
    for name, table in tables.items():
        lines.append(f"\n[{name}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in table.items()]
    return "\n".join(lines) + "\n"


def make_plugin(
    root: Path,
    *,
    dist: str = "toy-plugin",
    module: str = "toy_plugin",
    doc: dict[str, Any] | None = None,
    version: str = "1.2.3",
    connector_source: str | None = CONNECTOR_SOURCE,
    manifest_text: str | None = None,
    with_manifest_entry_point: bool = True,
    exporter_files: dict[str, str] | None = None,
) -> Path:
    """Write a fake installed plugin under ``root`` (which must be on ``sys.path``)."""
    doc = dict(GOOD if doc is None else doc)
    pkg = root / module
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    if connector_source is not None:
        (pkg / "connector.py").write_text(connector_source)
    if with_manifest_entry_point:
        (pkg / "plugin.toml").write_text(manifest_text if manifest_text is not None else _toml(doc))
    for rel, text in (exporter_files or {}).items():
        f = pkg / "files" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    info = root / f"{dist.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n")
    eps = ""
    if with_manifest_entry_point:
        eps += f"[frontier.plugins]\n{doc['id']} = {module}\n\n"
    if doc.get("connectors"):
        eps += "[frontier.connectors]\n"
        eps += "".join(f"{k} = {v}\n" for k, v in doc["connectors"].items()) + "\n"
    if doc.get("exporter"):
        eps += f"[frontier.exporters]\n{doc['exporter']['vendor']} = {module}\n"
    (info / "entry_points.txt").write_text(eps)
    return pkg


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An extra ``site-packages`` directory at the front of ``sys.path``."""
    d = tmp_path / "site"
    d.mkdir()
    monkeypatch.syspath_prepend(str(d))
    yield d
    for name in [m for m in sys.modules if m.split(".")[0] in ("toy_plugin", "bad_plugin")]:
        del sys.modules[name]
    registry.reload()


# -- manifest -----------------------------------------------------------------------------------


def test_manifest_validates_and_reports_every_problem_at_once() -> None:
    m = manifest.validate(dict(GOOD))
    assert m.id == "toyfmt" and m.kind == "connector" and m.maturity == "verified-synthetic"
    bad = dict(GOOD) | {
        "id": "Toy Format",
        "kind": "plugin",
        "version": "soon",
        "maturity": "works-on-my-machine",
        "connectors": {"toyfmt": "not a target"},
    }
    del bad["licence"]
    with pytest.raises(manifest.ManifestError) as e:
        manifest.validate(bad, source="x/plugin.toml")
    text = str(e.value)
    for needle in ("id 'Toy Format'", "kind 'plugin'", "version 'soon'", "maturity", "licence"):
        assert needle in text, needle
    assert len(e.value.problems) >= 6 and text.startswith("x/plugin.toml: ")
    # British and American spelling are both accepted
    us = dict(GOOD)
    us["license"] = us.pop("licence")
    assert manifest.validate(us).licence == "Apache-2.0"
    with pytest.raises(manifest.ManifestError, match=r"needs an `\[exporter\]` table"):
        manifest.validate(dict(GOOD) | {"kind": "exporter", "connectors": {}})
    with pytest.raises(manifest.ManifestError, match="relative path inside the package"):
        manifest.validate(
            dict(GOOD)
            | {"kind": "exporter", "connectors": {}, "exporter": {"vendor": "x", "files": "../.."}}
        )


def test_data_model_compatibility_rule() -> None:
    assert manifest.DATA_MODEL_VERSION == "1.1"
    assert manifest.data_model_compatible("1.1") is None
    assert manifest.data_model_compatible("1.0") is None  # older minor: still served
    assert "newer" in (manifest.data_model_compatible("1.2") or "")
    assert "major" in (manifest.data_model_compatible("2.0") or "")
    assert manifest.data_model_compatible("banana") is not None


def test_first_party_manifests_and_generated_files_are_in_sync() -> None:
    """Every first-party plugin validates, and pyproject.toml / catalog.json were regenerated."""
    out = subprocess.run(
        [sys.executable, str(PKG_ROOT / "tools" / "sync_plugins.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if (REPO / "nx.json").is_file():
        assert out.returncode == 0, out.stdout + out.stderr
    ids = {e["id"] for e in discovery.catalog()}
    assert {"openfoam", "arrays", "catalyst"} <= ids
    installed = {r.id: r for r in plugins.discover()}
    for pid in ids:
        rec = installed.get(pid)
        assert rec is not None, f"{pid} is in the catalog but not installed in the workspace"
        assert rec.status == "enabled", (pid, rec.problems)
        assert rec.first_party and rec.manifest is not None
        assert rec.manifest.data_model_version == manifest.DATA_MODEL_VERSION


# -- state --------------------------------------------------------------------------------------


def test_state_file_round_trip_and_damage(tmp_path: Path) -> None:
    path = state_mod.state_path()
    assert path.name == "plugins.toml" and not path.exists()
    st = state_mod.load()
    assert st.set_enabled("plugins", "luminary", False) and not st.set_enabled(
        "plugins", "luminary", False
    )
    st.set_enabled("connectors", "hdf5", False)
    st.find_links = ['/mnt/usb/wheels "x"']
    st.deployed.append(state_mod.Deployed("matlab", "exporter-matlab", "0.2.0", "/c/m", "now"))
    state_mod.save(st)
    again = state_mod.load()
    assert again.is_disabled("plugins", "luminary") and again.is_disabled(
        "connectors", "hdf5"
    )
    assert again.find_links == ['/mnt/usb/wheels "x"'] and again.deployed[0].vendor == "matlab"
    assert not list(path.parent.glob(".plugins-*"))  # atomic write leaves no temp file
    path.write_text("this is [not toml")
    damaged = state_mod.load()
    assert damaged.error and not damaged.is_disabled("plugins", "luminary")
    with pytest.raises(ConnectorError, match="could not be read"):
        plugins.set_enabled("vtk", False)
    assert {"vtk", "openfoam"} <= set(registry.connectors())  # registry unaffected


# -- discovery, isolation -----------------------------------------------------------------------


def test_third_party_plugin_is_discovered_loaded_and_converted(site: Path, tmp_path: Path) -> None:
    make_plugin(site)
    registry.reload()
    rec = plugins.find("toyfmt")
    assert rec is not None and rec.status == "enabled" and not rec.first_party
    assert rec.dist == "toy-plugin" and rec.version == "1.2.3"
    row = next(i for i in fc.list_connectors() if i.id == "toyfmt")
    assert row.plugin == "toyfmt" and row.source == "plugin:toyfmt" and row.status == "available"
    f = tmp_path / "a.toy"
    f.write_text("x")
    assert fc.detect(f)[0].connector_id == "toyfmt"
    res = fc.convert(f, tmp_path / "ds")
    assert res.sample_ids == ["toy"]


@pytest.mark.parametrize(
    ("change", "status", "needle"),
    [
        ({"data_model_version": "2.0"}, "incompatible", "different major version"),
        ({"data_model_version": "1.9"}, "incompatible", "upgrade frontier-connectors"),
        ({"maturity": "great"}, "invalid", "maturity"),
        ({"version": "9.9.9"}, "invalid", "differs from the distribution version"),
        ({"id": "other"}, "invalid", "differs from the entry-point name"),
    ],
)
def test_invalid_and_incompatible_plugins_are_listed_but_never_loaded(
    site: Path, change: dict[str, Any], status: str, needle: str
) -> None:
    doc = dict(GOOD) | change
    pkg = make_plugin(site)
    (pkg / "plugin.toml").write_text(_toml(doc))
    registry.reload()
    rec = plugins.find("toy-plugin")
    assert rec is not None and rec.status == status
    assert any(needle in p for p in rec.problems), rec.problems
    assert "toyfmt" not in registry.connectors()
    assert {"vtk", "openfoam"} <= set(registry.connectors())
    report = plugins.doctor()
    row = next(r for r in report["plugins"] if r["dist"] == "toy-plugin")
    assert not row["ok"] and not report["ok"]


def test_unparsable_manifest_and_broken_import_do_not_break_the_registry(
    site: Path, tmp_path: Path
) -> None:
    make_plugin(site, manifest_text="id = [unterminated")
    make_plugin(
        site,
        dist="bad-plugin",
        module="bad_plugin",
        doc=dict(GOOD) | {"id": "badfmt", "connectors": {"badfmt": "bad_plugin.connector:Bad"}},
        connector_source="import a_module_that_does_not_exist\n",
    )
    registry.reload()
    toy, bad = plugins.find("toy-plugin"), plugins.find("badfmt")
    assert toy is not None and toy.status == "invalid" and "not valid TOML" in toy.problems[0]
    assert bad is not None and bad.status == "enabled"  # the manifest is fine, the code is not
    rows = {i.id: i for i in fc.list_connectors()}
    assert (
        rows["badfmt"].status == "broken" and "a_module_that_does_not_exist" in rows["badfmt"].note
    )
    assert rows["vtk"].status == "available"
    # detection and reading through the registry keep working with the broken plugin installed
    vtu = tmp_path / "x.unknownext"
    vtu.write_text("?")
    assert fc.detect(vtu) == []
    with pytest.raises(ConnectorError, match="failed to load"):
        registry.get("badfmt").describe(vtu)
    report = plugins.doctor()
    row = next(r for r in report["plugins"] if r["id"] == "badfmt")
    assert not row["ok"] and "ModuleNotFoundError" in row["problems"][0]


def test_plugin_without_a_manifest_still_works(site: Path) -> None:
    """The form documented before manifests existed: only a ``frontier.connectors`` entry."""
    make_plugin(site, with_manifest_entry_point=False)
    registry.reload()
    rec = plugins.find("toy-plugin")
    assert rec is not None and rec.manifest is None and rec.status == "enabled"
    assert "no plugin.toml" in rec.problems[0]
    assert "toyfmt" in registry.connectors()


def test_enable_disable_plugin_and_single_connector(site: Path) -> None:
    make_plugin(site)
    registry.reload()
    assert "toyfmt" in registry.connectors()
    result = plugins.set_enabled("toyfmt", False)
    assert result == {
        "scope": "plugins",
        "id": "toyfmt",
        "enabled": False,
        "changed": True,
        "state": str(state_mod.state_path()),
    }
    assert "toyfmt" not in registry.connectors()
    rec = plugins.find("toyfmt")
    assert rec is not None and rec.status == "disabled"  # still installed
    with pytest.raises(fc.UnsupportedSourceError) as e:
        registry.get("toyfmt")
    assert "plugins enable toyfmt" in (e.value.hint or "")
    assert plugins.set_enabled("toyfmt", True)["changed"]
    assert "toyfmt" in registry.connectors()

    # one connector of a multi-connector first-party plugin
    assert {"netcdf", "hdf5"} <= set(registry.connectors())
    assert plugins.set_enabled("connector:hdf5", False)["scope"] == "connectors"
    assert "hdf5" not in registry.connectors() and "netcdf" in registry.connectors()
    with pytest.raises(fc.UnsupportedSourceError) as e2:
        registry.get("hdf5")
    assert "plugins enable connector:hdf5" in (e2.value.hint or "")
    plugins.set_enabled("hdf5", True)  # bare connector id also resolves
    assert "hdf5" in registry.connectors()
    with pytest.raises(ConnectorError, match="nothing installed is called"):
        plugins.set_enabled("no-such-thing", False)


def test_unknown_connector_hint_names_the_plugin_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pretend nothing is installed: the catalog still knows who provides 'fluent-cff'
    monkeypatch.setattr(discovery, "discover", lambda state=None: [])
    registry.reload()
    with pytest.raises(fc.UnsupportedSourceError) as e:
        registry.get("fluent-cff")
    assert "plugins install fluent" in (e.value.hint or "")
    assert discovery.missing_hint("exporter", "ansys_mechanical") is not None
    assert discovery.missing_hint("connector", "definitely-not-a-format") is None


def test_legacy_import_paths_still_resolve() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from frontier_connector_vtk.connector import VtkConnector
        from frontier_connectors.connectors.vtk import VtkConnector as Legacy  # type: ignore[import-not-found]

        from frontier_connectors.connectors import vendors  # type: ignore[attr-defined]

        assert Legacy is VtkConnector
        assert vendors.PalabosConnector.id == "palabos"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    with pytest.raises(ModuleNotFoundError):
        __import__("frontier_connectors.connectors.no_such_module")


# -- importers ----------------------------------------------------------------------------------


class _ToyImporter(importers.BaseImporter):
    id = "toycloud"
    name = "Toy cloud"
    credentials_env = ["TOYCLOUD_API_KEY"]
    source_help = "<case-id>"

    def plan(self, source: str, *, options: Any = None) -> list[importers.PlannedCall]:
        return [
            importers.PlannedCall(
                "GET",
                f"https://api.example.invalid/cases/{source}",
                "case metadata",
                headers={"Authorization": "$TOYCLOUD_API_KEY"},
            )
        ]

    def fetch(self, source: str, dest: Path, *, options: Any = None) -> Path:
        self.credentials()
        (dest / "a.toy").write_text("x")
        return dest / "a.toy"


def test_importer_dry_run_fetch_and_credentials(
    site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_plugin(site)
    registry.reload()
    importers.register(_ToyImporter())
    try:
        monkeypatch.delenv("TOYCLOUD_API_KEY", raising=False)
        assert main(["import", "toycloud", "case-7", "--dry-run", "--json"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["dry_run"] and plan["calls"][0]["target"].endswith("/cases/case-7")
        assert plan["calls"][0]["headers"] == {"Authorization": "$TOYCLOUD_API_KEY"}
        # without credentials the real run fails with the variable name, and nothing is written
        assert main(["import", "toycloud", "case-7", "--out", str(tmp_path / "ds"), "--json"]) == 2
        err = json.loads(capsys.readouterr().out)
        assert err["error"] == "NeedsCredentialsError" and "TOYCLOUD_API_KEY" in err["hint"]
        monkeypatch.setenv("TOYCLOUD_API_KEY", "not-a-real-key")
        result = importers.run_import(
            "toycloud", "case-7", tmp_path / "ds", download_dir=tmp_path / "raw"
        )
        assert result["dataset"]["sample_ids"] == ["toy"]
        assert (tmp_path / "raw" / "a.toy").is_file()
        rows = {r["id"]: r for r in importers.list_importers()}
        assert rows["toycloud"]["credentials_present"] == ["TOYCLOUD_API_KEY"]
        assert "not-a-real-key" not in json.dumps(rows)
    finally:
        importers.unregister("toycloud")
    with pytest.raises(fc.UnsupportedSourceError, match="unknown importer"):
        importers.get("toycloud")


# -- exporters ----------------------------------------------------------------------------------


def test_install_and_uninstall_exporter_with_version_stamp(site: Path, tmp_path: Path) -> None:
    doc = {
        "schema": 1,
        "id": "exporter-toytool",
        "kind": "exporter",
        "name": "Toy tool scripts",
        "version": "1.2.3",
        "data_model_version": "1.1",
        "licence": "Apache-2.0",
        "exporter": {"vendor": "toytool", "tool": "ToyTool 9", "usage": "toytool -batch run.mac"},
    }
    pkg = make_plugin(
        site,
        dist="frontier-exporter-toytool",
        module="toy_plugin",
        doc=doc,
        connector_source=None,
        exporter_files={"run.mac": "// v1\n", "lib/helper.mac": "// helper\n"},
    )
    assert exporters.exporter_files("toytool") == ["lib/helper.mac", "run.mac"]
    case = tmp_path / "case"
    result = plugins.install_exporter("toytool", case)
    dest = case / "toytool"
    assert result["path"] == str(dest.resolve()) and result["usage"] == "toytool -batch run.mac"
    stamp = json.loads((dest / exporters.STAMP).read_text())
    assert stamp["version"] == "1.2.3" and stamp["plugin"] == "exporter-toytool"
    assert {f["path"] for f in stamp["files"]} == {"lib/helper.mac", "run.mac"}
    assert all(len(f["sha256"]) == 64 for f in stamp["files"])
    assert state_mod.load().deployed[0].path == str(dest.resolve())
    assert exporters.stale_copies() == []

    # a newer plugin version makes the copy stale, and a reinstall drops files that disappeared
    (pkg / "files" / "lib" / "helper.mac").unlink()
    (pkg / "plugin.toml").write_text(_toml(doc | {"version": "1.3.0"}))
    info = next(site.glob("frontier_exporter_toytool-*.dist-info"))
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: frontier-exporter-toytool\nVersion: 1.3.0\n"
    )
    info.rename(site / "frontier_exporter_toytool-1.3.0.dist-info")
    stale = exporters.stale_copies()
    assert len(stale) == 1 and "copy is version 1.2.3" in stale[0]["problem"]
    again = plugins.install_exporter("toytool", case)
    assert again["replaced_version"] == "1.2.3" and not (dest / "lib" / "helper.mac").exists()

    # the user edits a script: uninstall keeps it and says so; --force removes it
    (dest / "run.mac").write_text("// edited by the user\n")
    out = plugins.uninstall_exporter("toytool")
    assert out["kept_modified"] == [str(dest / "run.mac")] and (dest / "run.mac").is_file()
    plugins.install_exporter("toytool", case, force=True)
    out = plugins.uninstall_exporter("toytool", case)
    assert not dest.exists() and not out["kept_modified"]
    assert state_mod.load().deployed == []

    # refuses to copy into / delete a directory it does not own
    (case / "toytool").mkdir()
    (case / "toytool" / "mine.txt").write_text("keep")
    with pytest.raises(ConnectorError, match="was not created by install-exporter"):
        plugins.install_exporter("toytool", case)
    with pytest.raises(ConnectorError, match="not touching it"):
        plugins.uninstall_exporter("toytool", case)
    with pytest.raises(ConnectorError, match="no exporter scripts for 'nope'"):
        plugins.exporter_dir("nope")




# -- installer ----------------------------------------------------------------------------------


def _wheel(directory: Path, doc: dict[str, Any], dist: str, module: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    whl = directory / f"{dist.replace('-', '_')}-{doc['version']}-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr(f"{module}/__init__.py", "")
        zf.writestr(f"{module}/plugin.toml", _toml(doc))
    return whl


def test_search_and_install_command_from_a_wheel_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheels = tmp_path / "wheels"
    _wheel(wheels, dict(GOOD) | {"summary": "reads toy files"}, "acme-frontier-toy", "toy_plugin")
    (wheels / "not-a-plugin-1.0-py3-none-any.whl").write_bytes(b"not a zip")
    rows = installer.search("toy", find_links=[str(wheels)])
    assert [r["id"] for r in rows] == ["toyfmt"]
    assert rows[0]["dist"] == "acme-frontier-toy" and rows[0]["status"] == "not-installed"
    assert rows[0]["connectors"] == ["toyfmt"] and rows[0]["origin"] == str(wheels)
    # the built-in catalog answers without any wheel directory, by format / vendor / connector id
    assert "openfoam" in {r["id"] for r in installer.search("openfoam")}
    assert {r["id"] for r in installer.search("exodus")} == {"meshio"}
    assert installer.search("openfoam")[0]["installed"] is True

    monkeypatch.setenv(installer.INSTALLER_ENV, "uv")
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    result = installer.install(
        ["toyfmt", "fluent"], find_links=[str(wheels)], no_index=True, dry_run=True
    )
    cmd = result["command"]
    assert cmd[1:3] == ["pip", "install"] and cmd[3:5] == ["--python", sys.executable]
    assert ["--find-links", str(wheels.resolve())] == cmd[5:7] and "--no-index" in cmd
    # ids become distribution names; a connector plugin pulls in its suggested exporter plugin
    assert cmd[-3:] == [
        "acme-frontier-toy",
        "frontier-connector-fluent",
        "frontier-exporter-fluent",
    ]
    slim = installer.install(["fluent"], with_suggested=False, dry_run=True)
    assert slim["command"][-1] == "frontier-connector-fluent"
    # paths and VCS URLs are passed through untouched
    url = "git+https://example.invalid/acme/frontier-toy.git@v1"
    assert installer.install([url], dry_run=True)["command"][-1] == url
    local = installer.install(
        [str(wheels / "acme_frontier_toy-1.2.3-py3-none-any.whl")], dry_run=True
    )
    assert local["command"][-1].endswith("acme_frontier_toy-1.2.3-py3-none-any.whl")

    un = installer.uninstall(["fluent"], dry_run=True)
    assert (
        un["command"][1:3] == ["pip", "uninstall"]
        and un["command"][-1] == "frontier-connector-fluent"
    )
    with pytest.raises(ConnectorError, match="refusing to uninstall the framework"):
        installer.uninstall(["frontier-connectors"], dry_run=True)
    with pytest.raises(ConnectorError, match="no installed plugin"):
        installer.uninstall(["toyfmt"], dry_run=True)


# -- CLI (the JSON documents the Rust crate parses) ---------------------------------------------






def test_wheel_index_matches_the_catalog_when_wheels_were_built() -> None:
    index = REPO / "dist" / "wheels" / "connectors" / installer.INDEX_FILE
    if not index.is_file():
        pytest.skip("wheels not built (run packages/frontier-connectors/tools/build_wheels.py)")
    built = {p["id"]: p for p in json.loads(index.read_text())["plugins"]}
    assert set(built) == {e["id"] for e in discovery.catalog()}
    for pid, row in built.items():
        assert (index.parent / row["wheel"]).is_file(), pid


class _ListingImporter(importers.BaseImporter):
    id = "toylist"
    name = "Toy listing"

    def plan(self, source: str, *, options: Any = None) -> list[importers.PlannedCall]:
        return []

    def fetch(self, source: str, dest: Path, *, options: Any = None) -> Path:
        doc = {"list": "cases", "count": 2, "items": [{"id": "c-1", "name": "wing"}, {"id": "c-2"}]}
        (dest / "listing.json").write_text(json.dumps(doc))
        return dest


def test_import_of_a_list_source_prints_the_listing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    importers.register(_ListingImporter())
    try:
        raw = str(tmp_path / "raw")
        assert main(["import", "toylist", "list:cases", "--download-dir", raw]) == 0
        text = capsys.readouterr().out
        assert "list:cases: 2 item(s)" in text and "c-1  name=wing" in text
        assert main(["import", "toylist", "list:cases", "--download-dir", raw, "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["listing"]["count"] == 2
    finally:
        importers.unregister("toylist")


# ---------------------------------------------------------------------------------- marketplace


def _minimal_manifest(**extra: object) -> dict[str, object]:
    return {
        "id": "x",
        "kind": "connector",
        "name": "X",
        "version": "0.2.0",
        "data_model_version": "1.1",
        "licence": "MIT",
        "connectors": {"x": "pkg.mod:X"},
    } | extra


def test_official_is_derived_from_the_author_company() -> None:
    from frontier_connectors.plugins.manifest import ManifestError, validate

    ours = validate(
        _minimal_manifest(author={"name": "DimensionLab", "company": "DimensionLab s.r.o."})
    )
    assert ours.official
    theirs = validate(_minimal_manifest(author={"name": "Jane", "company": "Acme GmbH"}))
    assert not theirs.official and theirs.author["company"] == "Acme GmbH"
    assert not validate(_minimal_manifest()).official
    assert validate(_minimal_manifest()).readme == "README.md"
    # a manifest cannot declare itself official, and marketplace fields are checked
    for bad in (
        {"author": {"official": "yes"}},
        {"category": "best"},
        {"repository": "git@github.com:x/y.git"},
        {"icon": "../../etc/passwd"},
        {"updated": "yesterday"},
        {"tags": "cfd"},
    ):
        with pytest.raises(ManifestError):
            validate(_minimal_manifest(**bad))


def test_marketplace_catalog_describes_every_plugin() -> None:
    path = PKG_ROOT / "src" / "frontier_connectors" / "plugins" / "marketplace.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "frontier-plugin-marketplace/1"
    assert {p["id"] for p in doc["plugins"]} == {e["id"] for e in discovery.catalog()}
    provided = {c for p in doc["plugins"] for c in p["connectors"]}
    assert {r["id"] for r in doc["connectors"]} == provided
    for p in doc["plugins"]:
        assert p["name"] and p["description"] and p["version"], p["id"]
        assert p["official"] and p["author"]["company"] == "DimensionLab s.r.o.", p["id"]
        assert p["category"] in ("solver", "file-format", "cloud-service", "exporter", "importer")
        assert p["readme"].startswith("# "), p["id"]
        # optional: present only when the manifest names a repository, never empty
        assert p.get("repository", "https://").startswith(("https://", "http://")), p["id"]
        if p["kind"] == "exporter":
            assert p["exporter"]["files"], p["id"]
    # nothing that depends on the machine the file was generated on
    assert not any({"missing", "status", "source"} & set(r) for r in doc["connectors"])


def test_a_locked_wheel_directory_is_verified_before_install(tmp_path: Path) -> None:
    """A wheel swapped or added after the directory was built stops the install."""
    import hashlib

    from frontier_connectors.plugins import installer

    d = tmp_path / "wheels"
    d.mkdir()
    good = d / "frontier_connector_demo-1.0.0-py3-none-any.whl"
    good.write_bytes(b"wheel bytes")
    # no lock: a plain find-links directory, not verified
    assert installer.verify_wheel_dir(d)["locked"] is False
    lock = {
        "schema": "frontier-plugins-lock/1",
        "wheels": [
            {
                "dist": "frontier-connector-demo",
                "version": "1.0.0",
                "file": good.name,
                "sha256": hashlib.sha256(b"wheel bytes").hexdigest(),
            }
        ],
    }
    (d / installer.LOCK_FILE).write_text(json.dumps(lock))
    assert installer.verify_wheel_dir(d) == {"locked": True, "directory": str(d), "wheels": 1}
    # altered content
    good.write_bytes(b"other bytes")
    with pytest.raises(ConnectorError, match="content differs"):
        installer.verify_wheel_dir(d)
    good.write_bytes(b"wheel bytes")
    # a wheel that was added afterwards
    (d / "frontier_connector_extra-9.9.9-py3-none-any.whl").write_bytes(b"x")
    with pytest.raises(ConnectorError, match="not listed"):
        installer.verify_wheel_dir(d)
    # install() refuses before running any installer
    with pytest.raises(ConnectorError, match="failed verification"):
        installer.install(["frontier-connector-demo"], find_links=[str(d)], no_index=True)
