# SPDX-License-Identifier: Apache-2.0
"""Prove plugin install / uninstall / enable / disable in a brand-new virtual environment.

    uv run python packages/frontier-connectors/tools/build_wheels.py --simple-index
    uv run python packages/frontier-connectors/tools/clean_venv_proof.py [--keep] [--offline-deps]

Creates a temporary ``uv venv``, installs only the framework from the local wheel directory,
then drives ``python -m frontier_connectors plugins ...`` exactly as a user would and prints a
transcript. Covered: (a) an index (the wheel directory served over HTTP as a PEP 503 index),
(b) a local wheel directory (``--find-links``), (c) a third-party plugin from a path and from a
git URL. Third-party dependencies (numpy, vtk, ...) come from PyPI unless ``--offline-deps``
points ``uv`` at an already populated cache only.

Exit code 0 means every step did what the transcript claims (each step is asserted).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WHEELS = REPO / "dist" / "wheels" / "connectors"

TOY_PYPROJECT = """\
[project]
name = "acme-frontier-toy"
version = "1.0.0"
description = "Third-party Frontier connector plugin used by the clean-venv proof"
requires-python = ">=3.11"
dependencies = ["frontier-connectors>=0.2"]

[project.entry-points."frontier.plugins"]
acme-toy = "acme_frontier_toy"

[project.entry-points."frontier.connectors"]
acme-toy = "acme_frontier_toy.connector:ToyConnector"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
acme_frontier_toy = ["plugin.toml"]
"""

TOY_MANIFEST = """\
schema = 1
id = "acme-toy"
kind = "connector"
name = "ACME toy point files"
summary = "x y z p text files"
version = "1.0.0"
data_model_version = "1.1"
vendor = "ACME"
formats = ["acme-xyzp"]
maturity = "verified-synthetic"
licence = "Apache-2.0"

[connectors]
acme-toy = "acme_frontier_toy.connector:ToyConnector"
"""

TOY_CONNECTOR = """\
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np

import frontier_connectors as fc


class ToyConnector(fc.BaseConnector):
    id = "acme-toy"
    name = "ACME toy point files"
    formats = ["acme-xyzp"]
    vendor = "ACME"
    extensions = [".xyzp"]
    maturity = "verified-synthetic"
    kinds = ["point_cloud"]

    def describe(self, path, *, options=None):
        return fc.SourceDescription(connector_id=self.id, path=Path(path), format="acme-xyzp")

    def read(self, path, *, zones=None, fields=None, times=None, options=None):
        a = np.loadtxt(path, ndmin=2)
        yield fc.RawSample(
            id=Path(path).stem,
            kind="point_cloud",
            points=a[:, :3].astype(np.float32),
            fields={"p": fc.as_field(a[:, 3], "point", units="Pa")},
        )
"""

LEGACY_VTK = """\
# vtk DataFile Version 3.0
two tetrahedra
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 5 float
0 0 0  1 0 0  0 1 0  0 0 1  1 1 1
CELLS 2 10
4 0 1 2 3
4 1 2 3 4
CELL_TYPES 2
10
10
POINT_DATA 5
SCALARS pressure float 1
LOOKUP_TABLE default
1 2 3 4 5
VECTORS velocity float
1 0 0  0 1 0  0 0 1  1 1 0  0 1 1
"""


class Proof:
    def __init__(self, work: Path, env: dict[str, str]) -> None:
        self.work = work
        self.env = env
        self.py = work / "venv" / "bin" / "python"
        self.step = 0

    def title(self, text: str) -> None:
        self.step += 1
        print(f"\n### {self.step}. {text}", flush=True)

    def run(self, cmd: list[str], *, expect: int | None = 0, show: int = 40) -> str:
        shown = " ".join(
            c.replace(str(self.work), "$WORK")
            .replace(str(WHEELS), "$WHEELS")
            .replace(str(REPO), ".")
            for c in cmd
        )
        print(f"$ {shown}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=self.env, check=False)
        out = (
            (proc.stdout + proc.stderr)
            .replace(str(self.work), "$WORK")
            .replace(str(WHEELS), "$WHEELS")
        )
        lines = out.rstrip().splitlines()
        for line in lines[:show]:
            print(line)
        if len(lines) > show:
            print(f"... ({len(lines) - show} more lines)")
        if expect is not None and proc.returncode != expect:
            raise SystemExit(f"FAILED: exit code {proc.returncode}, expected {expect}")
        return proc.stdout

    def fc(self, *args: str, expect: int | None = 0, show: int = 40) -> str:
        return self.run(
            [str(self.py), "-m", "frontier_connectors", *args], expect=expect, show=show
        )

    def plugin_ids(self) -> dict[str, str]:
        proc = subprocess.run(
            [str(self.py), "-m", "frontier_connectors", "plugins", "list", "--json"],
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        return {p["id"]: p["status"] for p in json.loads(proc.stdout)["plugins"]}

    def connector_ids(self) -> set[str]:
        proc = subprocess.run(
            [str(self.py), "-m", "frontier_connectors", "list", "--json"],
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        return {c["id"] for c in json.loads(proc.stdout)}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return None


@contextlib.contextmanager
def serve(directory: Path) -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/simple"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def check(condition: bool, what: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {what}")
    print(f"   ok: {what}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean-venv proof of the plugin manager.")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--python", default="3.12")
    args = ap.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required")
    if not (WHEELS / "plugins-index.json").is_file() or not (WHEELS / "simple").is_dir():
        raise SystemExit(f"build the wheels first: tools/build_wheels.py --simple-index ({WHEELS})")
    work = Path(tempfile.mkdtemp(prefix="frontier-plugin-proof-")).resolve()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("VIRTUAL_ENV", "PYTHONPATH", "FRONTIER_PLUGIN_INSTALLER", "UV_PROJECT_ENVIRONMENT")
    }
    env["FRONTIER_HOME"] = str(work / "frontier-home")
    env["FRONTIER_CACHE"] = str(work / "cache")
    p = Proof(work, env)
    wheels = str(WHEELS)
    try:
        # uv caches local wheels by name and version: drop cached copies of our own
        # distributions so a rebuilt 0.2.0 wheel is the one that gets installed
        dists = sorted({w.name.split("-")[0].replace("_", "-") for w in WHEELS.glob("*.whl")})
        subprocess.run([uv, "cache", "clean", *dists], capture_output=True, env=env, check=False)
        p.title("a new, empty virtual environment; only the framework is installed")
        p.run([uv, "venv", "--python", args.python, str(work / "venv")], show=3)
        p.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(p.py),
                "--find-links",
                wheels,
                "frontier-connectors",
            ],
            show=12,
        )
        p.fc("plugins", "list")
        check(p.plugin_ids() == {} and p.connector_ids() == set(), "no plugin, no connector")

        p.title("search the local wheel directory, then install two plugins from it (--find-links)")
        p.fc("plugins", "search", "fluent", "--find-links", wheels)
        p.fc("plugins", "install", "vtk", "fluent", "--find-links", wheels, show=12)
        p.fc("plugins", "list")
        ids = p.plugin_ids()
        check(
            ids == {"exporter-fluent": "enabled", "fluent": "enabled", "vtk": "enabled"},
            "vtk and fluent are installed; fluent brought its suggested exporter scripts",
        )
        p.fc("list")
        check(
            p.connector_ids() == {"vtk", "fluent-cff", "fluent-legacy", "fluent-live"},
            "exactly the connectors of the two plugins are loaded",
        )

        p.title("convert a sample with an installed plugin")
        sample = work / "two_tets.vtk"
        sample.write_text(LEGACY_VTK)
        p.fc("detect", str(sample))
        p.fc("convert", str(sample), "--out", str(work / "ds"), "--domain", "cfd")
        manifest = json.loads((work / "ds" / "manifest.json").read_text())
        check(
            manifest["format"] == "frontier-dataset" and manifest["minor"] == 1,
            "a frontier-dataset v1.1 was written",
        )
        check(len(manifest["samples"]) == 1, "one sample in the manifest")

        p.title("disable without uninstalling, then enable again")
        p.fc("plugins", "disable", "fluent")
        p.fc("plugins", "list")
        check(p.plugin_ids()["fluent"] == "disabled", "fluent is installed but disabled")
        check(p.connector_ids() == {"vtk"}, "its connectors are gone from the registry")
        p.run(["cat", str(work / "frontier-home" / "plugins.toml")])
        p.fc("describe", str(sample), "--connector", "fluent-cff", expect=2)
        p.fc("plugins", "enable", "fluent")
        check("fluent-cff" in p.connector_ids(), "fluent connectors are back")

        p.title("copy the vendor-side exporter scripts next to a case, with a version stamp")
        p.fc("plugins", "install-exporter", "fluent", "--to", str(work / "case"))
        p.run(["ls", "-A", str(work / "case" / "fluent")])
        check((work / "case" / "fluent" / ".frontier-exporter.json").is_file(), "stamp written")
        p.fc("plugins", "uninstall-exporter", "fluent")
        check(not (work / "case" / "fluent").exists(), "the copy is removed again")

        p.title("uninstall one plugin")
        p.fc("plugins", "uninstall", "fluent")
        p.fc("plugins", "list")
        check(
            p.plugin_ids() == {"exporter-fluent": "enabled", "vtk": "enabled"},
            "fluent is gone, vtk and the independent exporter plugin remain",
        )
        p.fc("describe", str(sample), "--connector", "fluent-cff", expect=2)
        check(p.connector_ids() == {"vtk"}, "only the vtk connector is loaded")
        p.fc("convert", str(sample), "--out", str(work / "ds2"))
        check((work / "ds2" / "manifest.json").is_file(), "conversion still works")

        p.title("plugins added in 0.2: a cloud importer (dry run, no credentials) and Radioss")
        p.fc("plugins", "install", "rescale", "optistruct", "--find-links", wheels, show=12)
        ids = p.plugin_ids()
        check(
            {"rescale", "optistruct", "exporter-optistruct", "nastran"} <= set(ids),
            "rescale and optistruct installed; optistruct brought nastran and its exporter",
        )
        check({"rescale", "optistruct", "radioss", "nastran"} <= p.connector_ids(), "connectors")
        p.fc("importers")
        out = p.fc("import", "rescale", "AbCdEf", "--dry-run")
        check("/api/v2/jobs/AbCdEf/" in out, "the dry run lists the REST calls, contacting nothing")
        p.fc("import", "rescale", "AbCdEf", "--out", str(work / "ds-rescale"), expect=2)
        p.fc("plugins", "uninstall", "rescale", "optistruct", "exporter-optistruct", "nastran")
        check(
            p.plugin_ids() == {"exporter-fluent": "enabled", "vtk": "enabled"},
            "all four are gone again",
        )

        p.title("install from an index (the wheel directory served as a PEP 503 index over HTTP)")
        with serve(WHEELS) as index:
            p.fc("plugins", "install", "ensight", "--extra-index-url", index, show=12)
        check(p.plugin_ids().get("ensight") == "enabled", "ensight came from the index")

        p.title("a third-party plugin from a local path and from a git URL")
        toy = work / "acme-frontier-toy"
        (toy / "src" / "acme_frontier_toy").mkdir(parents=True)
        (toy / "pyproject.toml").write_text(TOY_PYPROJECT)
        (toy / "src" / "acme_frontier_toy" / "__init__.py").write_text("")
        (toy / "src" / "acme_frontier_toy" / "plugin.toml").write_text(TOY_MANIFEST)
        (toy / "src" / "acme_frontier_toy" / "connector.py").write_text(TOY_CONNECTOR)
        p.fc("plugins", "install", str(toy), "--find-links", wheels, show=12)
        check(p.plugin_ids().get("acme-toy") == "enabled", "third-party plugin loaded from a path")
        pts = work / "probe.xyzp"
        pts.write_text("0 0 0 1.0\n1 0 0 2.0\n0 1 0 3.0\n")
        p.fc("convert", str(pts), "--out", str(work / "ds3"))
        p.fc("plugins", "uninstall", "acme-toy")
        check("acme-toy" not in p.plugin_ids(), "third-party plugin uninstalled")
        git_env = env | {
            "GIT_AUTHOR_NAME": "proof",
            "GIT_AUTHOR_EMAIL": "proof@example.invalid",
            "GIT_COMMITTER_NAME": "proof",
            "GIT_COMMITTER_EMAIL": "proof@example.invalid",
        }
        for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "toy plugin"]):
            subprocess.run(["git", *cmd], cwd=toy, env=git_env, check=True, capture_output=True)
        p.fc("plugins", "install", f"git+file://{toy}", "--find-links", wheels, show=12)
        check(p.plugin_ids().get("acme-toy") == "enabled", "third-party plugin loaded from git")

        p.title("doctor")
        p.fc("plugins", "doctor", show=30)
        print("\nPROOF PASSED")
        return 0
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
