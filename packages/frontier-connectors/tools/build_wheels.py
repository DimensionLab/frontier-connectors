# SPDX-License-Identifier: Apache-2.0
"""Build the wheels of the framework and of every plugin into one directory.

    uv run python packages/frontier-connectors/tools/build_wheels.py     # dist/wheels/connectors
    uv run python packages/frontier-connectors/tools/build_wheels.py --out DIR --simple-index

The directory is a ready ``--find-links`` source (and, copied to a USB stick together with the
wheels of the third-party dependencies, an air-gapped bundle):

    python -m frontier_connectors plugins install fluent --find-links dist/wheels/connectors

``plugins-index.json`` (the catalog of what was built) is written next to the wheels; with
``--simple-index`` a PEP 503 "simple" tree is written to ``<out>/simple`` so the same directory
can be served as a package index (``python -m http.server`` in ``<out>``, then
``--index-url http://localhost:8000/simple``).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PLUGINS = ROOT.parent / "connectors"


def _build(project: Path, out: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to build the wheels (https://docs.astral.sh/uv/)")
    proc = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out), str(project)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"building {project} failed:\n{proc.stdout}{proc.stderr}")


def _simple_index(out: Path) -> None:
    simple = out / "simple"
    if simple.exists():
        shutil.rmtree(simple)
    by_dist: dict[str, list[Path]] = {}
    for whl in sorted(out.glob("*.whl")):
        by_dist.setdefault(whl.name.split("-")[0].replace("_", "-").lower(), []).append(whl)
    simple.mkdir()
    links = "".join(f'<a href="{d}/">{d}</a><br>\n' for d in sorted(by_dist))
    (simple / "index.html").write_text(f"<!DOCTYPE html><html><body>\n{links}</body></html>\n")
    for dist, wheels in by_dist.items():
        d = simple / dist
        d.mkdir()
        rows = ""
        for whl in wheels:
            digest = hashlib.sha256(whl.read_bytes()).hexdigest()
            name = html.escape(whl.name)
            up = "../" * 2  # from simple/<project>/ back to the wheel directory
            rows += f'<a href="{up}{name}#sha256={digest}">{name}</a><br>\n'
        (d / "index.html").write_text(f"<!DOCTYPE html><html><body>\n{rows}</body></html>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build framework + plugin wheels into one directory.")
    ap.add_argument("--out", default=str(REPO / "dist" / "wheels" / "connectors"))
    ap.add_argument("--simple-index", action="store_true", help="also write a PEP 503 index tree")
    ap.add_argument("--keep", action="store_true", help="do not delete wheels already in --out")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.keep:
        for old in out.glob("frontier_*.whl"):
            old.unlink()
    projects = [ROOT, *sorted(p.parent for p in PLUGINS.glob("*/pyproject.toml"))]
    for i, project in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] {project.name}", flush=True)
        _build(project, out)

    sys.path.insert(0, str(ROOT / "src"))
    from frontier_connectors.plugins.installer import INDEX_FILE, wheel_manifest

    rows = [e for e in map(wheel_manifest, sorted(out.glob("*.whl"))) if e]
    index = {
        "schema": "frontier-plugin-catalog/1",
        "generated_by": "tools/build_wheels.py",
        "plugins": sorted(rows, key=lambda r: r["id"]),
    }
    (out / INDEX_FILE).write_text(json.dumps(index, indent=1) + "\n")
    if args.simple_index:
        _simple_index(out)
    wheels = sorted(out.glob("*.whl"))
    size = sum(w.stat().st_size for w in wheels)
    print(f"{len(wheels)} wheels ({len(rows)} plugins), {size / 1e6:.1f} MB in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
