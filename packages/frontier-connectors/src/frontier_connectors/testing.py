# SPDX-License-Identifier: Apache-2.0
"""Helpers for testing connectors: synthetic meshes, a conformance check, sample downloads.

Third-party plugins can use :func:`check_connector` in their own test-suite::

    from frontier_connectors.testing import check_connector
    report = check_connector(MyConnector(), "tests/data/case.xyz", tmp_path)
    assert report.n_steps == 10
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors.convert import ConvertResult, frontier_data_available, to_frontier_dataset
from frontier_connectors.model import Connector, RawSample, SourceDescription

NETWORK_ENV = "FRONTIER_NETWORK_TESTS"


def check_plugin_distribution(name: str) -> None:
    """Check an installed wheel's manifest, entry points and exporter resources."""
    import importlib.metadata
    import importlib.resources

    from frontier_connectors.plugins.manifest import validate

    dist = importlib.metadata.distribution(name)
    entries = list(dist.entry_points)
    plugin_entries = [ep for ep in entries if ep.group == "frontier.plugins"]
    if not plugin_entries:
        # Meta-distributions have requirements but no plugin entry points.
        assert name in {"frontier-connectors-all", "frontier-connectors-oss"}
        for requirement in dist.requires or []:
            from packaging.requirements import Requirement

            req = Requirement(requirement)
            installed = importlib.metadata.version(req.name)
            assert installed in req.specifier, f"{req}: installed {installed}"
        return
    assert len(plugin_entries) == 1, f"{name}: expected one plugin manifest entry point"
    entry = plugin_entries[0]
    package = entry.load()
    import tomllib

    files = importlib.resources.files(package)
    manifest = validate(tomllib.loads(files.joinpath("plugin.toml").read_text()))
    assert manifest.id == entry.name
    assert manifest.version == dist.version
    for ep in entries:
        if ep.group in {"frontier.connectors", "frontier.importers", "frontier.exporters"}:
            assert ep.load() is not None, f"{name}: broken entry point {ep.name}"
    if manifest.exporter:
        directory = files.joinpath(manifest.exporter.files)
        assert directory.is_dir() and any(directory.iterdir()), f"{name}: exporter files missing"


def network_enabled() -> bool:
    """Network-marked tests run only when ``FRONTIER_NETWORK_TESTS=1``."""
    return os.environ.get(NETWORK_ENV, "") not in ("", "0", "false", "no")


def sample_cache() -> Path:
    root = os.environ.get("FRONTIER_CONNECTORS_CACHE")
    p = Path(root) if root else Path.home() / ".cache" / "frontier-connectors" / "samples"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_sample(url: str, name: str | None = None, *, sha256: str | None = None) -> Path:
    """Download a public sample file once into the cache (never into the repository).

    Each download gets its own directory so that untrusted files never sit next to code.
    """
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    dest_dir = sample_cache() / key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (name or url.rsplit("/", 1)[-1].split("?")[0])
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        from frontier_connectors.network import require_online

        require_online("downloading a public sample file", url=url)
        req = urllib.request.Request(url, headers={"User-Agent": "frontier-connectors-tests"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    if sha256 is not None:
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != sha256:
            raise RuntimeError(f"{dest} sha256 {got} != expected {sha256}")
    return dest


def open_store(path: Path | str) -> Any:
    """Open a zarr store read-only (typed ``Any`` so tests can index freely)."""
    import zarr

    return zarr.open_group(str(path), mode="r")


# -- synthetic geometry -----------------------------------------------------------------------


def hex_grid(
    nx: int = 3, ny: int = 2, nz: int = 2, size: tuple[float, float, float] = (1.0, 1.0, 1.0)
):
    """Structured block of ``nx*ny*nz`` hexahedra. Returns ``(points (N,3) f64, cells (K,8))``
    with VTK hexahedron node order; points are ordered x-fastest."""
    xs = np.linspace(0.0, size[0], nx + 1)
    ys = np.linspace(0.0, size[1], ny + 1)
    zs = np.linspace(0.0, size[2], nz + 1)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    def pid(i: Any, j: Any, k: Any) -> Any:
        return i + (nx + 1) * (j + (ny + 1) * k)

    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    cells = np.stack(
        [
            pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k),
            pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1),
        ],
        axis=1,
    )  # fmt: skip
    return points, cells.astype(np.int64)


def tet_grid(nx: int = 2, ny: int = 2, nz: int = 2):
    """Tetrahedral mesh of a unit box (each hexahedron split into 6 tets)."""
    points, hexes = hex_grid(nx, ny, nz)
    split = np.array(
        [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    )
    tets = hexes[:, split].reshape(-1, 4)
    return points, tets


def quad_grid(nx: int = 4, ny: int = 3, size: tuple[float, float] = (1.0, 1.0)):
    """Planar quad mesh in z = 0. Returns ``(points (N,3), quads (K,4))``."""
    xs = np.linspace(0.0, size[0], nx + 1)
    ys = np.linspace(0.0, size[1], ny + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a = i + (nx + 1) * j
    quads = np.stack([a, a + 1, a + nx + 2, a + nx + 1], axis=1)
    return points, quads.astype(np.int64)


def tri_grid(nx: int = 4, ny: int = 3):
    """Planar triangle mesh in z = 0."""
    points, quads = quad_grid(nx, ny)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return points, tris


def analytic_fields(points: np.ndarray, t: float = 0.0) -> dict[str, np.ndarray]:
    """Deterministic smooth test fields: scalar ``p`` and vector ``U`` at time ``t``."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    p = np.sin(2.0 * x + t) + 0.5 * y - 0.25 * z
    u = np.stack([1.0 + y * (1.0 + t), -x + 0.1 * t, 0.5 * z + t], axis=1)
    return {"p": p, "U": u}


def make_ugrid(
    points: np.ndarray,
    cells: np.ndarray,
    vtk_type: int,
    point_data: Mapping[str, np.ndarray] | None = None,
    cell_data: Mapping[str, np.ndarray] | None = None,
) -> Any:
    """Build a ``vtkUnstructuredGrid`` with one cell type."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    grid = vtkUnstructuredGrid()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    grid.SetPoints(pts)
    for row in np.asarray(cells):
        grid.InsertNextCell(int(vtk_type), len(row), [int(v) for v in row])
    for name, arr in (point_data or {}).items():
        a = numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
        a.SetName(name)
        grid.GetPointData().AddArray(a)
    for name, arr in (cell_data or {}).items():
        a = numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
        a.SetName(name)
        grid.GetCellData().AddArray(a)
    return grid


def write_vtk_xml(obj: Any, path: Path | str, *, binary: bool = True) -> Path:
    """Write any VTK data object with the matching XML writer (extension must fit the type)."""
    from vtkmodules.vtkIOXML import vtkXMLDataSetWriter, vtkXMLMultiBlockDataWriter

    path = Path(path)
    w = vtkXMLMultiBlockDataWriter() if path.suffix == ".vtm" else vtkXMLDataSetWriter()
    w.SetFileName(str(path))
    w.SetInputData(obj)
    if not binary:
        w.SetDataModeToAscii()
    if w.Write() != 1:
        raise RuntimeError(f"VTK failed to write {path}")
    return path


# -- conformance ------------------------------------------------------------------------------


@dataclass
class ConformanceReport:
    description: SourceDescription
    samples: dict[str, list[RawSample]] = field(default_factory=dict)
    results: dict[str, ConvertResult] = field(default_factory=dict)

    @property
    def sample_ids(self) -> list[str]:
        return list(self.samples)

    @property
    def n_steps(self) -> int:
        return max((len(v) for v in self.samples.values()), default=0)

    def first(self, sample_id: str | None = None) -> RawSample:
        return self.samples[sample_id or self.sample_ids[0]][0]


def check_connector(
    connector: Any,
    path: Path | str,
    tmp_path: Path | str,
    *,
    zones: list[str] | None = None,
    fields: list[str] | None = None,
    times: Any = None,
    options: Mapping[str, Any] | None = None,
    min_confidence: float = 0.5,
    convert: bool = True,
    time_as_samples: bool = False,
) -> ConformanceReport:
    """Run the full contract against a real source and assert the invariants every connector
    must satisfy. Returns what was read so the caller can assert the physics.

    Checks: protocol attributes; ``detect`` recognises the path; ``describe`` is JSON-clean;
    every ``RawSample`` validates; zones/fields/times read are the ones described; step-form
    times are increasing; the samples convert to a dataset with the streaming zarr writer and,
    when ``frontier_data`` is installed, with its writers too, and that dataset validates.
    """
    import json

    path = Path(path)
    assert isinstance(connector, Connector), "object does not satisfy the Connector protocol"
    for attr in ("id", "name", "formats", "vendor", "license"):
        assert getattr(connector, attr), f"connector.{attr} must be set"
    det = connector.detect(path)
    assert det is not None, f"{connector.id}.detect({path}) returned None"
    assert det.connector_id == connector.id
    assert 0.0 < det.confidence <= 1.0
    assert det.confidence >= min_confidence, f"confidence {det.confidence} < {min_confidence}"

    kw: dict[str, Any] = {"options": options} if options else {}
    desc = connector.describe(path, **kw)
    assert desc.connector_id == connector.id
    json.dumps(desc.to_dict())  # must be JSON-clean
    assert desc.zones, "describe() returned no zones"
    zone_names = {z.name for z in desc.zones}
    assert len(zone_names) == len(desc.zones), "zone names must be unique"

    report = ConformanceReport(description=desc)
    last_time: dict[str, float] = {}
    for s in connector.read(path, zones=zones, fields=fields, times=times, **kw):
        s.validate()
        if s.zone is not None:
            assert s.zone in zone_names, f"read() yielded undescribed zone {s.zone!r}"
        if s.time is not None:
            assert s.id not in last_time or s.time > last_time[s.id], "times must increase"
            last_time[s.id] = s.time
            assert not desc.time.steady, "read() yielded time steps but describe() says steady"
        if not report.samples.get(s.id) and s.kind != "grid":
            assert s.points is not None, "first sample of an id must carry points"
        for name, f in s.fields.items():
            assert np.asarray(f.data[:1]).dtype.kind in "fiub", f"field {name!r} is not numeric"
        report.samples.setdefault(s.id, []).append(s)
    assert report.samples, "read() yielded nothing"

    if convert:
        backends = ["zarr"] + (["frontier_data"] if frontier_data_available() else [])
        for backend in backends:
            flat = [s for group in report.samples.values() for s in group]
            out = Path(tmp_path) / f"dataset-{connector.id}-{backend}"
            res = to_frontier_dataset(
                iter(flat),
                out,
                description=desc,
                backend="auto" if backend == "frontier_data" else "zarr",
                overwrite=True,
                time_as_samples=time_as_samples,
            )
            report.results[backend] = res
            manifest = json.loads((out / "manifest.json").read_text())
            assert manifest["format"] == "frontier-dataset" and manifest["minor"] == 1
            assert len(manifest["samples"]) == len(res.sample_ids)
            if frontier_data_available():
                validate_with_frontier_data(out)
    return report


def validate_with_frontier_data(out: Path) -> None:
    """Open the dataset with the reference reader and run its validator when present."""
    from frontier_data.dataset import Dataset

    ds = Dataset(out)
    assert len(ds) > 0
    try:
        from frontier_data.validate import validate_dataset
    except ImportError:
        return
    rep = validate_dataset(out)
    errors = [i for i in getattr(rep, "issues", []) if getattr(i, "level", "") == "error"]
    assert not errors, f"frontier_data.validate found errors: {errors[:5]}"
