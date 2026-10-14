# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import frontier_connectors as fc
from frontier_connectors import meshutil, registry, units
from frontier_connectors.cli import main
from frontier_connectors.convert import frontier_data_available
from frontier_connectors.errors import SelectionError
from frontier_connectors.model import match_extension
from frontier_connectors.select import match_names, parse_time_selection, select_time_indices
from frontier_connectors.testing import analytic_fields, hex_grid, open_store, quad_grid

BACKENDS = ["zarr"] + (["auto"] if frontier_data_available() else [])


def _mesh_steps(n_steps: int = 4, moving: bool = False) -> Iterator[fc.RawSample]:
    points, cells = hex_grid(3, 2, 2)
    types = np.full(len(cells), meshutil.VTK_HEXAHEDRON, dtype=np.uint8)
    for i in range(n_steps):
        t = 0.1 * i
        f = analytic_fields(points, t)
        cell_p = f["p"][cells].mean(axis=1)
        yield fc.RawSample(
            id="case",
            kind="mesh",
            zone="internal",
            points=(points + (0.01 * i if moving else 0.0)) if (i == 0 or moving) else None,
            cells=cells if i == 0 else None,
            cell_types=types if i == 0 else None,
            fields={
                "p": fc.as_field(f["p"], "point", units="Pa"),
                "U": fc.as_field(f["U"], "point", units="m/s"),
                "p_cell": fc.as_field(cell_p, "cell"),
            },
            static={"wall_distance": fc.as_field(points[:, 1], "point")} if i == 0 else {},
            time=t,
            time_index=i,
            metadata={"solver": "synthetic", "parameters": {"Re": 100.0}, "source": "unit-test"},
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_streamed_transient_mesh_roundtrip(tmp_path: Path, backend: str) -> None:
    out = tmp_path / "ds"
    res = fc.to_frontier_dataset(_mesh_steps(4), out, backend=backend, domain="cfd")
    assert res.sample_ids == ["case"] and res.n_steps["case"] == 4
    assert res.backends["case"] == "zarr-stream"  # step form always streams
    g = open_store(out / "samples/case/mesh.zarr")
    points, cells = hex_grid(3, 2, 2)
    assert g["points"].shape == (len(points), 3)
    assert g["cells"].shape == cells.shape and g["cell_types"][0] == 12
    assert g["fields/U"].shape == (4, len(points), 3)
    assert g["fields/U"].attrs["dims"] == ["time", "point", "component"]
    assert g["fields/U"].chunks[0] == 16  # tiny steps are grouped, see the next test
    assert g["cell_fields/p_cell"].shape == (4, len(cells), 1)
    np.testing.assert_allclose(g["times"][...], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(g["fields/p"][2, :, 0], analytic_fields(points, 0.2)["p"], rtol=1e-6)
    assert g["static/wall_distance"].shape == (len(points), 1)
    assert g.attrs["temporal"] == "transient" and g.attrs["n_steps"] == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["minor"] == 1 and manifest["temporal"] == "transient"
    assert manifest["sample_kind"] == "mesh" and manifest["domain"] == "cfd"
    assert manifest["time"]["n_steps"] == 4 and manifest["time"]["dt"] == pytest.approx(0.1)
    meta = json.loads((out / "samples/case/meta.json").read_text())
    assert meta["solver"] == "synthetic" and meta["parameters"] == {"Re": 100.0}
    assert meta["provenance"]["ingest_tool"] == "frontier-connectors"
    if frontier_data_available():
        from frontier_data.dataset import Dataset
        from frontier_data.validate import validate_dataset

        s = Dataset(out).sample("case")
        assert s.is_transient and s.n_steps == 4
        assert validate_dataset(out).ok, validate_dataset(out).to_dict()


def test_one_step_per_chunk_for_real_sized_steps(tmp_path: Path) -> None:
    n = 8000  # 8000 x 3 float32 = 96 kB per step: above the small-step threshold

    def steps() -> Iterator[fc.RawSample]:
        for i in range(3):
            yield fc.RawSample(
                id="cloud",
                kind="point_cloud",
                points=np.zeros((n, 3)) if i == 0 else None,
                fields={"U": fc.as_field(np.full((n, 3), float(i)), "point")},
                time=float(i),
                time_index=i,
            )

    fc.to_frontier_dataset(steps(), tmp_path / "ds", backend="zarr")
    a = open_store(tmp_path / "ds/samples/cloud/points.zarr")["fields/U"]
    assert a.chunks[0] == 1 and float(a[2, 0, 0]) == 2.0


def test_moving_mesh_writes_points_t(tmp_path: Path) -> None:
    fc.to_frontier_dataset(_mesh_steps(3, moving=True), tmp_path / "ds", backend="zarr")
    g = open_store(tmp_path / "ds/samples/case/mesh.zarr")
    assert g["points_t"].shape == (3, g["points"].shape[0], 3)
    np.testing.assert_allclose(g["points_t"][2] - g["points"][...], 0.02, atol=1e-6)


def test_same_counts_different_connectivity_is_refused(tmp_path: Path) -> None:
    points, cells = hex_grid(2, 1, 1)

    def steps() -> Iterator[fc.RawSample]:
        for i in range(2):
            yield fc.RawSample(
                id="remesh",
                kind="mesh",
                points=points,
                cells=cells if i == 0 else cells[::-1],
                cell_types=np.full(len(cells), 12, dtype=np.uint8),
                fields={"p": fc.as_field(points[:, 0], "point")},
                time=float(i),
                time_index=i,
            )

    with pytest.raises(fc.ConnectorError, match="different connectivity"):
        fc.to_frontier_dataset(steps(), tmp_path / "a", backend="zarr")
    assert not (tmp_path / "a").exists()  # nothing half-written is left behind


def test_topology_change_needs_time_as_samples(tmp_path: Path) -> None:
    def steps() -> Iterator[fc.RawSample]:
        for i, n in enumerate((3, 4)):
            points, cells = hex_grid(n, 2, 2)
            yield fc.RawSample(
                id="amr",
                kind="mesh",
                points=points,
                cells=cells,
                cell_types=np.full(len(cells), 12, dtype=np.uint8),
                fields={"p": fc.as_field(points[:, 0], "point")},
                time=float(i),
                time_index=i,
            )

    with pytest.raises(fc.ConnectorError, match="time-as-samples"):
        fc.to_frontier_dataset(steps(), tmp_path / "a", backend="zarr")
    res = fc.to_frontier_dataset(steps(), tmp_path / "b", backend="zarr", time_as_samples=True)
    assert res.sample_ids == ["amr_t00000", "amr_t00001"]
    meta = json.loads((tmp_path / "b/samples/amr_t00001/meta.json").read_text())
    assert meta["group"] == "amr" and meta["time"]["t0"] == 1.0 and meta["temporal"] == "steady"


@pytest.mark.parametrize("backend", BACKENDS)
def test_steady_surface_grid_and_points(tmp_path: Path, backend: str) -> None:
    points, quads = quad_grid(4, 3)
    faces, parent = meshutil.triangulate(quads)
    assert len(faces) == 2 * len(quads)
    surf = fc.RawSample(
        id="wall",
        kind="surface",
        points=points,
        faces=faces,
        fields={
            "p": fc.as_field(points[:, 0], "point"),
            "wss": fc.as_field(np.ones((len(quads), 3))[parent], "cell"),
        },
        metadata={"solver": "synthetic"},
    )
    x, y = np.linspace(0, 1, 6), np.linspace(0, 2, 5)
    times = np.array([0.0, 0.5, 1.0])
    data = np.random.default_rng(0).normal(size=(3, 6, 5, 2)).astype(np.float32)
    grid = fc.RawSample(
        id="grid",
        kind="grid",
        coords={"x": x, "y": y},
        times=times,
        fields={
            "u": fc.FieldData(data, ["time", "x", "y", "component"], "grid", units="m/s"),
        },
        static={
            "mask": fc.FieldData(np.ones((6, 5, 1), np.float32), ["x", "y", "component"], "grid")
        },
    )
    cloud = fc.RawSample(
        id="probes",
        kind="point_cloud",
        points=points[:5],
        fields={"T": fc.as_field(np.arange(5.0), "point", units="K")},
    )
    out = tmp_path / "ds"
    res = fc.to_frontier_dataset([surf, grid, cloud], out, backend=backend)
    assert sorted(res.sample_ids) == ["grid", "probes", "wall"]
    if backend == "auto":
        assert set(res.backends.values()) == {"frontier_data"}, res.warnings
    s = open_store(out / "samples/wall/surface.zarr")
    assert s["faces"].shape == (24, 3) and s["cell_area"].shape == (24,)
    assert float(np.sum(s["cell_area"][...])) == pytest.approx(1.0, rel=1e-5)
    assert s["cell_fields/wss"].shape == (24, 3)
    g = open_store(out / "samples/grid/grid.zarr")
    assert g["fields/u"].shape == (3, 6, 5, 2)
    assert list(g["fields/u"].attrs["dims"]) == ["time", "x", "y", "component"]
    np.testing.assert_allclose(g["fields/u"][1], data[1])
    np.testing.assert_allclose(g["coords/y"][...], y)
    assert g.attrs["spacing"] == pytest.approx([0.2, 0.5])
    p = open_store(out / "samples/probes/points.zarr")
    assert p["fields/T"].shape == (5, 1)
    if frontier_data_available():
        from frontier_data.validate import validate_dataset

        rep = validate_dataset(out)
        assert rep.ok, rep.to_dict()


def test_lazy_block_is_streamed(tmp_path: Path) -> None:
    import h5py

    with h5py.File(tmp_path / "lazy.h5", "w") as f:
        f["u"] = np.arange(5 * 4 * 3, dtype=np.float64).reshape(5, 4, 3, 1)
    with h5py.File(tmp_path / "lazy.h5", "r") as f:
        s = fc.RawSample(
            id="lazy",
            kind="grid",
            coords={"x": np.arange(4.0), "y": np.arange(3.0)},
            times=np.arange(5.0),
            fields={"u": fc.FieldData(f["u"], ["time", "x", "y", "component"], "grid")},
        )
        res = fc.to_frontier_dataset([s], tmp_path / "ds")
    assert res.backends["lazy"] == "zarr-stream"
    g = open_store(tmp_path / "ds/samples/lazy/grid.zarr")
    assert g["fields/u"].shape == (5, 4, 3, 1) and float(g["fields/u"][4, 3, 2, 0]) == 59.0


def test_long_tiny_series_is_buffered_and_chunked(tmp_path: Path) -> None:
    def steps() -> Iterator[fc.RawSample]:
        buf = np.zeros((1, 3))
        for i in range(3000):
            buf[:] = i  # a reader that reuses its buffer must not corrupt earlier steps
            yield fc.RawSample(
                id="probe",
                kind="point_cloud",
                points=np.zeros((1, 3)) if i == 0 else None,
                fields={"force": fc.FieldData(buf, ["point", "component"], "point")}
                if i != 7
                else {},
                time=float(i),
                time_index=i,
            )

    fc.to_frontier_dataset(steps(), tmp_path / "ds", backend="zarr")
    g = open_store(tmp_path / "ds/samples/probe/points.zarr")
    a = g["fields/force"]
    assert a.shape == (3000, 1, 3) and a.chunks[0] == 16
    data = a[...]
    np.testing.assert_allclose(data[[0, 6, 8, 2999], 0, 0], [0, 6, 8, 2999])
    assert np.isnan(data[7]).all()


def test_overwrite_guard(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "precious.txt").write_text("keep")
    with pytest.raises(fc.ConnectorError, match="not empty"):
        fc.to_frontier_dataset(_mesh_steps(1), tmp_path / "x")
    with pytest.raises(fc.ConnectorError, match="refusing"):
        fc.to_frontier_dataset(_mesh_steps(1), tmp_path / "x", overwrite=True)
    assert (tmp_path / "x" / "precious.txt").exists()


def test_selection_helpers() -> None:
    names = ["internalMesh", "inlet", "outlet", "wall_top", "wall_bottom"]
    assert match_names(names, None) == names
    assert match_names(names, ["wall_*", "inlet"]) == ["inlet", "wall_top", "wall_bottom"]
    assert match_names(names, ["!wall_*"]) == ["internalMesh", "inlet", "outlet"]
    with pytest.raises(SelectionError, match="available zones"):
        match_names(names, ["nope"])
    times = [0.0, 0.1, 0.2, 0.3]
    assert select_time_indices(times, None) == [0, 1, 2, 3]
    assert select_time_indices(times, "last") == [3]
    assert select_time_indices(times, -2) == [2]
    assert select_time_indices(times, slice(0, None, 2)) == [0, 2]
    assert select_time_indices(times, [0.11, 0.29]) == [1, 3]
    assert select_time_indices([], "last") == [0]
    assert parse_time_selection("0:10:2") == slice(0, 10, 2)
    assert parse_time_selection("t=0.1,0.2") == [0.1, 0.2]
    assert parse_time_selection("1,3") == [1, 3]
    with pytest.raises(SelectionError):
        select_time_indices(times, 9)


def test_units() -> None:
    v, si = units.to_si([1.0, 2.0], "mm")
    assert si == "m" and np.allclose(v, [1e-3, 2e-3])
    assert units.to_si(20.0, "degC")[0] == pytest.approx(293.15)
    assert units.si_factor("mm/s^2")[0] == pytest.approx(1e-3)
    assert units.length_scale("in", "mm") == pytest.approx(25.4)
    assert units.foam_dimensions_to_unit([0, 2, -2, 0, 0, 0, 0]) == "m^2/s^2"
    assert units.foam_dimensions_to_unit([1, -1, -2, 0, 0, 0, 0]) == "kg/m/s^2"
    assert units.normalize_unit("Meter") == "m"
    points, cells = hex_grid(1, 1, 1)
    s = fc.RawSample(id="a", kind="point_cloud", points=points * 1000.0)
    units.convert_sample_units(s, length="m", source_length="mm")
    assert s.points is not None and s.points.max() == pytest.approx(1.0)


def test_meshutil() -> None:
    conn = np.array([0, 1, 2, 3, 4, 5, 6])
    cells = meshutil.pad_cells(conn, np.array([0, 3, 7]))
    assert cells.tolist() == [[0, 1, 2, -1], [3, 4, 5, 6]]
    faces, parent = meshutil.triangulate(cells)
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6]] and parent.tolist() == [0, 1, 1]
    c, t = meshutil.pad_cell_blocks([(10, np.zeros((2, 4), int)), (12, np.zeros((1, 8), int))])
    assert c.shape == (3, 8) and t.tolist() == [10, 10, 12] and (c[0, 4:] == -1).all()
    assert meshutil.census(t) == {"tetra": 2, "hexahedron": 1}
    assert match_extension(Path("a/B.CAS.H5"), [".h5", ".cas.h5"]) == ".cas.h5"
    assert meshutil.default_sample_id("x/run 1.cas.h5") == "run_1"


class _ToyConnector(fc.BaseConnector):
    id = "toy"
    name = "Toy"
    formats = ["toy"]
    vendor = "tests"
    extensions = [".toy"]
    maturity = "verified-synthetic"

    def describe(self, path: Path, *, options=None) -> fc.SourceDescription:
        return fc.SourceDescription(self.id, path, zones=[fc.ZoneInfo("main", "point_cloud")])

    def read(self, path: Path, *, zones=None, fields=None, times=None, options=None):
        yield fc.RawSample(id="toy", kind="point_cloud", zone="main", points=np.zeros((2, 3)))


def test_registry_runtime_plugin_and_entry_points(tmp_path: Path) -> None:
    f = tmp_path / "a.toy"
    f.write_text("x")
    registry.register(_ToyConnector())
    try:
        assert registry.is_connector(registry.get("toy"))
        assert fc.detect(f)[0].connector_id == "toy"
        assert next(iter(fc.read(f))).n_points == 2
        row = next(i for i in fc.list_connectors() if i.id == "toy")
        assert row.status == "available" and row.source == "runtime"
    finally:
        registry.unregister("toy")
    with pytest.raises(fc.UnsupportedSourceError, match="known ids"):
        registry.get("toy")
    with pytest.raises(fc.UnsupportedSourceError, match="no connector recognises"):
        registry.resolve(f)


def test_every_installed_connector_loads_and_is_labelled() -> None:
    rows = {i.id: i for i in fc.list_connectors()}
    broken = {k: v.note for k, v in rows.items() if v.status == "broken"}
    assert not broken, broken
    assert {"vtk", "openfoam", "netcdf"} <= set(rows)
    for row in rows.values():
        assert row.maturity in fc.MATURITIES
        assert row.license, f"{row.id} has no licence string"
        assert row.plugin, f"{row.id} does not come from a plugin"
        assert row.source == f"plugin:{row.plugin}"


def test_readme_support_matrix_is_generated_from_the_registry() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    start = readme.index("<!-- matrix:start -->") + len("<!-- matrix:start -->")
    table = readme[start : readme.index("<!-- matrix:end -->")].strip()
    # An installation may contain only the open plugins. Every installed row must be
    # documented, even when the monorepo's matrix also describes deferred plugins.
    for row in registry.support_matrix_markdown().splitlines()[2:]:
        assert row in table, f"README support matrix is missing {row}"


def test_cli_list_detect_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {"vtk", "openfoam", "netcdf"} <= {r["id"] for r in rows}
    (tmp_path / "x.unknown").write_text("?")
    assert main(["detect", str(tmp_path / "x.unknown"), "--json"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["error"] == "ConnectorError" and "hint" in err
    assert main(["describe", str(tmp_path / "missing.vtu")]) == 2
    assert "does not exist" in capsys.readouterr().err
    main(["doctor", "--json"])
    assert json.loads(capsys.readouterr().out)["modules"][0]["module"] == "numpy"


def test_field_names_differing_only_in_case_do_not_share_an_array(tmp_path: Path) -> None:
    """macOS and Windows file systems are case-insensitive: ``Extra`` and ``extra`` would be
    one zarr array. The second gets a suffix, the same one on every time step."""
    points = np.zeros((3, 3), np.float32)

    def steps() -> Iterator[fc.RawSample]:
        for i in range(3):
            yield fc.RawSample(
                id="sig",
                kind="point_cloud",
                points=points if i == 0 else None,
                fields={
                    "Extra": fc.as_field(np.full(3, 1.0 + i), "point"),
                    "extra": fc.as_field(np.full(3, 10.0 + i), "point"),
                    "EXTRA": fc.as_field(np.full(3, 100.0 + i), "point"),
                },
                time=float(i),
                time_index=i,
            )

    res = fc.to_frontier_dataset(steps(), tmp_path / "ds", backend="zarr")
    assert sum("only in case" in w for w in res.warnings) == 2
    g = open_store(tmp_path / "ds/samples/sig/points.zarr")
    np.testing.assert_allclose(g["fields/Extra"][:, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(g["fields/extra_2"][:, 0, 0], [10, 11, 12])
    np.testing.assert_allclose(g["fields/EXTRA_3"][:, 0, 0], [100, 101, 102])
