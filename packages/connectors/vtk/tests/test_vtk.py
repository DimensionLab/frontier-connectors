# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from frontier_connector_vtk.connector import VtkConnector

import frontier_connectors as fc
from frontier_connectors.cli import main
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    hex_grid,
    make_ugrid,
    open_store,
    quad_grid,
    tet_grid,
    write_vtk_xml,
)

TIMES = [0.0, 0.25, 0.5, 1.0]


def _write_series(tmp_path: Path) -> Path:
    points, cells = hex_grid(3, 2, 2)
    lines = ['<VTKFile type="Collection" version="0.1"><Collection>']
    for i, t in enumerate(TIMES):
        f = analytic_fields(points, t)
        grid = make_ugrid(
            points, cells, 12, {"p": f["p"], "U": f["U"]}, {"cellp": f["p"][cells].mean(axis=1)}
        )
        write_vtk_xml(grid, tmp_path / f"flow_{i:04d}.vtu")
        lines.append(f'<DataSet timestep="{t}" part="0" file="flow_{i:04d}.vtu"/>')
    lines.append("</Collection></VTKFile>")
    (tmp_path / "flow.pvd").write_text("\n".join(lines))
    (tmp_path / "flow.vtu.series").write_text(
        json.dumps(
            {
                "file-series-version": "1.0",
                "files": [{"name": f"flow_{i:04d}.vtu", "time": t} for i, t in enumerate(TIMES)],
            }
        )
    )
    return tmp_path / "flow.pvd"


def test_transient_pvd_series_and_directory(tmp_path: Path) -> None:
    pvd = _write_series(tmp_path)
    c = VtkConnector()
    rep = check_connector(c, pvd, tmp_path / "out")
    assert rep.description.temporal == "transient" and rep.description.time.values == TIMES
    assert {f.name: f.components for f in rep.description.fields} == {"p": 1, "U": 3, "cellp": 1}
    assert rep.description.zones[0].cell_types == {"hexahedron": 12}
    steps = rep.samples["flow"]
    assert [s.time for s in steps] == TIMES and steps[0].kind == "mesh"
    points, cells = hex_grid(3, 2, 2)
    np.testing.assert_allclose(
        steps[2].fields["U"].data, analytic_fields(points, 0.5)["U"], rtol=1e-6
    )
    assert steps[0].fields["cellp"].location == "cell"
    np.testing.assert_array_equal(steps[0].cells, cells)

    last = list(c.read(tmp_path / "flow.vtu.series", times="last", fields=["p"]))
    assert len(last) == 1 and last[0].time == 1.0 and list(last[0].fields) == ["p"]
    by_dir = list(c.read(tmp_path, times=[0, 3]))
    assert [s.time for s in by_dir] == [0.0, 3.0]  # file numbers stand in for time
    assert fc.detect(tmp_path)[0].connector_id == "vtk"


def test_multiblock_surface_and_volume_zones(tmp_path: Path) -> None:
    from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet, vtkMultiBlockDataSet
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    points, tets = tet_grid(2, 2, 2)
    vol = make_ugrid(points, tets, 10, {"T": points[:, 0]})
    qp, quads = quad_grid(3, 2)
    surf_u = make_ugrid(qp, quads, 9, {"p": qp[:, 0]}, {"wss": np.arange(len(quads), dtype=float)})
    gf = vtkGeometryFilter()
    gf.SetInputData(surf_u)
    gf.Update()
    mb = vtkMultiBlockDataSet()
    mb.SetBlock(0, vol)
    mb.GetMetaData(0).Set(vtkCompositeDataSet.NAME(), "fluid")
    mb.SetBlock(1, gf.GetOutput())
    mb.GetMetaData(1).Set(vtkCompositeDataSet.NAME(), "wall")
    path = write_vtk_xml(mb, tmp_path / "case.vtm")

    rep = check_connector(VtkConnector(), path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    assert zones["fluid"].kind == "mesh" and zones["wall"].kind == "surface"
    wall = rep.samples["case__wall"][0]
    assert wall.faces is not None and wall.faces.shape == (2 * len(quads), 3)
    # per-quad cell data is carried to both triangles of each quad
    np.testing.assert_allclose(wall.fields["wss"].data[:, 0], np.repeat(np.arange(len(quads)), 2))
    only = list(VtkConnector().read(path, zones=["wall"]))
    assert [s.id for s in only] == ["case"]
    with pytest.raises(fc.SelectionError, match="available zones: fluid, wall"):
        list(VtkConnector().read(path, zones=["nope"]))


def test_image_data_becomes_grid_and_legacy_detects(tmp_path: Path) -> None:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkIOLegacy import vtkDataSetWriter

    img = vtkImageData()
    img.SetDimensions(4, 3, 2)
    img.SetSpacing(0.5, 1.0, 2.0)
    img.SetOrigin(1.0, 0.0, 0.0)
    # value = x index + 10*y index + 100*z index, VTK point order is x-fastest
    k, j, i = np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij")
    arr = numpy_to_vtk((i + 10 * j + 100 * k).ravel().astype(np.float64), deep=True)
    arr.SetName("rho")
    img.GetPointData().AddArray(arr)
    path = write_vtk_xml(img, tmp_path / "img.vti")
    rep = check_connector(VtkConnector(), path, tmp_path / "out")
    s = rep.first()
    assert s.kind == "grid" and s.fields["rho"].dims == ["x", "y", "z", "component"]
    assert s.fields["rho"].shape == (4, 3, 2, 1)
    assert float(s.fields["rho"].data[3, 2, 1, 0]) == 123.0
    np.testing.assert_allclose(s.coords["x"], [1.0, 1.5, 2.0, 2.5])

    # cell data only -> lossless grid of cell centres; point + cell data -> averaged to nodes
    carr = numpy_to_vtk(np.arange(3 * 2 * 1, dtype=np.float64), deep=True)
    carr.SetName("T")
    img.GetCellData().AddArray(carr)
    mixed = next(iter(VtkConnector().read(write_vtk_xml(img, tmp_path / "mixed.vti"))))
    assert mixed.fields["T"].shape == (4, 3, 2, 1)
    assert float(mixed.fields["T"].data[0, 0, 0, 0]) == 0.0  # corner node: its one cell
    assert float(mixed.fields["T"].data[1, 0, 0, 0]) == 0.5  # mean of cells 0 and 1
    assert mixed.metadata["cell_fields_averaged_to_nodes"] == ["T"]
    img.GetPointData().RemoveArray("rho")
    rep2 = check_connector(VtkConnector(), write_vtk_xml(img, tmp_path / "fv.vti"), tmp_path / "o2")
    fv = rep2.first()
    assert fv.fields["T"].shape == (3, 2, 1, 1) and fv.metadata["grid_location"] == "cell_centers"
    np.testing.assert_allclose(fv.coords["x"], [1.25, 1.75, 2.25])
    assert float(fv.fields["T"].data[2, 1, 0, 0]) == 5.0

    w = vtkDataSetWriter()
    w.SetFileName(str(tmp_path / "legacy.vtk"))
    w.SetInputData(img)
    w.Write()
    det = VtkConnector().detect(tmp_path / "legacy.vtk")
    assert det is not None and det.confidence > 0.9
    assert next(iter(VtkConnector().read(tmp_path / "legacy.vtk"))).kind == "grid"


def test_cli_describe_and_convert(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pvd = _write_series(tmp_path)
    assert main(["describe", str(pvd), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["temporal"] == "transient" and d["time"]["n_steps"] == 4
    out = tmp_path / "ds"
    rc = main(["convert", str(pvd), "--out", str(out), "--times", "1:", "--fields", "U", "--json"])
    assert rc == 0
    res = json.loads(capsys.readouterr().out)
    assert res["n_steps"] == {"flow": 3} and res["backends"] == {"flow": "zarr-stream"}
    assert main(["convert", str(pvd), "--out", str(out)]) == 2  # exists
    assert "--overwrite" in capsys.readouterr().err
    assert main(["describe", str(pvd)]) == 0
    assert "transient (4 steps" in capsys.readouterr().out


def test_convert_rescales_lengths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    points, cells = hex_grid(2, 1, 1, size=(2000.0, 1000.0, 1000.0))  # millimetres
    path = write_vtk_xml(make_ugrid(points, cells, 12, {"p": points[:, 0]}), tmp_path / "mm.vtu")
    out = tmp_path / "ds"
    assert main(["convert", str(path), "--out", str(out), "--length-unit", "m"]) == 2
    assert "--source-length" in capsys.readouterr().err
    args = ["convert", str(path), "--out", str(out), "--overwrite"]
    assert main([*args, "--length-unit", "m", "--source-length", "mm"]) == 0
    g = open_store(out / "samples/mm/mesh.zarr")
    assert float(g["points"][...].max()) == pytest.approx(2.0)
    assert float(g["fields/p"][...].max()) == pytest.approx(2000.0)  # fields are not rescaled
    meta = json.loads((out / "samples/mm/meta.json").read_text())
    assert meta["extra"]["units"] == {"length": "m", "length_converted_from": "mm"}


@pytest.mark.network
def test_public_vtk_samples(tmp_path: Path) -> None:
    """Real files from the VTK testing data mirrored in pyvista/vtk-data: a ParaView-era XML
    PolyData surface, a legacy shell mesh with point arrays and a legacy hexahedral volume."""
    from frontier_connectors.testing import fetch_sample

    base = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data/"
    cases = [("cow.vtp", "surface"), ("blow.vtk", "surface"), ("hexa.vtk", "mesh")]
    for name, kind in cases:
        path = fetch_sample(base + name)
        rep = check_connector(VtkConnector(), path, tmp_path / name)
        first = rep.first()
        assert first.kind == kind and first.n_points > 500 and first.n_cells > 400
        if name == "blow.vtk":  # blow-moulding shell with ten displacement/thickness states
            assert first.fields["displacement9"].shape == (first.n_points, 3)
            assert first.fields["thickness0"].shape == (first.n_points, 1)
        if name == "hexa.vtk":
            assert first.cell_types is not None and set(first.cell_types.tolist()) == {12}
