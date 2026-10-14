# SPDX-License-Identifier: Apache-2.0
"""Exodus II, XDMF, Gmsh, MED, SU2 and the generic meshio connector.

Fixtures are written by an independent implementation wherever one exists (VTK's Exodus writer,
meshio's writers, plain h5py / text following the format specifications) and the values read
back are compared with the analytic fields they were generated from, or with a second reader
(scipy's NetCDF reader, meshio, VTK's Xdmf reader).
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Any

import h5py
import meshio
import numpy as np
import pytest
from frontier_connector_meshio.connector import (
    ExodusConnector,
    GmshConnector,
    MedConnector,
    MeshioConnector,
    Su2Connector,
    XdmfConnector,
    parse_su2,
    scan_gmsh_views,
    scan_xdmf,
)

import frontier_connectors as fc
from frontier_connectors.model import RawSample
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    fetch_sample,
    hex_grid,
    make_ugrid,
    quad_grid,
    tet_grid,
    tri_grid,
)

TIMES = [0.0, 0.5, 1.5]
PYVISTA = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data/"
MESHIO = "https://media.githubusercontent.com/media/nschloe/meshio/main/tests/meshes/"
VTK_DATA = "https://www.vtk.org/files/ExternalData/SHA512/"
#: VTK testing data is content-addressed: SHA-512 of Testing/Data/<name> in the VTK repository
VTK_FLOW1D = (
    "52e8b2212d9eb7ce49c2f44b10cf95024edc49bc67ced47057f5ad765222efac"
    "1bc35ee58b3dffc69beb979f91064adddd9ba51de274880f001c29bdca2bcfd3"
)
VTK_CORECT = (
    "b6ce25bd083b34ef694af1c7a47b4d521d87650767ca249e3a51347ec720e75d"
    "225e0069c511e94e782c800e2d8e55f563e3834ba6259c01ea53de766b5b0c5b"
)
VTK_SPATIAL_TIME = (
    "4d47dcd2eeee9616a0aad68ce5910382661b2c5cb751453e4a15d7632e5f81cd"
    "04cf705d1567d6a7029730505f9d41b7fa6dbe3638e2b373a45d25efa4aa77c8"
)


def signed_volumes(points: np.ndarray, cells: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Corner-0 Jacobian of linear 3-D cells: positive when the cell follows VTK's ordering."""
    p = np.asarray(points, dtype=np.float64)
    out = np.ones(len(cells))
    for vtk_type, (a, b, c) in {10: (1, 2, 3), 12: (1, 3, 4), 13: (2, 1, 3), 14: (1, 3, 4)}.items():
        rows = np.nonzero(types == vtk_type)[0]
        if len(rows):
            conn = cells[rows]
            o = p[conn[:, 0]]
            out[rows] = np.einsum(
                "ij,ij->i", np.cross(p[conn[:, a]] - o, p[conn[:, b]] - o), p[conn[:, c]] - o
            )
    return out


def pts(s: RawSample) -> np.ndarray:
    assert s.points is not None
    return s.points


def cells(s: RawSample) -> np.ndarray:
    assert s.cells is not None
    return s.cells


def faces(s: RawSample) -> np.ndarray:
    assert s.faces is not None
    return s.faces


def volumes(s: RawSample) -> np.ndarray:
    assert s.cell_types is not None
    return signed_volumes(pts(s), cells(s), s.cell_types)


def same_mesh(s: RawSample, points: np.ndarray, conn: np.ndarray) -> bool:
    return same_cells(pts(s), cells(s), points, conn)


def same_cells(pa: np.ndarray, ca: np.ndarray, pb: np.ndarray, cb: np.ndarray) -> bool:
    """Two meshes hold the same cells, whatever the point and cell numbering."""

    def key(points: np.ndarray, cells: np.ndarray) -> list[tuple[float, ...]]:
        corners = np.round(np.asarray(points, dtype=np.float64)[cells], 5)
        order = np.lexsort(corners.transpose(2, 0, 1)[::-1], axis=-1)
        rows = np.take_along_axis(corners, order[:, :, None], axis=1).reshape(len(cells), -1)
        return sorted(map(tuple, rows.tolist()))

    return key(pa, ca) == key(pb, cb)


def tets_with_wall() -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Unit-box tet mesh split into two regions (z < 0.5 / z > 0.5) plus the boundary
    triangles on x = 0. Returns ``(points, tets sorted by region, n_lower, triangles)``."""
    points, tets = tet_grid(2, 2, 2)
    lower = points[tets].mean(axis=1)[:, 2] < 0.5
    tets = tets[np.argsort(~lower, kind="stable")]
    tris = [t[f] for t in tets for f in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])]
    tris = np.array([t for t in tris if np.all(points[t, 0] == 0.0)])
    return points, tets, int(lower.sum()), tris


# -- Exodus II ----------------------------------------------------------------------------------


def write_exodus(path: Path, *, planar: bool = False) -> Path:
    """Transient two-block Exodus file written by VTK's vtkExodusIIWriter from a time-aware
    source (hexahedra named ``fluid`` and tetrahedra named ``solid``; or one quad block)."""
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
    from vtkmodules.vtkCommonDataModel import (
        vtkCompositeDataSet,
        vtkDataObject,
        vtkMultiBlockDataSet,
    )
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as SDDP  # noqa: N817
    from vtkmodules.vtkIOExodus import vtkExodusIIWriter

    class Source(VTKPythonAlgorithmBase):
        def __init__(self) -> None:
            super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet")

        def RequestInformation(self, request: Any, inInfo: Any, outInfo: Any) -> Any:  # noqa: N802, N803
            info = outInfo.GetInformationObject(0)
            info.Set(SDDP.TIME_STEPS(), TIMES, len(TIMES))
            info.Set(SDDP.TIME_RANGE(), [TIMES[0], TIMES[-1]], 2)
            return 1

        def RequestData(self, request: Any, inInfo: Any, outInfo: Any) -> Any:  # noqa: N802, N803
            info = outInfo.GetInformationObject(0)
            t = info.Get(SDDP.UPDATE_TIME_STEP()) if info.Has(SDDP.UPDATE_TIME_STEP()) else 0.0
            out = vtkMultiBlockDataSet.GetData(outInfo, 0)
            blocks = []
            if planar:
                qp, quads = quad_grid(4, 3)
                blocks.append(("plate", qp, quads, 9))
            else:
                hp, hexes = hex_grid(3, 2, 2)
                tp, tets = tet_grid(2, 2, 2)
                blocks += [("fluid", hp, hexes, 12), ("solid", tp + [2.0, 0.0, 0.0], tets, 10)]
            for i, (name, pts, cells, vtk_type) in enumerate(blocks):
                f = analytic_fields(pts, t)
                grid = make_ugrid(
                    pts,
                    cells,
                    vtk_type,
                    {"p": f["p"], "U": f["U"]},
                    {"cellp": f["p"][cells].mean(axis=1)},
                )
                out.SetBlock(i, grid)
                out.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), name)
            out.GetInformation().Set(vtkDataObject.DATA_TIME_STEP(), t)
            return 1

    source = Source()  # keep a reference: the pipeline does not own the Python object
    writer = vtkExodusIIWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(source.GetOutputPort())
    writer.WriteAllTimeStepsOn()
    assert writer.Write() == 1
    return path


def test_exodus_transient_blocks_against_netcdf(tmp_path: Path) -> None:
    from scipy.io import netcdf_file

    path = write_exodus(tmp_path / "two.e")
    c = ExodusConnector()
    rep = check_connector(c, path, tmp_path / "out")
    desc = rep.description
    assert desc.temporal == "transient" and desc.time.values == TIMES
    zones = {z.name: z for z in desc.zones}
    assert zones["fluid"].cell_types == {"hexahedron": 12} and zones["fluid"].default
    assert zones["solid"].cell_types == {"tetra": 48} and zones["solid"].kind == "mesh"
    assert {(f.name, f.location, f.components) for f in desc.fields} == {
        ("p", "point", 1),
        ("U", "point", 3),
        ("cellp", "cell", 1),
    }
    fluid, solid = rep.samples["two__fluid"], rep.samples["two__solid"]
    assert [s.time for s in fluid] == TIMES and [s.time_index for s in solid] == [0, 1, 2]
    assert fluid[0].points is not None and fluid[1].points is None  # geometry once
    # the reader numbers the points of each block in connectivity order: compare as sets
    hp, hexes = hex_grid(3, 2, 2)
    tp, tets = tet_grid(2, 2, 2)
    assert same_mesh(fluid[0], hp, hexes)
    assert same_mesh(solid[0], tp + [2.0, 0.0, 0.0], tets)
    for s, t in zip(fluid, TIMES, strict=True):
        f = analytic_fields(fluid[0].points, t)
        np.testing.assert_allclose(s.fields["U"].data, f["U"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s.fields["p"].data[:, 0], f["p"], rtol=1e-5, atol=1e-6)
    expected = analytic_fields(pts(solid[0]), 0.5)["p"]
    np.testing.assert_allclose(solid[1].fields["p"].data[:, 0], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        solid[1].fields["cellp"].data[:, 0],
        expected[solid[0].cells].mean(axis=1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert (volumes(solid[0]) > 0).all()

    # independent reader: the raw NetCDF variables (classic format, scipy)
    nc = netcdf_file(str(path), "r", mmap=False)
    np.testing.assert_allclose(nc.variables["time_whole"][:], TIMES)
    names = [b"".join(r).split(b"\x00")[0].decode() for r in nc.variables["name_nod_var"][:]]
    assert names == ["p", "UX", "UY", "UZ"]
    raw_p = np.array(nc.variables["vals_nod_var1"][2])  # all nodes of the file at t = 1.5
    raw_x = np.array(nc.variables["coordx"][:])
    nc.close()
    step = np.concatenate([fluid[2].fields["p"].data[:, 0], solid[2].fields["p"].data[:, 0]])
    xs = np.concatenate([pts(fluid[0])[:, 0], pts(solid[0])[:, 0]])
    np.testing.assert_allclose(np.sort(step), np.sort(raw_p), rtol=1e-6)
    np.testing.assert_allclose(np.sort(xs), np.sort(raw_x), atol=1e-6)

    last = list(c.read(path, zones=["solid"], fields=["U"], times="last"))
    assert [(s.id, s.time, list(s.fields)) for s in last] == [("two", 1.5, ["U"])]
    with pytest.raises(fc.SelectionError, match="available zones: fluid, solid"):
        list(c.read(path, zones=["nope"]))


def test_exodus_detection_and_planar_rule(tmp_path: Path) -> None:
    c = ExodusConnector()
    path = write_exodus(tmp_path / "plate.exo", planar=True)
    # restart / parallel suffixes and foreign extensions are recognised by content
    for name, conf in (("plate.e-s0002", 0.97), ("plate.e.4.0", 0.97), ("plate.nc", 0.9)):
        det = c.detect(Path(shutil.copy(path, tmp_path / name)))
        assert det is not None and det.confidence == conf, name
    (tmp_path / "cow.g").write_text("1 2 3 4\n0.0 1.0 2.0\n")  # a BYU file, not NetCDF
    assert c.detect(tmp_path / "cow.g") is None and c.detect(tmp_path) is None
    with pytest.raises(fc.UnsupportedSourceError, match="not an Exodus II"):
        c.describe(tmp_path / "cow.g")

    rep = check_connector(c, path, tmp_path / "out")
    s = rep.first()
    qp, quads = quad_grid(4, 3)
    # a planar mesh is a 2-D domain by default ...
    assert s.kind == "mesh" and pts(s).shape == (len(qp), 2)
    assert rep.description.zones[0].kind == "mesh" and rep.description.zones[0].dim == 2
    assert same_mesh(s, qp[:, :2], quads)
    # ... and a triangulated surface on request
    surf = next(iter(c.read(path, options={"planar_as": "surface"})))
    assert surf.kind == "surface" and faces(surf).shape == (2 * len(quads), 3)
    assert pts(surf).shape[1] == 3
    np.testing.assert_allclose(
        surf.fields["cellp"].data[:, 0], np.repeat(s.fields["cellp"].data[:, 0], 2)
    )
    with pytest.raises(fc.ConnectorError, match="planar_as"):
        list(c.read(path, options={"planar_as": "volume"}))

    truncated = tmp_path / "cut.e"
    truncated.write_bytes(path.read_bytes()[:600])
    with pytest.raises(fc.ConnectorError):
        list(c.read(truncated))


# -- XDMF ---------------------------------------------------------------------------------------


def test_xdmf_meshio_time_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # meshio writes the .h5 into the working directory
    points, tris = tri_grid(4, 3)
    path = tmp_path / "series.xdmf"
    with meshio.xdmf.TimeSeriesWriter(str(path)) as w:
        w.write_points_cells(points[:, :2], [("triangle", tris)])
        for t in TIMES:
            f = analytic_fields(points, t)
            w.write_data(
                t,
                point_data={"p": f["p"], "U": f["U"][:, :2]},
                cell_data={"cellp": [f["p"][tris].mean(axis=1)]},
            )
    scan = scan_xdmf(path)
    assert scan.meshio_series and scan.times == TIMES and scan.topologies == ["Triangle"]
    c = XdmfConnector()
    rep = check_connector(c, path, tmp_path / "out")
    assert rep.description.time.values == TIMES and rep.description.metadata["engine"] == "meshio"
    steps = rep.samples["series"]
    assert [s.time for s in steps] == TIMES and steps[0].kind == "mesh"
    assert pts(steps[0]).shape == (len(points), 2) and steps[1].points is None
    np.testing.assert_array_equal(steps[0].cells, tris)
    f = analytic_fields(points, 1.5)
    np.testing.assert_allclose(steps[2].fields["U"].data, f["U"][:, :2], rtol=1e-6)
    np.testing.assert_allclose(
        steps[2].fields["cellp"].data[:, 0], f["p"][tris].mean(axis=1), rtol=1e-6
    )
    one = list(c.read(path, times=[0.5], fields=["p"]))
    assert [(s.time, s.time_index, list(s.fields)) for s in one] == [(0.5, 1, ["p"])]


def write_dolfinx_style(path: Path, *, quadratic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """XDMF3 temporal collection as dolfinx writes it: every step repeats Topology/Geometry and
    points at HDF5 heavy data."""
    points, tets = tet_grid(2, 2, 2)
    h5 = path.with_suffix(".h5")
    grids = []
    with h5py.File(h5, "w") as handle:
        f: Any = handle
        f["Mesh/mesh/topology"] = tets.astype(np.int64)
        f["Mesh/mesh/geometry"] = points
        for i, t in enumerate(TIMES):
            fields = analytic_fields(points, t)
            f[f"Function/p/{i}"] = fields["p"][:, None]
            f[f"Function/U/{i}"] = fields["U"]
            f[f"Function/cellp/{i}"] = fields["p"][tets].mean(axis=1)[:, None]
            grids.append(
                f'<Grid Name="mesh" GridType="Uniform">'
                f'<Topology TopologyType="Tetrahedron" NumberOfElements="{len(tets)}" '
                f'NodesPerElement="4"><DataItem Dimensions="{len(tets)} 4" NumberType="Int" '
                f'Precision="8" Format="HDF">{h5.name}:/Mesh/mesh/topology</DataItem></Topology>'
                f'<Geometry GeometryType="XYZ"><DataItem Dimensions="{len(points)} 3" '
                f'Precision="8" Format="HDF">{h5.name}:/Mesh/mesh/geometry</DataItem></Geometry>'
                f'<Time Value="{t}" />'
                f'<Attribute Name="p" AttributeType="Scalar" Center="Node"><DataItem '
                f'Dimensions="{len(points)} 1" Precision="8" Format="HDF">{h5.name}:/Function/p/{i}'
                f"</DataItem></Attribute>"
                f'<Attribute Name="U" AttributeType="Vector" Center="Node"><DataItem '
                f'Dimensions="{len(points)} 3" Precision="8" Format="HDF">{h5.name}:/Function/U/{i}'
                f"</DataItem></Attribute>"
                f'<Attribute Name="cellp" AttributeType="Scalar" Center="Cell"><DataItem '
                f'Dimensions="{len(tets)} 1" Precision="8" Format="HDF">'
                f"{h5.name}:/Function/cellp/{i}</DataItem></Attribute></Grid>"
            )
    path.write_text(
        '<?xml version="1.0"?>\n<Xdmf Version="3.0"><Domain>'
        '<Grid Name="p" GridType="Collection" CollectionType="Temporal">'
        + "".join(grids)
        + "</Grid></Domain></Xdmf>\n"
    )
    return points, tets


def test_xdmf_temporal_collection_through_vtk(tmp_path: Path) -> None:
    path = tmp_path / "flow.xdmf"
    points, tets = write_dolfinx_style(path)
    c = XdmfConnector()
    rep = check_connector(c, path, tmp_path / "out")
    desc = rep.description
    assert desc.metadata["engine"] == "vtk" and desc.time.values == TIMES
    assert desc.zones[0].cell_types == {"tetra": len(tets)}
    steps = rep.samples["flow"]
    assert [s.time for s in steps] == TIMES and steps[1].points is None
    np.testing.assert_allclose(pts(steps[0]), points, atol=1e-7)
    assert sorted(map(tuple, np.sort(steps[0].cells, axis=1))) == sorted(
        map(tuple, np.sort(tets, axis=1))
    )
    for s, t in zip(steps, TIMES, strict=True):
        f = analytic_fields(points, t)
        np.testing.assert_allclose(s.fields["U"].data, f["U"], rtol=1e-6)
        np.testing.assert_allclose(
            s.fields["cellp"].data[:, 0], f["p"][tets].mean(axis=1), rtol=1e-6
        )
    # the heavy data is required: without it the error names the file's constructs
    broken = tmp_path / "alone" / "flow.xdmf"
    broken.parent.mkdir()
    shutil.copy(path, broken)
    with pytest.raises(fc.ConnectorError, match=r"heavy-data file\(s\) not found: flow.h5"):
        list(c.read(broken))
    # VTK's reader crashes on what it cannot parse: the child-process probe turns that
    # into an error that names the problem
    weird = tmp_path / "weird.xdmf"
    weird.write_text(path.read_text().replace("Tetrahedron", "Hypercube"))
    with pytest.raises(fc.ConnectorError, match="Bad Topology Type : Hypercube"):
        c.describe(weird)


def test_xdmf_structured_grid_and_reader_cross_check(tmp_path: Path) -> None:
    # 3DCoRectMesh (XDMF lists dimensions and origin as Z Y X) -> grid
    nz, ny, nx = 2, 3, 4
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    values = (i + 10 * j + 100 * k).ravel()
    grid = tmp_path / "grid.xmf"
    grid.write_text(
        '<Xdmf Version="2.1"><Domain><Grid Name="box" GridType="Uniform">'
        f'<Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>'
        '<Geometry Type="ORIGIN_DXDYDZ"><DataItem Format="XML" Dimensions="3">0.0 1.0 2.0'
        '</DataItem><DataItem Format="XML" Dimensions="3">3.0 0.5 0.25</DataItem></Geometry>'
        f'<Attribute Name="rho" Center="Node"><DataItem Format="XML" Dimensions="{nz} {ny} {nx}">'
        + " ".join(str(v) for v in values)
        + '</DataItem></Attribute><Attribute Name="cell_i" Center="Cell"><DataItem Format="XML" '
        f'Dimensions="{nz - 1} {ny - 1} {nx - 1}">'
        + " ".join(str(v) for v in np.tile(np.arange(nx - 1), (nz - 1) * (ny - 1)))
        + "</DataItem></Attribute></Grid></Domain></Xdmf>"
    )
    c = XdmfConnector()
    rep = check_connector(c, grid, tmp_path / "out")
    s = rep.first()
    assert s.kind == "grid" and s.fields["rho"].shape == (nx, ny, nz, 1)
    assert float(s.fields["rho"].data[3, 2, 1, 0]) == 123.0
    np.testing.assert_allclose(s.coords["x"], 2.0 + 0.25 * np.arange(nx))
    np.testing.assert_allclose(s.coords["z"], 0.0 + 3.0 * np.arange(nz))
    # the dataset format has no cell-centred grid arrays: every node takes the mean of the
    # cells around it (cell value = x index, so nodes get 0, 0.5, 1.5, 2 along x)
    assert s.fields["cell_i"].dims == ["x", "y", "z", "component"]
    np.testing.assert_allclose(s.fields["cell_i"].data[:, 1, 0, 0], [0.0, 0.5, 1.5, 2.0])
    assert s.metadata["cell_fields_averaged_to_nodes"] == ["cell_i"]

    # one unstructured grid written by meshio, read by both engines (meshio and VTK's Xdmf2)
    points, tets, _, tris = tets_with_wall()
    f = analytic_fields(points)
    mesh = meshio.Mesh(points, [("tetra", tets)], point_data={"p": f["p"], "U": f["U"]})
    single = tmp_path / "single.xdmf"
    meshio.write(single, mesh, data_format="HDF")
    by_meshio = check_connector(c, single, tmp_path / "o1").first()
    by_vtk = next(iter(c.read(single, options={"engine": "vtk"})))
    assert c.describe(single).metadata["engine"] == "meshio"
    np.testing.assert_allclose(pts(by_meshio), pts(by_vtk), atol=1e-7)
    np.testing.assert_array_equal(by_meshio.cells, by_vtk.cells)
    np.testing.assert_allclose(by_meshio.fields["U"].data, by_vtk.fields["U"].data, rtol=1e-6)
    np.testing.assert_allclose(by_meshio.fields["p"].data[:, 0], f["p"], rtol=1e-6)

    bad = tmp_path / "bad.xdmf"
    bad.write_text('<Xdmf Version="3.0"><Domain><Grid>')
    assert c.detect(bad) is not None
    with pytest.raises(fc.ConnectorError, match="not valid XDMF XML"):
        c.describe(bad)
    assert c.detect(tmp_path / "missing.xdmf") is None


# -- Gmsh ---------------------------------------------------------------------------------------


def gmsh_mesh() -> tuple[meshio.Mesh, np.ndarray, np.ndarray, int, np.ndarray]:
    points, tets, n1, tris = tets_with_wall()
    mesh = meshio.Mesh(
        points,
        [("tetra", tets[:n1]), ("tetra", tets[n1:]), ("triangle", tris)],
        cell_data={
            "gmsh:physical": [np.full(n1, 1), np.full(len(tets) - n1, 2), np.full(len(tris), 3)],
            "gmsh:geometrical": [np.full(n1, 1), np.full(len(tets) - n1, 2), np.full(len(tris), 7)],
        },
        field_data={"fluid": np.array([1, 3]), "solid": np.array([2, 3]), "wall": np.array([3, 2])},
    )
    return mesh, points, tets, n1, tris


def write_msh41(
    path: Path, binary: bool
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """MSH 4.1 written per the Gmsh reference manual: two volumes and one surface, each in a
    physical group, with sparse node tags (tag = 10 + 3 * index)."""
    _, points, tets, n1, tris = gmsh_mesh()
    tags = 10 + 3 * np.arange(len(points))
    # (dim, entity tag, physical tag, gmsh element type, connectivity)
    blocks = [(3, 1, 1, 4, tets[:n1]), (3, 2, 2, 4, tets[n1:]), (2, 7, 3, 2, tris)]
    box = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    out = bytearray()

    def text(s: str) -> None:
        out.extend(s.encode())

    def numbers(fmt: str, *values: Any) -> None:
        if binary:
            out.extend(struct.pack("<" + fmt, *values))
        else:
            text(" ".join(str(v) for v in values) + "\n")

    text(f"$MeshFormat\n4.1 {int(binary)} 8\n")
    if binary:
        out.extend(struct.pack("<i", 1))
        text("\n")
    text('$EndMeshFormat\n$PhysicalNames\n3\n3 1 "fluid"\n3 2 "solid"\n2 3 "wall"\n')
    text("$EndPhysicalNames\n$Entities\n")
    numbers("4Q", 0, 0, 1, 2)
    numbers("i6dQiQ", 7, *box, 1, 3, 0)
    for entity, physical in ((1, 1), (2, 2)):
        numbers("i6dQiQ", entity, *box, 1, physical, 0)
    if binary:
        text("\n")
    text("$EndEntities\n$Nodes\n")
    numbers("4Q", 1, len(points), int(tags.min()), int(tags.max()))
    numbers("3iQ", 3, 1, 0, len(points))
    if binary:
        out.extend(tags.astype("<u8").tobytes() + points.astype("<f8").tobytes() + b"\n")
    else:
        text("\n".join(str(t) for t in tags) + "\n")
        text("\n".join(" ".join(repr(float(v)) for v in p) for p in points) + "\n")
    text("$EndNodes\n$Elements\n")
    numbers("4Q", len(blocks), len(tets) + len(tris), 1, len(tets) + len(tris))
    element = 1
    for dim, entity, _, etype, conn in blocks:
        numbers("3iQ", dim, entity, etype, len(conn))
        rows = np.column_stack([np.arange(element, element + len(conn)), tags[conn]])
        element += len(conn)
        if binary:
            out.extend(rows.astype("<u8").tobytes())
        else:
            text("\n".join(" ".join(str(v) for v in r) for r in rows) + "\n")
    if binary:
        text("\n")
    text("$EndElements\n")
    path.write_bytes(bytes(out))
    return points, tets, n1, tris, tags


def gmsh_view(
    kind: str, name: str, time: float, step: int, tags: np.ndarray, values: np.ndarray, binary: bool
) -> bytes:
    values = values.reshape(len(tags), -1)
    head = f'${kind}\n1\n"{name}"\n1\n{time}\n3\n{step}\n{values.shape[1]}\n{len(tags)}\n'
    if binary:
        rec = np.zeros(len(tags), dtype=[("t", "<i4"), ("v", "<f8", (values.shape[1],))])
        rec["t"], rec["v"] = tags, values
        body = rec.tobytes() + b"\n"
    else:
        body = "".join(
            f"{t} " + " ".join(repr(float(v)) for v in row) + "\n"
            for t, row in zip(tags, values, strict=True)
        ).encode()
    return head.encode() + body + f"$End{kind}\n".encode()


@pytest.mark.parametrize(
    ("version", "binary"), [("2.2", False), ("2.2", True), ("4.1", False), ("4.1", True)]
)
def test_gmsh_groups_and_transient_views(tmp_path: Path, version: str, binary: bool) -> None:
    path = tmp_path / "case.msh"
    if version == "2.2":
        mesh, points, tets, n1, tris = gmsh_mesh()
        meshio.write(path, mesh, file_format="gmsh22", binary=binary)
        tags = 1 + np.arange(len(points))
    else:
        points, tets, n1, tris, tags = write_msh41(path, binary)
    n_cells = len(tets) + len(tris)
    cell_id = np.arange(1, n_cells + 1)
    with open(path, "ab") as f:
        for step, t in enumerate(TIMES):
            fields = analytic_fields(points, t)
            # views are keyed by node tag: write them in reverse order to prove the mapping
            f.write(gmsh_view("NodeData", "T", t, step, tags[::-1], fields["p"][::-1], binary))
            f.write(gmsh_view("NodeData", "U", t, step, tags, fields["U"], binary))
            f.write(gmsh_view("ElementData", "q", t, step, cell_id, cell_id * (1.0 + t), binary))
        f.write(gmsh_view("NodeData", "wall_distance", 0.0, 0, tags, points[:, 0], binary))
    assert len(scan_gmsh_views(path)) == 10

    c = GmshConnector()
    rep = check_connector(c, path, tmp_path / "out")
    desc = rep.description
    assert desc.format == f"gmsh-msh{version}" and desc.time.values == TIMES
    zones = {z.name: z for z in desc.zones}
    assert list(zones) == ["fluid", "solid", "wall"]
    assert zones["fluid"].default and zones["solid"].cell_types == {"tetra": len(tets) - n1}
    assert zones["wall"].kind == "surface" and not zones["wall"].default
    assert zones["wall"].n_cells == len(tris)
    assert {f.name for f in desc.fields} == {"T", "U", "q", "wall_distance"}

    fluid = rep.samples["case__fluid"]
    assert [s.time for s in fluid] == TIMES and fluid[1].points is None
    assert set(fluid[0].static) >= {"wall_distance", "gmsh_geometrical"}
    xyz = pts(fluid[0])
    np.testing.assert_allclose(fluid[0].static["wall_distance"].data[:, 0], xyz[:, 0])
    assert (volumes(fluid[0]) > 0).all()
    assert "reoriented_cells" not in fluid[0].metadata
    for s, t in zip(fluid, TIMES, strict=True):
        f = analytic_fields(xyz, t)
        np.testing.assert_allclose(s.fields["T"].data[:, 0], f["p"], rtol=1e-6)
        np.testing.assert_allclose(s.fields["U"].data, f["U"], rtol=1e-6)
        np.testing.assert_allclose(s.fields["q"].data[:, 0], np.arange(1, n1 + 1) * (1.0 + t))
    solid = rep.samples["case__solid"]
    np.testing.assert_allclose(
        solid[2].fields["q"].data[:, 0], np.arange(n1 + 1, len(tets) + 1) * 2.5
    )

    wall = list(c.read(path, zones=["wall"], times="last", fields=["T", "q"]))
    assert len(wall) == 1 and wall[0].kind == "surface" and wall[0].time == 1.5
    assert faces(wall[0]).shape == (len(tris), 3) and (pts(wall[0])[:, 0] == 0).all()
    np.testing.assert_allclose(
        wall[0].fields["q"].data[:, 0], np.arange(len(tets) + 1, n_cells + 1) * 2.5
    )
    np.testing.assert_allclose(
        wall[0].fields["T"].data[:, 0], analytic_fields(pts(wall[0]), 1.5)["p"], rtol=1e-6
    )


def test_gmsh_quadratic_planar_and_malformed(tmp_path: Path) -> None:
    c = GmshConnector()
    # 2-D quadratic triangles: a planar mesh stays a 2-D mesh with VTK's quadratic type
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]], float
    )
    mesh = meshio.Mesh(points, [("triangle6", np.array([[0, 1, 2, 3, 4, 5]]))])
    path = tmp_path / "tri6.msh"
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    s = check_connector(c, path, tmp_path / "out").first()
    assert s.kind == "mesh" and pts(s).shape == (6, 2) and np.asarray(s.cell_types).tolist() == [22]
    surf = next(iter(c.read(path, options={"planar_as": "surface"})))
    assert surf.kind == "surface" and faces(surf).tolist() == [[0, 1, 2]]  # corner nodes only

    fluent = tmp_path / "fluent.msh"
    fluent.write_text('(0 "Fluent mesh")\n(2 3)\n')
    assert c.detect(fluent) is None
    with pytest.raises(fc.UnsupportedSourceError, match="fluent-legacy"):
        c.describe(fluent)
    cut = tmp_path / "cut.msh"
    cut.write_bytes(path.read_bytes()[:120])
    assert c.detect(cut) is not None
    with pytest.raises(fc.ConnectorError, match="meshio could not read"):
        list(c.read(cut))
    view = tmp_path / "view.msh"
    view.write_bytes(
        path.read_bytes() + b'$NodeData\n1\n"T"\n1\n0.0\n3\n0\n1\n6\n1 1.0\n2 oops\n$EndNodeData\n'
    )
    with pytest.raises(fc.ConnectorError, match="non-numeric|numbers, expected"):
        list(c.read(view))


# -- MED ----------------------------------------------------------------------------------------

#: VTK node order -> MED node order (MED numbers the base of 3-D cells the other way round)
VTK_TO_MED = {"tetra": [0, 2, 1, 3], "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5]}


def write_med(
    path: Path, *, gauss_field: bool = True
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """MED 3 file: meshio's writer for the mesh, families and one field step; h5py for what
    meshio cannot write (further time steps, component names and units, a Gauss-point field)."""
    points, tets, n1, tris = tets_with_wall()
    f0 = analytic_fields(points, TIMES[0])
    fam = np.where(np.arange(len(tets)) < n1, -1, -2)
    mesh = meshio.Mesh(
        points,
        [("tetra", tets[:, VTK_TO_MED["tetra"]]), ("triangle", tris)],
        point_data={
            "TEMP": f0["p"],
            "resu____DEPL": f0["U"],
            "point_tags": np.where(points[:, 0] == 0.0, 1, 0),
        },
        cell_data={
            "cell_tags": [fam, np.full(len(tris), -3)],
            "POROSITY": [
                np.arange(len(tets), dtype=float),
                np.arange(len(tris), dtype=float) + 500,
            ],
        },
    )
    # family number -> group names, the way meshio's MED writer expects them
    vars(mesh)["cell_tags"] = {-1: ["fluid"], -2: ["solid"], -3: ["wall", "all_faces"]}
    vars(mesh)["point_tags"] = {1: ["inlet_nodes"]}
    meshio.write(path, mesh, file_format="med")
    with h5py.File(path, "a") as h5:
        f: Any = h5
        f["ENS_MAA/mesh"].attrs["DES"] = np.bytes_("CREE PAR CODE_ASTER")
        f["ENS_MAA/mesh"].attrs["UNI"] = np.bytes_("mm".ljust(16) * 3)
        for name, key, ncomp in (("TEMP", "p", 1), ("resu____DEPL", "U", 3)):
            field = f["CHA"][name]
            first = next(iter(field))
            if ncomp == 3:
                field.attrs["NOM"] = np.bytes_("".join(c.ljust(16) for c in ("DX", "DY", "DZ")))
                field.attrs["UNI"] = np.bytes_("mm".ljust(16) * 3)
            else:
                field.attrs["UNI"] = np.bytes_("degC".ljust(16))
            for i, t in enumerate(TIMES[1:], start=2):
                step = f"{i:020d}{i:020d}"
                f.copy(field[first], field, name=step)
                field[step].attrs["NDT"], field[step].attrs["NOR"] = i, i
                field[step].attrs["PDT"] = t
                values = np.asarray(analytic_fields(points, t)[key]).reshape(len(points), -1)
                # MED stores component by component
                field[step]["NOE/MED_NO_PROFILE_INTERNAL/CO"][...] = values.flatten(order="F")
        if not gauss_field:
            return points, tets, n1, tris
        # element field at 4 Gauss points per tetrahedron: value = cell number + 0.1 * gauss
        gauss = np.arange(len(tets))[:, None] + 0.1 * np.arange(4)[None, :]
        elga = f["CHA"].create_group("SIGMA_ELGA")
        elga.attrs["MAI"], elga.attrs["NCO"] = np.bytes_("mesh"), 1
        elga.attrs["NOM"], elga.attrs["UNI"] = (
            np.bytes_("SIXX".ljust(16)),
            np.bytes_("MPa".ljust(16)),
        )
        step = elga.create_group("0000000000000000000100000000000000000001")
        step.attrs["NDT"], step.attrs["NOR"], step.attrs["PDT"] = 1, 1, 0.0
        support = step.create_group("MAI.TE4")
        support.attrs["PFL"] = np.bytes_("MED_NO_PROFILE_INTERNAL")
        prof = support.create_group("MED_NO_PROFILE_INTERNAL")
        prof.attrs["NBR"], prof.attrs["NGA"] = len(tets), 4
        prof.create_dataset("CO", data=gauss.reshape(-1))
    return points, tets, n1, tris


def test_med_groups_fields_and_time_steps(tmp_path: Path) -> None:
    path = tmp_path / "case.med"
    points, tets, n1, tris = write_med(path)
    c = MedConnector()
    rep = check_connector(c, path, tmp_path / "out")
    desc = rep.description
    assert desc.solver == "Code_Aster" and desc.domain == "structural"
    assert desc.time.values == TIMES and desc.units == {"length": "mm"}
    zones = {z.name: z for z in desc.zones}
    assert [n for n, z in zones.items() if z.default] == ["fluid", "solid"]
    assert zones["wall"].kind == "surface" and zones["all_faces"].n_cells == len(tris)
    assert zones["inlet_nodes"].kind == "point_cloud" and zones["inlet_nodes"].n_points == 9
    fields = {f.name: f for f in desc.fields}
    assert fields["resu____DEPL"].component_names == ["DX", "DY", "DZ"]
    assert fields["resu____DEPL"].units == "mm" and fields["TEMP"].units == "degC"
    assert fields["SIGMA_ELGA"].location == "cell" and fields["SIGMA_ELGA"].units == "MPa"
    assert not fields["POROSITY"].time_dependent

    fluid = rep.samples["case__fluid"]
    assert [s.time for s in fluid] == TIMES and fluid[1].points is None
    first = fluid[0]
    # the reader turns MED's node order into VTK's: nothing is left for the generic repair
    assert (volumes(first) > 0).all()
    assert "reoriented_cells" not in first.metadata
    assert same_mesh(first, points, tets[:n1])
    for s, t in zip(fluid, TIMES, strict=True):
        f = analytic_fields(pts(first), t)
        np.testing.assert_allclose(s.fields["TEMP"].data[:, 0], f["p"], rtol=1e-6)
        np.testing.assert_allclose(s.fields["resu____DEPL"].data, f["U"], rtol=1e-6)
    assert set(first.static) == {"POROSITY", "SIGMA_ELGA"} and "TEMP" not in fluid[1].static
    np.testing.assert_allclose(first.static["POROSITY"].data[:, 0], np.arange(n1))
    # Gauss-point values are averaged per cell: mean of n + 0.1 * (0, 1, 2, 3)
    np.testing.assert_allclose(first.static["SIGMA_ELGA"].data[:, 0], np.arange(n1) + 0.15)
    solid = rep.samples["case__solid"][0]
    np.testing.assert_allclose(solid.static["POROSITY"].data[:, 0], np.arange(n1, len(tets)))

    # second reader: meshio sees the same mesh (in MED's node order) and the same last step
    # (it cannot read a field that lives on one cell type only, hence the plainer file)
    write_med(tmp_path / "plain.med", gauss_field=False)
    ref = meshio.read(tmp_path / "plain.med")
    np.testing.assert_allclose(ref.points, points)
    np.testing.assert_array_equal(cells(ref)[0].data[:, VTK_TO_MED["tetra"]], tets)
    last_key = next(k for k in ref.point_data if k.startswith("TEMP[2]"))
    for s in c.read(path, times="last"):
        # match the zone's points to the whole-mesh numbering through their coordinates
        index = np.argmin(np.abs(pts(s)[:, None, :] - points[None, :, :]).sum(axis=2), axis=1)
        np.testing.assert_allclose(s.fields["TEMP"].data[:, 0], ref.point_data[last_key][index])
    wall = next(iter(c.read(path, zones=["wall"], times=[0])))
    assert wall.kind == "surface" and faces(wall).shape == (len(tris), 3)
    np.testing.assert_allclose(wall.static["POROSITY"].data[:, 0], 500 + np.arange(len(tris)))
    nodes = next(iter(c.read(path, zones=["inlet_nodes"], times="first")))
    assert nodes.kind == "point_cloud" and (pts(nodes)[:, 0] == 0).all()


def test_med_detection_and_malformed(tmp_path: Path) -> None:
    c = MedConnector()
    plain = tmp_path / "plain.med"
    with h5py.File(plain, "w") as h5:
        h5["data"] = np.arange(3)
    assert c.detect(plain) is None and c.detect(tmp_path / "none.med") is None
    with pytest.raises(fc.UnsupportedSourceError, match="ENS_MAA"):
        c.describe(plain)
    path = tmp_path / "case.med"
    write_med(path)
    det = c.detect(path)
    assert det is not None and det.confidence > 0.9 and "MED 3.0.0" in det.notes[0]
    with h5py.File(path, "a") as h5:
        f: Any = h5
        del f["ENS_MAA/mesh/-0000000000000000001-0000000000000000001/NOE/COO"]
    with pytest.raises(fc.ConnectorError, match="unexpected MED layout"):
        c.describe(path)
    write_med(path)
    with h5py.File(path, "a") as h5:
        f: Any = h5
        co = f["CHA/TEMP"][next(iter(f["CHA/TEMP"]))]["NOE/MED_NO_PROFILE_INTERNAL"]
        del co["CO"]
        co.create_dataset("CO", data=np.zeros(5))
    with pytest.raises(fc.ConnectorError, match="holds 5 entities, mesh has 27"):
        list(c.read(path))

    # a field stored on a profile (a subset of the nodes) is NaN elsewhere
    write_med(path, gauss_field=False)
    with h5py.File(path, "a") as h5:
        f = h5
        support = f["CHA/TEMP"][next(iter(f["CHA/TEMP"]))]["NOE"]
        support.attrs["PFL"] = np.bytes_("first_five")
        del support["MED_NO_PROFILE_INTERNAL"]
        part = support.create_group("first_five")
        part.attrs["NBR"], part.attrs["NGA"] = 5, 1
        part.create_dataset("CO", data=np.arange(5, dtype=float) + 100.0)
        prof = f.create_group("PROFILS/first_five")
        prof.attrs["NBR"] = 5
        prof.create_dataset("PFL", data=np.array([1, 3, 5, 7, 9], dtype=np.int32))
    desc = c.describe(path)
    assert any("profile" in w for w in desc.warnings)
    whole = np.full(27, np.nan)
    whole[[0, 2, 4, 6, 8]] = 100.0 + np.arange(5)
    points = tets_with_wall()[0]
    for s in c.read(path, times="first"):  # only the first step was put on the profile
        index = np.argmin(np.abs(pts(s)[:, None, :] - points[None, :, :]).sum(axis=2), axis=1)
        np.testing.assert_allclose(s.fields["TEMP"].data[:, 0], whole[index], equal_nan=True)


# -- SU2 ----------------------------------------------------------------------------------------

SU2_CFG = """\
% SU2 configuration (the handful of keys the connector reads)
SOLVER= RANS
KIND_TURB_MODEL= SST
MATH_PROBLEM= DIRECT
MACH_NUMBER= 0.8
AOA= 1.25
REYNOLDS_NUMBER= 6.5E6
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_SYM= ( NONE )
MARKER_INLET= ( inlet, 288.6, 102010.0, 1.0, 0.0, 0.0 )
MESH_FILENAME= wing.su2
"""


def write_su2(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native SU2 mesh per the SU2 mesh-file documentation: VTK element codes, 0-based node
    ids, an optional trailing index column, markers."""
    points, hexes = hex_grid(2, 2, 2)
    # VTK wedges: the base triangle is numbered so that its normal points away from the top
    wedges = np.concatenate([hexes[:4][:, [0, 3, 1, 4, 7, 5]], hexes[:4][:, [1, 3, 2, 5, 7, 6]]])
    hexes = hexes[4:]
    lines = ["% test mesh", "NDIME= 3", f"NELEM= {len(hexes) + len(wedges)}"]
    lines += [f"12 {' '.join(map(str, h))}" for h in hexes]
    lines += [f"13\t{' '.join(map(str, w))}" for w in wedges]
    lines += [f"NPOIN= {len(points)}"]
    lines += [f"{float(p[0])!r} {float(p[1])!r} {float(p[2])!r} {i}" for i, p in enumerate(points)]
    quads = {"wall": [], "farfield": [], "inlet": []}
    for h in hexes:  # top hexahedra: z = 1 faces are the far field
        quads["farfield"].append(h[[4, 5, 6, 7]])
    for h in np.concatenate([hexes, np.zeros((0, 8), int)]):
        if np.all(points[h[[0, 3, 7, 4]], 0] == 0.0):
            quads["inlet"].append(h[[0, 3, 7, 4]])
    tris = [w[[0, 1, 2]] for w in wedges]  # bottom triangles: the wall
    lines += ["NMARK= 3", "MARKER_TAG= wall", f"MARKER_ELEMS= {len(tris)}"]
    lines += [f"5 {' '.join(map(str, t))}" for t in tris]
    for name in ("farfield", "inlet"):
        lines += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(quads[name])}"]
        lines += [f"9 {' '.join(map(str, q))}" for q in quads[name]]
    path.write_text("\n".join(lines) + "\n")
    return points, hexes, wedges


def test_su2_mesh_markers_and_config(tmp_path: Path) -> None:
    path = tmp_path / "wing.su2"
    points, hexes, wedges = write_su2(path)
    c = Su2Connector()
    bare = c.describe(path)
    assert bare.solver == "SU2" and "no .cfg found" in bare.warnings[0]
    assert all(z.bc_type is None for z in bare.zones)
    (tmp_path / "turb_wing.cfg").write_text(SU2_CFG)  # found through MESH_FILENAME
    rep = check_connector(c, path, tmp_path / "out")
    desc = rep.description
    zones = {z.name: z for z in desc.zones}
    assert list(zones) == ["volume", "wall", "farfield", "inlet"]
    assert zones["volume"].cell_types == {"hexahedron": 4, "wedge": 8} and zones["volume"].default
    assert [zones[n].bc_type for n in ("wall", "farfield", "inlet")] == [
        "wall",
        "farfield",
        "inlet",
    ]
    assert zones["wall"].kind == "surface" and not zones["wall"].default
    assert desc.physics == {
        "solver": "RANS",
        "turbulence_model": "SST",
        "math_problem": "DIRECT",
        "compressible": True,
    }
    assert desc.parameters == {"mach_number": 0.8, "aoa": 1.25, "reynolds_number": 6.5e6}
    assert desc.boundary_conditions["inlet"] == {
        "type": "inlet",
        "su2_option": "MARKER_INLET",
        "values": [288.6, 102010.0, 1.0, 0.0, 0.0],
    }
    assert desc.domain == "cfd" and desc.temporal == "steady"
    vol = rep.first()
    assert vol.kind == "mesh" and vol.metadata["boundary_conditions"]["wall"]["type"] == "wall"
    np.testing.assert_allclose(pts(vol), points)
    # SU2 uses VTK's node order: the cells come out as written, none needs mirroring
    np.testing.assert_array_equal(cells(vol)[vol.cell_types == 12], hexes)
    assert (volumes(vol) > 0).all()
    assert "reoriented_cells" not in vol.metadata
    wall = next(iter(c.read(path, zones=["wall"])))
    assert wall.metadata["bc_type"] == "wall" and (pts(wall)[:, 2] == 0).all()
    assert faces(wall).shape == (8, 3)

    # second parser: meshio's SU2 reader agrees on points and element counts
    ref = meshio.read(path)
    np.testing.assert_allclose(ref.points, points)
    counts = {b.type: len(b.data) for b in ref.cells}
    assert counts == {"hexahedron": 4, "wedge": 8, "triangle": 8, "quad": 6}
    data = parse_su2(path)
    assert data["markers"] == ["wall", "farfield", "inlet"]
    np.testing.assert_array_equal(dict(data["blocks"])["hexahedron"], cells(ref)[0].data)


def test_su2_planar_and_malformed(tmp_path: Path) -> None:
    c = Su2Connector()
    points, tris = tri_grid(2, 2)
    path = tmp_path / "square.su2"
    path.write_text(
        "NDIME= 2\nNELEM= 8\n"
        + "".join(f"5 {t[0]} {t[1]} {t[2]}\n" for t in tris)
        + "NPOIN= 9\n"
        + "".join(f"{float(p[0])!r} {float(p[1])!r}\n" for p in points)
        + "NMARK= 1\nMARKER_TAG= lower\nMARKER_ELEMS= 2\n3 0 1\n3 1 2\n"
    )
    rep = check_connector(c, path, tmp_path / "out")
    s = rep.first()
    assert s.kind == "mesh" and pts(s).shape == (9, 2) and rep.description.zones[0].dim == 2
    np.testing.assert_array_equal(s.cells, tris)
    marker = rep.description.zones[1]
    assert (marker.kind, marker.role, marker.cell_types) == ("mesh", "boundary", {"line": 2})

    text = path.read_text()
    short = tmp_path / "short.su2"
    short.write_text(text.replace("NPOIN= 9", "NPOIN= 12"))
    with pytest.raises(fc.ConnectorError, match=r"line \d+: expected 12 point rows"):
        c.describe(short)
    odd = tmp_path / "odd.su2"
    odd.write_text(text.replace("\n5 ", "\n77 ", 1))
    with pytest.raises(fc.ConnectorError, match="unknown SU2 element type 77"):
        list(c.read(odd))
    assert c.detect(odd) is not None and c.detect(tmp_path / "wing.cfg") is None
    empty = tmp_path / "empty.su2"
    empty.write_text("% nothing\n")
    assert c.detect(empty) is None
    with pytest.raises(fc.ConnectorError, match="not a complete SU2 mesh"):
        c.describe(empty)


# -- generic meshio formats ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "regions"),
    [
        ("mesh.mesh", True),
        ("mesh.meshb", True),
        ("mesh.vol", True),
        ("mesh.lb8.ugrid", False),
        ("mesh.f3grid", True),
        ("mesh.node", False),
        ("mesh.h5m", False),
        ("mesh.xml", False),
        ("mesh.avs", False),
        ("mesh.mdpa", False),
        ("mesh.post", False),
        ("mesh.hmf", False),
    ],
)
def test_generic_meshio_formats(tmp_path: Path, name: str, regions: bool) -> None:
    points, tets, n1, tris = tets_with_wall()
    ext = name[name.index(".") :]
    label = {".mesh": "medit:ref", ".meshb": "medit:ref", ".vol": "netgen:index"}
    region = np.where(np.arange(len(tets)) < n1, 1, 2)
    if ext == ".f3grid":
        mesh = meshio.Mesh(
            points,
            [("tetra", tets)],
            cell_sets={"lower": [np.arange(n1)], "upper": [np.arange(n1, len(tets))]},
        )
    elif regions:
        mesh = meshio.Mesh(
            points,
            [("tetra", tets), ("triangle", tris)],
            cell_data={label[ext]: [region, np.full(len(tris), 5)]},
        )
        if ext in (".mesh", ".meshb"):
            mesh.point_data["medit:ref"] = np.zeros(len(points), int)
    else:
        mesh = meshio.Mesh(points, [("tetra", tets)])
    path = tmp_path / name
    try:
        meshio.write(path, mesh)
    except Exception as e:  # noqa: BLE001 - optional writer dependencies (e.g. lxml)
        pytest.skip(f"meshio cannot write {ext} here: {e}")
    c = MeshioConnector()
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    volume = [s[0] for s in rep.samples.values()]
    assert sum(s.n_cells for s in volume) == len(tets) and all(s.kind == "mesh" for s in volume)
    for s in volume:
        assert (volumes(s) > 0).all()
        assert pts(s).shape[1] == 3 and not np.isnan(s.points).any()
    if ext == ".f3grid":
        assert [z.n_cells for z in rep.description.zones] == [n1, len(tets) - n1]
        assert list(zones) == ["lower", "upper"]
    elif regions:
        # region ids become zones (meshio's Netgen reader names them cd<codim>_<id>)
        default = [z for z in rep.description.zones if z.default]
        assert [z.n_cells for z in default] == [n1, len(tets) - n1]
        (wall,) = [z for z in rep.description.zones if not z.default]
        assert wall.kind == "surface" and wall.n_cells == len(tris)
        assert wall.name in ("region_5", "cd1_5")
        assert same_mesh(volume[0], points, tets[:n1])
    else:
        assert same_mesh(volume[0], points, tets)


def test_generic_meshio_detection_is_header_checked(tmp_path: Path) -> None:
    c = MeshioConnector()
    (tmp_path / "config.xml").write_text("<settings><a/></settings>")
    (tmp_path / "notes.mesh").write_text("just some text\n")
    (tmp_path / "lonely.node").write_text("4 3 0 0\n")
    (tmp_path / "table.vol").write_text("1 2 3\n")
    for name in ("config.xml", "notes.mesh", "lonely.node", "table.vol", "absent.ugrid"):
        assert c.detect(tmp_path / name) is None, name
    for foreign in (
        "a.vtu",
        "a.cgns",
        "a.inp",
        "a.bdf",
        "a.stl",
        "a.dat",
        "a.msh",
        "a.h5",
        "a.case",
    ):
        (tmp_path / foreign).write_text("x")
        assert c.detect(tmp_path / foreign) is None, foreign
    broken = tmp_path / "broken.mesh"
    broken.write_text("MeshVersionFormatted 2\nDimension 3\nVertices\n5\n0 0 0 1\n")
    assert c.detect(broken) is not None
    with pytest.raises(fc.ConnectorError, match="meshio could not read"):
        c.describe(broken)
    with pytest.raises(fc.UnsupportedSourceError):
        c.describe(tmp_path / "absent.ugrid")


# -- real public files --------------------------------------------------------------------------


def fetch_into(directory: Path, *sources: tuple[str, str]) -> list[Path]:
    """Download ``(url, file name)`` pairs and put them side by side (heavy-data files and
    configs are looked up next to the main file)."""
    directory.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy(fetch_sample(url, name), directory / name)) for url, name in sources]


@pytest.mark.network
def test_public_exodus_files(tmp_path: Path) -> None:
    c = ExodusConnector()
    # CUBIT mesh + SALSA reacting-flow results (VTK / ParaView tutorial data)
    rep = check_connector(c, fetch_sample(PYVISTA + "disk_out_ref.ex2"), tmp_path / "disk")
    desc = rep.description
    assert desc.solver == "salsa" and desc.temporal == "steady" and desc.domain == "cfd"
    zones = {z.name: z for z in desc.zones}
    assert zones["block_1"].cell_types == {"hexahedron": 7472} and zones["block_1"].n_points == 8499
    assert [z.name for z in desc.zones if z.kind == "surface"] == [
        f"sideset_{i}" for i in range(1, 8)
    ]
    s = rep.first()
    assert (volumes(s) > 0).all() and "reoriented_cells" not in s.metadata
    temp = s.fields["Temp"].data
    assert abs(float(temp.min()) - 293.15) < 1e-3 and abs(float(temp.max()) - 913.15) < 1e-3
    assert s.fields["V"].shape == (8499, 3)
    side = next(iter(c.read(fetch_sample(PYVISTA + "disk_out_ref.ex2"), zones=["sideset_7"])))
    assert side.kind == "surface" and faces(side).shape == (2 * 964, 3)
    assert np.allclose(pts(side)[:, 0], 0.0) and "face_parent" in side.static

    # LaGriT wedge mesh: Exodus numbers the wedge base the other way round than VTK
    rep = check_connector(c, fetch_sample(PYVISTA + "mesh_fs8.exo"), tmp_path / "fs8")
    assert [z.cell_types for z in rep.description.zones[:3]] == [
        {"wedge": 336},
        {"wedge": 336},
        {"wedge": 672},
    ]
    for group in rep.samples.values():
        assert (volumes(group[0]) > 0).all() and group[0].metadata["reoriented_cells"] > 0

    # one piece of a 4-way decomposition of the PRONTO3D "can" impact (44 time steps)
    path = fetch_sample(PYVISTA + "ParallelExodus/can.e.4.0")
    det = c.detect(path)
    assert det is not None and "4-way parallel" in det.notes[1]
    rep = check_connector(c, path, tmp_path / "can", times=[0, 21, 43])
    desc = rep.description
    assert desc.time.n_steps == 44 and desc.domain == "structural"
    assert any("parallel decomposition" in w for w in desc.warnings)
    steps = rep.samples["can.e.4"]
    assert [s.time_index for s in steps] == [0, 21, 43] and steps[1].points is None
    assert float(np.abs(steps[0].fields["DISPL"].data).max()) == 0.0
    assert abs(float(steps[2].fields["DISPL"].data.min()) + 12.3037) < 1e-3
    assert steps[2].fields["EQPS"].location == "cell" and steps[2].metadata["globals"]["KE"] > 0
    nodes = next(iter(c.read(path, zones=["nodeset_100"], times="first", fields=["VEL"])))
    assert (
        nodes.kind == "point_cloud"
        and pts(nodes).shape == (120, 3)
        and list(nodes.fields) == ["VEL"]
    )
    moved = next(iter(c.read(path, times="last", options={"apply_displacements": True})))
    np.testing.assert_allclose(
        pts(moved), pts(steps[0]) + steps[2].fields["DISPL"].data, rtol=1e-5, atol=1e-5
    )

    # MOOSE thermal-hydraulics pipe (1-D elements, element and global variables)
    flow = fetch_sample(VTK_DATA + VTK_FLOW1D, "Flow1D.e")
    rep = check_connector(c, flow, tmp_path / "flow", times="last")
    assert rep.description.solver == "MOOSE" and rep.description.time.n_steps == 51
    pipe = rep.first()
    assert pipe.time == 0.5 and np.asarray(pipe.cell_types).tolist() == [3] * 9
    assert pipe.fields["T"].location == "cell" and "T_air_in" in pipe.metadata["globals"]


@pytest.mark.network
def test_public_xdmf_files(tmp_path: Path) -> None:
    c = XdmfConnector()
    # dolfinx XDMFFile.write_function output (pyvista/vtk-data)
    (path, _) = fetch_into(
        tmp_path / "dolfinx",
        (PYVISTA + "dolfinx/no_checkpoint.xdmf", "no_checkpoint.xdmf"),
        (PYVISTA + "dolfinx/no_checkpoint.h5", "no_checkpoint.h5"),
    )
    rep = check_connector(c, path, tmp_path / "o1")
    desc = rep.description
    assert desc.metadata["engine"] == "vtk" and desc.time.values == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert desc.zones[0].cell_types == {"tetra": 3072} and desc.zones[0].n_points == 729
    steps = rep.samples["no_checkpoint"]
    assert len(steps) == 5 and steps[4].fields["f_5"].shape == (729, 1)
    # FEniCS does not orient its tetrahedra: half of them are mirrored to VTK's convention
    assert (volumes(steps[0]) > 0).all() and steps[0].metadata["reoriented_cells"] == 1536
    # dolfinx mesh with quadratic triangles on a sphere (inline XML heavy data): meshio engine
    sphere = check_connector(c, fetch_sample(PYVISTA + "dolfinx/mesh.xdmf"), tmp_path / "o2")
    assert sphere.description.metadata["engine"] == "meshio"
    assert sphere.description.zones[0].cell_types == {"quadratic_triangle": 484}
    assert sphere.first().kind == "surface" and faces(sphere.first()).shape == (484, 3)
    np.testing.assert_allclose(np.linalg.norm(pts(sphere.first()), axis=1), 1.0, atol=1e-6)
    # dolfinx checkpoint: FiniteElementFunction attributes are reported, the mesh is read
    (path, _) = fetch_into(
        tmp_path / "checkpoint",
        (PYVISTA + "dolfinx/checkpoint.xdmf", "checkpoint.xdmf"),
        (PYVISTA + "dolfinx/checkpoint.h5", "checkpoint.h5"),
    )
    desc = c.describe(path)
    assert any("FiniteElementFunction" in w for w in desc.warnings) and not desc.fields
    # meshio's own time series
    (path, _) = fetch_into(
        tmp_path / "meshio",
        (PYVISTA + "meshio/out.xdmf", "out.xdmf"),
        (PYVISTA + "meshio/out.h5", "out.h5"),
    )
    rep = check_connector(c, path, tmp_path / "o3")
    assert rep.description.metadata["engine"] == "meshio"
    assert rep.description.time.values == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert pts(rep.first()).shape == (4, 2) and set(rep.first().fields) == {"phi", "u", "a"}
    # VTK testing data: structured and spatial/temporal collections with inline data (XDMF 2)
    grid = fetch_sample(VTK_DATA + VTK_CORECT, "3DCoRectMesh.xmf")
    s = check_connector(c, grid, tmp_path / "o4").first()
    assert s.kind == "grid" and s.fields["NodeScalars"].shape == (3, 2, 2, 1)
    # origin and spacing are listed Z Y X in the file: (0, 0, 3) and (1, 1.5, 2)
    np.testing.assert_allclose(s.coords["x"], [3.0, 5.0, 7.0])
    np.testing.assert_allclose(s.coords["y"], [0.0, 1.5])
    assert float(s.fields["NodeScalars"].data[2, 1, 1, 0]) == 1200.0
    coll = fetch_sample(VTK_DATA + VTK_SPATIAL_TIME, "TetrahedronSpatialTimestep.xmf")
    rep = check_connector(c, coll, tmp_path / "o5")
    assert rep.description.time.values == [0.0, 1.0, 2.0] and len(rep.samples) == 2
    assert all(len(group) == 3 for group in rep.samples.values())


@pytest.mark.network
def test_public_gmsh_med_su2_files(tmp_path: Path) -> None:
    # Gmsh 4.2 output of the same geometry in MSH 2.2 and 4.1 (meshio test meshes)
    c: Any = GmshConnector()
    seen = []
    for name in ("insulated-2.2.msh", "insulated-4.1.msh"):
        rep = check_connector(c, fetch_sample(MESHIO + "msh/" + name), tmp_path / name, zones=["*"])
        zones = {z.name: z for z in rep.description.zones}
        assert [(n, z.n_cells, z.default) for n, z in zones.items()] == [
            ("wire", 45, True),
            ("insulation", 66, True),
            ("convection", 21, False),
        ]
        assert zones["wire"].kind == "mesh" and zones["wire"].dim == 2
        assert zones["convection"].cell_types == {"line": 21}
        wire = rep.samples[f"{name[:-4]}__wire"][0]
        assert pts(wire).shape == (31, 2)
        seen.append(np.sort(pts(wire), axis=0))
    np.testing.assert_allclose(seen[0], seen[1], atol=1e-12)

    # Code_Aster result written through Salome's MED library (meshio test meshes)
    c = MedConnector()
    rep = check_connector(c, fetch_sample(MESHIO + "med/box.med"), tmp_path / "box")
    desc = rep.description
    assert desc.solver == "Code_Aster" and desc.metadata["med_version"] == "3.3.1"
    fields = {f.name: f for f in desc.fields}
    assert fields["resu____DEPL"].component_names == ["DX", "DY", "DZ"]
    assert (
        fields["resu____SIEF_ELNO"].components == 6
        and fields["resu____SIEF_ELNO"].location == "cell"
    )
    box = rep.first()
    assert (volumes(box) > 0).all() and "reoriented_cells" not in box.metadata
    assert box.fields["resu____ENEL_ELEM"].shape == (1, 1)
    assert abs(float(box.fields["resu____ENEL_ELEM"].data[0, 0]) - 4550.399) < 1e-2
    # Salome mesh with overlapping groups, pyramids and tetrahedra
    rep = check_connector(c, fetch_sample(MESHIO + "med/cylinder.med"), tmp_path / "cyl")
    zones = {z.name: z for z in rep.description.zones}
    assert zones["volume"].cell_types == {"pyramid": 18, "tetra": 63} and zones["volume"].default
    assert not zones["A"].default and zones["Top"].kind == "surface"
    assert zones["Top circle"].cell_types == {"line": 8} and zones["Side"].kind == "point_cloud"
    cyl = rep.first()
    # MED -> VTK node order is right for tetrahedra and pyramids: nothing left to mirror
    assert (volumes(cyl) > 0).all() and "reoriented_cells" not in cyl.metadata

    # SU2: the mesh of the SU2 mesh-file documentation and the QuickStart NACA0012 case
    c = Su2Connector()
    mix = check_connector(c, fetch_sample(MESHIO + "su2/mixgrid.su2"), tmp_path / "mix")
    assert mix.description.zones[0].cell_types == {"tetra": 6, "wedge": 4}
    assert [z.name for z in mix.description.zones[1:]] == [
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
    ]
    assert (volumes(mix.first()) > 0).all() and "reoriented_cells" not in mix.first().metadata
    quick = "https://raw.githubusercontent.com/su2code/SU2/master/QuickStart/"
    (mesh, _) = fetch_into(
        tmp_path / "naca",
        (quick + "mesh_NACA0012_inv.su2", "mesh_NACA0012_inv.su2"),
        (quick + "inv_NACA0012.cfg", "inv_NACA0012.cfg"),
    )
    rep = check_connector(c, mesh, tmp_path / "naca-out", zones=["*"])
    desc = rep.description
    assert [(z.name, z.n_cells, z.bc_type) for z in desc.zones] == [
        ("volume", 10216, None),
        ("airfoil", 200, "wall"),
        ("farfield", 50, "farfield"),
    ]
    assert desc.physics["solver"] == "EULER" and desc.parameters["mach_number"] == 0.8
    assert desc.parameters["aoa"] == 1.25 and desc.boundary_conditions["airfoil"]["type"] == "wall"
    airfoil = rep.samples["mesh_NACA0012_inv__airfoil"][0]
    assert pts(airfoil).shape == (200, 2) and float(pts(airfoil)[:, 0].max()) == 1.0
    ref = meshio.read(mesh)  # second parser
    np.testing.assert_allclose(pts(rep.first()), ref.points)
    np.testing.assert_array_equal(cells(rep.first()), ref.cells[0].data)


@pytest.mark.network
def test_public_generic_meshio_files(tmp_path: Path) -> None:
    c = MeshioConnector()
    expected = {
        "medit/cube86.mesh": {"tetra": 86},
        "medit/hch_strct.4.meshb": {"wedge": 96, "hexahedron": 144},
        "medit/sphere_mixed.1.meshb": {"tetra": 9072, "wedge": 3024},
        "netgen/periodic_3d.vol": {"tetra": 166},
        "ugrid/hch_strct.4.lb8.ugrid": {"wedge": 96, "hexahedron": 144},
        "ugrid/sphere_mixed.1.lb8.ugrid": {"tetra": 9072, "wedge": 3024},
        "flac3d/flac3d_mesh_ex.f3grid": {"hexahedron": 75, "pyramid": 15, "wedge": 15, "tetra": 3},
        "flac3d/flac3d_mesh_ex_bin.f3grid": {
            "hexahedron": 75,
            "pyramid": 15,
            "wedge": 15,
            "tetra": 3,
        },
    }
    for name, census in expected.items():
        rep = check_connector(c, fetch_sample(MESHIO + name), tmp_path / name.replace("/", "_"))
        total: dict[str, int] = {}
        for z in rep.description.zones:
            if z.default:
                for k, v in z.cell_types.items():
                    total[k] = total.get(k, 0) + v
        assert total == census, name
        for group in rep.samples.values():
            assert (volumes(group[0]) > 0).all(), name
            # meshio's wedge is converted to VTK's by the reader itself (FLAC3D's own
            # numbering is left to the orientation repair)
            assert "reoriented_cells" not in group[0].metadata or "flac3d" in name, name
    flac = c.describe(fetch_sample(MESHIO + "flac3d/flac3d_mesh_ex.f3grid"))
    assert [z.name for z in flac.zones] == [
        "Brick1",
        "Pyramid2",
        "Tetrahedron4",
        "Wedge3",
        "bottom",
    ]
    assert flac.zones[-1].kind == "surface" and not flac.zones[-1].default
    ugrid = c.describe(fetch_sample(MESHIO + "ugrid/hch_strct.4.lb8.ugrid"))
    assert [(z.name, z.n_cells) for z in ugrid.zones[1:]] == [
        ("region_1", 15),
        ("region_2", 15),
        ("region_3", 160),
    ]
    # TetGen writes a .node/.ele pair: either file opens the mesh
    node, ele = fetch_into(
        tmp_path / "tetgen",
        (MESHIO + "tetgen/mesh.node", "mesh.node"),
        (MESHIO + "tetgen/mesh.ele", "mesh.ele"),
    )
    for path in (node, ele):
        rep = check_connector(c, path, tmp_path / f"tetgen-{path.suffix[1:]}")
        assert [z.n_cells for z in rep.description.zones] == [235, 69]
    planar = check_connector(c, fetch_sample(MESHIO + "netgen/periodic_2d.vol"), tmp_path / "ng2d")
    assert planar.first().kind == "mesh" and pts(planar.first()).shape == (40, 2)
