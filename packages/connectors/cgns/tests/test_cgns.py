# SPDX-License-Identifier: Apache-2.0
"""CGNS connector tests.

The fixtures are CGNS/HDF5 files written here with h5py following the public SIDS-to-HDF5
mapping. Every fixture is also opened with VTK's independent ``vtkCGNSReader`` (which embeds
the CGNS library): it has to accept the file and agree with the h5py reader on counts,
connectivity and field values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from frontier_connector_cgns.connector import CgnsConnector, bc_type_id, file_format

import frontier_connectors as fc
from frontier_connectors import _vtk
from frontier_connectors.errors import ConnectorError, SelectionError, UnsupportedSourceError
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    fetch_sample,
    hex_grid,
    open_store,
    quad_grid,
    tet_grid,
)

TRI_3, QUAD_4, TETRA_4, TETRA_10, PENTA_6, PENTA_15, HEXA_8, HEXA_20, HEXA_27 = (
    5, 7, 10, 11, 14, 15, 17, 18, 19,
)  # fmt: skip
MIXED, NGON_N, NFACE_N = 20, 22, 23
_CODES = {"int32": "I4", "int64": "I8", "float32": "R4", "float64": "R8", "int8": "C1"}


# -- CGNS/HDF5 writer (SIDS-to-HDF5) -------------------------------------------------------------


def node(parent: Any, name: str, label: str, data: Any = None) -> Any:
    """One CGNS node. ``data`` is given in HDF5 order (the Fortran array transposed); a ``str``
    becomes ``C1`` character data."""
    g = parent.create_group(name, track_order=True)
    g.attrs.create("name", np.bytes_(name), dtype="S33")
    g.attrs.create("label", np.bytes_(label), dtype="S33")
    arr = None
    if isinstance(data, str):
        arr = np.frombuffer(data.encode("ascii"), dtype=np.int8)
    elif data is not None:
        arr = np.asarray(data)
    code = "MT" if arr is None else _CODES[arr.dtype.name]
    g.attrs.create("type", np.bytes_(code), dtype="S3")
    g.attrs.create("flags", np.array([1], dtype=np.int32))
    if arr is not None:
        g.create_dataset(" data", data=arr)
    return g


def chars(rows: list[str], width: int = 32) -> np.ndarray:
    """Fortran ``character*32`` rows as stored in HDF5: ``(n, 32)`` int8, blank padded."""
    return np.array([np.frombuffer(r.ljust(width).encode(), dtype=np.int8) for r in rows])


def new_file(path: Path, version: float = 4.2) -> Any:
    f = h5py.File(path, "w", track_order=True)
    f.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype="S33")
    f.attrs.create("label", np.bytes_("Root Node of HDF5 File"), dtype="S33")
    f.attrs.create("type", np.bytes_("MT"), dtype="S3")
    f.create_dataset(" format", data=np.frombuffer(b"NATIVE\x00", dtype=np.int8))
    hdf5 = b"HDF5 Version 1.14.0".ljust(33, b"\x00")
    f.create_dataset(" hdf5version", data=np.frombuffer(hdf5, dtype=np.int8))
    node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", np.array([version], np.float32))
    return f


def base(f: Any, name: str = "Base", cell_dim: int = 3, phys_dim: int = 3) -> Any:
    return node(f, name, "CGNSBase_t", np.array([cell_dim, phys_dim], np.int32))


def coordinates(zone: Any, points: np.ndarray, shape: tuple[int, ...], name: str) -> Any:
    g = node(zone, name, "GridCoordinates_t")
    for i, axis in enumerate("XYZ"[: points.shape[1]]):
        node(g, f"Coordinate{axis}", "DataArray_t", points[:, i].reshape(shape))
    return g


def unstructured_zone(b: Any, name: str, points: np.ndarray, n_cells: int) -> Any:
    z = node(b, name, "Zone_t", np.array([[len(points)], [n_cells], [0]], np.int32))
    node(z, "ZoneType", "ZoneType_t", "Unstructured")
    coordinates(z, points, (len(points),), "GridCoordinates")
    return z


def structured_zone(b: Any, name: str, points: np.ndarray, dims: tuple[int, ...]) -> Any:
    """``dims`` = vertices per direction (ni, nj[, nk]); ``points`` ordered i-fastest."""
    size = np.array([dims, [d - 1 for d in dims], [0] * len(dims)], np.int32)
    z = node(b, name, "Zone_t", size)
    node(z, "ZoneType", "ZoneType_t", "Structured")
    coordinates(z, points, tuple(reversed(dims)), "GridCoordinates")
    return z


def section(
    zone: Any, name: str, etype: int, start: int, n: int, conn: Any, offsets: Any = None
) -> Any:
    s = node(zone, name, "Elements_t", np.array([etype, 0], np.int32))
    node(s, "ElementRange", "IndexRange_t", np.array([start, start + n - 1], np.int32))
    if offsets is not None:
        node(s, "ElementStartOffset", "DataArray_t", np.asarray(offsets, np.int32))
    node(s, "ElementConnectivity", "DataArray_t", np.asarray(conn, np.int32).ravel())
    return s


def solution(
    zone: Any,
    name: str,
    arrays: dict[str, np.ndarray],
    location: str | None = None,
    shape: tuple[int, ...] | None = None,
) -> Any:
    s = node(zone, name, "FlowSolution_t")
    if location is not None:
        node(s, "GridLocation", "GridLocation_t", location)
    for key, values in arrays.items():
        a = np.asarray(values, dtype=np.float64)
        node(s, key, "DataArray_t", a.reshape(shape) if shape else a)
    return s


def bc(
    zone: Any,
    name: str,
    bc_type: str,
    *,
    point_range: Any = None,
    point_list: Any = None,
    location: str | None = None,
    family: str | None = None,
) -> Any:
    zbc = zone["ZoneBC"] if "ZoneBC" in zone else node(zone, "ZoneBC", "ZoneBC_t")
    b = node(zbc, name, "BC_t", bc_type)
    if point_range is not None:
        node(b, "PointRange", "IndexRange_t", np.asarray(point_range, np.int32))
    if point_list is not None:
        node(b, "PointList", "IndexArray_t", np.asarray(point_list, np.int32).reshape(-1, 1))
    if location is not None:
        node(b, "GridLocation", "GridLocation_t", location)
    if family is not None:
        node(b, "FamilyName", "FamilyName_t", family)
    return b


def split_xyz(name: str, vectors: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{name}{axis}": vectors[:, i] for i, axis in enumerate("XYZ")}


def boundary_faces(points: np.ndarray, tets: np.ndarray, axis: int, value: float) -> np.ndarray:
    """Triangles of a tet mesh that lie in the plane ``x[axis] == value``."""
    faces = np.concatenate([tets[:, list(f)] for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    on = np.isclose(points[faces][:, :, axis], value).all(axis=1)
    return faces[on]


# -- VTK cross-check -----------------------------------------------------------------------------


def vtk_zones(path: Path, *, patches: bool = False, time: float | None = None) -> dict[str, Any]:
    """Leaf datasets of ``vtkCGNSReader`` keyed by block path. Fails when VTK reports an error."""
    from vtkmodules.vtkCommonCore import vtkCommand
    from vtkmodules.vtkIOCGNSReader import vtkCGNSReader

    errors: list[str] = []

    def on_error(obj: Any, event: str, message: str) -> None:
        errors.append(message)

    on_error.CallDataType = "string0"  # type: ignore[attr-defined]
    reader: Any = vtkCGNSReader()
    reader.AddObserver(vtkCommand.ErrorEvent, on_error)
    assert reader.CanReadFile(str(path)), f"VTK cannot open {path}"
    reader.SetFileName(str(path))
    reader.SetLoadBndPatch(patches)
    reader.UpdateInformation()
    reader.EnableAllBases()
    reader.EnableAllPointArrays()
    reader.EnableAllCellArrays()
    obj = _vtk.update_to_time(reader, time)
    assert not errors, errors
    out = dict(_vtk.iter_leaves(obj))
    out["__times__"] = _vtk.reader_times(reader)
    return out


def vtk_sample(ds: Any) -> Any:
    return _vtk.dataset_to_sample(ds, "vtk")


def only(samples: Any) -> Any:
    """The single sample of ``samples``, untyped so geometry can be compared directly."""
    (sample,) = list(samples)
    return sample


def edit(path: Path) -> Any:
    return h5py.File(path, "r+")


def row_sets(cells: np.ndarray) -> list[frozenset[int]]:
    return [frozenset(int(v) for v in row if v >= 0) for row in cells]


# -- structured ----------------------------------------------------------------------------------


def write_structured(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points, cells = hex_grid(3, 2, 2, size=(3.0, 1.0, 0.5))
    dims = (4, 3, 3)
    f = new_file(path)
    b = base(f)
    node(b, "DataClass", "DataClass_t", "Dimensional")
    units = chars(["Kilogram", "Millimeter", "Second", "Kelvin", "Degree"])
    node(b, "DimensionalUnits", "DimensionalUnits_t", units)
    ref = node(b, "ReferenceState", "ReferenceState_t")
    node(ref, "Mach", "DataArray_t", np.array([0.8]))
    node(ref, "Reynolds", "DataArray_t", np.array([6.5e6]))
    node(ref, "AngleofAttack", "DataArray_t", np.array([2.31]))
    eqs = node(b, "FlowEquationSet", "FlowEquationSet_t")
    node(eqs, "EquationDimension", '"int"', np.array([3], np.int32))
    node(eqs, "GoverningEquations", "GoverningEquations_t", "NSTurbulent")
    node(eqs, "TurbulenceModel", "TurbulenceModel_t", "OneEquation_SpalartAllmaras")
    fam = node(b, "Hull", "Family_t")
    node(fam, "FamilyBC", "FamilyBC_t", "BCWallViscous")
    z = structured_zone(b, "blk-1", points, dims)
    node(z, "FamilyName", "FamilyName_t", "Fluid")
    a = analytic_fields(points)
    solution(z, "Nodes", {"Pressure": a["p"], **split_xyz("Velocity", a["U"])}, shape=dims[::-1])
    centres = points[cells].mean(axis=1)
    cc = analytic_fields(centres)
    solution(
        z, "Cells", {"Pressure": cc["p"], "Density": 1.0 + centres[:, 0]}, "CellCenter", (2, 2, 3)
    )
    bc(z, "inflow", "BCInflowSubsonic", point_range=[[1, 1, 1], [1, 3, 3]])
    bc(z, "hull", "FamilySpecified", point_range=[[1, 1, 1], [4, 3, 1]], family="Hull")
    bc(z, "top", "BCSymmetryPlane", point_range=[[1, 1, 3], [3, 2, 3]], location="KFaceCenter")
    f.close()
    return points, cells


def test_structured_zone_metadata_fields_and_patches(tmp_path: Path) -> None:
    path = tmp_path / "block.cgns"
    points, cells = write_structured(path)
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out", min_confidence=0.95)
    d = rep.description
    assert d.format == "cgns-hdf5" and d.domain == "cfd" and d.temporal == "steady"
    assert d.units == {
        "mass": "kg", "length": "mm", "time": "s", "temperature": "K", "angle": "deg"
    }  # fmt: skip
    assert d.parameters["Mach"] == 0.8 and d.parameters["Ma"] == 0.8
    assert d.parameters["Re"] == 6.5e6 and d.parameters["AngleofAttack"] == 2.31
    assert d.physics == {
        "equation_dimension": 3,
        "governing_equations": "NSTurbulent",
        "turbulence_model": "OneEquation_SpalartAllmaras",
    }
    assert d.metadata["cgns_version"] == pytest.approx(4.2)
    zones = {z.name: z for z in d.zones}
    assert list(zones) == ["blk-1", "blk-1/inflow", "blk-1/hull", "blk-1/top"]
    vol = zones["blk-1"]
    assert (vol.kind, vol.role, vol.default, vol.shape) == ("mesh", "volume", True, [4, 3, 3])
    assert (vol.n_points, vol.n_cells, vol.cell_types) == (36, 12, {"hexahedron": 12})
    assert [zones[n].bc_type for n in list(zones)[1:]] == ["inlet", "wall", "symmetry"]
    assert all(z.kind == "surface" and not z.default for z in list(zones.values())[1:])
    assert (zones["blk-1/inflow"].n_cells, zones["blk-1/hull"].n_cells) == (4, 6)
    assert zones["blk-1/top"].n_cells == 6
    assert d.boundary_conditions["blk-1/hull"] == {
        "type": "wall", "cgns_type": "BCWallViscous", "family": "Hull", "location": "Vertex"
    }  # fmt: skip
    fields = {f.name: (f.location, f.components, f.units) for f in d.fields}
    assert fields == {
        "Pressure": ("point", 1, "kg/mm/s^2"),
        "Velocity": ("point", 3, "mm/s"),
        "Pressure__cell": ("cell", 1, "kg/mm/s^2"),
        "Density": ("cell", 1, "kg/mm^3"),
    }

    s = only(rep.samples["block"])
    assert s.kind == "mesh" and s.time is None and s.metadata["structured_shape"] == [4, 3, 3]
    assert s.metadata["family"] == "Fluid" and s.metadata["units"]["length"] == "mm"
    assert s.metadata["parameters"]["Re"] == 6.5e6
    np.testing.assert_allclose(s.points, points)
    np.testing.assert_array_equal(s.cells, cells)
    assert set(s.cell_types.tolist()) == {12}
    a = analytic_fields(points)
    np.testing.assert_allclose(s.fields["Pressure"].data[:, 0], a["p"], rtol=1e-6)
    np.testing.assert_allclose(s.fields["Velocity"].data, a["U"], rtol=1e-6)
    assert s.fields["Velocity"].component_names == ["x", "y", "z"]
    centres = points[cells].mean(axis=1)
    np.testing.assert_allclose(s.fields["Density"].data[:, 0], 1.0 + centres[:, 0], rtol=1e-6)
    assert s.fields["Pressure__cell"].location == "cell"

    # the independent CGNS reader in VTK accepts the file and sees the same data
    v = vtk_zones(path)["Base/blk-1"]
    assert (v.GetNumberOfPoints(), v.GetNumberOfCells()) == (36, 12)
    np.testing.assert_allclose(_vtk.points(v), s.points)
    vp, vc = _vtk.attributes(v.GetPointData(), None), _vtk.attributes(v.GetCellData(), None)
    np.testing.assert_allclose(vp["Velocity"], s.fields["Velocity"].data, rtol=1e-6)
    np.testing.assert_allclose(vp["Pressure"], s.fields["Pressure"].data[:, 0], rtol=1e-6)
    np.testing.assert_allclose(vc["Density"], s.fields["Density"].data[:, 0], rtol=1e-6)

    # boundary patches: geometry, vertex data on the patch points, adjacent-cell data
    patches = {s.zone: s for s in c.read(path, zones=["blk-1/*"])}
    assert set(patches) == {"blk-1/inflow", "blk-1/hull", "blk-1/top"}
    inflow = patches["blk-1/inflow"]
    assert inflow.kind == "surface" and inflow.metadata["bc_type"] == "inlet"
    assert inflow.points is not None and inflow.faces is not None
    assert inflow.points.shape == (9, 3) and inflow.faces.shape == (8, 3)
    np.testing.assert_allclose(inflow.points[:, 0], 0.0)
    np.testing.assert_allclose(
        inflow.fields["Pressure"].data[:, 0], analytic_fields(inflow.points)["p"], rtol=1e-6
    )
    hull = patches["blk-1/hull"]
    assert hull.points is not None and hull.faces is not None
    np.testing.assert_allclose(hull.points[:, 2], 0.0)
    assert hull.points.shape == (12, 3) and hull.faces.shape == (12, 3)
    # cell data of a patch = the cell layer that touches it (x of the cell centre + 1 here)
    tri_x = hull.points[hull.faces].mean(axis=1)[:, 0]
    quad_x = np.floor(tri_x) + 0.5
    np.testing.assert_allclose(hull.fields["Density"].data[:, 0], 1.0 + quad_x, rtol=1e-6)
    top = patches["blk-1/top"]
    assert top.points is not None and top.faces is not None
    np.testing.assert_allclose(top.points[:, 2], 0.5)
    assert top.faces.shape == (12, 3)  # 3 x 2 faces listed as face centres
    # VTK builds patches for family-specified BCs only: same faces, same adjacent-cell data
    vhull = vtk_zones(path, patches=True)["Base/blk-1/Patches/hull"]
    assert (vhull.GetNumberOfPoints(), vhull.GetNumberOfCells()) == (12, 6)
    per_quad = np.zeros(6)
    per_quad[hull.static["face_parent"].data[:, 0]] = hull.fields["Density"].data[:, 0]
    np.testing.assert_allclose(
        np.sort(_vtk.attributes(vhull.GetCellData(), None)["Density"]), np.sort(per_quad)
    )


def test_structured_2d_zone_and_rind(tmp_path: Path) -> None:
    points, quads = quad_grid(4, 3)
    path = tmp_path / "plate.cgns"
    f = new_file(path)
    b = base(f, "Plate", cell_dim=2, phys_dim=2)
    z = structured_zone(b, "surf", points[:, :2], (5, 4))
    # cell-centred solution with one rind (ghost) layer in +/- i and none in j
    core = (points[quads].mean(axis=1)[:, 0]).reshape(3, 4)
    padded = np.full((3, 6), -99.0)
    padded[:, 1:5] = core
    s = solution(z, "Sol", {"Temperature": padded}, "CellCenter")
    node(s, "Rind", "Rind_t", np.array([1, 1, 0, 0], np.int32))
    solution(z, "Nodal", {"VelocityX": points[:, 1], "VelocityY": -points[:, 0]}, shape=(4, 5))
    f.close()

    rep = check_connector(CgnsConnector(), path, tmp_path / "out")
    (zone,) = rep.description.zones
    assert (zone.kind, zone.role, zone.dim, zone.cell_types) == (
        "surface",
        "volume",
        2,
        {"quad": 12},
    )
    s = only(rep.samples["plate"])
    assert s.kind == "surface" and s.points is not None and s.faces is not None
    assert s.points.shape == (20, 3) and s.faces.shape == (24, 3)
    np.testing.assert_allclose(s.points, points)
    parent = s.static["face_parent"].data[:, 0]
    np.testing.assert_allclose(
        s.fields["Temperature"].data[:, 0], points[quads].mean(axis=1)[parent, 0], rtol=1e-6
    )
    # a 2-D base stores two velocity components: they still form one vector
    assert s.fields["Velocity"].component_names == ["x", "y"]
    np.testing.assert_allclose(s.fields["Velocity"].data[:, 1], -points[:, 0], rtol=1e-6)
    v = vtk_zones(path)["Plate/surf"]
    assert (v.GetNumberOfPoints(), v.GetNumberOfCells()) == (20, 12)
    np.testing.assert_allclose(
        _vtk.attributes(v.GetCellData(), None)["Temperature"], core.ravel(), rtol=1e-6
    )


# -- unstructured --------------------------------------------------------------------------------


def write_tets(path: Path) -> dict[str, np.ndarray]:
    points, tets = tet_grid(2, 2, 2)
    wall = boundary_faces(points, tets, 2, 0.0)
    inlet = boundary_faces(points, tets, 0, 0.0)
    extra = boundary_faces(points, tets, 1, 1.0)
    f = new_file(path)
    b = base(f)
    fam = node(b, "walls", "Family_t")
    node(fam, "FamBC", "FamilyBC_t", "BCWall")
    z = unstructured_zone(b, "fluid", points, len(tets))
    start = 1
    section(z, "tets", TETRA_4, start, len(tets), tets + 1)
    start += len(tets)
    section(z, "wall_tris", TRI_3, start, len(wall), wall + 1)
    bc(
        z, "wall", "FamilySpecified", location="FaceCenter", family="walls",
        point_range=[[start], [start + len(wall) - 1]],
    )  # fmt: skip
    start += len(wall)
    section(z, "inlet_tris", TRI_3, start, len(inlet), inlet + 1)
    # a patch given as the list of its vertices
    bc(z, "inlet", "BCInflow", point_list=np.unique(inlet) + 1, location="Vertex")
    start += len(inlet)
    section(z, "extra", TRI_3, start, len(extra), extra + 1)
    a = analytic_fields(points)
    solution(z, "FlowSolution", {"Pressure": a["p"], **split_xyz("Momentum", a["U"])})
    cc = analytic_fields(points[tets].mean(axis=1))
    solution(z, "CellSolution", {"TurbulentDistance": cc["p"]}, "CellCenter")
    f.close()
    return {"points": points, "tets": tets, "wall": wall, "inlet": inlet, "extra": extra}


def test_unstructured_tets_sections_and_bc_patches(tmp_path: Path) -> None:
    path = tmp_path / "tets.cgns"
    m = write_tets(path)
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    # sections that a BC covers exactly are not listed twice; 'extra' has no BC
    assert list(zones) == ["fluid", "fluid/wall", "fluid/inlet", "fluid/extra"]
    assert zones["fluid"].cell_types == {"tetra": 48} and zones["fluid"].n_points == 27
    assert zones["fluid/wall"].bc_type == "wall" and zones["fluid/inlet"].bc_type == "inlet"
    assert zones["fluid/extra"].bc_type is None and zones["fluid/extra"].role == "boundary"
    assert [zones[n].n_cells for n in ("fluid/wall", "fluid/inlet", "fluid/extra")] == [8, 8, 8]
    assert [z.default for z in zones.values()] == [True, False, False, False]
    assert rep.sample_ids == ["tets"]  # only the volume zone is read by default

    s = only(rep.samples["tets"])
    np.testing.assert_array_equal(s.cells, m["tets"])
    assert set(s.cell_types.tolist()) == {10}
    a = analytic_fields(m["points"])
    np.testing.assert_allclose(s.fields["Momentum"].data, a["U"], rtol=1e-6)
    cc = analytic_fields(m["points"][m["tets"]].mean(axis=1))
    np.testing.assert_allclose(s.fields["TurbulentDistance"].data[:, 0], cc["p"], rtol=1e-6)
    assert s.fields["TurbulentDistance"].location == "cell"
    assert s.metadata["boundary_conditions"]["fluid/wall"]["cgns_type"] == "BCWall"

    v = vtk_zones(path)["Base/fluid"]
    vs = vtk_sample(v)
    np.testing.assert_array_equal(vs.cells, s.cells)
    np.testing.assert_allclose(vs.points, s.points)
    np.testing.assert_allclose(vs.fields["Momentum"].data, s.fields["Momentum"].data)
    np.testing.assert_allclose(
        vs.fields["TurbulentDistance"].data, s.fields["TurbulentDistance"].data
    )

    surf = {s.zone: s for s in c.read(path, zones=["fluid/*"], fields=["Pressure"])}
    assert set(surf) == {"fluid/wall", "fluid/inlet", "fluid/extra"}
    for name, axis, value in (("wall", 2, 0.0), ("inlet", 0, 0.0), ("extra", 1, 1.0)):
        p = surf[f"fluid/{name}"]
        assert p.kind == "surface" and p.points is not None and p.faces is not None
        assert p.points.shape == (9, 3) and p.faces.shape == (8, 3)
        np.testing.assert_allclose(p.points[:, axis], value)
        # vertex data is restricted to the points of the patch
        assert list(p.fields) == ["Pressure"]
        np.testing.assert_allclose(
            p.fields["Pressure"].data[:, 0], analytic_fields(p.points)["p"], rtol=1e-6
        )
    assert surf["fluid/wall"].metadata["bc_type"] == "wall"
    assert "bc_type" not in surf["fluid/extra"].metadata
    vwall = vtk_zones(path, patches=True)["Base/fluid/Patches/wall"]
    assert vwall.GetNumberOfCells() == 8

    both = check_connector(c, path, tmp_path / "out2", zones=["fluid", "fluid/wall"])
    assert both.sample_ids == ["tets__fluid", "tets__fluid_wall"]


def write_mixed(path: Path, version: float, one_section: bool = False) -> dict[str, np.ndarray]:
    """One hexahedron and two wedges (a split hexahedron) in a MIXED section and the boundary
    quad and triangles in a second MIXED section, the way CFX writes a domain;
    ``one_section`` puts volume and boundary elements into the same section."""
    points, hexes = hex_grid(2, 1, 1)
    h = hexes[0]
    g = hexes[1]
    wedges = np.array([[g[0], g[1], g[2], g[4], g[5], g[6]], [g[0], g[2], g[3], g[4], g[6], g[7]]])
    quads = np.array([[h[0], h[3], h[2], h[1]]])
    tris = np.array([[g[0], g[2], g[1]], [g[0], g[3], g[2]]])
    volume = [[HEXA_8, *(h + 1)]] + [[PENTA_6, *(w + 1)] for w in wedges]
    surface = [[QUAD_4, *(q + 1)] for q in quads] + [[TRI_3, *(t + 1)] for t in tris]
    f = new_file(path, version)
    b = base(f)
    z = unstructured_zone(b, "Domain 1", points, 3)
    start = 1
    groups = {"Elements": volume + surface} if one_section else {"Vol": volume, "Bnd": surface}
    for name, records in groups.items():
        conn = np.concatenate([np.asarray(r) for r in records])
        offsets = np.concatenate([[0], np.cumsum([len(r) for r in records])])
        section(z, name, MIXED, start, len(records), conn, offsets if version >= 4 else None)
        start += len(records)
    bc(z, "floor", "BCWall", point_list=[4, 5, 6], location="FaceCenter")
    a = analytic_fields(points)
    solution(z, "FlowSolution", {"Pressure": a["p"]})
    solution(z, "Cells", {"Density": np.array([1.0, 2.0, 3.0])}, "CellCenter")
    f.close()
    return {"points": points, "hex": h, "wedges": wedges, "quads": quads, "tris": tris}


@pytest.mark.parametrize("version", [4.2, 3.3])
def test_mixed_section_with_and_without_start_offset(tmp_path: Path, version: float) -> None:
    path = tmp_path / "mixed.cgns"
    m = write_mixed(path, version)
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    assert list(zones) == ["Domain 1", "Domain 1/floor"]
    assert zones["Domain 1"].cell_types == {"hexahedron": 1, "wedge": 2}
    assert zones["Domain 1/floor"].cell_types == {"triangle": 2, "quad": 1}
    s = only(rep.samples["mixed"])
    assert s.cell_types.tolist() == [12, 13, 13]
    np.testing.assert_array_equal(s.cells[0], m["hex"])
    np.testing.assert_array_equal(s.cells[1:, :6], m["wedges"])
    assert (s.cells[1:, 6:] == -1).all()
    np.testing.assert_allclose(s.fields["Density"].data[:, 0], [1.0, 2.0, 3.0])

    vs = vtk_sample(vtk_zones(path)["Base/Domain 1"])
    np.testing.assert_array_equal(vs.cells, s.cells)
    np.testing.assert_array_equal(vs.cell_types, s.cell_types)
    np.testing.assert_allclose(vs.fields["Density"].data, s.fields["Density"].data)

    (floor,) = c.read(path, zones=["Domain 1/floor"])
    assert floor.points is not None and floor.faces is not None
    assert floor.faces.shape == (4, 3) and floor.points.shape == (6, 3)
    np.testing.assert_allclose(floor.points[:, 2], 0.0)
    np.testing.assert_allclose(
        floor.fields["Pressure"].data[:, 0], analytic_fields(floor.points)["p"], rtol=1e-6
    )

    # volume and boundary elements in one MIXED section are separated by dimension
    # (vtkCGNSReader takes all six as cells, so this layout is checked without it)
    single = tmp_path / "single.cgns"
    write_mixed(single, version, one_section=True)
    d = c.describe(single)
    assert {z.name: z.cell_types for z in d.zones} == {
        "Domain 1": {"hexahedron": 1, "wedge": 2},
        "Domain 1/floor": {"triangle": 2, "quad": 1},
    }
    vol = only(c.read(single))
    np.testing.assert_array_equal(vol.cells, s.cells)
    np.testing.assert_allclose(vol.fields["Density"].data[:, 0], [1.0, 2.0, 3.0])


def _reference_cell(vtk_type: int) -> np.ndarray:
    """Node coordinates of a VTK cell in VTK node order (its parametric coordinates)."""
    from vtkmodules.vtkCommonDataModel import vtkGenericCell

    cell = vtkGenericCell()
    cell.SetCellType(vtk_type)
    n = cell.GetNumberOfPoints()
    pc = cell.GetParametricCoords()
    return np.array([[pc[3 * i + k] for k in range(3)] for i in range(n)], dtype=np.float64)


@pytest.mark.parametrize(
    ("etype", "vtk_type", "name"),
    [
        (TETRA_10, 24, "quadratic_tetra"),
        (PENTA_15, 26, "quadratic_wedge"),
        (HEXA_20, 25, "quadratic_hexahedron"),
        (HEXA_27, 29, "triquadratic_hexahedron"),
    ],
)
def test_quadratic_elements_use_vtk_node_order(
    tmp_path: Path, etype: int, vtk_type: int, name: str
) -> None:
    """The CGNS node order of prisms and hexahedra differs from VTK's. The fixture stores a
    reference element in a scrambled CGNS order; both readers must return a valid VTK cell,
    i.e. the same connectivity."""
    ref = _reference_cell(vtk_type)
    n = len(ref)
    rng = np.random.default_rng(etype)
    order = rng.permutation(n)
    points = ref[order]
    path = tmp_path / f"{name}.cgns"
    f = new_file(path)
    z = unstructured_zone(base(f), "one", points, 1)
    # connectivity in CGNS order: corner nodes are shared with VTK, so the edge/face nodes are
    # found geometrically from the SIDS definition (mid-points of the listed node pairs)
    section(z, "cell", etype, 1, 1, _cgns_order(etype, points) + 1)
    f.close()

    rep = check_connector(CgnsConnector(), path, tmp_path / "out")
    assert rep.description.zones[0].cell_types == {name: 1}
    s = only([rep.first()])
    assert s.cell_types.tolist() == [vtk_type]
    np.testing.assert_allclose(s.points[s.cells[0]], ref, atol=1e-12)
    vs = vtk_sample(vtk_zones(path)["Base/one"])
    assert vs.cell_types.tolist() == [vtk_type]
    np.testing.assert_array_equal(vs.cells, s.cells)


#: SIDS node numbering (0-based corner pairs whose mid-point is the node; longer tuples are
#: face / body centres)
_SIDS_EXTRA: dict[int, list[tuple[int, ...]]] = {
    TETRA_10: [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    PENTA_15: [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (5, 3)],
    HEXA_20: [
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
    ],
}  # fmt: skip
_SIDS_EXTRA[HEXA_27] = _SIDS_EXTRA[HEXA_20] + [
    (0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 3, 7, 4), (4, 5, 6, 7),
    (0, 1, 2, 3, 4, 5, 6, 7),
]  # fmt: skip
_CORNERS = {TETRA_10: 4, PENTA_15: 6, HEXA_20: 8, HEXA_27: 8}
_LINEAR_VTK = {TETRA_10: 10, PENTA_15: 13, HEXA_20: 12, HEXA_27: 12}


def _cgns_order(etype: int, points: np.ndarray) -> np.ndarray:
    """Point ids of the single reference element listed in SIDS node order."""
    corners_ref = _reference_cell(_LINEAR_VTK[etype])[: _CORNERS[etype]]

    def find(target: np.ndarray) -> int:
        d = np.abs(points - target).sum(axis=1)
        assert d.min() < 1e-9
        return int(d.argmin())

    ids = [find(p) for p in corners_ref]
    for combo in _SIDS_EXTRA[etype]:
        ids.append(find(corners_ref[list(combo)].mean(axis=0)))
    assert len(set(ids)) == len(points)
    return np.array(ids)


def hex_faces(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique quad faces of hexahedra and, per cell, the 6 (1-based, signed) face numbers."""
    local = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    faces: list[tuple[int, ...]] = []
    index: dict[frozenset[int], int] = {}
    cell_faces = np.zeros((len(cells), 6), dtype=np.int64)
    for ci, cell in enumerate(cells):
        for fi, loc in enumerate(local):
            nodes = tuple(int(cell[i]) for i in loc)
            key = frozenset(nodes)
            if key in index:
                cell_faces[ci, fi] = -(index[key] + 1)
            else:
                index[key] = len(faces)
                faces.append(nodes)
                cell_faces[ci, fi] = len(faces)
    return np.array(faces), cell_faces


@pytest.mark.parametrize("version", [4.2, 3.3])
def test_ngon_nface_polyhedra(tmp_path: Path, version: float) -> None:
    points, cells = hex_grid(3, 2, 2)
    faces, cell_faces = hex_faces(cells)
    nf, nc = len(faces), len(cells)
    path = tmp_path / "poly.cgns"
    f = new_file(path, version)
    z = unstructured_zone(base(f), "poly", points, nc)
    if version >= 4:
        section(z, "faces", NGON_N, 1, nf, faces + 1, np.arange(nf + 1) * 4)
        signed = np.sign(cell_faces) * (np.abs(cell_faces) + 0)
        section(z, "cells", NFACE_N, nf + 1, nc, signed, np.arange(nc + 1) * 6)
    else:  # CGNS 3: every record starts with its item count
        section(z, "faces", NGON_N, 1, nf, np.column_stack([np.full(nf, 4), faces + 1]))
        section(z, "cells", NFACE_N, nf + 1, nc, np.column_stack([np.full(nc, 6), cell_faces]))
    floor = np.nonzero(np.isclose(points[faces][:, :, 2], 0.0).all(axis=1))[0]
    bc(z, "floor", "BCWallInviscid", point_list=floor + 1, location="FaceCenter")
    centres = points[cells].mean(axis=1)
    solution(z, "Sol", {"Pressure": analytic_fields(centres)["p"]}, "CellCenter")
    f.close()

    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    # the NGON section holds interior faces too, so it is not offered as a boundary
    assert list(zones) == ["poly", "poly/floor"]
    assert zones["poly"].cell_types == {"polyhedron": 12}
    assert zones["poly/floor"].cell_types == {"polygon": 6}
    s = only(rep.samples["poly"])
    assert set(s.cell_types.tolist()) == {42} and s.metadata["polyhedral"] is True
    assert row_sets(s.cells) == row_sets(cells)  # unique point ids of every polyhedron
    np.testing.assert_allclose(
        s.fields["Pressure"].data[:, 0], analytic_fields(centres)["p"], rtol=1e-6
    )
    v = vtk_zones(path)["Base/poly"]
    assert (v.GetNumberOfPoints(), v.GetNumberOfCells()) == (36, 12)
    vs = vtk_sample(v)
    assert row_sets(vs.cells) == row_sets(s.cells)
    np.testing.assert_allclose(vs.fields["Pressure"].data, s.fields["Pressure"].data)

    (floor_s,) = c.read(path, zones=["poly/floor"])
    assert floor_s.points is not None and floor_s.faces is not None
    assert floor_s.faces.shape == (12, 3) and floor_s.points.shape == (12, 3)
    np.testing.assert_allclose(floor_s.points[:, 2], 0.0)
    assert floor_s.metadata["bc_type"] == "wall"


# -- transient -----------------------------------------------------------------------------------

TIMES = [0.0, 0.5, 1.0, 2.5]


def write_transient(path: Path, *, moving: bool = False, pointers: bool = True) -> dict[str, Any]:
    points, cells = hex_grid(2, 2, 1)
    f = new_file(path)
    b = base(f)
    node(b, "SimulationType", "SimulationType_t", "TimeAccurate")
    it = node(b, "TimeIterValues", "BaseIterativeData_t", np.array([len(TIMES)], np.int32))
    node(it, "TimeValues", "DataArray_t", np.array(TIMES))
    node(it, "IterationValues", "DataArray_t", np.arange(1, len(TIMES) + 1, dtype=np.int32) * 10)
    z = unstructured_zone(b, "zone", points, len(cells))
    section(z, "hexes", HEXA_8, 1, len(cells), cells + 1)
    names, grids = [], []
    for i, t in enumerate(TIMES):
        a = analytic_fields(points, t)
        name = f"FlowSolution#{i + 1:04d}"
        names.append(name)
        solution(z, name, {"Pressure": a["p"], **split_xyz("Velocity", a["U"])})
        if moving:
            grid = "GridCoordinates" if i == 0 else f"GridCoordinates#{i + 1:04d}"
            grids.append(grid)
            if i:
                coordinates(z, points + [0.1 * t, 0.0, 0.0], (len(points),), grid)
    if pointers:
        solution(z, "WallDistance", {"TurbulentDistance": points[:, 2]})
        zi = node(z, "ZoneIterativeData", "ZoneIterativeData_t")
        node(zi, "FlowSolutionPointers", "DataArray_t", chars(names))
        if moving:
            node(zi, "GridCoordinatesPointers", "DataArray_t", chars(grids))
    f.close()
    return {"points": points, "cells": cells}


def test_transient_flow_solution_pointers(tmp_path: Path) -> None:
    path = tmp_path / "unsteady.cgns"
    m = write_transient(path)
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    d = rep.description
    assert d.temporal == "transient" and d.time.values == TIMES and d.time.quantity == "time"
    assert d.physics["simulation_type"] == "TimeAccurate"
    assert d.metadata["iterations"] == [10, 20, 30, 40]
    fields = {f.name: f.time_dependent for f in d.fields}
    assert fields == {"Pressure": True, "Velocity": True, "TurbulentDistance": False}
    steps = rep.samples["unsteady"]
    assert [s.time for s in steps] == TIMES and [s.time_index for s in steps] == [0, 1, 2, 3]
    assert steps[0].points is not None and steps[0].cells is not None
    assert all(s.points is None and s.cells is None for s in steps[1:])  # static mesh: once
    for s, t in zip(steps, TIMES, strict=True):
        a = analytic_fields(m["points"], t)
        np.testing.assert_allclose(s.fields["Velocity"].data, a["U"], rtol=1e-6)
        np.testing.assert_allclose(s.fields["Pressure"].data[:, 0], a["p"], rtol=1e-6)
        assert s.metadata["time_units"] == "s"
    # the solution no step points at is time-independent
    assert list(steps[0].static) == ["TurbulentDistance"] and not steps[1].static
    np.testing.assert_allclose(steps[0].static["TurbulentDistance"].data[:, 0], m["points"][:, 2])

    last = list(c.read(path, times="last", fields=["Pressure"]))
    assert len(last) == 1 and last[0].time == 2.5 and list(last[0].fields) == ["Pressure"]
    assert last[0].points is not None  # the first step that is read carries the mesh
    picked = [s.time for s in c.read(path, times=[1, 3])]
    assert picked == [0.5, 2.5]
    assert [s.time for s in c.read(path, times=[0.6])] == [0.5]

    v = vtk_zones(path, time=1.0)
    assert v["__times__"] == TIMES
    vp = _vtk.attributes(v["Base/zone"].GetPointData(), None)
    np.testing.assert_allclose(vp["Velocity"], steps[2].fields["Velocity"].data, rtol=1e-6)

    # the converted dataset holds the four steps on one mesh
    result = rep.results["zarr"]
    assert result.n_steps == {"unsteady": 4}
    store = open_store(result.out / "samples/unsteady/mesh.zarr")
    assert store["points"].shape == (18, 3)
    assert store["fields/Velocity"].shape == (4, 18, 3)
    np.testing.assert_allclose(
        store["fields/Velocity"][3], analytic_fields(m["points"], 2.5)["U"], rtol=1e-6
    )
    with pytest.raises(SelectionError):
        list(c.read(path, times=9))


def test_transient_moving_grid_and_solutions_without_pointers(tmp_path: Path) -> None:
    path = tmp_path / "moving.cgns"
    m = write_transient(path, moving=True)
    rep = check_connector(CgnsConnector(), path, tmp_path / "out")
    assert rep.description.metadata["moving_grid"] is True
    steps = rep.samples["moving"]
    for s, t in zip(steps, TIMES, strict=True):
        assert s.points is not None  # GridCoordinatesPointers: new points every step
        np.testing.assert_allclose(s.points, m["points"] + [0.1 * t, 0.0, 0.0])
    assert steps[0].cells is not None and steps[1].cells is None

    # no ZoneIterativeData: as many solutions as steps are taken in order
    bare = tmp_path / "bare.cgns"
    write_transient(bare, pointers=False)
    rep2 = check_connector(CgnsConnector(), bare, tmp_path / "out2")
    steps2 = rep2.samples["bare"]
    assert [s.time for s in steps2] == TIMES
    np.testing.assert_allclose(
        steps2[3].fields["Pressure"].data[:, 0], analytic_fields(m["points"], 2.5)["p"], rtol=1e-6
    )


def write_steady_step(path: Path, t: float, stamp: bool = True) -> np.ndarray:
    points, cells = hex_grid(2, 1, 1)
    f = new_file(path)
    b = base(f)
    if stamp:
        it = node(b, "BaseIterativeData", "BaseIterativeData_t", np.array([1], np.int32))
        node(it, "TimeValues", "DataArray_t", np.array([t]))
    z = unstructured_zone(b, "zone", points, len(cells))
    section(z, "hexes", HEXA_8, 1, len(cells), cells + 1)
    solution(z, "FlowSolution", {"Pressure": analytic_fields(points, t)["p"]})
    f.close()
    return points


def test_one_file_per_step_series(tmp_path: Path) -> None:
    series = tmp_path / "run"
    series.mkdir()
    times = [0.1, 0.2, 0.4]
    points = np.zeros((0, 3))
    for i, t in enumerate(times):
        points = write_steady_step(series / f"sol_{i + 1:04d}.cgns", t)
    c = CgnsConnector()
    det = c.detect(series)
    assert det is not None and det.format == "cgns-series"
    assert fc.detect(series)[0].connector_id == "cgns"
    rep = check_connector(c, series, tmp_path / "out", min_confidence=0.5)
    assert rep.description.time.values == times and rep.description.temporal == "transient"
    assert rep.description.metadata["series_files"] == [f"sol_{i:04d}.cgns" for i in (1, 2, 3)]
    steps = rep.samples["run"]
    assert [s.time for s in steps] == times
    assert steps[0].points is not None and steps[1].points is None and steps[2].cells is None
    np.testing.assert_allclose(
        steps[2].fields["Pressure"].data[:, 0], analytic_fields(points, 0.4)["p"], rtol=1e-6
    )
    # a member file stands for the series; series=False reads just that file
    member = series / "sol_0002.cgns"
    by_member = list(c.read(member, times="last"))
    assert [s.time for s in by_member] == [0.4] and by_member[0].id == "sol"
    (alone,) = c.read(member, options={"series": False})
    assert alone.time is None and alone.metadata["time_value"] == 0.2 and alone.id == "sol_0002"
    assert c.describe(member, options={"series": False}).temporal == "steady"

    # without TimeValues the file numbers are the axis
    plain = tmp_path / "plain"
    plain.mkdir()
    for n in (5, 10, 20):
        write_steady_step(plain / f"flow-{n}.cgns", float(n), stamp=False)
    d = c.describe(plain)
    assert d.time.values == [5.0, 10.0, 20.0] and d.time.quantity == "iteration"
    assert [s.time for s in c.read(plain)] == [5.0, 10.0, 20.0]

    mixed = tmp_path / "unrelated"
    mixed.mkdir()
    write_steady_step(mixed / "wing.cgns", 0.0)
    write_steady_step(mixed / "body.cgns", 0.0)
    assert c.detect(mixed) is None
    with pytest.raises(UnsupportedSourceError, match="not one numbered series"):
        c.describe(mixed)


# -- structure of the file -----------------------------------------------------------------------


def test_links_multiple_bases_and_raw_names(tmp_path: Path) -> None:
    points, cells = hex_grid(2, 1, 1)
    grid_file = tmp_path / "grid.cgns"
    f = new_file(grid_file)
    z = unstructured_zone(base(f, "Mesh"), "zone", points, len(cells))
    section(z, "hexes", HEXA_8, 1, len(cells), cells + 1)
    f.close()

    path = tmp_path / "linked.cgns"
    f = new_file(path)
    b = base(f, "Flow")
    z = unstructured_zone(b, "zone", points, len(cells))
    # the element section lives in another file (external link), as mesh/solution splits do
    link = node(z, "hexes", "Elements_t")
    link.attrs.modify("type", np.bytes_("LK"))
    link.create_dataset(" path", data=np.frombuffer(b"/Mesh/zone/hexes\x00", dtype=np.int8))
    link.create_dataset(" file", data=np.frombuffer(b"grid.cgns\x00", dtype=np.int8))
    link[" link"] = h5py.ExternalLink("grid.cgns", "/Mesh/zone/hexes")
    a = analytic_fields(points)
    solution(z, "FlowSolution", {"Pressure": a["p"], **split_xyz("Velocity", a["U"])})
    # a second base with a zone of the same name whose coordinates are a link inside the file
    b2 = base(f, "Copy")
    z2 = node(b2, "zone", "Zone_t", np.array([[len(points)], [len(cells)], [0]], np.int32))
    node(z2, "ZoneType", "ZoneType_t", "Unstructured")
    soft = node(z2, "GridCoordinates", "GridCoordinates_t")
    soft.attrs.modify("type", np.bytes_("LK"))
    target = b"/Flow/zone/GridCoordinates\x00"
    soft.create_dataset(" path", data=np.frombuffer(target, dtype=np.int8))
    soft[" link"] = h5py.SoftLink("/Flow/zone/GridCoordinates")
    section(z2, "hexes", HEXA_8, 1, len(cells), cells + 1)
    f.close()

    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    assert [z.name for z in rep.description.zones] == ["Flow/zone", "Copy/zone"]
    assert rep.sample_ids == ["linked__Flow_zone", "linked__Copy_zone"]
    for sid in rep.sample_ids:
        s = only([rep.first(sid)])
        np.testing.assert_array_equal(s.cells, cells)
        np.testing.assert_allclose(s.points, points)
    assert list(rep.first("linked__Flow_zone").fields) == ["Pressure", "Velocity"]
    assert not rep.first("linked__Copy_zone").fields
    assert rep.description.fields[0].zones == ["Flow/zone"]
    v = vtk_zones(path)
    assert v["Flow/zone"].GetNumberOfCells() == 2 and v["Copy/zone"].GetNumberOfCells() == 2

    (raw,) = c.read(path, zones=["Flow/zone"], options={"raw_names": True})
    assert list(raw.fields) == ["Pressure", "VelocityX", "VelocityY", "VelocityZ"]
    np.testing.assert_allclose(raw.fields["VelocityY"].data[:, 0], a["U"][:, 1], rtol=1e-6)

    # a link whose target file is gone is reported, not fatal for describe
    grid_file.unlink()
    d = c.describe(path)
    assert any("missing" in w for w in d.warnings)
    assert {z.name: z.n_cells for z in d.zones} == {"Flow/zone": 0, "Copy/zone": 2}


def test_surface_base_is_not_read_by_default(tmp_path: Path) -> None:
    """Solvers that write the boundaries as zones of a second, 2-D base (as VTK's own sample
    files do): those zones are offered, the volume base is what a plain read returns."""
    points, tets = tet_grid(1, 1, 1)
    wall = boundary_faces(points, tets, 2, 0.0)
    path = tmp_path / "two_bases.cgns"
    f = new_file(path)
    z = unstructured_zone(base(f, "Volume"), "interior", points, len(tets))
    section(z, "tets", TETRA_4, 1, len(tets), tets + 1)
    zs = unstructured_zone(base(f, "Surface", cell_dim=2), "wall", points, len(wall))
    section(zs, "tris", TRI_3, 1, len(wall), wall + 1)
    solution(zs, "Sol", {"SkinFrictionX": np.arange(len(wall), dtype=float)}, "CellCenter")
    f.close()
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    assert [(z.kind, z.role, z.default) for z in zones.values()] == [
        ("mesh", "volume", True),
        ("surface", "boundary", False),
    ]
    assert rep.sample_ids == ["two_bases"]
    (s,) = c.read(path, zones=["Surface/wall"])
    assert s.kind == "surface" and s.points is not None and s.faces is not None
    assert s.points.shape == (4, 3) and s.faces.shape == (2, 3)  # 8 zone points, 4 used
    np.testing.assert_allclose(s.fields["SkinFrictionX"].data[:, 0], [0.0, 1.0])


def test_detect_selection_and_cli_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "tets.cgns"
    write_tets(path)
    c = CgnsConnector()
    # CGNS content in a generically named HDF5 file is recognised by its node labels
    as_h5 = tmp_path / "result.h5"
    as_h5.write_bytes(path.read_bytes())
    det = c.detect(as_h5)
    assert det is not None and det.confidence == 0.9 and det.format == "cgns-hdf5"
    assert fc.detect(as_h5)[0].connector_id == "cgns"
    assert fc.detect(path)[0].connector_id == "cgns"
    other = tmp_path / "plain.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("x", data=np.arange(3))
    assert c.detect(other) is None
    renamed = tmp_path / "plain.cgns"
    renamed.write_bytes(other.read_bytes())
    assert c.detect(renamed) is None
    with pytest.raises(UnsupportedSourceError, match="without a CGNSBase_t"):
        c.describe(renamed)
    assert c.detect(tmp_path / "missing.cgns") is None and c.detect(tmp_path) is None

    with pytest.raises(SelectionError, match="fluid/wall"):
        list(c.read(path, zones=["nope"]))
    with pytest.raises(ConnectorError, match="unknown CGNS engine"):
        c.describe(path, options={"engine": "adf"})
    assert [s.zone for s in c.read(path, zones=["!fluid/*"])] == ["fluid"]
    assert (
        bc_type_id("BCOutflowSubsonic") == "outlet" and bc_type_id("BCExtrapolate") == "extrapolate"
    )
    assert bc_type_id("FamilySpecified") == "unknown"


def test_vtk_engine_reads_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "tets.cgns"
    m = write_tets(path)
    c = CgnsConnector()
    rep = check_connector(c, path, tmp_path / "out", options={"engine": "vtk"})
    d = rep.description
    assert d.metadata["engine"] == "vtk" and any("vtkCGNSReader" in w for w in d.warnings)
    zones = {z.name: (z.kind, z.role, z.default) for z in d.zones}
    assert zones["fluid"] == ("mesh", "volume", True)
    assert zones["fluid/wall"] == ("surface", "boundary", False)
    s = only(rep.samples["tets"])
    np.testing.assert_array_equal(s.cells, m["tets"])
    np.testing.assert_allclose(
        s.fields["Momentum"].data, analytic_fields(m["points"])["U"], rtol=1e-6
    )


# -- malformed input -----------------------------------------------------------------------------


def test_malformed_truncated_and_adf_files(tmp_path: Path) -> None:
    c = CgnsConnector()
    good = tmp_path / "tets.cgns"
    write_tets(good)
    raw = good.read_bytes()

    cut = tmp_path / "cut.cgns"
    cut.write_bytes(raw[: len(raw) // 3])
    det = c.detect(cut)
    assert det is not None and det.confidence == 0.5  # HDF5 header, cannot be opened
    with pytest.raises(ConnectorError, match="cannot be opened as HDF5") as e:
        c.describe(cut)
    assert e.value.hint and "cgnscheck" in e.value.hint
    with pytest.raises(ConnectorError):
        list(c.read(cut))

    garbage = tmp_path / "garbage.cgns"
    garbage.write_bytes(b"this is not a CGNS file\n" * 10)
    assert c.detect(garbage) is None
    with pytest.raises(ConnectorError, match="neither an HDF5 nor an ADF header"):
        c.describe(garbage)

    # connectivity shorter than the element range says
    short = tmp_path / "short.cgns"
    short.write_bytes(raw)
    with edit(short) as f:
        sec = f["Base/fluid/tets"]
        conn = sec["ElementConnectivity/ data"][()]
        del sec["ElementConnectivity/ data"]
        sec["ElementConnectivity"].create_dataset(" data", data=conn[:-7])
    with pytest.raises(ConnectorError, match="connectivity entries for 48 TETRA_4") as e:
        list(c.read(short))
    assert e.value.hint

    # node ids beyond the vertex count
    wild = tmp_path / "wild.cgns"
    wild.write_bytes(raw)
    with edit(wild) as f:
        f["Base/fluid/tets/ElementConnectivity/ data"][5] = 999
    with pytest.raises(ConnectorError, match=r"node ids outside 1\.\.27"):
        list(c.read(wild))

    # a zone without ZoneType_t / without coordinates
    for victim, message in (("ZoneType", "no valid ZoneType_t"), ("GridCoordinates", "no Grid")):
        broken = tmp_path / f"no_{victim}.cgns"
        broken.write_bytes(raw)
        with edit(broken) as f:
            del f[f"Base/fluid/{victim}"]
        with pytest.raises(ConnectorError, match=message):
            list(c.read(broken))

    # cubic elements are named and the VTK engine is suggested
    cubic = tmp_path / "cubic.cgns"
    f = new_file(cubic)
    z = unstructured_zone(base(f), "ho", np.zeros((20, 3)), 1)
    section(z, "cell", 30, 1, 1, np.arange(1, 21))  # TETRA_20
    f.close()
    with pytest.raises(UnsupportedSourceError, match="TETRA_20") as e2:
        c.describe(cubic)
    assert e2.value.hint and "engine" in e2.value.hint

    # ADF (pre-HDF5) container: recognised by its header; the h5py engine says how to convert
    adf = tmp_path / "old.cgns"
    adf.write_bytes(b"\xc0\xa8\xa3\xa9ADF Database Version A02011>AdF0" + b"\x00" * 600)
    assert file_format(adf) == "adf"
    det = c.detect(adf)
    assert det is not None and det.format == "cgns-adf" and det.confidence == 0.9
    with pytest.raises(ConnectorError, match="ADF-based") as e3:
        c.describe(adf, options={"engine": "h5py"})
    assert e3.value.hint and "cgnsconvert -h" in e3.value.hint
    # default engine: handed to VTK, which rejects this stub cleanly
    with pytest.raises(ConnectorError, match="VTK"):
        list(c.read(adf))


# -- real files ----------------------------------------------------------------------------------

VTK_DATA = "https://www.vtk.org/files/ExternalData/SHA512/"
PYVISTA = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data/cgns/"
#: VTK testing data (Testing/Data/<name>.sha512 in the VTK repository), CGNS/HDF5
VTK_FILES = {
    "test_node_and_cell.cgns": "a4ffcf82da6071872a2d71d83c1dd60ccd54a3c9134fdda4390616d248c4d55b"
    "56d093ada4ef348a9cd0ea55fc5487d7590f0548596dd09ed532a150aba18fcd",
    "EngineSector.cgns": "1f69b4276708003a23d7d510bd4b7ce10e055df43e843f14f9af7f8eea44ee940b7f1"
    "2bef7e5dcf0e094f46235b66a896555461a77cbb9daada86d3280187e78",
    "MixedElementNodes.cgns": "c3987f5f4f32b2aad4d35776538f4919cb04d57b18af2d3acc6fe5fb2a993f65"
    "26e91a303d11848f841c743a77392365548be80ad431834b0b54328f5b244946",
    "Example_ngon_2d_base.cgns": "2175784b4aa63c41158fd94dd35216f3949d8ed0a4f3a4f29ccbeb5595826"
    "5bda86cd312d10178197d34a9676609e2340933eafc838cfd315554042e20e920f7",
    "HO_hexa_elem.cgns": "f202df34f6a2b1450b4f7907246c0a572033caa6fc0212404b0fb0fac3c479edd3bd8"
    "4664a31034556ebed2fd78329f9dbb1900d4c3c6832e0d4396d6f81a3b0",
}


def _vtk_file(name: str) -> Path:
    return fetch_sample(VTK_DATA + VTK_FILES[name], name)


def _compare_with_vtk(sample: Any, ds: Any) -> None:
    """A volume sample of the h5py reader against the VTK dataset of the same zone."""
    vs = vtk_sample(ds)
    assert sample.cells is not None and vs.cells is not None
    assert (len(sample.points), len(sample.cells)) == (len(vs.points), len(vs.cells))
    np.testing.assert_allclose(sample.points, vs.points, rtol=1e-6)
    np.testing.assert_array_equal(sample.cell_types, vs.cell_types)
    poly = sample.cell_types == 42
    np.testing.assert_array_equal(sample.cells[~poly], vs.cells[~poly][:, : sample.cells.shape[1]])
    assert row_sets(sample.cells[poly]) == row_sets(vs.cells[poly])
    assert set(sample.fields) == set(vs.fields)
    for name, f in sample.fields.items():
        np.testing.assert_allclose(f.data, vs.fields[name].data, rtol=1e-6, err_msg=name)


@pytest.mark.network
def test_real_structured_file_matches_vtk(tmp_path: Path) -> None:
    """``test_node_and_cell.cgns`` (VTK testing data, CGNS 3.3 HDF5): structured zone with a
    vertex and a cell-centred solution of the same name."""
    path = _vtk_file("test_node_and_cell.cgns")
    rep = check_connector(CgnsConnector(), path, tmp_path / "out", min_confidence=0.95)
    (zone,) = rep.description.zones
    assert (zone.name, zone.shape, zone.n_cells) == ("Zone_1", [20, 20, 10], 3249)
    assert rep.description.metadata["cgns_version"] == pytest.approx(3.3)
    s = only([rep.first()])
    assert s.metadata["family"] == "Fluid"
    v = vtk_zones(path)["TestBase#1/Zone_1"]
    np.testing.assert_allclose(s.points, _vtk.points(v))
    np.testing.assert_allclose(
        s.fields["Pressure"].data[:, 0], _vtk.attributes(v.GetPointData(), None)["Pressure"]
    )
    np.testing.assert_allclose(
        s.fields["Pressure__cell"].data[:, 0], _vtk.attributes(v.GetCellData(), None)["Pressure"]
    )


@pytest.mark.network
def test_real_polyhedral_files_match_vtk(tmp_path: Path) -> None:
    """``EngineSector.cgns`` and ``MixedElementNodes.cgns`` (VTK testing data, CGNS 4.2 HDF5,
    CONVERGE-style trees): NGON_n/NFACE_n cells, HEXA_8 next to polyhedra, face-list BCs."""
    c = CgnsConnector()
    path = _vtk_file("EngineSector.cgns")
    rep = check_connector(c, path, tmp_path / "engine")
    d = rep.description
    assert d.units["length"] == "m" and d.units["mass"] == "kg"
    zones = {z.name: z for z in d.zones}
    assert zones["STREAM_0"].cell_types == {"polyhedron": 1956}
    assert zones["STREAM_0"].n_points == 7556
    assert zones["STREAM_0/BOUNDARIES"].n_cells == 6209
    assert zones["STREAM_0/BOUNDARIES"].bc_type == "dirichlet"
    fields = {f.name: f.components for f in d.fields}
    assert fields["VELOCITY"] == 3 and fields["PRESSURE"] == 1 and len(fields) == 19
    s = only([rep.first()])
    _compare_with_vtk(s, vtk_zones(path)["MESH_DATA/STREAM_0"])
    (bnd,) = c.read(path, zones=["STREAM_0/BOUNDARIES"])
    assert bnd.kind == "surface" and bnd.faces is not None and len(bnd.faces) >= 6209
    vb = vtk_zones(path, patches=True)["MESH_DATA/STREAM_0/Patches/BOUNDARIES"]
    assert vb.GetNumberOfCells() == 6209

    path = _vtk_file("MixedElementNodes.cgns")
    rep = check_connector(c, path, tmp_path / "mixed")
    zones = {z.name: z for z in rep.description.zones}
    assert zones["ZONE"].cell_types == {"hexahedron": 4000, "polyhedron": 4000}
    assert zones["ZONE/TRI_BOUNDARY"].cell_types == {"triangle": 38}
    assert zones["ZONE/MIXED_BOUNDARY"].n_cells == 10
    assert zones["ZONE/SURFACE_TRIANGLES"].n_cells == 3836
    _compare_with_vtk(rep.first(), vtk_zones(path)["BASE/ZONE"])
    vp = vtk_zones(path, patches=True)
    assert vp["BASE/ZONE/Patches/TRI_BOUNDARY"].GetNumberOfCells() == 38
    assert vp["BASE/ZONE/Patches/MIXED_BOUNDARY"].GetNumberOfCells() == 10


@pytest.mark.network
def test_real_two_base_file_matches_vtk(tmp_path: Path) -> None:
    """``Example_ngon_2d_base.cgns`` (VTK testing data, CGNS 4.1 HDF5): a polyhedral volume
    base plus a 2-D base whose zones are the boundaries as NGON_n polygons."""
    c = CgnsConnector()
    path = _vtk_file("Example_ngon_2d_base.cgns")
    rep = check_connector(c, path, tmp_path / "out")
    zones = {z.name: z for z in rep.description.zones}
    assert len(zones) == 8
    interior = zones["Base_Volume_elements/Zone_Interior"]
    assert interior.default and interior.cell_types == {"polyhedron": 4096}
    assert not zones["Base_Surface_elements/Zone_INLET"].default
    assert rep.sample_ids == ["Example_ngon_2d_base"]
    v = vtk_zones(path)
    _compare_with_vtk(rep.first(), v["Base_Volume_elements/Zone_Interior"])
    (inlet,) = c.read(path, zones=["Base_Surface_elements/Zone_INLET"])
    vin = v["Base_Surface_elements/Zone_INLET"]
    assert inlet.faces is not None and inlet.points is not None
    assert vin.GetNumberOfCells() == 64 and len(inlet.faces) == 128
    parent = inlet.static["face_parent"].data[:, 0]
    np.testing.assert_allclose(
        inlet.fields["ApproxCurvDeviation"].data[:, 0],
        _vtk.attributes(vin.GetCellData(), None)["ApproxCurvDeviation"][parent],
        rtol=1e-6,
    )


@pytest.mark.network
def test_real_high_order_file_needs_the_vtk_engine(tmp_path: Path) -> None:
    """``HO_hexa_elem.cgns`` (VTK testing data): HEXA_64 / HEXA_125 are not mapped natively."""
    c = CgnsConnector()
    path = _vtk_file("HO_hexa_elem.cgns")
    with pytest.raises(UnsupportedSourceError, match="HEXA_125"):
        c.describe(path)
    rep = check_connector(c, path, tmp_path / "out", options={"engine": "vtk"})
    assert {z.name: (z.n_points, z.n_cells) for z in rep.description.zones} == {
        "HEXA125": (125, 1),
        "HEXA64": (64, 1),
    }


@pytest.mark.network
def test_real_adf_files_fall_back_to_vtk(tmp_path: Path) -> None:
    """``sqnz_s.adf.cgns`` and ``multi.cgns`` (pyvista/vtk-data): ADF containers, read through
    ``vtkCGNSReader`` automatically."""
    c = CgnsConnector()
    path = fetch_sample(PYVISTA + "sqnz_s.adf.cgns")
    det = c.detect(path)
    assert det is not None and det.format == "cgns-adf"
    rep = check_connector(c, path, tmp_path / "sqnz")
    d = rep.description
    assert d.format == "cgns-adf" and d.metadata["engine"] == "vtk"
    volumes = [z for z in d.zones if z.default]
    assert len(volumes) == 12 and sum(z.n_cells or 0 for z in volumes) == 11264
    assert volumes[0].name == "dom1_1_1_1" and volumes[0].shape == [15, 9, 9]
    patches = [z for z in d.zones if not z.default]
    assert patches and all(z.role == "boundary" and z.kind == "surface" for z in patches)
    assert {f.name: (f.location, f.components) for f in d.fields} == {
        "Density": ("cell", 1),
        "Momentum": ("cell", 3),
        "EnergyStagnationDensity": ("cell", 1),
    }
    assert len(rep.samples) == 12
    s = only([rep.first()])
    assert s.metadata["structured_shape"] == [15, 9, 9] and s.fields["Momentum"].shape == (896, 3)
    with pytest.raises(ConnectorError, match="ADF-based"):
        c.describe(path, options={"engine": "h5py"})

    path = fetch_sample(PYVISTA + "multi.cgns")
    d = c.describe(path)
    assert [z.name for z in d.zones if z.default] == [f"Zone   {i}" for i in (1, 2, 3, 4)]
    assert sum(z.n_cells or 0 for z in d.zones if z.default) == 28168
    (s,) = c.read(path, zones=["Zone   1"], fields=["Density"])
    assert s.n_cells == 2928 and list(s.fields) == ["Density"]
