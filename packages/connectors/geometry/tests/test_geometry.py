# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_geometry import connector as geometry
from frontier_connector_geometry.connector import GeometryConnector, merge_vertices, surface_stats

import frontier_connectors as fc
from frontier_connectors.testing import check_connector, fetch_sample, open_store

C = GeometryConnector()
# typed Any: samples expose optional arrays and the assertions fail loudly on None anyway
close: Any = np.testing.assert_allclose
equal: Any = np.testing.assert_array_equal


def _cube_polydata(size: float = 1.0) -> Any:
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.vtkFiltersSources import vtkCubeSource

    src = vtkCubeSource()
    src.SetXLength(size)
    src.SetYLength(size)
    src.SetZLength(size)
    src.SetCenter(size / 2, size / 2, size / 2)
    tri = vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def _write_stl(poly: Any, path: Path, binary: bool) -> Path:
    from vtkmodules.vtkIOGeometry import vtkSTLWriter

    w = vtkSTLWriter()
    w.SetFileName(str(path))
    w.SetInputData(poly)
    if binary:
        w.SetFileTypeToBinary()
    else:
        w.SetFileTypeToASCII()
    assert w.Write() == 1
    return path


def _vtk_reference(path: Path) -> dict[str, float]:
    """Point/triangle counts, area and volume as VTK's own reader sees the file."""
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkMassProperties, vtkTriangleFilter
    from vtkmodules.vtkIOGeometry import vtkOBJReader, vtkSTLReader
    from vtkmodules.vtkIOPLY import vtkPLYReader

    reader: Any = {".stl": vtkSTLReader, ".obj": vtkOBJReader, ".ply": vtkPLYReader}[
        path.suffix.lower()
    ]()
    reader.SetFileName(str(path))
    tri = vtkTriangleFilter()
    tri.SetInputConnection(reader.GetOutputPort())
    clean = vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.SetTolerance(0.0)
    clean.PointMergingOn()
    mass = vtkMassProperties()
    mass.SetInputConnection(clean.GetOutputPort())
    mass.Update()
    out = clean.GetOutput()
    return {
        "points": out.GetNumberOfPoints(),
        "triangles": out.GetNumberOfPolys(),
        "area": mass.GetSurfaceArea(),
        "volume": mass.GetVolume(),
    }


@pytest.mark.parametrize("binary", [True, False])
def test_cube_stl_written_by_vtk(tmp_path: Path, binary: bool) -> None:
    path = _write_stl(_cube_polydata(), tmp_path / "cube.stl", binary)
    assert geometry.stl_is_binary(path) is binary
    det = C.detect(path)
    assert det is not None and det.confidence == 0.9
    assert det.format == ("stl-binary" if binary else "stl-ascii")
    rep = check_connector(C, path, tmp_path / "out", options={"units": "mm"})
    d = rep.description
    assert d.solver == "geometry" and d.domain == "other" and d.temporal == "steady"
    assert d.units == {"length": "mm"} and not d.fields and not d.warnings
    z = d.zones[0]
    assert (z.name, z.kind, z.role, z.n_points, z.n_cells) == (
        "surface",
        "surface",
        "geometry",
        8,
        12,
    )
    assert z.bounds == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0] and z.cell_types == {"triangle": 12}
    st = d.metadata["geometry"]["surface"]
    assert st["watertight"] and st["consistently_oriented"] and st["normals"] == "outward"
    assert st["area"] == pytest.approx(6.0) and st["volume"] == pytest.approx(1.0)
    assert st["boundary_edges"] == 0 and st["nonmanifold_edges"] == 0
    s = rep.first()
    assert s.id == "cube" and s.kind == "surface" and s.time is None and not s.fields
    assert s.points is not None and s.points.shape == (8, 3)  # 36 STL vertices merged
    assert s.faces is not None and s.faces.shape == (12, 3) and int(s.faces.max()) == 7
    assert s.metadata["units"] == {"length": "mm"} and s.metadata["geometry"]["volume"] == 1.0
    ref = _vtk_reference(path)
    assert (ref["points"], ref["triangles"]) == (8, 12)
    assert st["area"] == pytest.approx(ref["area"]) and st["volume"] == pytest.approx(ref["volume"])
    g = open_store(rep.results["zarr"].out / "samples/cube/surface.zarr")
    assert g["points"].shape == (8, 3) and g["faces"].shape == (12, 3)
    assert float(np.sum(g["cell_area"][...])) == pytest.approx(6.0)


def _numpy_binary_stl(path: Path, tris: np.ndarray, header: bytes = b"") -> Path:
    rec = np.zeros(len(tris), dtype=geometry._STL_RECORD)
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(header.ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())
    return path


def _tetra() -> np.ndarray:
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f4")
    return p[np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])]


def test_binary_stl_named_solid_open_surface_and_truncation(tmp_path: Path) -> None:
    tris = _tetra()
    # binary files exported by some CAD tools start with the word "solid"
    path = _numpy_binary_stl(tmp_path / "tet.stl", tris, b"solid exported by cad")
    assert geometry.stl_is_binary(path) is True
    (s,) = C.read(path)
    st = s.metadata["geometry"]
    assert st["n_triangles"] == 4 and st["n_points"] == 4 and st["watertight"]
    assert st["volume"] == pytest.approx(1.0 / 6.0) and st["title"] == "solid exported by cad"

    # one face removed: three boundary edges, no volume
    opened = _numpy_binary_stl(tmp_path / "open.stl", tris[:3])
    desc = C.describe(opened)
    st = desc.metadata["geometry"]["surface"]
    assert not st["watertight"] and st["boundary_edges"] == 3 and st["volume"] is None
    assert any("not watertight (3 boundary edges" in w for w in desc.warnings)

    # inward-facing triangles are reported, the volume stays positive
    flipped = _numpy_binary_stl(tmp_path / "flipped.stl", tris[:, ::-1])
    st = C.describe(flipped).metadata["geometry"]["surface"]
    assert st["normals"] == "inward" and st["volume"] == pytest.approx(1.0 / 6.0)

    # a degenerate facet is dropped; vertices closer than the tolerance are merged
    noisy = tris.copy()
    noisy[1, 0] += 1e-7
    extra = np.concatenate([noisy, tris[:1, [0, 0, 1]]])
    (loose,) = C.read(_numpy_binary_stl(tmp_path / "noisy.stl", extra))
    assert loose.metadata["geometry"]["n_points"] == 5
    assert loose.metadata["geometry"]["degenerate_triangles_dropped"] == 1
    (tight,) = C.read(tmp_path / "noisy.stl", options={"merge_tolerance": 1e-4})
    assert tight.metadata["geometry"]["n_points"] == 4 and tight.metadata["geometry"]["watertight"]

    cut = tmp_path / "cut.stl"
    cut.write_bytes(path.read_bytes()[:-30])
    det = C.detect(cut)
    assert det is not None and det.confidence == 0.4
    with pytest.raises(fc.ConnectorError, match="announces 4 triangles") as err:
        list(C.read(cut))
    assert "truncated" in str(err.value)
    ascii_cut = tmp_path / "ascii_cut.stl"
    text = _write_stl(_cube_polydata(), tmp_path / "a.stl", False).read_text()
    ascii_cut.write_text(text[: text.rindex("vertex")])
    with pytest.raises(fc.ConnectorError, match="not a multiple of 3"):
        C.describe(ascii_cut)


def _ascii_solid(name: str, tris: np.ndarray) -> str:
    out = [f"solid {name}"]
    for t in tris:
        out.append(" facet normal 0 0 0\n  outer loop")
        out.extend(f"   vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}" for v in t)
        out.append("  endloop\n endfacet")
    out.append(f"endsolid {name}")
    return "\n".join(out)


def test_multi_solid_ascii_stl_zones(tmp_path: Path) -> None:
    tris = _tetra()
    path = tmp_path / "assembly.stl"
    path.write_text(_ascii_solid("inlet", tris[:1]) + "\n" + _ascii_solid("wall patch", tris[1:]))
    rep = check_connector(C, path, tmp_path / "out")
    zones = {z.name: z.n_cells for z in rep.description.zones}
    assert zones == {"inlet": 1, "wall_patch": 3}
    assert rep.sample_ids == ["assembly__inlet", "assembly__wall_patch"]
    wall = rep.samples["assembly__wall_patch"][0]
    assert wall.zone == "wall_patch" and wall.faces is not None and wall.faces.shape == (3, 3)
    (only,) = C.read(path, zones=["inlet"])
    assert only.id == "assembly" and only.n_cells == 1
    with pytest.raises(fc.SelectionError, match="available zones: inlet, wall_patch"):
        list(C.read(path, zones=["nope"]))
    (merged,) = C.read(path, options={"merge_zones": True})
    assert merged.id == "assembly" and merged.n_cells == 4
    assert merged.fields["zone_id"].location == "cell"
    assert merged.fields["zone_id"].data[:, 0].tolist() == [0, 1, 1, 1]
    assert merged.metadata["geometry"]["zones"] == ["inlet", "wall_patch"]


OBJ_TEXT = """# unit cube: quads, two groups, mixed index styles
mtllib cube.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
vn 0 0 1
vt 0.5 0.5
g bottom_top
f 4 3 2 1
f 5/1 6/1 7/1 8/1
g sides
usemtl steel
f 1//1 2//1 6//1 5//1
f 2/1/1 3/1/1 7/1/1 6/1/1
f -6 -5 -1 -2
f 4 1 5
f 4 5 8
"""


def test_obj_quads_groups_and_vtk_cross_check(tmp_path: Path) -> None:
    path = tmp_path / "cube.obj"
    path.write_text(OBJ_TEXT)
    det = C.detect(path)
    assert det is not None and det.confidence == 0.8
    rep = check_connector(C, path, tmp_path / "out")
    zones = {z.name: (z.n_points, z.n_cells) for z in rep.description.zones}
    assert zones == {"bottom_top": (8, 4), "sides": (8, 8)}
    sides = rep.samples["cube__sides"][0]
    assert sides.metadata["geometry"]["polygons"] == 5
    assert sides.metadata["geometry"]["area"] == pytest.approx(4.0)
    (whole,) = C.read(path, options={"merge_zones": True})
    # the two groups do not share vertices in the output: weld them to judge the whole cube
    pts, inv = merge_vertices(np.asarray(whole.points))
    st = surface_stats(pts, inv[whole.faces])
    assert st["n_points"] == 8 and st["n_triangles"] == 12 and st["watertight"]
    assert st["area"] == pytest.approx(6.0) and st["volume"] == pytest.approx(1.0)
    ref = _vtk_reference(path)
    assert ref["triangles"] == 12 and ref["area"] == pytest.approx(st["area"])
    assert ref["volume"] == pytest.approx(st["volume"])

    bad = tmp_path / "bad.obj"
    bad.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
    with pytest.raises(fc.ConnectorError, match="references vertex 9"):
        list(C.read(bad))
    worse = tmp_path / "worse.obj"
    worse.write_text("v 0 0 zero\n")
    with pytest.raises(fc.ConnectorError, match="OBJ line 1 cannot be parsed"):
        C.describe(worse)
    binary = tmp_path / "prog.obj"
    binary.write_bytes(b"\x7fELF\x00\x01\x02" * 40)
    assert C.detect(binary) is None


def test_binary_ply_with_vertex_properties(tmp_path: Path) -> None:
    import meshio

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype="f4")
    tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
    quality = np.linspace(0.0, 1.0, len(points)).astype("f4")
    path = tmp_path / "pyramid.ply"
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("triangle", tris)],
            point_data={
                "quality": quality,
                "red": np.arange(5, dtype="u1") * 50,
                "green": np.full(5, 7, dtype="u1"),
                "blue": np.full(5, 9, dtype="u1"),
            },
        ),
        binary=True,
    )
    head = path.read_bytes()[:300]
    assert b"binary_little_endian" in head and b"property float quality" in head
    rep = check_connector(C, path, tmp_path / "out")
    d = rep.description
    assert {f.name: (f.components, f.location) for f in d.fields} == {
        "quality": (1, "point"),
        "color": (3, "point"),
    }
    s = rep.first()
    close(s.points, points)
    equal(s.faces, tris)
    close(s.fields["quality"].data[:, 0], quality)
    assert s.fields["color"].data[3].tolist() == [150, 7, 9]
    assert s.fields["color"].component_names == ["red", "green", "blue"]
    st = s.metadata["geometry"]
    assert st["watertight"] and st["volume"] == pytest.approx(1.0 / 3.0)
    ref = _vtk_reference(path)
    assert ref["points"] == 5 and ref["triangles"] == 6
    assert ref["volume"] == pytest.approx(st["volume"]) and ref["area"] == pytest.approx(st["area"])
    (no_fields,) = C.read(path, fields=["quality"])
    assert list(no_fields.fields) == ["quality"]
    g = open_store(rep.results["zarr"].out / "samples/pyramid/surface.zarr")
    assert g["fields/color"].shape == (5, 3) and g["fields/quality"].shape == (5, 1)

    cut = tmp_path / "cut.ply"
    cut.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(fc.ConnectorError, match="truncated"):
        list(C.read(cut))


def test_ascii_and_big_endian_ply_with_polygons_and_face_data(tmp_path: Path) -> None:
    header = (
        "ply\nformat {fmt} 1.0\ncomment made by hand\nelement vertex 5\n"
        "property double x\nproperty double y\nproperty double z\n"
        "element face 3\nproperty uchar patch\nproperty list uchar int vertex_indices\n"
        "end_header\n"
    )
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [1, 2, 0]], dtype=">f8")
    faces = [(7, [0, 1, 2, 3]), (9, [3, 2, 4]), (11, [0, 3, 4])]
    ascii_path = tmp_path / "plate.ply"
    body = "\n".join(" ".join(f"{v:g}" for v in row) for row in verts)
    body += "\n" + "\n".join(f"{p} {len(ix)} " + " ".join(map(str, ix)) for p, ix in faces) + "\n"
    ascii_path.write_text(header.format(fmt="ascii") + body)
    be_path = tmp_path / "plate_be.ply"
    blob = verts.tobytes()
    for p, ix in faces:
        blob += bytes([p, len(ix)]) + np.asarray(ix, dtype=">i4").tobytes()
    be_path.write_bytes(header.format(fmt="binary_big_endian").encode() + blob)
    for path in (ascii_path, be_path):
        rep = check_connector(C, path, tmp_path / f"out-{path.stem}")
        s = rep.first()
        close(s.points, verts.astype(float))
        equal(s.faces, [[0, 1, 2], [0, 2, 3], [3, 2, 4], [0, 3, 4]])
        patch = s.fields["patch"]
        assert patch.location == "cell" and patch.data[:, 0].tolist() == [7, 7, 9, 11]
        st = s.metadata["geometry"]
        assert st["polygons"] == 3 and st["area"] == pytest.approx(2.0 + 1.0 + 0.5)
        assert not st["watertight"]
    ref = _vtk_reference(ascii_path)
    assert ref["triangles"] == 4 and ref["area"] == pytest.approx(3.5)

    cloud = tmp_path / "cloud.ply"
    cloud.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nproperty float intensity\nend_header\n0 0 0 1\n1 0 0 2\n0 1 0 3\n"
    )
    rep = check_connector(C, cloud, tmp_path / "out-cloud")
    s = rep.first()
    assert (
        s.kind == "point_cloud"
        and s.faces is None
        and rep.description.zones[0].kind == "point_cloud"
    )
    assert s.fields["intensity"].data[:, 0].tolist() == [1.0, 2.0, 3.0]
    not_ply = tmp_path / "fake.ply"
    not_ply.write_text("plain text\n")
    assert C.detect(not_ply) is None
    with pytest.raises(fc.ConnectorError, match="missing 'ply' magic"):
        C.describe(not_ply)


def test_off_and_directory_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    off = tmp_path / "tet.off"
    off.write_text(
        "OFF\n# a tetrahedron\n4 4 6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "3 0 2 1\n3 0 1 3\n3 1 2 3\n3 0 3 2\n"
    )
    rep = check_connector(C, off, tmp_path / "off-out")
    assert rep.first().metadata["geometry"]["volume"] == pytest.approx(1.0 / 6.0)

    sweep = tmp_path / "sweep"
    sweep.mkdir()
    for k, size in enumerate((1.0, 2.0, 3.0)):
        _write_stl(_cube_polydata(size), sweep / f"design_{k:03d}.stl", binary=bool(k % 2))
    (sweep / ".DS_Store").write_bytes(b"\x00")
    det = C.detect(sweep)
    assert det is not None and det.format == "geometry-dir" and det.confidence == 0.55
    rep = check_connector(C, sweep, tmp_path / "sweep-out")
    d = rep.description
    assert [z.name for z in d.zones] == ["design_000", "design_001", "design_002"]
    assert d.metadata["n_files"] == 3
    assert rep.sample_ids == ["design_000", "design_001", "design_002"]
    volumes = [rep.samples[i][0].metadata["geometry"]["volume"] for i in rep.sample_ids]
    assert volumes == pytest.approx([1.0, 8.0, 27.0])
    assert rep.results["zarr"].sample_ids == rep.sample_ids
    (two,) = C.read(sweep, zones=["design_002"])
    assert two.id == "design_002" and two.zone == "design_002"
    assert [s.id for s in C.read(sweep, zones=["design_00[01]"])] == ["design_000", "design_001"]
    with pytest.raises(fc.SelectionError, match="available zones: design_000"):
        list(C.read(sweep, zones=["nope"]))
    # large directories are described from headers only
    monkeypatch.setattr(geometry, "STATS_LIMIT", 1)
    d = C.describe(sweep)
    assert d.zones[1].n_cells == 12 and d.zones[1].n_points is None  # binary STL header count
    assert list(d.metadata["geometry"]) == ["design_000"]
    assert any("first 1 of 3 files" in w for w in d.warnings)
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_stl(_cube_polydata(), mixed / "a.stl", True)
    for n in ("b.txt", "c.txt", "d.txt"):
        (mixed / n).write_text("notes")
    assert C.detect(mixed) is None and C.detect(tmp_path / "missing.stl") is None


STEP_STUB = (
    "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('stub'),'2;1');\nENDSEC;\n"
    "DATA;\nENDSEC;\nEND-ISO-10303-21;\n"
)


def test_step_without_tessellator_names_the_licences(tmp_path: Path) -> None:
    path = tmp_path / "part.step"
    path.write_text(STEP_STUB)
    det = C.detect(path)
    assert det is not None and det.format == "step" and det.confidence == 0.9
    if importlib.util.find_spec("OCP") is None:
        assert any("cadquery-ocp" in n for n in det.needs)
        with pytest.raises(fc.MissingDependencyError) as err:
            list(C.read(path))
        text = str(err.value)
        assert "cadquery-ocp" in text and "LGPL-2.1" in text and "GPL-2.0-or-later gmsh" in text
        assert "pip install cadquery-ocp" in text
    if importlib.util.find_spec("gmsh") is None:
        with pytest.raises(fc.MissingDependencyError, match="GPL-2.0-or-later"):
            C.describe(path, options={"tessellator": "gmsh"})
    with pytest.raises(fc.ConnectorError, match="unknown tessellator"):
        C.describe(path, options={"tessellator": "magic"})
    assert C.detect(tmp_path / "x.stp") is None
    fake = tmp_path / "fake.step"
    fake.write_text("not a step file")
    assert C.detect(fake) is None


def test_step_box_tessellated_with_opencascade(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder  # type: ignore[import-not-found]
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer  # type: ignore[import-not-found]

    box = tmp_path / "box.step"
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape(), STEPControl_AsIs)
    w.Write(str(box))
    rep = check_connector(C, box, tmp_path / "out")
    st = rep.description.metadata["geometry"]["surface"]
    assert st["watertight"] and st["volume"] == pytest.approx(24.0, rel=1e-9)
    assert st["area"] == pytest.approx(52.0, rel=1e-9) and st["cad_faces"] == 6
    assert st["n_points"] == 8 and st["n_triangles"] == 12 and st["normals"] == "outward"
    s = rep.first()
    assert sorted(set(s.fields["cad_face"].data[:, 0].tolist())) == [0, 1, 2, 3, 4, 5]

    cyl = tmp_path / "cyl.step"
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape(), STEPControl_AsIs)
    w.Write(str(cyl))
    coarse = C.describe(cyl, options={"tessellation": {"linear_deflection": 0.1}})
    fine = C.describe(cyl, options={"tessellation": {"linear_deflection": 0.001}})
    vc = coarse.metadata["geometry"]["surface"]
    vf = fine.metadata["geometry"]["surface"]
    assert vf["n_triangles"] > 3 * vc["n_triangles"] and vf["watertight"] and vc["watertight"]
    exact = 2.0 * np.pi
    assert abs(vf["volume"] - exact) < abs(vc["volume"] - exact) and vf["volume"] == pytest.approx(
        exact, rel=2e-3
    )


VTK_DATA = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data/"


@pytest.mark.network
@pytest.mark.parametrize(
    ("name", "min_triangles"),
    [("42400-IDGH.stl", 10000), ("bunny.ply", 60000), ("trumpet.obj", 5000), ("cow.obj", 5000)],
)
def test_public_geometry_samples(tmp_path: Path, name: str, min_triangles: int) -> None:
    """Real files from pyvista/vtk-data: a binary STL CAD export, the Stanford bunny (PLY) and
    two Wavefront OBJ models. Counts, area and volume are compared with VTK's readers."""
    path = fetch_sample(VTK_DATA + name)
    rep = check_connector(C, path, tmp_path / "out", options={"merge_zones": True})
    s = rep.first()
    assert s.kind == "surface" and s.faces is not None and s.points is not None
    pts, inv = merge_vertices(s.points)
    welded = inv[s.faces]
    st = surface_stats(pts, welded)
    ref = _vtk_reference(path)
    assert st["n_triangles"] >= min_triangles
    # the OBJ models hold non-planar quads: a fan and VTK's triangulation pick different
    # diagonals, so areas agree to a fraction of a percent only; triangle files agree exactly
    rel = 5e-3 if name.endswith(".obj") else 1e-5
    assert st["area"] == pytest.approx(ref["area"], rel=rel)
    # VTK's cleaner drops degenerate cells and unreferenced points
    assert abs(st["n_triangles"] - ref["triangles"]) <= max(5, ref["triangles"] // 500)
    assert abs(len(np.unique(welded)) - ref["points"]) <= max(5, ref["points"] // 500)
    assert int(s.faces.min()) == 0 and int(s.faces.max()) <= len(s.points) - 1
    if st["watertight"] and st["consistently_oriented"]:
        assert st["volume"] == pytest.approx(ref["volume"], rel=rel)
    if name == "bunny.ply":
        assert len(s.points) == 35947 and len(s.faces) == 69451  # the published counts
