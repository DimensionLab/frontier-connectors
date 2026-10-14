# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_ensight.connector import (
    EnSightConnector,
    geometry_description,
    parse_case,
    scan_geometry,
    write_ensight_gold,
)

import frontier_connectors as fc
from frontier_connectors.errors import ConnectorError
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    hex_grid,
    make_ugrid,
    quad_grid,
    sample_cache,
)

TIMES = [0.0, 0.1, 0.25, 0.5]
#: the wall part mixes two element types: the first quad is split into two triangles
WALL_TRIS = np.array([[0, 1, 5], [0, 5, 4]])


def _parts(t: float, scale: float = 1.0) -> list[dict[str, Any]]:
    """One hexa8 volume part and one tria3 + quad4 surface part with analytic fields."""
    hp, hc = hex_grid(3, 2, 2)
    qp, qc = quad_grid(3, 2)
    hp, qp = hp * scale, qp * scale
    vol, wall = analytic_fields(hp, t), analytic_fields(qp, t)
    vol_c = analytic_fields(hp[hc].mean(axis=1), t)
    wall_cells = [WALL_TRIS, qc[1:]]
    wall_c = analytic_fields(np.concatenate([qp[c].mean(axis=1) for c in wall_cells]), t)
    return [
        {
            "name": "fluid volume",
            "points": hp,
            "elements": [("hexa8", hc)],
            "node": {"pressure": vol["p"], "velocity": vol["U"]},
            # 'pressure' exists per node and per element on purpose (name collision)
            "element": {"pressure": vol_c["p"], "cell_U": vol_c["U"]},
        },
        {
            "name": "wall",
            "points": qp,
            "elements": [("tria3", WALL_TRIS), ("quad4", qc[1:])],
            "node": {"pressure": wall["p"], "velocity": wall["U"]},
            # 'wss' is a partial variable: only the wall carries it
            "element": {"pressure": wall_c["p"], "cell_U": wall_c["U"], "wss": 2.0 * wall_c["U"]},
        },
    ]


def _transient(tmp_path: Path, encoding: str, *, moving: bool = False) -> Path:
    steps = [_parts(t, 1.0 + t if moving else 1.0) for t in TIMES]
    return write_ensight_gold(
        tmp_path / "flow.case",
        steps,
        times=TIMES,
        encoding=encoding,
        moving=moving,
        constants={"Re": 1200.0},
    )


def _vtk_parts(case: Path, time: float | None = None) -> dict[str, Any]:
    """Independent read with VTK's combined reader (not the one the connector uses)."""
    from vtkmodules.vtkIOEnSight import vtkEnSightGoldCombinedReader

    from frontier_connectors import _vtk

    reader = vtkEnSightGoldCombinedReader()
    reader.SetCaseFileName(str(case))
    reader.UpdateInformation()
    obj = _vtk.update_to_time(reader, time)
    return {name.strip(): ds for name, ds in _vtk.iter_leaves(obj)}


def test_parse_case_sections_time_sets_and_patterns(tmp_path: Path) -> None:
    case = tmp_path / "run.case"
    case.write_text(
        "# Exported by STAR-CCM+ 2402.0001\n"
        "FORMAT\ntype:  ensight gold\n\n"
        "GEOMETRY\nmodel: 2 run.geo.*** change_coords_only\nmeasured: 1 run.mgeo\n\n"
        "VARIABLE\n"
        "constant per case: Mach 0.3\n"
        "constant per case: 1 Lift 1.0 2.0 3.0\n"
        'scalar per node: 1 "p" run.p.****\n'
        "vector per element: 1 1 U run.U.****\n"
        "tensor symm per node: stress run.stress\n"
        "complex scalar per node: 1 cp run.cp_r.**** run.cp_i.**** 50.0\n\n"
        "TIME\n"
        "time set: 1 flow\nnumber of steps: 3\nfilename numbers: 10 20\n30\n"
        "time values: 0.0 0.5\n1.0\n"
        "time set: 2\nnumber of steps: 2\nfilename start number: 5\nfilename increment: 5\n"
        "time values:\n 0.0\n 1.0\n\n"
        "FILE\nfile set: 1\nnumber of steps: 3\n"
    )
    c = parse_case(case)
    assert c.version == "gold" and c.model == "run.geo.***" and c.model_time_set == 2
    assert c.change_coords_only and c.moving and c.measured == "run.mgeo"
    assert c.times() == [0.0, 0.5, 1.0]
    assert c.time_sets[1].numbers == [10, 20, 30] and c.time_sets[2].values == [0.0, 1.0]
    by_name = {v.name: v for v in c.variables}
    assert (by_name["p"].type, by_name["p"].location, by_name["p"].time_set) == (
        "scalar",
        "node",
        1,
    )
    assert (by_name["U"].location, by_name["U"].file_set, by_name["U"].components) == (
        "element",
        1,
        3,
    )
    assert by_name["stress"].components == 6 and not by_name["stress"].time_dependent
    assert by_name["cp"].complex and by_name["cp"].files == ["run.cp_r.****", "run.cp_i.****"]
    assert c.constants(0) == {"Mach": 0.3, "Lift": 1.0} and c.constants(2)["Lift"] == 3.0
    # wildcards take the file number of the step that is current at that time, per time set
    assert c.resolve("run.p.****", 1, 0.5).name == "run.p.0020"
    assert c.geometry_file(0.5).name == "run.geo.005"  # set 2 is still at its first step
    assert c.geometry_file(1.0).name == "run.geo.010"
    assert file_set_steps(c) == {1: [3]}


def file_set_steps(c: Any) -> dict[int, list[int]]:
    return dict(c.file_sets)


@pytest.mark.parametrize("encoding", ["ascii", "c", "fortran"])
def test_transient_parts_values_and_selection(tmp_path: Path, encoding: str) -> None:
    case = _transient(tmp_path, encoding)
    c = EnSightConnector()
    rtol = 2e-5 if encoding == "ascii" else 1e-6
    rep = check_connector(c, case, tmp_path / "out", zones=["*"], min_confidence=0.9)
    desc = rep.description
    assert desc.format == "ensight-gold" and desc.metadata["encoding"] == encoding
    assert desc.temporal == "transient" and desc.time.values == TIMES
    assert desc.parameters == {"Re": 1200.0}
    zones = {z.name: z for z in desc.zones}
    assert list(zones) == ["fluid volume", "wall"]
    assert (zones["fluid volume"].kind, zones["fluid volume"].role) == ("mesh", "volume")
    assert zones["fluid volume"].cell_types == {"hexahedron": 12} and zones["fluid volume"].default
    assert (zones["wall"].kind, zones["wall"].role, zones["wall"].default) == (
        "surface",
        "boundary",
        False,
    )
    assert zones["wall"].cell_types == {"triangle": 2, "quad": 5}
    fields = {f.name: (f.location, f.components, f.zones) for f in desc.fields}
    assert fields == {
        "pressure": ("point", 1, None),
        "velocity": ("point", 3, None),
        "pressure__cell": ("cell", 1, None),
        "cell_U": ("cell", 3, None),
        "wss": ("cell", 3, ["wall"]),
    }
    assert not desc.warnings

    hp, hc = hex_grid(3, 2, 2)
    qp, qc = quad_grid(3, 2)
    vol = rep.samples["flow__fluid_volume"]
    assert [s.time for s in vol] == TIMES and [s.time_index for s in vol] == [0, 1, 2, 3]
    assert vol[0].points is not None and vol[0].cells is not None
    np.testing.assert_allclose(vol[0].points, hp, rtol=rtol, atol=1e-6)
    np.testing.assert_array_equal(vol[0].cells, hc)
    assert vol[1].points is None and vol[1].cells is None  # static geometry is sent once
    for s, t in zip(vol, TIMES, strict=True):
        want = analytic_fields(hp, t)
        np.testing.assert_allclose(s.fields["velocity"].data, want["U"], rtol=rtol, atol=1e-5)
        np.testing.assert_allclose(s.fields["pressure"].data[:, 0], want["p"], rtol=rtol, atol=1e-5)
        cell = analytic_fields(hp[hc].mean(axis=1), t)
        assert s.fields["pressure__cell"].location == "cell"
        np.testing.assert_allclose(s.fields["cell_U"].data, cell["U"], rtol=rtol, atol=1e-5)
        np.testing.assert_allclose(
            s.fields["pressure__cell"].data[:, 0], cell["p"], rtol=rtol, atol=1e-5
        )
        assert "wss" not in s.fields and s.metadata["parameters"] == {"Re": 1200.0}

    wall = rep.samples["flow__wall"]
    assert wall[0].kind == "surface" and wall[0].faces is not None
    assert wall[0].faces.shape == (12, 3)  # 2 triangles + 5 quads split in two
    np.testing.assert_array_equal(wall[0].faces[:2], WALL_TRIS)
    parent = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    cells = [WALL_TRIS, qc[1:]]
    centres = np.concatenate([qp[k].mean(axis=1) for k in cells])
    want = 2.0 * analytic_fields(centres, TIMES[2])["U"][parent]
    np.testing.assert_allclose(wall[2].fields["wss"].data, want, rtol=rtol, atol=1e-5)

    # default selection: only the parts of highest dimension (the volume)
    default = list(c.read(case))
    assert {s.zone for s in default} == {"fluid volume"} and {s.id for s in default} == {"flow"}
    only = list(c.read(case, zones=["wall"], fields=["wss", "pressure"], times=[1, 3]))
    assert [(s.id, s.time) for s in only] == [("flow", 0.1), ("flow", 0.5)]
    assert sorted(only[0].fields) == ["pressure", "wss"]
    last = list(c.read(case, times="last", fields=["pressure__cell"]))
    assert [s.time for s in last] == [0.5] and list(last[0].fields) == ["pressure__cell"]
    near = list(c.read(case, times=[0.26]))
    assert [s.time_index for s in near] == [2]
    with pytest.raises(fc.SelectionError, match="available zones: fluid volume, wall"):
        list(c.read(case, zones=["nope"]))

    # the record scanner and VTK's other EnSight reader agree with what was described
    scan = scan_geometry(tmp_path / "flow.geo")
    assert scan.mode == encoding
    assert [(p.name, p.n_points, p.n_cells, p.dim) for p in scan.parts] == [
        ("fluid volume", 36, 12, 3),
        ("wall", 12, 7, 2),
    ]
    other = _vtk_parts(case, TIMES[1])
    assert {n: (d.GetNumberOfPoints(), d.GetNumberOfCells()) for n, d in other.items()} == {
        z.name: (z.n_points, z.n_cells) for z in desc.zones
    }
    from vtkmodules.util.numpy_support import vtk_to_numpy

    np.testing.assert_allclose(
        vol[1].fields["velocity"].data,
        vtk_to_numpy(other["fluid volume"].GetPointData().GetArray("velocity")),
        rtol=1e-6,
    )
    assert fc.detect(case)[0].connector_id == "ensight"


@pytest.mark.parametrize("encoding", ["ascii", "c"])
def test_moving_geometry_sends_points_every_step(tmp_path: Path, encoding: str) -> None:
    case = _transient(tmp_path, encoding, moving=True)
    assert sorted(p.name for p in tmp_path.glob("flow.geo*")) == [
        f"flow.geo.{i:04d}" for i in range(4)
    ]
    rep = check_connector(EnSightConnector(), case, tmp_path / "out")
    assert rep.description.metadata["moving_geometry"] is True
    hp, _ = hex_grid(3, 2, 2)
    steps = rep.samples["flow"]
    assert len(steps) == 4
    for s, t in zip(steps, TIMES, strict=True):
        assert s.points is not None
        np.testing.assert_allclose(s.points, hp * (1.0 + t), rtol=2e-5, atol=1e-6)
        want = analytic_fields(hp * (1.0 + t), t)["p"]
        np.testing.assert_allclose(s.fields["pressure"].data[:, 0], want, rtol=2e-5, atol=1e-5)


def test_steady_case_directory_and_tensor(tmp_path: Path) -> None:
    parts = _parts(0.0)
    hp = parts[0]["points"]
    stress = np.stack([hp[:, 0] * (i + 1) for i in range(6)], axis=1)
    parts[0]["node"]["stress"] = stress
    case = write_ensight_gold(tmp_path / "steady.encas", [parts], encoding="c")
    c = EnSightConnector()
    rep = check_connector(c, case, tmp_path / "out")
    assert rep.description.temporal == "steady" and rep.description.time.values == []
    s = rep.first()
    assert s.time is None and s.time_index is None
    assert s.fields["stress"].shape == (36, 6)
    # the file stores 11 22 33 12 13 23; VTK hands back xx yy zz xy yz xz and is labelled so
    assert s.fields["stress"].component_names == ["xx", "yy", "zz", "xy", "yz", "xz"]
    np.testing.assert_allclose(s.fields["stress"].data, stress[:, [0, 1, 2, 3, 5, 4]], rtol=1e-6)
    info = {f.name: f for f in rep.description.fields}
    assert info["stress"].zones == ["fluid volume"] and info["stress"].description == "tensor symm"
    assert not info["stress"].time_dependent

    # a directory holding exactly one case file is accepted, at lower confidence
    det = c.detect(tmp_path)
    assert det is not None and det.confidence == 0.5
    assert [x.id for x in c.read(tmp_path)] == ["steady"]
    # .case files of other tools are not claimed
    other = tmp_path / "other.case"
    other.write_text("solver settings\nnot ensight\n")
    assert c.detect(other) is None and c.detect(tmp_path / "missing.case") is None
    binary = tmp_path / "binary.case"
    binary.write_bytes(b"\x00\x01\x02" * 100)
    assert c.detect(binary) is None


@pytest.mark.parametrize("encoding", ["c", "fortran"])
def test_big_endian_binary_and_geometry_every_step(tmp_path: Path, encoding: str) -> None:
    steps = [_parts(t) for t in TIMES[:2]]
    case = write_ensight_gold(
        tmp_path / "be.case", steps, times=TIMES[:2], encoding=encoding, byte_order=">"
    )
    scan = scan_geometry(tmp_path / "be.geo")
    assert scan.byte_order == ">" and [p.n_points for p in scan.parts] == [36, 12]
    assert geometry_description(tmp_path / "be.geo")[0].startswith("EnSight Gold geometry")
    c = EnSightConnector()
    rep = check_connector(c, case, tmp_path / "out")
    hp, _ = hex_grid(3, 2, 2)
    np.testing.assert_allclose(
        rep.samples["be"][1].fields["velocity"].data, analytic_fields(hp, TIMES[1])["U"], rtol=1e-6
    )
    assert rep.samples["be"][1].points is None
    # options geometry_every_step repeats the static mesh (for convert --time-as-samples)
    every = list(c.read(case, options={"geometry_every_step": True}))
    assert all(s.points is not None and s.cells is not None for s in every)


def test_part_names_that_collide_as_ids_stay_separate(tmp_path: Path) -> None:
    parts = _parts(0.0)
    parts[0]["name"], parts[1]["name"] = "side wall", "side wall@"
    twin = dict(parts[1])
    twin["name"] = "side wall@"  # the same description twice
    case = write_ensight_gold(tmp_path / "dup.case", [[*parts, twin]], encoding="ascii")
    c = EnSightConnector()
    assert [z.name for z in c.describe(case).zones] == ["side wall", "side wall@", "side wall@_2"]
    got = {s.zone: s.id for s in c.read(case, zones=["*"])}
    assert got == {
        "side wall": "dup__side_wall",
        "side wall@": "dup__side_wall_2",
        "side wall@_2": "dup__side_wall__2",
    }
    assert len(set(got.values())) == 3


def test_reads_what_vtk_ensight_writer_wrote(tmp_path: Path) -> None:
    """Cross-check against an independent writer: VTK's ``vtkEnSightWriter`` (C binary, parts
    split by the ``BlockId`` cell array)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkIOParallel import vtkEnSightWriter

    points, cells = hex_grid(3, 2, 2)
    f = analytic_fields(points, 0.3)
    cell_p = f["p"][cells].mean(axis=1)
    grid = make_ugrid(points, cells, 12, {"p": f["p"], "U": f["U"]}, {"cp": cell_p})
    block = np.where(np.arange(len(cells)) < 6, 1, 2).astype(np.int32)
    bid = numpy_to_vtk(block, deep=True)
    bid.SetName("BlockId")
    grid.GetCellData().AddArray(bid)
    w = vtkEnSightWriter()
    w.SetInputData(grid)
    w.SetPath(str(tmp_path))
    w.SetBaseName("vtkw")
    w.SetFileName(str(tmp_path / "vtkw.case"))
    w.Write()
    w.WriteCaseFile(1)
    case = tmp_path / "vtkw.0.case"
    assert case.is_file()

    rep = check_connector(EnSightConnector(), case, tmp_path / "out", min_confidence=0.9)
    desc = rep.description
    assert desc.metadata["encoding"] == "c" and desc.temporal == "steady"
    assert [(z.name, z.n_cells, z.kind) for z in desc.zones] == [
        ("VTK Part", 6, "mesh"),
        ("VTK Part_2", 6, "mesh"),
    ]
    assert {f.name: f.location for f in desc.fields} == {
        "p_n": "point",
        "U_n": "point",
        "cp_c": "cell",
    }
    scan = scan_geometry(tmp_path / "vtkw.0.00000.geo")
    assert [(p.n_points, p.n_cells) for p in scan.parts] == [(24, 6), (24, 6)]
    for sid, sel in (("vtkw.0__VTK_Part", block == 1), ("vtkw.0__VTK_Part_2", block == 2)):
        s = rep.samples[sid][0]
        assert s.points is not None and s.cells is not None
        np.testing.assert_allclose(s.fields["cp_c"].data[:, 0], cell_p[sel], rtol=1e-6)
        want = analytic_fields(s.points.astype(np.float64), 0.3)
        np.testing.assert_allclose(s.fields["U_n"].data, want["U"], rtol=1e-5, atol=1e-6)
        # same hexahedra, renumbered per part
        np.testing.assert_allclose(
            np.sort(s.points[s.cells].mean(axis=1), axis=0),
            np.sort(points[cells[sel]].mean(axis=1), axis=0),
            rtol=1e-6,
        )


def test_malformed_case_files(tmp_path: Path) -> None:
    c = EnSightConnector()
    good = _transient(tmp_path, "ascii")
    text = good.read_text()

    def variant(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    bad_kind = variant("bad_kind.case", text.replace("scalar per node", "scalar per banana"))
    with pytest.raises(ConnectorError, match="unknown VARIABLE line") as e:
        c.describe(bad_kind)
    assert e.value.hint and "valid kinds" in e.value.hint

    no_model = variant("no_model.case", text.replace("model: flow.geo", ""))
    with pytest.raises(ConnectorError, match="no 'model:' line") as e:
        c.describe(no_model)
    assert e.value.hint

    no_format = variant("no_format.case", text.replace("FORMAT\ntype: ensight gold\n", ""))
    assert c.detect(no_format) is None
    with pytest.raises(ConnectorError, match="no FORMAT section"):
        list(c.read(no_format))

    junk = variant("junk.case", text.replace("GEOMETRY", "GEOMETRY\nthis is not a key line"))
    with pytest.raises(ConnectorError, match="cannot parse line"):
        c.describe(junk)

    short = variant("short.case", text.replace("number of steps: 4", "number of steps: 9"))
    with pytest.raises(ConnectorError, match="declares 9 steps but lists 4"):
        c.describe(short)

    missing = variant("missing.case", text.replace("model: flow.geo", "model: nowhere.geo"))
    with pytest.raises(ConnectorError, match="nowhere.geo .* is missing") as e:
        c.describe(missing)
    assert e.value.hint and "next to the case file" in e.value.hint
    with pytest.raises(ConnectorError, match="is missing"):
        list(c.read(missing))


@pytest.mark.parametrize("encoding", ["ascii", "c", "fortran"])
def test_truncated_and_missing_data_files_raise(tmp_path: Path, encoding: str) -> None:
    """VTK's EnSight readers segfault on a truncated geometry file and silently drop a truncated
    variable; the connector must turn both into errors before VTK sees the files."""
    c = EnSightConnector()
    src = tmp_path / "src"
    src.mkdir()
    _transient(src, encoding)

    def copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copytree(src, dst)
        return dst / "flow.case"

    case = copy("cut_geo")
    geo = case.parent / "flow.geo"
    geo.write_bytes(geo.read_bytes()[: geo.stat().st_size // 3])
    with pytest.raises(ConnectorError, match="truncated") as e:
        c.describe(case)
    assert e.value.hint and "export again" in e.value.hint
    with pytest.raises(ConnectorError, match="truncated"):
        list(c.read(case))

    case = copy("garbage_geo")
    (case.parent / "flow.geo").write_bytes(b"\x00\x01garbage" * 50)
    with pytest.raises(ConnectorError, match="not an EnSight Gold geometry file"):
        list(c.read(case))

    case = copy("cut_var")
    var = case.parent / "flow.pressure_n.0002"
    var.write_bytes(var.read_bytes()[: var.stat().st_size // 2])
    assert len(list(c.read(case, times=[0, 1]))) == 2
    with pytest.raises(
        ConnectorError, match=r"flow.pressure_n.0002 \(pressure\) is truncated"
    ) as e:
        list(c.read(case))
    assert e.value.hint and "!pressure" in e.value.hint
    assert len(list(c.read(case, fields=["!pressure"]))) == 4

    case = copy("no_var")
    (case.parent / "flow.velocity_n.0000").unlink()
    desc = c.describe(case)
    assert any("velocity" in w and "missing" in w for w in desc.warnings)
    assert "velocity" not in {f.name for f in desc.fields}
    with pytest.raises(ConnectorError, match=r"flow.velocity_n.0000 \(velocity\) .* is missing"):
        list(c.read(case))
    assert len(list(c.read(case, fields=["pressure"]))) == 4


# -- real exports (network) --------------------------------------------------------------------

VTK_DATA = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data/EnSight/"


def fetch_case(base_url: str, names: list[str]) -> Path:
    """Download the files of one multi-file EnSight case into a single cache directory (the
    case file is ``names[0]``). Files are only parsed, never executed."""
    key = hashlib.sha256((base_url + names[0]).encode()).hexdigest()[:16]
    dest = sample_cache() / f"ensight-{key}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest / name.rsplit("/", 1)[-1]
        if target.exists():
            continue
        req = urllib.request.Request(  # noqa: S310
            base_url + name, headers={"User-Agent": "frontier-connectors-tests"}
        )
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(target)
    return dest / names[0].rsplit("/", 1)[-1]


def _scan_matches_description(case: Path, desc: Any) -> None:
    geo = parse_case(case).geometry_file(None)
    scan = scan_geometry(geo)
    assert [(p.n_points, p.n_cells) for p in scan.parts if p.n_points] == [
        (z.n_points, z.n_cells) for z in desc.zones
    ]


@pytest.mark.network
def test_real_fluent_export_steady_2d(tmp_path: Path) -> None:
    """``cylinder_Re35``: EnSight Gold C binary written by ANSYS Fluent 20.1 (pyvista/vtk-data)."""
    stem = "CylinderCrossflow/cylinder_Re35"
    case = fetch_case(
        VTK_DATA, [f"{stem}.case", f"{stem}.geo", f"{stem}.scl1", f"{stem}.scl2", f"{stem}.vel"]
    )
    c = EnSightConnector()
    rep = check_connector(c, case, tmp_path / "out", min_confidence=0.9)
    desc = rep.description
    assert (desc.solver, desc.solver_version, desc.domain) == ("ANSYS Fluent", "20.1", "cfd")
    assert desc.temporal == "steady" and desc.metadata["encoding"] == "c"
    assert [(z.name, z.kind, z.role, z.default) for z in desc.zones] == [
        ("fluid", "surface", "part", True),
        ("inlet", "mesh", "boundary", False),
        ("outlet", "mesh", "boundary", False),
        ("outlet_top", "mesh", "boundary", False),
        ("outlet_bottom", "mesh", "boundary", False),
        ("wall_cylinder", "mesh", "boundary", False),
    ]
    assert desc.zones[0].n_points == 14831 and desc.zones[0].cell_types == {
        "triangle": 39,
        "quad": 14555,
    }
    assert {f.name: (f.location, f.components) for f in desc.fields} == {
        "pressure": ("point", 1),
        "vorticity_mag": ("point", 1),
        "velocity": ("point", 3),
    }
    _scan_matches_description(case, desc)
    fluid = rep.first()
    assert fluid.id == "cylinder_Re35" and fluid.kind == "surface" and fluid.n_points == 14831
    speed = np.linalg.norm(fluid.fields["velocity"].data, axis=1)
    assert np.isfinite(speed).all() and 35.0 < float(speed.max()) < 60.0
    # physics: the cylinder wall is no-slip, the inlet is not
    wall = next(iter(c.read(case, zones=["wall_cylinder"])))
    assert wall.kind == "mesh" and wall.cell_types is not None and set(wall.cell_types) == {3}
    assert float(np.abs(wall.fields["velocity"].data).max()) < 1e-6
    inlet = next(iter(c.read(case, zones=["inlet"], fields=["velocity"])))
    np.testing.assert_allclose(np.linalg.norm(inlet.fields["velocity"].data, axis=1), 35.0)


@pytest.mark.network
def test_real_ensight_transient_structured_blocks(tmp_path: Path) -> None:
    """``naca.bin.case``: written by EnSight 7.3.1; three structured ``block`` parts, two time
    steps with ``filename numbers: 1 3``."""
    case = fetch_case(
        VTK_DATA,
        ["naca.bin.case", "naca.gold.bin.geo", "naca.gold.bin.DENS_1", "naca.gold.bin.DENS_3"],
    )
    rep = check_connector(EnSightConnector(), case, tmp_path / "out", min_confidence=0.9)
    desc = rep.description
    assert desc.temporal == "transient" and desc.time.values == [1.0, 3.0]
    assert [z.shape for z in desc.zones] == [[73, 65, 1], [241, 65, 1], [73, 65, 1]]
    assert all(z.kind == "surface" and z.default for z in desc.zones)
    assert desc.zones[1].name == "block 2, Mins/Maxs=(I:1/241,J:1/65,K:1/1)"
    assert [(f.name, f.location) for f in desc.fields] == [("DENS", "point")]
    _scan_matches_description(case, desc)
    assert len(rep.samples) == 3 and rep.n_steps == 2
    steps = rep.samples[rep.sample_ids[1]]
    assert [s.time for s in steps] == [1.0, 3.0] and steps[1].points is None
    assert steps[0].n_points == 241 * 65 and steps[0].metadata["structured_shape"] == [241, 65, 1]
    d0, d1 = steps[0].fields["DENS"].data, steps[1].fields["DENS"].data
    assert 0.5 < float(d0.min()) < float(d0.max()) < 2.0  # normalised density of a NACA flow
    assert not np.allclose(d0, d1)


@pytest.mark.network
def test_real_code_saturne_polyhedral_export(tmp_path: Path) -> None:
    """``TEST.case``: ASCII EnSight Gold written by Code_Saturne's FVM library 0.3.0 with
    ``nfaced`` polyhedra, ``nsided`` polygons and per-element partial variables."""
    names = ["TEST.case", "test.geo", "test.pressure.0001", "test.velocity.0001"]
    names += ["test.extr_velocity.0001", "test.extr_pressure.0001"]
    case = fetch_case(VTK_DATA, names)
    rep = check_connector(EnSightConnector(), case, tmp_path / "out", zones=["*"])
    desc = rep.description
    assert (desc.solver, desc.solver_version) == ("Code_Saturne", "0.3.0")
    assert desc.temporal == "steady" and desc.metadata["time_value"] == 0.3
    zones = {z.name: z for z in desc.zones}
    assert zones["Volume"].cell_types == {"hexahedron": 122, "polyhedron": 18}
    assert zones["Surface"].cell_types == {"polygon": 18, "quad": 133}
    assert {f.name: (f.location, f.components, f.zones) for f in desc.fields} == {
        "Pressure": ("cell", 1, ["Volume"]),
        "Velocity": ("cell", 3, ["Volume"]),
        "Face_Velocity": ("cell", 3, ["Surface"]),
        "Face_Pressure": ("cell", 1, ["Surface"]),
    }
    _scan_matches_description(case, desc)
    vol = rep.samples["TEST__Volume"][0]
    assert vol.metadata.get("polyhedral") is True and vol.fields["Velocity"].shape == (140, 3)
    assert np.isfinite(vol.fields["Pressure"].data).all()
    surf = rep.samples["TEST__Surface"][0]
    assert surf.kind == "surface" and surf.fields["Face_Pressure"].shape[0] == surf.n_cells


@pytest.mark.network
def test_real_single_file_transient_and_wildcard_series(tmp_path: Path) -> None:
    """``LShape.case`` (EdelweissFE, C binary, all 11 steps in one file, two time sets) and
    ``blow2_ascii.case`` (VTK's writer, one file per step, geometry rewritten every step)."""
    c = EnSightConnector()
    case = fetch_case(VTK_DATA, ["LShape.case", "LShape_geometry.geo", "LShape_displacement.var"])
    rep = check_connector(c, case, tmp_path / "out1")
    desc = rep.description
    assert desc.solver == "EdelweissFE" and desc.time.n_steps == 11
    np.testing.assert_allclose(desc.time.values, np.linspace(0.0, 1.0, 11))
    assert [(z.name, z.kind, z.default) for z in desc.zones][:2] == [
        ("all", "mesh", True),
        ("NSET_ALLNODES", "point_cloud", False),
    ]
    steps = rep.samples["LShape"]
    assert len(steps) == 11 and steps[0].cells is not None and steps[0].cells.shape == (3, 8)
    mag = [float(np.abs(s.fields["displacement"].data).max()) for s in steps]
    assert mag[0] == 0.0 and all(b > a for a, b in zip(mag, mag[1:], strict=False))

    names = ["blow2_ascii.case"]
    for i in ("000", "001"):
        names += [f"blow2_ascii.geo{i}", f"blow2_ascii_pd_displacement{i}"]
        names += [f"blow2_ascii_pd_thickness{i}", f"blow2_ascii_cd_displacement{i}"]
        names += [f"blow2_ascii_cd_thickness{i}"]
    case = fetch_case(VTK_DATA, names)
    rep = check_connector(c, case, tmp_path / "out2")
    assert rep.description.time.values == [0.0, 0.1]
    assert rep.description.metadata["moving_geometry"] is True
    assert {f.name: f.location for f in rep.description.fields} == {
        "displacement": "point",
        "thickness": "point",
        "displacement__cell": "cell",
        "thickness__cell": "cell",
    }
    steps = rep.samples["blow2_ascii"]
    assert len(steps) == 2 and steps[1].points is not None and steps[0].kind == "surface"
    assert steps[0].fields["thickness__cell"].location == "cell"


@pytest.mark.network
def test_real_all_element_types_gold_and_ensight6(tmp_path: Path) -> None:
    """VTK's ``elements.case`` (every EnSight Gold element type, tensors, complex variables)
    and its EnSight 6 twin ``elements6.case``."""
    suffixes = ["geo", "Nsca", "Nvec", "Nten", "NCsca_r", "NCsca_i", "NCvec_r"]
    suffixes += ["Esca", "Evec", "Eten", "ECsca_r", "ECsca_i", "ECvec_r", "ECvec_i"]
    c = EnSightConnector()
    case = fetch_case(VTK_DATA, ["elements.case"] + [f"elements.{s}" for s in suffixes])
    rep = check_connector(c, case, tmp_path / "out1", zones=["*"])
    desc = rep.description
    census = desc.zones[0].cell_types
    for name in (
        "tetra",
        "hexahedron",
        "wedge",
        "pyramid",
        "quadratic_hexahedron",
        "quadratic_tetra",
    ):
        assert census.get(name) == 1, census
    assert desc.zones[1].cell_types == {"polygon": 1}
    info = {f.name: f for f in desc.fields}
    assert info["pointTensors"].components == 6 and info["cellTensors"].location == "cell"
    assert info["pointCScalars"].component_names == ["re", "im"]
    assert info["cellScalars"].zones == [desc.zones[0].name]
    _scan_matches_description(case, desc)
    s = rep.first()
    assert s.fields["pointTensors"].component_names == ["xx", "yy", "zz", "xy", "yz", "xz"]
    assert s.fields["cellVectors"].shape == (15, 3)

    case6 = fetch_case(VTK_DATA, ["elements6.case"] + [f"elements6.{s}" for s in suffixes])
    det = c.detect(case6)
    assert det is not None and det.format == "ensight6"
    rep6 = check_connector(c, case6, tmp_path / "out2")
    assert rep6.description.format == "ensight6" and rep6.first().n_cells == 15
    assert rep6.first().fields["pointVectors"].shape == (108, 3)
