# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_tabular import connector as tabular
from frontier_connector_tabular.connector import TabularConnector, split_unit

import frontier_connectors as fc
from frontier_connectors.cli import main
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    fetch_sample,
    hex_grid,
    open_store,
)

C = TabularConnector()
# typed Any: samples expose optional arrays and the assertions fail loudly on None anyway
close: Any = np.testing.assert_allclose
equal: Any = np.testing.assert_array_equal


def test_paraview_csv_cross_checked_with_vtk_reader(tmp_path: Path) -> None:
    """A CSV in the layout ParaView's "Save Data" writes (quoted ``Points:0`` / ``U:0`` names);
    VTK's own delimited-text reader parses the same file as an independent reference."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader

    points, _ = hex_grid(3, 2, 2)
    f = analytic_fields(points, 0.5)
    path = tmp_path / "slice.csv"
    cols = ["p", "U:0", "U:1", "U:2", "Points:0", "Points:1", "Points:2"]
    data = np.column_stack([f["p"], f["U"], points])
    lines = [",".join(f'"{c}"' for c in cols)]
    lines += [",".join(f"{v:.9g}" for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n")
    reader = vtkDelimitedTextReader()
    reader.SetFileName(str(path))
    reader.SetHaveHeaders(True)
    reader.SetDetectNumericColumns(True)
    reader.Update()
    table = reader.GetOutput()
    ref = {c: vtk_to_numpy(table.GetColumnByName(c)).astype(float) for c in cols}
    assert table.GetNumberOfRows() == len(points)

    det = C.detect(path)
    assert det is not None and det.confidence == 0.5 and "points table" in det.notes[0]
    rep = check_connector(C, path, tmp_path / "out")
    d = rep.description
    assert d.format == "text-table/points" and d.temporal == "steady"
    assert {f.name: f.components for f in d.fields} == {"p": 1, "U": 3}
    assert d.zones[0].n_points == len(points) and d.zones[0].kind == "point_cloud"
    s = rep.first()
    assert s.id == "slice" and s.kind == "point_cloud"
    close(s.points, points, rtol=1e-8)
    close(s.fields["U"].data, f["U"], rtol=1e-6)
    close(s.fields["p"].data[:, 0], f["p"], rtol=1e-6)
    for k in range(3):
        close(np.asarray(s.points)[:, k], ref[f"Points:{k}"], rtol=1e-12)
        close(s.fields["U"].data[:, k], ref[f"U:{k}"], rtol=1e-6)
    g = open_store(rep.results["zarr"].out / "samples/slice/points.zarr")
    assert g["fields/U"].shape == (len(points), 3)
    (only,) = C.read(path, fields=["p"])
    assert list(only.fields) == ["p"]
    with pytest.raises(fc.SelectionError, match="available fields: p, U"):
        list(C.read(path, fields=["nope"]))


def test_units_vector_suffixes_and_fluent_ascii(tmp_path: Path) -> None:
    assert split_unit("p [Pa]") == ("p", "Pa")
    assert split_unit("Temperature (K)") == ("Temperature", "K")
    assert split_unit("Velocity u [m s^-1]") == ("Velocity u", "m s^-1")
    assert split_unit("f (x)") == ("f (x)", "") and split_unit('"x-coordinate"') == (
        "x-coordinate",
        "",
    )
    groups = tabular._component_groups(
        ["U_x", "U_y", "U_z", "velocity-x", "velocity-y", "max", "may", "T", "x-wss", "y-wss"]
    )
    assert groups["U"] == ["U_x", "U_y", "U_z"] and groups["velocity"] == [
        "velocity-x",
        "velocity-y",
    ]
    assert groups["max"] == ["max"] and groups["may"] == ["may"] and groups["T"] == ["T"]
    assert groups["wss"] == ["x-wss", "y-wss"]
    assert tabular._component_groups(["Ux", "Uy", "Uz"]) == {"U": ["Ux", "Uy", "Uz"]}

    # CFD-Post style export: units in brackets, 2-D
    cfx = tmp_path / "plane.csv"
    cfx.write_text(
        "X [ mm ], Y [ mm ], Pressure [ Pa ], Velocity u [ m s^-1 ], Velocity v [ m s^-1 ]\n"
        "0.0, 0.0, 101325.0, 1.0, 0.5\n"
        "10.0, 0.0, 101300.0, 2.0, -0.5\n"
        "10.0, 5.0, , 3.0, 0.0\n"
    )
    rep = check_connector(C, cfx, tmp_path / "cfx")
    d = rep.description
    assert d.units == {"length": "mm"} and d.zones[0].dim == 2
    info = {f.name: f.units for f in d.fields}
    assert info == {"Pressure": "Pa", "Velocity u": "m s^-1", "Velocity v": "m s^-1"}
    s = rep.first()
    assert (
        s.points is not None
        and s.points.shape == (3, 2)
        and s.metadata["units"] == {"length": "mm"}
    )
    assert np.isnan(s.fields["Pressure"].data[2, 0])  # empty cell
    assert s.fields["Pressure"].data[1, 0] == pytest.approx(101300.0)

    # Fluent "File > Export > ASCII" (space delimited): node number + coordinates + fields
    fluent = tmp_path / "export.dat"
    fluent.write_text(
        "nodenumber    x-coordinate    y-coordinate    z-coordinate    pressure    x-velocity"
        "    y-velocity    z-velocity\n"
        "    1  0.0e+00  0.0e+00  0.0e+00  1.5e+00  1.0e+00  0.0e+00  2.0e+00\n"
        "    2  1.0e+00  0.0e+00  0.5e+00  2.5e+00  1.1e+00  0.1e+00  2.1e+00\n"
    )
    det = C.detect(fluent)
    assert det is not None and det.confidence == 0.45
    rep = check_connector(C, fluent, tmp_path / "fluent", min_confidence=0.4)
    s = rep.first()
    assert set(s.fields) == {"pressure", "velocity"} and s.fields["velocity"].shape == (2, 3)
    close(s.fields["velocity"].data[1], [1.1, 0.1, 2.1])
    close(s.points, [[0, 0, 0], [1, 0, 0.5]])
    assert any("nodenumber" in w for w in rep.description.warnings)


def _long_table(n_t: int = 4) -> tuple[str, np.ndarray, np.ndarray]:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.0, 1.0, 0.0]])
    names = ["inlet", "mid", "outlet"]
    rows = ["time;probe;x;y;z;p;u;v"]
    vals = np.zeros((n_t, 3, 3))
    for i in range(n_t):
        for k in (2, 0, 1):  # deliberately not in probe order
            p, u, v = 10.0 * i + k, 0.1 * i, -1.0 * k
            vals[i, k] = [p, u, v]
            rows.append(f"{0.5 * i};{names[k]};{pos[k, 0]};{pos[k, 1]};{pos[k, 2]};{p};{u};{v}")
    return "\n".join(rows) + "\n", pos, vals


def test_long_format_probe_series(tmp_path: Path) -> None:
    text, pos, vals = _long_table()
    path = tmp_path / "probes.csv"
    path.write_text(text)
    rep = check_connector(C, path, tmp_path / "out")
    d = rep.description
    assert d.metadata["layout"] == "long" and d.metadata["probes"] == ["outlet", "inlet", "mid"]
    assert d.time.values == [0.0, 0.5, 1.0, 1.5] and d.zones[0].n_points == 3
    s = rep.first()
    order = [2, 0, 1]  # first appearance in the file
    close(s.points, pos[order])
    assert s.fields["p"].dims == ["time", "point", "component"] and s.fields["p"].shape == (4, 3, 1)
    close(s.fields["p"].data[..., 0], vals[:, order, 0])
    assert "u" in s.fields and "v" in s.fields  # u/v are not suffix-named, so not grouped
    g = open_store(rep.results["zarr"].out / "samples/probes/points.zarr")
    assert g["fields/p"].shape == (4, 3, 1) and g["points"].shape == (3, 3)
    close(g["times"][...], [0.0, 0.5, 1.0, 1.5])

    opts = {"mapping": {"fields": {"U": ["u", "v"]}, "units": {"U": "m/s", "time": "ms"}}}
    (m,) = C.read(path, options=opts, times=[1, 3])
    assert list(m.fields) == ["U"] and m.fields["U"].units == "m/s"
    assert m.metadata["time_units"] == "ms"
    close(m.times, [0.5, 1.5])
    close(m.fields["U"].data, vals[[1, 3]][:, order, 1:])

    # ragged: one probe stops early
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("\n".join(text.splitlines()[:-2]) + "\n")
    with pytest.raises(fc.ConnectorError, match="ragged") as err:
        list(C.read(ragged))
    assert "per_probe" in str(err.value)
    per = list(C.read(ragged, options={"per_probe": True}))
    assert sorted(x.id for x in per) == ["ragged_inlet", "ragged_mid", "ragged_outlet"]
    lengths = {x.id: len(x.times) for x in per if x.times is not None}
    assert lengths == {"ragged_outlet": 4, "ragged_inlet": 3, "ragged_mid": 3}
    assert all(x.points is not None and x.points.shape == (1, 3) for x in per)


def test_moving_points_without_ids_are_read_step_by_step(tmp_path: Path) -> None:
    rows = ["t,x,y,z,T"]
    for i in range(3):
        for k in range(4):
            rows.append(f"{i},{k + 0.1 * i},{k},0,{100 * i + k}")
    path = tmp_path / "particles.csv"
    path.write_text("\n".join(rows))
    rep = check_connector(C, path, tmp_path / "out")
    steps = rep.samples["particles"]
    assert [s.time for s in steps] == [0.0, 1.0, 2.0]
    assert steps[2].points is not None and steps[2].points[3, 0] == pytest.approx(3.2)
    assert steps[1].fields["T"].data[:, 0].tolist() == [100.0, 101.0, 102.0, 103.0]
    g = open_store(rep.results["zarr"].out / "samples/particles/points.zarr")
    assert g["points_t"].shape == (3, 4, 3)


def test_monitor_series_and_mapping_positions(tmp_path: Path) -> None:
    # Fluent report-file layout: title line, parenthesised quoted header
    out = tmp_path / "report-def-0-rfile.out"
    out.write_text(
        '"report-def-0-rfile"\n'
        '"Time Step" "cd" "cl etc.." "flow-time"\n'
        '("Time Step" "cd" "cl" "flow-time")\n'
        "1 0.31 0.02 0.001\n2 0.30 0.03 0.002\n3 0.29 0.04 0.003\n"
    )
    rep = check_connector(C, out, tmp_path / "o1", min_confidence=0.4)
    d = rep.description
    assert d.metadata["layout"] == "series" and d.time.values == [0.001, 0.002, 0.003]
    assert [f.name for f in d.fields] == ["cd", "cl"]
    assert any("single point at the origin" in w for w in d.warnings)
    s = rep.first()
    close(s.points, [[0.0, 0.0, 0.0]])
    assert s.fields["cd"].shape == (3, 1, 1)
    close(s.fields["cd"].data[:, 0, 0], [0.31, 0.30, 0.29])

    # wide probe table (COMSOL-like '%' header, time unit) + positions from a mapping
    wide = tmp_path / "probe_table.txt"
    wide.write_text(
        "% Time (s)    Temperature (K), Point 1    Temperature (K), Point 2\n"
        "0   300.0   310.0\n0.5   301.0   311.0\n1.0   302.0   312.0\n"
    )
    plain = C.describe(wide)
    assert [f.name for f in plain.fields] == [
        "Temperature (K), Point 1",
        "Temperature (K), Point 2",
    ]
    mapping = {
        "time": "Time",
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
        "fields": {"T": ["Temperature (K), Point 1", "Temperature (K), Point 2"]},
        "units": {"T": "K"},
        "parameters": {"power": 50.0},
        "domain": "thermal",
        "solver": "COMSOL",
    }
    rep = check_connector(
        C, wide, tmp_path / "o2", options={"mapping": mapping}, min_confidence=0.4
    )
    d = rep.description
    assert d.domain == "thermal" and d.solver == "COMSOL" and d.parameters == {"power": 50.0}
    assert d.zones[0].n_points == 2 and not d.warnings
    s = rep.first()
    assert s.fields["T"].shape == (3, 2, 1) and s.fields["T"].units == "K"
    close(s.fields["T"].data[:, 1, 0], [310.0, 311.0, 312.0])
    close(s.times, [0.0, 0.5, 1.0])
    assert s.metadata["parameters"] == {"power": 50.0}


FOAM_U = """# Probe 0 (0.1 0.2 0.3)
# Probe 1 (0.4 0.5 0.6)
# Probe 2 (1 1 1)
#       Probe             0             1             2
#        Time
{rows}
"""


def _foam_case(root: Path) -> Path:
    probes = root / "postProcessing" / "probes"
    for start, times in (("0", [0.1, 0.2, 0.3]), ("0.25", [0.3, 0.4])):
        d = probes / start
        d.mkdir(parents=True)
        u_rows, p_rows = [], []
        for t in times:
            run = 1.0 if start == "0" else 2.0  # the restart rewrites t = 0.3
            vec = "   ".join(f"({run * (k + t):.6g} {k} {-t})" for k in range(3))
            u_rows.append(f"           {t}   {vec}")
            p_rows.append(
                f"           {t}   " + "   ".join(f"{run * 10 * k + t:.6g}" for k in range(3))
            )
        (d / "U").write_text(FOAM_U.format(rows="\n".join(u_rows)))
        (d / "p").write_text(FOAM_U.format(rows="\n".join(p_rows)))
    return probes


def test_openfoam_probes_directory_with_restart(tmp_path: Path) -> None:
    probes = _foam_case(tmp_path)
    det = C.detect(probes)
    assert det is not None and det.format == "openfoam-probes" and det.confidence == 0.85
    one = C.detect(probes / "0" / "U")
    assert one is not None and one.confidence == 0.9
    rep = check_connector(C, probes, tmp_path / "out")
    d = rep.description
    assert d.solver == "OpenFOAM" and d.domain == "cfd" and d.zones[0].n_points == 3
    assert {f.name: f.components for f in d.fields} == {"U": 3, "p": 1}
    assert d.time.values == pytest.approx([0.1, 0.2, 0.3, 0.4])
    s = rep.first()
    close(s.points, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1, 1, 1]])
    u = s.fields["U"].data
    assert u.shape == (4, 3, 3)
    close(u[1, 2], [2.2, 2.0, -0.2], rtol=1e-6)  # first run
    close(u[2, 1], [2.0 * 1.3, 1.0, -0.3], rtol=1e-6)  # restart wins
    close(s.fields["p"].data[3, :, 0], [0.4, 20.4, 40.4], rtol=1e-6)
    (single,) = C.read(probes / "0" / "U", times="last")
    assert list(single.fields) == ["U"] and single.times is not None and single.times[0] == 0.3
    # a whole case directory is not a table
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("FoamFile {}")
    assert C.detect(tmp_path) is None


def test_openfoam_force_coefficients(tmp_path: Path) -> None:
    path = tmp_path / "coefficient.dat"
    path.write_text(
        "# Force coefficients\n"
        "# dragDir               : (1 0 0)\n"
        "# magUInf               : 30\n"
        "# lRef                  : 1.5\n"
        "# Aref                  : 2.1\n"
        "# Time          \tCd            \tCl            \tCmPitch\n"
        "1               \t3.5e-01       \t1.0e-02       \t-2.0e-02\n"
        "2               \t3.4e-01       \t1.1e-02       \t-2.1e-02\n"
    )
    det = C.detect(path)
    assert det is not None and det.confidence == 0.45
    rep = check_connector(C, path, tmp_path / "out", min_confidence=0.4)
    d = rep.description
    assert [f.name for f in d.fields] == ["Cd", "Cl", "CmPitch"] and d.time.values == [1.0, 2.0]
    assert d.metadata["header_values"]["magUInf"] == "30"
    s = rep.first()
    close(s.fields["Cd"].data[:, 0, 0], [0.35, 0.34])
    assert s.points is not None and s.points.shape == (1, 3)


def test_whitespace_decimal_comma_and_large_file_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eu = tmp_path / "eu.csv"
    eu.write_text("x;y;Druck [bar]\n0,5;1,0;1,25\n1,5;2,0;2,5\n")
    (s,) = C.read(eu)
    close(s.points, [[0.5, 1.0], [1.5, 2.0]])
    close(s.fields["Druck"].data[:, 0], [1.25, 2.5])
    assert s.fields["Druck"].units == "bar"

    n = 2500
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n, 5))
    big = tmp_path / "big.txt"
    with open(big, "w") as f:
        f.write("# generated\nx y z T rho\n")
        for i, row in enumerate(data):
            f.write(" ".join(f"{v:.9e}" for v in row) + "\n")
            if i == 1200:
                f.write("\n# a comment in the middle\n")
    monkeypatch.setattr(tabular, "CHUNK_LINES", 700)  # forces four blocks
    (s,) = C.read(big)
    assert s.points is not None and s.points.shape == (n, 3)
    close(s.points, data[:, :3], rtol=1e-8)
    close(s.fields["rho"].data[:, 0], data[:, 4], rtol=1e-6)


def test_parquet_row_groups_and_vector_columns(tmp_path: Path) -> None:
    # exc_type: a pyarrow built for another numpy major raises ImportError, not ModuleNotFound
    pa = pytest.importorskip("pyarrow", exc_type=ImportError)
    pq = pytest.importorskip("pyarrow.parquet", exc_type=ImportError)
    points, _ = hex_grid(4, 3, 2)
    f = analytic_fields(points, 0.0)
    table = pa.table(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "p [Pa]": f["p"],
            "U": pa.FixedSizeListArray.from_arrays(pa.array(f["U"].ravel()), 3),
            "wss_x": f["U"][:, 0],
            "wss_y": f["U"][:, 1],
            "tag": pa.array(["a"] * len(points)),
        }
    )
    path = tmp_path / "cloud.parquet"
    pq.write_table(table, path, row_group_size=16)
    assert pq.ParquetFile(path).metadata.num_row_groups > 2
    det = C.detect(path)
    assert det is not None and det.format == "parquet" and det.confidence == 0.6
    rep = check_connector(C, path, tmp_path / "out")
    d = rep.description
    assert {f.name: (f.components, f.units) for f in d.fields} == {
        "U": (3, ""),
        "p": (1, "Pa"),
        "wss": (2, ""),
    }
    s = rep.first()
    close(s.points, points)
    close(s.fields["U"].data, f["U"], rtol=1e-6)
    close(s.fields["wss"].data, f["U"][:, :2], rtol=1e-6)

    # long-format parquet with an integer probe id
    t = np.repeat(np.arange(5) * 0.1, 3)
    pid = np.tile(np.arange(3), 5)
    long = pa.table({"time": t, "probe_id": pid, "x": pid * 1.0, "y": pid * 0.0, "T": t * 10 + pid})
    lpath = tmp_path / "series.pq"
    pq.write_table(long, lpath)
    rep = check_connector(C, lpath, tmp_path / "long")
    s = rep.first()
    assert s.fields["T"].shape == (5, 3, 1) and s.points is not None and s.points.shape == (3, 2)
    close(s.fields["T"].data[4, :, 0], [4.0, 5.0, 6.0], rtol=1e-6)

    bad = tmp_path / "bad.parquet"
    bad.write_bytes(path.read_bytes()[:200])
    with pytest.raises(fc.ConnectorError, match="not a readable Parquet file"):
        C.describe(bad)


def test_mapping_override_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "rig.txt"
    path.write_text(
        "rig export v2\nsecond junk line\na|b|c|d|e\n0.0|1.0|5.0|7.0|9.0\n2.0|3.0|6.0|8.0|10.0\n"
    )
    mapping = {
        "skip_rows": 2,
        "delimiter": "|",
        "x": "a",
        "y": "b",
        "fields": {"q": ["c", "d"]},
        "units": {"q": "W/m^2", "length": "mm"},
    }
    with pytest.raises(fc.MappingError, match="no coordinate columns") as err:
        C.describe(path, options={"mapping": {"skip_rows": 2}})
    assert "'a', 'b', 'c', 'd', 'e'" in str(err.value)
    det = C.detect(path)
    assert det is None or det.confidence <= 0.3
    rep = (
        check_connector(
            C,
            path,
            tmp_path / "out",
            options={"mapping": mapping},
            convert=True,
            min_confidence=0.0,
        )
        if det
        else None
    )
    (s,) = C.read(path, options={"mapping": mapping})
    assert list(s.fields) == ["q"] and s.fields["q"].units == "W/m^2"
    close(s.fields["q"].data, [[5.0, 7.0], [6.0, 8.0]])
    close(s.points, [[0.0, 1.0], [2.0, 3.0]])
    assert s.metadata["units"] == {"length": "mm"}
    del rep
    with pytest.raises(fc.MappingError, match="not in rig.txt"):
        C.describe(path, options={"mapping": {**mapping, "x": "nope"}})
    with pytest.raises(fc.MappingError, match="unknown mapping key"):
        C.describe(path, options={"mapping": {**mapping, "feilds": {}}})

    mfile = tmp_path / "mapping.json"
    mfile.write_text(json.dumps(mapping))
    out = tmp_path / "ds"
    args = ["--connector", "tabular", "--mapping", str(mfile)]
    assert main(["convert", str(path), "--out", str(out), "--json", *args]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["sample_ids"] == ["rig"]


def test_malformed_and_foreign_tables(tmp_path: Path) -> None:
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("x,y,z,p\n0,0,0,1\n1,0,0\n2,0,0,3\n")
    with pytest.raises(fc.ConnectorError, match="line 3 has 3 columns, the header has 4"):
        list(C.read(ragged))
    words = tmp_path / "words.csv"
    words.write_text("x,y,z,p\n0,0,0,1\n1,0,0,oops\n")
    with pytest.raises(fc.ConnectorError, match="line 3, column 'p': 'oops' is not a number"):
        list(C.read(words))
    nocoord = tmp_path / "nocoord.csv"
    nocoord.write_text("alpha,beta,gamma\n1,2,3\n4,5,6\n")
    det = C.detect(nocoord)
    assert det is not None and det.confidence == 0.3 and det.needs
    with pytest.raises(fc.MappingError) as err:
        C.describe(nocoord)
    assert "'alpha', 'beta', 'gamma'" in str(err.value) and "--mapping" in str(err.value)

    tecplot = tmp_path / "flow.dat"
    tecplot.write_text('TITLE = "x"\nVARIABLES = "X", "Y", "P"\nZONE T="a", I=2\n0 0 1\n1 0 2\n')
    comsol = tmp_path / "comsol.txt"
    comsol.write_text("% Model:              heat.mph\n% Version: 6.1\n% x y T\n0 0 300\n1 0 301\n")
    prose = tmp_path / "readme.txt"
    prose.write_text("This is a readme.\nNothing tabular here.\n")
    binary = tmp_path / "blob.dat"
    binary.write_bytes(bytes(range(256)) * 8)
    for p in (tecplot, comsol, prose, binary, tmp_path / "missing.csv", tmp_path):
        assert C.detect(p) is None, p
    with pytest.raises(fc.ConnectorError, match="not a delimited numeric table"):
        C.describe(prose)
    dates = tmp_path / "dates.csv"
    dates.write_text("time,p\n2024-01-01,1\n2024-01-02,2\n")
    with pytest.raises(fc.MappingError, match="date strings are not parsed"):
        C.describe(dates)


FLUIDFOAM = (
    "https://raw.githubusercontent.com/fluiddyn/fluidfoam/master/output_samples/ascii/"
    "postProcessing/"
)


@pytest.mark.network
def test_public_openfoam_postprocessing_samples(tmp_path: Path) -> None:
    """Real OpenFOAM function-object output (probes with restarts, forceCoeffs, forces) from
    the sample data of the fluidfoam project. The files are downloaded to the cache and laid
    out as ``postProcessing/<name>/<time>/<file>`` in the temporary directory."""
    import shutil

    root = tmp_path / "postProcessing"
    wanted = [(f"probes/{t}/{f}") for t in ("0", "0.1", "0.2") for f in ("U", "p")]
    wanted += ["forces/100/coefficient.dat", "forces/100/forces.dat"]
    for rel in wanted:
        src = fetch_sample(FLUIDFOAM + rel, name=rel.replace("/", "_"))
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, root / rel)

    rep = check_connector(C, root / "probes", tmp_path / "probes-out")
    d = rep.description
    assert d.format == "openfoam-probes" and d.solver == "OpenFOAM"
    # in this sample p was sampled on one probe and U on four: two probe sets
    assert {z.name: z.n_points for z in d.zones} == {"probes": 4, "probes_p": 1}
    u = rep.samples["probes__probes"][0]
    close(u.points, [[0.05, 0.05, 0], [0.2, 0, 0], [0.2, 0.05, 0], [0.2, -0.05, 0]])
    assert u.fields["U"].shape == (13, 4, 3) and u.times is not None
    assert u.times[0] == pytest.approx(92.0002) and u.times[-1] == pytest.approx(92.0026)
    assert np.all(np.diff(u.times) > 0)  # the restart overlap was merged
    close(u.fields["U"].data[0, 1], [0.209107, -0.211087, 0.0], atol=1e-6)
    # 92.0016 exists in both runs: the restart (directory 0.2) wins
    k = int(np.argmin(np.abs(u.times - 92.0016)))
    close(u.fields["U"].data[k, 0, :2], [0.00250531, 0.0347091], rtol=1e-5)
    p = rep.samples["probes__probes_p"][0]
    assert p.fields["p"].shape == (13, 1, 1)

    coeff = root / "forces/100/coefficient.dat"
    # every 50th step: one chunk per step makes a 10 000-step conversion slow for a unit test
    rep = check_connector(C, coeff, tmp_path / "coeff-out", min_confidence=0.4, times="::50")
    d = rep.description
    assert [f.name for f in d.fields] == ["Cd", "Cl"] and d.time.n_steps == 10000
    assert d.metadata["header_values"]["magUInf"] == "1.000000e-04"
    s = rep.first()
    assert s.fields["Cd"].data[0, 0, 0] == pytest.approx(1.6845)
    assert s.times is not None and s.times[0] == pytest.approx(100.01)

    forces = root / "forces/100/forces.dat"
    rep = check_connector(C, forces, tmp_path / "forces-out", min_confidence=0.4)
    names = {f.name: f.components for f in rep.description.fields}
    assert names == {
        "forces_pressure": 3, "forces_viscous": 3, "forces_porous": 3,
        "moments_pressure": 3, "moments_viscous": 3, "moments_porous": 3,
    }  # fmt: skip
    s = rep.first()
    close(s.fields["forces_pressure"].data[0, 0], [-222.5302, 119.4625, 0.0], atol=1e-3)
    close(s.fields["moments_viscous"].data[0, 0, 2], 0.3857731, rtol=1e-6)
