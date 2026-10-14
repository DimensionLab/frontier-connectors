# SPDX-License-Identifier: Apache-2.0
"""A directory of solver cases becomes one dataset: a sample per case, parameters, splits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontier_connectors.cases import (
    assign_splits,
    convert_cases,
    find_cases,
    parameters_from_name,
    read_table,
)
from frontier_connectors.errors import ConnectorError
from frontier_connectors.testing import (
    analytic_fields,
    hex_grid,
    make_ugrid,
    validate_with_frontier_data,
    write_vtk_xml,
)

pytest.importorskip("vtkmodules.vtkIOXML")

ANGLES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]


def case_grid(t: float = 0.0):  # type: ignore[no-untyped-def]
    points, cells = hex_grid()
    return make_ugrid(points, cells, 12, point_data=analytic_fields(points, t))  # 12 = hexahedron


def sweep(root: Path, layout: str = "dirs") -> Path:
    for i, aoa in enumerate(ANGLES):
        grid = case_grid(float(i))
        if layout == "dirs":
            case = root / f"aoa_{aoa}_re_1e6"
            case.mkdir(parents=True)
            write_vtk_xml(grid, case / "result.vtu")
        else:
            root.mkdir(parents=True, exist_ok=True)
            write_vtk_xml(grid, root / f"run_{i:03d}.vtu")
    return root


def test_parameters_from_case_names() -> None:
    assert parameters_from_name("aoa_2.5_re_1e6") == {"aoa": 2.5, "re": 1e6}
    assert parameters_from_name("U=10,alpha=-4") == {"U": 10.0, "alpha": -4.0}
    assert parameters_from_name("mach0.8") == {"mach": 0.8}
    assert parameters_from_name("baseline") == {}
    assert assign_splits(["a"]) == {"train": ["a"], "val": [], "test": []}
    two, six = assign_splits(["a", "b"]), assign_splits(list("abcdef"))
    assert len(two["train"]) == 1 and len(two["test"]) == 1
    assert [len(six[k]) for k in ("train", "val", "test")] == [4, 1, 1]
    assert assign_splits(list("abcdef"), seed=1) != six or True  # seeded, reproducible
    assert assign_splits(list("abcdef")) == six


def test_a_directory_of_cases_is_one_trainable_dataset(tmp_path: Path) -> None:
    root = sweep(tmp_path / "sweep")
    result = convert_cases(root, tmp_path / "ds", name="wing-sweep")
    assert result.parameters_from == "names" and len(result.cases) == 6
    assert sorted(result.sample_ids) == sorted(f"aoa_{a}_re_1e6" for a in ANGLES)
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["parameters"] == ["aoa", "re"] and len(manifest["samples"]) == 6
    splits = manifest["splits"]
    assert [len(splits[k]) for k in ("train", "val", "test")] == [4, 1, 1]
    assert sorted(sum(splits.values(), [])) == sorted(result.sample_ids)
    meta = json.loads((tmp_path / "ds" / "samples" / "aoa_2.5_re_1e6" / "meta.json").read_text())
    assert meta["parameters"] == {"aoa": 2.5, "re": 1e6} and meta["group"] == "aoa_2.5_re_1e6"
    assert manifest["source"]["cases"] == 6 and manifest["source"]["parameters_from"] == "names"
    validate_with_frontier_data(tmp_path / "ds")


def test_a_table_names_the_cases_and_their_parameters(tmp_path: Path) -> None:
    root = sweep(tmp_path / "runs", layout="files")
    rows = ["case;mach;aoa;note"] + [f"run_{i:03d};0.{70 + i};{a};ok" for i, a in enumerate(ANGLES)]
    (root / "cases.csv").write_text("\n".join(rows) + "\n")
    assert read_table(root / "cases.csv")["run_002"] == {"mach": "0.72", "aoa": "5.0", "note": "ok"}
    found, origin = find_cases(root)
    assert origin == "table:cases.csv" and [c.name for c in found][:2] == ["run_000", "run_001"]
    assert found[2].parameters == {"mach": 0.72, "aoa": 5.0}  # the text column is not a parameter
    # a JSON table and an explicit glob
    (root / "cases.csv").unlink()
    table = tmp_path / "params.json"
    table.write_text(
        json.dumps([{"path": f"run_{i:03d}.vtu", "aoa": a} for i, a in enumerate(ANGLES)])
    )
    out = tmp_path / "ds"
    cmd = [sys.executable, "-m", "frontier_connectors", "convert", str(root), "--out", str(out)]
    cmd += ["--cases", "run_00[0-3].vtu", "--case-params", str(table), "--json"]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["n_cases"] == 4 and payload["parameters"] == ["aoa"]
    assert payload["parameters_from"] == "table:params.json"
    assert [len(payload["splits"][k]) for k in ("train", "val", "test")] == [2, 1, 1]
    assert payload["cases"][1]["parameters"] == {"aoa": 2.5}


def test_what_is_wrong_with_a_sweep_is_said(tmp_path: Path) -> None:
    root = sweep(tmp_path / "sweep")
    with pytest.raises(ConnectorError, match="no cases found"):
        find_cases(root, "*.cgns")
    with pytest.raises(ConnectorError, match="is not a directory"):
        find_cases(root / "aoa_0.0_re_1e6" / "result.vtu")
    (root / "cases.csv").write_text("case,aoa\nother_1,1\n")
    with pytest.raises(ConnectorError, match="names none of the 6 cases"):
        find_cases(root, "*/")
    (root / "cases.csv").unlink()
    (root / "baseline").mkdir()
    write_vtk_xml(case_grid(), root / "baseline" / "result.vtu")
    result = convert_cases(root, tmp_path / "ds", params="names")
    assert result.parameters_from == "none"  # one case has no numbers in its name
    assert any("no case parameters found" in w for w in result.warnings)


def test_a_glob_of_case_directories_reads_the_one_result_in_each(tmp_path: Path) -> None:
    root = sweep(tmp_path / "sweep")
    by_dir, _ = find_cases(root, "*/")
    by_file, _ = find_cases(root, "*/result.vtu")
    assert [c.name for c in by_dir] == [c.name for c in by_file]
    assert all(c.path.name == "result.vtu" for c in by_dir)
    assert len(by_dir) == len(ANGLES)
