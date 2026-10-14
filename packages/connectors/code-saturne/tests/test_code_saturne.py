# SPDX-License-Identifier: Apache-2.0
"""code_saturne: a complete public results directory written by code_saturne 9.1.

The layout test on generated files and the FVM 0.3 sample are in ``test_vendors.py``; this file
holds the checks against a real ``RESU/<run>`` (EnSight output, ``setup.xml``, solver log).
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from frontier_connector_code_saturne.connector import CodeSaturneConnector

import frontier_connectors as fc
from frontier_connectors.testing import check_connector, fetch_sample

# "RANS resistance ... of the DARPA SUBOFF AFF-8 submarine, code_saturne 9.1", F. Hermet
# (Simvia), Zenodo record 22692726, DOI 10.5281/zenodo.22692726, licence CC-BY-4.0. Downloaded
# at test time, never committed. The fields are one zstd-compressed tar (30 MB) per run.
SUBOFF = "https://zenodo.org/api/records/22692726/files/{}/content"
SUBOFF_FILES = {
    "fields_ensight__RANS_run_exp_5.93kn_fields_ensight.tar.zst": (
        "8d8ed9d1691b44366b203ef071a7a06eefd0aa21c0249dd0ad431e884e91e6cb"
    ),
    "results__RANS__run_exp_5.93kn__setup.xml": (
        "325fe82b374562ae93ee33f1ccfc3a025f4ad031b0e4a41cd9a11865ae21bedc"
    ),
    "results__RANS__run_exp_5.93kn__logs__run_solver.log.zst": (
        "ede52383ccb9225ba0d33a02d91e65dd22fd7b12131fc87b833af47644124670"
    ),
}


def _unzstd(src: Path, dest: Path) -> None:
    """Decompress a .zst file (Python 3.14 ``compression.zstd``, else the ``zstd`` program)."""
    try:
        from compression import zstd  # type: ignore[import-not-found]
    except ImportError:
        exe = shutil.which("zstd")
        if exe is None:
            pytest.skip("needs Python 3.14 (compression.zstd) or the zstd program")
        subprocess.run([exe, "-d", "-q", "-f", str(src), "-o", str(dest)], check=True)  # noqa: S603
        return
    with zstd.open(src, "rb") as f, open(dest, "wb") as out:
        shutil.copyfileobj(f, out)


def _suboff_case(tmp_path: Path) -> Path:
    """``SUBOFF/RANS/RESU/run_exp_5.93kn`` rebuilt from the record's flat file names (the
    author's run directory, as printed in the log)."""
    files = {
        name: fetch_sample(SUBOFF.format(name), name, sha256=sha)
        for name, sha in SUBOFF_FILES.items()
    }
    case_dir = tmp_path / "SUBOFF" / "RANS"
    run = case_dir / "RESU" / "run_exp_5.93kn"
    run.mkdir(parents=True)
    (case_dir / "DATA").mkdir()
    names = list(files)
    _unzstd(files[names[0]], tmp_path / "fields.tar")
    with tarfile.open(tmp_path / "fields.tar") as tar:
        tar.extractall(run, filter="data")  # members are postprocessing/*
    (tmp_path / "fields.tar").unlink()
    shutil.copy(files[names[1]], run / "setup.xml")
    _unzstd(files[names[2]], run / "run_solver.log")
    return case_dir


SETUP_XML = """<?xml version="1.0" encoding="utf-8"?>
<Code_Saturne_GUI case="CASE1" solver_version="8.0" study="STUDY" version="2.0">
  <analysis_control><time_parameters>
    <iterations>50</iterations><time_passing>0</time_passing><time_step_ref>0.01</time_step_ref>
  </time_parameters></analysis_control>
  <boundary_conditions>
    <boundary label="inlet" name="1" nature="inlet">1</boundary>
    <boundary label="wall" name="2" nature="wall">2 or 3</boundary>
    <inlet label="inlet">
      <turbulence choice="hydraulic_diameter"><hydraulic_diameter>1</hydraulic_diameter>
      </turbulence>
      <velocity_pressure choice="norm" direction="normal"><norm>1.5</norm></velocity_pressure>
    </inlet>
    <wall label="wall"><velocity_pressure choice="off"/></wall>
  </boundary_conditions>
  <numerical_parameters><velocity_pressure_algo choice="simplec"/></numerical_parameters>
  <physical_properties>
    <fluid_properties>
      <property choice="constant" label="Density" name="density">
        <initial_value>1.17862</initial_value></property>
      <property choice="user_law" label="LamVisc" name="molecular_viscosity">
        <initial_value>1.83e-05</initial_value></property>
      <reference_pressure>101325</reference_pressure>
    </fluid_properties>
    <notebook><var name="u_in" value="1.5"/><var name="label" value="abc"/></notebook>
  </physical_properties>
  <solution_domain><meshes_list><mesh name="pipe.med"/></meshes_list></solution_domain>
  <thermophysical_models>
    <thermal_scalar model="temperature_celsius"/>
    <turbulence model="k-epsilon-PL"/>
    <radiative_transfer model="off"/>
  </thermophysical_models>
</Code_Saturne_GUI>
"""


def test_setup_xml_and_solver_log_metadata(tmp_path: Path) -> None:
    """What ``setup.xml`` and ``run_solver.log`` add (element names as in the real code_saturne
    9.1 case below). The inlet's own ``<turbulence choice=...>`` precedes the model element."""
    from frontier_connector_ensight.connector import write_ensight_gold

    from frontier_connectors.testing import hex_grid

    run = tmp_path / "STUDY" / "CASE1" / "RESU" / "20240102-1030"
    (tmp_path / "STUDY" / "CASE1" / "DATA").mkdir(parents=True)
    hp, hc = hex_grid(2, 2, 2)
    parts = [[{"name": "inlet", "points": hp, "elements": [("hexa8", hc)], "node": {"p": hp[:, 0]},
               "element": {}}]]  # fmt: skip
    (run / "postprocessing").mkdir(parents=True)
    write_ensight_gold(run / "postprocessing" / "RESULTS_FLUID_DOMAIN.case", parts, encoding="c")
    (run / "setup.xml").write_text(SETUP_XML)
    c = CodeSaturneConnector()
    d = c.describe(run)
    assert d.solver_version == "8.0"  # nothing better than the GUI's version attribute
    assert d.physics == {
        "turbulence_model": "k-epsilon-PL",
        "thermal_scalar": "temperature_celsius",
        "velocity_pressure_algorithm": "simplec",
        "time_stepping": "constant time step",
    }
    # constant properties only; the notebook's numeric variables
    assert d.parameters == {
        "time_step_ref": 0.01, "iterations": 50.0, "density": 1.17862,
        "reference_pressure": 101325.0, "u_in": 1.5,
    }  # fmt: skip
    assert d.metadata["notebook"] == {"u_in": 1.5} and d.metadata["meshes"] == ["pipe.med"]
    assert d.boundary_conditions == {
        "inlet": {"type": "inlet", "selection": "1", "zone_id": 1, "velocity_pressure": "norm",
                  "norm": 1.5},
        "wall": {"type": "wall", "selection": "2 or 3", "zone_id": 2},
    }  # fmt: skip
    assert d.zone("inlet").bc_type == "inlet"  # a part named like a boundary label

    (run / "run_solver.log").write_text(
        "command: \n ./cs_solver\n\n" + "*" * 60 + "\n\n            code_saturne\n\n"
        "            Version 8.0.3\n\n  Copyright (C) 1998-2023 EDF S.A., France\n"
    )
    assert c.describe(run).solver_version == "8.0.3"
    (run / "setup.xml").write_text("<Code_Saturne_GUI><broken")
    d = c.describe(run)
    assert d.physics == {} and any("could not be parsed" in w for w in d.warnings)


@pytest.mark.network
def test_real_code_saturne_9_results_directory(tmp_path: Path) -> None:
    case_dir = _suboff_case(tmp_path)
    post = case_dir / "RESU" / "run_exp_5.93kn" / "postprocessing"
    c = CodeSaturneConnector()
    assert fc.detect(case_dir)[0].connector_id == "code-saturne"
    rep = check_connector(c, case_dir, tmp_path / "out", min_confidence=0.95)
    d = rep.description
    # version from the solver banner / the EnSight description line, not the GUI's "9.1"
    assert (d.solver, d.solver_version, d.domain) == ("Code_Saturne", "9.1.0", "cfd")
    assert d.metadata["run"] == "run_exp_5.93kn" and d.metadata["route"] == "ensight"
    assert d.metadata["routed_path"] == str(post / "RESULTS_FLUID_DOMAIN.case")
    assert d.temporal == "steady"

    # setup.xml: the inlet's <turbulence choice=...> must not be taken for the model
    assert d.physics["turbulence_model"] == "k-omega-SST"
    assert d.physics["velocity_pressure_algorithm"] == "simplec"
    assert d.physics["time_stepping"].startswith("local time step")
    ux = d.parameters["UX"]
    assert ux == 3.0506 and d.metadata["notebook"] == {"UX": 3.0506}
    assert d.parameters["density"] == 998.2 and d.parameters["molecular_viscosity"] == 1.003e-3
    assert d.parameters["iterations"] == 2000 and d.metadata["meshes"] == ["darpa_mesh.med"]
    bcs = d.boundary_conditions
    assert [(k, v["type"]) for k, v in bcs.items()][:2] == [
        ("Inlet", "inlet"),
        ("Outlet", "outlet"),
    ]
    assert {k for k, v in bcs.items() if v["type"] == "wall"} == {"Hull", "Sail", "Fins"}
    assert bcs["Inlet"]["norm_formula"] == "u_norm = UX;" and bcs["Inlet"]["direction"] == [1, 0, 0]
    assert bcs["Hull"]["selection"] == "hull_surface" and bcs["Hull"]["zone_id"] == 7

    # the log reports 459378 cells; the half model spans z >= 0
    (zone,) = d.zones
    assert zone.name == "Fluid domain" and zone.n_points == 533377
    assert zone.cell_types == {"hexahedron": 402503, "wedge": 11053, "polyhedron": 45822}
    s = rep.first()
    assert s.metadata["solver"] == "Code_Saturne" and s.points is not None
    np.testing.assert_allclose(s.points.min(0), [-6.4, -6.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(s.points.max(0), [15.0, 6.5, 6.5], atol=1e-6)
    velocity = s.fields["Velocity"]
    assert velocity.location == "cell" and velocity.shape == (459378, 3)
    # most of the domain is free stream at the inlet speed UX, along +x
    np.testing.assert_allclose(np.median(velocity.data, axis=0), [ux, 0, 0], atol=0.05 * ux)
    assert s.fields["Wall_distance"].data.min() > 0 and s.fields["k"].data.min() > 0
    assert {"Pressure", "omega", "TurbVisc", "total_pressure"} <= set(s.fields)

    # the boundary writer: nine zones, numbered as the <boundary name="..."> of setup.xml
    boundary = next(iter(c.read(post / "RESULTS_BOUNDARY.case")))
    assert boundary.kind == "surface" and boundary.zone == "Boundary"
    ids = np.unique(boundary.fields["boundary_zone_id"].data)
    assert ids.tolist() == sorted(v["zone_id"] for v in bcs.values())
    assert c.describe(post / "RESULTS_BOUNDARY.case").zones[0].n_cells == 61619

    # the user-defined surface mesh: the SUBOFF hull is 4.356 m long
    body_desc = c.describe(post / "RESULTS_SUBOFF_BODY.case")
    assert body_desc.physics["turbulence_model"] == "k-omega-SST"
    body = next(iter(c.read(post / "RESULTS_SUBOFF_BODY.case")))
    assert body.points is not None
    assert body.points[:, 0].max() - body.points[:, 0].min() == pytest.approx(4.356, abs=2e-3)
    yplus = body.fields["Yplus"].data
    assert 10 < np.median(yplus) < 100 and body.fields["Shear_Stress"].shape[1] == 3

    # every level of the real layout resolves to the same run
    for path in (case_dir / "RESU", post.parent, post, post / "RESULTS_FLUID_DOMAIN.case"):
        det = c.detect(path)
        assert det is not None and det.confidence > 0.95, path
