# SPDX-License-Identifier: Apache-2.0
"""Palabos: output directories written by Palabos itself (``plbLog.dat`` + VTK files).

The layout test on generated files is in ``test_vendors.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from frontier_connector_palabos.connector import PalabosConnector, read_plb_log

import frontier_connectors as fc
from frontier_connectors.testing import check_connector, fetch_sample, write_vtk_xml

RAW = "https://raw.githubusercontent.com/"
# Flow around a cylinder in a channel (Palabos tutorial 1.5), 40 states in tmp_5/. Repository
# einsteinzhang/LBM_palabos_Neymar-zhang, no licence file: fetched for testing only, never
# redistributed.
CYLINDER = (
    RAW + "einsteinzhang/LBM_palabos_Neymar-zhang/7a91fd43b034009f21692c8187b56b747998394f/tmp_5/"
)
CYLINDER_FILES = {
    "plbLog.dat": "170cd3f66fceb77d149b832e7f29816344981178b1b8cf22dcd29dd750ad1b03",
    "vtk000000.vti": "e860aef8bbaff0a76c7f852d1bf754a7099b64a9864d0f07968a953294fcf5e2",
    "vtk010000.vti": "9ed06435fccde0aa34457097d0f4e7564d075b0ac0582c48d45e9f042ef3ed4a",
    "vtk039000.vti": "af74e191a0711ed6b944995068faccbb98297efb748c7cc0ec6e28e388fb5ab7",
}
# Couette flow written with VtkStructuredImageOutput2D (.vts). Repository
# xzagvozdina/palabosCouette, no licence file: fetched for testing only.
COUETTE = (
    RAW + "xzagvozdina/palabosCouette/10836c4f3262f55f2b8c33a4b043eab44a2da904/tmp/"
    "free%20slip/periodic/Re%20=%201.5,%20T%20=%20101/"
)
COUETTE_FILES = {
    "plbLog.dat": "772963c8d34c6fc98c3a7bce17a7ce398ae3b10cac75a668bbb15c98b6d67e33",
    "vtk100000.vts": "a39949390c14b047ff39fd99ccd8c93f647b8b36a346a92635b11cdcd835663a",
}


def _output_dir(root: Path, base: str, files: dict[str, str]) -> Path:
    """The downloaded files side by side, as Palabos left them in its output directory."""
    root.mkdir(parents=True)
    for name, sha in files.items():
        (root / name).symlink_to(fetch_sample(base + name, name, sha256=sha))
    return root


def _image(path: Path, iteration: int) -> None:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData

    img = vtkImageData()
    img.SetDimensions(5, 4, 1)
    img.SetSpacing(0.25, 0.25, 0.25)
    a = numpy_to_vtk(np.full(20, 1.0 + 1e-3 * iteration, dtype=np.float32), deep=True)
    a.SetName("density")
    img.GetPointData().AddArray(a)
    write_vtk_xml(img, path)


def test_plb_log_parameters_and_single_state(tmp_path: Path) -> None:
    """The log layout of Palabos' ``writeLogFile`` (as in the real files below) and an output
    directory with one state only."""
    out = tmp_path / "tmp"
    out.mkdir()
    (out / "plbLog.dat").write_text(
        "Poiseuille flow\n\nVelocity in lattice units: u=0.01\n"
        "Reynolds number:           Re=300\nLattice resolution:        N=4\n"
        "Relaxation frequency:      omega=1.9\nExtent of the system:      lx=1\n"
        "Grid spacing deltaX:       dx=0.25\nTime step deltaT:          dt=0.0025\n"
        "Custom value:              tag=abc\nsome free text written through plbLogFile\n"
    )
    title, values = read_plb_log(out / "plbLog.dat")
    assert title == "Poiseuille flow" and values == {
        "u": 0.01, "Re": 300.0, "N": 4.0, "omega": 1.9, "lx": 1.0, "dx": 0.25, "dt": 0.0025
    }  # fmt: skip
    assert read_plb_log(tmp_path / "missing.dat") == ("", {})
    _image(out / "vtk000100.vti", 100)
    c = PalabosConnector()
    d = c.describe(out)
    assert d.temporal == "steady" and d.metadata["title"] == "Poiseuille flow"
    assert d.parameters["Re"] == 300.0 and d.parameters["relaxation_frequency"] == 1.9
    assert "time_per_iteration" not in d.metadata and "one VTK file" in d.metadata["recognised_by"]
    _image(out / "vtk000200.vti", 200)
    rep = check_connector(c, out, tmp_path / "ds", min_confidence=0.95)
    d = rep.description
    assert (d.time.quantity, d.time.values) == ("iteration", [100.0, 200.0])
    assert d.metadata["time_per_iteration"] == 0.0025
    assert [w for w in d.warnings if "dt=0.0025" in w] and len(d.warnings) == 1
    # a log without the parameter block (the file exists in every Palabos output directory)
    (out / "plbLog.dat").write_text("")
    d = c.describe(out)
    assert d.parameters == {} and "title" not in d.metadata


@pytest.mark.network
def test_real_palabos_vti_series_with_log(tmp_path: Path) -> None:
    out = _output_dir(tmp_path / "tmp_5", CYLINDER, CYLINDER_FILES)
    c = PalabosConnector()
    assert fc.detect(out)[0].connector_id == "palabos"
    rep = check_connector(c, out, tmp_path / "out", min_confidence=0.95)
    d = rep.description
    assert (d.solver, d.domain, d.physics) == ("Palabos", "cfd", {"method": "lattice Boltzmann"})
    assert d.metadata["title"] == "Poiseuille flow"
    # plbLog.dat (writeLogFile): every symbol becomes a parameter
    p = d.parameters
    assert (p["Re"], p["resolution"], p["lattice_velocity"]) == (300.0, 299.0, 0.01)
    assert p["relaxation_frequency"] == 1.88715 and (p["lx"], p["ly"], p["lz"]) == (3.0, 1.0, 0.0)
    # dx = 1 / N and dt = u_lb * dx, as Palabos derives them
    assert p["dx"] == pytest.approx(1.0 / p["resolution"], rel=1e-5)
    assert p["dt"] == pytest.approx(p["lattice_velocity"] * p["dx"], rel=1e-5)
    # file numbers are iterations; the log gives the factor to time
    assert (d.time.quantity, d.time.values) == ("iteration", [0.0, 10000.0, 39000.0])
    assert d.metadata["time_per_iteration"] == p["dt"]
    assert any("dt=3.34448e-05" in w for w in d.warnings)
    assert not any("file numbers" in w for w in d.warnings)
    assert [(f.name, f.components) for f in d.fields] == [("density", 1), ("velocity", 2)]

    steps = rep.samples[rep.sample_ids[0]]
    assert [s.time for s in steps] == [0.0, 10000.0, 39000.0] and steps[0].kind == "grid"
    first = steps[0]
    # lx * N + 1 by ly * N + 1 nodes, spaced dx
    assert first.fields["velocity"].shape == (898, 300, 2)
    x, y = first.coords["x"], first.coords["y"]
    assert x[1] - x[0] == pytest.approx(p["dx"], rel=1e-5) and x[-1] == pytest.approx(3.0, 1e-3)
    assert y[-1] == pytest.approx(1.0, rel=1e-3)
    # iteration 0: fluid at rest density 1 with the Poiseuille profile (peak 1) imposed
    rho0, u0 = first.fields["density"].data[..., 0], first.fields["velocity"].data
    assert set(np.unique(rho0).round(6)) <= {0.0, 1.0} and (rho0 == 1.0).mean() > 0.95
    assert u0[..., 0].max() == pytest.approx(1.0, abs=1e-3) and np.abs(u0[..., 1]).max() < 1e-6
    profile = u0[0, :, 0]  # inlet column: 4 y (1 - y)
    yc = np.linspace(0.0, 1.0, len(profile))
    np.testing.assert_allclose(profile, 4.0 * yc * (1.0 - yc), atol=0.02)
    # later the wake sheds: transverse velocity appears and the density stays near 1
    last = steps[-1]
    assert np.abs(last.fields["velocity"].data[..., 1]).max() > 0.3
    fluid = last.fields["density"].data[..., 0]
    assert 0.99 < np.median(fluid[fluid > 0]) < 1.01
    assert last.metadata["solver"] == "Palabos" and last.metadata["route"] == "vtk"


@pytest.mark.network
def test_real_palabos_structured_output_single_state(tmp_path: Path) -> None:
    out = _output_dir(tmp_path / "couette", COUETTE, COUETTE_FILES)
    c = PalabosConnector()
    title, values = read_plb_log(out / "plbLog.dat")
    assert title == "Couette flow" and values["Re"] == 1.5 and values["omega"] == 0.4
    rep = check_connector(c, out, tmp_path / "out", min_confidence=0.95)
    d = rep.description
    # one .vts next to the log is one state, not a numbered family
    assert d.temporal == "steady" and "one VTK file" in d.metadata["recognised_by"]
    assert d.parameters["Re"] == 1.5 and d.parameters["resolution"] == 100.0
    s = rep.first()
    assert s.points is not None and s.n_points == 201 * 101
    np.testing.assert_allclose(s.points.max(0), [2.0, 1.0, 0.0], atol=1e-6)
    u = s.fields["velocity"].data
    # the driven wall moves at velocity 1; the opposite wall is a free-slip wall (the run
    # directory is "free slip"): no flow through it
    top = np.isclose(s.points[:, 1], 1.0)
    bottom = np.isclose(s.points[:, 1], 0.0)
    assert u[top, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert np.abs(u[bottom, 1]).max() < 1e-6 and u[bottom, 0].max() < 1.0
    np.testing.assert_allclose(
        s.fields["velocityNorm"].data[:, 0], np.linalg.norm(u, axis=1), atol=1e-5
    )
