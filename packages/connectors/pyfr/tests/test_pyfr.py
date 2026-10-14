# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from frontier_connector_pyfr.connector import (
    PyfrConnector,
    corner_indices,
    shape_interpolation,
    std_element,
    tensor_points,
)

import frontier_connectors as fc
from frontier_connectors.testing import check_connector, fetch_sample

UUID = "0f0e0d0c-1111-2222-3333-444455556666"
GAMMA = 1.4
CASES = "https://raw.githubusercontent.com/PyFR/PyFR-Test-Cases/main/"

CONFIG = """[backend]
precision = double

[constants]
gamma = 1.4
Re = 1600
M = 0.4

[solver]
system = {system}
order = {order}

[solver-time-integrator]
scheme = rk45
tstart = 0.0

[solver-interfaces]
riemann-solver = rusanov

[solver-elements-quad]
soln-pts = {rule}

[solver-elements-hex]
soln-pts = {rule}

[solver-elements-tri]
soln-pts = williams-shunn

[soln-bcs-wall]
type = no-slp-adia-wall
"""


def warp(ref: np.ndarray, centre: np.ndarray, order: int) -> np.ndarray:
    """Polynomial map reference -> physical of degree <= ``order`` in the element's own space
    (affine for order 1), so the exact image of any reference point is known independently of
    the reader's shape functions."""
    d = ref.shape[-1]
    out = 0.5 * ref + centre
    if order >= 2:
        bend = 0.05 * ref[..., 0] ** 2 if d == 2 else 0.05 * ref[..., 0] * ref[..., 0]
        out = out.copy()
        out[..., 1] += bend
    return out


def primitive(xyz: np.ndarray, t: float) -> dict[str, np.ndarray]:
    x, y = xyz[..., 0], xyz[..., 1]
    z = xyz[..., 2] if xyz.shape[-1] == 3 else np.zeros_like(x)
    return {
        "rho": 1.0 + 0.1 * x + 0.01 * t,
        "u": y + t,
        "v": -x,
        "w": 0.5 * z,
        "p": 2.0 + 0.3 * x * y + 0.1 * z,
    }


def conservative(xyz: np.ndarray, t: float) -> np.ndarray:
    """``(..., nvars)`` rho, rho*u, rho*v[, rho*w], E."""
    d = xyz.shape[-1]
    q = primitive(xyz, t)
    vel = [q["u"], q["v"], q["w"]][:d]
    energy = q["p"] / (GAMMA - 1.0) + 0.5 * q["rho"] * sum(v * v for v in vel)
    return np.stack([q["rho"], *[q["rho"] * v for v in vel], energy], axis=-1)


def reference_points(etype: str, n: int = 4) -> np.ndarray:
    """A few interior reference points per element type (stand-ins for PyFR's solution
    points; the v3 layout stores them in the ``pts`` attribute)."""
    if etype in ("quad", "hex"):
        return tensor_points(etype, 3 ** (2 if etype == "quad" else 3), "gauss-legendre")
    rng = np.random.default_rng(len(etype))
    d = 2 if etype == "tri" else 3
    bary = rng.dirichlet(np.ones(3 if etype in ("tri", "pri") else 4), size=n)
    corners = std_element(etype if etype != "pri" else "tri", 1)[:, : 2 if etype == "pri" else d]
    pts = bary @ corners
    if etype == "pri":
        pts = np.concatenate([pts, rng.uniform(-0.9, 0.9, size=(n, 1))], axis=1)
    return pts


def element_centres(ne: int, d: int, offset: float) -> np.ndarray:
    c = np.zeros((ne, d))
    c[:, 0] = np.arange(ne) * 1.5 + offset
    c[:, 1] = offset
    return c


class Case:
    """Elements of several types, each a warped copy of the reference element."""

    def __init__(self, spec: dict[str, tuple[int, int]]) -> None:
        self.spec = spec  # etype -> (n elements, geometry order)
        self.ndims = 2 if next(iter(spec)) in ("quad", "tri") else 3
        self.centres = {
            et: element_centres(ne, self.ndims, 10.0 * i)
            for i, (et, (ne, _)) in enumerate(spec.items())
        }
        self.upts = {et: reference_points(et) for et in spec}

    def shape_points(self, et: str) -> np.ndarray:
        order = self.spec[et][1]
        return warp(std_element(et, order)[None], self.centres[et][:, None, :], order)

    def soln_points(self, et: str) -> np.ndarray:
        return warp(self.upts[et][None], self.centres[et][:, None, :], self.spec[et][1])

    def all_points(self) -> np.ndarray:
        # PyFR (and the reader) order element types alphabetically
        return np.concatenate(
            [self.soln_points(et).reshape(-1, self.ndims) for et in sorted(self.spec)]
        )


def stats(t: float, ndims: int, fields: str | None = None) -> str:
    names = fields or ("rho,rhou,rhov,rhow,E" if ndims == 3 else "rho,rhou,rhov,E")
    return (
        f"[data]\nfields = {names}\nprefix = soln\n\n"
        f"[solver-time-integrator]\ntcurr = {t!r}\nnsteps = 10\n"
    )


def write_mesh_v3(path: Path, case: Case, uuid: str = UUID) -> Path:
    """PyFR 3 mesh layout: shared ``nodes`` table plus one compound ``eles/<etype>`` array."""
    locs: list[np.ndarray] = []
    with h5py.File(path, "w") as f:
        offset = 0
        codec = [b"bc/wall"]
        for et in case.spec:
            spts = case.shape_points(et)
            ne, ns, _ = spts.shape
            nfaces = {"quad": 4, "tri": 3, "hex": 6, "tet": 4, "pri": 5}[et]
            dt = np.dtype(
                [
                    ("nodes", "<i8", (ns,)),
                    ("curved", "?"),
                    ("faces", [("cidx", "<i2"), ("off", "<i8")], (nfaces,)),
                ]
            )
            eles = np.zeros(ne, dtype=dt)
            eles["nodes"] = offset + np.arange(ne * ns).reshape(ne, ns)
            eles["curved"] = case.spec[et][1] > 1
            eles["faces"]["off"] = -1
            f[f"eles/{et}"] = eles
            locs.append(spts.reshape(ne * ns, -1))
            offset += ne * ns
            codec += [f"eles/{et}/{i}".encode() for i in range(nfaces)]
        loc = np.concatenate(locs)
        nodes = np.zeros(len(loc), dtype=[("location", "<f8", (case.ndims,)), ("valency", "<u2")])
        nodes["location"] = loc
        f["nodes"] = nodes
        f["codec"] = np.array(codec, dtype="S")
        f["creator"] = np.array("pyfr 3.1", dtype="S")
        f["mesh-uuid"] = np.array(uuid, dtype="S")
        f["version"] = 1
    return path


def write_soln_v3(
    path: Path, case: Case, t: float, *, system: str = "euler", uuid: str = UUID
) -> Path:
    with h5py.File(path, "w") as f:
        f["config"] = np.array(CONFIG.format(system=system, order=2, rule="gauss-legendre"), "S")
        f["stats"] = np.array(stats(t, case.ndims), dtype="S")
        f["mesh-uuid"] = np.array(uuid, dtype="S")
        f["creator"] = np.array("pyfr 3.1", dtype="S")
        f["version"] = 1
        g = f.create_group("soln")
        for et in case.spec:
            data = conservative(case.soln_points(et), t)  # (neles, nupts, nvars)
            ds = g.create_dataset(f"p2-{et}", data=data.transpose(0, 2, 1).astype(np.float32))
            ds.attrs["pts"] = case.upts[et]
            g.create_dataset(f"p2-{et}-parts", data=np.zeros(len(data), dtype=np.int32))
    return path


def write_legacy(tmp: Path, case: Case, times: list[float], rule: str, parts: int = 2) -> None:
    """PyFR <= 2.1 layout: per-partition ``spt_``/``soln_`` arrays, no stored solution points."""
    with h5py.File(tmp / "mesh.pyfrm", "w") as f:
        f["mesh_uuid"] = np.array(UUID, dtype="S")
        for et in case.spec:
            spts = case.shape_points(et)
            for p, chunk in enumerate(np.array_split(spts, parts)):
                f[f"spt_{et}_p{p}"] = chunk.transpose(1, 0, 2)  # (nspts, neles, ndims)
        f["bcon_wall_p0"] = np.zeros(2, dtype=[("f0", "S4"), ("f1", "<i4"), ("f2", "i1")])
    for t in times:
        with h5py.File(tmp / f"flow-{t:.2f}.pyfrs", "w") as f:
            f["config"] = np.array(CONFIG.format(system="navier-stokes", order=2, rule=rule), "S")
            f["stats"] = np.array(stats(t, case.ndims), dtype="S")
            f["mesh_uuid"] = np.array(UUID, dtype="S")
            for et in case.spec:
                upts = tensor_points(et, 3**case.ndims, rule)
                phys = warp(upts[None], case.centres[et][:, None, :], case.spec[et][1])
                data = conservative(phys, t)  # (neles, nupts, nvars)
                for p, chunk in enumerate(np.array_split(data, parts)):
                    f[f"soln_{et}_p{p}"] = chunk.transpose(1, 2, 0)  # (nupts, nvars, neles)


def test_shape_functions_and_corners() -> None:
    for et, order in [("quad", 3), ("hex", 2), ("tri", 3), ("tet", 2), ("pri", 2), ("pyr", 2)]:
        std = std_element(et, order)
        if et != "pyr":
            # nodal property: interpolating at the shape points gives the identity
            m = shape_interpolation(et, len(std), std)
            np.testing.assert_allclose(m, np.eye(len(std)), atol=1e-9)
            np.testing.assert_allclose(shape_interpolation(et, len(std), std[:3] * 0.3).sum(1), 1)
        corners = std[corner_indices(et, order)]
        assert np.abs(corners).max() == 1.0 and len(np.unique(corners, axis=0)) == len(corners)
    np.testing.assert_allclose(
        std_element("hex", 1)[corner_indices("hex", 1)][:4, :2],
        [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    )
    with pytest.raises(fc.UnsupportedSourceError, match="pyr"):
        shape_interpolation("pyr", 5, np.zeros((1, 3)))


def test_v3_transient_mixed_2d_primitive(tmp_path: Path) -> None:
    case = Case({"quad": (3, 2), "tri": (4, 2)})
    write_mesh_v3(tmp_path / "wing.pyfrm", case)
    times = [0.0, 0.5, 1.25]
    for t in times:
        write_soln_v3(tmp_path / f"wing-{t:.2f}.pyfrs", case, t)
    c = PyfrConnector()
    rep = check_connector(c, tmp_path / "wing-0.50.pyfrs", tmp_path / "out")
    d = rep.description
    assert d.solver == "PyFR" and d.solver_version == "3.1" and d.domain == "cfd"
    assert d.time.values == times and d.format == "pyfr-v3"
    assert d.physics["system"] == "euler" and d.physics["polynomial_order"] == 2
    assert d.parameters == {"gamma": 1.4, "Re": 1600.0, "M": 0.4}
    assert d.boundary_conditions == {"wall": {"type": "no-slp-adia-wall"}}
    assert [f.name for f in d.fields] == ["rho", "u", "v", "p"]
    assert d.zones[0].kind == "point_cloud" and d.zones[0].cell_types == {"quad": 3, "tri": 4}
    assert d.metadata["stored_fields"] == ["rho", "rhou", "rhov", "E"]
    steps = rep.samples["wing"]
    assert [s.time for s in steps] == times and steps[0].kind == "point_cloud"
    assert steps[0].points is not None and steps[1].points is None
    want = case.all_points()
    np.testing.assert_allclose(steps[0].points, want, atol=1e-12)
    for s in steps:
        assert s.time is not None
        q = primitive(want, s.time)
        for name in ("rho", "u", "v", "p"):
            np.testing.assert_allclose(s.fields[name].data[:, 0], q[name], rtol=2e-5, atol=2e-5)
    assert set(np.unique(steps[0].static["element_type"].data)) == {5, 9}

    raw = list(c.read(tmp_path, times="last", fields=["rhou", "E"], options={"primitive": False}))
    assert len(raw) == 1 and raw[0].time == 1.25 and list(raw[0].fields) == ["rhou", "E"]
    np.testing.assert_allclose(
        raw[0].fields["E"].data[:, 0], conservative(want, 1.25)[:, 3], rtol=1e-6
    )
    one = list(c.read(tmp_path / "wing-0.50.pyfrs", options={"series": False}))
    assert len(one) == 1 and one[0].time is None and one[0].points is not None
    assert fc.detect(tmp_path)[0].connector_id == "pyfr"
    with pytest.raises(fc.SelectionError, match="solution_points"):
        list(c.read(tmp_path, zones=["wall"]))


def test_v3_3d_elements_and_incompressible_fields(tmp_path: Path) -> None:
    case = Case({"hex": (2, 2), "tet": (3, 1), "pri": (2, 2)})
    write_mesh_v3(tmp_path / "m.pyfrm", case)
    path = write_soln_v3(tmp_path / "s.pyfrs", case, 0.25, system="ac-euler")
    with h5py.File(path, "a") as f:  # artificial compressibility stores p, u, v, w directly
        del f["stats"]
        f["stats"] = np.array(stats(0.25, 3, "p,u,v,w,extra"), dtype="S")
    c = PyfrConnector()
    rep = check_connector(c, path, tmp_path / "out")
    assert rep.description.temporal == "steady" and rep.description.metadata["time_value"] == 0.25
    assert [f.name for f in rep.description.fields] == ["p", "u", "v", "w", "extra"]
    assert rep.description.physics["compressible"] is False
    s = rep.first()
    assert s.points is not None
    np.testing.assert_allclose(s.points, case.all_points(), atol=1e-12)
    # no conversion for a non-conservative system: columns come back as stored
    np.testing.assert_allclose(
        s.fields["u"].data[:, 0], conservative(case.all_points(), 0.25)[:, 1], rtol=1e-6
    )


@pytest.mark.parametrize("rule", ["gauss-legendre", "gauss-legendre-lobatto"])
def test_legacy_layout_partitions_and_time_series(tmp_path: Path, rule: str) -> None:
    case = Case({"hex": (5, 2)})
    times = [0.0, 0.1, 0.2]
    write_legacy(tmp_path, case, times, rule)
    c = PyfrConnector()
    rep = check_connector(c, tmp_path / "flow-0.00.pyfrs", tmp_path / "out")
    d = rep.description
    assert d.format == "pyfr-legacy" and d.solver_version is None and d.time.values == times
    assert d.physics["viscous"] is True and d.zones[0].n_points == 5 * 27
    steps = rep.samples["flow"]
    upts = tensor_points("hex", 27, rule)
    want = warp(upts[None], case.centres["hex"][:, None, :], 2).reshape(-1, 3)
    assert steps[0].points is not None
    np.testing.assert_allclose(steps[0].points, want, atol=1e-12)
    for s in steps:
        assert s.time is not None
        q = primitive(want, s.time)
        for name in ("rho", "u", "v", "w", "p"):
            np.testing.assert_allclose(s.fields[name].data[:, 0], q[name], rtol=1e-5, atol=1e-5)


def test_legacy_simplex_solution_points_are_refused(tmp_path: Path) -> None:
    case = Case({"tri": (2, 1)})
    with h5py.File(tmp_path / "t.pyfrm", "w") as f:
        f["mesh_uuid"] = np.array(UUID, dtype="S")
        f["spt_tri_p0"] = case.shape_points("tri").transpose(1, 0, 2)
    with h5py.File(tmp_path / "t.pyfrs", "w") as f:
        f["config"] = np.array(CONFIG.format(system="euler", order=1, rule="gauss-legendre"), "S")
        f["stats"] = np.array(stats(0.0, 2), dtype="S")
        f["mesh_uuid"] = np.array(UUID, dtype="S")
        f["soln_tri_p0"] = np.ones((3, 4, 2))
    c = PyfrConnector()
    assert c.describe(tmp_path / "t.pyfrs").zones[0].n_points == 6
    with pytest.raises(fc.UnsupportedSourceError, match="tabulated rules") as e:
        list(c.read(tmp_path / "t.pyfrs"))
    assert "pyfr export" in (e.value.hint or "")


def _measure(points: np.ndarray, cells: np.ndarray, vtk_type: int) -> np.ndarray:
    p = points[cells[:, 0]]
    a, b = points[cells[:, 1]] - p, points[cells[:, 2 if vtk_type != 9 else 3]] - p
    if points.shape[1] == 2:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    top = {10: 3, 12: 4, 13: 3, 14: 4}[vtk_type]
    if vtk_type == 12:
        b = points[cells[:, 3]] - p
    return np.einsum("ij,ij->i", np.cross(a, b), points[cells[:, top]] - p)


def test_mesh_files_become_linear_meshes(tmp_path: Path) -> None:
    c = PyfrConnector()
    case = Case({"hex": (2, 2), "tet": (3, 1), "pri": (2, 2)})
    path = write_mesh_v3(tmp_path / "m3.pyfrm", case)
    rep = check_connector(c, path, tmp_path / "out3")
    d = rep.description
    assert d.zones[0].cell_types == {"hexahedron": 2, "tetra": 3, "wedge": 2}
    assert d.metadata["geometry_order"] == {"hex": 2, "tet": 1, "pri": 2}
    assert list(d.boundary_conditions) == ["wall"] and d.warnings
    s = rep.first()
    assert s.kind == "mesh" and s.points is not None and s.cells is not None
    assert s.cell_types is not None and s.cells.shape == (7, 8)
    assert len(s.points) == 2 * 8 + 3 * 4 + 2 * 6  # only the corner nodes are kept
    for vt in (10, 12, 13):
        sel = s.cell_types == vt
        n = {10: 4, 12: 8, 13: 6}[vt]
        assert (s.cells[sel][:, n:] == -1).all()
        assert _measure(s.points, s.cells[sel][:, :n], vt).min() > 0

    case2 = Case({"quad": (4, 1)})
    write_legacy(tmp_path, case2, [], "gauss-legendre")
    rep2 = check_connector(c, tmp_path / "mesh.pyfrm", tmp_path / "out2")
    s2 = rep2.first()
    assert rep2.description.format == "pyfr-legacy" and rep2.description.metadata["partitions"] == 2
    assert s2.points is not None and s2.cells is not None and s2.cells.shape == (4, 4)
    assert _measure(s2.points, s2.cells, 9).min() > 0
    np.testing.assert_allclose(_measure(s2.points, s2.cells, 9), 1.0)


def test_missing_mesh_wrong_mesh_and_corrupt_files(tmp_path: Path) -> None:
    case = Case({"quad": (2, 1)})
    c = PyfrConnector()
    soln = write_soln_v3(tmp_path / "a.pyfrs", case, 0.0)
    desc = c.describe(soln)
    assert any("no matching .pyfrm" in w for w in desc.warnings)
    with pytest.raises(fc.ConnectorError, match="no .pyfrm with mesh uuid") as e:
        list(c.read(soln))
    assert "mesh" in (e.value.hint or "")

    other = tmp_path / "elsewhere"
    other.mkdir()
    mesh = write_mesh_v3(other / "other.pyfrm", case, uuid="ffffffff-0000-0000-0000-000000000000")
    with pytest.raises(fc.ConnectorError, match="does not belong"):
        list(c.read(soln, options={"mesh": str(mesh)}))
    good = write_mesh_v3(other / "good.pyfrm", case)
    (s,) = c.read(soln, options={"mesh": str(good)})
    assert s.points is not None and len(s.points) == 2 * 9

    cut = tmp_path / "cut.pyfrs"
    cut.write_bytes(soln.read_bytes()[:700])
    assert c.detect(cut) is None
    with pytest.raises(fc.ConnectorError, match="not a readable HDF5") as e2:
        c.describe(cut)
    assert e2.value.hint

    alien = tmp_path / "alien.pyfrs"
    with h5py.File(alien, "w") as f:
        f["something"] = np.arange(3)
    det = c.detect(alien)
    assert det is not None and det.confidence < 0.5
    with pytest.raises(fc.ConnectorError, match="neither the PyFR legacy"):
        c.describe(alien)
    assert c.detect(tmp_path / "nope.pyfrs") is None

    sub = tmp_path / "sub"
    sub.mkdir()
    write_mesh_v3(sub / "m.pyfrm", case)
    part = write_soln_v3(sub / "p.pyfrs", case, 0.0)
    with h5py.File(part, "a") as f:
        f["soln/p2-quad-idxs"] = np.arange(2)
    with pytest.raises(fc.UnsupportedSourceError, match="region subset"):
        list(c.read(part))


def _pyfr(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, PyFR from the test environment
        [sys.executable, "-m", "pyfr", *args], cwd=cwd, capture_output=True, text=True, timeout=600
    )


@pytest.mark.network
def test_files_written_by_real_pyfr(tmp_path: Path) -> None:
    """Opt-in end-to-end check against PyFR itself (BSD-3, external tool, never a dependency):
    import a public CC-BY test-case mesh, run two tiny steps with initial conditions equal to
    the coordinates, and require the reader's solution-point positions to match the stored
    velocity (both layouts: the installed PyFR decides which one is written). Needs a
    throwaway environment with ``pyfr`` and a working backend
    (``FRONTIER_PYFR_BACKEND``, default ``metal`` on macOS, ``openmp`` elsewhere)."""
    pytest.importorskip("pyfr")
    backend = os.environ.get(
        "FRONTIER_PYFR_BACKEND", "metal" if sys.platform == "darwin" else "openmp"
    )
    msh = fetch_sample(CASES + "2d-couette-flow/couette-flow.msh")
    work = tmp_path / "run"
    work.mkdir()
    (work / "c.msh").write_bytes(msh.read_bytes())
    r = _pyfr(["import", "c.msh", "c.pyfrm"], work)
    assert r.returncode == 0, r.stderr[-2000:]
    c = PyfrConnector()
    mesh = check_connector(c, work / "c.pyfrm", tmp_path / "mesh-out").first()
    assert mesh.cells is not None and mesh.cell_types is not None
    assert sorted(np.unique(mesh.cell_types).tolist()) == [5, 9] and len(mesh.cells) == 47

    vortex = fetch_sample(CASES + "2d-euler-vortex/euler-vortex.msh")
    (work / "v.msh").write_bytes(vortex.read_bytes())
    r = _pyfr(["import", "v.msh", "v.pyfrm"], work)
    assert r.returncode == 0, r.stderr[-2000:]
    (work / "v.ini").write_text(
        "[backend]\nprecision = single\n[constants]\ngamma = 1.4\n"
        "[solver]\nsystem = euler\norder = 3\n"
        "[solver-time-integrator]\nscheme = rk4\ncontroller = none\ntstart = 0\n"
        "tend = 2e-6\ndt = 1e-6\n"
        "[solver-interfaces]\nriemann-solver = rusanov\n"
        "[solver-interfaces-line]\nflux-pts = gauss-legendre\n"
        "[solver-elements-quad]\nsoln-pts = gauss-legendre\n"
        "[soln-plugin-writer]\ndt-out = 1e-6\nbasedir = .\nbasename = v-{n:02d}\n"
        "[soln-ics]\nrho = 1\nu = x\nv = y\np = 1000\n"
    )
    r = _pyfr(["run", "-b", backend, "v.pyfrm", "v.ini"], work)
    solns = sorted(work.glob("v-*.pyfrs"))
    if r.returncode != 0 or not solns:
        pytest.skip(f"PyFR backend {backend!r} did not run here: {r.stderr[-300:]}")
    rep = check_connector(c, solns[0], tmp_path / "soln-out", options={"series": False})
    s = rep.first()
    assert s.points is not None and s.points.shape == (400 * 16, 2)
    assert rep.description.physics["system"] == "euler" and rep.description.parameters["gamma"]
    err = np.abs(np.stack([s.fields["u"].data[:, 0], s.fields["v"].data[:, 0]], axis=1) - s.points)
    # the 20 x 20 periodic box makes u = x jump at the border: only the outer ring of elements
    # sees that within two 1e-6 steps, everything else must sit on the coordinates
    inner = (np.abs(s.points) < 9.0).all(axis=1)
    assert inner.sum() == 324 * 16 and err[inner].max() < 2e-4
    np.testing.assert_allclose(s.fields["p"].data[inner, 0], 1000.0, rtol=1e-3)


def test_registry_row() -> None:
    from frontier_connectors import registry

    info: Any = registry.info(PyfrConnector())
    assert info.id == "pyfr" and info.maturity == "verified-synthetic"
