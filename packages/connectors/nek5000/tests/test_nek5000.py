# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_nek5000.connector import (
    Nek5000Connector,
    parse_rdcode,
    read_header,
    subcell_template,
)

import frontier_connectors as fc
from frontier_connectors.testing import check_connector, fetch_sample

SAMPLES = "https://raw.githubusercontent.com/eX-Mech/pymech-test-data/main/nek/"


def gll(n: int) -> np.ndarray:
    """Gauss-Lobatto-Legendre nodes on [-1, 1] (roots of (1 - x^2) P'_{n-1})."""
    if n == 2:
        return np.array([-1.0, 1.0])
    inner = np.polynomial.legendre.Legendre.basis(n - 1).deriv().roots()
    return np.concatenate([[-1.0], np.sort(inner.real), [1.0]])


def box_elements(ne: tuple[int, int, int], lx: int, ndim: int) -> np.ndarray:
    """GLL node coordinates ``(nel, npts, ndim)`` of a box of ``ne`` elements, nodes x-fastest
    inside each element as Nek5000 stores them. The box is sheared so axes are not symmetric."""
    r = 0.5 * (gll(lx) + 1.0)
    out = []
    for ez in range(ne[2] if ndim == 3 else 1):
        for ey in range(ne[1]):
            for ex in range(ne[0]):
                zs = (ez + r) if ndim == 3 else np.zeros(1)
                z, y, x = np.meshgrid(zs, ey + r, ex + r, indexing="ij")
                xyz = [x.ravel() + 0.2 * y.ravel(), 0.5 * y.ravel(), 2.0 * z.ravel()]
                out.append(np.stack(xyz[:ndim], axis=1))
    return np.asarray(out)


def exact(xyz: np.ndarray, t: float) -> dict[str, np.ndarray]:
    x, y = xyz[..., 0], xyz[..., 1]
    z = xyz[..., 2] if xyz.shape[-1] == 3 else np.zeros_like(x)
    u = np.stack([1.0 + y + t, 2.0 * x - z, 0.25 * z + x * y], axis=-1)[..., : xyz.shape[-1]]
    return {
        "U": u,
        "P": (x - 2.0 * y + 3.0 * z + t)[..., None],
        "T": (10.0 + x * y + t)[..., None],
        "S1": (x + 100.0)[..., None],
        "S2": (y - 100.0 + t)[..., None],
    }


def write_fld(
    path: Path,
    xyz: np.ndarray,
    *,
    rdcode: str = "XUPT",
    time: float = 0.0,
    istep: int = 0,
    wdsize: int = 4,
    endian: str = "<",
    lz1: int | None = None,
    ids: np.ndarray | None = None,
    fid: int = 0,
    nfiles: int = 1,
    nelgt: int | None = None,
    extra: str = "",
) -> Path:
    """Write a field file following the Nek5000 ``#std`` layout (header, endian tag, element
    map, then per variable element-by-element, component-by-component, and for 3-D the
    per-element min/max metadata that readers skip)."""
    nel, npts, ndim = xyz.shape
    lx1 = round(npts ** (1.0 / ndim))
    lz = lz1 if lz1 is not None else (lx1 if ndim == 3 else 1)
    header = (
        f"#std {wdsize:1d} {lx1:2d} {lx1:2d} {lz:2d} {nel:10d} {nelgt or nel:10d} "
        f"{time:20.13E} {istep:9d} {fid:6d} {nfiles:6d} {rdcode}{extra}"
    )
    values = exact(xyz, time)
    blocks: list[np.ndarray] = []
    i = 0
    while i < len(rdcode):
        c = rdcode[i]
        i += 1
        if c == "X":
            blocks.append(xyz)
        elif c == "S":
            n = int(rdcode[i : i + 2])
            i += 2
            blocks.extend(values[f"S{k}"] for k in range(1, n + 1))
        else:
            blocks.append(values[c])
    real = np.dtype(f"{endian}f{wdsize}")
    with open(path, "wb") as f:
        f.write(header.ljust(132).encode("ascii"))
        f.write(np.array([6.54321], dtype=f"{endian}f4").tobytes())
        f.write((np.arange(1, nel + 1) if ids is None else ids).astype(f"{endian}i4").tobytes())
        for b in blocks:  # (nel, npts, comp) -> element, component, node
            f.write(np.ascontiguousarray(b.transpose(0, 2, 1)).astype(real).tobytes())
        if ndim == 3:
            for b in blocks:
                mm = np.stack([b.min(axis=1), b.max(axis=1)], axis=2)
                f.write(mm.astype(f"{endian}f4").tobytes())
    return path


def signed_measure(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a = points[cells[:, 1]] - points[cells[:, 0]]
    b = points[cells[:, 3]] - points[cells[:, 0]]
    if cells.shape[1] == 4:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    c = points[cells[:, 4]] - points[cells[:, 0]]
    return np.einsum("ij,ij->i", np.cross(a, b), c)


@pytest.mark.parametrize(
    ("ndim", "wdsize", "endian"), [(3, 4, "<"), (3, 8, ">"), (2, 8, "<"), (2, 4, ">")]
)
def test_single_dump_values_and_subcells(
    tmp_path: Path, ndim: int, wdsize: int, endian: str
) -> None:
    lx = 4
    xyz = box_elements((2, 2, 2), lx, ndim)
    path = write_fld(
        tmp_path / "box0.f00001", xyz, rdcode="XUPTS02", time=0.5, wdsize=wdsize, endian=endian
    )
    c = Nek5000Connector()
    det = c.detect(path)
    assert det is not None and det.confidence > 0.95
    rep = check_connector(c, path, tmp_path / "out")
    d = rep.description
    assert d.solver == "Nek5000" and d.domain == "cfd" and d.temporal == "steady"
    assert d.physics["polynomial_order"] == lx - 1 and d.physics["dimension"] == ndim
    assert d.metadata["precision"] == ("double" if wdsize == 8 else "single")
    assert d.metadata["byte_order"] == ("little" if endian == "<" else "big")
    assert {f.name: f.components for f in d.fields} == {
        "velocity": ndim, "pressure": 1, "temperature": 1, "scalar_1": 1, "scalar_2": 1,
    }  # fmt: skip
    s = rep.first()
    nel = xyz.shape[0]
    assert s.kind == "mesh" and s.time is None and s.points is not None and s.cells is not None
    assert s.points.shape == (nel * lx**ndim, ndim)
    assert s.cells.shape == (nel * (lx - 1) ** ndim, 2**ndim)
    assert d.zones[0].n_cells == len(s.cells) and d.zones[0].n_points == len(s.points)
    np.testing.assert_allclose(s.points, xyz.reshape(-1, ndim), rtol=1e-6)
    want = exact(xyz.reshape(-1, ndim), 0.5)
    for name, key in [("velocity", "U"), ("pressure", "P"), ("temperature", "T")]:
        np.testing.assert_allclose(s.fields[name].data, want[key], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(s.fields["scalar_2"].data, want["S2"], rtol=1e-6)
    assert s.fields["velocity"].data.dtype == np.float32
    assert set(np.unique(s.cell_types)) == {12 if ndim == 3 else 9}
    # sub-cells are right-handed and tile the element exactly
    measure = signed_measure(s.points.astype(np.float64), s.cells)
    assert measure.min() > 0
    np.testing.assert_allclose(measure.sum(), nel * (0.5 if ndim == 2 else 1.0), rtol=1e-5)
    np.testing.assert_array_equal(
        s.static["element_id"].data[:, 0], np.repeat(np.arange(1, nel + 1), (lx - 1) ** ndim)
    )


def test_subcell_template_matches_vtk_hexahedron_order() -> None:
    t = subcell_template(3, 3, 3)
    assert t.shape == (8, 8)
    np.testing.assert_array_equal(t[0], [0, 1, 4, 3, 9, 10, 13, 12])
    np.testing.assert_array_equal(subcell_template(3, 2, 1), [[0, 1, 4, 3], [1, 2, 5, 4]])
    assert [v.name for v in parse_rdcode("XUPTS03", 2)][-3:] == ["scalar_1", "scalar_2", "scalar_3"]


def _write_series(tmp_path: Path, times: list[float], **kw: Any) -> np.ndarray:
    xyz = box_elements((2, 1, 1), 3, 3)
    for i, t in enumerate(times):
        # like Nek5000, only the first dump carries the mesh
        write_fld(
            tmp_path / f"cav0.f{i + 1:05d}",
            xyz,
            rdcode="XUP" if i == 0 else "UPT",
            time=t,
            istep=10 * i,
            **kw,
        )
    (tmp_path / "cav.nek5000").write_text(
        f" filetemplate: cav%01d.f%05d\n firsttimestep: 1\n numtimesteps: {len(times)}\n"
    )
    return xyz


def test_series_reuses_mesh_and_nek5000_template(tmp_path: Path) -> None:
    times = [0.0, 0.1, 0.25, 0.6]
    xyz = _write_series(tmp_path, times)
    c = Nek5000Connector()
    for entry in (tmp_path / "cav.nek5000", tmp_path / "cav0.f00003", tmp_path):
        det = c.detect(entry)
        assert det is not None and det.connector_id == "nek5000"
    rep = check_connector(c, tmp_path / "cav.nek5000", tmp_path / "out")
    assert rep.description.time.values == times
    # temperature appears only from the second dump on: describe scans every header
    assert [f.name for f in rep.description.fields] == ["velocity", "pressure", "temperature"]
    steps = rep.samples["cav"]
    assert [s.time for s in steps] == times and [s.time_index for s in steps] == [0, 1, 2, 3]
    assert steps[0].points is not None and all(s.points is None for s in steps[1:])
    flat = xyz.reshape(-1, 3)
    for s in steps:
        assert s.time is not None
        np.testing.assert_allclose(s.fields["pressure"].data, exact(flat, s.time)["P"], rtol=1e-6)
    assert "temperature" not in steps[0].fields and "temperature" in steps[1].fields
    assert steps[2].metadata["istep"] == 20

    # a later dump alone still gets its geometry from the first file
    late = list(c.read(tmp_path / "cav0.f00003", times=[0.25], fields=["velocity"]))
    assert len(late) == 1 and late[0].time == 0.25 and list(late[0].fields) == ["velocity"]
    assert late[0].points is not None and late[0].points.shape == flat.shape
    np.testing.assert_allclose(late[0].fields["velocity"].data, exact(flat, 0.25)["U"], rtol=1e-6)
    assert [s.time for s in c.read(tmp_path, times="last")] == [0.6]
    assert fc.detect(tmp_path / "cav0.f00002")[0].connector_id == "nek5000"
    with pytest.raises(fc.SelectionError, match="available zones: fluid"):
        list(c.read(tmp_path, zones=["solid"]))

    # the template restricts the series to the listed dumps
    (tmp_path / "cav.nek5000").write_text(
        "filetemplate: cav%01d.f%05d\nfirsttimestep: 2\nnumtimesteps: 2\n"
    )
    with pytest.raises(fc.ConnectorError, match="contains the mesh"):
        list(c.read(tmp_path / "cav.nek5000"))


def test_single_file_option_and_moving_mesh(tmp_path: Path) -> None:
    xyz = box_elements((1, 2, 1), 3, 3)
    write_fld(tmp_path / "mv0.f00001", xyz, rdcode="XU", time=0.0)
    write_fld(tmp_path / "mv0.f00002", xyz, rdcode="XU", time=1.0)  # mesh repeated, unchanged
    write_fld(tmp_path / "mv0.f00003", xyz + 0.5, rdcode="XU", time=2.0)
    c = Nek5000Connector()
    steps = list(c.read(tmp_path / "mv0.f00001"))
    assert [s.time for s in steps] == [0.0, 1.0, 2.0]
    assert steps[1].points is None and steps[2].points is not None
    np.testing.assert_allclose(steps[2].points, xyz.reshape(-1, 3) + 0.5, rtol=1e-6)
    one = list(c.read(tmp_path / "mv0.f00003", options={"series": False}))
    assert len(one) == 1 and one[0].time is None
    assert c.describe(tmp_path / "mv0.f00003", options={"series": False}).temporal == "steady"


def test_merge_points_averages_coincident_gll_nodes(tmp_path: Path) -> None:
    lx = 4
    xyz = box_elements((2, 2, 2), lx, 3)
    path = write_fld(tmp_path / "m0.f00001", xyz, rdcode="XUP", wdsize=8)
    c = Nek5000Connector()
    (s,) = c.read(path, options={"merge_points": True})
    assert s.points is not None and s.cells is not None
    n_side = 2 * (lx - 1) + 1
    assert s.points.shape == (n_side**3, 3) and s.cells.shape == (8 * (lx - 1) ** 3, 8)
    assert signed_measure(s.points, s.cells).min() > 0
    np.testing.assert_allclose(s.fields["pressure"].data, exact(s.points, 0.0)["P"], atol=1e-6)
    rep = check_connector(c, path, tmp_path / "out", options={"merge_points": True})
    assert rep.first().metadata["merged_points"] is True


def test_split_dump_in_output_directories(tmp_path: Path) -> None:
    xyz = box_elements((2, 2, 1), 3, 3)
    ids = np.arange(1, 5)
    for step, t in enumerate([0.0, 1.0], start=1):
        for fid, sel in enumerate([[0, 2], [3, 1]]):
            d = tmp_path / f"A{fid + 1:02d}"
            d.mkdir(exist_ok=True)
            # the second dump lists the elements of each part in another order
            sel = sel if step == 1 else sel[::-1]
            write_fld(
                d / f"sp{fid}.f{step:05d}",
                xyz[sel],
                rdcode="XP" if step == 1 else "P",
                time=t,
                ids=ids[sel],
                fid=fid,
                nfiles=2,
                nelgt=4,
            )
    c = Nek5000Connector()
    rep = check_connector(c, tmp_path / "A01" / "sp0.f00001", tmp_path / "out")
    assert rep.description.metadata["files_per_dump"] == 2
    assert rep.description.zones[0].n_points == 4 * 27
    a, b = rep.samples["sp"]
    assert a.points is not None
    np.testing.assert_allclose(a.points, xyz[[0, 2, 3, 1]].reshape(-1, 3), rtol=1e-6)
    np.testing.assert_allclose(b.fields["pressure"].data, exact(a.points, 1.0)["P"], rtol=1e-5)

    (tmp_path / "A02" / "sp1.f00002").unlink()
    with pytest.raises(fc.ConnectorError, match="split into 2 files"):
        list(c.read(tmp_path / "A01" / "sp0.f00001"))


def test_malformed_and_unsupported_files(tmp_path: Path) -> None:
    c = Nek5000Connector()
    xyz = box_elements((2, 1, 1), 3, 3)
    good = write_fld(tmp_path / "t0.f00001", xyz, rdcode="XUP")
    raw = good.read_bytes()

    cut = tmp_path / "cut0.f00001"
    cut.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(fc.ConnectorError, match="truncated") as e:
        c.describe(cut)
    assert e.value.hint

    bad_tag = tmp_path / "tag0.f00001"
    bad_tag.write_bytes(raw[:132] + b"\x00\x00\x00\x00" + raw[136:])
    with pytest.raises(fc.ConnectorError, match="endian tag"):
        list(c.read(bad_tag))

    bad_code = tmp_path / "code0.f00001"
    bad_code.write_bytes(raw[:132].replace(b"XUP", b"XQP") + raw[132:])
    with pytest.raises(fc.ConnectorError, match="unknown variable code"):
        c.describe(bad_code)

    garbage = tmp_path / "junk0.f00001"
    garbage.write_bytes(b"#std not a header" + b" " * 200)
    with pytest.raises(fc.ConnectorError, match="malformed header"):
        c.describe(garbage)
    assert c.detect(tmp_path / "missing0.f00001") is None
    (tmp_path / "notes.txt").write_text("hello")
    assert c.detect(tmp_path / "notes.txt") is None

    pn2 = write_fld(tmp_path / "pn0.f00001", xyz, rdcode="XUP", extra="   0.0000000E+00 T")
    assert read_header(pn2).press_mesh
    with pytest.raises(fc.UnsupportedSourceError, match="PN-2"):
        c.describe(pn2)

    legacy = tmp_path / "old.fld"
    legacy.write_text("    2  3  3  3  0.1000000E+01    10  XUP\n" + "0.0 " * 50)
    det = c.detect(legacy)
    assert det is not None and det.confidence < 0.5 and det.needs
    with pytest.raises(fc.UnsupportedSourceError, match="legacy"):
        c.describe(legacy)


def test_vtk_nek5000_reader_agrees_on_synthetic_file(tmp_path: Path) -> None:
    """Independent check of the test writer and of this reader: VTK ships its own Nek5000
    reader (BSD), which must see the same points and values in the same file."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    try:
        from vtkmodules.vtkIOParallel import vtkNek5000Reader
    except ImportError:
        pytest.skip("this VTK build has no vtkNek5000Reader")
    xyz = box_elements((2, 2, 1), 4, 3)
    for i, t in enumerate([0.0, 0.5]):
        write_fld(tmp_path / f"v0.f{i + 1:05d}", xyz, rdcode="XUPT", time=t)
    meta = tmp_path / "v.nek5000"
    meta.write_text("filetemplate: v%01d.f%05d\nfirsttimestep: 1\nnumtimesteps: 2\n")
    r: Any = vtkNek5000Reader()
    r.SetFileName(str(meta))
    r.UpdateInformation()
    r.SetPointArrayStatus("Velocity", 1)
    r.SetPointArrayStatus("Pressure", 1)
    r.UpdateTimeStep(0.5)
    out = r.GetOutput()
    vp = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    assert len(vp) > 0
    ours = list(Nek5000Connector().read(meta, times=[0.5]))[0]
    assert ours.points is not None
    mine = ours.points.astype(np.float64)

    def keyed(points: np.ndarray, values: np.ndarray) -> dict[tuple[float, ...], np.ndarray]:
        return {tuple(np.round(p, 4)): v for p, v in zip(points.tolist(), values, strict=True)}

    pd = out.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    assert "Pressure" in names and "Velocity" in names
    theirs_p = keyed(vp, vtk_to_numpy(out.GetPointData().GetArray("Pressure")))
    theirs_u = keyed(vp, vtk_to_numpy(out.GetPointData().GetArray("Velocity")))
    ours_p = keyed(mine, ours.fields["pressure"].data[:, 0])
    ours_u = keyed(mine, ours.fields["velocity"].data)
    assert set(theirs_p) == set(ours_p)
    for k, v in theirs_p.items():
        assert abs(float(v) - float(ours_p[k])) < 1e-5
        np.testing.assert_allclose(theirs_u[k], ours_u[k], atol=1e-5)


@pytest.mark.network
def test_public_nek5000_files(tmp_path: Path) -> None:
    """Real Nek5000 dumps from the pymech test-data repository (data only; GPL-3.0 repo)."""
    c = Nek5000Connector()
    path = fetch_sample(SAMPLES + "channel3D_0.f00001")
    rep = check_connector(c, path, tmp_path / "channel")
    d = rep.description
    assert d.metadata["elements"] == 512 and d.metadata["gll_points_per_element"] == [8, 8, 8]
    assert [f.name for f in d.fields] == ["velocity", "pressure"]
    s = rep.first()
    assert s.points is not None and s.cells is not None
    assert s.points.shape == (512 * 512, 3) and s.cells.shape == (512 * 343, 8)
    # turbulent channel: x in [0, 2 pi], walls at y = +-1, z in [0, pi]
    np.testing.assert_allclose(s.points.min(axis=0), [0.0, -1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(s.points.max(axis=0), [2 * np.pi, 1.0, np.pi], atol=1e-5)
    assert signed_measure(s.points.astype(np.float64), s.cells).min() > 0
    np.testing.assert_allclose(
        signed_measure(s.points.astype(np.float64), s.cells).sum(), 4 * np.pi**2, rtol=1e-4
    )
    u = s.fields["velocity"].data
    wall = np.abs(np.abs(s.points[:, 1]) - 1.0) < 1e-6
    assert wall.sum() > 1000 and np.abs(u[wall]).max() < 1e-6  # no-slip walls
    assert 0.5 < float(u[:, 0].mean()) < 1.3

    # 2-D double-precision statistics file with 44 passive scalars (rdcode XS44)
    stats = fetch_sample(SAMPLES + "stsabl0.f00001")
    h = read_header(stats)
    assert (h.wdsize, h.lx1, h.lz1, h.nelt, h.rdcode) == (8, 12, 1, 120, "XS44")
    rep2 = check_connector(c, stats, tmp_path / "stats", fields=["scalar_1", "scalar_44"])
    s2 = rep2.first()
    assert s2.points is not None and s2.cells is not None
    assert s2.points.shape == (120 * 144, 2) and s2.cells.shape == (120 * 121, 4)
    assert len(rep2.description.fields) == 44 and set(s2.fields) == {"scalar_1", "scalar_44"}
    assert signed_measure(s2.points, s2.cells).min() > 0
    merged = list(c.read(stats, fields=["scalar_1"], options={"merge_points": True}))[0]
    assert merged.points is not None and len(merged.points) < len(s2.points)


@pytest.mark.network
def test_public_file_matches_pymech_oracle() -> None:
    """Cross-check against pymech (GPL-3.0): an external oracle only, never a dependency.
    Runs in a throwaway environment that happens to have pymech installed."""
    pymech = pytest.importorskip("pymech")
    path = fetch_sample(SAMPLES + "phill0.f00010")
    ref = pymech.readnek(str(path))
    (s,) = Nek5000Connector().read(path)
    assert s.points is not None
    n = ref.nel
    lx, ly, lz = (int(v) for v in ref.lr1)
    pts = s.points.reshape(n, lz, ly, lx, 3)
    vel = s.fields["velocity"].data.reshape(n, lz, ly, lx, 3)
    pres = s.fields["pressure"].data.reshape(n, lz, ly, lx)
    assert s.metadata["nek_time"] == pytest.approx(ref.time)
    # this reader keeps the file's element order; pymech sorts by global element id
    gids = s.static["element_id"].data[:: (lx - 1) * (ly - 1) * (lz - 1), 0]
    assert sorted(gids.tolist()) == list(range(1, n + 1))
    for e in (0, 1, n // 2, n - 1):
        el = ref.elem[int(gids[e]) - 1]
        np.testing.assert_allclose(pts[e].transpose(3, 0, 1, 2), el.pos, rtol=1e-12)
        np.testing.assert_allclose(vel[e].transpose(3, 0, 1, 2), el.vel, rtol=1e-6)
        np.testing.assert_allclose(pres[e], el.pres[0], rtol=1e-6, atol=1e-7)
