# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import zarr
from frontier_connector_arrays.connector import (
    Hdf5Connector,
    NetcdfConnector,
    ZarrConnector,
)

import frontier_connectors as fc
from frontier_connectors import _arraymap as am
from frontier_connectors import meshutil
from frontier_connectors.cli import main
from frontier_connectors.testing import (
    analytic_fields,
    check_connector,
    fetch_sample,
    hex_grid,
    open_store,
    tet_grid,
    tri_grid,
)

convert_mod = importlib.import_module("frontier_connectors.convert")
# typed Any: samples expose optional arrays and the assertions fail loudly on None anyway
close: Any = np.testing.assert_allclose
equal: Any = np.testing.assert_array_equal
NT, NZ, NY, NX = 5, 2, 3, 4
FILL_AT = (1, 1, 2, 3)  # (time, lev, lat, lon) of the masked value


def _packed() -> np.ndarray:
    """int16 codes on disk in (time, lev, lat, lon) order: 1000 t + 100 k + 10 j + i."""
    t, k, j, i = np.meshgrid(
        np.arange(NT), np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij"
    )
    return (1000 * t + 100 * k + 10 * j + i).astype(np.int16)


def _expected(t: int, i: int, j: int, k: int) -> float:
    return 280.0 + 0.01 * (1000 * t + 100 * k + 10 * j + i)


def _write_cf_nc4(path: Path) -> Path:
    """NetCDF-4 written at the HDF5 level the way the netCDF-C library lays it out: dimension
    scales, ``_Netcdf4Dimid``, ``DIMENSION_LIST``, an unlimited time dimension."""
    with h5py.File(path, "w") as f:
        f.attrs["_NCProperties"] = np.bytes_("version=2,netcdf=4.9.2,hdf5=1.14.3")
        f.attrs["Conventions"] = np.bytes_("CF-1.8")
        f.attrs["title"] = np.bytes_("synthetic CF file")
        time = f.create_dataset("time", data=np.arange(NT) * 6.0, maxshape=(None,))
        time.attrs["units"] = np.bytes_("hours since 2000-01-01 00:00:00")
        time.attrs["calendar"] = np.bytes_("standard")
        lev = f.create_dataset("lev", data=np.array([100.0, 500.0]))
        lev.attrs["units"] = np.bytes_("m")
        lev.attrs["positive"] = np.bytes_("up")
        lat = f.create_dataset("lat", data=np.array([-30.0, 0.0, 30.0]))
        lat.attrs["units"] = np.bytes_("degrees_north")
        lat.attrs["standard_name"] = np.bytes_("latitude")
        lon = f.create_dataset("lon", data=np.array([0.0, 90.0, 180.0, 270.0]))
        lon.attrs["units"] = np.bytes_("degrees_east")
        for dimid, ds in enumerate((time, lev, lat, lon)):
            ds.make_scale(str(ds.name).strip("/"))
            ds.attrs["_Netcdf4Dimid"] = np.int32(dimid)
        # a dimension without a coordinate variable, as netCDF-4 stores it
        bnds = f.create_dataset("bnds", data=np.zeros(2, dtype=">f4"))
        bnds.make_scale("This is a netCDF dimension but not a netCDF variable.         2")
        bnds.attrs["_Netcdf4Dimid"] = np.int32(4)
        data = _packed()
        data[FILL_AT] = -32768
        ta = f.create_dataset("ta", data=data, maxshape=(None, NZ, NY, NX), chunks=(1, NZ, NY, NX))
        ta.attrs["_FillValue"] = np.array([-32768], dtype=np.int16)
        ta.attrs["scale_factor"] = np.float64(0.01)
        ta.attrs["add_offset"] = np.float64(280.0)
        ta.attrs["units"] = np.bytes_("K")
        ta.attrs["long_name"] = np.bytes_("air temperature")
        for axis, scale in enumerate((time, lev, lat, lon)):
            ta.dims[axis].attach_scale(scale)
        orog = f.create_dataset("orog", data=np.arange(NY * NX, dtype="f4").reshape(NY, NX))
        orog.attrs["units"] = np.bytes_("m")
        orog.dims[0].attach_scale(lat)
        orog.dims[1].attach_scale(lon)
        tb = f.create_dataset("time_bnds", data=np.zeros((NT, 2)))
        tb.dims[0].attach_scale(time)
        tb.dims[1].attach_scale(bnds)
        time.attrs["bounds"] = np.bytes_("time_bnds")
        gw = f.create_dataset("gw", data=np.ones(NY))
        gw.dims[0].attach_scale(lat)
    return path


def test_cf_netcdf4_transposes_scales_and_masks(tmp_path: Path) -> None:
    path = _write_cf_nc4(tmp_path / "atmos.nc")
    c = NetcdfConnector()
    det = c.detect(path)
    assert det is not None and det.confidence >= 0.9 and det.format == "netcdf4"
    rep = check_connector(c, path, tmp_path / "out")
    d = rep.description
    assert d.format == "netcdf4" and d.domain == "geophysical" and d.temporal == "transient"
    zones = {z.name: z for z in d.zones}
    assert set(zones) == {"lon_lat_lev", "lon_lat"}
    assert zones["lon_lat_lev"].shape == [NX, NY, NZ] and zones["lon_lat"].shape == [NX, NY]
    assert d.time.units == "s" and d.time.values == [0.0, 21600.0, 43200.0, 64800.0, 86400.0]
    assert d.metadata["time_reference"] == "2000-01-01 00:00:00"
    assert any("gw(lat)" in w for w in d.warnings)
    ta_info = next(f for f in d.fields if f.name == "ta")
    assert ta_info.units == "K" and ta_info.time_dependent and ta_info.location == "grid"
    assert not any(f.name in ("time_bnds", "bnds", "lat") for f in d.fields)

    s = rep.samples["atmos__lon_lat_lev"][0]
    assert s.kind == "grid" and list(s.coords) == ["x", "y", "z"]
    close(s.coords["x"], [0.0, 90.0, 180.0, 270.0])
    close(s.coords["z"], [100.0, 500.0])
    assert s.metadata["axis_names"] == {"x": "lon", "y": "lat", "z": "lev"}
    ta = s.fields["ta"]
    assert not isinstance(ta.data, np.ndarray)  # lazy
    assert ta.dims == ["time", "x", "y", "z", "component"] and ta.shape == (NT, NX, NY, NZ, 1)
    block = np.asarray(ta.data[:])
    assert block.dtype == np.float32
    for t, i, j, k in [(0, 0, 0, 0), (4, 3, 2, 1), (2, 1, 2, 0), (3, 2, 0, 1)]:
        assert block[t, i, j, k, 0] == pytest.approx(_expected(t, i, j, k), abs=1e-4)
    ft, fk, fj, fi = FILL_AT
    assert np.isnan(block[ft, fi, fj, fk, 0]) and np.isnan(block).sum() == 1
    close(ta.data[3], block[3])
    close(ta.data[[4, 1]], block[[4, 1]])

    orog = rep.samples["atmos__lon_lat"][0]
    assert orog.times is None and orog.fields["orog"].shape == (NX, NY, 1)
    assert float(orog.fields["orog"].data[3, 1, 0]) == 1 * NX + 3

    g = open_store(rep.results["zarr"].out / "samples/atmos__lon_lat_lev/grid.zarr")
    assert g["fields/ta"].shape == (NT, NX, NY, NZ, 1)
    assert g["fields/ta"][4, 3, 2, 1, 0] == pytest.approx(_expected(4, 3, 2, 1), abs=1e-4)
    close(g["times"][...], d.time.values)


def test_netcdf_time_field_zone_selection_and_vector_grouping(tmp_path: Path) -> None:
    path = _write_cf_nc4(tmp_path / "atmos.nc")
    c = NetcdfConnector()
    (s,) = c.read(path, zones=["lon_lat_lev"], times=[1, 3], fields=["ta"])
    assert s.id == "atmos" and list(s.fields) == ["ta"] and s.fields["ta"].shape[0] == 2
    close(s.times, [21600.0, 64800.0])
    assert float(s.fields["ta"].data[1, 2, 0, 1, 0]) == pytest.approx(_expected(3, 2, 0, 1), 1e-6)
    (last,) = c.read(path, zones=["lon_lat_lev"], times="last")
    assert last.fields["ta"].shape[0] == 1 and last.times is not None and last.times[0] == 86400.0
    with pytest.raises(fc.SelectionError, match="available zones: lon_lat, lon_lat_lev"):
        list(c.read(path, zones=["nope"]))
    with pytest.raises(fc.SelectionError, match="available fields: orog, ta"):
        list(c.read(path, fields=["nope"]))

    # partial mapping: stack two detected scalars into one vector field
    opts = {"mapping": {"fields": {"pair": {"paths": ["ta", "ta"], "units": "K"}}, "domain": "cfd"}}
    desc = c.describe(path, options=opts)
    assert desc.domain == "cfd"
    assert {f.name: f.components for f in desc.fields} == {"pair": 2, "orog": 1}
    (v,) = c.read(path, zones=["lon_lat_lev"], options=opts)
    step = v.fields["pair"].data[2:3]
    assert step.shape == (1, NX, NY, NZ, 2)
    close(step[..., 0], step[..., 1])
    assert step[0, 1, 2, 0, 1] == pytest.approx(_expected(2, 1, 2, 0), abs=1e-4)


def _write_nc3(path: Path, nt: int = 6) -> tuple[Path, np.ndarray]:
    from scipy.io import netcdf_file

    ny, nx = 3, 5
    t, j, i = np.meshgrid(np.arange(nt), np.arange(ny), np.arange(nx), indexing="ij")
    u = (100.0 * t + 10.0 * j + i).astype("f4")
    with netcdf_file(str(path), "w", version=1) as f:
        f.title = "synthetic classic file"
        f.createDimension("time", None)
        f.createDimension("y", ny)
        f.createDimension("x", nx)
        tv = f.createVariable("time", "d", ("time",))
        tv.units = "seconds"
        yv = f.createVariable("y", "f", ("y",))
        yv[:] = [0.0, 0.5, 1.0]
        xv = f.createVariable("x", "f", ("x",))
        xv[:] = np.linspace(0.0, 2.0, nx)
        uv = f.createVariable("u", "f", ("time", "y", "x"))
        uv.units = "m/s"
        pv = f.createVariable("p", "h", ("time", "y", "x"))
        pv.scale_factor = 0.5
        mask = f.createVariable("mask", "i", ("y", "x"))
        mask[:] = (np.arange(ny * nx).reshape(ny, nx) % 2).astype("i4")
        tv[:] = 0.25 * np.arange(nt)
        uv[:] = u
        pv[:] = (2 * u).astype("i2")
    return path, u


def test_netcdf3_classic_is_memory_mapped(tmp_path: Path) -> None:
    path, u = _write_nc3(tmp_path / "flow.nc")
    assert path.read_bytes()[:4] == b"CDF\x01"
    c = NetcdfConnector()
    det = c.detect(path)
    assert det is not None and det.format == "netcdf3" and det.confidence == 0.95
    rep = check_connector(c, path, tmp_path / "out")
    d = rep.description
    assert d.format == "netcdf3" and d.domain == "other" and d.time.dt == pytest.approx(0.25)
    assert [z.name for z in d.zones] == ["grid"] and d.zones[0].shape == [5, 3]
    s = rep.first()
    assert s.id == "flow" and set(s.fields) == {"u", "p"} and set(s.static) == {"mask"}
    assert s.fields["u"].units == "m/s" and not isinstance(s.fields["u"].data, np.ndarray)
    got = np.asarray(s.fields["u"].data[:])
    close(got[..., 0], u.transpose(0, 2, 1))
    close(np.asarray(s.fields["p"].data[5])[..., 0], u[5].T, atol=0.5)
    assert s.static["mask"].shape == (5, 3, 1) and s.static["mask"].data[1, 2, 0] == 1
    # the lazy view sits on a numpy memory map, not on a loaded copy
    handle = s.fields["u"].data.sources[0].handle
    assert isinstance(handle.base, np.memmap) or isinstance(handle, np.memmap)


def test_reads_stay_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, u = _write_nc3(tmp_path / "long.nc", nt=40)
    requested: list[int] = []
    original = am.LazyArray._read

    def spy(self: am.LazyArray, lead: np.ndarray | None) -> np.ndarray:
        requested.append(self.shape[0] if lead is None else len(lead))
        return original(self, lead)

    monkeypatch.setattr(am.LazyArray, "_read", spy)
    monkeypatch.setattr(convert_mod, "STEP_CHUNK_BYTES", 2 * 5 * 3 * 4)  # two steps per batch
    samples = NetcdfConnector().read(path, fields=["u"])
    res = fc.to_frontier_dataset(samples, tmp_path / "ds", backend="zarr")
    assert res.backends == {"long": "zarr-stream"} and res.n_steps == {"long": 40}
    assert requested and max(requested) == 2 and len(requested) == 20
    g = open_store(tmp_path / "ds/samples/long/grid.zarr")
    close(g["fields/u"][...][..., 0], u.transpose(0, 2, 1))


def _write_pdebench(path: Path) -> dict[str, np.ndarray]:
    nb, nt, nx, ny = 3, 4, 6, 5
    b, t, i, j = np.meshgrid(
        np.arange(nb), np.arange(nt), np.arange(nx), np.arange(ny), indexing="ij"
    )
    arrays = {
        "Vx": (1000.0 * b + 100.0 * t + 10.0 * i + j).astype("f4"),
        "Vy": -(1000.0 * b + 100.0 * t + 10.0 * i + j).astype("f4"),
        "density": (b + 0.1 * t + 0.0 * i).astype("f4"),
    }
    with h5py.File(path, "w") as f:
        for name, a in arrays.items():
            f.create_dataset(name, data=a)
        f.create_dataset("x-coordinate", data=(np.arange(nx) + 0.5) / nx)
        f.create_dataset("y-coordinate", data=(np.arange(ny) + 0.5) / ny)
        f.create_dataset("t-coordinate", data=0.05 * np.arange(nt + 1))  # T + 1 edges
        f.attrs["eta"] = 0.01
    return arrays


def test_pdebench_like_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "2D_CFD_Rand.hdf5"
    arrays = _write_pdebench(path)
    c = Hdf5Connector()
    det = c.detect(path)
    assert det is not None and det.confidence <= 0.3
    opts = {"mapping": {"fields": {"velocity": {"paths": ["Vx", "Vy"], "units": "m/s"}}}}
    rep = check_connector(c, path, tmp_path / "out", options=opts, min_confidence=0.2)
    d = rep.description
    assert d.metadata["n_samples"] == 3 and d.metadata["mapping_source"] == "auto-detected"
    assert d.metadata["mapping"]["samples"] == {"axis": "sample"}
    assert d.time.n_steps == 4 and d.time.dt == pytest.approx(0.05)
    assert any("last one was dropped" in w for w in d.warnings)
    assert {f.name: f.components for f in d.fields} == {"velocity": 2, "density": 1}
    assert d.parameters == {"eta": 0.01}
    assert rep.sample_ids == ["2D_CFD_Rand_0000", "2D_CFD_Rand_0001", "2D_CFD_Rand_0002"]
    s = rep.samples["2D_CFD_Rand_0002"][0]
    assert s.metadata["parameters"] == {"eta": 0.01} and s.metadata["sample"] == "0002"
    v = np.asarray(s.fields["velocity"].data[:])
    assert v.shape == (4, 6, 5, 2)
    close(v[..., 0], arrays["Vx"][2])
    close(v[..., 1], arrays["Vy"][2])
    close(s.coords["y"], (np.arange(5) + 0.5) / 5)
    only = list(c.read(path, options={"samples": [1]}, fields=["density"], times=[3]))
    assert [x.id for x in only] == ["2D_CFD_Rand_0001"]
    close(only[0].fields["density"].data[0], arrays["density"][1, 3, :, :, None])
    assert len(list(c.read(path, options={"samples": "0:2"}))) == 2


def test_pdebench_group_per_sample(tmp_path: Path) -> None:
    path = tmp_path / "swe.h5"
    nt, nx, ny = 3, 4, 5
    with h5py.File(path, "w") as f:
        for n in range(3):
            g = f.create_group(f"{n:04d}")
            data = np.full((nt, nx, ny, 1), float(n), dtype="f4")
            data += np.arange(nt, dtype="f4")[:, None, None, None] / 10
            g.create_dataset("data", data=data)
            g.create_dataset("grid/x", data=np.linspace(-1, 1, nx))
            g.create_dataset("grid/y", data=np.linspace(-1, 1, ny))
            g.create_dataset("grid/t", data=np.linspace(0, 1, nt))
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    assert rep.sample_ids == ["swe_0000", "swe_0001", "swe_0002"]
    assert rep.description.metadata["mapping"]["samples"] == {"groups": "*"}
    s = rep.samples["swe_0002"][0]
    assert s.fields["data"].shape == (nt, nx, ny, 1)
    assert float(s.fields["data"].data[2, 0, 0, 0]) == pytest.approx(2.2)
    close(s.times, [0.0, 0.5, 1.0])


def test_the_well_like_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "shear_flow.hdf5"
    ns, nt, nx, ny = 2, 3, 6, 4
    s_, t_, i_, j_ = np.meshgrid(
        np.arange(ns), np.arange(nt), np.arange(nx), np.arange(ny), indexing="ij"
    )
    pressure = (1000.0 * s_ + 100.0 * t_ + 10.0 * i_ + j_).astype("f4")
    velocity = np.stack([pressure, -pressure], axis=-1)
    with h5py.File(path, "w") as f:
        f.attrs.update(
            dataset_name="shear_flow", grid_type="cartesian", n_spatial_dims=2, n_trajectories=ns
        )
        f.attrs["simulation_parameters"] = np.array(["Reynolds"], dtype="S")
        f.attrs["Reynolds"] = 1e4
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = np.array(["x", "y"], dtype="S")
        for name, vals in (
            ("time", 0.5 * np.arange(nt)),
            ("x", np.linspace(0, 1, nx)),
            ("y", np.linspace(0, 2, ny)),
        ):
            ds = dims.create_dataset(name, data=vals)
            ds.attrs["sample_varying"] = False
        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = np.array(["pressure", "mask"], dtype="S")
        p = t0.create_dataset("pressure", data=pressure)
        p.attrs.update(dim_varying=[True, True], sample_varying=True, time_varying=True)
        m = t0.create_dataset("mask", data=np.arange(nx * ny, dtype="f4").reshape(nx, ny))
        m.attrs.update(dim_varying=[True, True], sample_varying=False, time_varying=False)
        c0 = t0.create_dataset("profile", data=np.zeros((ns, nt, ny), dtype="f4"))
        c0.attrs.update(dim_varying=[False, True], sample_varying=True, time_varying=True)
        t1 = f.create_group("t1_fields")
        v = t1.create_dataset("velocity", data=velocity)
        v.attrs.update(dim_varying=[True, True], sample_varying=True, time_varying=True)
        f.create_group("t2_fields")
        sc = f.create_group("scalars")
        sc.create_dataset("Schmidt", data=np.array([1.0, 2.0]))
        bc = f.create_group("boundary_conditions/x_periodic")
        bc.attrs.update(bc_type="PERIODIC", associated_dims=np.array(["x"], dtype="S"))
        bc.create_dataset("mask", data=np.zeros(nx, dtype=bool))
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    d = rep.description
    assert d.metadata["mapping"]["metadata"]["layout"] == "the-well"
    assert {f.name: (f.components, f.time_dependent) for f in d.fields} == {
        "pressure": (1, True),
        "mask": (1, False),
        "velocity": (2, True),
    }
    assert any("profile is constant along ['x']" in w for w in d.warnings)
    assert d.boundary_conditions["x_periodic"]["bc_type"] == "PERIODIC"
    assert d.parameters == {"Schmidt": [1.0, 2.0], "Reynolds": 1e4}
    s = rep.samples["shear_flow_0001"][0]
    assert s.metadata["parameters"] == {"Schmidt": 2.0, "Reynolds": 1e4}
    close(s.fields["velocity"].data[:], velocity[1])
    close(s.fields["pressure"].data[2], pressure[1, 2][..., None])
    assert s.static["mask"].shape == (nx, ny, 1) and s.static["mask"].data[2, 3, 0] == 2 * ny + 3
    close(s.times, [0.0, 0.5, 1.0])


def test_the_well_reference_dummy_layout(tmp_path: Path) -> None:
    """Same structure, attributes and dtypes as ``the_well.utils.dummy_data.write_dummy_data``
    (PolymathicAI/the_well, the project's own reference writer for the format)."""
    path = tmp_path / "dummy_dataset.hdf5"
    rng = np.random.default_rng(1)
    n_traj, nx, ny, nt = 2, 32, 32, 10
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    t = np.linspace(0, 1, nt, dtype=np.float32)
    t1 = rng.random((n_traj, nt, nx, ny, 2)).astype(np.float32)
    t0_const = rng.random((n_traj, nx, ny)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["a"] = 0.25
        f.attrs["b"] = 0.75
        f.attrs["dataset_name"] = "dummy_dataset"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = n_traj
        f.attrs["simulation_parameters"] = ["a", "b"]
        group = f.create_group("boundary_conditions")
        for key, val in (("x_periodic", x), ("y_periodic", y)):
            sub = group.create_group(key)
            sub.attrs["associated_dims"] = key[0]
            sub.attrs["associated_fields"] = []
            sub.attrs["bc_type"] = "PERIODIC"
            sub.attrs["sample_varying"] = False
            sub.attrs["time_varying"] = False
            sub.create_dataset("mask", data=np.zeros_like(val).astype(bool))
        group = f.create_group("dimensions")
        group.attrs["spatial_dims"] = ["x", "y"]
        for key, val in (("time", t), ("x", x), ("y", y)):
            group.create_dataset(key, data=val)
            group[key].attrs["sample_varying"] = False
        group = f.create_group("scalars")
        group.attrs["field_names"] = ["a", "b", "time_varying_scalar"]
        for key, val in (("a", 0.25), ("b", 0.75)):
            group.create_dataset(key, data=np.array(val))
            group[key].attrs["time_varying"] = False
            group[key].attrs["sample_varying"] = False
        dset = group.create_dataset("time_varying_scalar", data=rng.random(nt))
        dset.attrs["time_varying"] = True
        dset.attrs["sample_varying"] = False
        group = f.create_group("t0_fields")
        group.attrs["field_names"] = ["constant_field"]
        dset = group.create_dataset("constant_field", data=t0_const)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = False
        group = f.create_group("t1_fields")
        group.attrs["field_names"] = ["field"]
        dset = group.create_dataset("field", data=t1)
        dset.attrs["dim_varying"] = [True, True]
        dset.attrs["sample_varying"] = True
        dset.attrs["time_varying"] = True
        f.create_group("t2_fields").attrs["field_names"] = []
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    d = rep.description
    assert d.metadata["n_samples"] == 2 and d.zones[0].shape == [nx, ny]
    assert d.parameters == {"a": 0.25, "b": 0.75} and d.time.n_steps == nt
    assert set(d.boundary_conditions) == {"x_periodic", "y_periodic"}
    assert d.boundary_conditions["x_periodic"]["associated_dims"] == "x"
    assert {f.name: (f.components, f.time_dependent) for f in d.fields} == {
        "constant_field": (1, False),
        "field": (2, True),
    }
    s = rep.samples["dummy_dataset_0001"][0]
    close(s.fields["field"].data[:], t1[1])
    close(s.static["constant_field"].data[..., 0], t0_const[1])  # per-trajectory static input
    close(s.coords["x"], x)
    close(s.times, t)


def _mesh_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, tets = tet_grid(2, 2, 1)
    times = 0.1 * np.arange(4)
    temp = np.stack([analytic_fields(points, t)["p"] for t in times])  # (T, N)
    vel = np.stack([analytic_fields(points, t)["U"] for t in times], axis=-1)  # (N, 3, T)
    with h5py.File(path, "w") as f:
        f.create_dataset("mesh/coordinates", data=points)
        f.create_dataset("mesh/topology", data=tets + 1)  # 1-based, as many solvers write
        f.create_dataset("results/T", data=temp)
        f.create_dataset("results/U", data=vel)
        f.create_dataset("results/cell_volume", data=np.arange(len(tets), dtype="f8"))
        f.create_group("sim").attrs["dt"] = 0.1
        f["sim"].attrs["Re"] = 250.0
    return points, tets, temp, vel


MESH_MAPPING: dict[str, Any] = {
    "kind": "mesh",
    "points": "/mesh/coordinates",
    "cells": "/mesh/topology",
    "cell_type": "tetra",
    "index_base": 1,
    "time": {"attr": "/sim@dt"},
    "fields": {
        "T": {"path": "/results/T", "units": "K"},
        "U": {"path": "/results/U", "dims": ["point", "component", "time"], "units": "m/s"},
    },
    "static": {"volume": {"path": "/results/cell_volume", "dims": ["cell"]}},
    "parameters": {"Re": "/sim@Re", "Ma": 0.3},
    "units": {"length": "m"},
    "domain": "cfd",
    "solver": "in-house",
}


def test_mesh_hdf5_with_mapping(tmp_path: Path) -> None:
    path = tmp_path / "run.h5"
    points, tets, temp, vel = _mesh_file(path)
    c = Hdf5Connector()
    # without a mapping the points+cells layout is recognised, but U (N, 3, T) is not
    auto = c.describe(path)
    assert [f.name for f in auto.fields] == ["T", "cell_volume"]
    assert auto.metadata["mapping"]["points"] == "/mesh/coordinates"
    (guess,) = c.read(path)
    equal(guess.cells, tets)  # 1-based indices were recognised

    opts = {"mapping": MESH_MAPPING}
    rep = check_connector(c, path, tmp_path / "out", options=opts, min_confidence=0.2)
    d = rep.description
    assert d.domain == "cfd" and d.solver == "in-house" and d.units == {"length": "m"}
    assert d.parameters == {"Re": 250.0, "Ma": 0.3} and d.metadata["mapping_source"] == "user"
    assert d.zones[0].cell_types == {"tetra": len(tets)} and d.zones[0].n_points == len(points)
    close(d.time.values, [0.0, 0.1, 0.2, 0.3])
    s = rep.first()
    assert s.kind == "mesh"
    equal(s.cells, tets)
    assert s.cell_types is not None and set(s.cell_types.tolist()) == {meshutil.VTK_TETRA}
    close(s.points, points)
    u = s.fields["U"]
    assert u.dims == ["time", "point", "component"] and u.shape == (4, len(points), 3)
    close(u.data[:], vel.transpose(2, 0, 1), rtol=1e-6)
    close(s.fields["T"].data[3][:, 0], temp[3], rtol=1e-6)
    assert s.static["volume"].location == "cell" and s.static["volume"].shape == (len(tets), 1)
    g = open_store(rep.results["zarr"].out / "samples/run/mesh.zarr")
    assert g["fields/U"].shape == (4, len(points), 3) and g["static/volume"].shape[0] == len(tets)

    bad = {**MESH_MAPPING, "index_base": 0}
    with pytest.raises(fc.MappingError, match="reference points outside"):
        list(c.read(path, options={"mapping": bad}))
    bad = {**MESH_MAPPING, "fields": {"U": {"path": "/results/U", "dims": ["point", "time"]}}}
    with pytest.raises(fc.MappingError, match="2 dims named for an array of shape"):
        c.describe(path, options={"mapping": bad})
    with pytest.raises(fc.MappingError, match="similar arrays: /mesh/topology"):
        c.describe(path, options={"mapping": {**MESH_MAPPING, "cells": "/topology"}})
    with pytest.raises(fc.MappingError, match="unknown mapping key"):
        c.describe(path, options={"mapping": {**MESH_MAPPING, "feilds": {}}})


def test_mesh_layout_autodetect_surface_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "rollout.h5"
    points, tris = tri_grid(4, 2)
    points = points + np.array([0.0, 0.0, 1.0])
    times = np.array([0.0, 0.5, 1.5])
    p = np.stack([analytic_fields(points, t)["p"] for t in times])
    load = np.arange(len(tris), dtype="f4")
    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=points)
        f.create_dataset("cells", data=tris.astype("i4"))
        f.create_dataset("pressure", data=p)
        f.create_dataset("load", data=np.tile(load, (3, 1)))
        f.create_dataset("time", data=times)
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    s = rep.first()
    assert s.kind == "surface" and s.faces is not None and s.faces.shape == tris.shape
    assert s.fields["load"].location == "cell" and s.fields["pressure"].location == "point"
    close(s.fields["pressure"].data[:][..., 0], p, rtol=1e-6)
    close(s.times, times)
    assert any("VTK node ordering" in w for w in rep.description.warnings)

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(MESH_MAPPING))
    run = tmp_path / "run.h5"
    _mesh_file(run)
    out = tmp_path / "ds"
    args = ["--connector", "hdf5", "--mapping", str(mapping)]
    assert main(["convert", str(run), "--out", str(out), "--times", "1:3", "--json", *args]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["n_steps"] == {"run": 2} and res["backends"] == {"run": "zarr-stream"}
    odd = _ambiguous(tmp_path / "odd.h5")
    assert main(["describe", str(odd), "--json", "--connector", "hdf5"]) == 2
    assert "suggested mapping skeleton" in capsys.readouterr().out


def _ambiguous(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("run/a", data=np.zeros((10, 7, 5), dtype="f4"))
        f.create_dataset("run/b", data=np.zeros((10, 7, 5), dtype="f4"))
        f.create_dataset("axis0", data=np.arange(7.0))
        f.create_dataset("labels", data=np.array([b"a", b"b"]))
        f.attrs["code"] = "in-house"
    return path


def test_ambiguous_hdf5_raises_mapping_error_with_tree_and_skeleton(tmp_path: Path) -> None:
    path = _ambiguous(tmp_path / "odd.h5")
    c = Hdf5Connector()
    with pytest.raises(fc.MappingError) as err:
        c.describe(path)
    e = err.value
    assert "not self-describing" in e.message and e.hint is not None
    assert "/run/a  (10, 7, 5)  float32" in e.hint and "/axis0  (7,)  float64" in e.hint
    assert "/ attributes: code" in e.hint and "--mapping mapping.json" in e.hint
    skeleton, _ = json.JSONDecoder().raw_decode(e.hint[e.hint.index("{\n") :])
    assert skeleton["fields"]["a"] == {"path": "/run/a", "dims": ["time", "x", "component"]}
    assert skeleton["coords"]["x"] == "/axis0"
    with pytest.raises(fc.MappingError):
        list(c.read(path))
    # the corrected skeleton is a working mapping
    skeleton["coords"] = {"x": "/axis0", "y": {"linspace": [0, 1]}}
    skeleton["fields"] = {
        "ab": {"paths": ["/run/a", "/run/b"], "dims": ["time", "x", "y"]},
    }
    rep = check_connector(
        c, path, tmp_path / "out", options={"mapping": skeleton}, min_confidence=0.2
    )
    assert rep.first().fields["ab"].shape == (10, 7, 5, 2)


def test_hdf5_dimension_scales_and_foreign_signatures(tmp_path: Path) -> None:
    path = tmp_path / "scales.h5"
    data = np.arange(3 * 4 * 5, dtype="f8").reshape(4, 3, 5)  # (y, time, x)
    with h5py.File(path, "w") as f:
        f.create_dataset("xc", data=np.linspace(0, 1, 5)).make_scale("x")
        f.create_dataset("yc", data=np.linspace(0, 1, 4)).make_scale("y")
        t = f.create_dataset("t", data=np.array([0.0, 1.0, 2.0]))
        t.make_scale("time")
        t.attrs["units"] = "ms"
        d = f.create_dataset("phi", data=data)
        for axis, name in enumerate(("yc", "t", "xc")):
            d.dims[axis].attach_scale(f[name])
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    s = rep.first()
    phi = s.fields["phi"]
    assert phi.dims == ["time", "x", "y", "component"] and phi.shape == (3, 5, 4, 1)
    close(phi.data[:][..., 0], data.transpose(1, 2, 0))
    close(s.times, [0.0, 0.001, 0.002])

    c = Hdf5Connector()
    with h5py.File(tmp_path / "x.hdf", "w") as f:
        f.create_group("VTKHDF")
    with h5py.File(tmp_path / "case.cas.h5", "w") as f:
        f.create_group("meshes")
    with h5py.File(tmp_path / "grid.h5", "w") as f:
        f.create_group("Base")
        f.create_dataset("CGNSLibraryVersion", data=[4.2])
    for name in ("x.hdf", "case.cas.h5", "grid.h5"):
        assert c.detect(tmp_path / name) is None
        assert NetcdfConnector().detect(tmp_path / name) is None
    (tmp_path / "notes.h5").write_text("not hdf5")
    assert c.detect(tmp_path / "notes.h5") is None and c.detect(tmp_path) is None


def _write_ugrid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import netcdf_file

    x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    faces = np.array([[1, 2, 5, 4], [2, 3, 6, -999], [3, 6, 5, -999]], dtype="i4")
    with netcdf_file(str(path), "w") as f:
        f.Conventions = "CF-1.8 UGRID-1.0"
        f.createDimension("time", None)
        f.createDimension("node", 6)
        f.createDimension("face", 3)
        f.createDimension("nmax", 4)
        # scipy's writer corrupts record data when a 0-d variable is present, so the topology
        # variable (a dummy integer that only carries attributes) gets a length-1 dimension
        f.createDimension("one", 1)
        mesh = f.createVariable("mesh2d", "i", ("one",))
        mesh[:] = 0
        mesh.cf_role = "mesh_topology"
        mesh.topology_dimension = 2
        mesh.node_coordinates = "node_x node_y"
        mesh.face_node_connectivity = "face_nodes"
        f.createVariable("node_x", "d", ("node",))[:] = x
        f.createVariable("node_y", "d", ("node",))[:] = y
        fn = f.createVariable("face_nodes", "i", ("face", "nmax"))
        fn.start_index = 1
        fn._FillValue = -999
        fn[:] = faces
        tv = f.createVariable("time", "d", ("time",))
        tv.units = "seconds since 2020-01-01"
        h = f.createVariable("waterlevel", "d", ("time", "node"))
        h.mesh = "mesh2d"
        h.location = "node"
        q = f.createVariable("discharge", "d", ("time", "face"))
        q.mesh = "mesh2d"
        q.location = "face"
        tv[:] = [0.0, 60.0]
        h[:] = np.array([x + y, x + y + 1.0])
        q[:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return np.stack([x, y], axis=1), faces


def test_ugrid_2d_mesh(tmp_path: Path) -> None:
    path = tmp_path / "map.nc"
    points, faces = _write_ugrid(path)
    rep = check_connector(NetcdfConnector(), path, tmp_path / "out")
    z = rep.description.zones[0]
    assert (z.name, z.kind, z.n_points, z.n_cells) == ("mesh2d", "mesh", 6, 3)
    s = rep.first()
    close(s.points, points)
    equal(s.cells, [[0, 1, 4, 3], [1, 2, 5, -1], [2, 5, 4, -1]])
    assert s.cell_types is not None and s.cell_types.tolist() == [9, 5, 5]
    assert s.fields["waterlevel"].location == "point" and s.fields["discharge"].location == "cell"
    close(s.fields["discharge"].data[1][:, 0], [4.0, 5.0, 6.0])
    close(s.times, [0.0, 60.0])


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_zarr_xarray_style(tmp_path: Path, zarr_format: int) -> None:
    path = tmp_path / f"ocean_v{zarr_format}.zarr"
    nt, ny, nx = 4, 3, 5
    t, j, i = np.meshgrid(np.arange(nt), np.arange(ny), np.arange(nx), indexing="ij")
    sst = (100.0 * t + 10.0 * j + i).astype("f4")
    fmt: Any = zarr_format
    root: Any = zarr.open_group(str(path), mode="w", zarr_format=fmt)

    def put(name: str, data: np.ndarray, dims: list[str], **attrs: Any) -> None:
        kw: dict[str, Any] = {"dimension_names": dims} if zarr_format == 3 else {}
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=data.shape, **kw)
        arr[...] = data
        if zarr_format == 2:
            attrs["_ARRAY_DIMENSIONS"] = dims
        arr.attrs.update(attrs)

    put("time", np.arange(nt, dtype="f8"), ["time"], units="days since 2001-01-01")
    put("latitude", np.array([10.0, 0.0, -10.0]), ["latitude"], units="degrees_north")
    put("longitude", np.linspace(100.0, 140.0, nx), ["longitude"], units="degrees_east")
    put("sst", sst, ["time", "latitude", "longitude"], units="degC", _FillValue=213.0)
    put("depth", np.ones((ny, nx), dtype="f4"), ["latitude", "longitude"], units="m")
    c = ZarrConnector()
    det = c.detect(path)
    assert det is not None and det.format == f"zarr-v{zarr_format}" and det.confidence == 0.95
    rep = check_connector(c, path, tmp_path / "out")
    d = rep.description
    assert d.domain == "geophysical" and d.time.dt == 86400.0 and d.zones[0].shape == [nx, ny]
    s = rep.first()
    assert s.id == f"ocean_v{zarr_format}" and s.fields["sst"].units == "degC"
    assert not isinstance(s.fields["sst"].data, np.ndarray)
    got = np.asarray(s.fields["sst"].data[:])[..., 0]
    want = sst.transpose(0, 2, 1).copy()
    want[want == 213.0] = np.nan
    close(got, want)
    assert np.isnan(got[2, 3, 1]) and np.isnan(got).sum() == 1
    close(s.coords["y"], [10.0, 0.0, -10.0])
    assert s.static["depth"].shape == (nx, ny, 1)


def test_zarr_with_mapping_and_sample_axis(tmp_path: Path) -> None:
    path = tmp_path / "runs.zarr"
    root: Any = zarr.open_group(str(path), mode="w", zarr_format=3)
    nb, nx, ny, nt = 2, 4, 3, 5
    b, i, j, t = np.meshgrid(
        np.arange(nb), np.arange(nx), np.arange(ny), np.arange(nt), indexing="ij"
    )
    data = (1000.0 * b + 100.0 * t + 10.0 * i + j).astype("f4")  # FNO order (batch, x, y, time)
    arr = root.create_array("u", shape=data.shape, dtype="f4", chunks=(1, nx, ny, nt))
    arr[...] = data
    nu = root.create_array("nu", shape=(nb,), dtype="f8", chunks=(nb,))
    nu[...] = [1e-3, 1e-4]
    with pytest.raises(fc.MappingError, match="suggested mapping skeleton"):
        ZarrConnector().describe(path)
    mapping = {
        "coords": {"x": {"linspace": [0, 1]}, "y": {"start": 0.0, "step": 0.5}},
        "time": {"start": 1.0, "dt": 0.25},
        "samples": {"axis": "batch", "ids": ["low", "high"]},
        "fields": {"u": {"path": "u", "dims": ["batch", "x", "y", "time"]}},
        "parameters": {"nu": "/nu"},
    }
    rep = check_connector(ZarrConnector(), path, tmp_path / "out", options={"mapping": mapping})
    assert rep.sample_ids == ["runs_low", "runs_high"]
    s = rep.samples["runs_high"][0]
    assert s.metadata["parameters"] == {"nu": 1e-4}
    close(s.times, 1.0 + 0.25 * np.arange(nt))
    close(s.coords["x"], np.linspace(0, 1, nx))
    close(s.fields["u"].data[:][..., 0], data[1].transpose(2, 0, 1))
    (one,) = ZarrConnector().read(path, options={"mapping": mapping, "samples": ["low"]})
    assert one.id == "runs_low"


def _dataset(tmp_path: Path, moving: bool = False) -> Path:
    points, cells = hex_grid(3, 2, 2)
    types = np.full(len(cells), meshutil.VTK_HEXAHEDRON, dtype=np.uint8)

    def steps() -> Any:
        for n in range(4):
            f = analytic_fields(points, 0.1 * n)
            yield fc.RawSample(
                id="case",
                kind="mesh",
                points=(points + 0.01 * n) if (n == 0 or moving) else None,
                cells=cells if n == 0 else None,
                cell_types=types if n == 0 else None,
                fields={
                    "p": fc.as_field(f["p"], "point", units="Pa"),
                    "U": fc.as_field(f["U"], "point", units="m/s"),
                    "pc": fc.as_field(f["p"][cells].mean(axis=1), "cell"),
                },
                static={"wall_distance": fc.as_field(points[:, 1], "point")} if n == 0 else {},
                time=0.1 * n,
                time_index=n,
                metadata={"solver": "synthetic", "parameters": {"Re": 100.0}},
            )
        tp, tris = tri_grid(3, 2)
        yield fc.RawSample(
            id="wall",
            kind="surface",
            points=tp,
            faces=tris,
            fields={"cp": fc.as_field(tp[:, 0], "point")},
            metadata={"solver": "synthetic"},
        )
        xs, ys = np.linspace(0, 1, 4), np.linspace(0, 2, 3)
        rho = (xs[:, None] + 10 * ys[None, :])[None, :, :, None] * np.ones((2, 1, 1, 1))
        yield fc.RawSample(
            id="box",
            kind="grid",
            coords={"x": xs, "y": ys},
            fields={
                "rho": fc.FieldData(
                    rho.astype("f4"), ["time", "x", "y", "component"], location="grid"
                )
            },
            times=np.array([0.0, 2.0]),
            metadata={"solver": "synthetic"},
        )

    out = tmp_path / "dataset"
    fc.to_frontier_dataset(steps(), out, backend="zarr", domain="cfd")
    return out


def test_frontier_dataset_round_trip(tmp_path: Path) -> None:
    out = _dataset(tmp_path)
    c = ZarrConnector()
    det = c.detect(out)
    assert det is not None and det.format == "frontier-dataset" and det.confidence == 0.9
    assert fc.detect(out)[0].connector_id == "zarr"
    rep = check_connector(c, out, tmp_path / "again")
    d = rep.description
    assert {z.name for z in d.zones} == {"mesh", "surface", "grid"} and d.domain == "cfd"
    assert d.metadata["n_samples"] == 3 and d.temporal == "transient"
    points, cells = hex_grid(3, 2, 2)
    case = rep.samples["case"][0]
    assert case.kind == "mesh" and case.metadata["parameters"] == {"Re": 100.0}
    assert case.metadata["solver"] == "synthetic"
    equal(case.cells, cells)
    close(case.points, points, rtol=1e-6)
    close(case.times, [0.0, 0.1, 0.2, 0.3])
    u = case.fields["U"]
    assert not isinstance(u.data, np.ndarray) and u.units == "m/s"
    assert u.dims == ["time", "point", "component"]
    close(u.data[2], analytic_fields(points, 0.2)["U"], rtol=1e-6)
    assert case.fields["pc"].location == "cell" and "wall_distance" in case.static
    wall = rep.samples["wall"][0]
    assert wall.kind == "surface" and wall.times is None and wall.fields["cp"].shape[1] == 1
    box = rep.samples["box"][0]
    assert (
        box.fields["rho"].shape == (2, 4, 3, 1)
        and float(box.fields["rho"].data[1, 3, 2, 0]) == 21.0
    )
    # the copy of the copy is identical to the first conversion
    a = open_store(out / "samples/case/mesh.zarr")
    b = open_store(rep.results["zarr"].out / "samples/case/mesh.zarr")
    for name in ("points", "cells", "cell_types", "times", "fields/U", "cell_fields/pc"):
        equal(a[name][...], b[name][...])

    (only,) = c.read(out, options={"samples": ["ca*"]}, times=[1, 2], fields=["p"])
    assert only.id == "case" and list(only.fields) == ["p"] and only.fields["p"].shape[0] == 2
    close(only.times, [0.1, 0.2])
    assert [s.id for s in c.read(out, zones=["surface"])] == ["wall"]
    (single,) = c.read(out / "samples" / "box" / "grid.zarr")
    assert single.id == "box" and single.kind == "grid"
    (by_dir,) = c.read(out / "samples" / "wall")
    assert by_dir.id == "wall" and by_dir.faces is not None


def test_frontier_dataset_moving_mesh_reads_step_form(tmp_path: Path) -> None:
    out = _dataset(tmp_path, moving=True)
    steps = [s for s in ZarrConnector().read(out, options={"samples": ["case"]})]
    assert [s.time for s in steps] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    points, _ = hex_grid(3, 2, 2)
    assert steps[0].cells is not None and steps[2].cells is None
    close(steps[2].points, points + 0.02, rtol=1e-5)
    assert steps[1].fields["U"].dims == ["point", "component"]


def test_corrupt_and_unsupported_inputs(tmp_path: Path) -> None:
    good = _write_cf_nc4(tmp_path / "good.nc")
    blob = good.read_bytes()
    cut = tmp_path / "cut.nc"
    cut.write_bytes(blob[: len(blob) // 3])
    with pytest.raises(fc.ConnectorError, match="hint"):
        NetcdfConnector().describe(cut)
    junk = tmp_path / "junk.nc"
    junk.write_bytes(b"\x00" * 64)
    det = NetcdfConnector().detect(junk)
    assert det is not None and det.confidence <= 0.3
    with pytest.raises(fc.ConnectorError, match="neither NetCDF-3"):
        list(NetcdfConnector().read(junk))
    nc3, _ = _write_nc3(tmp_path / "c.nc")
    short = tmp_path / "short.nc"
    short.write_bytes(nc3.read_bytes()[:40])
    with pytest.raises(fc.ConnectorError):
        NetcdfConnector().describe(short)
    cdf5 = tmp_path / "five.nc"
    cdf5.write_bytes(b"CDF\x05" + b"\x00" * 60)
    with pytest.raises(fc.ConnectorError, match="CDF-5"):
        NetcdfConnector().describe(cdf5)
    bad_h5 = tmp_path / "bad.h5"
    bad_h5.write_bytes(blob[:600])
    with pytest.raises(fc.ConnectorError, match="not a readable HDF5 file"):
        Hdf5Connector().describe(bad_h5)
    empty = tmp_path / "empty.zarr"
    empty.mkdir()
    assert ZarrConnector().detect(empty) is not None
    with pytest.raises(fc.ConnectorError):
        ZarrConnector().describe(empty)
    if importlib.util.find_spec("xarray") is None:
        with pytest.raises(fc.MissingDependencyError, match=r"xarray.*Apache-2.0"):
            NetcdfConnector().describe(good, options={"engine": "xarray"})
    for c in (NetcdfConnector(), Hdf5Connector(), ZarrConnector()):
        assert c.detect(tmp_path / "missing.nc") is None


def test_xarray_engine_and_netcdf4_cross_check(tmp_path: Path) -> None:
    xr: Any = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    path = _write_cf_nc4(tmp_path / "atmos.nc")
    # the netCDF-C library accepts the h5py-written fixture and decodes the same values
    ds = xr.open_dataset(path, decode_times=False)
    ref = ds["ta"].transpose("time", "lon", "lat", "lev").values
    c = NetcdfConnector()
    (ours,) = c.read(path, zones=["lon_lat_lev"])
    close(ours.fields["ta"].data[:][..., 0], ref, rtol=1e-6, equal_nan=True)
    (via,) = c.read(path, zones=["lon_lat_lev"], options={"engine": "xarray"})
    close(via.fields["ta"].data[:][..., 0], ref, rtol=1e-6, equal_nan=True)
    # and a file written by netCDF4 itself reads the same through h5py
    out = tmp_path / "written_by_netcdf4.nc"
    ds.to_netcdf(out, engine="netcdf4")
    (again,) = c.read(out, zones=["lon_lat_lev"])
    close(again.fields["ta"].data[:][..., 0], ref, rtol=1e-6, equal_nan=True)
    assert again.coords["z"].tolist() == [100.0, 500.0]


UNIDATA = "https://archive.unidata.ucar.edu/software/netcdf/examples/"


@pytest.mark.network
def test_public_netcdf3_ccsm_sample(tmp_path: Path) -> None:
    """NCAR CCSM3 output (CMIP3 SRES A1B), NetCDF-3 classic written by the model's NetCDF/CMOR
    stack; one of Unidata's example files."""
    path = fetch_sample(UNIDATA + "sresa1b_ncar_ccsm3-example.nc")
    rep = check_connector(NetcdfConnector(), path, tmp_path / "out")
    d = rep.description
    assert d.format == "netcdf3" and d.domain == "geophysical"
    zones = {z.name: z for z in d.zones}
    assert zones["lon_lat"].shape == [256, 128] and zones["lon_lat_plev"].shape == [256, 128, 17]
    names = {f.name for f in d.fields}
    assert {"tas", "pr", "ua", "area"} <= names and "lat_bnds" not in names
    s = rep.samples[next(i for i in rep.sample_ids if i.endswith("lon_lat"))][0]
    tas = np.asarray(s.fields["tas"].data[:])
    assert tas.shape == (1, 256, 128, 1) and 200.0 < np.nanmin(tas) < np.nanmax(tas) < 330.0
    # poles are colder than the equator: latitude really ended up on the y axis
    assert tas[0, :, 64, 0].mean() > tas[0, :, 0, 0].mean() + 20.0
    ua = rep.samples[next(i for i in rep.sample_ids if i.endswith("plev"))][0].fields["ua"]
    assert np.isnan(np.asarray(ua.data[:])).any()  # below-ground levels are masked


XARRAY_DATA = "https://github.com/pydata/xarray-data/raw/master/"


@pytest.mark.network
def test_public_netcdf_samples(tmp_path: Path) -> None:
    """PCMDI ``tos`` (NetCDF-3 classic, CMOR) from Unidata and an ERA-Interim extract (NetCDF-3
    64-bit offset, packed shorts, written by ECMWF's MARS converter) from pydata/xarray-data."""
    path = fetch_sample(UNIDATA + "tos_O1_2001-2002.nc")
    rep = check_connector(NetcdfConnector(), path, tmp_path / "tos")
    d = rep.description
    assert d.time.n_steps == 24 and d.zones[0].shape == [180, 170]
    assert d.metadata["calendar"] == "360_day" and d.time.dt == pytest.approx(30 * 86400.0)
    tos = rep.first().fields["tos"]
    assert tos.units == "K" and not isinstance(tos.data, np.ndarray)
    step = np.asarray(tos.data[5])
    assert step.shape == (180, 170, 1) and np.isnan(step).any()  # land is masked
    assert 265.0 < np.nanmin(step) < np.nanmax(step) < 310.0

    era = fetch_sample(XARRAY_DATA + "eraint_uvz.nc")
    assert era.read_bytes()[:4] == b"CDF\x02"
    rep = check_connector(NetcdfConnector(), era, tmp_path / "era", fields=["u", "z"])
    d = rep.description
    assert d.zones[0].shape == [480, 241, 3] and d.time.values == [1.0, 7.0]
    assert d.time.units == "month" and d.metadata["axis_names"]["z"] == "level"
    s = rep.first()
    u = np.asarray(s.fields["u"].data[:])
    assert u.shape == (2, 480, 241, 3, 1) and -40.0 < u.min() < u.max() < 90.0  # m/s, unpacked
    z = np.asarray(s.fields["z"].data[0])
    # geopotential grows with height: level axis is (200, 500, 850) hPa
    assert z[..., 0, 0].mean() > z[..., 1, 0].mean() > z[..., 2, 0].mean() > 1.0e4


@pytest.mark.network
def test_public_netcdf4_samples(tmp_path: Path) -> None:
    """NetCDF-4 files written through the netCDF-C library: ROMS ocean model output on a
    curvilinear grid (pydata/xarray-data) and ``test_hgroups.nc`` with nested groups (Unidata)."""
    c = NetcdfConnector()
    roms = fetch_sample(XARRAY_DATA + "ROMS_example.nc")
    assert roms.read_bytes()[:4] == b"\x89HDF"
    rep = check_connector(c, roms, tmp_path / "roms", times=[0])
    d = rep.description
    zones = {z.name: z.shape for z in d.zones}
    assert zones == {"xi_rho_eta_rho": [371, 191], "xi_rho_eta_rho_s_rho": [371, 191, 30]}
    assert d.domain == "geophysical" and d.time.n_steps == 2
    assert d.time.units == "s" and d.metadata["time_source_units"].startswith("seconds since")
    assert any("index coordinates are used" in w for w in d.warnings)
    flat = rep.samples["ROMS_example__xi_rho_eta_rho"][0]
    assert set(flat.static) == {"h", "lat_rho", "lon_rho"} and set(flat.fields) == {"zeta"}
    lat = flat.static["lat_rho"].data
    assert 27.0 < lat.min() < lat.max() < 31.0 and lat[0, -1, 0] > lat[0, 0, 0]  # y = eta
    salt = np.asarray(rep.samples["ROMS_example__xi_rho_eta_rho_s_rho"][0].fields["salt"].data[0])
    assert salt.shape == (371, 191, 30, 1) and np.isnan(salt).any()  # land mask
    assert 0.0 <= np.nanmin(salt) < np.nanmax(salt) < 40.0

    groups = fetch_sample(UNIDATA + "test_hgroups.nc")
    det = c.detect(groups)
    assert det is not None and det.format == "netcdf4"
    store = am.open_h5(groups)
    assert any("/" in p for p in store.nodes)
    node = store.nodes["mozaic_flight_2012030319051051_descent/CO"]
    assert node.dims == ["air_press"] and node.scales is not None
    with pytest.raises(fc.MappingError, match="suggested mapping skeleton"):
        c.describe(groups)  # aircraft profiles (CF discrete sampling geometry), not a grid


@pytest.mark.network
def test_public_ugrid_samples(tmp_path: Path) -> None:
    """UGRID meshes: an LFRic cubed-sphere face field from SciTools/iris-test-data and an ADH
    shallow-water run (triangles, 49 steps) from Deltares/xugrid."""
    c = NetcdfConnector()
    path = fetch_sample(
        "https://raw.githubusercontent.com/SciTools/iris-test-data/master/test_data/NetCDF/"
        "unstructured_grid/data_C4.nc"
    )
    rep = check_connector(c, path, tmp_path / "c4")
    s = rep.first()
    assert s.kind == "mesh" and s.points is not None and s.points.shape == (98, 2)
    assert s.cells is not None and s.cells.shape == (96, 4) and int(s.cells.max()) == 97
    assert s.cell_types is not None and set(s.cell_types.tolist()) == {meshutil.VTK_QUAD}
    assert s.fields["sample_data"].location == "cell" and s.fields["sample_data"].shape == (96, 1)

    adh = fetch_sample(
        "https://raw.githubusercontent.com/Deltares/xugrid/main/data/ADH_SanDiego.nc"
    )
    rep = check_connector(c, adh, tmp_path / "adh", times="0:49:12")
    d = rep.description
    assert d.zones[0].n_points == 9140 and d.zones[0].n_cells == 16869 and d.time.n_steps == 49
    s = rep.first()
    assert s.cell_types is not None and set(s.cell_types.tolist()) == {meshutil.VTK_TRIANGLE}
    assert s.cells is not None and int(s.cells.min()) == 0 and int(s.cells.max()) == 9139
    depth = s.fields["depth"]
    assert depth.shape == (5, 9140, 1) and depth.location == "point"
    assert s.times is not None and s.times[0] == 0.0 and s.times[-1] == 86400.0
    assert len(s.times) == 5 and np.all(np.diff(s.times) > 0)
    assert "elevation" in s.static
    # triangles are consistently oriented (positive area): node order survived the read
    p, t = np.asarray(s.points), np.asarray(s.cells)
    a, b = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    assert (area > 0).all() or (area < 0).all()


@pytest.mark.network
def test_public_pdebench_shock_tube(tmp_path: Path) -> None:
    """``Sod1.hdf5`` (1 MB) from the PDEBench dataset on DaRUS: a 1-D compressible shock tube
    written by PDEBench's own solver (``1D/CFD/Test/ShockTube``)."""
    path = fetch_sample("https://darus.uni-stuttgart.de/api/access/datafile/133145", "Sod1.hdf5")
    rep = check_connector(Hdf5Connector(), path, tmp_path / "out", min_confidence=0.2)
    d = rep.description
    assert d.zones[0].shape == [1024] and d.time.n_steps == 41
    assert (
        d.time.dt == pytest.approx(0.01, rel=1e-5)
        and d.metadata["mapping_source"] == "auto-detected"
    )
    assert {f.name for f in d.fields} == {"Vx", "density", "pressure"}
    assert any("42 entries for 41 steps" in w for w in d.warnings)
    s = rep.first()
    rho = s.fields["density"]
    assert not isinstance(rho.data, np.ndarray) and rho.dims == ["time", "x", "component"]
    first = np.asarray(rho.data[0])[:, 0]
    x = s.coords["x"]
    assert x[0] == pytest.approx(-1.0, abs=2e-3) and x[-1] == pytest.approx(1.0, abs=2e-3)
    # Toro's shock-tube test 1: (rho, u, p) = (1, 0.75, 1) on the left, (0.125, 0, 0.1) right
    assert first[0] == pytest.approx(1.0) and first[-1] == pytest.approx(0.125)
    assert np.all(np.diff(first) <= 0) and set(np.unique(first).round(3)) == {0.125, 1.0}
    p0 = np.asarray(s.fields["pressure"].data[0])[:, 0]
    assert p0[0] == pytest.approx(1.0) and p0[-1] == pytest.approx(0.1)
    u0 = np.asarray(s.fields["Vx"].data[0])[:, 0]
    assert u0[0] == pytest.approx(0.75) and u0[-1] == 0.0
    # later the gas between the rarefaction and the shock moves faster than either side
    assert float(np.asarray(s.fields["Vx"].data[40]).max()) > 1.2
