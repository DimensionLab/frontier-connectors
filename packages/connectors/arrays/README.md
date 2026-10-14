# frontier-connector-arrays

Reads NetCDF 3/4, generic HDF5 and Zarr v2/v3. Common layouts are detected automatically; any other layout takes a mapping config.

## Install

```bash
python -m frontier_connectors plugins install arrays
python -m frontier_connectors plugins info arrays
```

## Status

**`hdf5`** (`verified-public-sample`): Detects datasets with dimension scales, The Well, PDEBench / FNO-style grids and points + cells mesh layouts. Any other layout needs a mapping config: the error prints the dataset tree and a mapping skeleton. The Well, dimension-scale and mesh layouts are validated on generated files only. Fluent CFF, CGNS, MATLAB v7.3, VTKHDF, Exodus, XDMF heavy data, MED and PyFR files belong to their own connectors.

**`netcdf`** (`verified-public-sample`): NetCDF-4 is read with h5py, NetCDF-3 classic and 64-bit-offset with scipy (memory-mapped). CF grids with 1-D coordinates and UGRID 2-D face meshes are detected automatically. Curvilinear grids, CDF-5, UGRID 1-D / 3-D-layered and DSG feature types need a mapping or options={'engine': 'xarray'}.

**`zarr`** (`verified-synthetic`): Reads directory and .zip stores in Zarr v2 and v3. Dimension names come from \_ARRAY_DIMENSIONS (v2) or dimension_names (v3); CF attributes are honoured as for NetCDF. Remote stores (s3://, gs://) are not opened. Validated on generated files only.

## Getting the data out

**`hdf5`**: Write the arrays with h5py / HDF5 from any solver or script, then describe the layout with a mapping config (--mapping file.json) unless it is one of the auto-detected layouts (dimension scales, PDEBench, The Well, points+cells+fields).

**`netcdf`**: Most climate/ocean/weather models and many CFD codes write NetCDF directly; from Python use xarray.Dataset.to_netcdf, from ParaView/VisIt use the NetCDF writer. OPeNDAP URLs, GRIB and multi-file datasets go through options={'engine': 'xarray'}.

**`zarr`**: xarray: Dataset.to_zarr(path). Any zarr group written by a solver or script can be described with a mapping config. A frontier-dataset directory (or one of its sample stores) is read back as-is.

## Requirements and licence

Licence: Apache-2.0. Installs `h5py>=3.10`, `scipy>=1.13`, `zarr>=3.0`.

| Optional package | Licence    | Used to                                                    | Note |
| ---------------- | ---------- | ---------------------------------------------------------- | ---- |
| `xarray`         | Apache-2.0 | open GRIB, OPeNDAP and multi-file datasets (engine=xarray) |      |
