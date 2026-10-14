# frontier-connector-palabos

Reads Palabos VTK image output directories (.vti, .vts) with parameters from plbLog.dat.

## Install

```bash
python -m frontier_connectors plugins install palabos
python -m frontier_connectors plugins info palabos
```

## Status

`verified-public-sample`. Reads Palabos output directories (.vti series and .vts states) with Re, resolution, relaxation frequency, dx and dt from plbLog.dat. An output directory is recognised by plbLog.dat next to its VTK files; file numbers are iterations, converted to time with the log's dt, and values are in the units the application wrote. The XML parameter file is not read. 3-D volume series with a log, parallel .pvti and sparse multi-block output are not validated.

## Getting the data out

No export step: Palabos writes VTK image data (VtkImageOutput2D/3D -> tmp/vtk*.vti, VtkStructuredImageOutput -> .vts). Point at the output directory (the one holding plbLog.dat) or at a single .vti file.

## Requirements and licence

Licence: Apache-2.0.
