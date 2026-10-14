# frontier-connector-cgns

Reads CGNS/HDF5 with h5py and ADF files through VTK.

## Install

```bash
python -m frontier_connectors plugins install cgns
python -m frontier_connectors plugins info cgns
```

## Status

`verified-public-sample`. Reads CGNS/HDF5 with h5py: structured and unstructured zones, linear and quadratic elements, MIXED, NGON_n / NFACE_n, boundary patches as surface zones, vertex and cell-centred solutions, rind, links, FlowSolutionPointers, file-per-step series, units, reference state and flow equations. Cubic and higher elements, face-centred solutions and non-Cartesian coordinates are not supported. ADF files and options={'engine': 'vtk'} go through VTK's vtkCGNSReader, without the CGNS metadata.

## Getting the data out

Export CGNS from the solver or post-processor (Ansys CFX: cfx5export -cgns; Fluent: File > Export > CGNS; STAR-CCM+: File > Export > CGNS; Fidelity/FINE, SU2, elsA, Pointwise write it natively) and point at the .cgns file or a directory holding a numbered series.

## Requirements and licence

Licence: Apache-2.0. Installs `h5py>=3.10`, `vtk>=9.3`.
