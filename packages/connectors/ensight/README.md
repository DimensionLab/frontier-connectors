# frontier-connector-ensight

Reads EnSight Gold and EnSight 6 case files, steady and transient, ASCII and binary.

## Install

```bash
python -m frontier_connectors plugins install ensight
python -m frontier_connectors plugins info ensight
```

## Status

`verified-public-sample`. Reads ASCII, C binary and Fortran binary files: unstructured and structured parts, several time sets, moving geometry. Parts of the highest dimension are read by default, boundary parts with zones=[...]. Server-of-servers (.sos) files are not read.

## Getting the data out

Export 'EnSight Gold' from the solver (STAR-CCM+: File > Export or the shipped macro; Fluent: File > Export > EnSight Case Gold; CONVERGE: post_convert; OpenFOAM: foamToEnsight) and point at the .case file.

## Requirements and licence

Licence: Apache-2.0. Installs `vtk>=9.3`.
