# frontier-connector-openfoam

Reads OpenFOAM case directories (Foundation, ESI, foam-extend): ASCII, binary and gzip, serial and decomposed, with moving meshes.

## Install

```bash
python -m frontier_connectors plugins install openfoam
python -m frontier_connectors plugins info openfoam
```

## Status

`verified-public-sample`. Reads ASCII, binary and gzip cases, serial and decomposed (uncollated), with moving points; multi-region with options={'region': ...}. Zones: internalMesh (default) and patch/<name> (opt-in). Steady-solver cases read the last iteration unless options={'iterations': 'all'}. Not read: surface (flux) fields, lagrangian clouds, cell and face zones, collated decomposition.

## Getting the data out

Point at the case directory (system/, constant/polyMesh, time directories), a .foam stub inside it or system/controlDict. Decomposed cases (processor\*/) are read directly; run reconstructPar first for the collated format (processors<N>/).

## Requirements and licence

Licence: Apache-2.0. Installs `vtk>=9.3`, `scipy>=1.13`.
