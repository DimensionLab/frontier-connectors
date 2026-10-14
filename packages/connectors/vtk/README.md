# frontier-connector-vtk

Reads VTK XML, legacy and VTKHDF files, and PVD and .series time series.

## Install

```bash
python -m frontier_connectors plugins install vtk
python -m frontier_connectors plugins info vtk
```

## Getting the data out

ParaView: File > Save Data (.vtu/.vtm, or .pvd with 'Write time series'). OpenFOAM: foamToVTK. SU2/Palabos/Flow360/Luminary/SimScale write VTU directly.

## Requirements and licence

Licence: Apache-2.0. Installs `vtk>=9.3`.
