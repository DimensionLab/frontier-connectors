# frontier-connector-pyfr

Reads PyFR native mesh (.pyfrm) and solution (.pyfrs) files, PyFR <= 2.1 and 3.x. Validated on generated files and on PyFR 3.1 and 2.1 output.

## Install

```bash
python -m frontier_connectors plugins install pyfr
python -m frontier_connectors plugins info pyfr
```

## Status

`verified-synthetic`. Reads PyFR <= 2.1 files (spt*\* / soln*\* datasets) and PyFR 3.x files (eles / nodes, <prefix>/p<order>-<etype> with 'pts'). A solution becomes a point cloud of solution points (quad, hex, tri, tet, pri; legacy files only quad and hex with Gauss-Legendre or Lobatto points; no pyramids, no region subsets); a mesh alone becomes a linear corner mesh. Validated on generated files and on PyFR 3.1 and 2.1 output.

## Getting the data out

PyFR writes soln-<t>.pyfrs next to the .pyfrm mesh; read them directly as a cloud of solution points, or run `pyfr export mesh.pyfrm soln.pyfrs out.vtu` and read the VTU with the vtk connector when cell connectivity is needed.

## Requirements and licence

Licence: Apache-2.0. Installs `h5py>=3.10`.
