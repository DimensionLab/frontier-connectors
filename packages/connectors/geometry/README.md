# frontier-connector-geometry

Reads STL, OBJ, PLY and OFF surfaces. STEP, IGES and BREP need an optional tessellator (cadquery-ocp, or gmsh).

## Install

```bash
python -m frontier_connectors plugins install geometry
python -m frontier_connectors plugins info geometry
```

## Status

`verified-public-sample`. STL, OBJ, PLY and OFF are read with built-in parsers. STEP, IGES and BREP need cadquery-ocp, or gmsh on request. OBJ materials and textures, PLY tristrips and binary STL colour attributes are ignored.

## Getting the data out

Any CAD or meshing tool exports STL / OBJ / PLY. For STEP / IGES / BREP install cadquery-ocp (pip install cadquery-ocp) and tune options tessellation={'linear_deflection': ..., 'angular_deflection': ...}. Put a design sweep in one directory to convert it as one dataset.

## Requirements and licence

Licence: Apache-2.0.

| Optional package | Licence | Used to | Note |
|---|---|---|---|
| `cadquery-ocp` | Apache-2.0 | tessellate STEP / IGES / BREP | wraps OpenCascade (LGPL-2.1 with exception) |
| `gmsh` | GPL-2.0-or-later | tessellate CAD with gmsh (only with tessellator='gmsh') | GPL: never imported unless requested |
