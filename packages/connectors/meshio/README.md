# frontier-connector-meshio

Reads Exodus II, XDMF, Gmsh MSH and MED files, SU2 meshes, and the other mesh formats meshio supports.

## Install

```bash
python -m frontier_connectors plugins install meshio
python -m frontier_connectors plugins info meshio
```

## Status

**`exodus`** (`verified-public-sample`): Element blocks are the default zones; side sets and node sets are extra zones. Reads nodal, element and global variables at all time steps. One piece of a parallel decomposition is read as it is, without merging.

**`gmsh`** (`verified-public-sample`): Physical groups become zones. $NodeData and $ElementData views are read, including several time steps of one view; element views assume the consecutive element tags Gmsh writes. $ElementNodeData and .pos parsed views are not read. MSH 4 files need every saved entity in a physical group. Views and binary files are validated on generated files only.

**`med`** (`verified-public-sample`): Groups (families) become zones; node groups are point-cloud zones. Reads fields on nodes and cells for every stored time step. Gauss-point (ELGA) and element-node (ELNO) fields are averaged per cell; their storage order for more than one cell is not validated on a real file. Fields on a profile are NaN outside it. Structured MED meshes and polyhedra are not read. Several time steps are validated on generated files only.

**`meshio`** (`verified-public-sample`): Reads Medit .mesh / .meshb, Netgen .vol, TetGen .node / .ele, UGRID and FLAC3D .f3grid. Region and material ids become zones. MOAB .h5m, DOLFIN XML, AVS-UCD, PERMAS, Kratos .mdpa and HMF are validated on meshio-written files only.

**`su2`** (`verified-public-sample`): Reads the mesh, markers as boundary zones and multi-zone meshes; solver, turbulence model, free-stream parameters and the boundary condition of each marker come from a .cfg next to the mesh. No solution fields.

**`xdmf`** (`verified-public-sample`): Unstructured grids and meshio time series are read through meshio; structured topologies (CoRect, Rect, SMesh), spatial collections and dolfinx / ParaView temporal collections through VTK's Xdmf2 reader. FiniteElementFunction (dolfinx checkpoint) attributes are not evaluated: only the mesh of such files is read. Cell-centred data of regular grids is averaged to the grid nodes.

## Getting the data out

**`exodus`**: MOOSE: [Outputs] exodus = true. Sierra/SEACAS/Cubit/libMesh write .e/.exo/.g directly; join parallel pieces (file.e.N.i) with SEACAS epu first.

**`gmsh`**: Gmsh: File > Export > .msh (version 4.1 or 2.2; tick 'Save all elements' only if you have no physical groups). Post-processing views: Save 'Mesh-based' views into the .msh; parsed-view .pos files must be re-saved that way.

**`med`**: Code_Aster: IMPR_RESU(FORMAT='MED', RESU=\_F(RESULTAT=...)); Salome: File > Export > MED; Code_Saturne: writer format 'MED'. MED 2.1 files need Salome's medimport.

**`meshio`**: These are mesh formats: results usually live elsewhere (FUN3D: Tecplot/VTK, NGSolve: VTK, FEniCS: XDMF/PVD). Any other meshio format can be converted with `meshio convert in.xxx out.vtu` and read by the vtk connector.

**`su2`**: The .su2 file is the mesh only. SU2 results are written as VTU/PVD (OUTPUT_FILES= PARAVIEW), Tecplot, CSV (surface_flow.csv, restart_flow.csv) or CGNS: read those with the vtk, tecplot, tabular or cgns connectors.

**`xdmf`**: dolfinx: XDMFFile.write_mesh / write_function(u, t); legacy FEniCS: XDMFFile.write; ParaView: Save Data as Xdmf. Keep the .h5 heavy-data file next to the .xdmf.

## Requirements and licence

Licence: Apache-2.0. Installs `meshio>=5.3`, `vtk>=9.3`, `h5py>=3.10`.
