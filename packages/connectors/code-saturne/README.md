# frontier-connector-code-saturne

Reads code_saturne RESU/<run>/postprocessing output (EnSight Gold, MED or CGNS) with the physics from setup.xml and run_solver.log.

## Install

```bash
python -m frontier_connectors plugins install code-saturne
python -m frontier_connectors plugins info code-saturne
```

## Status

`verified-public-sample`. Reads RESU/<run>/postprocessing output (code_saturne 9.1, FVM 0.3): mesh including polyhedra, cell fields, solver version, turbulence model, fluid properties, notebook variables and boundary natures. Transient EnSight output and the MED and CGNS writers are validated on generated files only.

## Getting the data out

No export step: code_saturne writes EnSight Gold by default into RESU/<run>/postprocessing/ (RESULTS_FLUID_DOMAIN.case; MED or CGNS when selected in the writer settings). Point at the case directory, the run directory or the .case; setup.xml and run_solver.log next to postprocessing/ supply the physics.

## Requirements and licence

Licence: Apache-2.0.
