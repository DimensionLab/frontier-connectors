# frontier-exporter-catalyst

A numpy-only step emitter, a ParaView Catalyst 2 pipeline script and adaptor notes that run next to the solver and write what the catalyst connector reads. The pipeline script is not validated against ParaView.

## Install

```bash
python -m frontier_connectors plugins install exporter-catalyst
python -m frontier_connectors plugins info exporter-catalyst
```

Copy the scripts next to a case:

```bash
python -m frontier_connectors plugins install-exporter catalyst --to <dir>
```

Files: `README.txt`, `adaptor_example.txt`, `frontier_catalyst_pipeline.py`, `frontier_emit.py`. Usage is in `README.txt`.

| File | Purpose |
| --- | --- |
| `frontier_emit.py` | numpy-only emitter of the `frontier-spool` protocol: atomic step files, header, end marker, restart support, mesh builders. It imports nothing from Frontier; copy it into the solver repository. |
| `frontier_catalyst_pipeline.py` | ParaView Catalyst 2 pipeline script. `--mode spool` converts a channel's VTK data to Blueprint and emits it (serial); `--mode vtk` adds a VTPD extractor (parallel; read with the `vtk` connector). Not validated against ParaView. |
| `adaptor_example.txt` | the Conduit node a Catalyst 2 adaptor passes (C++, Fortran, Python), `CATALYST_IMPLEMENTATION_NAME` / `CATALYST_IMPLEMENTATION_PATHS`, other transports |

```python
from frontier_emit import Emitter, uniform_mesh

with Emitter("spool", solver="my-solver") as out:  # writes the end marker on exit
    for step in range(n_steps):
        ...
        out.write_step(step, t, uniform_mesh((nx, ny), spacing=(dx, dy), point_fields={"T": T}))
```

```bash
python -m frontier_connectors import catalyst spool --out dataset      # while the solver runs
```

## Status

`frontier_emit.py` is validated. The Catalyst pipeline script is not validated against ParaView.

## Requirements and licence

Licence: Apache-2.0.
