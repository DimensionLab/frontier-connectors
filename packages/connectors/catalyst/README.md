# frontier-connector-catalyst

Reads Conduit Mesh Blueprint steps from a spool directory, Conduit relay files or ADIOS2 (BP5 / SST), during or after the run. Not validated against ParaView Catalyst.

## Install

```bash
python -m frontier_connectors plugins install catalyst
python -m frontier_connectors plugins info catalyst
```

A running solver feeds a Frontier dataset step by step. The unit of exchange is the Conduit Mesh
Blueprint tree, the same tree a ParaView Catalyst 2 adaptor passes under
`catalyst/channels/<name>/data`.

```bash
python -m frontier_connectors import catalyst run/spool --out dataset --option timeout=600
python -m frontier_connectors import catalyst run/spool --out dataset --dry-run
python -m frontier_connectors convert run/spool --out dataset      # after the run
```

## What it reads

| Source | Written by | Needs |
| --- | --- | --- |
| spool directory, `frontier-spool` v1 (`frontier-stream.json`, `step_<n>.npz`, `frontier-stream.done`) | `frontier_emit.py` in the solver or in the Catalyst pipeline script, both in `exporter-catalyst` | numpy |
| Conduit relay Blueprint series (`<name>.cycle_<n>.root` + domain files; hdf5, json, yaml, conduit_json, conduit_base64_json, conduit_bin) | `conduit.relay.io.blueprint.save_mesh`, Ascent relay extracts | `h5py` for hdf5, `pyyaml` for yaml; Conduit itself is not needed |
| ADIOS2 BP4 / BP5 file (also while written) or SST stream | variables named by Blueprint path, or the Fides predefined-model attributes | `adios2` (Apache-2.0) |
| `conduit.Node` or nested `dict` in memory | `frontier_connector_catalyst.blueprint.mesh_to_samples`, `StepConverter` | LLNL Conduit only for `Node` objects |

Files a Catalyst pipeline writes with plain VTK extractors (`.vtpd`, `.vtm`, VTKHDF) are read by
the `vtk` connector.

## Blueprint coverage

- Coordsets: `uniform`, `rectilinear`, `explicit` (Cartesian).
- Topologies: `points`, `uniform`, `rectilinear`, `structured`, `unstructured` with
  `point line tri quad tet hex wedge pyramid`, `polygonal`, `polyhedral` (with `subelements`) and
  `mixed` (`shape_map`).
- Fields: `vertex` or `element`, scalar or mcarray, `units`.
- `state/{time,cycle,domain_id}`, multi-domain trees, Catalyst `mesh` and `multimesh` channels.

Uniform and rectilinear topologies become `grid` samples, all-2-D topologies `surface`
(triangulated, cell data follows), point topologies `point_cloud`, the rest `mesh`. One zone per
(domain, topology); MPI-decomposed domains stay separate zones.

Not supported: non-Cartesian coordsets, strided structured topologies, high-order meshes and
fields. Skipped with a warning: material fields, matsets, specsets, adjsets, nestsets.

## Following a live stream

`--option follow=true` (the importer's default) tails the source. Each step is yielded once, in
increasing step number, and only when complete; the dataset is appended along time.

| Option | Default | Meaning |
| --- | --- | --- |
| `timeout` | 60 | seconds without a new complete step before giving up; also the wait for the stream to appear |
| `poll` | 0.2 | seconds between looks |
| `until` | `done-marker` | `timeout`: ignore end markers, stop after `timeout` idle seconds (relay series) |
| `on_timeout` | `stop` | keep what arrived; `error` raises and leaves no dataset |
| `on_restart` | `skip` | steps re-emitted after a restart from a checkpoint are not read twice; `error` raises |
| `order_grace` | 2 | seconds to wait for a step the producer's `prev` chain says is missing |
| `partial_grace` | 5 | seconds an unreadable visible file may stay unchanged before it is declared corrupt |
| `settle` | 0.5 | seconds a relay file must be unmodified before it is read |
| `max_steps` | - | stop after N steps |
| `geometry` | `changed` | `always` sends the mesh with every step (needed with `--time-as-samples` for changing topology) |
| `name` | - | relay series base name when a directory holds several |
| `engine`, `adios_parameters`, `fides_order` | `BPFile`, -, `zyx` | ADIOS2 engine (`BP5`, `BP4`, `SST`), engine parameters, grid array order of Fides models |

While following, `--times` accepts `all` or a forward range `start:stop:step`. Timeouts,
restarts, lost steps and failed end markers are logged and added to the conversion warnings.
Hidden `.<name>.tmp-<pid>` files are never read. A visible file that does not decode is retried.

## ADIOS2 variable convention

Each Blueprint array is a variable named by its path (`coordsets/coords/values/x`,
`topologies/mesh/elements/connectivity`, `fields/T/values`, `state/time`, `state/cycle`), and
each string leaf an attribute (`topologies/mesh/type = "unstructured"`,
`fields/T/association = "vertex"`, `fields/T/units = "K"`). The mesh may be written in the first
step only. Several domains share a prefix (`fluid/coordsets/...`). The attributes
`frontier/solver`, `frontier/solver_version`, `frontier/domain` and `frontier/time_units` go to
the dataset metadata. The Fides predefined models `uniform`, `rectilinear`,
`unstructured_single` and `unstructured` are translated; the Fides JSON file, XGC and GTC are not.

## Status

`verified-synthetic`. Reads frontier-spool directories, Conduit relay Blueprint files (hdf5, json, yaml, conduit_*) and ADIOS2 BP4 / BP5 files and BP5 / SST streams, during or after the run. Validated with Conduit 0.9.5 and ADIOS2 2.12.1. Not validated against ParaView Catalyst (the pipeline script), Fides or MPI-parallel writers.

## Getting the data out

Get the solver-side scripts with `python -m frontier_connectors plugins install exporter-catalyst`, then use one of: (1) copy frontier_emit.py (numpy only) next to the solver and call Emitter.write_step(step, time, blueprint_mesh) at each output step; (2) run the Catalyst 2 pipeline frontier_catalyst_pipeline.py (CATALYST_IMPLEMENTATION_NAME=paraview), which writes the same spool; (3) write Conduit relay Blueprint files (relay.io.blueprint.save_mesh, Ascent) or ADIOS2 BP5 / SST with Blueprint paths as variable names. Follow the run with `python -m frontier_connectors import catalyst <spool|stream> --out ds`. Files a Catalyst pipeline writes with plain VTK extractors (.vtpd / .vtm series, VTKHDF) are read by the `vtk` connector.

## Requirements and licence

Licence: Apache-2.0.

| Optional package | Licence | Used to | Note |
|---|---|---|---|
| `adios2` | Apache-2.0 | read ADIOS2 BP4/BP5 files and follow BP5/SST streams |  |
| `h5py` | BSD-3-Clause | read Conduit relay Blueprint files written with the hdf5 protocol |  |
| `pyyaml` | MIT | read Conduit relay Blueprint files written with the yaml protocol |  |
| `conduit` | BSD-3-Clause | pass conduit.Node objects to the Blueprint bridge directly | LLNL Conduit is not on PyPI (the PyPI project named conduit is unrelated): build it from https://github.com/LLNL/conduit or use Spack |
