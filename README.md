# frontier-connectors

Connector plugins that turn CFD/CAE solver output (OpenFOAM, VTK, Fluent, STAR-CCM+, CGNS, LS-DYNA, ...) into Frontier datasets. Write your own in an afternoon.

Years of solver output sit in case directories nobody can train on. A connector reads one
tool's native output and yields solver-neutral samples (mesh, fields with named dimensions,
time steps, units, solver metadata) that are written as a
[Frontier dataset](https://github.com/DimensionLab/frontier-format): chunked Zarr that a training
loop, a viewer or a catalog can read without the solver installed.

```
pip install frontier-connectors
python -m frontier_connectors list                 # what is installed, what needs an extra
python -m frontier_connectors detect  path/to/case
python -m frontier_connectors describe path/to/case
python -m frontier_connectors convert path/to/case --to datasets/my-cases
```

## What is in this repository

- `crates/frontier-connector-openfoam/`: Rust crate `frontier-connector-openfoam`
- `crates/frontier-connector-vtk/`: Rust crate `frontier-connector-vtk`
- `crates/frontier-connectors/`: Rust crate `frontier-connectors`
- `packages/connectors/arrays/`: Python package `frontier-connector-arrays`
- `packages/connectors/catalyst/`: Python package `frontier-connector-catalyst`
- `packages/connectors/cgns/`: Python package `frontier-connector-cgns`
- `packages/connectors/code-saturne/`: Python package `frontier-connector-code-saturne`
- `packages/connectors/ensight/`: Python package `frontier-connector-ensight`
- `packages/connectors/exporter-catalyst/`: Python package `frontier-exporter-catalyst`
- `packages/connectors/geometry/`: Python package `frontier-connector-geometry`
- `packages/connectors/meshio/`: Python package `frontier-connector-meshio`
- `packages/connectors/nek5000/`: Python package `frontier-connector-nek5000`
- `packages/connectors/openfoam/`: Python package `frontier-connector-openfoam`
- `packages/connectors/oss/`: Python package `frontier-connectors-oss`
- `packages/connectors/palabos/`: Python package `frontier-connector-palabos`
- `packages/connectors/pyfr/`: Python package `frontier-connector-pyfr`
- `packages/connectors/tabular/`: Python package `frontier-connector-tabular`
- `packages/connectors/vtk/`: Python package `frontier-connector-vtk`
- `packages/frontier-connectors/`: Python package `frontier-connectors`

The Python package is where formats are read. The Rust crates define the same contract for
the native pipeline and keep `detect` fast enough to walk an archive of millions of files.

## Write a connector

A connector is a class with three methods, registered through a Python entry point. It
lives in *your* package and *your* repository; nothing here needs to change for Frontier
(the open-source tools and the commercial platform alike) to discover it.

```python
# my_solver_connector/__init__.py
from pathlib import Path
from frontier_connectors import BaseConnector, Detection, RawSample, SourceDescription


class MySolverConnector(BaseConnector):
    id = "my-solver"
    name = "MySolver results"
    formats = ["mysolver-h5"]
    vendor = "Example Corp"
    license = "Apache-2.0"  # licence of THIS connector
    extensions = [".msr"]

    def detect(
        self, path: Path
    ) -> Detection | None: ...  # cheap: look at names and magic bytes, never parse

    def describe(
        self, path: Path
    ) -> SourceDescription: ...  # zones, fields, time steps, units, solver settings

    def read(self, path, *, zones=None, fields=None, times=None):
        yield RawSample(...)  # one sample at a time; never load an archive eagerly
```

```toml
# pyproject.toml of your package
[project.entry-points."frontier.connectors"]
my-solver = "my_solver_connector:MySolverConnector"
```

`pip install` your package next to `frontier-connectors` and it appears in
`python -m frontier_connectors list`, in the `frontier` CLI, and in the dataset builder of
the Frontier platform. `frontier_connectors.testing` has a conformance suite to run against
your connector in CI.

### Rules every connector follows

- **No vendor licence at import time.** Importing a connector, listing it and calling
  `detect` must work on a machine without the solver. Readers that need vendor software
  ship an *exporter script* that runs inside the vendor tool and a reader for what the
  script writes.
- **Lazy.** `read` is a generator; a 10^8-cell transient case is streamed step by step.
- **Honest metadata.** `license` states the licence of the connector, `requires` the
  optional packages it needs, `maturity` how far it has been tested.
- **No proprietary bytes in the repository.** Tests use synthetic files or small public
  samples with a stated source.

### Compatibility

Connectors target a *data-model version* (`frontier_core::DATA_MODEL_VERSION`, mirrored in
the Python package). Frontier loads a plugin when its data-model version matches and
reports the mismatch otherwise; the version changes only with a major release of the
dataset format, announced one minor release ahead.

## Ecosystem

| Repository | What it is |
|---|---|
| [frontier-format](https://github.com/DimensionLab/frontier-format) | Frontier data model and dataset format: the frontier-dataset specification, the plugin contract, and reference readers/writers in Rust and Python. |
| **frontier-connectors** (this repository) | Connector plugins that turn CFD/CAE solver output (OpenFOAM, VTK, Fluent, STAR-CCM+, CGNS, LS-DYNA, ...) into Frontier datasets. Write your own in an afternoon. |
| [frontier-sdk](https://github.com/DimensionLab/frontier-sdk) | Frontier Python SDK: offline-first experiment tracking for physics ML, the native Rust core, and the NVIDIA PhysicsNeMo bridge. |
| [frontier-physicsnemo](https://github.com/DimensionLab/frontier-physicsnemo) | NVIDIA PhysicsNeMo plugin for Frontier: logging, model registry, checkpoint compatibility and dataset adapters. Installable independently of the platform. |
| [frontier-hub](https://github.com/DimensionLab/frontier-hub) | Curated hub of public physics-AI datasets and benchmarks: catalog, fetchers and converters to the Frontier dataset format. |
| [frontier-editor](https://github.com/DimensionLab/frontier-editor) | Editor core and GPU text view for physics-ML configs, recipes and logs: rope document, undo tree, hunks with per-hunk approval, tree-sitter syntax, JSON-Schema diagnostics, provenance events, and a wgpu renderer that draws into the host's render pass. |
| [frontier-viz](https://github.com/DimensionLab/frontier-viz) | GPU-accelerated (wgpu) rendering core for meshes, fields and neural-network topology: native, browser (WebGPU) and headless. |
| [foldstate](https://github.com/DimensionLab/foldstate) | Statecharts as data with an event log: a pure, deterministic kernel (state, event) -> (state, effects), replay, fork, restore and explanations. no_std + alloc, runs natively and in WebAssembly. Not specific to Frontier. |
| [frontier-zed](https://github.com/DimensionLab/frontier-zed) | Frontier for Zed: JSON schemas, snippets and the Frontier MCP context server for physics-ML training configs, RunSpecs, data-prep recipes, pipelines, hub entries and plugin manifests. A Zed extension (extension.toml plus a small wasm32-wasip2 crate), installable as a dev extension today and submitted to the Zed extension registry later. |

Orchestration across SLURM/Kubernetes/cloud, the canvas application, V&V reporting and
enterprise governance are part of the commercial [Frontier platform](https://dimensionlab.org),
which uses these same open components.

## Contributing

New formats are the most valuable contribution: see [CONTRIBUTING.md](CONTRIBUTING.md) and
open a *Connector request* issue first. Licensed under [Apache-2.0](LICENSE).
