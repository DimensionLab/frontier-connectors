# Writing a connector plugin

A connector turns the output of one simulation tool into `RawSample` objects; the framework
writes them as a `frontier-dataset` v1.1 directory. A connector is a small Python
distribution. Nothing in `frontier-connectors` has to change for yours to appear in
`python -m frontier_connectors list`, and users can install, disable and uninstall it like
the first-party ones.

There are three kinds of plugin, and one distribution may provide more than one:

| Kind | What it does | Entry-point group |
|---|---|---|
| connector | reads a path: `detect`, `describe`, `read` | `frontier.connectors` |
| exporter | carries scripts that run inside the vendor tool (a macro, a journal, a MATLAB package) | `frontier.exporters` |
| importer | gets data that is not a local file yet: a cloud API, a live solver session, an in-situ stream | `frontier.importers` |

## A complete connector in one file

```
acme-frontier-xyzp/
  pyproject.toml
  src/acme_frontier_xyzp/
    __init__.py
    plugin.toml
    connector.py
```

`connector.py`:

```python
from pathlib import Path

import numpy as np

import frontier_connectors as fc


class XyzpConnector(fc.BaseConnector):
    id = "acme-xyzp"
    name = "ACME probe files (x y z p)"
    formats = ["acme-xyzp"]
    vendor = "ACME"
    license = "Apache-2.0"
    extensions = [".xyzp"]
    maturity = "verified-synthetic"
    temporal = ["steady"]
    kinds = ["point_cloud"]
    export_help = "ACME Solver: File > Export > Probe table (.xyzp)."

    def describe(self, path, *, options=None):
        n = sum(1 for _ in Path(path).open())
        return fc.SourceDescription(
            connector_id=self.id,
            path=Path(path),
            format="acme-xyzp",
            solver="ACME Solver",
            domain="cfd",
            zones=[fc.ZoneInfo(name="probes", kind="point_cloud", role="probes", n_points=n)],
            fields=[fc.FieldInfo(name="p", location="point", units="Pa")],
        )

    def read(self, path, *, zones=None, fields=None, times=None, options=None):
        a = np.loadtxt(path, ndmin=2)
        yield fc.RawSample(
            id=Path(path).stem,
            kind="point_cloud",
            zone="probes",
            points=a[:, :3].astype(np.float32),
            fields={"p": fc.as_field(a[:, 3], "point", units="Pa")},
            metadata={"solver": "ACME Solver"},
        )
```

`BaseConnector.detect` matches `extensions`. Override it when the format has magic bytes or a
directory layout, and return a `Detection` with a confidence: 1.0 for an unambiguous
signature, about 0.6 for an extension, below 0.5 for a generic container you might fit.
Never claim a generic file (`.h5`, `.csv`, `.case`) on the extension alone.

`plugin.toml`, the manifest:

```toml
schema = 1
id = "acme-xyzp"                 # [a-z0-9-], unique among installed plugins
kind = "connector"               # connector | exporter | importer
name = "ACME probe files"
summary = "x y z p text files written by ACME Solver"
version = "1.0.0"                # must equal the distribution version
data_model_version = "1.1"       # the frontier-dataset version you wrote against
vendor = "ACME"
formats = ["acme-xyzp"]
maturity = "verified-synthetic"
licence = "Apache-2.0"
requires = []                    # pip requirements (repeat them in pyproject.toml)
plugins = []                     # ids of other Frontier plugins you delegate to
suggests = []                    # e.g. ["exporter-acme"]: installed alongside by default

[connectors]
acme-xyzp = "acme_frontier_xyzp.connector:XyzpConnector"
```

`pyproject.toml`:

```toml
[project]
name = "acme-frontier-xyzp"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = ["frontier-connectors>=0.2"]

[project.entry-points."frontier.plugins"]
acme-xyzp = "acme_frontier_xyzp"          # the import package that holds plugin.toml

[project.entry-points."frontier.connectors"]
acme-xyzp = "acme_frontier_xyzp.connector:XyzpConnector"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
acme_frontier_xyzp = ["plugin.toml"]
```

Install and use it:

```bash
python -m frontier_connectors plugins install ./acme-frontier-xyzp        # or a wheel, or git+https://...
python -m frontier_connectors plugins info acme-xyzp
python -m frontier_connectors convert probes.xyzp --out datasets/probes
python -m frontier_connectors plugins disable acme-xyzp                   # keep it installed, switch it off
python -m frontier_connectors plugins uninstall acme-xyzp
```

A distribution that only declares `frontier.connectors` (no `frontier.plugins` entry and no
`plugin.toml`) still loads. It is listed without manifest data, cannot be checked for
data-model compatibility, and `plugins doctor` says so.

## What the framework does with the manifest

The manifest is read with `tomllib` from the package directory **without importing your
code**, so a plugin whose import fails can still be listed, explained and uninstalled.

A plugin gets one of four states:

| State | Meaning | Loaded |
|---|---|---|
| `enabled` | manifest valid, compatible, not switched off | yes |
| `disabled` | switched off in `~/.frontier/plugins.toml` | no |
| `incompatible` | `data_model_version` has another major version, or a newer minor, than the framework | no |
| `invalid` | manifest missing, not TOML, a field wrong, or it disagrees with the entry points or the distribution version | no |

The compatibility rule: same major version, and your minor not newer than the framework's.
Minor versions of the dataset format only add things, so an older plugin keeps working.

One failing plugin never affects the others. An exception while importing your connector
turns it into a registry row with status `broken` and the error text; an exception in your
`detect` is logged at debug level and skipped.

## The sample model

`read` yields `RawSample` objects in one of two forms.

- **Step form**, for meshes. One sample per zone and time step, all with the same `id`. The
  first step carries `points`, `cells`, `cell_types` (VTK ids, connectivity padded with -1) or
  `faces` (triangles of a surface). Later steps set them to `None`, or pass new `points` when
  the mesh moves. Set `time` and `time_index`; field `dims` have no time axis.
- **Block form**, for grids and array stores. `times` is set and every time-dependent field
  has `dims[0] == "time"`. The data may be a lazy array (`h5py.Dataset`, `zarr.Array`,
  `numpy.memmap`); the writer slices it step by step.

Never build a whole transient in memory. `dims` names every axis and ends with
`"component"`; `fc.as_field(values, "point" | "cell")` builds the common case. Fields are
stored as float32. Put time-independent inputs (wall distance, material ids) in `static`.

`describe` must be cheap: headers and metadata only. Fill what the format records: `solver`,
`solver_version`, `domain`, `physics` (turbulence model, compressibility, ...),
`boundary_conditions` per patch, `units`, `parameters` (Reynolds number, angle of attack),
and one `ZoneInfo` per thing a user can select. Mark boundary patches of a volume case
`default=False`.

Selections arrive as `zones` and `fields` (names or globs) and `times` (see
`frontier_connectors.select`: `select_time_indices`, `match_names`, `wants`). A selection
that matches nothing raises `SelectionError`.

## Errors

Raise the classes in `frontier_connectors.errors`. Every one takes a `hint` that says what
the user should do next, and the CLI prints it.

| Error | When |
|---|---|
| `UnsupportedSourceError` | the path is not yours, or a variant you do not read |
| `NeedsExportError` | the native file is proprietary: name the exporter script and the command |
| `MissingDependencyError` | an optional package is needed for this file |
| `NeedsCredentialsError` | a cloud path without credentials: name the environment variable |
| `SelectionError` | a zone, field or time selection matched nothing |
| `MappingError` | a mapping config is missing or ambiguous |

## Dependencies and licences

- `requires` are hard dependencies. Keep them permissive (BSD, MIT, Apache).
- Anything GPL, LGPL, proprietary, or that needs a vendor licence to run is **optional**: do
  not list it in `requires`, import it inside the function that needs it, and declare it so
  that `plugins doctor` can explain it:

  ```toml
  [[optional]]
  package = "vendor-sdk"
  module = "vendorsdk"             # import name, when it differs
  licence = "LGPL-2.1-only"
  purpose = "download results through the vendor SDK"
  flag = "LGPL; needs a vendor account"
  ```

- Importing your module must work with only `requires` installed, and must never need a
  vendor licence.
- Do not copy vendor SDK code or vendor documentation into your plugin. A proprietary format
  gets an exporter script that runs inside the vendor tool plus a reader for what it writes.

## Exporter plugins

Scripts that run inside the vendor tool live in their own distribution, so they can be
installed on the machine that has the tool and versioned with the reader that understands
their output:

```
acme-frontier-exporter/src/acme_frontier_exporter/
  __init__.py
  plugin.toml
  files/export_for_frontier.mac        # anything; copied verbatim
```

```toml
schema = 1
id = "exporter-acme"
kind = "exporter"
name = "ACME Solver exporter scripts"
version = "1.0.0"
data_model_version = "1.1"
maturity = "exporter+reader"
licence = "Apache-2.0"

[exporter]
vendor = "acme"                        # the key of `install-exporter`
tool = "ACME Solver"
files = "files"
usage = "acme -batch export_for_frontier.mac case.acm"
```

```toml
[project.entry-points."frontier.plugins"]
exporter-acme = "acme_frontier_exporter"

[project.entry-points."frontier.exporters"]
acme = "acme_frontier_exporter"
```

`python -m frontier_connectors plugins install-exporter acme --to <dir>` copies `files/` to
`<dir>/acme/` and writes `.frontier-exporter.json` there with the plugin version and a hash
per file. `uninstall-exporter acme` removes exactly those files and keeps any the user
edited. In code, `frontier_connectors.plugins.exporter_dir("acme")` returns the directory.

Write a JSON sidecar from the script (solver version, physics, boundary conditions, units)
and read it in `describe`: the interchange file rarely carries that metadata.

## Importer plugins

An importer fetches; a connector reads what was fetched. Keeping them apart lets an
air-gapped site download on one machine and convert on another.

```python
from frontier_connectors.importers import BaseImporter, PlannedCall


class AcmeCloudImporter(BaseImporter):
    id = "acme-cloud"
    name = "ACME Cloud"
    vendor = "ACME"
    maturity = "implemented-unverified-live"
    credentials_env = ["ACME_API_KEY"]
    source_help = "<case-id>"
    api_reference = "https://docs.acme.example/api (read 2026-10-10)"

    def plan(self, source, *, options=None):
        return [
            PlannedCall(
                "GET",
                f"https://api.acme.example/v1/cases/{source}",
                "case metadata",
                headers={"Authorization": "$ACME_API_KEY"},
            ),
            PlannedCall("GET", f"https://api.acme.example/v1/cases/{source}/results", "download"),
        ]

    def fetch(self, source, dest, *, options=None):
        key = self.credentials()["ACME_API_KEY"]  # raises NeedsCredentialsError when unset
        ...  # download into dest
        return dest
```

Register it under `[importers]` in `plugin.toml` and
`[project.entry-points."frontier.importers"]`. Then:

```bash
python -m frontier_connectors import acme-cloud case-42 --dry-run     # prints plan(), contacts nothing
python -m frontier_connectors import acme-cloud case-42 --out datasets/case-42
```

Rules: credentials come from environment variables only, never appear in `PlannedCall`
(put the variable name there), in logs, in error messages or in files you write; `plan`
must work without credentials and without the vendor SDK installed; plain `http://` is
refused.

## Maturity

Pick the level that matches how the connector was validated. `status_note` is printed next to it
in the support matrix: say what is read, what is not, and the limit of validation in one
sentence (`Experimental. Not validated against a licensed installation.`).

| Level | Meaning |
|---|---|
| `verified-public-sample` | validated on files written by the tool itself |
| `verified-synthetic` | validated on generated files only |
| `exporter+reader` | exporter script plus reader; the script is not validated against a licensed installation |
| `implemented-unverified-live` | cloud or live path; not validated against the live service or a licensed installation |
| `planned` | a stub |

## Testing

`frontier_connectors.testing.check_connector` runs the whole contract (detect, describe,
read, shape and dims invariants, conversion to a dataset, validation with `frontier-data`
when it is installed) and returns what was read, so your test asserts the physics:

```python
from frontier_connectors.testing import check_connector


def test_xyzp(tmp_path):
    f = tmp_path / "probes.xyzp"
    f.write_text("0 0 0 1.0\n1 0 0 2.0\n")
    report = check_connector(XyzpConnector(), f, tmp_path / "out")
    assert report.samples["probes"][0].fields["p"].data[1, 0] == 2.0
```

Guidelines the first-party plugins follow:

- Generate small fixtures in the test from the format specification; commit no binary files.
- Tests that download public samples are marked `@pytest.mark.network`, use
  `frontier_connectors.testing.fetch_sample` (cached outside the repository, sha256 pinned)
  and run only with `FRONTIER_NETWORK_TESTS=1`.
- Tests that need an optional package or an external program skip when it is absent.
- Write only below `tmp_path`.

## In this repository

First-party plugins have their own package directories, for example `packages/connectors/openfoam`.
`plugin.toml` is the single source
of truth there: `pyproject.toml`, the catalog used by `plugins search`, and the
`frontier-connectors-oss` meta-distribution are generated from it.

```bash
uv run python packages/frontier-connectors/tools/sync_plugins.py        # after editing a plugin.toml
uv sync                                                                  # picks up new entry points
uv run pytest packages/frontier-connectors
uv run python packages/frontier-connectors/tools/build_wheels.py --simple-index
uv run python packages/frontier-connectors/tools/clean_venv_proof.py    # install/uninstall in a fresh venv
```

`python -m frontier_connectors list --markdown` prints the support matrix of the README; a
test fails when the README is stale.
