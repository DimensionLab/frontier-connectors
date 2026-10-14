# frontier-connector-tabular

Reads CSV, TSV and text tables, Parquet, and OpenFOAM probes and force coefficients. Parquet needs pyarrow.

## Install

```bash
python -m frontier_connectors plugins install tabular
python -m frontier_connectors plugins info tabular
```

## Status

`verified-public-sample`. Layouts: point snapshot, long (time, probe) tables, time + signal columns. Global signals are stored on one point at the origin. Parquet needs pyarrow. ParaView, Fluent, CFX and COMSOL style tables and Parquet are validated on generated files only. Tecplot ASCII and COMSOL '% Model:' files belong to their own connectors. Date-string time columns and ragged probe series (without per_probe) are not supported.

## Getting the data out

ParaView: File > Save Data > CSV (point data). Fluent: File > Export > ASCII, or report-file monitors (.out). CFX / COMSOL / STAR-CCM+: export monitor or probe tables as CSV/text. OpenFOAM: the probes and forceCoeffs function objects write postProcessing/<name>/<time>/ files that are read directly.

## Requirements and licence

Licence: Apache-2.0.

| Optional package | Licence    | Used to      | Note |
| ---------------- | ---------- | ------------ | ---- |
| `pyarrow`        | Apache-2.0 | read Parquet |      |
