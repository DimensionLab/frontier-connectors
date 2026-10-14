# frontier-connector-nek5000

Reads Nek5000 and NekRS binary field files.

## Install

```bash
python -m frontier_connectors plugins install nek5000
python -m frontier_connectors plugins info nek5000
```

## Status

`verified-public-sample`. Reads binary #std files: 2-D and 3-D, single and double precision, both byte orders, split dumps (nfiles > 1); geometry is reused when later dumps omit X. Legacy .fld files and pressure on the PN-2 mesh are not read. Spectral elements are exported as linear GLL sub-cells.

## Getting the data out

Nek5000 and NekRS write case0.f00001, case0.f00002, ... directly (plus case.nek5000 from the visnek script); point the connector at any of them or at the .nek5000 file.

## Requirements and licence

Licence: Apache-2.0. Installs `scipy>=1.13`.
