# SPDX-License-Identifier: Apache-2.0
"""Many solver cases as one dataset: one sample per case, with the case parameters.

A parameter study is usually a directory of cases (``sweep/aoa_2/``, ``sweep/aoa_4/`` ...), a
set of result files (``runs/*.cgns``) or a table that lists the cases. :func:`convert_cases`
reads every case with its connector, names the sample after the case, attaches the case
parameters and assigns train, validation and test splits by case, so the result can be
trained on.

Cases are found by ``cases``:

* a glob relative to the root (``"*/"``, ``"run_*/case.foam"``, ``"**/*.cgns"``);
* ``"auto"``: every direct child of the root that a connector recognises (a child directory
  that holds exactly one recognised result file counts as that file);
* a parameter table that has a ``path`` column: the paths it lists.

Parameters come from ``params``:

* a CSV or JSON table: one row per case, the case in the column ``case`` (or ``id``, ``name``,
  ``path``; else the first column), every numeric column a parameter. Looked for under the
  root as ``cases.csv``, ``parameters.csv``, ``sweep.csv``, ``cases.json`` or
  ``parameters.json`` when not named;
* ``"names"``: numbers in the case names (``aoa_2.5_re_1e6``, ``U=10,alpha=4``), used when
  every case yields the same parameter names;
* ``"none"``.

The default tries the table, then the names.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors.errors import ConnectorError
from frontier_connectors.model import RawSample

TABLE_NAMES = ("cases.csv", "parameters.csv", "sweep.csv", "cases.json", "parameters.json")
CASE_COLUMNS = ("case", "id", "name", "path")
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PAIR = re.compile(rf"([A-Za-z][A-Za-z]*(?:_[A-Za-z]+)*?)[=_\-]?({_NUMBER})(?![A-Za-z\d])")


@dataclass
class Case:
    name: str
    path: Path
    parameters: dict[str, float] = field(default_factory=dict)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value) if np.isfinite(value) else None
    try:
        out = float(str(value).strip())
    except ValueError:
        return None
    return out if np.isfinite(out) else None


def read_table(path: Path) -> dict[str, dict[str, Any]]:
    """``{case: {column: value}}`` from a CSV or JSON table (see the module docstring)."""
    if path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("cases"), list | dict):
            doc = doc["cases"]
        if isinstance(doc, dict):
            return {str(k): dict(v) for k, v in doc.items() if isinstance(v, dict)}
        rows = [r for r in doc if isinstance(r, dict)] if isinstance(doc, list) else []
    else:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            rows = [dict(r) for r in csv.DictReader(fh, dialect=dialect)]
    if not rows:
        raise ConnectorError(f"{path} lists no cases", hint="one row per case with a header row")
    keys = [k for k in rows[0] if k is not None]
    lower = {str(k).strip().lower(): k for k in keys}
    column = next((lower[c] for c in CASE_COLUMNS if c in lower), keys[0])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case = str(row.get(column, "")).strip()
        if case:
            out[case] = {str(k).strip(): v for k, v in row.items() if k is not None and k != column}
    return out


def parameters_from_name(name: str) -> dict[str, float]:
    """``aoa_2.5_re_1e6`` -> ``{"aoa": 2.5, "re": 1e6}``; a name without pairs gives nothing."""
    out: dict[str, float] = {}
    for key, value in _PAIR.findall(name):
        number = _number(value)
        if number is not None:
            out[key.strip("_")] = number
    return out


def _find_table(root: Path) -> Path | None:
    return next((root / n for n in TABLE_NAMES if (root / n).is_file()), None)


def _recognised(path: Path, connector: str | None) -> bool:
    from frontier_connectors import registry

    if connector:
        try:
            return registry.get(connector).detect(path) is not None
        except Exception:  # noqa: BLE001 - a connector that cannot look at the path does not match
            return False
    try:
        return bool(registry.detect(path))
    except ConnectorError:
        return False


def find_cases(
    root: Path,
    cases: str = "auto",
    *,
    params: str | Path | None = None,
    connector: str | None = None,
) -> tuple[list[Case], str]:
    """The cases under ``root`` and where their parameters came from
    (``table:<file>``, ``names`` or ``none``)."""
    if not root.is_dir():
        raise ConnectorError(
            f"{root} is not a directory", hint="--cases needs the directory that holds the cases"
        )
    table_path: Path | None = None
    mode = "auto" if params is None else str(params)
    if mode not in ("auto", "names", "none"):
        table_path = Path(mode) if Path(mode).is_absolute() else root / mode
        if not table_path.is_file():
            table_path = Path(mode)
        if not table_path.is_file():
            raise ConnectorError(f"no parameter table {mode}", hint="a CSV or JSON file")
    elif mode == "auto":
        table_path = _find_table(root)
    table = read_table(table_path) if table_path else {}

    if cases == "auto":
        listed = [root / c for c in table if (root / c).exists()]
        if table and len(listed) == len(table):
            paths = listed
        else:
            skip = {p.name for p in (table_path,) if p}
            paths = []
            for child in sorted(root.iterdir()):
                if child.name in skip or child.name.startswith("."):
                    continue
                if _recognised(child, connector):
                    paths.append(child)
                elif child.is_dir():
                    # a case directory that holds one result file
                    inner = [
                        f
                        for f in sorted(child.iterdir())
                        if not f.name.startswith(".") and _recognised(f, connector)
                    ]
                    if len(inner) == 1:
                        paths.append(inner[0])
    else:
        paths = []
        for match in sorted(p for p in root.glob(cases) if not p.name.startswith(".")):
            # `*/` names the case directories: one that no connector reads as a whole and
            # that holds one result file is that file
            if match.is_dir() and not _recognised(match, connector):
                inner = [
                    f
                    for f in sorted(match.iterdir())
                    if not f.name.startswith(".") and _recognised(f, connector)
                ]
                if len(inner) == 1:
                    match = inner[0]
            paths.append(match)
    if not paths:
        raise ConnectorError(
            f"no cases found under {root}" + ("" if cases == "auto" else f" for {cases!r}"),
            hint="pass --cases with a glob such as '*/' or 'run_*/case.foam', or a table with "
            "a `path` column",
        )

    def label(p: Path) -> str:
        rel = p.relative_to(root)
        parts = rel.parts if p.is_dir() else (*rel.parts[:-1], rel.stem)
        generic = {"case", "results", "result", "output", "solution"}
        if len(parts) > 1 and parts[-1].lower() in generic:
            parts = parts[:-1]
        return re.sub(r"[^A-Za-z0-9._=+-]+", "_", "__".join(parts)).strip("_") or "case"

    found = [Case(label(p), p) for p in paths]
    names = [c.name for c in found]
    if len(set(names)) != len(names):
        raise ConnectorError(
            "two cases have the same name", hint="narrow --cases so every match is one case"
        )
    origin = "none"
    if table:
        rel = {c.path.relative_to(root).as_posix().rstrip("/"): c for c in found}
        by_name = {c.name: c for c in found}
        matched = 0
        for key, row in table.items():
            case = by_name.get(key) or rel.get(key.rstrip("/")) or by_name.get(Path(key).stem)
            if case is None:
                continue
            matched += 1
            case.parameters = {k: n for k, v in row.items() if (n := _number(v)) is not None}
        if matched == 0:
            raise ConnectorError(
                f"{table_path} names none of the {len(found)} cases",
                hint=f"its first column should hold case names such as {names[0]!r}",
            )
        origin = f"table:{table_path.name}" if table_path else "table"
    elif mode in ("auto", "names"):
        parsed = [parameters_from_name(c.name) for c in found]
        keys = set(parsed[0])
        if keys and all(set(p) == keys for p in parsed):
            for case, values in zip(found, parsed, strict=True):
                case.parameters = values
            origin = "names"
    return found, origin


def assign_splits(names: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Seeded split by case: one test case from 2 cases on, one validation case from 4 on,
    then about 10 % each."""
    n = len(names)
    order = [names[i] for i in np.random.default_rng(seed).permutation(n)]
    n_test = 0 if n < 2 else max(1, round(0.1 * n))
    n_val = 0 if n < 4 else max(1, round(0.1 * n))
    n_train = n - n_test - n_val
    return {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train : n_train + n_val]),
        "test": sorted(order[n_train + n_val :]),
    }


def read_cases(
    found: list[Case],
    connector: str | None,
    ids: dict[str, str],
    **read_kwargs: Any,
) -> Iterator[RawSample]:
    """Every sample of every case, renamed after its case. ``ids`` is filled with
    ``sample id -> case name``. The first zone of a case is the sample ``<case>``; further
    zones are ``<case>__<zone id>``."""
    from frontier_connectors import registry

    for case in found:
        seen: dict[str, str] = {}
        for sample in registry.read(case.path, connector, **read_kwargs):
            new = seen.get(sample.id)
            if new is None:
                new = case.name if not seen else f"{case.name}__{sample.id}"
                seen[sample.id] = new
                ids[new] = case.name
            metadata = dict(sample.metadata)
            own = metadata.get("parameters")
            metadata["parameters"] = {
                **(dict(own) if isinstance(own, Mapping) else {}),
                **case.parameters,
            }
            metadata["group"] = case.name
            metadata.setdefault("source", str(case.path))
            sample.id = new
            sample.metadata = metadata
            yield sample
        if not seen:
            raise ConnectorError(
                f"case {case.name} ({case.path}) has no samples",
                hint="check the case, or exclude it with a narrower --cases glob",
            )


def write_splits(out: Path, ids: dict[str, str], seed: int) -> dict[str, list[str]]:
    """Assign the splits by case and record them in ``manifest.json``."""
    by_case = assign_splits(sorted(set(ids.values())), seed)
    where = {case: split for split, cases in by_case.items() for case in cases}
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid in sorted(ids):
        splits[where[ids[sid]]].append(sid)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["splits"] = splits
    manifest["split_strategy"] = {"name": "random-by-case", "seed": seed, "n": len(where)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(path)
    return splits


def convert_cases(
    root: str | Path,
    out: str | Path,
    *,
    cases: str = "auto",
    params: str | Path | None = None,
    connector: str | None = None,
    split_seed: int = 0,
    zones: list[str] | None = None,
    fields: list[str] | None = None,
    times: Any = None,
    options: Mapping[str, Any] | None = None,
    name: str | None = None,
    length_unit: str | None = None,
    source_length: str | None = None,
    **kwargs: Any,
) -> Any:
    """Convert the cases under ``root`` into one dataset (see the module docstring). Returns
    the ``ConvertResult`` with ``cases`` (name, path, parameters), ``parameters_from`` and
    ``splits`` set."""
    from frontier_connectors import registry
    from frontier_connectors.convert import to_frontier_dataset

    root = Path(root)
    found, origin = find_cases(root, cases, params=params, connector=connector)
    first = registry.resolve(found[0].path, connector)
    description = registry.describe(found[0].path, first.id, options=options)
    ids: dict[str, str] = {}
    stream = read_cases(
        found, connector, ids, zones=zones, fields=fields, times=times, options=options
    )
    if length_unit:
        from frontier_connectors.units import convert_sample_units

        def rescaled(samples: Iterator[RawSample]) -> Iterator[RawSample]:
            for s in samples:
                try:
                    yield convert_sample_units(s, length=length_unit, source_length=source_length)
                except KeyError as e:
                    raise ConnectorError(
                        f"cannot convert lengths to {length_unit!r}: {e.args[0]}",
                        hint="this format does not record its length unit: pass source_length "
                        "(CLI: --source-length mm)",
                    ) from e

        stream = rescaled(stream)
    source = {"path": str(root), "cases": len(found), "parameters_from": origin}
    kwargs.setdefault("description", description)
    result = to_frontier_dataset(stream, out, name=name, source=source, **kwargs)
    result.warnings = list(description.warnings) + result.warnings
    result.splits = write_splits(Path(out), ids, split_seed)
    result.parameters_from = origin
    result.cases = [
        {"name": c.name, "path": str(c.path), "parameters": c.parameters} for c in found
    ]
    if origin == "none":
        result.warnings.append(
            "no case parameters found: add a cases.csv (one row per case) or name the cases "
            "like aoa_2.5_re_1e6"
        )
    if len(found) < 3:
        result.warnings.append(
            f"{len(found)} case(s): too few to hold one out for validation and one for testing"
        )
    return result
