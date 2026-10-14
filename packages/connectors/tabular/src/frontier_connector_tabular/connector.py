# SPDX-License-Identifier: Apache-2.0
"""Tabular point data: CSV / TSV / whitespace text and Parquet tables of point clouds, probe
time series and monitor signals; OpenFOAM ``postProcessing`` probes and force coefficients.

Layouts (chosen from the column names, overridable with a mapping):

``points``
    One row per point: coordinate columns (``x,y,z``, ``X [m]``, ParaView ``Points:0/1/2``,
    Fluent ``x-coordinate``, ``CoordinateX`` ...) plus field columns -> a steady ``point_cloud``.
``long``
    One row per (time, probe): a time column, coordinates and optionally a probe id column
    (``probe``, ``id``, ``name`` ... or mapping ``group_by``) -> a transient ``point_cloud`` with
    static probe positions (moving points are yielded step by step). Probes whose series differ
    in length raise an error unless ``options={"per_probe": True}`` (one sample per probe).
``series``
    A time column and signal columns without coordinates (solver monitors, force coefficients,
    COMSOL / CFX / Fluent probe exports) -> a transient ``point_cloud``. The dataset contract
    has no store for global signals, so they are attached to a single point at the origin (a
    warning says so) unless the mapping gives ``positions`` and per-probe column lists.

Columns are grouped into vectors by suffix (``U:0 U:1 U:2``, ``U_x U_y U_z``, ``velocity-x``,
``Ux Uy Uz``, ``x-velocity``); units come from headers such as ``p [Pa]`` or ``T (K)``.

Mapping config (``options={"mapping": {...}}`` / ``--mapping file.json``), every key optional:

.. code-block:: json

    {"layout": "points | long | series",
     "x": "Points:0", "y": "Points:1", "z": "Points:2",
     "time": "Time",
     "group_by": "probe_id",
     "fields": {"U": ["u", "v", "w"], "p": "pressure"},
     "positions": [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
     "drop": ["Point ID"],
     "delimiter": ";", "skip_rows": 2, "comment": "#", "header": ["t", "p0", "p1"],
     "units": {"p": "Pa", "length": "mm", "time": "ms"},
     "parameters": {"Re": 1.0e5}, "domain": "cfd", "solver": "..."}

With ``fields`` only the listed fields are read. In the ``series`` layout with ``positions``
each field lists one column per probe (``{"p": ["p0", "p1"]}``; vectors:
``{"U": [["u0", "v0"], ["u1", "v1"]]}``).

Text files are parsed in blocks of lines straight into numpy arrays; Parquet files are read
row group by row group with ``pyarrow`` (optional, Apache-2.0).
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import itertools
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil, units
from frontier_connectors.errors import (
    ConnectorError,
    MappingError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldData,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    ZoneInfo,
    match_extension,
)
from frontier_connectors.select import match_names, select_time_indices, wants

TEXT_EXTENSIONS = [".csv", ".tsv", ".txt", ".dat", ".out", ".xy", ".prn"]
PARQUET_EXTENSIONS = [".parquet", ".pq"]
CHUNK_LINES = 200_000
COMMENT_CHARS = ("#", "%", "//", "!")
MAPPING_KEYS = (
    "layout", "x", "y", "z", "time", "group_by", "fields", "positions", "drop", "delimiter",
    "skip_rows", "comment", "header", "units", "parameters", "domain", "solver", "metadata",
)  # fmt: skip

_AXIS_TEMPLATES = (
    "{a}", "points:{i}", "points_{i}", "point:{i}", "{a}-coordinate", "{a}_coordinate",
    "{a}coordinate", "coordinate{a}", "coordinate {a}", "coordinate_{a}", "coordinates:{i}",
    "coordinates_{i}", "coords:{i}", "coords_{i}", "{a} coordinate", "{a}coord", "{a}_coord",
    "pos_{a}", "pos{a}", "position_{a}", "position:{i}", "position {a}", "{a}_pos", "{a}pos",
    "centroid_{a}", "node_{a}", "{a}c", "points:{a}", "points_{a}", "{a}-coord", "{a} coord",
    "location_{a}", "center_{a}", "centre_{a}",
)  # fmt: skip
AXIS_NAMES: dict[str, tuple[str, ...]] = {
    a: tuple(t.format(a=a, i=i) for t in _AXIS_TEMPLATES) for i, a in enumerate("xyz")
}
TIME_NAMES = (
    "time", "t", "flow-time", "flow time", "flowtime", "flow_time", "time_s", "physical time",
    "physical_time", "simulation time", "simulation_time", "times", "timevalue", "time value",
)  # fmt: skip
ITERATION_NAMES = (
    "iteration", "iter", "iterations", "step", "timestep", "time step", "time-step", "time_step",
    "accumulated timestep", "accumulated time step", "accumulated iteration", "it", "nstep",
)  # fmt: skip
GROUP_NAMES = (
    "probe", "probe_id", "probeid", "probe id", "id", "point_id", "pointid", "point id", "name",
    "station", "sensor", "label", "probe_name", "point", "node", "nodenumber", "particle_id",
)  # fmt: skip
ID_NAMES = (
    "nodenumber", "cellnumber", "node number", "cell number", "point id", "pointid", "point_id",
    "cell id", "cellid", "cell_id", "index", "vtkoriginalpointids", "vtkoriginalcellids", "",
    "unnamed: 0", "node id", "node_id", "nodeid", "id",
)  # fmt: skip
_NAN_TOKENS = {"", "nan", "na", "n/a", "null", "none", "-", "--", "inf", "-inf", "+inf"}
_FOAM_PROBE = re.compile(r"^#\s*Probe\s+(\d+)\s*\(\s*([^)]*)\)")
_FOAM_AXIS = re.compile(r"^#\s*([xyz])\s+(.+)$")
_FOAM_COEFF_FILES = ("coefficient.dat", "forcecoeffs.dat", "force.dat", "forces.dat", "moment.dat")
_UNIT_TAIL = re.compile(r"^(.*?)\s*(?:\[([^\]]*)\]|\(([^()]*)\))\s*$")
_PARENS = str.maketrans("()", "  ")
_DECIMAL_COMMA = re.compile(r"[-+]?\d+,\d+(?:[eE][-+]?\d+)?")


# -- header helpers --------------------------------------------------------------------------


def split_unit(header: str) -> tuple[str, str]:
    """``"p [Pa]"`` -> ``("p", "Pa")``; ``"T (K)"`` -> ``("T", "K")``; no unit -> ``(name, "")``.
    Parenthesised tails count as units only when they look like one (``"f (x)"`` stays)."""
    h = header.strip().strip('"').strip()
    m = _UNIT_TAIL.match(h)
    if not m or not m.group(1):
        return h, ""
    name, bracket, paren = m.group(1).strip(), m.group(2), m.group(3)
    if bracket is not None:
        return name, bracket.strip()
    unit = (paren or "").strip()
    known = True
    try:
        units.si_factor(unit)
    except (KeyError, ValueError, ZeroDivisionError):
        known = False
    if (
        unit
        and (known or re.fullmatch(r"[A-Za-zµ°%/*^.\-0-9 ]{1,12}", unit))
        and not unit.isdigit()
    ):
        if known or not re.fullmatch(r"[xyzXYZ0-9]", unit):
            return name, unit
    return h, ""


def _is_number(tok: str) -> bool:
    t = tok.strip().strip('"')
    if t.lower() in ("nan", "inf", "-inf", "+inf"):
        return True
    try:
        float(t)
    except ValueError:
        return False
    return True


def _tokens(line: str, delimiter: str | None) -> list[str]:
    line = line.strip("\r\n")
    if delimiter is None:
        return [a or b for a, b in re.findall(r'"([^"]*)"|(\S+)', line)]
    return [t.strip() for t in next(csv.reader([line], delimiter=delimiter))]


def _guess_delimiter(lines: Sequence[str]) -> str | None:
    """Delimiter with the same count on (nearly) every line, header included so that a decimal
    comma is not mistaken for one; up to two leading title lines are tolerated."""
    for skip in (0, 1, 2):
        rows = lines[skip:]
        if len(rows) < (1 if skip == 0 else 2):
            break
        for d in (";", "\t", "|", ","):
            counts = [ln.count(d) for ln in rows]
            mode = max(set(counts), key=counts.count)
            # a few ragged rows must not hide the delimiter (they are reported when read)
            if mode > 0 and counts.count(mode) >= 0.6 * len(counts):
                return d
    return None


def _component_groups(names: Sequence[str]) -> dict[str, list[str]]:
    """Group scalar columns into vector fields by their component suffix/prefix.
    Returns ``field name -> [column, ...]`` in column order of first appearance."""
    patterns = (
        (re.compile(r"^(.+?)[:_\- ]([0-8])$"), "index"),
        (re.compile(r"^(.+?)[:_\- ]([xyzXYZ])$"), "letter"),
        (re.compile(r"^([xyzXYZ])[_\- ](.+)$"), "prefix"),
        (re.compile(r"^(.*[A-Za-z])([xyz])$"), "glued"),
    )
    parsed: dict[str, tuple[str, int, str]] = {}
    for n in names:
        for rx, kind in patterns:
            m = rx.match(n)
            if not m:
                continue
            if kind == "prefix":
                base, comp = m.group(2), "xyz".index(m.group(1).lower())
            elif kind == "index":
                base, comp = m.group(1), int(m.group(2))
            else:
                base, comp = m.group(1), "xyz".index(m.group(2).lower())
            parsed[n] = (base.strip(), comp, kind)
            break
    by_base: dict[tuple[str, str], dict[int, str]] = {}
    for n, (base, comp, kind) in parsed.items():
        by_base.setdefault((base, kind), {}).setdefault(comp, n)
    grouped: dict[str, list[str]] = {}
    member: dict[str, str] = {}
    for (base, kind), comps in by_base.items():
        idx = sorted(comps)
        if len(idx) < 2 or idx != list(range(len(idx))) or base in names:
            continue
        if (
            kind == "glued"
            and len(base) > 1
            and not any(c.isupper() for c in base)
            and len(idx) < 3
        ):
            continue  # "max"/"may" style look-alikes: only take lowercase glued pairs when x,y,z
        cols = [comps[i] for i in idx]
        if any(c in member for c in cols) or base in grouped:
            continue
        grouped[base] = cols
        for c in cols:
            member[c] = base
    out: dict[str, list[str]] = {}
    for n in names:
        if n in member:
            out.setdefault(member[n], grouped[member[n]])
        else:
            out[n] = [n]
    return out


# -- sources ---------------------------------------------------------------------------------


@dataclass
class _Column:
    name: str  # header text without the unit
    raw: str
    unit: str = ""
    numeric: bool = True
    #: vector columns exist in Parquet only (list-typed); text files are always scalar
    width: int = 1


@dataclass
class _Source:
    """A table whose columns can be loaded selectively."""

    path: Path
    format: str
    columns: list[_Column]
    notes: dict[str, Any] = field(default_factory=dict)
    delimiter: str | None = None
    data_start: int = 0
    comments: tuple[str, ...] = COMMENT_CHARS
    parens: bool = False
    decimal_comma: bool = False

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def index(self, name: str) -> int:
        return self.names().index(name)

    def load(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        want = list(dict.fromkeys(names))
        if self.format == "parquet":
            return _load_parquet(self, want)
        return _load_text(self, want)


def _unique_names(raw: Sequence[str]) -> list[_Column]:
    cols: list[_Column] = []
    seen: dict[str, int] = {}
    for r in raw:
        name, unit = split_unit(r)
        name = name or f"col{len(cols)}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(_Column(name=name, raw=r, unit=unit))
    return cols


def _comment_header(lines: Sequence[str], n_cols: int, delimiter: str | None) -> list[str] | None:
    """Column names from the comment line just above the data (``# Time Cd Cl``,
    ``% Time (s)    Probe 1 (K)``) when it has as many entries as the data has columns."""
    for line in reversed(lines):
        body = line.lstrip()
        for c in COMMENT_CHARS:
            if body.startswith(c):
                body = body[len(c) :]
                break
        body = body.strip()
        if not body:
            continue
        candidates = [_tokens(body, delimiter)]
        if delimiter is None:
            candidates.append([t.strip() for t in re.split(r"\s{2,}|\t", body) if t.strip()])
        else:
            candidates.append(_tokens(body, None))
        for toks in candidates:
            if len(toks) == n_cols and not all(_is_number(t) for t in toks):
                return toks
        return None
    return None


def _paren_header(lines: Sequence[str], n_cols: int) -> list[str] | None:
    """Expand an OpenFOAM header such as ``# Time forces(pressure viscous porous)
    moments(pressure viscous porous)`` into one name per value column
    (``forces_pressure_x`` ...), when the counts work out with 3-vectors."""
    for line in reversed(lines):
        body = line.lstrip().lstrip("#%!/").strip()
        if "(" not in body:
            continue
        parts = re.findall(r"(\w+)\(([^)]*)\)|(\S+)", body)
        plain = sum(1 for g, _, p in parts if p and not g)
        subs = sum(len(inner.split()) for g, inner, _ in parts if g)
        if not subs or (n_cols - plain) % subs:
            return None
        width = (n_cols - plain) // subs
        if width not in (1, 3):
            return None
        names: list[str] = []
        for g, inner, p in parts:
            if not g:
                names.append(p)
                continue
            for sub in inner.split():
                names.extend([f"{g}_{sub}"] if width == 1 else [f"{g}_{sub}_{c}" for c in "xyz"])
        return names if len(names) == n_cols else None
    return None


def sniff_text(path: Path, mapping: Mapping[str, Any] | None = None) -> _Source | None:
    """Find the delimiter, header and first data line of a text table (reads the head only).
    ``None`` when the file does not look like a numeric table."""
    m = mapping or {}
    try:
        with open(path, "rb") as f:
            head = f.read(1 << 18)
    except OSError:
        return None
    if b"\x00" in head[:4096]:
        return None
    lines = head.decode("utf-8-sig", "replace").splitlines()
    if len(head) == 1 << 18 and lines:
        lines.pop()  # possibly cut in the middle
    skip = int(m.get("skip_rows") or 0)
    comment_chars = (str(m["comment"]),) if m.get("comment") else COMMENT_CHARS
    comments: list[str] = []
    notes: dict[str, Any] = {}
    body: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if i < skip or not line.strip():
            continue
        if line.lstrip().startswith(comment_chars):
            if not body:
                comments.append(line)
            continue
        body.append((i, line))
        if len(body) >= 40:
            break
    if not body:
        return None
    probe = [ln for _, ln in body[:12]]
    delimiter: str | None = m.get("delimiter") or _guess_delimiter(probe)
    if delimiter in ("whitespace", " ", ""):
        delimiter = None
    grouped_rows = [ln for ln in probe if "(" in ln and re.match(r"\s*[-+.\d(]", ln)]
    if grouped_rows and not m.get("delimiter") and delimiter != ",":
        delimiter = None  # OpenFOAM writes vectors as "(a b c)": parentheses become blanks
    header: list[str] | None = None
    start = 0
    parens = False
    # leading non-numeric lines: optional title lines, then the header (Fluent monitor files
    # put the header in parentheses)
    for k, (_, line) in enumerate(body[:6]):
        clean = line.translate(_PARENS) if delimiter is None else line
        toks = _tokens(clean, delimiter)
        numeric = sum(_is_number(t) or bool(_DECIMAL_COMMA.fullmatch(t)) for t in toks)
        blank = sum(t.strip().lower() in _NAN_TOKENS for t in toks)
        # a data row may carry text columns (probe names), a header row has no numbers
        if toks and numeric and (numeric + blank == len(toks) or 2 * numeric >= len(toks)):
            if header is not None or numeric + blank == len(toks):
                start = k
                break
        header = toks
    else:
        return None
    first = body[start][1]
    if delimiter is None and "(" in first:
        parens = True
    data_tokens = _tokens(first.translate(_PARENS) if parens else first, delimiter)
    n_cols = len(data_tokens)
    if n_cols < 1:
        return None
    decimal_comma = delimiter != "," and any(_DECIMAL_COMMA.fullmatch(t) for t in data_tokens)
    if isinstance(m.get("header"), list):
        header = [str(h) for h in m["header"]]
    if header is None or len(header) != n_cols:
        from_comment = _comment_header(comments, n_cols, delimiter) or _paren_header(
            comments, n_cols
        )
        if from_comment is not None:
            header = from_comment
        elif header is not None and len(header) != n_cols:
            # mixed text/number rows (a string id column) keep the header; otherwise give up
            return None
    if header is None:
        header = [f"col{i}" for i in range(n_cols)]
        notes["generated_header"] = True
    cols = _unique_names(header)
    for c, tok in zip(cols, data_tokens, strict=True):
        c.numeric = _is_number(tok) or tok.strip().lower() in _NAN_TOKENS or decimal_comma
    for line in comments:
        mm = re.match(r"^\s*[#%!/]+\s*([A-Za-z_][\w ]*?)\s*[:=]\s*(.+?)\s*$", line)
        if mm:
            notes.setdefault("header_values", {})[mm.group(1).strip()] = mm.group(2)
    notes["comment_lines"] = comments[:50]
    return _Source(
        path=path,
        format="text",
        columns=cols,
        notes=notes,
        delimiter=delimiter,
        data_start=body[start][0],
        comments=comment_chars,
        parens=parens,
        decimal_comma=decimal_comma,
    )


def _slow_rows(
    src: _Source, block: Sequence[str], first_line: int, cols: Sequence[int]
) -> np.ndarray:
    """Per-line parse used when the fast path fails: tolerates missing values and reports the
    exact line of a ragged or non-numeric row."""
    n_cols = len(src.columns)
    out = np.empty((len(block), len(cols)), dtype=np.float64)
    for r, line in enumerate(block):
        toks = _tokens(line, src.delimiter)
        if len(toks) != n_cols:
            raise ConnectorError(
                f"{src.path.name}: line {first_line + r + 1} has {len(toks)} columns, the "
                f"header has {n_cols}",
                hint="the table is ragged (truncated write, or unquoted delimiters inside a "
                "field); fix the row or set mapping 'delimiter'/'skip_rows'",
            )
        for k, c in enumerate(cols):
            t = toks[c].strip().strip('"')
            if t.lower() in _NAN_TOKENS:
                out[r, k] = np.nan if t.lower() not in ("inf", "+inf", "-inf") else float(t)
                continue
            try:
                out[r, k] = float(t)
            except ValueError:
                raise ConnectorError(
                    f"{src.path.name}: line {first_line + r + 1}, column "
                    f"{src.columns[c].raw!r}: {t!r} is not a number",
                    hint="list the column under mapping 'drop' or use it as 'group_by'",
                ) from None
    return out


def _load_text(src: _Source, names: Sequence[str]) -> dict[str, np.ndarray]:
    idx = [src.index(n) for n in names]
    num = [i for i in idx if src.columns[i].numeric]
    txt = [i for i in idx if not src.columns[i].numeric]
    blocks: list[np.ndarray] = []
    text: dict[int, list[str]] = {i: [] for i in txt}
    line_no = src.data_start
    try:
        fh = open(src.path, encoding="utf-8-sig", errors="replace", newline="")  # noqa: SIM115
    except OSError as e:
        raise ConnectorError(f"{src.path} cannot be read ({e})") from e
    with fh:
        for _ in itertools.islice(fh, src.data_start):
            pass
        while True:
            raw = list(itertools.islice(fh, CHUNK_LINES))
            if not raw:
                break
            block = [ln for ln in raw if ln.strip() and not ln.lstrip().startswith(src.comments)]
            if src.parens:
                block = [ln.translate(_PARENS) for ln in block]
            if src.decimal_comma:
                block = [ln.replace(",", ".") for ln in block]
            if block and num:
                try:
                    arr = np.loadtxt(
                        block,
                        delimiter=src.delimiter,
                        comments=None,
                        usecols=num,
                        dtype=np.float64,
                        ndmin=2,
                        quotechar='"',
                    )
                except (ValueError, IndexError):
                    arr = _slow_rows(src, block, line_no, num)
                blocks.append(arr)
            for i in txt:
                text[i].extend(_tokens(ln, src.delimiter)[i].strip('"') for ln in block)
            line_no += len(raw)
    data = np.concatenate(blocks) if blocks else np.zeros((0, len(num)))
    out: dict[str, np.ndarray] = {}
    for k, i in enumerate(num):
        out[src.columns[i].name] = data[:, k]
    for i in txt:
        out[src.columns[i].name] = np.asarray(text[i], dtype=object)
    return out


def _pyarrow() -> Any:
    try:
        return importlib.import_module("pyarrow"), importlib.import_module("pyarrow.parquet")
    except ImportError as e:
        raise MissingDependencyError(
            "pyarrow",
            extra="frontier-connector-tabular[parquet]",
            purpose="read Parquet tables",
            license="Apache-2.0",
        ) from e


def open_parquet(path: Path) -> _Source:
    pa, pq = _pyarrow()
    try:
        pf = pq.ParquetFile(str(path))
    except Exception as e:  # noqa: BLE001 - pyarrow.lib.ArrowInvalid and friends
        raise ConnectorError(
            f"{path} is not a readable Parquet file ({e})",
            hint="the file is truncated (the footer is written last) or not Parquet",
        ) from e
    cols: list[_Column] = []
    schema = pf.schema_arrow
    for f in schema:
        name, unit = split_unit(f.name)
        c = _Column(name=name or f.name, raw=f.name, unit=unit)
        t = f.type
        if pa.types.is_fixed_size_list(t):
            c.width = int(t.list_size)
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            c.width = -1  # resolved from the first row when loaded
        elif not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)):
            c.numeric = False
        meta = f.metadata or {}
        if not c.unit and b"units" in meta:
            c.unit = meta[b"units"].decode("utf-8", "replace")
        cols.append(c)
    notes = {"n_rows": int(pf.metadata.num_rows), "row_groups": int(pf.metadata.num_row_groups)}
    return _Source(path=path, format="parquet", columns=cols, notes=notes)


def _load_parquet(src: _Source, names: Sequence[str]) -> dict[str, np.ndarray]:
    _pa, pq = _pyarrow()
    raw = {src.columns[src.index(n)].raw: n for n in names}
    chunks: dict[str, list[np.ndarray]] = {n: [] for n in names}
    try:
        pf = pq.ParquetFile(str(src.path))
        for batch in pf.iter_batches(batch_size=1 << 18, columns=list(raw)):
            for r, n in raw.items():
                col = batch.column(batch.schema.get_field_index(r))
                c = src.columns[src.index(n)]
                if c.width != 1:
                    flat = col.flatten().to_numpy(zero_copy_only=False)
                    width = c.width if c.width > 0 else (len(flat) // max(len(col), 1))
                    if width * len(col) != len(flat):
                        raise ConnectorError(
                            f"{src.path.name}: list column {r!r} has rows of different length",
                            hint="only fixed-length vector columns can become fields",
                        )
                    chunks[n].append(np.asarray(flat, dtype=np.float64).reshape(len(col), width))
                elif c.numeric:
                    arr = col.to_numpy(zero_copy_only=False)
                    chunks[n].append(np.asarray(arr, dtype=np.float64))
                else:
                    chunks[n].append(np.asarray(col.to_pylist(), dtype=object))
    except ConnectorError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConnectorError(
            f"{src.path}: Parquet data could not be read ({e})",
            hint="the file is corrupt or uses a codec this pyarrow build lacks",
        ) from e
    return {n: (np.concatenate(v) if v else np.zeros((0,))) for n, v in chunks.items()}


# -- OpenFOAM probes -------------------------------------------------------------------------


def foam_probe_header(path: Path) -> np.ndarray | None:
    """Probe positions ``(P, 3)`` from an OpenFOAM ``probes`` function-object file header."""
    pos: dict[int, list[float]] = {}
    axes: dict[str, list[float]] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in itertools.islice(f, 20000):
                if not line.startswith("#"):
                    break
                m = _FOAM_PROBE.match(line)
                if m:
                    try:
                        pos[int(m.group(1))] = [float(v) for v in m.group(2).split()]
                    except ValueError:
                        return None
                    continue
                m = _FOAM_AXIS.match(line)
                if m:
                    try:
                        axes[m.group(1)] = [float(v) for v in m.group(2).split()]
                    except ValueError:
                        continue
    except OSError:
        return None
    if pos and all(len(p) == 3 for p in pos.values()):
        return np.asarray([pos[k] for k in sorted(pos)], dtype=np.float64)
    if set(axes) == {"x", "y", "z"} and len({len(v) for v in axes.values()}) == 1:
        return np.stack([axes["x"], axes["y"], axes["z"]], axis=1)
    return None


def _foam_probe_files(path: Path) -> dict[str, list[Path]]:
    """``field -> [file per start-time directory, in time order]`` for a probes file, a time
    directory or the function-object directory."""
    if path.is_file():
        return {path.name: [path]}
    direct = [f for f in sorted(path.iterdir()) if f.is_file() and not f.name.startswith(".")]
    if any(foam_probe_header(f) is not None for f in direct):
        return {f.name: [f] for f in direct if foam_probe_header(f) is not None}
    out: dict[str, list[Path]] = {}
    subs = []
    for d in path.iterdir():
        if d.is_dir():
            try:
                subs.append((float(d.name), d))
            except ValueError:
                continue
    for _, d in sorted(subs, key=lambda t: t[0]):
        for f in sorted(d.iterdir()):
            if f.is_file() and foam_probe_header(f) is not None:
                out.setdefault(f.name, []).append(f)
    return out


def _read_foam_probe_file(path: Path, n_probes: int) -> tuple[np.ndarray, np.ndarray]:
    src = _Source(path=path, format="text", columns=[], parens=True)
    with open(path, encoding="utf-8", errors="replace") as f:
        first = next((ln for ln in f if ln.strip() and not ln.startswith("#")), "")
    n_cols = len(first.translate(_PARENS).split())
    if not first:
        return np.zeros(0), np.zeros((0, n_probes, 1))
    if n_cols < 2 or (n_cols - 1) % n_probes:
        raise ConnectorError(
            f"{path}: {n_cols - 1} values per row do not divide over {n_probes} probes",
            hint="the probes file is truncated or its header lists a different probe set",
        )
    src.columns = [_Column(name=f"c{i}", raw=f"c{i}") for i in range(n_cols)]
    table = _load_text(src, src.names())
    data = np.stack([table[f"c{i}"] for i in range(n_cols)], axis=1)
    return data[:, 0], data[:, 1:].reshape(len(data), n_probes, -1)


# -- interpretation --------------------------------------------------------------------------


@dataclass
class _Layout:
    layout: str  # points | long | series
    coords: list[str]
    time: str | None
    group_by: str | None
    #: field name -> columns (one per component); for series-with-positions: per probe
    fields: dict[str, list[Any]]
    units: dict[str, str]
    length_unit: str = ""
    time_unit: str = "s"
    time_quantity: str = "time"
    positions: np.ndarray | None = None
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _find(names: Sequence[str], candidates: Sequence[str], exact_case: str = "") -> str | None:
    lower = {n.lower().strip(): n for n in reversed(names)}
    for cand in candidates:
        if len(cand) == 1 and cand == exact_case:
            if cand in names:
                return cand
            continue
        if cand in lower:
            return lower[cand]
    return None


def interpret(src: _Source, mapping: Mapping[str, Any] | None) -> _Layout:
    m = dict(mapping or {})
    unknown = [k for k in m if k not in MAPPING_KEYS]
    if unknown:
        raise MappingError(
            f"unknown mapping key(s): {', '.join(map(str, unknown))}",
            hint="valid keys: " + ", ".join(MAPPING_KEYS),
        )
    names = src.names()
    by_raw = {c.raw.strip().strip('"'): c.name for c in src.columns}

    def col(value: Any, what: str) -> str:
        v = str(value)
        if v in names:
            return v
        if v in by_raw:
            return by_raw[v]
        low = {n.lower(): n for n in names} | {r.lower(): n for r, n in by_raw.items()}
        if v.lower() in low:
            return low[v.lower()]
        raise MappingError(
            f"mapping {what} names column {v!r}, which is not in {src.path.name}",
            hint="columns: " + ", ".join(repr(c.raw) for c in src.columns[:60]),
        )

    coords: list[str] = []
    for ax in "xyz":
        if m.get(ax) is not None:
            coords.append(col(m[ax], ax))
        else:
            found = _find(names, AXIS_NAMES[ax])
            if found is not None and (ax == "x" or coords):
                coords.append(found)
            elif ax == "x":
                break
            elif found is None and ax == "y":
                break
    if m.get("time") is not None:
        time = col(m["time"], "time")
        quantity = "iteration" if time.lower() in ITERATION_NAMES else "time"
    else:
        time = _find([n for n in names if n not in coords], TIME_NAMES, exact_case="t")
        quantity = "time"
        if time is None and not coords:
            time = _find(names, ITERATION_NAMES)
            quantity = "iteration"
    if time is not None and not src.columns[src.index(time)].numeric:
        raise MappingError(
            f"time column {time!r} is not numeric (date strings are not parsed)",
            hint="convert the column to seconds, or map another column as 'time'",
        )
    layout = m.get("layout")
    if layout is None:
        if coords and time is None:
            layout = "points"
        elif coords:
            layout = "long"
        elif time is not None:
            layout = "series"
        else:
            raise MappingError(
                f"{src.path.name}: no coordinate columns (x, y, z) and no time column found",
                hint="columns: "
                + ", ".join(repr(c.raw) for c in src.columns[:60])
                + '; map them, e.g. --mapping with {"x": "<col>", "y": "<col>", "z": "<col>"} '
                'or {"time": "<col>"}',
            )
    if layout not in ("points", "long", "series"):
        raise MappingError(f"unknown layout {layout!r}", hint="one of points | long | series")
    if layout in ("points", "long") and not coords:
        raise MappingError(
            f"layout {layout!r} needs coordinate columns",
            hint="columns: " + ", ".join(repr(c.raw) for c in src.columns[:60]),
        )
    if layout in ("long", "series") and time is None:
        raise MappingError(
            f"layout {layout!r} needs a time column",
            hint="columns: " + ", ".join(repr(c.raw) for c in src.columns[:60]),
        )
    if layout == "points":
        time = None
    group_by = None
    if layout == "long":
        if m.get("group_by") is not None:
            group_by = col(m["group_by"], "group_by")
        else:
            group_by = _find([n for n in names if n not in coords and n != time], GROUP_NAMES)
    reserved = {*coords, *([time] if time else []), *([group_by] if group_by else [])}
    dropped = [col(d, "drop") for d in m.get("drop") or []]
    lay = _Layout(layout, coords, time, group_by, {}, {}, time_quantity=quantity)
    user_units = {str(k): str(v) for k, v in (m.get("units") or {}).items()}
    unit_of = {c.name: c.unit for c in src.columns}
    lay.length_unit = user_units.get("length") or next(
        (units.normalize_unit(unit_of[c]) for c in coords if unit_of[c]), ""
    )
    if time is not None:
        lay.time_unit = user_units.get("time") or units.normalize_unit(unit_of[time]) or "s"
    positions = m.get("positions")
    if positions is not None:
        lay.positions = np.asarray(positions, dtype=np.float64).reshape(len(positions), -1)
    if m.get("fields"):
        for fname, spec in dict(m["fields"]).items():
            items = [spec] if isinstance(spec, str) else list(spec)
            if lay.positions is not None and layout == "series":
                per_probe = [
                    [col(c, f"field {fname!r}") for c in ([p] if isinstance(p, str) else p)]
                    for p in items
                ]
                if len(per_probe) != len(lay.positions):
                    raise MappingError(
                        f"field {fname!r} lists {len(per_probe)} probes, 'positions' has "
                        f"{len(lay.positions)}"
                    )
                lay.fields[str(fname)] = per_probe
            else:
                lay.fields[str(fname)] = [col(c, f"field {fname!r}") for c in items]
    else:
        candidates = []
        for c in src.columns:
            if c.name in reserved or c.name in dropped:
                continue
            low = c.name.lower().strip()
            if not c.numeric:
                lay.dropped.append(c.name)
            elif low in ID_NAMES or (low in ITERATION_NAMES and time is not None):
                lay.dropped.append(c.name)
            elif re.search(r"[:_ ]magnitude$", low):
                lay.dropped.append(c.name)  # ParaView's derived |v| column
            else:
                candidates.append(c.name)
        wide = [c.name for c in src.columns if c.width != 1 and c.name in candidates]
        scalars = [n for n in candidates if n not in wide]
        lay.fields = {n: [n] for n in wide} | _component_groups(scalars)
        if lay.positions is not None and layout == "series":
            raise MappingError(
                "'positions' needs 'fields' listing one column per probe",
                hint='e.g. "fields": {"p": ["p_probe0", "p_probe1"]}',
            )
    for fname, cols in lay.fields.items():
        flat = [c for item in cols for c in ([item] if isinstance(item, str) else item)]
        bad = [c for c in flat if not src.columns[src.index(c)].numeric]
        if bad:
            raise MappingError(f"field {fname!r}: column {bad[0]!r} is not numeric")
        lay.units[fname] = user_units.get(fname) or next(
            (unit_of[c] for c in flat if unit_of[c]), ""
        )
    if lay.dropped:
        lay.warnings.append("columns not read as fields: " + ", ".join(lay.dropped[:20]))
    if layout == "series" and lay.positions is None:
        lay.warnings.append(
            "no coordinates in the table: the signals are stored on a single point at the "
            "origin (the dataset format has no global time-series store)"
        )
    return lay


def _field_columns(lay: _Layout, names: Sequence[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for item in lay.fields[n]:
            out.extend([item] if isinstance(item, str) else item)
    return out


def _stack(table: Mapping[str, np.ndarray], cols: Sequence[str]) -> np.ndarray:
    parts = [table[c] if table[c].ndim == 2 else table[c][:, None] for c in cols]
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _first_appearance(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unique values in order of first appearance, inverse)`` for 1-D or row keys."""
    if keys.ndim == 2:
        uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    else:
        uniq, first, inv = np.unique(
            keys.astype(str) if keys.dtype == object else keys,
            return_index=True,
            return_inverse=True,
        )
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return uniq[order], rank[np.asarray(inv).reshape(-1)]


@dataclass
class _Probes:
    """Pivoted long table: ``values[field]`` is ``(T, P, C)``."""

    times: np.ndarray
    positions: np.ndarray  # (P, D) or (T, P, D) when the points move
    labels: list[str]
    values: dict[str, np.ndarray]
    ragged: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] | None = None


def _pivot(
    src: _Source, lay: _Layout, table: Mapping[str, np.ndarray], names: Sequence[str]
) -> _Probes:
    assert lay.time is not None
    t = np.asarray(table[lay.time], dtype=np.float64)
    if np.isnan(t).any():
        raise ConnectorError(
            f"{src.path.name}: the time column {lay.time!r} has missing values",
            hint="remove incomplete rows or map another time column",
        )
    pos = _stack(table, lay.coords)
    times, t_inv = np.unique(t, return_inverse=True)
    n_t = len(times)
    if lay.group_by is not None:
        keys, k_inv = _first_appearance(table[lay.group_by])
        labels = [str(k) for k in keys]
    else:
        keys, k_inv = _first_appearance(pos)
        labels = [str(i) for i in range(len(keys))]
        if len(keys) * n_t != len(t):
            # points that move without an id column: match them by order within each time
            order = np.argsort(t_inv, kind="stable")
            starts = np.searchsorted(t_inv[order], np.arange(n_t))
            k_inv = np.empty(len(t), dtype=np.int64)
            k_inv[order] = np.arange(len(t)) - starts[t_inv[order]]
            labels = [str(i) for i in range(int(k_inv.max()) + 1 if len(t) else 0)]
    n_p = len(labels)
    flat = t_inv * n_p + k_inv
    dense = len(t) == n_t * n_p and len(np.unique(flat)) == len(t)
    values = {n: _stack(table, _field_columns(lay, [n])) for n in names}
    if not dense:
        ragged = {}
        for k, label in enumerate(labels):
            rows = np.nonzero(k_inv == k)[0]
            rows = rows[np.argsort(t[rows], kind="stable")]
            ragged[label] = (t[rows], pos[rows[0]], {n: v[rows] for n, v in values.items()})
        return _Probes(times, pos[:0], labels, {}, ragged)
    out_pos = np.empty((n_t, n_p, pos.shape[1]), dtype=np.float64)
    out_pos[t_inv, k_inv] = pos
    out_vals = {}
    for n, v in values.items():
        a = np.empty((n_t, n_p, v.shape[1]), dtype=np.float32)
        a[t_inv, k_inv] = v
        out_vals[n] = a
    scale = max(float(np.abs(out_pos).max(initial=0.0)), 1e-30)
    static = bool(np.all(np.abs(out_pos - out_pos[:1]) <= 1e-9 * scale))
    return _Probes(times, out_pos[0] if static else out_pos, labels, out_vals)


# -- connector -------------------------------------------------------------------------------


def _foreign_text(head: bytes) -> str | None:
    """Tecplot ASCII and COMSOL exports have their own connectors."""
    text = head[:4096].decode("utf-8", "replace")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("% Model:") or s.startswith("% Version:"):
            return "comsol"
        if re.match(r"^(TITLE|VARIABLES|ZONE|FILETYPE)\b\s*=?", s, re.IGNORECASE) and (
            "=" in s or s.upper().startswith("ZONE")
        ):
            return "tecplot"
        if s.startswith("#!TDV"):
            return "tecplot"
        if not s.startswith(COMMENT_CHARS):
            return None
    return None


class TabularConnector(BaseConnector):
    id = "tabular"
    name = "Tabular point data (CSV / TSV / text, Parquet, OpenFOAM probes and force coeffs)"
    formats = ["csv", "tsv", "text-table", "parquet", "openfoam-probes"]
    vendor = "open formats"
    license = "Apache-2.0 (optional: pyarrow, Apache-2.0)"
    extensions = TEXT_EXTENSIONS + PARQUET_EXTENSIONS
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["point_cloud"]
    requires = ["pyarrow"]
    extension_confidence = 0.3
    export_help = (
        "ParaView: File > Save Data > CSV (point data). Fluent: File > Export > ASCII, or "
        "report-file monitors (.out). CFX / COMSOL / STAR-CCM+: export monitor or probe tables "
        "as CSV/text. OpenFOAM: the probes and forceCoeffs function objects write "
        "postProcessing/<name>/<time>/ files that are read directly."
    )
    status_note = (
        "Layouts: point snapshot, long (time, probe) tables, time + signal columns. Global signals "
        "are stored on one point at the origin. Parquet needs pyarrow. ParaView, Fluent, CFX and "
        "COMSOL style tables and Parquet are validated on generated files only. Tecplot ASCII and "
        "COMSOL '% Model:' files belong to their own connectors. Date-string time columns and "
        "ragged probe series (without per_probe) are not supported."
    )

    # detect -----------------------------------------------------------------------------
    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        if path.is_dir():
            if (path / "system" / "controlDict").exists():
                return None
            try:
                files = _foreign_free_probe_files(path)
            except OSError:
                return None
            if not files:
                return None
            return Detection(
                self.id, path, 0.85, "openfoam-probes", [f"OpenFOAM probes: {sorted(files)}"]
            )
        if not path.is_file():
            return None
        ext = match_extension(path, self.extensions)
        try:
            with open(path, "rb") as f:
                head = f.read(4096)
        except OSError:
            return None
        if ext in PARQUET_EXTENSIONS or (ext is None and head[:4] == b"PAR1"):
            if head[:4] != b"PAR1":
                return None
            det = Detection(self.id, path, 0.6, "parquet", ["Parquet magic PAR1"])
            if importlib.util.find_spec("pyarrow") is None:
                det.needs.append("pyarrow (Apache-2.0)")
            return det
        if head.startswith(b"# Probe 0 (") or (ext is None and foam_probe_header(path) is not None):
            return Detection(self.id, path, 0.9, "openfoam-probes", ["'# Probe 0 (x y z)' header"])
        coeff = path.name.lower() in _FOAM_COEFF_FILES
        if ext is None and not coeff:
            return None
        if _foreign_text(head) is not None:
            return None
        src = sniff_text(path)
        if src is None or len(src.columns) < 2:
            return None
        try:
            lay = interpret(src, None)
        except MappingError:
            conf = 0.3 if ext in (".csv", ".tsv") else 0.25
            return Detection(
                self.id, path, conf, "text-table",
                [f"numeric table with {len(src.columns)} columns, no coordinate/time columns"],
                ["a mapping config naming the coordinate or time columns"],
            )  # fmt: skip
        conf = 0.5 if ext in (".csv", ".tsv") else 0.45
        return Detection(
            self.id,
            path,
            conf,
            "csv" if src.delimiter == "," else "text-table",
            [f"{lay.layout} table: coordinates {lay.coords or '-'}, time {lay.time or '-'}"],
        )

    # plan -------------------------------------------------------------------------------
    def _source(self, path: Path, mapping: Mapping[str, Any] | None) -> _Source:
        if not path.is_file():
            raise UnsupportedSourceError(f"{path} is not a file", hint="pass one table file")
        with open(path, "rb") as f:
            head = f.read(4)
        if head == b"PAR1" or match_extension(path, PARQUET_EXTENSIONS):
            return open_parquet(path)
        src = sniff_text(path, mapping)
        if src is None:
            raise ConnectorError(
                f"{path.name} is not a delimited numeric table",
                hint="expected a header line (or a '# name name ...' comment) followed by rows "
                "of numbers; set mapping 'delimiter', 'skip_rows' or 'header' if the sniffing "
                "is wrong",
            )
        return src

    @staticmethod
    def _mapping(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
        from frontier_connectors._arraymap import load_mapping

        return load_mapping((options or {}).get("mapping"))

    def _is_foam(self, path: Path) -> bool:
        if path.is_dir():
            return True
        return foam_probe_header(path) is not None

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        if not path.exists():
            raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
        mapping = self._mapping(options)
        desc = SourceDescription(connector_id=self.id, path=path)
        if self._is_foam(path):
            return self._describe_foam(path, desc)
        src = self._source(path, mapping)
        lay = interpret(src, mapping)
        desc.format = src.format if src.format == "parquet" else f"text-table/{lay.layout}"
        desc.warnings.extend(lay.warnings)
        m = mapping or {}
        desc.domain = str(m.get("domain") or (options or {}).get("domain") or "other")
        desc.solver = str(m.get("solver") or "unknown")
        desc.parameters = dict(m.get("parameters") or {})
        if lay.length_unit:
            desc.units["length"] = lay.length_unit
        zone = ZoneInfo(name="points" if lay.layout == "points" else "probes", kind="point_cloud")
        zone.role, zone.dim = "probes", max(len(lay.coords), 1)
        widths = {c.name: c.width for c in src.columns}
        for name, cols in lay.fields.items():
            first = cols[0]
            comps = len(first) if isinstance(first, list) else len(cols)
            if len(cols) == 1 and isinstance(first, str) and widths.get(first, 1) > 1:
                comps = widths[first]
            desc.fields.append(
                FieldInfo(
                    name=name,
                    location="point",
                    components=comps,
                    units=lay.units.get(name, ""),
                    time_dependent=lay.layout != "points",
                )
            )
        if lay.layout == "points":
            table = src.load(lay.coords)
            pts = _stack(table, lay.coords)
            zone.n_points, zone.bounds = len(pts), meshutil.bounds(pts)
        elif lay.layout == "series":
            assert lay.time is not None
            t = src.load([lay.time])[lay.time]
            desc.time = TimeInfo(
                values=[float(v) for v in t], units=lay.time_unit, quantity=lay.time_quantity
            )
            zone.n_points = 1 if lay.positions is None else len(lay.positions)
            zone.dim = 3 if lay.positions is None else lay.positions.shape[1]
        else:
            assert lay.time is not None
            need = [lay.time, *lay.coords, *([lay.group_by] if lay.group_by else [])]
            pr = _pivot(src, lay, src.load(need), [])
            zone.n_points = len(pr.labels)
            if pr.ragged is not None:
                desc.warnings.append(
                    "probe series differ in length: read with options={'per_probe': True}"
                )
            else:
                zone.bounds = meshutil.bounds(pr.positions.reshape(-1, pr.positions.shape[-1]))
            if len(pr.times) > 1:
                desc.time = TimeInfo(
                    values=[float(v) for v in pr.times],
                    units=lay.time_unit,
                    quantity=lay.time_quantity,
                )
            else:
                for f in desc.fields:
                    f.time_dependent = False
                desc.metadata["time_value"] = float(pr.times[0]) if len(pr.times) else None
            desc.metadata["probes"] = pr.labels[:50]
        desc.zones.append(zone)
        desc.metadata["layout"] = lay.layout
        desc.metadata["columns"] = [c.raw for c in src.columns[:200]]
        if src.notes.get("header_values"):
            desc.metadata["header_values"] = src.notes["header_values"]
        return desc

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        path = Path(path)
        if not path.exists():
            raise UnsupportedSourceError(f"{path} does not exist", hint="check the path")
        mapping = self._mapping(options)
        base = meshutil.default_sample_id(path)
        if self._is_foam(path):
            yield from self._read_foam(path, base, zones, fields, times)
            return
        src = self._source(path, mapping)
        lay = interpret(src, mapping)
        zone = "points" if lay.layout == "points" else "probes"
        match_names([zone], zones)
        if fields is not None:
            match_names(lay.fields, [f for f in fields if not f.startswith("!")], what="field")
        names = [n for n in lay.fields if wants(n, fields)]
        m = mapping or {}
        md: dict[str, Any] = {
            "solver": str(m.get("solver") or "unknown"),
            "source": str(path),
            "format": src.format,
            "layout": lay.layout,
        }
        if lay.length_unit:
            md["units"] = {"length": lay.length_unit}
        if m.get("parameters"):
            md["parameters"] = dict(m["parameters"])
        md.update(dict(m.get("metadata") or {}))
        if src.notes.get("header_values"):
            md["header_values"] = src.notes["header_values"]

        def fd(data: np.ndarray, name: str, lead: list[str]) -> FieldData:
            return FieldData(
                data=np.ascontiguousarray(data, dtype=np.float32),
                dims=[*lead, "point", "component"],
                location="point",
                units=lay.units.get(name, ""),
            )

        if lay.layout == "points":
            table = src.load([*lay.coords, *_field_columns(lay, names)])
            yield RawSample(
                id=base,
                kind="point_cloud",
                zone=zone,
                points=_stack(table, lay.coords),
                fields={n: fd(_stack(table, lay.fields[n]), n, []) for n in names},
                metadata=md,
            )
            return
        assert lay.time is not None
        md["time_units"] = lay.time_unit
        if lay.layout == "series":
            table = src.load([lay.time, *_field_columns(lay, names)])
            t = np.asarray(table[lay.time], dtype=np.float64)
            order = np.argsort(t, kind="stable")
            keep = order[np.concatenate([[True], np.diff(t[order]) > 0])] if len(t) else order
            if len(keep) != len(t):
                md["dropped_repeated_times"] = int(len(t) - len(keep))
            sel = np.asarray(select_time_indices(t[keep], times), dtype=np.int64)
            rows = keep[sel]
            sample = RawSample(
                id=base,
                kind="point_cloud",
                zone=zone,
                points=np.zeros((1, 3)) if lay.positions is None else lay.positions,
                times=t[rows],
                metadata=md,
            )
            for n in names:
                if lay.positions is None:
                    data = _stack(table, lay.fields[n])[rows][:, None, :]
                else:
                    data = np.stack([_stack(table, p)[rows] for p in lay.fields[n]], axis=1)
                sample.fields[n] = fd(data, n, ["time"])
            yield sample
            return
        need = [lay.time, *lay.coords, *([lay.group_by] if lay.group_by else [])]
        pr = _pivot(src, lay, src.load([*need, *_field_columns(lay, names)]), names)
        md["probes"] = pr.labels[:1000]
        if pr.ragged is not None:
            if not (options or {}).get("per_probe"):
                lengths = sorted({len(v[0]) for v in pr.ragged.values()})
                raise ConnectorError(
                    f"{path.name}: probe series are ragged (lengths {lengths[:6]}): the probes "
                    "do not share one time axis",
                    hint="pass options={'per_probe': True} (CLI: --option per_probe=true) to "
                    "write one sample per probe, or resample the series onto common times",
                )
            for label, (t, p0, vals) in pr.ragged.items():
                sel = select_time_indices(t, times)
                yield RawSample(
                    id=f"{base}_{meshutil.sanitize_id(label)}",
                    kind="point_cloud",
                    zone=zone,
                    points=p0[None, :],
                    times=t[sel],
                    fields={n: fd(v[sel][:, None, :], n, ["time"]) for n, v in vals.items()},
                    metadata={**md, "probe": label},
                )
            return
        if len(pr.times) <= 1:
            pts = pr.positions if pr.positions.ndim == 2 else pr.positions[0]
            if len(pr.times):
                md["time_value"] = float(pr.times[0])
            yield RawSample(
                id=base,
                kind="point_cloud",
                zone=zone,
                points=pts,
                fields={n: fd(pr.values[n][0], n, []) for n in names},
                metadata=md,
            )
            return
        sel = select_time_indices(pr.times, times)
        if pr.positions.ndim == 2:
            yield RawSample(
                id=base,
                kind="point_cloud",
                zone=zone,
                points=pr.positions,
                times=pr.times[sel],
                fields={n: fd(pr.values[n][sel], n, ["time"]) for n in names},
                metadata=md,
            )
            return
        for ti in sel:  # moving points: one step at a time carries the positions
            yield RawSample(
                id=base,
                kind="point_cloud",
                zone=zone,
                points=pr.positions[ti],
                time=float(pr.times[ti]),
                time_index=int(ti),
                fields={n: fd(pr.values[n][ti], n, []) for n in names},
                metadata=md,
            )

    # OpenFOAM probes --------------------------------------------------------------------
    def _foam(self, path: Path) -> dict[str, tuple[np.ndarray, dict[str, list[Path]]]]:
        """``zone -> (positions, {field: files})``. Fields sampled on the same probe set share
        a zone; a field written for another probe set gets its own (``probes_<field>``)."""
        files = _foam_probe_files(path)
        if not files:
            raise UnsupportedSourceError(
                f"{path} holds no OpenFOAM probes files",
                hint="point at postProcessing/<probes name>, one of its time directories, or "
                "one field file; other directories are not tables",
            )
        zones: dict[str, tuple[np.ndarray, dict[str, list[Path]]]] = {}
        for name, paths in files.items():
            pos = foam_probe_header(paths[0])
            assert pos is not None
            for _zone, (zpos, zfiles) in zones.items():
                if zpos.shape == pos.shape and np.allclose(zpos, pos):
                    zfiles[name] = paths
                    break
            else:
                zones["probes" if not zones else f"probes_{name}"] = (pos, {name: paths})
        return zones

    def _foam_field(self, paths: Sequence[Path], n_probes: int) -> tuple[np.ndarray, np.ndarray]:
        t_all: list[np.ndarray] = []
        v_all: list[np.ndarray] = []
        for p in paths:
            t, v = _read_foam_probe_file(p, n_probes)
            if not len(t):
                continue  # a restart directory whose file only has the header
            if t_all:
                # a restart rewrites the overlap: keep the later run
                keep = t_all[-1] < t[0]
                t_all[-1], v_all[-1] = t_all[-1][keep], v_all[-1][keep]
            t_all.append(t)
            v_all.append(v)
        if not t_all:
            raise ConnectorError(
                f"{paths[0]}: the probes file has a header but no rows",
                hint="the run stopped before the first write of the function object",
            )
        if len({v.shape[2] for v in v_all}) != 1:
            raise ConnectorError(f"{paths[0].name}: component count changes between restarts")
        return np.concatenate(t_all), np.concatenate(v_all)

    def _describe_foam(self, path: Path, desc: SourceDescription) -> SourceDescription:
        zones = self._foam(path)
        desc.format, desc.solver, desc.domain = "openfoam-probes", "OpenFOAM", "cfd"
        desc.units = {"length": "m"}
        times: np.ndarray | None = None
        for zone, (positions, files) in zones.items():
            desc.zones.append(
                ZoneInfo(
                    name=zone,
                    kind="point_cloud",
                    role="probes",
                    n_points=len(positions),
                    bounds=meshutil.bounds(positions),
                )
            )
            ztimes: np.ndarray | None = None
            for name, paths in files.items():
                t, v = self._foam_field(paths, len(positions))
                ztimes = t if ztimes is None else np.intersect1d(ztimes, t)
                desc.fields.append(
                    FieldInfo(
                        name=name,
                        location="point",
                        components=v.shape[2],
                        zones=[zone] if len(zones) > 1 else None,
                    )
                )
            if times is None and ztimes is not None:
                times = ztimes
        desc.time = TimeInfo(values=[float(v) for v in (times if times is not None else [])])
        desc.metadata["layout"] = "openfoam-probes"
        if len(zones) > 1:
            desc.warnings.append(
                "the field files list different probe sets: one zone per probe set "
                f"({', '.join(zones)})"
            )
        return desc

    def _read_foam(
        self,
        path: Path,
        base: str,
        zones: Sequence[str] | None,
        fields: Sequence[str] | None,
        times: TimeSelection,
    ) -> Iterator[RawSample]:
        found = self._foam(path)
        selected = match_names(found, zones)
        if fields is not None:
            names = [n for _, files in found.values() for n in files]
            match_names(names, [f for f in fields if not f.startswith("!")], what="field")
        if not path.is_dir():
            base = f"{base}_{path.parent.parent.name or 'probes'}"
        for zone in selected:
            positions, files = found[zone]
            loaded = {
                n: self._foam_field(p, len(positions)) for n, p in files.items() if wants(n, fields)
            }
            if not loaded:
                continue
            common: np.ndarray | None = None
            for t, _ in loaded.values():
                common = t if common is None else np.intersect1d(common, t)
            if common is None or not len(common):
                raise ConnectorError(
                    f"{path}: the probe files of zone {zone!r} share no time value",
                    hint="read one field at a time with fields=[...]",
                )
            common = np.unique(common)
            sel = select_time_indices(common, times)
            sample = RawSample(
                id=meshutil.zone_sample_id(base, zone, len(selected)),
                kind="point_cloud",
                zone=zone,
                points=positions,
                times=common[sel],
                metadata={
                    "solver": "OpenFOAM",
                    "source": str(path),
                    "format": "openfoam-probes",
                    "time_units": "s",
                    "units": {"length": "m"},
                },
            )
            for name, (t, v) in loaded.items():
                rows = np.searchsorted(t, common[sel])
                sample.fields[name] = FieldData(
                    data=np.ascontiguousarray(v[rows], dtype=np.float32),
                    dims=["time", "point", "component"],
                    location="point",
                )
            yield sample


def _foreign_free_probe_files(path: Path) -> dict[str, list[Path]]:
    """Probes files of a directory, looking at most two levels deep (never a whole case)."""
    try:
        entries = list(itertools.islice(path.iterdir(), 500))
    except OSError:
        return {}
    numeric_dirs = 0
    for e in entries:
        if e.is_dir():
            try:
                float(e.name)
                numeric_dirs += 1
            except ValueError:
                continue
    files = [e for e in entries if e.is_file()]
    if not numeric_dirs and not files:
        return {}
    return _foam_probe_files(path)
