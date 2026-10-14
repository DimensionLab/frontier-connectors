# SPDX-License-Identifier: Apache-2.0
"""EnSight Gold (and EnSight 6) case files: the interchange export of STAR-CCM+, Fluent,
CONVERGE, PowerFLOW, AcuSolve, Code_Saturne, OpenFOAM (``foamToEnsight``) and many FE codes.

Three layers:

* :func:`parse_case` reads the ``.case`` text file itself (FORMAT / GEOMETRY / VARIABLE / TIME /
  FILE sections), so time values, the variable list and the file patterns come from the case
  file and not from a reader's interpretation of it;
* :func:`scan_geometry` / :func:`scan_variable` walk the EnSight Gold geometry and variable files
  record by record *without loading the arrays* (ASCII, C binary, Fortran binary). They give the
  part list for ``describe`` and, more importantly, prove that the files are complete: VTK's
  EnSight readers crash the process (segmentation fault) on a truncated geometry file and
  silently drop a truncated variable, so nothing is handed to VTK before it passed this scan;
* the heavy arrays are read with ``vtkGenericEnSightReader`` one time step at a time. Every
  EnSight *part* becomes a zone named after its description line; element based variables come
  out with ``location="cell"``, node based ones with ``"point"``.

:func:`write_ensight_gold` is a small writer (ASCII / C binary / Fortran binary) used by the
test-suites of this connector and of the connectors that route through it.
"""

from __future__ import annotations

import itertools
import re
import struct
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from frontier_connectors import _vtk, meshutil
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    match_extension,
)
from frontier_connectors.paths import case_root_of, resolve_in_case
from frontier_connectors.select import select_time_indices, wants

CASE_EXTENSIONS = [".case", ".encas"]

#: nodes per element of the fixed-size EnSight element types
ELEMENT_NODES: dict[str, int] = {
    "point": 1, "bar2": 2, "bar3": 3, "tria3": 3, "tria6": 6, "quad4": 4, "quad8": 8,
    "tetra4": 4, "tetra10": 10, "pyramid5": 5, "pyramid13": 13, "penta6": 6, "penta15": 15,
    "hexa8": 8, "hexa20": 20,
}  # fmt: skip

#: topological dimension per EnSight element type
ELEMENT_DIM: dict[str, int] = {
    "point": 0, "bar2": 1, "bar3": 1, "tria3": 2, "tria6": 2, "quad4": 2, "quad8": 2,
    "nsided": 2, "tetra4": 3, "tetra10": 3, "pyramid5": 3, "pyramid13": 3, "penta6": 3,
    "penta15": 3, "hexa8": 3, "hexa20": 3, "nfaced": 3,
}  # fmt: skip

_COMPONENTS = {"scalar": 1, "vector": 3, "tensor symm": 6, "tensor asym": 9}
#: component labels of what VTK hands back. EnSight stores symmetric tensors as 11 22 33 12 13
#: 23; VTK's Gold readers reorder them to its own xx yy zz xy yz xz (verified in the tests).
_COMPONENT_NAMES: dict[str, list[str]] = {
    "tensor symm": ["xx", "yy", "zz", "xy", "yz", "xz"],
    "tensor asym": ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"],
}

#: text found in case comments / geometry description lines -> (solver, domain)
_SOLVER_MARKERS: list[tuple[str, str, str]] = [
    ("star-ccm", "STAR-CCM+", "cfd"),
    ("starccm", "STAR-CCM+", "cfd"),
    ("fluent", "ANSYS Fluent", "cfd"),
    ("fvm library", "Code_Saturne", "cfd"),
    ("code_saturne", "Code_Saturne", "cfd"),
    ("code saturne", "Code_Saturne", "cfd"),
    ("converge", "CONVERGE", "cfd"),
    ("powerflow", "PowerFLOW", "cfd"),
    ("acusolve", "AcuSolve", "cfd"),
    ("acutrans", "AcuSolve", "cfd"),
    ("openfoam", "OpenFOAM", "cfd"),
    ("edelweissfe", "EdelweissFE", "structural"),
    ("written by vtk", "VTK", "other"),
]

_CASE_HINT = (
    "an EnSight case file starts with 'FORMAT' / 'type: ensight gold' followed by GEOMETRY "
    "('model: <file>'), VARIABLE and TIME sections; re-export from the solver"
)


# -- case file ---------------------------------------------------------------------------------


@dataclass
class TimeSet:
    """One ``time set`` of the TIME section."""

    id: int
    n_steps: int | None = None
    start: int | None = None
    increment: int = 1
    numbers: list[int] | None = None
    values: list[float] = field(default_factory=list)

    def file_number(self, step: int) -> int:
        if self.numbers is not None:
            return self.numbers[min(step, len(self.numbers) - 1)]
        return (self.start if self.start is not None else 0) + step * self.increment

    def step_at(self, time: float | None) -> int:
        """Index of the last step of this set that is not later than ``time`` (the EnSight
        rule for combining time sets of different length)."""
        if time is None or not self.values:
            return 0
        vals = np.asarray(self.values, dtype=np.float64)
        tol = 1e-9 * max(1.0, float(np.abs(vals).max()))
        return max(int(np.searchsorted(vals, float(time) + tol, side="right")) - 1, 0)


@dataclass
class CaseVariable:
    """One line of the VARIABLE section."""

    name: str
    type: str  # scalar | vector | tensor symm | tensor asym
    location: str  # node | element | measured node | case
    complex: bool = False
    time_set: int | None = None
    file_set: int | None = None
    files: list[str] = field(default_factory=list)
    #: ``constant per case`` values (one, or one per time step)
    values: list[float] = field(default_factory=list)

    @property
    def components(self) -> int:
        return _COMPONENTS.get(self.type, 1)

    @property
    def time_dependent(self) -> bool:
        return any("*" in f for f in self.files) or self.file_set is not None


@dataclass
class CaseFile:
    """Parsed ``.case`` file."""

    path: Path
    version: str = "gold"  # gold | 6
    model: str | None = None
    model_time_set: int | None = None
    model_file_set: int | None = None
    change_coords_only: bool = False
    measured: str | None = None
    variables: list[CaseVariable] = field(default_factory=list)
    time_sets: dict[int, TimeSet] = field(default_factory=dict)
    #: file set id -> number of steps of each file of the set
    file_sets: dict[int, list[int]] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def moving(self) -> bool:
        """Geometry changes in time (wildcard pattern or single-file transient geometry)."""
        if self.model is None:
            return False
        if "*" in self.model:
            return True
        ts = self.time_set(self.model_time_set)
        return self.model_file_set is not None and ts is not None and len(ts.values) > 1

    def time_set(self, ts: int | None) -> TimeSet | None:
        if not self.time_sets:
            return None
        if ts is not None:
            return self.time_sets.get(ts)
        return self.time_sets.get(1) or next(iter(self.time_sets.values()))

    def times(self) -> list[float]:
        """Sorted union of the time values of every time set that a file pattern uses."""
        used: list[TimeSet] = []
        refs = [(self.model or "", self.model_time_set, self.model_file_set)]
        refs += [(f, v.time_set, v.file_set) for v in self.variables for f in v.files[:1]]
        for pattern, ts, fs in refs:
            if "*" in pattern or ts is not None or fs is not None:
                s = self.time_set(ts)
                if s is not None and s not in used:
                    used.append(s)
        if not used:
            used = list(self.time_sets.values())
        values = sorted({float(v) for s in used for v in s.values})
        return values

    def resolve(self, pattern: str, ts: int | None, time: float | None) -> Path:
        """File behind ``pattern`` at physical ``time`` (wildcards replaced by the step's file
        number, zero padded to the number of ``*``)."""
        name = pattern.replace("\\", "/")
        if "*" in name:
            s = self.time_set(ts)
            number = s.file_number(s.step_at(time)) if s is not None else 0
            name = re.sub(r"\*+", lambda m: str(number).zfill(len(m.group(0))), name)
        return resolve_in_case(
            case_root_of(self.path), name, what="EnSight case file reference", referrer=self.path
        )

    def geometry_file(self, time: float | None) -> Path:
        assert self.model is not None
        return self.resolve(self.model, self.model_time_set, time)

    def constants(self, time_index: int) -> dict[str, float]:
        out = {}
        for v in self.variables:
            if v.location == "case" and v.values:
                out[v.name] = float(v.values[min(time_index, len(v.values) - 1)])
        return out


_SECTION = re.compile(r"^[A-Z][A-Z_ ]*$")
_KEY = re.compile(r"^([A-Za-z_][A-Za-z_ ]*?)\s*:\s*(.*)$")
_TOKEN = re.compile(r'"[^"]*"|\S+')
_VAR_KEY = re.compile(
    r"^(complex )?(scalar|vector|tensor symm|tensor asym) per (node|element|measured node)$"
)
_TYPE_LINE = re.compile(r"^\s*type\s*:\s*ensight(\s+gold)?\s*$", re.IGNORECASE | re.MULTILINE)


def _tokens(text: str) -> list[str]:
    return [t.strip('"') for t in _TOKEN.findall(text)]


def _is_int(tok: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+", tok))


def _floats(tokens: Sequence[str], path: Path, what: str) -> list[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError as e:
        raise ConnectorError(f"{path}: bad number in '{what}': {e}", hint=_CASE_HINT) from e


def _int(tokens: Sequence[str], path: Path, what: str) -> int:
    try:
        return int(tokens[0])
    except (ValueError, IndexError) as e:
        raise ConnectorError(f"{path}: '{what}' needs an integer", hint=_CASE_HINT) from e


def parse_case(path: Path) -> CaseFile:
    """Parse an EnSight ``.case`` file. Raises :class:`ConnectorError` for anything that is not
    a well-formed case file (missing FORMAT / model line, unknown variable kind, bad numbers)."""
    path = Path(path)
    if not path.is_file():
        raise UnsupportedSourceError(f"{path} does not exist", hint="pass the .case file")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ConnectorError(
            f"{path} is too large to be an EnSight case file",
            hint="pass the small text .case file, not a geometry or variable file",
        )
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ConnectorError(f"{path} is binary, not an EnSight case file", hint=_CASE_HINT)
    case = CaseFile(path=path)
    case_root = case_root_of(path)
    other_refs: list[tuple[str, str]] = []
    section = ""
    seen_format = False
    time_set: TimeSet | None = None
    file_set: int | None = None
    continuation: list[Any] | None = None
    continuation_kind = float
    for raw_line in raw.decode("latin-1").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            case.comments.append(line.lstrip("#").strip())
            continue
        if _SECTION.match(line) and ":" not in line:
            section = line.split()[0]
            continuation = None
            continue
        m = _KEY.match(line)
        if m is None:
            if continuation is not None and section == "TIME":
                toks = _tokens(line)
                if continuation_kind is int:
                    continuation.extend(int(v) for v in _floats(toks, path, "filename numbers"))
                else:
                    continuation.extend(_floats(toks, path, "time values"))
                continue
            raise ConnectorError(
                f"{path}: cannot parse line {line[:60]!r} in section {section or '(none)'}",
                hint=_CASE_HINT,
            )
        key, value = m.group(1).strip().lower(), m.group(2).strip()
        toks = _tokens(value)
        continuation = None
        if not section:
            raise ConnectorError(f"{path}: '{key}:' appears before any section", hint=_CASE_HINT)
        if section == "FORMAT":
            if key == "type":
                kind = value.lower().split()
                if not kind or kind[0] != "ensight":
                    raise ConnectorError(
                        f"{path}: unsupported case type {value!r}",
                        hint="only 'type: ensight gold' and 'type: ensight' (EnSight 6) exist",
                    )
                case.version = "gold" if kind[1:2] == ["gold"] else "6"
                seen_format = True
        elif section == "GEOMETRY":
            if key == "model":
                if toks and toks[-1].isdigit() and "change_coords_only" in toks[:-1]:
                    toks = toks[:-1]  # optional 'cstep' after change_coords_only
                if "change_coords_only" in toks:
                    case.change_coords_only = True
                    toks = [t for t in toks if t != "change_coords_only"]
                lead = 0
                while lead < 2 and lead < len(toks) - 1 and _is_int(toks[lead]):
                    lead += 1
                if len(toks) - lead != 1:
                    raise ConnectorError(
                        f"{path}: cannot parse 'model: {value}'",
                        hint="expected 'model: [time set] [file set] <geometry file>'",
                    )
                ints = [int(t) for t in toks[:lead]]
                case.model_time_set = ints[0] if ints else None
                case.model_file_set = ints[1] if len(ints) > 1 else None
                case.model = toks[-1]
            elif key == "measured":
                case.measured = toks[-1] if toks else None
            elif toks:
                # match / boundary / rigid_body files are not needed to read the fields, but
                # the VTK reader opens them: they are confined like every other reference
                other_refs.append((key, toks[-1]))
        elif section == "VARIABLE":
            case.variables.append(_parse_variable(path, key, toks))
        elif section == "TIME":
            if key == "time set":
                time_set = TimeSet(id=_int(toks, path, key))
                case.time_sets[time_set.id] = time_set
                continue
            if time_set is None:
                time_set = TimeSet(id=1)
                case.time_sets[1] = time_set
            if key == "number of steps":
                time_set.n_steps = _int(toks, path, key)
            elif key == "filename start number":
                time_set.start = _int(toks, path, key)
            elif key == "filename increment":
                time_set.increment = _int(toks, path, key)
            elif key == "filename numbers":
                time_set.numbers = [int(v) for v in _floats(toks, path, key)]
                continuation, continuation_kind = time_set.numbers, int
            elif key == "time values":
                time_set.values = _floats(toks, path, key)
                continuation, continuation_kind = time_set.values, float
            elif key in ("time values file", "filename numbers file"):
                ref = resolve_in_case(
                    case_root, (toks[-1] if toks else "").replace("\\", "/"),
                    what=f"EnSight '{key}'", referrer=path,
                )  # fmt: skip
                if not ref.is_file():
                    raise ConnectorError(
                        f"{path}: '{key}' references missing file {ref}",
                        hint="copy the file next to the case file",
                    )
                vals = _floats(ref.read_text().split(), ref, key)
                if key.startswith("time"):
                    time_set.values = vals
                else:
                    time_set.numbers = [int(v) for v in vals]
        elif section == "FILE":
            if key == "file set":
                file_set = _int(toks, path, key)
                case.file_sets[file_set] = []
            elif key == "number of steps" and file_set is not None:
                case.file_sets[file_set].append(_int(toks, path, key))
        # MATERIAL, BLOCK_CONTINUATION, SCRIPTS, ...: not needed for the fields
    if not seen_format:
        raise ConnectorError(f"{path} has no FORMAT section with a 'type:' line", hint=_CASE_HINT)
    if case.model is None:
        raise ConnectorError(
            f"{path} has no 'model:' line in a GEOMETRY section",
            hint="the case file must name its geometry file: 'model: <name>.geo'",
        )
    for ts in case.time_sets.values():
        n = ts.n_steps if ts.n_steps is not None else len(ts.values)
        if not ts.values:
            case.warnings.append(
                f"time set {ts.id} has no 'time values': step numbers are used as time"
            )
            ts.values = [float(i) for i in range(n)]
        elif len(ts.values) < n:
            raise ConnectorError(
                f"{path}: time set {ts.id} declares {n} steps but lists {len(ts.values)} "
                "time values",
                hint="the case file is truncated or was edited by hand; fix the TIME section",
            )
        elif len(ts.values) > n:
            ts.values = ts.values[:n]
        if ts.numbers is not None and len(ts.numbers) < n:
            raise ConnectorError(
                f"{path}: time set {ts.id} declares {n} steps but lists {len(ts.numbers)} "
                "filename numbers",
                hint="fix 'filename numbers:' in the TIME section",
            )
    if case.measured:
        case.warnings.append("measured (particle) geometry is listed but not validated")
    # every file the case names must stay inside the case directory (the directory of the
    # case file): the VTK reader opens them on its own. `*` wildcards stand for digits.
    refs = [("model", case.model), ("measured", case.measured), *other_refs]
    refs += [(f"variable {v.name}", f) for v in case.variables for f in v.files]
    for label, ref in refs:
        if ref:
            resolve_in_case(
                case_root,
                ref.replace("\\", "/").replace("*", "0"),
                what=f"EnSight {label} file",
                referrer=path,
            )
    return case


def _parse_variable(path: Path, key: str, toks: list[str]) -> CaseVariable:
    if key == "constant per case":
        lead = 1 if len(toks) > 2 and _is_int(toks[0]) and not _is_number(toks[1]) else 0
        if len(toks) - lead < 2:
            raise ConnectorError(
                f"{path}: 'constant per case' needs a name and a value", hint=_CASE_HINT
            )
        return CaseVariable(
            name=toks[lead],
            type="scalar",
            location="case",
            time_set=int(toks[0]) if lead else None,
            values=_floats(toks[lead + 1 :], path, key),
        )
    if key == "constant per case file":
        return CaseVariable(name=toks[-2] if len(toks) > 1 else "constant", type="scalar",
                            location="case")  # fmt: skip
    m = _VAR_KEY.match(key)
    if m is None:
        raise ConnectorError(
            f"{path}: unknown VARIABLE line '{key}:'",
            hint="valid kinds: '[complex] scalar|vector|tensor symm|tensor asym per "
            "node|element|measured node' and 'constant per case'",
        )
    is_complex = m.group(1) is not None
    n_files = 2 if is_complex else 1
    tail = n_files + (1 if is_complex else 0)  # complex lines end with the frequency
    if is_complex and len(toks) >= 3 and not _is_number(toks[-1]):
        tail = n_files
    if len(toks) < tail + 1:
        raise ConnectorError(
            f"{path}: '{key}:' needs a description and a file name", hint=_CASE_HINT
        )
    head = toks[: len(toks) - tail]
    ints = [int(t) for t in head[:-1] if _is_int(t)][:2]
    return CaseVariable(
        name=head[-1],
        type=m.group(2),
        location=m.group(3),
        complex=is_complex,
        time_set=ints[0] if ints else None,
        file_set=ints[1] if len(ints) > 1 else None,
        files=toks[len(head) : len(head) + n_files],
    )


def _is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


# -- EnSight Gold record scanner ---------------------------------------------------------------


class _TruncatedError(Exception):
    """The file ends in the middle of a record."""


class _UnknownLayoutError(Exception):
    """A valid but rarely used layout this scanner does not follow (validation is skipped)."""


class _MalformedError(Exception):
    """The file is not EnSight Gold."""


@dataclass
class PartScan:
    number: int
    name: str
    n_points: int = 0
    #: ``[(element type as written, count)]`` in file order
    elements: list[tuple[str, int]] = field(default_factory=list)
    #: point dimensions of a structured ``block`` part
    shape: tuple[int, int, int] | None = None

    @property
    def n_cells(self) -> int:
        if self.shape is not None:
            return int(np.prod([max(d - 1, 1) for d in self.shape]))
        return sum(n for _, n in self.elements)

    @property
    def dim(self) -> int:
        if self.shape is not None:
            return sum(1 for d in self.shape if d > 1)
        return max((ELEMENT_DIM.get(_base_type(t), 3) for t, _ in self.elements), default=0)


@dataclass
class GeometryScan:
    path: Path
    mode: str  # ascii | c | fortran
    byte_order: str = "<"
    description: list[str] = field(default_factory=list)
    parts: list[PartScan] = field(default_factory=list)
    #: the file holds several time steps (``BEGIN TIME STEP``); only the first was scanned
    single_file: bool = False

    def part(self, number: int) -> PartScan | None:
        return next((p for p in self.parts if p.number == number), None)


def _base_type(etype: str) -> str:
    t = etype.lower()
    return t[2:] if t.startswith("g_") else t


class _GoldStream:
    """Record-level access to an EnSight Gold file in any of its three encodings."""

    def __init__(self, f: BinaryIO, size: int, mode: str, byte_order: str = "<") -> None:
        self.f = f
        self.size = size
        self.mode = mode
        self.order = byte_order

    def _take(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) < n:
            raise _TruncatedError
        return data

    def _marker(self) -> int:
        return int(struct.unpack(self.order + "i", self._take(4))[0])

    def string(self) -> str | None:
        """Next 80-character record (ASCII: next line); ``None`` at a clean end of file."""
        if self.mode == "ascii":
            line = self.f.readline()
            if not line:
                return None
            return line.decode("latin-1").strip()
        if self.f.tell() >= self.size:
            return None
        if self.mode == "fortran":
            n = self._marker()
            if n != 80:
                raise _MalformedError(f"expected an 80 byte text record, found {n} bytes")
            data = self._take(80)
            self._take(4)
        else:
            data = self._take(80)
        return data.split(b"\x00", 1)[0].decode("latin-1").strip()

    def need_string(self) -> str:
        s = self.string()
        if s is None:
            raise _TruncatedError
        return s

    def integers(self, n: int) -> np.ndarray:
        """``n`` integers (ASCII: any number per line)."""
        if n < 0:
            raise _MalformedError("negative count")
        if self.mode == "ascii":
            out: list[int] = []
            while len(out) < n:
                line = self.f.readline()
                if not line:
                    raise _TruncatedError
                try:
                    out.extend(int(t) for t in line.split())
                except ValueError as e:
                    raise _MalformedError(f"expected integers, found {line[:40]!r}") from e
            return np.asarray(out[:n], dtype=np.int64)
        if self.mode == "fortran":
            m = self._marker()
            if m != 4 * n:
                raise _MalformedError(f"expected a record of {n} integers, found {m} bytes")
        data = self._take(4 * n)
        if self.mode == "fortran":
            self._take(4)
        return np.frombuffer(data, dtype=self.order + "i4").astype(np.int64)

    def integer(self) -> int:
        return int(self.integers(1)[0])

    def skip(self, n_items: int, n_lines: int | None = None) -> None:
        """Skip one array of ``n_items`` 4-byte values (ASCII: ``n_lines`` lines, default one
        value per line), failing when the file ends first."""
        if n_items < 0:
            raise _MalformedError("negative count")
        if n_items == 0:
            return
        if self.mode == "ascii":
            want = n_items if n_lines is None else n_lines
            got = sum(1 for _ in itertools.islice(self.f, want))
            if got < want:
                raise _TruncatedError
            return
        nbytes = 4 * n_items
        if self.mode == "fortran":
            m = self._marker()
            if m != nbytes:
                raise _MalformedError(f"expected a record of {nbytes} bytes, found {m}")
            nbytes += 4
        pos = self.f.tell()
        if pos + nbytes > self.size:
            raise _TruncatedError
        self.f.seek(pos + nbytes)


def _open_geometry(f: BinaryIO, size: int) -> _GoldStream:
    head = f.read(256)
    if b"c binary" in head[:80].lower():
        f.seek(80)
        return _GoldStream(f, size, "c")
    if b"fortran binary" in head[:96].lower():
        order = "<" if head[:4] == struct.pack("<i", 80) else ">"
        f.seek(88)
        return _GoldStream(f, size, "fortran", order)
    if b"\x00" in head or not head.strip():
        raise _MalformedError("neither text nor a 'C Binary' / 'Fortran Binary' header")
    f.seek(0)
    return _GoldStream(f, size, "ascii")


def _fix_byte_order(s: _GoldStream) -> int:
    """Read the first part number and switch to big endian when it only makes sense that way
    (C binary files carry no byte-order mark)."""
    if s.mode != "c":
        return s.integer()
    raw = s._take(4)
    little = int(struct.unpack("<i", raw)[0])
    big = int(struct.unpack(">i", raw)[0])
    if not 0 <= little < 1 << 24 and 0 <= big < 1 << 24:
        s.order = ">"
        return big
    return little


def scan_geometry(path: Path) -> GeometryScan:
    """Walk an EnSight Gold geometry file and return its parts (names, node and element
    counts) without loading coordinates or connectivity.

    Raises :class:`ConnectorError` when the file is truncated or not EnSight Gold. A valid
    layout the scanner does not follow (ghost/ranged blocks) ends the scan early with the parts
    found so far."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        scan = GeometryScan(path=path, mode="ascii")
        try:
            s = _open_geometry(f, size)
            scan.mode = s.mode
            _scan_geometry(s, scan)
            scan.byte_order = s.order
        except _TruncatedError as e:
            raise ConnectorError(
                f"EnSight geometry file {path} is truncated ({size} bytes)",
                hint="the export was interrupted or the copy is incomplete; export again",
            ) from e
        except _MalformedError as e:
            raise ConnectorError(
                f"{path} is not an EnSight Gold geometry file: {e}",
                hint="check that the case file's 'model:' line points at the geometry file and "
                "that the export finished",
            ) from e
        except _UnknownLayoutError:
            pass
    return scan


def geometry_description(path: Path) -> list[str]:
    """The two free-text description lines at the top of an EnSight Gold geometry file (most
    writers put their name and version there). Reads a few hundred bytes; ``[]`` on failure."""
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            s = _open_geometry(f, size)
            line = s.need_string()
            if line.upper().startswith("BEGIN TIME STEP"):
                line = s.need_string()
            return [line[:200], s.need_string()[:200]]
    except (OSError, _TruncatedError, _MalformedError, UnicodeDecodeError):
        return []


def _scan_geometry(s: _GoldStream, scan: GeometryScan) -> None:
    line = s.need_string()
    if line.upper().startswith("BEGIN TIME STEP"):
        scan.single_file = True
        line = s.need_string()
    scan.description = [line, s.need_string()]
    node_line = s.need_string().lower()
    elem_line = s.need_string().lower()
    if not node_line.startswith("node id") or not elem_line.startswith("element id"):
        raise _MalformedError("the 'node id' / 'element id' header lines are missing")
    node_ids = node_line.split()[-1] in ("given", "ignore")
    elem_ids = elem_line.split()[-1] in ("given", "ignore")
    line = s.string()
    if line is not None and line.lower().startswith("extents"):
        s.skip(6, 3)
        line = s.string()
    first = True
    while line is not None and not line.upper().startswith("END TIME STEP"):
        if line.lower() != "part":
            raise _MalformedError(f"expected 'part', found {line[:40]!r}")
        number = _fix_byte_order(s) if first else s.integer()
        first = False
        part = PartScan(number=number, name=s.need_string())
        scan.parts.append(part)
        kind = s.need_string().lower().split()
        if kind[:1] == ["coordinates"]:
            part.n_points = s.integer()
            if node_ids:
                s.skip(part.n_points)
            for _ in range(3):
                s.skip(part.n_points)
            line = _scan_elements(s, part, elem_ids)
        elif kind[:1] == ["block"]:
            line = _scan_block(s, part, kind[1:])
        else:
            raise _MalformedError(f"expected 'coordinates' or 'block', found {kind}")


def _scan_elements(s: _GoldStream, part: PartScan, elem_ids: bool) -> str | None:
    while True:
        line = s.string()
        if line is None or line.lower() == "part" or line.upper().startswith("END TIME STEP"):
            return line
        words = line.split()
        etype = words[0] if words else ""
        base = _base_type(etype)
        if base not in ELEMENT_DIM:
            raise _MalformedError(f"unknown element type {line[:40]!r}")
        ne = s.integer()
        if elem_ids:
            s.skip(ne)
        if base == "nsided":
            counts = s.integers(ne)
            s.skip(int(counts.sum()), ne)
        elif base == "nfaced":
            n_faces = s.integers(ne)
            total = int(n_faces.sum())
            per_face = s.integers(total)
            s.skip(int(per_face.sum()), total)
        else:
            s.skip(ne * ELEMENT_NODES[base], ne)
        part.elements.append((etype, ne))


def _scan_block(s: _GoldStream, part: PartScan, flags: list[str]) -> str | None:
    if "with_ghost" in flags or "range" in flags:
        raise _UnknownLayoutError
    ijk = s.integers(3)
    ni, nj, nk = (int(v) for v in ijk)
    part.shape = (ni, nj, nk)
    n = ni * nj * nk
    part.n_points = n
    if "uniform" in flags:
        s.skip(3)
        s.skip(3)
    elif "rectilinear" in flags:
        s.skip(ni)
        s.skip(nj)
        s.skip(nk)
    else:
        for _ in range(3):
            s.skip(n)
    if "iblanked" in flags:
        s.skip(n)
    line = s.string()
    if line is not None and line.lower().split()[:1] in (["node_ids"], ["element_ids"]):
        raise _UnknownLayoutError
    return line


def scan_variable(path: Path, geometry: GeometryScan, variable: CaseVariable) -> list[int]:
    """Check that one EnSight Gold variable file is complete for the parts of ``geometry`` and
    return the numbers of the parts it carries (first time step of a single-file transient).

    Raises :class:`ConnectorError` when the file is truncated; returns ``[]`` for layouts the
    scanner does not follow."""
    path = Path(path)
    size = path.stat().st_size
    found: list[int] = []
    with open(path, "rb") as f:
        s = _GoldStream(f, size, geometry.mode, geometry.byte_order)
        try:
            _scan_variable(s, geometry, variable, found)
        except _TruncatedError as e:
            raise ConnectorError(
                f"EnSight variable file {path} ({variable.name}) is truncated or does not "
                f"match the geometry ({size} bytes)",
                hint="the export was interrupted, or the file belongs to another mesh; export "
                "again or exclude the field with fields=['!" + variable.name + "']",
            ) from e
        except (_MalformedError, _UnknownLayoutError, ValueError):
            return []
    return found


def _scan_variable(
    s: _GoldStream, geometry: GeometryScan, variable: CaseVariable, found: list[int]
) -> None:
    line = s.need_string()
    if line.upper().startswith("BEGIN TIME STEP"):
        s.need_string()
    nc = variable.components
    line = s.string()
    while line is not None and not line.upper().startswith("END TIME STEP"):
        if line.lower() != "part":
            raise _MalformedError("expected 'part'")
        part = geometry.part(s.integer())
        if part is None:
            raise _UnknownLayoutError
        found.append(part.number)
        line = s.string()
        while line is not None:
            words = line.lower().split()
            if not words or words[0] == "part" or line.upper().startswith("END TIME STEP"):
                break
            if words[0] == "coordinates":
                count = part.n_points
            elif words[0] == "block":
                count = part.n_points if variable.location == "node" else part.n_cells
            else:
                counts = {t.lower(): n for t, n in part.elements}
                if words[0] not in counts:
                    raise _UnknownLayoutError
                count = counts[words[0]]
            if "undef" in words[1:]:
                s.skip(1)
            elif "partial" in words[1:]:
                count = s.integer()
                s.skip(count)
            for _ in range(nc):
                s.skip(count)
            line = s.string()


# -- VTK plumbing ------------------------------------------------------------------------------


@contextmanager
def quiet_vtk() -> Iterator[None]:
    """Silence VTK's stderr log while a reader runs (errors are collected by an observer)."""
    from vtkmodules.vtkCommonCore import vtkLogger

    old = vtkLogger.GetCurrentVerbosityCutoff()
    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
    try:
        yield
    finally:
        vtkLogger.SetStderrVerbosity(old)


class VtkErrorLog:
    """VTK ``ErrorEvent`` observer that keeps the messages."""

    CallDataType = "string0"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, obj: Any, event: str, message: str) -> None:
        self.messages.append(str(message).strip().splitlines()[-1] if message else "error")


def _dataset_dim(ds: Any) -> int:
    """Topological dimension of a VTK dataset (highest cell dimension)."""
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    if isinstance(ds, vtkUnstructuredGrid):
        types = np.unique(_vtk.cell_types(ds))
        return max((meshutil.VTK_TYPE_DIM.get(int(t), 3) for t in types), default=0)
    if hasattr(ds, "GetDimensions"):
        return sum(1 for d in _vtk.grid_dims(ds) if d > 1)
    return 3 if ds.GetNumberOfCells() else 0


def unique_names(names: Sequence[str]) -> list[str]:
    """Strip the padding of binary part descriptions and make repeated names unique."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(names):
        name = raw.strip() or f"part{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        out.append(name)
    return out


def unique_sample_ids(base: str, zones: Sequence[str]) -> dict[str, str]:
    """``zone -> sample id`` for the selected zones. Zone names that only differ in characters
    an id cannot hold (``SIDE WALL`` / ``SIDE WALL@``) get a numeric suffix."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for zone in zones:
        sid = candidate = meshutil.zone_sample_id(base, zone, len(zones))
        n = 1
        while candidate in taken:
            n += 1
            candidate = f"{sid}_{n}"
        taken.add(candidate)
        out[zone] = candidate
    return out


def drop_unchanged_geometry(sample: RawSample, first: dict[str, RawSample]) -> None:
    """For sources that store a complete mesh in every step: clear the geometry of ``sample``
    when it equals the first step of the same id (``first`` keeps that step's geometry only),
    so that a static mesh is written once."""
    if sample.points is None:
        return
    ref = first.get(sample.id)
    if ref is None:
        first[sample.id] = RawSample(
            id=sample.id,
            kind=sample.kind,
            points=sample.points,
            cells=sample.cells,
            faces=sample.faces,
        )
        return

    def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
        if a is None or b is None:
            return a is b
        return a.shape == b.shape and bool(np.array_equal(a, b))

    if (
        same(ref.points, sample.points)
        and same(ref.cells, sample.cells)
        and same(ref.faces, sample.faces)
    ):
        sample.points = None
        sample.cells = None
        sample.cell_types = None
        sample.faces = None
        sample.static.pop("face_parent", None)


def sniff_solver(texts: Sequence[str]) -> tuple[str, str | None, str]:
    """``(solver, version, domain)`` guessed from case comments and geometry description lines."""
    blob = " | ".join(texts)
    low = blob.lower()
    for marker, solver, domain in _SOLVER_MARKERS:
        at = low.find(marker)
        if at >= 0:
            m = re.search(r"\d+(?:\.\d+)+", blob[at + len(marker) : at + len(marker) + 40])
            return solver, (m.group(0) if m else None), domain
    return "unknown", None, "other"


class _Session:
    """One opened case: parsed case file, validated files, configured VTK reader."""

    def __init__(self, case: CaseFile) -> None:
        self.case = case
        self.scans: dict[Path, GeometryScan] = {}
        self.checked: set[Path] = set()
        self.errors = VtkErrorLog()
        self.reader: Any = None
        self.vtk_times: list[float] = []
        self.warnings: list[str] = list(case.warnings)
        self.times = case.times()

    # -- validation ---------------------------------------------------------------------
    def geometry(self, time: float | None) -> GeometryScan | None:
        """Validate (once) and return the scan of the geometry file used at ``time``."""
        path = self.case.geometry_file(time)
        if path in self.scans:
            return self.scans[path]
        if not path.is_file():
            raise ConnectorError(
                f"geometry file {path.name} referenced by {self.case.path.name} is missing",
                hint=f"copy every file of the export next to the case file (looked for {path})",
            )
        if path.stat().st_size == 0:
            raise ConnectorError(f"geometry file {path} is empty", hint="export again")
        if self.case.version != "gold":
            return None
        scan = scan_geometry(path)
        if not scan.parts:
            raise ConnectorError(
                f"geometry file {path} contains no parts", hint="export at least one part"
            )
        self.scans[path] = scan
        return scan

    def variable_parts(self, var: CaseVariable, time: float | None) -> list[int] | None:
        """Validate the file(s) of ``var`` at ``time``; part numbers present (Gold only)."""
        scan = self.geometry(time)
        present: list[int] | None = None
        for pattern in var.files:
            path = self.case.resolve(pattern, var.time_set, time)
            if not path.is_file():
                raise ConnectorError(
                    f"variable file {path.name} ({var.name}) referenced by "
                    f"{self.case.path.name} is missing",
                    hint="copy every file of the export next to the case file, or leave the "
                    f"field out with fields=['!{var.name}']",
                )
            if scan is None or var.location == "measured node":
                continue
            if path in self.checked:
                continue
            self.checked.add(path)
            parts = scan_variable(path, scan, var)
            present = parts if present is None else present
        return present

    # -- reader -------------------------------------------------------------------------
    def open(self, wanted: Sequence[CaseVariable] | None) -> None:
        """Create the VTK reader with only the ``wanted`` variables enabled (``None`` = all)."""
        from vtkmodules.vtkCommonCore import vtkCommand
        from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader

        first = self.times[0] if self.times else None
        self.geometry(first)
        reader = vtkGenericEnSightReader()
        reader.AddObserver(vtkCommand.ErrorEvent, self.errors)
        reader.SetCaseFileName(str(self.case.path))
        reader.ReadAllVariablesOff()
        with quiet_vtk():
            reader.UpdateInformation()
        inner = reader.GetReader()
        if inner is None:
            raise ConnectorError(
                f"VTK cannot open EnSight case {self.case.path}"
                + (f": {self.errors.messages[0]}" if self.errors.messages else ""),
                hint="check the FORMAT section and that the geometry file is readable",
            )
        inner.AddObserver(vtkCommand.ErrorEvent, self.errors)
        selections = (
            ("node", reader.GetPointDataArraySelection()),
            ("element", reader.GetCellDataArraySelection()),
        )
        for location, sel in selections:
            enable = None if wanted is None else {v.name for v in wanted if v.location == location}
            for i in range(sel.GetNumberOfArrays()):
                name = sel.GetArrayName(i)
                if enable is None or name in enable or re.sub(r"_[ri]$", "", name) in enable:
                    sel.EnableArray(name)
                else:
                    sel.DisableArray(name)
        self.reader = reader
        self.vtk_times = _vtk.reader_times(reader)
        self.errors.messages.clear()
        if self.vtk_times and (
            len(self.vtk_times) != len(self.times)
            or not np.allclose(self.vtk_times, self.times, rtol=1e-6, atol=1e-12)
        ):
            self.warnings.append(
                f"the case file lists {len(self.times)} time values but the VTK reader "
                f"reports {len(self.vtk_times)}; the reader's values are used"
            )
            self.times = list(self.vtk_times)

    def vtk_names(self) -> set[str]:
        out = set()
        for sel in (
            self.reader.GetPointDataArraySelection(),
            self.reader.GetCellDataArraySelection(),
        ):
            out |= {sel.GetArrayName(i) for i in range(sel.GetNumberOfArrays())}
        return out

    def load(self, index: int) -> list[tuple[str, Any]]:
        """``(zone, dataset)`` leaves of time step ``index`` (one step in memory)."""
        t = self.vtk_times[index] if self.vtk_times else None
        self.errors.messages.clear()
        with quiet_vtk():
            obj = _vtk.update_to_time(self.reader, t)
        leaves = list(_vtk.iter_leaves(obj))
        if self.errors.messages or not leaves:
            detail = self.errors.messages[0] if self.errors.messages else "no parts were read"
            raise ConnectorError(
                f"VTK failed to read {self.case.path.name}"
                + (f" at step {index}" if self.vtk_times else "")
                + f": {detail}",
                hint="a geometry or variable file is damaged or uses an EnSight feature VTK "
                "does not support; re-export as EnSight Gold (C binary)",
            )
        names = unique_names([n for n, _ in leaves])
        return list(zip(names, (ds for _, ds in leaves), strict=True))


def _zone_rows(leaves: Sequence[tuple[str, Any]]) -> dict[str, tuple[int, str, bool]]:
    """``zone -> (dim, role, default)``: the parts of highest dimension are the domain and are
    read by default; lower-dimensional parts (boundaries, node sets) are opt-in."""
    dims = {name: _dataset_dim(ds) for name, ds in leaves}
    top = max(dims.values(), default=0)
    rows = {}
    for name, ds in leaves:
        d = dims[name]
        kind = _vtk.dataset_kind(ds)
        if d == top:
            role = {"mesh": "volume", "grid": "grid", "point_cloud": "probes"}.get(kind, "part")
        else:
            role = "boundary" if d >= 1 else "probes"
        rows[name] = (d, role, d == top)
    return rows


def _output_name(var: CaseVariable, collisions: set[str]) -> str:
    return f"{var.name}__cell" if var.location == "element" and var.name in collisions else var.name


def _component_names(case: CaseFile, var: CaseVariable) -> list[str] | None:
    if var.complex:
        return ["re", "im"] if var.type == "scalar" else None
    # the EnSight 6 readers were not checked for their tensor component order
    return _COMPONENT_NAMES.get(var.type) if case.version == "gold" else None


def _collisions(case: CaseFile) -> set[str]:
    node = {v.name for v in case.variables if v.location == "node"}
    return {v.name for v in case.variables if v.location == "element" and v.name in node}


def _rename_cell_arrays(ds: Any, collisions: set[str]) -> None:
    """Give element variables that share their name with a node variable a ``__cell`` suffix in
    every part (``dataset_to_sample`` would only do so in parts that carry both)."""
    cd = ds.GetCellData()
    for name in collisions:
        arr = cd.GetArray(name)
        if arr is not None:
            arr.SetName(f"{name}__cell")


def resolve_case(path: Path) -> Path:
    """The case file behind ``path`` (a ``.case``/``.encas`` file, or a directory holding
    exactly one)."""
    path = Path(path)
    if path.is_dir():
        cases = find_cases(path)
        if len(cases) == 1:
            return cases[0]
        if not cases:
            raise UnsupportedSourceError(
                f"{path} contains no EnSight case file", hint="pass the .case file"
            )
        raise UnsupportedSourceError(
            f"{path} contains {len(cases)} EnSight case files",
            hint="pass one of: " + ", ".join(c.name for c in cases[:10]),
        )
    if not path.is_file():
        raise UnsupportedSourceError(f"{path} does not exist", hint="pass the .case file")
    return path


def case_version(path: Path) -> str | None:
    """``"gold"`` / ``"6"`` when ``path`` starts like an EnSight case file, else ``None``."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if b"\x00" in head:
        return None
    text = head.decode("latin-1")
    m = _TYPE_LINE.search(text)
    if m is None or "FORMAT" not in text[: m.start()]:
        return None
    return "gold" if m.group(1) else "6"


def find_cases(directory: Path, limit: int = 2000) -> list[Path]:
    """EnSight case files directly inside ``directory`` (sorted by name)."""
    out = []
    try:
        entries = itertools.islice(directory.iterdir(), limit)
        for p in entries:
            if match_extension(p, CASE_EXTENSIONS) and p.is_file() and case_version(p):
                out.append(p)
    except OSError:
        return []
    return sorted(out)


class EnSightConnector(BaseConnector):
    id = "ensight"
    name = "EnSight Gold / EnSight 6 case"
    formats = ["ensight-gold", "ensight6"]
    vendor = "Ansys/CEI (openly documented format)"
    license = "Apache-2.0 (reader: VTK, BSD-3)"
    extensions = CASE_EXTENSIONS
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["surface", "mesh", "grid", "point_cloud"]
    export_help = (
        "Export 'EnSight Gold' from the solver (STAR-CCM+: File > Export or the shipped macro; "
        "Fluent: File > Export > EnSight Case Gold; CONVERGE: post_convert; OpenFOAM: "
        "foamToEnsight) and point at the .case file."
    )
    status_note = (
        "Reads ASCII, C binary and Fortran binary files: unstructured and structured parts, "
        "several time sets, moving geometry. Parts of the highest dimension are read by default, "
        "boundary parts with zones=[...]. Server-of-servers (.sos) files are not read."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            if path.is_dir():
                cases = find_cases(path)
                if not cases:
                    return None
                conf = 0.5 if len(cases) == 1 else 0.3
                notes = [f"directory with EnSight case file(s): {cases[0].name}"]
                return Detection(self.id, path, conf, "ensight-case-dir", notes)
            ext = match_extension(path, self.extensions)
            if ext is None or not path.is_file():
                return None
            version = case_version(path)
        except OSError:
            return None
        if version is None:
            return None
        fmt = "ensight-gold" if version == "gold" else "ensight6"
        conf = 0.97 if version == "gold" else 0.9
        return Detection(
            self.id, path, conf, fmt, [f"extension {ext}", "FORMAT section 'type: ensight'"]
        )

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        case_path = resolve_case(Path(path))
        case = parse_case(case_path)
        session = _Session(case)
        first = session.times[0] if session.times else None
        scan = session.geometry(first)
        session.open([])  # geometry only: describe never loads field arrays
        leaves = session.load(0)
        texts = list(case.comments) + (scan.description if scan is not None else [])
        solver, version, domain = sniff_solver(texts)
        desc = SourceDescription(
            connector_id=self.id,
            path=Path(path),
            format="ensight-gold" if case.version == "gold" else "ensight6",
            solver=solver,
            solver_version=version,
            domain=domain,
            time=TimeInfo(values=list(session.times) if len(session.times) > 1 else []),
            parameters=dict(case.constants(0)),
        )
        rows = _zone_rows(leaves)
        for name, ds in leaves:
            dim, role, default = rows[name]
            z = _vtk.zone_info(name, ds, role=role, default=default)
            z.name, z.dim = name, dim
            desc.zones.append(z)
        desc.metadata["case_file"] = str(case_path)
        desc.metadata["geometry_file"] = case.model
        desc.metadata["moving_geometry"] = case.moving
        if scan is not None:
            desc.metadata["encoding"] = scan.mode
            desc.metadata["geometry_description"] = scan.description
        if len(session.times) == 1:
            desc.metadata["time_value"] = session.times[0]
        self._describe_fields(desc, session, scan, leaves, first)
        desc.warnings.extend(session.warnings)
        return desc

    def _describe_fields(
        self,
        desc: SourceDescription,
        session: _Session,
        scan: GeometryScan | None,
        leaves: Sequence[tuple[str, Any]],
        first: float | None,
    ) -> None:
        case = session.case
        collisions = _collisions(case)
        # part number -> zone name, when the scan and the reader agree on the part list
        by_number: dict[int, str] = {}
        if scan is not None:
            scanned = [p for p in scan.parts if p.n_points > 0]
            if len(scanned) == len(leaves) and all(
                p.n_points == ds.GetNumberOfPoints()
                for p, (_, ds) in zip(scanned, leaves, strict=True)
            ):
                by_number = {p.number: n for p, (n, _) in zip(scanned, leaves, strict=True)}
            else:
                desc.warnings.append(
                    "the part list read by VTK differs from the geometry file scan; per-zone "
                    "field availability is not reported"
                )
        vtk_names = session.vtk_names()
        for var in case.variables:
            if var.location in ("case", "measured node"):
                continue
            zones: list[str] | None = None
            try:
                parts = session.variable_parts(var, first)
            except ConnectorError as e:
                desc.warnings.append(f"field {var.name!r} cannot be read: {e.message}")
                continue
            if parts and by_number and all(p in by_number for p in parts):
                zones = [by_number[p] for p in parts]
                if len(zones) == len(leaves):
                    zones = None
            if var.name not in vtk_names:
                desc.warnings.append(
                    f"variable {var.name!r} of the case file is not offered by the VTK reader"
                )
                continue
            names = [var.name]
            if var.complex and var.type != "scalar":
                names = [f"{var.name}_r", f"{var.name}_i"]
            for name in names:
                out = f"{name}__cell" if var.location == "element" and name in collisions else name
                desc.fields.append(
                    FieldInfo(
                        name=out,
                        location="point" if var.location == "node" else "cell",
                        components=2 if var.complex and var.type == "scalar" else var.components,
                        component_names=_component_names(case, var),
                        description=("complex " if var.complex else "") + var.type,
                        time_dependent=var.time_dependent,
                        zones=zones,
                    )
                )

    def read(
        self,
        path: Path,
        *,
        zones: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        times: TimeSelection = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[RawSample]:
        case_path = resolve_case(Path(path))
        case = parse_case(case_path)
        session = _Session(case)
        collisions = _collisions(case)
        wanted = [
            v
            for v in case.variables
            if v.location in ("node", "element")
            and (
                wants(_output_name(v, collisions), fields)
                or (v.complex and (wants(f"{v.name}_r", fields) or wants(f"{v.name}_i", fields)))
            )
        ]
        session.open(wanted)
        transient = len(session.times) > 1
        steps = select_time_indices(session.times if transient else [], times)
        first = session.times[0] if session.times else None
        scan = session.geometry(first)
        texts = list(case.comments) + (scan.description if scan is not None else [])
        solver, version, _ = sniff_solver(texts)
        base = meshutil.default_sample_id(case_path)
        tensor_names = {
            _output_name(v, collisions): _COMPONENT_NAMES[v.type]
            for v in case.variables
            if v.type in _COMPONENT_NAMES and not v.complex and case.version == "gold"
        }
        # options geometry_every_step: repeat a static mesh in every step (needed when the
        # steps are later stored as separate samples, convert --time-as-samples)
        every_step = bool((options or {}).get("geometry_every_step", False))
        selected: list[str] | None = None
        ids: dict[str, str] = {}
        rows: dict[str, tuple[int, str, bool]] = {}
        emitted: set[str] = set()
        for ti in steps:
            t = session.times[ti] if session.times else None
            session.geometry(t)
            for var in wanted:
                session.variable_parts(var, t)
            leaves = session.load(ti)
            if selected is None:
                rows = _zone_rows(leaves)
                names = [n for n, _ in leaves]
                defaults = [n for n in names if rows[n][2]]
                selected = _vtk.select_zones(names, zones, defaults)
                ids = unique_sample_ids(base, selected)
            for name, ds in leaves:
                if name not in ids:
                    continue
                _rename_cell_arrays(ds, collisions)
                meta: dict[str, Any] = {
                    "solver": solver,
                    "source": str(case_path),
                    "format": "ensight-gold" if case.version == "gold" else "ensight6",
                    "role": rows.get(name, (0, "part", True))[1],
                }
                if version:
                    meta["solver_version"] = version
                constants = case.constants(ti)
                if constants:
                    meta["parameters"] = constants
                if transient:
                    meta["time_units"] = "s"
                sid = ids[name]
                sample = _vtk.dataset_to_sample(
                    ds,
                    sid,
                    zone=name,
                    fields=fields,
                    time=float(session.times[ti]) if transient else None,
                    time_index=ti if transient else None,
                    geometry=every_step or case.moving or sid not in emitted,
                    metadata=meta,
                )
                emitted.add(sid)
                for fname, comp_names in tensor_names.items():
                    f = sample.fields.get(fname)
                    if f is not None and f.components == len(comp_names):
                        f.component_names = list(comp_names)
                yield sample


# -- writer (tests, round trips) ---------------------------------------------------------------


class _GoldWriter:
    def __init__(self, path: Path, mode: str, *, geometry: bool, order: str = "<") -> None:
        if mode not in ("ascii", "c", "fortran"):
            raise ValueError(f"unknown EnSight encoding {mode!r}")
        self.mode = mode
        self.order = order
        self.f = open(path, "wb")  # noqa: SIM115 - closed by close()
        if geometry and mode == "c":
            self.f.write(b"C Binary".ljust(80))
        elif geometry and mode == "fortran":
            self._record(b"Fortran Binary".ljust(80))

    def _record(self, data: bytes) -> None:
        n = struct.pack(self.order + "i", len(data))
        self.f.write(n + data + n)

    def string(self, text: str) -> None:
        if self.mode == "ascii":
            self.f.write(text.encode() + b"\n")
        elif self.mode == "c":
            self.f.write(text.encode().ljust(80)[:80])
        else:
            self._record(text.encode().ljust(80)[:80])

    def integer(self, value: int) -> None:
        if self.mode == "ascii":
            self.f.write(b"%10d\n" % value)
        elif self.mode == "c":
            self.f.write(struct.pack(self.order + "i", value))
        else:
            self._record(struct.pack(self.order + "i", value))

    def integers(self, rows: np.ndarray) -> None:
        a = np.asarray(rows, dtype=self.order + "i4")
        if self.mode == "ascii":
            for row in a.reshape(len(a), -1):
                self.f.write(("".join(f"{int(v):10d}" for v in row) + "\n").encode())
        elif self.mode == "c":
            self.f.write(a.tobytes())
        else:
            self._record(a.tobytes())

    def floats(self, values: np.ndarray) -> None:
        a = np.asarray(values, dtype=self.order + "f4").ravel()
        if self.mode == "ascii":
            self.f.write("".join(f"{float(v):12.5e}\n" for v in a).encode())
        elif self.mode == "c":
            self.f.write(a.tobytes())
        else:
            self._record(a.tobytes())

    def close(self) -> None:
        self.f.close()


def _write_geometry(
    path: Path, parts: Sequence[Mapping[str, Any]], mode: str, order: str = "<"
) -> None:
    w = _GoldWriter(path, mode, geometry=True, order=order)
    try:
        w.string("EnSight Gold geometry written by frontier-connectors")
        w.string("test fixture")
        w.string("node id assign")
        w.string("element id assign")
        for number, part in enumerate(parts, 1):
            w.string("part")
            w.integer(number)
            w.string(str(part["name"]))
            w.string("coordinates")
            pts = np.asarray(part["points"], dtype=np.float64)
            w.integer(len(pts))
            for d in range(3):
                w.floats(pts[:, d])
            for etype, conn in part["elements"]:
                w.string(etype)
                w.integer(len(conn))
                w.integers(np.asarray(conn) + 1)
    finally:
        w.close()


def _write_variable(
    path: Path,
    parts: Sequence[Mapping[str, Any]],
    name: str,
    location: str,
    mode: str,
    order: str = "<",
) -> None:
    w = _GoldWriter(path, mode, geometry=False, order=order)
    try:
        w.string(f"{name} ({location})")
        for number, part in enumerate(parts, 1):
            values = part.get(location, {}).get(name)
            if values is None:
                continue  # partial variable: this part does not carry it
            v = np.asarray(values, dtype=np.float64)
            v = v.reshape(len(v), -1)
            w.string("part")
            w.integer(number)
            if location == "node":
                w.string("coordinates")
                for c in range(v.shape[1]):
                    w.floats(v[:, c])
                continue
            start = 0
            for etype, conn in part["elements"]:
                w.string(etype)
                for c in range(v.shape[1]):
                    w.floats(v[start : start + len(conn), c])
                start += len(conn)
    finally:
        w.close()


_TYPE_BY_COMPONENTS = {1: "scalar", 3: "vector", 6: "tensor symm", 9: "tensor asym"}


def write_ensight_gold(
    case_path: Path,
    steps: Sequence[Sequence[Mapping[str, Any]]],
    *,
    times: Sequence[float] | None = None,
    encoding: str = "ascii",
    moving: bool = False,
    constants: Mapping[str, float] | None = None,
    comment: str = "EnSight Gold case written by frontier-connectors",
    byte_order: str = "<",
) -> Path:
    """Write a small EnSight Gold case (used by tests; not a general exporter).

    ``steps[i]`` is the list of parts at time step ``i``; a part is ``{"name": str, "points":
    (N, 3), "elements": [(ensight_type, 0-based connectivity (K, P)), ...], "node": {name:
    (N,) | (N, C)}, "element": {name: (K_total,) | (K_total, C)}}`` with element values in the
    order of ``elements``. ``times=None`` writes a steady case; ``moving=True`` writes one
    geometry file per step. ``encoding`` is ``ascii`` | ``c`` | ``fortran``; ``byte_order``
    (``<`` | ``>``) applies to the binary encodings.
    """
    case_path = Path(case_path)
    stem = case_path.stem
    transient = times is not None
    if transient and len(times or []) != len(steps):
        raise ValueError("one time value per step is required")
    variables: dict[tuple[str, str], int] = {}
    for part in steps[0]:
        for location in ("node", "element"):
            for name, values in part.get(location, {}).items():
                v = np.asarray(values)
                variables[(name, location)] = 1 if v.ndim == 1 else int(v.shape[1])
    star = ".****" if transient else ""
    for i, parts in enumerate(steps):
        number = f".{i:04d}" if transient else ""
        if i == 0 or moving:
            geo = case_path.parent / f"{stem}.geo{number if moving else ''}"
            _write_geometry(geo, parts, encoding, byte_order)
        for name, location in variables:
            safe = meshutil.sanitize_id(name)
            file = case_path.parent / f"{stem}.{safe}_{location[0]}{number}"
            _write_variable(file, parts, name, location, encoding, byte_order)
    lines = [f"# {comment}", "", "FORMAT", "type: ensight gold", "", "GEOMETRY"]
    lines.append(f"model: 1 {stem}.geo.****" if moving and transient else f"model: {stem}.geo")
    lines += ["", "VARIABLE"]
    for name, value in (constants or {}).items():
        lines.append(f"constant per case: {name} {value:g}")
    for (name, location), nc in variables.items():
        safe = meshutil.sanitize_id(name)
        ts = "1 " if transient else ""
        lines.append(
            f"{_TYPE_BY_COMPONENTS[nc]} per {location}: {ts}{name} "
            f"{stem}.{safe}_{location[0]}{star}"
        )
    if transient:
        values = [f"{float(t):g}" for t in (times or [])]
        lines += ["", "TIME", "time set: 1", f"number of steps: {len(values)}"]
        lines += ["filename start number: 0", "filename increment: 1"]
        # two lines on purpose: readers must follow 'time values:' across lines
        half = max(len(values) // 2, 1)
        lines.append("time values: " + " ".join(values[:half]))
        if values[half:]:
            lines.append(" ".join(values[half:]))
    case_path.write_text("\n".join(lines) + "\n")
    return case_path
