# SPDX-License-Identifier: Apache-2.0
"""OpenFOAM cases (Foundation openfoam.org, ESI openfoam.com, foam-extend).

Two layers:

* a self-contained **FoamFile parser** (:func:`parse_foam_file`, :func:`read_points`,
  :func:`read_faces`, :func:`read_labels`, :func:`read_boundary`, :func:`read_field`): header,
  nested dictionaries, ``#include``/``$var``, ``N(...)`` and ``N{v}`` lists, binary lists sized
  by ``arch``, ``faceCompactList`` and gzip. It serves ``describe`` (counts come from the list
  headers, boundary conditions from ``boundaryField``, units from ``dimensions``), validation
  and error diagnosis, and never needs an OpenFOAM installation;
* VTK's ``vtkOpenFOAMReader`` for the heavy lifting of ``read`` (cell shape recognition,
  polyhedra, patch values), driven one time step at a time.

Zones: ``internalMesh`` (volume, the only default zone) and ``patch/<name>`` for every boundary
patch with faces (surfaces, opt-in with ``zones=["patch/*"]``; ``empty`` patches of 2-D cases
are listed but are never a default). Volume fields are emitted at their native location
(``cell``); ``options={"point_data": True}`` adds VTK's cell-to-point interpolation as
``<name>__point``.

Options: ``decomposed`` (bool, default: automatic), ``point_data`` (bool), ``iterations``
(``"last"`` | ``"all"``: what the time axis of a *steady* solver case is; default ``"last"``,
i.e. the converged iteration is read as one steady sample), ``region`` (name of the mesh region
of a multi-region case).
"""

from __future__ import annotations

import gzip
import math
import mmap
import os
import re
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import _vtk, meshutil
from frontier_connectors.errors import ConnectorError, SelectionError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    ZoneInfo,
    as_field,
)
from frontier_connectors.paths import ReferenceEscapeError, is_inside, resolve_in_case
from frontier_connectors.select import select_time_indices, wants
from frontier_connectors.units import foam_dimensions_to_unit

INTERNAL_ZONE = "internalMesh"
PATCH_PREFIX = "patch/"

#: components of the primitive types that appear in ``List<...>`` and field classes
PRIMITIVE_COMPONENTS: dict[str, int] = {
    "scalar": 1,
    "label": 1,
    "vector": 3,
    "sphericalTensor": 1,
    "symmTensor": 6,
    "tensor": 9,
}
COMPONENT_NAMES: dict[int, list[str]] = {
    3: ["x", "y", "z"],
    6: ["xx", "xy", "xz", "yy", "yz", "zz"],
    9: ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"],
}
#: VTK stores symmetric tensors as XX YY ZZ XY YZ XZ; this picks them back in OpenFOAM order
#: (the permutation is its own inverse)
_SYMM_VTK_TO_FOAM = [0, 3, 5, 1, 4, 2]

_FIELD_CLASS = re.compile(
    r"^(vol|point)(Scalar|Vector|SphericalTensor|SymmTensor|Tensor)Field(::Internal)?$"
)
_STEADY_APPLICATIONS = {
    "simpleFoam",
    "rhoSimpleFoam",
    "buoyantSimpleFoam",
    "buoyantBoussinesqSimpleFoam",
    "SRFSimpleFoam",
    "porousSimpleFoam",
    "rhoPorousSimpleFoam",
    "simpleReactingParcelFoam",
    "potentialFoam",
    "steadyCompressibleFoam",
    "MRFSimpleFoam",
}
_MESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
_PROCESSOR_PATCH_TYPES = ("processor", "processorCyclic")

_MMAP_MIN = 1 << 20
_SCAN_CHUNK = 1 << 24
#: sized lists at least this long take the vectorised numeric path
_FAST_LIST = 64
#: compressed field files above this size are not decompressed just to list boundary conditions
_BC_GZ_LIMIT = 256 << 20

_TIME_NAME = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_NUMBER = _TIME_NAME
_INTEGER = re.compile(r"^[-+]?\d+$")
_PROCESSOR_DIR = re.compile(r"^processor(\d+)$")
_COLLATED_DIR = re.compile(r"^processors\d+(_\d+-\d+)?$")
_LIST_TYPE = re.compile(r"^(?:List|Field|UList)<(\w+)>$")
_NOTE_COUNTS = re.compile(
    r"nPoints:\s*(\d+)\s+nCells:\s*(\d+)\s+nFaces:\s*(\d+)\s+nInternalFaces:\s*(\d+)"
)
_BANNER_VERSION = re.compile(rb"Version:\s*([^\s|]+)")
_DIMENSIONS = re.compile(rb"dimensions\s*\[([^\]]*)\]")

_SKIP = re.compile(rb"(?:\s+|//[^\n]*|/\*.*?\*/)+", re.S)
_TOKEN = re.compile(rb'"(?:\\.|[^"\\])*"|#\{.*?#\}|[{};\[\]()]|[^\s{};\[\]()"]+', re.S)
_WORD_TAIL = re.compile(rb'[^\s{};\[\]()"]+')
_PUNCTUATION = b'"#{};[]()'
_PARENS_TO_SPACE = bytes.maketrans(b"()", b"  ")

_TRUNCATED_HINT = (
    "the file is truncated or corrupt: copy it again from the solver machine, or rewrite the "
    "case with 'foamFormatConvert' / 'reconstructPar'"
)


# -- parsed values ----------------------------------------------------------------------------


class FoamDict(dict[str, Any]):
    """A parsed dictionary. ``patterns`` lists the keys that were quoted (regular expressions)."""

    def __init__(self) -> None:
        super().__init__()
        self.patterns: list[str] = []


class FoamDims(list[Any]):
    """A ``[kg m s K mol A cd]`` dimension set."""

    @property
    def unit(self) -> str:
        try:
            return foam_dimensions_to_unit([float(v) for v in self])
        except (TypeError, ValueError):
            return ""


@dataclass
class FoamUniform:
    """``N{value}``: a list of ``n`` identical entries."""

    n: int
    value: Any


@dataclass
class FoamSkipped:
    """A long list that was stepped over instead of loaded (``load=False``)."""

    n: int
    components: int = 0


@dataclass
class FoamFile:
    """Header and body of one parsed file."""

    path: Path
    header: FoamDict
    body: FoamDict
    version: str | None = None
    distribution: str | None = None
    #: bytes after the header when only the head of the file was read (:func:`read_header`)
    rest: bytes = b""

    @property
    def cls(self) -> str:
        return str(self.header.get("class", ""))


@dataclass
class FoamField:
    """A ``vol*Field``/``point*Field`` file."""

    name: str
    cls: str
    location: str
    components: int
    dimensions: FoamDims | None
    #: ``np.ndarray (N[, C])`` | ``("uniform", value)`` | :class:`FoamSkipped`
    internal: Any
    boundary: FoamDict

    @property
    def units(self) -> str:
        return self.dimensions.unit if self.dimensions is not None else ""

    def values(self, n_cells: int) -> np.ndarray:
        """Internal field as ``(n_cells, components)`` float64, expanding ``uniform``."""
        if isinstance(self.internal, np.ndarray):
            return np.asarray(self.internal, dtype=np.float64).reshape(-1, self.components)
        if isinstance(self.internal, tuple) and self.internal[0] == "uniform":
            v = np.asarray(self.internal[1], dtype=np.float64).reshape(1, -1)
            return np.repeat(v, n_cells, axis=0)
        raise ConnectorError(
            f"field {self.name!r}: internalField was not loaded", hint="call read_field(load=True)"
        )


@dataclass
class FoamPatch:
    name: str
    type: str
    n_faces: int
    start_face: int
    groups: list[str] = field(default_factory=list)
    physical_type: str | None = None


# -- low-level file access --------------------------------------------------------------------


def resolve_file(path: Path) -> Path | None:
    """``path`` itself or its ``.gz`` sibling (OpenFOAM ``writeCompression on``)."""
    if path.is_file():
        return path
    gz = path.with_name(path.name + ".gz")
    return gz if gz.is_file() else None


def _read_error(path: Path, exc: Exception) -> ConnectorError:
    return ConnectorError(f"cannot read {path}: {exc}", hint=_TRUNCATED_HINT)


def _load(path: Path) -> Any:
    """Whole file as a bytes-like object (gzip is inflated; large plain files are mapped)."""
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return f.read()
        if path.stat().st_size < _MMAP_MIN:
            return path.read_bytes()
        with open(path, "rb") as f:
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, EOFError, zlib.error) as e:
        raise _read_error(path, e) from e


def _head(path: Path, size: int = 1 << 15) -> bytes:
    """First ``size`` bytes (of the inflated stream for ``.gz``)."""
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return f.read(size)
        with open(path, "rb") as f:
            return f.read(size)
    except (OSError, EOFError, zlib.error) as e:
        raise _read_error(path, e) from e


def _flag(value: Any) -> bool:
    """Option value as a bool (the CLI passes ``--option key=value`` strings)."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _number(text: str) -> Any:
    if _INTEGER.match(text):
        return int(text)
    if _NUMBER.match(text):
        return float(text)
    return text


def _parse_numbers(raw: Any, integer: bool) -> np.ndarray | None:
    """Whitespace/parenthesis separated numbers -> flat array; ``None`` when not numeric."""
    text = bytes(raw).translate(_PARENS_TO_SPACE)
    dtype = np.int64 if integer else np.float64
    try:
        return np.fromstring(text.decode("ascii"), dtype=dtype, sep=" ")
    except (ValueError, UnicodeDecodeError):
        if integer:
            return None
    try:  # 'nan'/'inf' spellings and other float forms numpy's fast path rejects
        return np.array(text.split(), dtype=np.float64)
    except ValueError:
        return None


# -- #include confinement ---------------------------------------------------------------------

_INCLUDE = re.compile(rb'#(?:include|includeIfPresent|sinclude)\s+"([^"\r\n]*)"')
#: files up to this size are searched completely for #include directives
INCLUDE_SCAN_FULL_BYTES = 256 << 20
#: for larger files (big binary lists) only this much of the start and of the end is searched
INCLUDE_SCAN_EDGE_BYTES = 8 << 20
INCLUDE_SCAN_DEPTH = 8
_FOAM_CASE_VARS = ("${FOAM_CASE}", "$FOAM_CASE", "<case>")
_ENV_VAR = re.compile(r"\$(?:\{([^}]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def foam_case_root(path: Path) -> Path:
    """Case root for a file of a case: the nearest directory above it that holds
    ``system/controlDict`` (the case directory), else the file's own directory."""
    here = Path(path).resolve().parent
    for cand in (here, *list(here.parents)[:8]):
        if (cand / "system" / "controlDict").is_file():
            return cand
    return here


def include_target(root: Path, holder: Path, rel: str) -> Path | None:
    """File an ``#include "rel"`` in ``holder`` names, confined to the case directory.

    Plain relative names go through the shared resolver. Names with ``$VAR`` / ``~`` are
    expanded the way OpenFOAM and VTK's reader expand them (an unset variable is empty):
    ``$FOAM_CASE`` is the case, a target inside the user's own ``$WM_PROJECT_DIR`` (the
    OpenFOAM installation, taken from the environment, never from the case) is allowed, a
    target that does not exist is skipped (the include cannot be read by anyone), and any
    other existing target outside the case is refused."""
    if "$" not in rel and not rel.startswith(("~", "<")):
        return resolve_in_case(root, rel, what="OpenFOAM #include", referrer=holder)
    text = rel
    for var in _FOAM_CASE_VARS:
        text = text.replace(var, str(root))
    text = _ENV_VAR.sub(lambda m: os.environ.get(m.group(1) or m.group(2) or "", ""), text)
    cand = Path(text).expanduser() if text.startswith("~") else Path(text)
    if not cand.is_absolute():
        cand = holder.parent / cand
    try:
        target = cand.resolve()
    except (OSError, RuntimeError):
        return None
    if target.is_relative_to(root.resolve()):
        return target
    install = os.environ.get("WM_PROJECT_DIR")
    if install and is_inside(Path(install), target):
        return target
    if not os.path.lexists(target):
        return None
    raise ReferenceEscapeError(
        case_file=holder,
        reference=rel,
        case_root=root,
        what="OpenFOAM #include",
        reason="expands to an existing file outside the case directory",
    )


def _include_names(path: Path) -> list[str]:
    """``#include`` targets written in ``path`` (see ``INCLUDE_SCAN_*`` for large files)."""
    found: list[bytes] = []
    try:
        if path.name.endswith(".gz"):
            carry = b""
            with gzip.open(path, "rb") as f:
                while chunk := f.read(1 << 24):
                    buf = carry + chunk
                    found += _INCLUDE.findall(buf)
                    carry = buf[-4096:]
        else:
            size = path.stat().st_size
            if size == 0:
                return []
            with open(path, "rb") as f:
                if size <= INCLUDE_SCAN_FULL_BYTES:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        found = _INCLUDE.findall(mm)
                else:
                    found = _INCLUDE.findall(f.read(INCLUDE_SCAN_EDGE_BYTES))
                    f.seek(size - INCLUDE_SCAN_EDGE_BYTES)
                    found += _INCLUDE.findall(f.read(INCLUDE_SCAN_EDGE_BYTES))
    except (OSError, EOFError, zlib.error, ValueError):
        return []
    return list(dict.fromkeys(b.decode("utf-8", "replace") for b in found))


def confine_includes(root: Path, files: Sequence[Path], seen: set[Path], depth: int = 0) -> None:
    """Check every ``#include`` in ``files`` (and in what they include) against the case
    root, before a reader that follows includes on its own (VTK) opens them."""
    for file in files:
        if file in seen or not file.is_file():
            continue
        seen.add(file)
        for rel in _include_names(file):
            target = include_target(root, file, rel)
            if target is not None and depth < INCLUDE_SCAN_DEPTH and is_inside(root, target):
                confine_includes(root, [target], seen, depth + 1)


def _files_below(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        return sorted(p for p in directory.rglob("*") if p.is_file())
    except OSError:
        return []


class _Parser:
    """Tokenizer and recursive-descent parser over one file's bytes."""

    def __init__(self, buf: Any, path: Path, *, load: bool = True, depth: int = 0) -> None:
        self.buf = buf
        self.n = len(buf)
        self.pos = 0
        self.path = path
        self.load = load
        self.depth = depth
        self.binary = False
        self.label = np.dtype("<i4")
        self.scalar = np.dtype("<f8")
        #: element type announced by the last ``List<type>`` token
        self.hint: str | None = None
        self.scopes: list[FoamDict] = []

    def error(self, message: str, *, hint: str | None = None) -> ConnectorError:
        return ConnectorError(
            f"{self.path}: {message} (near byte {self.pos})", hint=hint or _TRUNCATED_HINT
        )

    # -- tokens -------------------------------------------------------------------------
    def skip(self) -> None:
        m = _SKIP.match(self.buf, self.pos)
        if m:
            self.pos = m.end()

    def peek_byte(self) -> bytes:
        self.skip()
        return bytes(self.buf[self.pos : self.pos + 1])

    def next(self) -> bytes | None:
        self.skip()
        if self.pos >= self.n:
            return None
        m = _TOKEN.match(self.buf, self.pos)
        if m is None:
            raise self.error("unterminated string")
        tok = bytes(m.group())
        self.pos = m.end()
        if tok[:1] not in _PUNCTUATION and not tok.isdigit():
            # OpenFOAM words keep balanced parentheses: div(phi,U), interpolate(grad(p))
            while self.buf[self.pos : self.pos + 1] == b"(":
                try:
                    end = self.list_end(self.pos + 1)
                except ConnectorError:
                    break
                tail = _WORD_TAIL.match(self.buf, end + 1)
                stop = tail.end() if tail else end + 1
                tok += bytes(self.buf[self.pos : stop])
                self.pos = stop
        return tok

    def list_end(self, start: int) -> int:
        """Offset of the ``)`` closing the list whose first content byte is ``start``."""
        buf = self.buf
        first = buf.find(b")", start)
        if first < 0:
            self.pos = self.n
            raise self.error("list is not closed by ')': unexpected end of file")
        if buf.find(b"(", start, first) < 0:
            return first
        depth, pos = 1, start
        while pos < self.n:
            chunk = np.frombuffer(
                buf, dtype=np.uint8, count=min(_SCAN_CHUNK, self.n - pos), offset=pos
            )
            delta = (chunk == 40).astype(np.int8) - (chunk == 41).astype(np.int8)
            level = np.cumsum(delta, dtype=np.int64) + depth
            closed = np.flatnonzero(level == 0)
            if closed.size:
                return pos + int(closed[0])
            depth = int(level[-1])
            pos += len(chunk)
        self.pos = self.n
        raise self.error("list is not closed by ')': unexpected end of file")

    @staticmethod
    def text(tok: bytes) -> str:
        s = tok.decode("utf-8", "replace")
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1].replace('\\"', '"')
        return s

    # -- grammar ------------------------------------------------------------------------
    def header(self) -> FoamDict:
        start = self.pos
        if self.next() != b"FoamFile" or self.next() != b"{":
            self.pos = start
            return FoamDict()
        head = self.entries(b"}")
        self.binary = str(head.get("format", "ascii")).lower() == "binary"
        arch = str(head.get("arch", ""))
        order = ">" if "MSB" in arch else "<"
        m = re.search(r"label=(\d+)", arch)
        self.label = np.dtype(f"{order}i{int(m.group(1)) // 8 if m else 4}")
        m = re.search(r"scalar=(\d+)", arch)
        self.scalar = np.dtype(f"{order}f{int(m.group(1)) // 8 if m else 8}")
        return head

    def entries(self, end: bytes | None) -> FoamDict:
        d = FoamDict()
        self.scopes.append(d)
        try:
            while True:
                start = self.pos
                tok = self.next()
                if tok is None:
                    if end is not None:
                        raise self.error("dictionary is not closed by '}': unexpected end of file")
                    return d
                if tok == b"}":
                    if end is None:
                        raise self.error("unbalanced '}'")
                    return d
                if tok == b";":
                    continue
                if tok[:1] == b"#" and tok[:2] != b"#{":
                    self.directive(tok, d)
                    continue
                if tok == b"(" or tok.isdigit():
                    # anonymous top-level list (zone files, uniform/time, ...)
                    vals: list[Any] = []
                    self.value(tok, vals)
                    d[""] = vals[0] if len(vals) == 1 else tuple(vals)
                    continue
                if tok[:1] in b"{[])":
                    self.pos = start
                    raise self.error(f"unexpected {tok.decode()!r} where a keyword was expected")
                key = self.text(tok)
                if tok[:1] == b'"':
                    d.patterns.append(key)
                vals = []
                while True:
                    vstart = self.pos
                    t = self.next()
                    if t is None or t == b";":
                        break
                    if t == b"}":
                        self.pos = vstart
                        break
                    if t == b"{":
                        vals.append(self.entries(b"}"))
                        break
                    self.value(t, vals)
                new = None if not vals else vals[0] if len(vals) == 1 else tuple(vals)
                old = d.get(key)
                if isinstance(old, FoamDict) and isinstance(new, FoamDict):
                    old.update(new)  # OpenFOAM's default '#inputMode merge'
                    old.patterns.extend(new.patterns)
                else:
                    d[key] = new
        finally:
            self.scopes.pop()

    def directive(self, tok: bytes, d: FoamDict) -> None:
        name = tok.decode("ascii", "replace")
        if name in ("#include", "#includeIfPresent", "#sinclude", "#includeEtc"):
            target = self.next()
            if target is None or name == "#includeEtc" or target[:1] != b'"':
                return
            rel = self.text(target)
            # an include that leaves the case raises; it is not silently skipped
            inc = include_target(foam_case_root(self.path), self.path, rel)
            if inc is None or self.depth >= 8 or not inc.is_file():
                return
            try:
                sub = _Parser(_load(inc), inc, load=self.load, depth=self.depth + 1)
                sub.header()
                d.update(sub.entries(None))
            except ConnectorError:
                return  # an unreadable include must not hide the including file
            return
        # #inputMode, #remove, #calc, #includeFunc, #if ...: not evaluated, the line is skipped
        eol = self.buf.find(b"\n", self.pos)
        self.pos = self.n if eol < 0 else eol + 1

    def lookup(self, text: str) -> Any:
        name = text.lstrip("$").strip("{}").lstrip(":.")
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return text

    def value(self, t: bytes, out: list[Any]) -> None:
        c = t[:1]
        if c == b"(":
            out.append(self.items())
        elif c == b"[":
            out.append(self.dims())
        elif c == b'"' or t[:2] == b"#{":
            out.append(self.text(t))
        elif c in b")]}":
            raise self.error(f"unbalanced {t.decode()!r}")
        elif t.isdigit():
            save = self.pos
            nxt = self.peek_byte()
            if nxt == b"(":
                self.sized_list(int(t), out)
            elif nxt == b"{":
                self.pos += 1
                vals: list[Any] = []
                while True:
                    u = self.next()
                    if u is None:
                        raise self.error("uniform list is not closed by '}'")
                    if u == b"}":
                        break
                    self.value(u, vals)
                out.append(FoamUniform(int(t), vals[0] if len(vals) == 1 else tuple(vals)))
            else:
                self.pos = save
                out.append(int(t))
        elif c == b"$":
            out.append(self.lookup(self.text(t)))
        else:
            s = t.decode("utf-8", "replace")
            m = _LIST_TYPE.match(s)
            if m:
                self.hint = m.group(1)
            out.append(_number(s))

    def dims(self) -> FoamDims:
        out = FoamDims()
        while True:
            t = self.next()
            if t is None:
                raise self.error("dimension set is not closed by ']'")
            if t == b"]":
                return out
            out.append(_number(t.decode("utf-8", "replace")))

    def items(self) -> list[Any]:
        """Elements of a ``( ... )`` list whose opening parenthesis was consumed."""
        out: list[Any] = []
        while True:
            t = self.next()
            if t is None:
                raise self.error("list is not closed by ')': unexpected end of file")
            if t == b")":
                return out
            if t == b";":
                continue
            if t == b"{":
                out.append(self.entries(b"}"))
                continue
            if t[:1] not in _PUNCTUATION and not t.isdigit() and self.peek_byte() == b"{":
                self.pos += 1
                out.append((self.text(t), self.entries(b"}")))  # 'inlet { type patch; }'
                continue
            self.value(t, out)

    def sized_list(self, n: int, out: list[Any]) -> None:
        """``N(...)`` with ``self.pos`` at the opening parenthesis."""
        hint, self.hint = self.hint, None
        self.pos += 1
        start = self.pos
        comps = PRIMITIVE_COMPONENTS.get(hint or "", 0)
        if self.binary and comps and n > 0:
            dtype = self.label if hint == "label" else self.scalar
            nbytes = n * comps * dtype.itemsize
            if start + nbytes + 1 > self.n:
                self.pos = self.n
                raise self.error(
                    f"binary List<{hint}> announces {n} entries ({nbytes} bytes) but only "
                    f"{max(self.n - start, 0)} bytes follow: the file is truncated"
                )
            if self.buf[start + nbytes : start + nbytes + 1] != b")":
                raise self.error(
                    f"binary List<{hint}> of {n} entries is not closed by ')' after {nbytes} bytes",
                    hint="the label/scalar width differs from the header 'arch' entry (default "
                    "label=32, scalar=64), or the file is corrupt",
                )
            self.pos = start + nbytes + 1
            if not self.load:
                out.append(FoamSkipped(n, comps))
                return
            arr = np.frombuffer(self.buf, dtype=dtype, count=n * comps, offset=start)
            arr = arr.astype(dtype.newbyteorder("="))  # also detaches from the mapped file
            out.append(arr.reshape(n, comps) if comps > 1 else arr)
            return
        if n >= _FAST_LIST:
            end = self.list_end(start)
            if not self.load:
                self.pos = end + 1
                out.append(FoamSkipped(n, comps))
                return
            arr = _parse_numbers(self.buf[start:end], hint == "label")
            if arr is not None:
                if arr.size == 0 or arr.size % n:
                    raise self.error(
                        f"list announces {n} entries but holds {arr.size} numbers",
                        hint=_TRUNCATED_HINT,
                    )
                self.pos = end + 1
                out.append(arr.reshape(n, -1) if arr.size > n else arr)
                return
        self.pos = start
        got = self.items()
        if len(got) == n:
            out.append(got)
        elif n >= _FAST_LIST:
            raise self.error(f"list announces {n} entries but holds {len(got)}")
        else:  # 'key 3 (1 2);' is two tokens, not a sized list
            out.extend([n, got])

    def top_list(self, hint: str) -> np.ndarray:
        """A file whose body is one ``N(...)`` list of a primitive type."""
        t = self.next()
        if t is None or not t.isdigit() or self.peek_byte() != b"(":
            raise self.error("expected '<size> (' at the start of the list")
        self.hint = hint
        out: list[Any] = []
        self.sized_list(int(t), out)
        comps = PRIMITIVE_COMPONENTS[hint]
        dtype = np.int64 if hint == "label" else np.float64
        try:
            arr = np.asarray(out[0], dtype=dtype)
        except (ValueError, TypeError) as e:
            raise self.error(f"list of {hint} holds non-numeric entries") from e
        if arr.size != int(t) * comps:
            raise self.error(f"list announces {int(t)} {hint} entries but holds {arr.size} numbers")
        return arr.reshape(-1, comps) if comps > 1 else arr.reshape(-1)


def _banner(head: bytes) -> tuple[str | None, str | None]:
    """``(version, distribution)`` from the comment banner OpenFOAM writes above ``FoamFile``."""
    top = head[: head.find(b"FoamFile")] if b"FoamFile" in head else head[:1024]
    m = _BANNER_VERSION.search(top)
    version = m.group(1).decode("ascii", "replace") if m else None
    if version in ("", "|"):
        version = None
    low = top.lower()
    v = version or ""
    if b"extend" in low:
        dist = "foam-extend"
    elif re.match(r"^v?\d{4}$", v):
        dist = "ESI (openfoam.com)"  # vYYMM releases
    elif b"openfoam.org" in low or re.match(r"^(\d{1,2}|dev)$", v):
        dist = "Foundation (openfoam.org)"
    elif b"openfoam.com" in low:
        # OpenCFD's site before the 2011 fork also was openfoam.com (releases 1.x - 2.x)
        dist = "ESI (openfoam.com)" if not re.match(r"^[12]\.", v) else "OpenCFD (pre-fork)"
    else:
        dist = None
    return version, dist


def _open(path: Path, *, load: bool = True) -> tuple[_Parser, FoamFile]:
    buf = _load(path)
    parser = _Parser(buf, path, load=load)
    version, dist = _banner(bytes(buf[:2048]))
    head = parser.header()
    return parser, FoamFile(
        path=path, header=head, body=FoamDict(), version=version, distribution=dist
    )


def parse_foam_file(path: Path | str, *, load: bool = False) -> FoamFile:
    """Parse any FoamFile dictionary. Long lists become :class:`FoamSkipped` unless ``load``."""
    path = Path(path)
    parser, doc = _open(path, load=load)
    doc.body = parser.entries(None)
    return doc


def read_header(path: Path | str) -> FoamFile:
    """Header only (reads the first kilobytes, also of ``.gz`` files)."""
    path = Path(path)
    head = _head(path)
    parser = _Parser(head, path, load=False)
    version, dist = _banner(head[:2048])
    try:
        header = parser.header()
    except ConnectorError as e:
        raise ConnectorError(
            f"{path}: malformed FoamFile header", hint="the file is not an OpenFOAM file"
        ) from e
    doc = FoamFile(path=path, header=header, body=FoamDict(), version=version, distribution=dist)
    doc.rest = head[parser.pos :]
    return doc


def read_points(path: Path | str) -> np.ndarray:
    """``polyMesh/points`` -> ``(N, 3)`` float64."""
    parser, _ = _open(Path(path))
    return parser.top_list("vector")


def read_labels(path: Path | str) -> np.ndarray:
    """``polyMesh/owner`` | ``neighbour`` | ``*ProcAddressing`` -> ``(N,)`` int64."""
    parser, _ = _open(Path(path))
    return parser.top_list("label")


def read_faces(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """``polyMesh/faces`` -> CSR ``(offsets (N+1,), point labels)`` (ASCII ``faceList`` or binary
    ``faceCompactList``)."""
    path = Path(path)
    parser, doc = _open(path)
    if parser.binary:
        if doc.cls != "faceCompactList":
            raise ConnectorError(
                f"{path}: binary '{doc.cls}' face lists are not supported",
                hint="rewrite the mesh with 'foamFormatConvert' (ascii or faceCompactList)",
            )
        offsets = parser.top_list("label")
        labels = parser.top_list("label")
        if len(offsets) == 0 or offsets[-1] != len(labels) or (np.diff(offsets) < 0).any():
            raise parser.error("faceCompactList offsets do not match the label list")
        return offsets, labels
    t = parser.next()
    if t is None or not t.isdigit() or parser.peek_byte() != b"(":
        raise parser.error("expected '<size> (' at the start of the face list")
    n = int(t)
    start = parser.pos + 1
    end = parser.list_end(start)
    flat = _parse_numbers(parser.buf[start:end], True)
    if flat is None:
        raise parser.error("face list holds non-integer entries")
    if n == 0:
        return np.zeros((1,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    bad = parser.error(f"face list announces {n} faces but its entries do not add up")
    if flat.size == 0:
        raise bad
    k = int(flat[0])
    if k > 0 and flat.size == n * (k + 1) and (flat[:: k + 1] == k).all():
        return np.arange(n + 1, dtype=np.int64) * k, flat.reshape(n, k + 1)[:, 1:].reshape(-1)
    if n > flat.size:  # every face needs at least its size entry: n is a claim of the file
        raise bad
    sizes = np.empty((n,), dtype=np.int64)
    heads = np.empty((n,), dtype=np.int64)
    p = 0
    for i in range(n):  # mixed face sizes: each face is '<k> v0 .. vk-1'
        if p >= flat.size:
            raise bad
        heads[i] = p
        sizes[i] = flat[p]
        p += int(flat[p]) + 1
    if p != flat.size:
        raise bad
    keep = np.ones((flat.size,), dtype=bool)
    keep[heads] = False
    return np.concatenate([[0], np.cumsum(sizes)]), flat[keep]


def read_boundary(path: Path | str) -> list[FoamPatch]:
    """``polyMesh/boundary`` -> patches in file order."""
    path = Path(path)
    doc = parse_foam_file(path, load=True)
    entries = doc.body.get("")
    if isinstance(entries, tuple):  # hand-edited files: the announced patch count is off
        entries = next((e for e in entries if isinstance(e, list)), None)
    if not isinstance(entries, list):
        raise ConnectorError(
            f"{path}: no patch list found", hint="polyMesh/boundary must hold 'N ( patch {..} )'"
        )
    out = []
    for item in entries:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict)):
            raise ConnectorError(f"{path}: malformed patch entry {item!r}", hint=_TRUNCATED_HINT)
        name, d = item
        try:
            n_faces, start = int(d["nFaces"]), int(d["startFace"])
        except (KeyError, TypeError, ValueError) as e:
            raise ConnectorError(
                f"{path}: patch {name!r} lacks nFaces/startFace", hint=_TRUNCATED_HINT
            ) from e
        groups = d.get("inGroups")
        if isinstance(groups, str):
            groups = [groups]
        out.append(
            FoamPatch(
                name=str(name),
                type=str(d.get("type", "patch")),
                n_faces=n_faces,
                start_face=start,
                groups=[str(g) for g in groups] if isinstance(groups, list) else [],
                physical_type=str(d["physicalType"]) if "physicalType" in d else None,
            )
        )
    return out


def field_class(cls: str) -> tuple[str, int] | None:
    """``(location, components)`` of a supported field class, else ``None``."""
    m = _FIELD_CLASS.match(cls)
    if not m:
        return None
    kind = m.group(2)[0].lower() + m.group(2)[1:]
    return ("cell" if m.group(1) == "vol" else "point"), PRIMITIVE_COMPONENTS[kind]


def read_field(path: Path | str, *, load: bool = True) -> FoamField:
    """Parse a field file. With ``load=False`` the value lists are stepped over (cheap way to
    get ``dimensions`` and ``boundaryField``)."""
    path = Path(path)
    parser, doc = _open(path, load=load)
    name = path.name[:-3] if path.name.endswith(".gz") else path.name
    info = field_class(doc.cls)
    if info is None:
        raise ConnectorError(
            f"{path}: unsupported field class {doc.cls or '(none)'!r}",
            hint="supported: vol/point Scalar, Vector, SphericalTensor, SymmTensor, Tensor fields; "
            "surface (face flux) fields and lagrangian clouds are not read: leave them out of "
            "fields=",
        )
    location, comps = info
    body = parser.entries(None)
    dims = body.get("dimensions")
    raw = body.get("internalField", body.get("value"))
    internal: Any
    if isinstance(raw, tuple) and raw and raw[0] == "uniform" and len(raw) == 2:
        internal = ("uniform", raw[1])
    elif isinstance(raw, tuple) and raw and raw[0] == "nonuniform":
        internal = raw[-1]
        if isinstance(internal, list):
            try:
                internal = np.asarray(internal, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ConnectorError(
                    f"{path}: internalField holds non-numeric entries", hint=_TRUNCATED_HINT
                ) from e
        if isinstance(internal, np.ndarray):
            if internal.size % comps:
                raise ConnectorError(
                    f"{path}: internalField has {internal.size} numbers, not a multiple of "
                    f"{comps} components",
                    hint=_TRUNCATED_HINT,
                )
            internal = internal.reshape(-1, comps)
        elif not isinstance(internal, FoamSkipped):
            raise ConnectorError(f"{path}: cannot interpret internalField", hint=_TRUNCATED_HINT)
    else:
        raise ConnectorError(
            f"{path}: no 'internalField uniform|nonuniform' entry", hint=_TRUNCATED_HINT
        )
    boundary = body.get("boundaryField")
    return FoamField(
        name=name,
        cls=doc.cls,
        location=location,
        components=comps,
        dimensions=dims if isinstance(dims, FoamDims) else None,
        internal=internal,
        boundary=boundary if isinstance(boundary, FoamDict) else FoamDict(),
    )


# -- case layout ------------------------------------------------------------------------------


def time_directories(root: Path) -> list[tuple[float, str]]:
    """``(value, directory name)`` of the numeric time directories, sorted (``0.orig`` and
    other non-numeric names are skipped)."""
    out = []
    try:
        for p in root.iterdir():
            if _TIME_NAME.match(p.name) and p.is_dir():
                out.append((float(p.name), p.name))
    except OSError:
        return []
    return sorted(out)


def _processor_dirs(root: Path) -> list[Path]:
    found = []
    try:
        for p in root.iterdir():
            m = _PROCESSOR_DIR.match(p.name)
            if m and p.is_dir():
                found.append((int(m.group(1)), p))
    except OSError:
        return []
    return [p for _, p in sorted(found)]


def _collated_dirs(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.iterdir() if _COLLATED_DIR.match(p.name) and p.is_dir())
    except OSError:
        return []


def _mesh_dir(root: Path, region: str | None) -> Path:
    base = root / "constant"
    return (base / region if region else base) / "polyMesh"


def _has_mesh(root: Path, region: str | None = None) -> bool:
    d = _mesh_dir(root, region)
    return resolve_file(d / "faces") is not None or resolve_file(d / "boundary") is not None


def _regions(root: Path) -> list[str]:
    """Names of the mesh regions of a multi-region case (``constant/<region>/polyMesh``)."""
    try:
        return sorted(p.name for p in (root / "constant").iterdir() if _has_mesh(root, p.name))
    except OSError:
        return []


def case_root(path: Path) -> Path | None:
    """Case directory for a case dir, a ``.foam``/``.OpenFOAM`` stub or ``system/controlDict``."""
    try:
        if path.is_dir():
            root = path
        elif path.suffix.lower() in (".foam", ".openfoam") and path.is_file():
            root = path.parent
        elif path.name == "controlDict" and path.parent.name == "system" and path.is_file():
            root = path.parent.parent
        else:
            return None
        signs = (
            (root / "system" / "controlDict").is_file()
            or _has_mesh(root)
            or any(_has_mesh(p) for p in _processor_dirs(root)[:1])
            or bool(_collated_dirs(root))
            or bool(_regions(root))
        )
        return root if signs else None
    except OSError:
        return None


@dataclass
class _Layout:
    root: Path
    decomposed: bool
    #: directories holding ``constant/polyMesh`` and the time directories: the case itself, or
    #: its ``processor<i>`` directories
    parts: list[Path]
    region: str | None = None
    regions: list[str] = field(default_factory=list)

    def mesh_dirs(self) -> list[Path]:
        return [_mesh_dir(p, self.region) for p in self.parts]

    def times(self) -> list[tuple[float, str]]:
        return time_directories(self.parts[0])

    def field_dir(self, part: Path, time_name: str) -> Path:
        d = part / time_name
        return d / self.region if self.region else d


def _layout(path: Path, options: Mapping[str, Any] | None) -> _Layout:
    opts = dict(options or {})
    root = case_root(Path(path))
    if root is None:
        raise UnsupportedSourceError(
            f"{path} is not an OpenFOAM case",
            hint="point at the case directory (the one holding system/, constant/ and the time "
            "directories), at a .foam stub inside it or at system/controlDict",
        )
    procs = [p for p in _processor_dirs(root) if _has_mesh(p) or _regions(p)]
    region = opts.get("region")
    regions = _regions(root) or (_regions(procs[0]) if procs else [])
    if region is not None and str(region) not in regions:
        raise SelectionError(
            f"region {region!r} not found", hint=f"available regions: {', '.join(regions) or '-'}"
        )
    region = str(region) if region is not None else None
    serial = _has_mesh(root, region)
    parallel = bool(procs) and _has_mesh(procs[0], region)
    if not serial and not parallel:
        if regions and region is None:
            raise ConnectorError(
                f"{root} is a multi-region case (regions: {', '.join(regions)})",
                hint=f"pick one with options={{'region': '{regions[0]}'}}",
            )
        if _collated_dirs(root):
            raise ConnectorError(
                f"{root} is decomposed in the collated file format "
                f"({_collated_dirs(root)[0].name}/), which is not supported",
                hint="run 'reconstructPar' (and 'reconstructParMesh' if the mesh was built in "
                "parallel), or 'foamFormatConvert -fileHandler uncollated' in the case",
            )
        raise ConnectorError(
            f"{root} has no mesh (constant/polyMesh is missing)",
            hint="run the mesher (blockMesh / snappyHexMesh) or copy constant/polyMesh into the "
            "case; a controlDict alone carries no data",
        )
    want = opts.get("decomposed")
    if want is None:
        # results usually live in processor*/ while the case root keeps only 0/
        decomposed = parallel and (
            not serial or len(time_directories(procs[0])) > len(time_directories(root))
        )
    else:
        decomposed = _flag(want)
    if decomposed and not parallel:
        raise ConnectorError(
            f"{root} has no processor*/constant/polyMesh",
            hint="drop options={'decomposed': True}: the case is not decomposed",
        )
    if not decomposed and not serial:
        raise ConnectorError(
            f"{root} has no reconstructed mesh",
            hint="run 'reconstructPar' or drop options={'decomposed': False}",
        )
    return _Layout(
        root=root,
        decomposed=decomposed,
        parts=procs if decomposed else [root],
        region=region,
        regions=regions,
    )


# -- mesh summary -----------------------------------------------------------------------------


def _mesh_file(mesh_dir: Path, name: str) -> Path:
    found = resolve_file(mesh_dir / name)
    if found is None:
        raise ConnectorError(
            f"{mesh_dir} has no '{name}' file",
            hint="a polyMesh needs points, faces, owner, neighbour and boundary: re-run the "
            "mesher (blockMesh / snappyHexMesh) or copy the complete constant/polyMesh directory",
        )
    return found


def _list_size(path: Path) -> int:
    doc = read_header(path)
    m = re.match(rb"(?:\s+|//[^\n]*|/\*.*?\*/)*(\d+)", doc.rest, re.S)
    if not m:
        raise ConnectorError(f"{path}: no list size after the header", hint=_TRUNCATED_HINT)
    n = int(m.group(1))
    return n - 1 if doc.cls == "faceCompactList" else n


@dataclass
class _MeshCounts:
    n_points: int
    n_cells: int
    n_faces: int
    n_internal_faces: int


def mesh_counts(mesh_dir: Path) -> _MeshCounts:
    """Point/cell/face counts from the list headers (the ``note`` of ``owner`` when present,
    else the list sizes and ``max(owner) + 1``); the mesh itself is not loaded."""
    owner = _mesh_file(mesh_dir, "owner")
    m = _NOTE_COUNTS.search(str(read_header(owner).header.get("note", "")))
    if m:
        return _MeshCounts(*(int(g) for g in m.groups()))
    labels = read_labels(owner)
    return _MeshCounts(
        n_points=_list_size(_mesh_file(mesh_dir, "points")),
        n_cells=int(labels.max()) + 1 if len(labels) else 0,
        n_faces=len(labels),
        n_internal_faces=_list_size(_mesh_file(mesh_dir, "neighbour")),
    )


def _check_mesh(mesh_dir: Path) -> None:
    """Fail early, with the file named, on what VTK would report as an empty read: missing
    files, broken headers and (for uncompressed binary, where it costs nothing) short lists."""
    for name in _MESH_FILES:
        path = _mesh_file(mesh_dir, name)
        head = read_header(path)
        if not head.header:
            raise ConnectorError(
                f"{path} has no FoamFile header", hint="the file is not an OpenFOAM mesh file"
            )
        binary = str(head.header.get("format", "ascii")).lower() == "binary"
        if binary and name != "boundary" and not path.name.endswith(".gz"):
            _check_list_file(path, name)


def _check_list_file(path: Path, name: str) -> None:
    parser, doc = _open(path, load=False)
    hint = "vector" if name == "points" else "label"
    for _ in range(2 if (name == "faces" and doc.cls == "faceCompactList") else 1):
        t = parser.next()
        if t is None or not t.isdigit() or parser.peek_byte() != b"(":
            raise parser.error("expected '<size> (' at the start of the list")
        parser.hint = hint
        parser.sized_list(int(t), [])


def _diagnose(
    layout: _Layout, time_name: str | None, names: Sequence[str], *, mesh: bool = True
) -> None:
    """Re-read the files of one step with the strict parser so that a VTK failure turns into
    an error that names the broken file. ``mesh=False`` checks the fields only."""
    for part, mesh_dir in zip(layout.parts, layout.mesh_dirs(), strict=True):
        counts = mesh_counts(mesh_dir)
        if mesh:
            _diagnose_mesh(mesh_dir)
        if time_name is None:
            continue
        for name in names:
            path = resolve_file(layout.field_dir(part, time_name) / name)
            if path is None:
                continue
            fld = read_field(path)
            if fld.location != "cell" or not isinstance(fld.internal, np.ndarray):
                continue
            if len(fld.internal) != counts.n_cells:
                raise ConnectorError(
                    f"{path}: internalField has {len(fld.internal)} values but the mesh has "
                    f"{counts.n_cells} cells",
                    hint="the field belongs to another mesh (or decomposition), or is truncated",
                )


def _diagnose_mesh(mesh_dir: Path) -> None:
    """Load the four mesh lists with the strict parser and cross-check their sizes."""
    n_points = len(read_points(_mesh_file(mesh_dir, "points")))
    offsets, labels = read_faces(_mesh_file(mesh_dir, "faces"))
    owner = read_labels(_mesh_file(mesh_dir, "owner"))
    neighbour = read_labels(_mesh_file(mesh_dir, "neighbour"))
    problems = []
    if len(owner) != len(offsets) - 1:
        problems.append(f"owner has {len(owner)} entries for {len(offsets) - 1} faces")
    if len(neighbour) > len(owner):
        problems.append(f"neighbour ({len(neighbour)}) is longer than owner ({len(owner)})")
    if len(labels) and (labels.min() < 0 or labels.max() >= n_points):
        problems.append(f"faces reference point {int(labels.max())} of {n_points}")
    if problems:
        raise ConnectorError(
            f"inconsistent polyMesh in {mesh_dir}: " + "; ".join(problems),
            hint="the mesh files belong to different meshes or are corrupt; run 'checkMesh'",
        )


# -- fields and metadata ----------------------------------------------------------------------


@dataclass
class _FieldMeta:
    name: str
    cls: str
    path: Path
    location: str | None = None  # None: not a readable field class
    components: int = 1
    units: str = ""
    internal_only: bool = False


def scan_fields(directory: Path) -> dict[str, _FieldMeta]:
    """Field files of one time directory, classified from their headers (cheap)."""
    out: dict[str, _FieldMeta] = {}
    try:
        files = sorted(p for p in directory.iterdir() if p.is_file())
    except OSError:
        return out
    for path in files:
        name = path.name[:-3] if path.name.endswith(".gz") else path.name
        if name.startswith(".") or name.endswith((".orig", ".bak", "~")) or name in out:
            continue
        try:
            doc = read_header(path)
        except ConnectorError:
            continue
        if not doc.header:
            continue
        meta = _FieldMeta(name=name, cls=doc.cls, path=path)
        info = field_class(doc.cls)
        if info is not None:
            meta.location, meta.components = info
            meta.internal_only = doc.cls.endswith("::Internal")
            m = _DIMENSIONS.search(doc.rest)
            if m:
                try:
                    meta.units = foam_dimensions_to_unit([float(v) for v in m.group(1).split()])
                except ValueError:
                    meta.units = ""
        out[name] = meta
    return out


def _scalar(value: Any) -> float | None:
    """Numeric value of ``nu 1e-5;`` / ``nu [0 2 -1 0 0 0 0] 1e-5;`` / ``nu nu [..] 1e-5;``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, tuple) and value and isinstance(value[-1], int | float):
        return float(value[-1])
    return None


def _plain(value: Any, depth: int = 0) -> Any:
    """JSON-friendly rendering of a parsed value; long lists collapse to their length."""
    if isinstance(value, FoamSkipped):
        return {"nonuniform": value.n}
    if isinstance(value, FoamUniform):
        return _plain(value.value, depth)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 9 else {"nonuniform": int(len(value))}
    if isinstance(value, tuple):
        if value and value[0] == "uniform" and len(value) == 2:
            return _plain(value[1], depth)
        if value and value[0] == "nonuniform":
            inner = value[-1]
            return {"nonuniform": len(inner)} if isinstance(inner, list) else _plain(inner, depth)
        return [_plain(v, depth) for v in value]
    if isinstance(value, dict):
        if depth >= 3:
            return "{...}"
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v, depth) for v in value] if len(value) <= 9 else {"nonuniform": len(value)}
    return value


def _patch_entry(boundary: FoamDict, patch: FoamPatch) -> FoamDict | None:
    """``boundaryField`` entry of a patch: exact name, then patch group, then the last matching
    regular expression (OpenFOAM's precedence)."""
    exact = boundary.get(patch.name)
    if isinstance(exact, FoamDict) and patch.name not in boundary.patterns:
        return exact
    for group in patch.groups:
        entry = boundary.get(group)
        if isinstance(entry, FoamDict) and group not in boundary.patterns:
            return entry
    for pattern in reversed(boundary.patterns):
        entry = boundary.get(pattern)
        try:
            if isinstance(entry, FoamDict) and re.fullmatch(pattern, patch.name):
                return entry
        except re.error:
            continue
    return None


def _physical_patches(mesh_dirs: Sequence[Path]) -> list[FoamPatch]:
    """Boundary patches merged over the processors (inter-processor patches dropped)."""
    merged: dict[str, FoamPatch] = {}
    for mesh_dir in mesh_dirs:
        for p in read_boundary(_mesh_file(mesh_dir, "boundary")):
            if p.type in _PROCESSOR_PATCH_TYPES:
                continue
            if p.name in merged:
                merged[p.name].n_faces += p.n_faces
            else:
                merged[p.name] = p
    return list(merged.values())


def _safe_dict(path: Path, warnings: list[str]) -> FoamDict:
    found = resolve_file(path)
    if found is None:
        return FoamDict()
    try:
        return parse_foam_file(found, load=True).body
    except ConnectorError as e:
        warnings.append(f"could not parse {found.name}: {e.message}")
        return FoamDict()


def _turbulence(const: Path, physics: dict[str, Any], warnings: list[str]) -> None:
    turb = _safe_dict(const / "momentumTransport", warnings) or _safe_dict(
        const / "turbulenceProperties", warnings
    )
    sim = turb.get("simulationType")
    if not isinstance(sim, str):
        return
    kind = {"RASModel": "RAS", "LESModel": "LES"}.get(sim, sim)
    physics["simulation_type"] = kind
    model: Any = None
    if kind in ("RAS", "LES"):
        sub = turb.get(kind)
        if isinstance(sub, dict):
            model = sub.get("model", sub.get(f"{kind}Model"))
        if model is None:  # OpenFOAM <= 2.2 and foam-extend keep the model in its own file
            model = _safe_dict(const / f"{kind}Properties", warnings).get(f"{kind}Model")
    elif kind == "laminar":
        model = "laminar"
    if isinstance(model, str):
        physics["turbulence_model"] = model


def _flatten_numbers(
    d: Mapping[str, Any], prefix: str, out: dict[str, Any], depth: int = 0
) -> None:
    for key, value in d.items():
        if isinstance(value, dict):
            if depth < 3:
                _flatten_numbers(value, f"{prefix}{key}.", out, depth + 1)
            continue
        v = _scalar(value)
        if v is not None:
            out.setdefault(f"{prefix}{key}", v)


def _case_metadata(root: Path) -> dict[str, Any]:
    """Solver, physics and parameters from the dictionaries in ``system/`` and ``constant/``."""
    warnings: list[str] = []
    physics: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    control = _safe_dict(root / "system" / "controlDict", warnings)
    application = control.get("application")
    if isinstance(application, str):
        physics["application"] = application
    if isinstance(control.get("solver"), str):  # Foundation v11+: foamRun -solver <module>
        physics["solver_module"] = control["solver"]
    run: dict[str, Any] = {}
    for key in ("startTime", "endTime", "deltaT", "writeInterval"):
        v = _scalar(control.get(key))
        if v is not None:
            run[key] = v
    for key in ("writeFormat", "writeControl", "writeCompression"):
        if isinstance(control.get(key), str):
            run[key] = control[key]

    const = root / "constant"
    _turbulence(const, physics, warnings)
    transport = _safe_dict(const / "physicalProperties", warnings) or _safe_dict(
        const / "transportProperties", warnings
    )
    if isinstance(transport.get("transportModel"), str):
        physics["transport_model"] = transport["transportModel"]
    if isinstance(transport.get("phases"), list):
        physics["phases"] = [str(p) for p in transport["phases"]]
    _flatten_numbers(transport, "", parameters)
    thermo = _safe_dict(const / "thermophysicalProperties", warnings)
    if thermo:
        physics["compressible"] = True
        if isinstance(thermo.get("thermoType"), dict):
            physics["thermo"] = {str(k): str(v) for k, v in thermo["thermoType"].items()}
        mixture = thermo.get("mixture")
        if isinstance(mixture, dict):
            leaves: dict[str, Any] = {}
            _flatten_numbers(mixture, "", leaves)
            for key, v in leaves.items():
                parameters.setdefault(key.rsplit(".", 1)[-1], v)
    gravity = _safe_dict(const / "g", warnings).get("value")
    if isinstance(gravity, list) and len(gravity) == 3:
        physics["gravity"] = [float(v) for v in gravity]

    schemes = _safe_dict(root / "system" / "fvSchemes", warnings)
    ddt = schemes.get("ddtSchemes")
    ddt_default = ddt.get("default") if isinstance(ddt, dict) else None
    if isinstance(ddt_default, tuple):
        ddt_default = ddt_default[0]
    if isinstance(ddt_default, str):
        physics["ddt_scheme"] = ddt_default
        steady = ddt_default in ("steadyState", "steady")
    else:
        steady = application in _STEADY_APPLICATIONS
    physics["steady"] = bool(steady)
    return {
        "physics": physics,
        "parameters": parameters,
        "run": run,
        "warnings": warnings,
        "foam_version": control.get("foamVersion"),
    }


def _version(layout: _Layout, times: Sequence[tuple[float, str]]) -> tuple[str | None, str | None]:
    """Version and distribution from the banner of a solver-written file (latest time first:
    ``system/`` dictionaries are often copied from a tutorial of another release)."""
    candidates: list[Path] = []
    if times:
        d = layout.field_dir(layout.parts[0], times[-1][1])
        try:
            candidates += sorted(p for p in d.iterdir() if p.is_file())[:3]
        except OSError:
            pass
    for name in ("owner", "points"):
        found = resolve_file(layout.mesh_dirs()[0] / name)
        if found is not None:
            candidates.append(found)
    candidates.append(layout.root / "system" / "controlDict")
    version = dist = None
    for path in candidates:
        try:
            v, d2 = _banner(_head(path, 2048))
        except ConnectorError:
            continue
        version, dist = version or v, dist or d2
        if version and dist:
            break
    return version, dist


# -- VTK driver -------------------------------------------------------------------------------


class _Reader:
    """One ``vtkOpenFOAMReader`` on a case (or processor) directory, with errors captured."""

    def __init__(self, part: Path, mesh_dir: Path, *, point_data: bool) -> None:
        from vtkmodules.vtkIOGeometry import vtkOpenFOAMReader

        self.part = part
        self.errors: list[str] = []
        reader: Any = vtkOpenFOAMReader()
        # the stub does not have to exist: the reader only takes its directory
        reader.SetFileName(str(part / "case.foam"))
        reader.SetCreateCellToPoint(1 if point_data else 0)
        reader.SetCacheMesh(1)
        reader.SetReadZones(0)
        reader.SetSkipZeroTime(0)
        reader.SetAddDimensionsToArrayNames(0)
        # only a fallback for headers without 'arch' (old releases, foam-extend)
        arch = ""
        owner = resolve_file(mesh_dir / "owner")
        points = resolve_file(mesh_dir / "points")
        if owner is not None and points is not None:
            arch = str(read_header(owner).header.get("arch", ""))
            arch += str(read_header(points).header.get("arch", ""))
        reader.SetUse64BitLabels(1 if "label=64" in arch else 0)
        reader.SetUse64BitFloats(0 if "scalar=32" in arch else 1)
        self.reader = reader
        # vtkOpenFOAMReader follows #include on its own: confine them before it reads
        self.root = foam_case_root(part / "system" / "controlDict")
        if not (self.root / "system" / "controlDict").is_file():
            self.root = part.resolve()
        self._scanned: set[Path] = set()
        for base in dict.fromkeys((self.root, part.resolve())):
            for sub in ("system", "constant"):
                confine_includes(self.root, _files_below(base / sub), self._scanned)
        self._confine_time(None)
        self._quietly(reader.UpdateInformation)
        self.times = _vtk.reader_times(reader)

    def _quietly(self, call: Any) -> Any:
        """Run a VTK call with its messages collected in ``self.errors`` instead of printed
        (the reader reports through internal objects, so an observer on it sees nothing)."""
        from vtkmodules.vtkCommonCore import vtkLogger, vtkOutputWindow, vtkStringOutputWindow

        previous = vtkOutputWindow.GetInstance()
        verbosity = vtkLogger.GetCurrentVerbosityCutoff()
        window = vtkStringOutputWindow()
        vtkOutputWindow.SetInstance(window)
        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
        try:
            return call()
        finally:
            vtkLogger.SetStderrVerbosity(vtkLogger.ConvertToVerbosity(verbosity))
            vtkOutputWindow.SetInstance(previous)
            for line in (window.GetOutput() or "").splitlines():
                line = line.strip()
                if line and not line.startswith(("ERROR: In ", "Warning: In ")):
                    self.errors.append(line)

    def _confine_time(self, time: float | None) -> None:
        """Scan the time directory VTK is about to read (the first one for ``None``, which is
        what ``UpdateInformation`` opens to list the fields)."""
        dirs: list[tuple[float, Path]] = []
        try:
            for p in self.part.iterdir():
                if p.is_dir():
                    try:
                        dirs.append((float(p.name), p))
                    except ValueError:
                        continue
        except OSError:
            return
        if not dirs:
            return
        dirs.sort()
        if time is None:
            chosen = [dirs[0][1], dirs[-1][1]]
        else:
            chosen = [p for v, p in dirs if math.isclose(v, time, rel_tol=1e-9, abs_tol=1e-12)] or [
                min(dirs, key=lambda d: abs(d[0] - time))[1]
            ]
        for d in chosen:
            confine_includes(self.root, _files_below(d), self._scanned)

    def names(self, kind: str) -> list[str]:
        r = self.reader
        count = getattr(r, f"GetNumberOf{kind}Arrays")()
        return [getattr(r, f"Get{kind}ArrayName")(i) for i in range(count)]

    def select(self, kind: str, enabled: Sequence[str]) -> None:
        setter = getattr(self.reader, f"Set{kind}ArrayStatus")
        for name in self.names(kind):
            setter(name, 1 if name in enabled else 0)

    def load(self, time: float | None) -> list[tuple[str, Any]]:
        """Leaves ``(vtk block path, dataset)`` at ``time`` (nearest stored step)."""
        self.errors.clear()
        t = None
        if time is not None and self.times:
            t = min(self.times, key=lambda v: abs(v - time))
        self._confine_time(t)
        obj = self._quietly(lambda: _vtk.update_to_time(self.reader, t))
        return list(_vtk.iter_leaves(obj))


def _vtk_patch_names(zones: Sequence[str], region: str | None) -> list[str]:
    prefix = f"/{region}/" if region else ""
    out = []
    for z in zones:
        out.append(prefix + (z if z == INTERNAL_ZONE else PATCH_PREFIX + z[len(PATCH_PREFIX) :]))
    return out


def _zone_of_leaf(leaf: str, region: str | None) -> str | None:
    parts = leaf.strip("/").split("/")
    if region and parts and parts[0] == region:
        parts = parts[1:]
    if parts == [INTERNAL_ZONE]:
        return INTERNAL_ZONE
    if len(parts) >= 2 and parts[0] in ("boundary", "Patches", "patch"):
        return PATCH_PREFIX + "/".join(parts[1:])
    return None


def has_addressing(mesh_dirs: Sequence[Path]) -> bool:
    return all(
        resolve_file(d / "cellProcAddressing") is not None
        and resolve_file(d / "pointProcAddressing") is not None
        for d in mesh_dirs
    )


def _processor_weld(mesh_dirs: Sequence[Path]) -> tuple[list[np.ndarray], int]:
    """Global point ids for processor meshes that carry no ``pointProcAddressing`` (meshes
    built in parallel): points are identified only where they lie on inter-processor patches,
    so baffles and sliding interfaces with coincident points stay separate."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    starts = [0]
    ids, xyz = [], []
    for d in mesh_dirs:
        pts = read_points(_mesh_file(d, "points"))
        offsets, labels = read_faces(_mesh_file(d, "faces"))
        local = [np.zeros((0,), dtype=np.int64)]
        for p in read_boundary(_mesh_file(d, "boundary")):
            if p.type in _PROCESSOR_PATCH_TYPES and p.n_faces > 0:
                local.append(labels[offsets[p.start_face] : offsets[p.start_face + p.n_faces]])
        on_seam = np.unique(np.concatenate(local))
        ids.append(on_seam + starts[-1])
        xyz.append(pts[on_seam])
        starts.append(starts[-1] + len(pts))
    total = starts[-1]
    gid = np.arange(total, dtype=np.int64)
    seam_ids, seam_xyz = np.concatenate(ids), np.concatenate(xyz)
    if len(seam_ids) > 1:
        span = float(np.linalg.norm(seam_xyz.max(axis=0) - seam_xyz.min(axis=0))) or 1.0
        pairs = KDTree(seam_xyz).query_pairs(1e-9 * span, output_type="ndarray")
        if len(pairs):
            n = len(seam_ids)
            graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, comp = connected_components(graph, directed=False)
            first = np.full((int(comp.max()) + 1,), total, dtype=np.int64)
            np.minimum.at(first, comp, seam_ids)
            gid[seam_ids] = first[comp]
    _, compact = np.unique(gid, return_inverse=True)
    n_points = int(compact.max()) + 1 if total else 0
    return [compact[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)], n_points


class _Stitcher:
    """Joins the processor pieces of ``internalMesh`` into one mesh.

    With the ``cellProcAddressing``/``pointProcAddressing`` maps ``decomposePar`` writes, the
    undecomposed cell and point numbering is restored exactly. Without them the cells keep
    processor order and points are welded on the processor patches (:func:`_processor_weld`).
    The point map is built once, so a moving mesh keeps its point count."""

    def __init__(self, mesh_dirs: Sequence[Path]) -> None:
        self.mesh_dirs = list(mesh_dirs)
        self.mode = "addressing" if has_addressing(mesh_dirs) else "processor-patch weld"
        self.cells: list[int] = []
        self.points: list[np.ndarray] = []
        self.order: np.ndarray | None = None
        self.n_points = 0
        self.ready = False
        self.available = True
        #: patch -> (global point ids used, triangles, parent face, faces per processor)
        self.surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}

    def prepare(self) -> None:
        if self.ready:
            return
        self.ready = True
        try:
            if self.mode == "addressing":
                cells = [read_labels(_mesh_file(d, "cellProcAddressing")) for d in self.mesh_dirs]
                self.points = [
                    read_labels(_mesh_file(d, "pointProcAddressing")) for d in self.mesh_dirs
                ]
                self.cells = [len(c) for c in cells]
                joined = np.concatenate(cells)
                self.order = np.argsort(joined, kind="stable")
                self.n_points = max((int(p.max()) + 1 for p in self.points if len(p)), default=0)
                # every original cell exactly once, otherwise the maps are not a partition
                if not np.array_equal(joined[self.order], np.arange(len(joined))):
                    self.available = False
            else:
                self.points, self.n_points = _processor_weld(self.mesh_dirs)
                self.cells = [mesh_counts(d).n_cells for d in self.mesh_dirs]
        except ConnectorError:
            self.available = False

    def fits(self, datasets: Sequence[Any]) -> bool:
        self.prepare()
        return (
            self.available
            and len(datasets) == len(self.cells)
            and all(
                ds is not None and ds.GetNumberOfCells() == c and ds.GetNumberOfPoints() == len(p)
                for ds, c, p in zip(datasets, self.cells, self.points, strict=True)
            )
        )

    def prepare_patches(self, names: Sequence[str]) -> None:
        """Surface of each patch in the stitched point numbering, from the processor face
        lists. Patch pieces then share points across processors exactly, and a moving mesh
        keeps its point count (coincidence welding would not)."""
        self.prepare()
        if not self.available or not names:
            return
        conn: dict[str, list[np.ndarray]] = {n: [] for n in names}
        sizes: dict[str, list[np.ndarray]] = {n: [] for n in names}
        counts: dict[str, list[int]] = {n: [] for n in names}
        try:
            for i, d in enumerate(self.mesh_dirs):
                patches = {p.name: p for p in read_boundary(_mesh_file(d, "boundary"))}
                offsets = labels = None
                for n in names:
                    p = patches.get(n)
                    if p is None or p.n_faces <= 0:
                        counts[n].append(0)
                        continue
                    if offsets is None or labels is None:
                        offsets, labels = read_faces(_mesh_file(d, "faces"))
                    a, b = p.start_face, p.start_face + p.n_faces
                    if b >= len(offsets):
                        return
                    local = labels[offsets[a] : offsets[b]]
                    if len(local) and local.max() >= len(self.points[i]):
                        return
                    conn[n].append(self.points[i][local])
                    sizes[n].append(np.diff(offsets[a : b + 1]))
                    counts[n].append(p.n_faces)
        except ConnectorError:
            return
        for n in names:
            if not conn[n]:
                continue
            used, inverse = np.unique(np.concatenate(conn[n]), return_inverse=True)
            ends = np.concatenate([[0], np.cumsum(np.concatenate(sizes[n]))])
            faces, parent = meshutil.triangulate(meshutil.pad_cells(inverse, ends))
            self.surfaces[n] = (used, faces.astype(np.int32), parent, counts[n])

    def stitched_points(self, datasets: Sequence[Any]) -> np.ndarray:
        pieces = [_vtk._points(ds) for ds in datasets]
        pts = np.zeros((self.n_points, 3), dtype=pieces[0].dtype)
        for piece, addr in zip(pieces, self.points, strict=True):
            pts[addr] = piece
        return pts

    def patch_sample(
        self, name: str, parts: Sequence[Any], internal: Sequence[Any], sid: str, kw: dict[str, Any]
    ) -> RawSample | None:
        """One surface sample from the processor pieces of a patch (``None`` when the pieces
        do not match the face lists, so the caller can fall back)."""
        surface = self.surfaces.get(name)
        if surface is None:
            return None
        used, faces, parent, counts = surface
        expect = [c for c in counts if c > 0]
        if len(parts) != len(expect) or any(
            ds.GetNumberOfCells() != c for ds, c in zip(parts, expect, strict=True)
        ):
            return None
        steps = [_vtk.dataset_to_sample(ds, sid, **{**kw, "geometry": False}) for ds in parts]
        out = RawSample(
            id=sid,
            kind="surface",
            zone=kw["zone"],
            time=kw["time"],
            time_index=kw["time_index"],
            metadata=kw["metadata"],
        )
        for fname, f0 in steps[0].fields.items():
            # point values of patch pieces have no global numbering: only face data is kept
            if f0.location != "cell" or not all(fname in s.fields for s in steps):
                continue
            data = np.concatenate([s.fields[fname].data for s in steps])
            if len(data) != len(faces):
                return None
            out.fields[fname] = as_field(data, "cell")
        if kw["geometry"]:
            out.points = self.stitched_points(internal)[used]
            out.faces = faces
            if not np.array_equal(parent, np.arange(len(parent))):
                out.static["face_parent"] = as_field(parent.astype(np.int32), "cell")
        return out

    def _ordered(self, data: np.ndarray) -> np.ndarray:
        return data if self.order is None else data[self.order]

    def combine(self, pieces: Sequence[RawSample]) -> RawSample:
        first = pieces[0]
        out = RawSample(
            id=first.id,
            kind="mesh",
            zone=first.zone,
            time=first.time,
            time_index=first.time_index,
            metadata=first.metadata,
        )
        if first.points is not None:
            pts = np.zeros((self.n_points, first.points.shape[1]), dtype=first.points.dtype)
            for piece, addr in zip(pieces, self.points, strict=True):
                pts[addr] = piece.points
            out.points = pts
        if first.cells is not None:
            width = max(p.cells.shape[1] for p in pieces if p.cells is not None)
            blocks, types = [], []
            for piece, addr in zip(pieces, self.points, strict=True):
                c = np.asarray(piece.cells)
                g = np.full((len(c), width), -1, dtype=np.int64)
                g[:, : c.shape[1]] = np.where(c >= 0, addr[np.clip(c, 0, None)], -1)
                blocks.append(g)
                types.append(np.asarray(piece.cell_types))
            out.cells = self._ordered(np.concatenate(blocks))
            out.cell_types = self._ordered(np.concatenate(types))
        for name, f0 in first.fields.items():
            if not all(name in p.fields for p in pieces):
                continue
            if f0.location == "cell":
                data = self._ordered(np.concatenate([p.fields[name].data for p in pieces]))
            else:
                data = np.zeros((self.n_points, f0.shape[-1]), dtype=f0.data.dtype)
                for piece, addr in zip(pieces, self.points, strict=True):
                    data[addr] = piece.fields[name].data
            out.fields[name] = as_field(data, f0.location)
        return out


@dataclass
class _Plan:
    """Everything ``describe`` and ``read`` share about one case."""

    layout: _Layout
    patches: list[FoamPatch]
    zones: list[ZoneInfo]
    all_times: list[tuple[float, str]]
    axis: list[tuple[float, str]]
    steady_solver: bool
    info: dict[str, Any]


def _plan(path: Path, options: Mapping[str, Any] | None) -> _Plan:
    layout = _layout(path, options)
    opts = dict(options or {})
    mesh_dirs = layout.mesh_dirs()
    for d in mesh_dirs:
        for name in _MESH_FILES:
            _mesh_file(d, name)
    patches = _physical_patches(mesh_dirs)
    counts = [mesh_counts(d) for d in mesh_dirs]
    zones = [
        ZoneInfo(
            name=INTERNAL_ZONE,
            kind="mesh",
            role="volume",
            # shared processor-boundary points are counted once per processor: unknown total
            n_points=counts[0].n_points if not layout.decomposed else None,
            n_cells=sum(c.n_cells for c in counts),
            dim=3,
            default=True,
        )
    ]
    for p in patches:
        if p.n_faces <= 0:
            continue
        zones.append(
            ZoneInfo(
                name=PATCH_PREFIX + p.name,
                kind="surface",
                role="boundary",
                n_cells=p.n_faces,
                dim=3,
                bc_type=p.type,
                default=False,
            )
        )
    info = _case_metadata(layout.root)
    all_times = layout.times()
    steady_solver = bool(info["physics"].get("steady"))
    mode = str(opts.get("iterations", "last")).lower()
    if mode not in ("last", "all"):
        raise ConnectorError(f"option iterations={mode!r} is not valid", hint="use 'last' or 'all'")
    axis = all_times[-1:] if (steady_solver and mode == "last") else all_times
    return _Plan(layout, patches, zones, all_times, axis, steady_solver, info)


def _mesh_source(
    layout: _Layout, times: Sequence[tuple[float, str]], index: int
) -> tuple[str, str]:
    """Names of the directories that provide the points and the faces valid at step ``index``
    (a moving or changing mesh stores ``<time>/polyMesh``; the latest one at or before the
    step applies)."""
    points = faces = "constant"
    target = times[index][0] if times else None
    if target is None:
        return points, faces
    for value, name in layout.times():
        if value > target:
            break
        d = layout.field_dir(layout.parts[0], name) / "polyMesh"
        if resolve_file(d / "points") is not None:
            points = name
        if resolve_file(d / "faces") is not None:
            faces = name
    return points, faces


class OpenFoamConnector(BaseConnector):
    id = "openfoam"
    name = "OpenFOAM case (Foundation, ESI, foam-extend)"
    formats = ["openfoam"]
    vendor = "OpenFOAM Foundation / OpenCFD (ESI) / foam-extend (open format)"
    license = "Apache-2.0 (own FoamFile parser; mesh assembly by VTK, BSD-3)"
    extensions = [".foam", ".OpenFOAM"]
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["mesh", "surface"]
    requires: list[str] = []
    export_help = (
        "Point at the case directory (system/, constant/polyMesh, time directories), a .foam "
        "stub inside it or system/controlDict. Decomposed cases (processor*/) are read directly; "
        "run reconstructPar first for the collated format (processors<N>/)."
    )
    status_note = (
        "Reads ASCII, binary and gzip cases, serial and decomposed (uncollated), with moving "
        "points; multi-region with options={'region': ...}. Zones: internalMesh (default) and "
        "patch/<name> (opt-in). Steady-solver cases read the last iteration unless "
        "options={'iterations': 'all'}. Not read: surface (flux) fields, lagrangian clouds, cell "
        "and face zones, collated decomposition."
    )

    def detect(self, path: Path) -> Detection | None:
        try:
            path = Path(path)
            root = case_root(path)
            if root is None:
                return None
            control = (root / "system" / "controlDict").is_file()
            procs = _processor_dirs(root)
            serial = _has_mesh(root)
            parallel = bool(procs) and (_has_mesh(procs[0]) or bool(_regions(procs[0])))
            regions = _regions(root)
            collated = _collated_dirs(root)
            notes, needs = [], []
            if control:
                notes.append("system/controlDict")
            if serial:
                notes.append("constant/polyMesh")
            if parallel:
                notes.append(f"{len(procs)} processor directories")
            if regions:
                notes.append("regions: " + ", ".join(regions))
            n_times = len(time_directories(procs[0] if parallel and not serial else root))
            notes.append(f"{n_times} time directories")
            has_mesh = serial or parallel or bool(regions)
            if has_mesh:
                confidence = 1.0 if control else 0.9
            elif collated:
                confidence = 0.9 if control else 0.6
                needs.append("reconstructPar (collated decomposition is not read directly)")
            else:
                confidence = 0.5
                needs.append("a mesh: constant/polyMesh is missing (run blockMesh)")
            fmt = "openfoam-decomposed" if parallel and not serial else "openfoam"
            return Detection(self.id, path, confidence, fmt, notes, needs)
        except OSError:
            return None

    # -- describe -----------------------------------------------------------------------
    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        plan = _plan(path, options)
        layout, info = plan.layout, plan.info
        version, dist = _version(layout, plan.all_times)
        physics = dict(info["physics"])
        if dist:
            physics["distribution"] = dist
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format="openfoam-decomposed" if layout.decomposed else "openfoam",
            solver="OpenFOAM",
            solver_version=str(info["foam_version"]) if info["foam_version"] else version,
            domain="cfd",
            physics=physics,
            zones=plan.zones,
            time=TimeInfo(
                values=[t for t, _ in plan.axis],
                units="s",
                quantity="iteration" if plan.steady_solver else "time",
            ),
            units={"length": "m", "system": "si"},
            parameters=dict(info["parameters"]),
            warnings=list(info["warnings"]),
        )
        desc.metadata["run"] = info["run"]
        desc.metadata["case"] = str(layout.root)
        if layout.decomposed:
            desc.metadata["processors"] = len(layout.parts)
        if layout.region:
            desc.metadata["region"] = layout.region
        if any(p.type == "empty" for p in plan.patches):
            physics["two_dimensional"] = True

        metas: dict[str, _FieldMeta] = {}
        skipped: dict[str, str] = {}
        part = layout.parts[0]
        for _, name in dict.fromkeys([*plan.axis[:1], *plan.axis[-1:]]):
            for key, meta in scan_fields(layout.field_dir(part, name)).items():
                if meta.location is None:
                    skipped.setdefault(key, meta.cls)
                else:
                    metas.setdefault(key, meta)
        for meta in metas.values():
            desc.fields.append(
                FieldInfo(
                    name=meta.name,
                    location=meta.location or "cell",
                    components=meta.components,
                    units=meta.units,
                    component_names=COMPONENT_NAMES.get(meta.components),
                    zones=[INTERNAL_ZONE] if meta.internal_only else None,
                )
            )
        if skipped:
            desc.metadata["skipped_fields"] = skipped
            odd = {k: v for k, v in skipped.items() if not v.startswith("surface")}
            if odd:
                listing = ", ".join(f"{k} ({v or 'no class'})" for k, v in odd.items())
                desc.warnings.append(f"fields of unsupported classes are not read: {listing}")
        p_meta = metas.get("p")
        if p_meta is not None and p_meta.units:
            kinematic = p_meta.units == "m^2/s^2"
            if kinematic:
                physics.setdefault("compressible", False)
            physics["pressure"] = "kinematic (p/rho)" if kinematic else "static"

        desc.boundary_conditions = self._boundary_conditions(plan, desc.warnings)
        if plan.steady_solver and len(plan.all_times) > 1:
            listed = ", ".join(n for _, n in plan.all_times[:8])
            more = " ..." if len(plan.all_times) > 8 else ""
            desc.metadata["iterations"] = [t for t, _ in plan.all_times]
            if len(plan.axis) == 1:
                desc.warnings.append(
                    f"steady solver: the time directories ({listed}{more}) are iterations; only "
                    f"the last one ({plan.axis[0][1]}) is read, as a steady sample. Pass "
                    "options={'iterations': 'all'} to read every stored iteration"
                )
            else:
                desc.warnings.append(
                    f"steady solver: the time axis ({listed}{more}) counts iterations, not seconds"
                )
        if not plan.all_times:
            desc.warnings.append("no time directories: the case holds a mesh without fields")
        if layout.decomposed and not has_addressing(layout.mesh_dirs()):
            desc.warnings.append(
                "processor*/constant/polyMesh has no cellProcAddressing/pointProcAddressing: "
                "cells keep processor order and points are welded on the processor patches"
            )
        return desc

    def _boundary_conditions(self, plan: _Plan, warnings: list[str]) -> dict[str, Any]:
        """Patch type plus the ``boundaryField`` entry of every volume field, taken from the
        first time directory (where the case prescribes them)."""
        out: dict[str, Any] = {}
        for p in plan.patches:
            row: dict[str, Any] = {"type": p.type, "n_faces": p.n_faces}
            if p.groups:
                row["groups"] = p.groups
            if p.physical_type:
                row["physical_type"] = p.physical_type
            out[p.name] = row
        if not plan.all_times:
            return out
        layout = plan.layout
        directory = layout.field_dir(layout.parts[0], plan.all_times[0][1])
        for meta in scan_fields(directory).values():
            if meta.location != "cell" or meta.internal_only:
                continue
            try:
                if meta.path.name.endswith(".gz") and meta.path.stat().st_size > _BC_GZ_LIMIT:
                    warnings.append(f"boundary conditions of {meta.name} skipped (large .gz file)")
                    continue
                fld = read_field(meta.path, load=False)
            except (ConnectorError, OSError) as e:
                warnings.append(f"boundary conditions of {meta.name} not read: {e}")
                continue
            for p in plan.patches:
                entry = _patch_entry(fld.boundary, p)
                if entry is not None:
                    out[p.name][meta.name] = _plain(entry)
        return out

    # -- read ---------------------------------------------------------------------------
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
        opts = dict(options or {})
        plan = _plan(path, options)
        layout = plan.layout
        names = [z.name for z in plan.zones]
        selected = _vtk.select_zones(names, zones, [z.name for z in plan.zones if z.default])
        if not selected:
            raise SelectionError(
                "the zone selection is empty", hint=f"available zones: {', '.join(names)}"
            )
        axis = plan.axis
        indices = select_time_indices([t for t, _ in axis], times)
        transient = len(axis) > 1
        point_data = _flag(opts.get("point_data", False))
        for d in layout.mesh_dirs():
            _check_mesh(d)

        desc = self.describe(path, options=options)
        metadata: dict[str, Any] = {
            "solver": desc.solver,
            "solver_version": desc.solver_version,
            "physics": desc.physics,
            "parameters": desc.parameters,
            "units": desc.units,
            "source": str(layout.root),
            "format": desc.format,
            "time_units": "s",
        }
        base = meshutil.sanitize_id(layout.root.resolve().name)
        readers = [
            _Reader(part, mesh_dir, point_data=point_data)
            for part, mesh_dir in zip(layout.parts, layout.mesh_dirs(), strict=True)
        ]
        stitcher = _Stitcher(layout.mesh_dirs()) if layout.decomposed else None
        loaded = list(selected)
        patch_names = [z[len(PATCH_PREFIX) :] for z in selected if z != INTERNAL_ZONE]
        if stitcher is not None and patch_names:
            stitcher.prepare_patches(patch_names)
            if stitcher.surfaces and INTERNAL_ZONE not in loaded:
                loaded.append(INTERNAL_ZONE)  # its points carry the stitched patch surfaces
        vtk_patches = _vtk_patch_names(loaded, layout.region)
        for r in readers:
            r.select("Patch", vtk_patches)
            r.select("Lagrangian", [])
        emitted: dict[str, tuple[str, str]] = {}  # zone -> mesh source of the geometry sent
        seen: dict[str, dict[str, tuple[str, int]]] = {z: {} for z in selected}
        sizes: dict[str, dict[str, int]] = {z: {} for z in selected}

        for ti in indices:
            time_value, time_name = axis[ti] if axis else (None, None)
            metas: dict[str, _FieldMeta] = {}
            if time_name is not None:
                metas = scan_fields(layout.field_dir(layout.parts[0], time_name))
            self._check_requested(metas, fields)
            cell_names = [
                m.name for m in metas.values() if m.location == "cell" and wants(m.name, fields)
            ]
            point_names = [
                m.name for m in metas.values() if m.location == "point" and wants(m.name, fields)
            ]
            if time_name is not None:
                self._check_binary_fields(layout, time_name, cell_names + point_names)
            pieces: dict[str, list[Any]] = {z: [] for z in selected}
            internal_parts: list[Any] = []
            errors: list[str] = []
            for r in readers:
                r.select("Cell", cell_names)
                r.select("Point", point_names)
                found = None
                for leaf, ds in r.load(time_value):
                    zone = _zone_of_leaf(leaf, layout.region)
                    if zone == INTERNAL_ZONE:
                        found = ds
                    if zone in pieces:
                        pieces[zone].append(ds)
                internal_parts.append(found)
                errors += r.errors
            lost = self._check_step(
                layout, time_name, cell_names, pieces, internal_parts, errors, selected
            )
            if lost and time_name is not None:
                self._inject(layout, time_name, lost, internal_parts)
            stitched = stitcher is not None and stitcher.fits(internal_parts)

            source = _mesh_source(layout, axis, ti)
            for zone in selected:
                parts = pieces[zone]
                if not parts:
                    continue
                previous = emitted.get(zone)
                geometry = previous is None or previous != source
                sid = meshutil.zone_sample_id(base, zone, len(selected))
                meta = dict(metadata)
                meta["zone_role"] = "volume" if zone == INTERNAL_ZONE else "boundary"
                if zone == INTERNAL_ZONE:
                    meta["boundary_conditions"] = desc.boundary_conditions
                else:
                    pname = zone[len(PATCH_PREFIX) :]
                    bc = desc.boundary_conditions.get(pname, {})
                    meta["boundary_conditions"] = {pname: bc}
                    meta["bc_type"] = bc.get("type")
                kw: dict[str, Any] = {
                    "zone": zone,
                    "fields": fields,
                    "time": float(time_value) if transient and time_value is not None else None,
                    "time_index": ti if transient else None,
                    "geometry": geometry,
                    "metadata": meta,
                    "force_kind": "mesh" if zone == INTERNAL_ZONE else "surface",
                }
                sample: RawSample | None = None
                if stitcher is None:
                    sample = _vtk.dataset_to_sample(parts[0], sid, **kw)
                elif stitched and zone == INTERNAL_ZONE:
                    meta["decomposed_merge"] = stitcher.mode
                    sample = stitcher.combine(
                        [_vtk.dataset_to_sample(ds, sid, **kw) for ds in internal_parts]
                    )
                elif stitched:
                    sample = stitcher.patch_sample(
                        zone[len(PATCH_PREFIX) :], parts, internal_parts, sid, kw
                    )
                    if sample is not None:
                        meta["decomposed_merge"] = stitcher.mode
                if sample is None:
                    # last resort: weld whatever coincides (not stable for moving meshes)
                    meta["decomposed_merge"] = "coincident points"
                    merged = parts[0] if len(parts) == 1 else _vtk.merge_datasets(parts)
                    sample = _vtk.dataset_to_sample(merged, sid, **kw)
                if geometry and previous is not None and previous[1] == source[1]:
                    # moving mesh, same topology: only the points are new
                    sample.cells = sample.cell_types = sample.faces = None
                    sample.static.pop("face_parent", None)
                emitted[zone] = source
                self._finish(sample, metas, seen[zone], sizes[zone])
                yield sample

    @staticmethod
    def _check_requested(metas: Mapping[str, _FieldMeta], fields: Sequence[str] | None) -> None:
        """A field asked for by its exact name must be readable (globs just skip it)."""
        if fields is None:
            return
        for name in [fields] if isinstance(fields, str) else fields:
            meta = metas.get(name)
            if meta is not None and meta.location is None:
                raise ConnectorError(
                    f"field {name!r} has class {meta.cls or '(none)'!r}, which is not supported",
                    hint="supported: vol/point Scalar, Vector, SphericalTensor, SymmTensor and "
                    "Tensor fields; surface (face flux) fields are not read: drop it from fields=",
                )

    @staticmethod
    def _check_binary_fields(layout: _Layout, time_name: str, names: Sequence[str]) -> None:
        """Structural check of uncompressed binary fields: list sizes are known from the
        headers, so a truncated file is found without reading it."""
        for part in layout.parts:
            for name in names:
                path = layout.field_dir(part, time_name) / name
                if not path.is_file():
                    continue
                if str(read_header(path).header.get("format", "ascii")).lower() == "binary":
                    read_field(path, load=False)

    @staticmethod
    def _check_step(
        layout: _Layout,
        time_name: str | None,
        cell_names: Sequence[str],
        pieces: Mapping[str, list[Any]],
        internal: Sequence[Any],
        errors: Sequence[str],
        selected: Sequence[str],
    ) -> list[str]:
        """Raise when a selected zone came back empty or a file is broken; return the volume
        fields VTK skipped although their files are sound."""
        missing = [z for z in selected if not pieces[z]]
        lost: list[str] = []
        for ds in internal:
            if ds is None:
                continue
            data = ds.GetCellData()
            have = {data.GetArrayName(i) for i in range(data.GetNumberOfArrays())}
            lost += [n for n in cell_names if n not in have and n not in lost]
        if not missing and not lost:
            return []
        # the strict parser names the broken file when there is one
        _diagnose(layout, time_name, lost or cell_names, mesh=bool(missing))
        if not missing:
            return lost
        where = f"time {time_name}" if time_name is not None else "the mesh"
        detail = "; ".join(dict.fromkeys(errors))
        raise ConnectorError(
            f"VTK's OpenFOAM reader returned no data for {', '.join(missing)} at {where} of "
            f"{layout.root}" + (f": {detail}" if detail else ""),
            hint="check the case with 'checkMesh' / 'foamDictionary'; files written by a newer "
            "OpenFOAM release may need 'foamFormatConvert'",
        )

    @staticmethod
    def _inject(
        layout: _Layout, time_name: str, names: Sequence[str], internal: Sequence[Any]
    ) -> None:
        """Volume fields that parse correctly but that VTK did not deliver are loaded with the
        connector's own parser and attached to the internal mesh pieces."""
        from vtkmodules.util.numpy_support import numpy_to_vtk

        if any(ds is None for ds in internal):
            return
        for name in names:
            arrays = []
            try:
                for part, ds in zip(layout.parts, internal, strict=True):
                    path = resolve_file(layout.field_dir(part, time_name) / name)
                    if path is None:
                        break
                    values = read_field(path).values(ds.GetNumberOfCells())
                    if len(values) != ds.GetNumberOfCells():
                        break
                    if values.shape[1] == 6:  # stored like VTK's own symmetric tensors
                        values = values[:, _SYMM_VTK_TO_FOAM]
                    arrays.append(values)
            except ConnectorError:
                continue
            if len(arrays) != len(internal):
                continue
            for ds, values in zip(internal, arrays, strict=True):
                arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
                arr.SetName(name)
                ds.GetCellData().AddArray(arr)

    @staticmethod
    def _finish(
        sample: RawSample,
        metas: Mapping[str, _FieldMeta],
        seen: dict[str, tuple[str, int]],
        sizes: dict[str, int],
    ) -> None:
        """Native names and units: cell values keep the field name, VTK's interpolated point
        copies become ``<name>__point``; symmetric tensors go back to OpenFOAM component order;
        a field absent at this step is NaN-filled so every step has the same arrays."""
        fields = sample.fields
        for name in [n for n in fields if n.endswith("__cell")]:
            stem = name[: -len("__cell")]
            point = fields.pop(stem)
            fields[stem] = fields.pop(name)
            fields[f"{stem}__point"] = point
        for name, f in fields.items():
            stem = name[: -len("__point")] if name.endswith("__point") else name
            meta = metas.get(stem)
            if meta is not None:
                f.units = meta.units
            comps = f.shape[-1]
            if comps == 6:
                f.data = np.ascontiguousarray(f.data[:, _SYMM_VTK_TO_FOAM])
            f.component_names = COMPONENT_NAMES.get(comps)
            seen[name] = (f.location, comps)
            sizes[f.location] = f.shape[0]
        for name, (location, comps) in seen.items():
            if name not in fields and location in sizes:
                fields[name] = as_field(
                    np.full((sizes[location], comps), np.nan, dtype=np.float32), location
                )
