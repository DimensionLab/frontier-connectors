# SPDX-License-Identifier: Apache-2.0
"""Zone / field / time selection shared by connectors."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from frontier_connectors.errors import SelectionError
from frontier_connectors.model import TimeSelection


def match_names(
    available: Iterable[str], patterns: Sequence[str] | None, *, what: str = "zone"
) -> list[str]:
    """Names from ``available`` matching any of ``patterns`` (exact names or ``fnmatch`` globs,
    ``!pattern`` excludes), in source order. ``None`` selects everything. Raises
    :class:`SelectionError` naming the available choices when a pattern matches nothing."""
    names = list(available)
    if patterns is None:
        return names
    if isinstance(patterns, str):
        patterns = [patterns]
    include = [p for p in patterns if not p.startswith("!")]
    exclude = [p[1:] for p in patterns if p.startswith("!")]
    for p in include:
        if not any(n == p or fnmatch.fnmatchcase(n, p) for n in names):
            shown = ", ".join(names[:30]) + (" ..." if len(names) > 30 else "")
            raise SelectionError(
                f"{what} {p!r} not found", hint=f"available {what}s: {shown or '(none)'}"
            )
    out = []
    for n in names:
        if include and not any(n == p or fnmatch.fnmatchcase(n, p) for p in include):
            continue
        if any(n == p or fnmatch.fnmatchcase(n, p) for p in exclude):
            continue
        out.append(n)
    return out


def wants(name: str, patterns: Sequence[str] | None) -> bool:
    """``True`` when ``name`` passes ``patterns`` (no error for unmatched patterns)."""
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    include = [p for p in patterns if not p.startswith("!")]
    exclude = [p[1:] for p in patterns if p.startswith("!")]
    if include and not any(name == p or fnmatch.fnmatchcase(name, p) for p in include):
        return False
    return not any(name == p or fnmatch.fnmatchcase(name, p) for p in exclude)


def select_time_indices(times: Sequence[float] | np.ndarray, selection: TimeSelection) -> list[int]:
    """Resolve a :data:`TimeSelection` against the stored ``times`` into sorted step indices.

    A steady source (``len(times) == 0``) resolves to ``[0]`` so callers can always loop.
    """
    n = len(times)
    if n == 0:
        return [0]
    if selection is None:
        return list(range(n))
    if isinstance(selection, str):
        key = selection.strip().lower()
        if key in ("all", "*", ""):
            return list(range(n))
        if key == "last":
            return [n - 1]
        if key == "first":
            return [0]
        return select_time_indices(times, parse_time_selection(selection))
    if isinstance(selection, bool):
        raise SelectionError("time selection cannot be a bool")
    if isinstance(selection, int | np.integer):
        i = int(selection)
        if not -n <= i < n:
            raise SelectionError(
                f"time index {i} out of range", hint=f"the source has {n} steps (0..{n - 1})"
            )
        return [i % n]
    if isinstance(selection, slice):
        return list(range(n))[selection]
    items = list(selection)
    if not items:
        raise SelectionError("empty time selection", hint="omit --times to read every step")
    if all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in items):
        out = []
        for v in items:
            out.extend(select_time_indices(times, int(v)))
        return sorted(set(out))
    arr = np.asarray(times, dtype=np.float64)
    out = []
    span = float(arr.max() - arr.min()) if n > 1 else max(abs(float(arr[0])), 1.0)
    for v in items:
        i = int(np.argmin(np.abs(arr - float(v))))
        if abs(arr[i] - float(v)) > 0.5 * max(span, 1e-30):
            raise SelectionError(
                f"time {v} is far from every stored step",
                hint=f"stored times span {arr.min():g} .. {arr.max():g}",
            )
        out.append(i)
    return sorted(set(out))


_SLICE = re.compile(r"^(-?\d*):(-?\d*)(?::(-?\d*))?$")


def parse_time_selection(text: str | None) -> Any:
    """Parse the CLI form: ``all`` | ``first`` | ``last`` | ``5`` | ``0:100:10`` | ``1,5,9`` |
    ``t=0.1,0.2`` (physical times, nearest step)."""
    if text is None:
        return None
    s = text.strip()
    if s.lower() in ("all", "*", ""):
        return None
    if s.lower() in ("first", "last"):
        return s.lower()
    if s.lower().startswith("t="):
        try:
            return [float(v) for v in s[2:].split(",") if v.strip()]
        except ValueError as e:
            raise SelectionError(f"bad time list {text!r}", hint="use t=0.1,0.2") from e
    m = _SLICE.match(s)
    if m:
        a, b, c = (int(g) if g not in (None, "") else None for g in m.groups())
        return slice(a, b, c)
    try:
        vals = [int(v) for v in s.split(",") if v.strip()]
    except ValueError as e:
        raise SelectionError(
            f"bad time selection {text!r}",
            hint="use all | first | last | <index> | start:stop:step | i,j,k | t=<time>,<time>",
        ) from e
    return vals[0] if len(vals) == 1 else vals
