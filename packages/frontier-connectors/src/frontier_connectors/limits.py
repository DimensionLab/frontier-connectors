# SPDX-License-Identifier: Apache-2.0
"""Bounds on sizes a file announces about itself.

A count or shape read from a header is a claim of whoever wrote the file. Before it sizes an
allocation or a read it is checked against something the file cannot lie about:

* :func:`check_available`: ``count * item_bytes`` must fit in the bytes that are actually
  there (the rest of the file, a section, a parsed buffer);
* :func:`check_dense`: an array that is *synthesised* from a few header numbers (a dense
  matrix from a diagonal, a permutation matrix, an "empty" array with dimensions) has no
  bytes in the file to compare with, so it is capped at :data:`MAX_SYNTHESISED_BYTES`;
* :func:`empty_dims`: the dimensions of an array a file declares empty must multiply to 0.

The limits are module constants so that a deployment can change them in one place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from frontier_connectors.errors import ConnectorError

__all__ = [
    "MAX_DIMENSIONS",
    "MAX_SYNTHESISED_BYTES",
    "SizeLimitError",
    "check_available",
    "check_dense",
    "empty_dims",
]

#: largest array a reader builds from header numbers alone (no matching bytes in the file)
MAX_SYNTHESISED_BYTES = 1 << 30
#: most dimensions an array may announce (numpy's own limit is 64 from 2.0, 32 before)
MAX_DIMENSIONS = 32


class SizeLimitError(ConnectorError):
    """A size announced by a file is impossible or above a limit of this module."""


def _where(path: Any) -> str:
    return f"{path}: " if path else ""


def check_available(
    count: int, item_bytes: int, available: int, *, what: str, path: Any = None
) -> int:
    """``count`` when ``count`` items of ``item_bytes`` fit into ``available`` bytes."""
    if count < 0 or item_bytes < 0 or count * item_bytes > max(available, 0):
        raise SizeLimitError(
            f"{_where(path)}{what} announces {count} entries ({count * item_bytes} bytes) but "
            f"only {max(available, 0)} bytes are there",
            hint="the file is truncated or its header is corrupt",
        )
    return count


def check_dense(
    shape: Sequence[int],
    dtype: Any = np.float64,
    *,
    what: str,
    path: Any = None,
    limit: int | None = None,
) -> tuple[int, ...]:
    """``shape`` as a tuple when a dense array of it stays under the limit
    (default :data:`MAX_SYNTHESISED_BYTES`, read at call time)."""
    cap = MAX_SYNTHESISED_BYTES if limit is None else limit
    dims = tuple(int(s) for s in shape)
    if len(dims) > MAX_DIMENSIONS or any(d < 0 for d in dims):
        raise SizeLimitError(
            f"{_where(path)}{what} announces an impossible shape "
            f"({len(dims)} dimensions{', negative size' if any(d < 0 for d in dims) else ''})",
            hint="the file header is corrupt",
        )
    nbytes = math.prod(dims) * np.dtype(dtype).itemsize
    if nbytes > cap:
        raise SizeLimitError(
            f"{_where(path)}{what} of shape {'x'.join(map(str, dims))} would need {nbytes} "
            f"bytes, above the limit of {cap}",
            hint="the header announces more than the file holds; if the array is real, "
            "export it in full (dense) form or raise frontier_connectors.limits."
            "MAX_SYNTHESISED_BYTES",
        )
    return dims


def empty_dims(dims: Sequence[int], *, what: str, path: Any = None) -> tuple[int, ...]:
    """Dimensions of an array the file marks as empty: they must multiply to zero."""
    out = tuple(int(d) for d in dims)
    if len(out) > MAX_DIMENSIONS or any(d < 0 for d in out) or (out and math.prod(out) != 0):
        raise SizeLimitError(
            f"{_where(path)}{what} is marked empty but announces the shape "
            f"{'x'.join(map(str, out[:MAX_DIMENSIONS]))}",
            hint="the file header is corrupt",
        )
    return out
