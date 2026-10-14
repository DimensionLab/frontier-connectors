# SPDX-License-Identifier: Apache-2.0
"""Numpy-only mesh helpers shared by connectors (cell padding, triangulation, VTK type ids)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# VTK linear cell type ids (the vocabulary of ``mesh.zarr/cell_types``)
VTK_VERTEX = 1
VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_POLYGON = 7
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14
VTK_QUADRATIC_EDGE = 21
VTK_QUADRATIC_TRIANGLE = 22
VTK_QUADRATIC_QUAD = 23
VTK_QUADRATIC_TETRA = 24
VTK_QUADRATIC_HEXAHEDRON = 25
VTK_QUADRATIC_WEDGE = 26
VTK_QUADRATIC_PYRAMID = 27
VTK_POLYHEDRON = 42

VTK_TYPE_NAMES: dict[int, str] = {
    1: "vertex",
    2: "poly_vertex",
    3: "line",
    4: "poly_line",
    5: "triangle",
    6: "triangle_strip",
    7: "polygon",
    8: "pixel",
    9: "quad",
    10: "tetra",
    11: "voxel",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
    15: "pentagonal_prism",
    16: "hexagonal_prism",
    21: "quadratic_edge",
    22: "quadratic_triangle",
    23: "quadratic_quad",
    24: "quadratic_tetra",
    25: "quadratic_hexahedron",
    26: "quadratic_wedge",
    27: "quadratic_pyramid",
    28: "biquadratic_quad",
    29: "triquadratic_hexahedron",
    30: "quadratic_linear_quad",
    31: "quadratic_linear_wedge",
    32: "biquadratic_quadratic_wedge",
    33: "biquadratic_quadratic_hexahedron",
    34: "biquadratic_triangle",
    35: "cubic_line",
    36: "quadratic_polygon",
    37: "triquadratic_pyramid",
    41: "convex_point_set",
    42: "polyhedron",
}

#: topological dimension per VTK type
VTK_TYPE_DIM: dict[int, int] = {
    1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
    15: 3, 16: 3, 21: 1, 22: 2, 23: 2, 24: 3, 25: 3, 26: 3, 27: 3, 28: 2, 29: 3, 30: 2,
    31: 3, 32: 3, 33: 3, 34: 2, 35: 1, 36: 2, 37: 3, 41: 3, 42: 3,
}  # fmt: skip

#: corner-node count of the quadratic/2-D types (used to take the linear skeleton)
_CORNERS = {22: 3, 23: 4, 28: 4, 5: 3, 9: 4}


def census(cell_types: np.ndarray) -> dict[str, int]:
    """Element census ``{"tetra": 120, ...}`` of a ``cell_types`` array."""
    ids, counts = np.unique(np.asarray(cell_types), return_counts=True)
    return {
        VTK_TYPE_NAMES.get(int(i), f"vtk_{int(i)}"): int(c)
        for i, c in zip(ids, counts, strict=True)
    }


def pad_cells(connectivity: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """CSR connectivity (``offsets`` of length K+1) -> ``(K, P)`` int64 matrix padded with -1."""
    offsets = np.asarray(offsets, dtype=np.int64)
    conn = np.asarray(connectivity, dtype=np.int64)
    k = len(offsets) - 1
    if k <= 0:
        return np.full((0, 1), -1, dtype=np.int64)
    sizes = np.diff(offsets)
    width = int(sizes.max()) if k else 1
    out = np.full((k, max(width, 1)), -1, dtype=np.int64)
    if (sizes == width).all():
        return conn[offsets[0] : offsets[-1]].reshape(k, width)
    cols = np.arange(width)[None, :]
    mask = cols < sizes[:, None]
    out[mask] = conn[offsets[0] : offsets[-1]]
    return out


def pad_cell_blocks(blocks: Sequence[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``[(vtk_type, (k, p) connectivity), ...]`` into padded ``cells`` and ``cell_types``."""
    blocks = [(t, np.asarray(c, dtype=np.int64).reshape(len(c), -1)) for t, c in blocks if len(c)]
    if not blocks:
        return np.full((0, 1), -1, dtype=np.int64), np.zeros((0,), dtype=np.uint8)
    width = max(c.shape[1] for _, c in blocks)
    total = sum(len(c) for _, c in blocks)
    cells = np.full((total, width), -1, dtype=np.int64)
    types = np.empty((total,), dtype=np.uint8)
    row = 0
    for t, c in blocks:
        cells[row : row + len(c), : c.shape[1]] = c
        types[row : row + len(c)] = t
        row += len(c)
    return cells, types


def triangulate(
    cells: np.ndarray, cell_types: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate padded 2-D cells (triangles, quads, polygons, quadratic tri/quad).

    Returns ``(faces (M, 3) int64, parent (M,) int64)`` where ``parent`` maps every triangle to
    the cell it came from, so cell data can be carried over with ``data[parent]``.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2:
        raise ValueError("cells must be a padded (K, P) matrix")
    k = cells.shape[0]
    sizes = (cells >= 0).sum(axis=1)
    if cell_types is not None:
        ct = np.asarray(cell_types)
        for t, n in _CORNERS.items():
            sizes = np.where(ct == t, np.minimum(sizes, n), sizes)
        if np.any(ct == 8):  # VTK_PIXEL: swap the last two nodes to get a ring
            cells = cells.copy()
            pix = ct == 8
            cells[pix, 2], cells[pix, 3] = cells[pix, 3].copy(), cells[pix, 2].copy()
    faces, parents = [], []
    for j in range(1, int(sizes.max(initial=0)) - 1):
        sel = np.nonzero(sizes > j + 1)[0]
        if len(sel) == 0:
            break
        faces.append(np.stack([cells[sel, 0], cells[sel, j], cells[sel, j + 1]], axis=1))
        parents.append(sel)
    if not faces:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    f = np.concatenate(faces)
    p = np.concatenate(parents)
    order = np.argsort(p, kind="stable")
    del k
    return f[order], p[order]


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area per triangle (float64). 2-D points are treated as z = 0."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[1] == 2:
        p = np.concatenate([p, np.zeros((len(p), 1))], axis=1)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.zeros((0,), dtype=np.float64)
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def bounds(points: np.ndarray | None) -> list[float]:
    """``[xmin, xmax, ymin, ymax, zmin, zmax]`` (zeros for missing axes / empty input)."""
    out = [0.0] * 6
    if points is None or len(points) == 0:
        return out
    p = np.asarray(points)
    for d in range(min(p.shape[1], 3)):
        out[2 * d] = float(np.nanmin(p[:, d]))
        out[2 * d + 1] = float(np.nanmax(p[:, d]))
    return out


def compact_points(
    points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop points not referenced by ``cells`` (padded with -1). Returns ``(points, cells,
    used)`` where ``used`` are the kept original point indices (for remapping point data)."""
    cells = np.asarray(cells, dtype=np.int64)
    valid = cells >= 0
    used = np.unique(cells[valid])
    remap = np.full((len(points),), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = np.full_like(cells, -1)
    out[valid] = remap[cells[valid]]
    return np.asarray(points)[used], out, used


def all_surface(cell_types: np.ndarray) -> bool:
    """``True`` when every cell is 2-D (so the zone is a surface, not a volume)."""
    ct = np.unique(np.asarray(cell_types))
    return len(ct) > 0 and all(VTK_TYPE_DIM.get(int(t), 3) == 2 for t in ct)


def sanitize_id(text: str) -> str:
    """Filesystem/URL-safe id: keeps ``[A-Za-z0-9._-]``, everything else becomes ``_``."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return s or "sample"


def default_sample_id(path: Path | str) -> str:
    """Sample id derived from a source path (file stem without compound suffixes, or dir name)."""
    p = Path(path)
    name = p.name
    for suffix in (".cas.h5", ".dat.h5", ".cas.gz", ".dat.gz", ".msh.h5", ".msh.gz", ".tar.gz"):
        if name.lower().endswith(suffix):
            return sanitize_id(name[: -len(suffix)])
    return sanitize_id(p.stem if p.is_file() or p.suffix else name)


def zone_sample_id(base: str, zone: str | None, n_zones: int) -> str:
    """``base`` when the source has a single selected zone, else ``base__zone``."""
    if zone is None or n_zones <= 1:
        return base
    return f"{base}__{sanitize_id(zone)}"
