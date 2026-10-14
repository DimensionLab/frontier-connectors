# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: UP006, UP007, UP031, UP035, UP037, UP045, B905
# (old-style typing and % formatting on purpose: this file runs under the solver's Python, 3.8+)
"""frontier_emit: write simulation steps into a spool directory that DimensionLab Frontier
follows while the solver runs.

This file is self-contained (Python >= 3.8, numpy only, no Frontier imports): copy it next to
your solver, a Catalyst Python pipeline or a post-processing hook.

.. code-block:: python

    from frontier_emit import Emitter, uniform_mesh

    with Emitter("spool", solver="my-solver", time_units="s") as out:
        for step in range(n_steps):
            ...  # advance the solution
            out.write_step(step, t, uniform_mesh((nx, ny), spacing=(dx, dy), point_fields={"T": T}))

and on the reading side, at the same time::

    python -m frontier_connectors import catalyst spool --out dataset --option follow=true

The ``mesh`` passed to :meth:`Emitter.write_step` is a nested ``dict`` that follows the Conduit
Mesh Blueprint (https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html): the same
tree a Catalyst 2 adaptor puts under ``catalyst/channels/<name>/data``. It is either one domain
(``{"coordsets": ..., "topologies": ..., "fields": ..., "state": ...}``) or several
(``{"fluid": {...}, "wall": {...}}``). Leaves are numpy arrays, numbers or strings. The helpers
:func:`uniform_mesh`, :func:`rectilinear_mesh`, :func:`structured_mesh` and
:func:`unstructured_mesh` build the common cases; :func:`from_conduit` converts a
``conduit.Node``.

Spool protocol ``frontier-spool`` version 1
-------------------------------------------

``frontier-stream.json``
    Stream header, written when the emitter opens: protocol, version, ``run_id``, solver
    name/version, domain, physics, parameters, units, ``time_units``, ``static_mesh``, restarts.
``step_<8 digits>.npz``
    One step. A plain (optionally deflated) numpy ``.npz`` without pickles. Every array leaf of
    the mesh tree is one member whose name is its Blueprint path
    (``coordsets/coords/values/x``, ``fields/T/values`` ...). The member ``__frontier_step__``
    is a JSON document: ``{"protocol", "version", "run_id", "step", "time", "cycle", "prev",
    "strings": {path: text}, "has_mesh": bool}``. ``prev`` is the index of the step emitted
    before this one (``null`` for the first), so that a reader can tell a missing step from a
    sparse numbering.
``frontier-stream.done``
    End marker (JSON: ``status`` = ``complete`` | ``failed``, ``n_steps``, ``last_step``,
    ``message``), written by :meth:`Emitter.close`.

Every file is written under a hidden temporary name (``.<name>.tmp-<pid>``) in the same
directory and moved into place with ``os.replace`` (atomic on POSIX and NTFS), so a file that is
visible under its final name is complete. Readers must ignore names starting with ``.``.
"""

from __future__ import annotations

import io
import json
import os
import re
import time as _time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

PROTOCOL = "frontier-spool"
VERSION = 1
HEADER_NAME = "frontier-stream.json"
DONE_NAME = "frontier-stream.done"
STEP_META = "__frontier_step__"
STEP_PATTERN = re.compile(r"^step_(\d{1,12})\.npz$")

__all__ = [
    "DONE_NAME",
    "HEADER_NAME",
    "PROTOCOL",
    "STEP_META",
    "VERSION",
    "Emitter",
    "flatten",
    "from_conduit",
    "rectilinear_mesh",
    "step_name",
    "structured_mesh",
    "uniform_mesh",
    "unstructured_mesh",
]


def step_name(step: int) -> str:
    return "step_%08d.npz" % int(step)


def flatten(
    tree: Mapping[str, Any], prefix: str = ""
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Split a nested mesh tree into ``({path: array}, {path: string})``."""
    arrays: Dict[str, np.ndarray] = {}
    strings: Dict[str, str] = {}
    for key, value in tree.items():
        key = str(key)
        if not key or key.startswith("__"):
            raise ValueError("bad key %r under %r" % (key, prefix))
        path = prefix + key.strip("/")
        if isinstance(value, Mapping):
            a, s = flatten(value, path + "/")
            arrays.update(a)
            strings.update(s)
        elif isinstance(value, (str, bytes)):
            strings[path] = value.decode() if isinstance(value, bytes) else value
        elif value is None:
            continue
        else:
            arr = np.asarray(value)
            if arr.dtype.kind not in "fiub":
                raise TypeError("%s: unsupported leaf of dtype %s" % (path, arr.dtype))
            arrays[path] = arr
    return arrays, strings


def from_conduit(node: Any) -> Any:
    """Convert a ``conduit.Node`` into the nested ``dict`` form (arrays are copied)."""
    names = list(node.child_names()) if hasattr(node, "child_names") else []
    if names:
        return {name: from_conduit(node.fetch(name)) for name in names}
    count = int(node.number_of_children()) if hasattr(node, "number_of_children") else 0
    if count:  # list node, e.g. unnamed domains
        return {"domain_%06d" % i: from_conduit(node.child(i)) for i in range(count)}
    value = node.value() if hasattr(node, "value") else node
    return np.array(value, copy=True) if isinstance(value, np.ndarray) else value


def _atomic_write(path: Path, data: bytes, fsync: bool) -> None:
    tmp = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    with open(tmp, "wb") as f:
        f.write(data)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


class Emitter:
    """Writes the spool protocol described in the module docstring.

    ``static_mesh=True`` stores ``coordsets``/``topologies`` only with the first step (and the
    first step after a restart); the reader keeps the last mesh it saw.

    A spool that already holds steps is refused unless ``overwrite=True`` (start over) or
    ``restart_from=k`` is given: the solver restarted from a checkpoint taken at step ``k``, so
    the steps after ``k`` and the end marker are removed and the stream continues.
    """

    def __init__(
        self,
        spool: "str | os.PathLike[str]",
        *,
        stream_id: Optional[str] = None,
        solver: str = "unknown",
        solver_version: Optional[str] = None,
        domain: str = "other",
        physics: Optional[Mapping[str, Any]] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        units: Optional[Mapping[str, str]] = None,
        time_units: str = "s",
        static_mesh: bool = False,
        overwrite: bool = False,
        restart_from: Optional[int] = None,
        compress: bool = False,
        fsync: bool = False,
    ) -> None:
        self.spool = Path(spool)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.static_mesh = bool(static_mesh)
        self.compress = bool(compress)
        self.fsync = bool(fsync)
        self.run_id = uuid.uuid4().hex
        self.n_written = 0
        self.closed = False
        self._mesh_sent = False
        existing = self._existing()
        restarts: list = []
        old: Dict[str, Any] = {}
        if (self.spool / HEADER_NAME).exists():
            try:
                old = json.loads((self.spool / HEADER_NAME).read_text())
            except (OSError, ValueError):
                old = {}
        if overwrite:
            for idx in existing:
                (self.spool / step_name(idx)).unlink()
            existing = []
            self._remove_done()
        elif restart_from is not None:
            for idx in [i for i in existing if i > int(restart_from)]:
                (self.spool / step_name(idx)).unlink()
            existing = [i for i in existing if i <= int(restart_from)]
            self._remove_done()
            restarts = list(old.get("restarts", []))
            restarts.append(
                {"from_step": int(restart_from), "run_id": self.run_id, "at": _time.time()}
            )
        elif existing:
            raise FileExistsError(
                "%s already holds %d step(s): pass overwrite=True to start over or "
                "restart_from=<step> to continue after a solver restart"
                % (self.spool, len(existing))
            )
        else:
            self._remove_done()
        self.prev: Optional[int] = existing[-1] if existing else None
        self.header: Dict[str, Any] = {
            "protocol": PROTOCOL,
            "version": VERSION,
            "stream_id": stream_id or old.get("stream_id") or self.spool.resolve().name,
            "run_id": self.run_id,
            "created": _time.time(),
            "solver": solver,
            "solver_version": solver_version,
            "domain": domain,
            "physics": dict(physics or {}),
            "parameters": dict(parameters or {}),
            "units": dict(units or {}),
            "time_units": time_units,
            "static_mesh": self.static_mesh,
            "restarts": restarts,
        }
        self._json(HEADER_NAME, self.header)

    # -- helpers ---------------------------------------------------------------------------
    def _existing(self) -> list:
        found = []
        for name in os.listdir(self.spool):
            m = STEP_PATTERN.match(name)
            if m:
                found.append(int(m.group(1)))
        return sorted(found)

    def _remove_done(self) -> None:
        try:
            (self.spool / DONE_NAME).unlink()
        except FileNotFoundError:
            pass

    def _json(self, name: str, doc: Mapping[str, Any]) -> None:
        _atomic_write(self.spool / name, json.dumps(doc, indent=1).encode(), self.fsync)

    # -- API -------------------------------------------------------------------------------
    def write_step(
        self, step: int, time: float, mesh: Mapping[str, Any], *, cycle: Optional[int] = None
    ) -> Path:
        """Write one step atomically and return its path. ``step`` must increase."""
        if self.closed:
            raise RuntimeError("the emitter is closed")
        step = int(step)
        if step < 0:
            raise ValueError("step must be >= 0")
        if self.prev is not None and step <= self.prev:
            raise ValueError("step %d is not after the previous step %d" % (step, self.prev))
        tree: Mapping[str, Any] = mesh
        has_mesh = True
        if self.static_mesh and self._mesh_sent:
            tree = _without_mesh(mesh)
            has_mesh = False
        arrays, strings = flatten(tree)
        meta = {
            "protocol": PROTOCOL,
            "version": VERSION,
            "run_id": self.run_id,
            "step": step,
            "time": float(time),
            "cycle": int(step if cycle is None else cycle),
            "prev": self.prev,
            "has_mesh": has_mesh,
            "strings": strings,
        }
        buf = io.BytesIO()
        mode = zipfile.ZIP_DEFLATED if self.compress else zipfile.ZIP_STORED
        with zipfile.ZipFile(buf, "w", mode, allowZip64=True) as zf:
            _put(zf, STEP_META, np.array(json.dumps(meta)))
            for path, arr in arrays.items():
                _put(zf, path, np.ascontiguousarray(arr))
        dest = self.spool / step_name(step)
        _atomic_write(dest, buf.getvalue(), self.fsync)
        self.prev = step
        self.n_written += 1
        self._mesh_sent = True
        return dest

    def close(self, status: str = "complete", message: str = "") -> None:
        """Write the end marker. ``status`` is ``complete`` or ``failed``."""
        if self.closed:
            return
        self.closed = True
        self._json(
            DONE_NAME,
            {
                "protocol": PROTOCOL,
                "version": VERSION,
                "run_id": self.run_id,
                "status": status,
                "message": message,
                "n_steps": len(self._existing()),
                "last_step": self.prev,
                "finished": _time.time(),
            },
        )

    def __enter__(self) -> "Emitter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.close("failed", "%s: %s" % (getattr(exc_type, "__name__", "error"), exc))


def _put(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    zf.writestr(name + ".npy", buf.getvalue())


def _without_mesh(mesh: Mapping[str, Any]) -> Dict[str, Any]:
    if "coordsets" in mesh or "topologies" in mesh or "fields" in mesh:
        return {k: v for k, v in mesh.items() if k not in ("coordsets", "topologies")}
    return {k: _without_mesh(v) if isinstance(v, Mapping) else v for k, v in mesh.items()}


# -- mesh builders ----------------------------------------------------------------------------

_AXES = ("x", "y", "z")


def _fields(
    point_fields: Optional[Mapping[str, Any]],
    cell_fields: Optional[Mapping[str, Any]],
    units: Optional[Mapping[str, str]],
    topology: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for assoc, group in (("vertex", point_fields), ("element", cell_fields)):
        for name, values in (group or {}).items():
            if name in out:
                raise ValueError("field %r is given as point and as cell field" % name)
            if isinstance(values, Mapping):
                vals: Any = {str(k): np.asarray(v).reshape(-1) for k, v in values.items()}
            else:
                vals = np.asarray(values)
            f: Dict[str, Any] = {"association": assoc, "topology": topology, "values": vals}
            if units and name in units:
                f["units"] = units[name]
            out[name] = f
    return out


def uniform_mesh(
    dims: Sequence[int],
    *,
    origin: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    point_fields: Optional[Mapping[str, Any]] = None,
    cell_fields: Optional[Mapping[str, Any]] = None,
    units: Optional[Mapping[str, str]] = None,
    topology: str = "mesh",
) -> Dict[str, Any]:
    """Uniform grid with ``dims`` points per axis (2-D or 3-D).

    Field arrays are flat with the first axis varying fastest (Blueprint order), or shaped
    ``(nx, ny[, nz])`` / ``(nx, ny[, nz], C)`` in which case they are reordered here.
    """
    dims = [int(d) for d in dims]
    cs: Dict[str, Any] = {"type": "uniform", "dims": dict(zip("ijk", dims))}
    if origin is not None:
        cs["origin"] = {a: float(v) for a, v in zip(_AXES, origin)}
    if spacing is not None:
        cs["spacing"] = {"d" + a: float(v) for a, v in zip(_AXES, spacing)}
    cdims = [max(d - 1, 1) for d in dims]
    return {
        "coordsets": {"coords": cs},
        "topologies": {topology: {"type": "uniform", "coordset": "coords"}},
        "fields": _fields(
            {k: _grid_values(v, dims) for k, v in (point_fields or {}).items()},
            {k: _grid_values(v, cdims) for k, v in (cell_fields or {}).items()},
            units,
            topology,
        ),
    }


def rectilinear_mesh(
    axes: Sequence[Any],
    *,
    point_fields: Optional[Mapping[str, Any]] = None,
    cell_fields: Optional[Mapping[str, Any]] = None,
    units: Optional[Mapping[str, str]] = None,
    topology: str = "mesh",
) -> Dict[str, Any]:
    """Rectilinear grid from 1-D coordinate axes ``(x, y[, z])``."""
    coords = [np.asarray(a, dtype=np.float64).reshape(-1) for a in axes]
    dims = [len(c) for c in coords]
    cdims = [max(d - 1, 1) for d in dims]
    return {
        "coordsets": {"coords": {"type": "rectilinear", "values": dict(zip(_AXES, coords))}},
        "topologies": {topology: {"type": "rectilinear", "coordset": "coords"}},
        "fields": _fields(
            {k: _grid_values(v, dims) for k, v in (point_fields or {}).items()},
            {k: _grid_values(v, cdims) for k, v in (cell_fields or {}).items()},
            units,
            topology,
        ),
    }


def structured_mesh(
    points: Any,
    dims: Sequence[int],
    *,
    point_fields: Optional[Mapping[str, Any]] = None,
    cell_fields: Optional[Mapping[str, Any]] = None,
    units: Optional[Mapping[str, str]] = None,
    topology: str = "mesh",
) -> Dict[str, Any]:
    """Curvilinear grid: ``points`` ``(N, 2|3)`` ordered first-axis-fastest, ``dims`` points
    per axis."""
    pts = np.asarray(points, dtype=np.float64)
    dims = [int(d) for d in dims]
    return {
        "coordsets": {
            "coords": {
                "type": "explicit",
                "values": {a: pts[:, i].copy() for i, a in enumerate(_AXES[: pts.shape[1]])},
            }
        },
        "topologies": {
            topology: {
                "type": "structured",
                "coordset": "coords",
                "elements": {"dims": dict(zip("ijk", [d - 1 for d in dims]))},
            }
        },
        "fields": _fields(point_fields, cell_fields, units, topology),
    }


_SHAPE_NODES = {
    "point": 1,
    "line": 2,
    "tri": 3,
    "quad": 4,
    "tet": 4,
    "hex": 8,
    "wedge": 6,
    "pyramid": 5,
}


def unstructured_mesh(
    points: Any,
    cells: "Any | Iterable[Tuple[str, Any]]",
    shape: Optional[str] = None,
    *,
    point_fields: Optional[Mapping[str, Any]] = None,
    cell_fields: Optional[Mapping[str, Any]] = None,
    units: Optional[Mapping[str, str]] = None,
    topology: str = "mesh",
) -> Dict[str, Any]:
    """Unstructured mesh.

    Single shape: ``cells`` is a ``(K, P)`` (or flat) connectivity and ``shape`` one of
    ``point line tri quad tet hex wedge pyramid`` (VTK node order). Mixed: ``cells`` is a list
    of ``(shape, connectivity)`` blocks and ``shape`` is omitted; cell fields follow the block
    order.
    """
    pts = np.asarray(points, dtype=np.float64)
    elements: Dict[str, Any]
    if shape is not None:
        if shape not in _SHAPE_NODES:
            raise ValueError("unknown shape %r" % shape)
        elements = {"shape": shape, "connectivity": np.asarray(cells, dtype=np.int64).reshape(-1)}
    else:
        blocks = [(str(s), np.asarray(c, dtype=np.int64)) for s, c in cells]  # type: ignore[union-attr]
        names = sorted({s for s, _ in blocks})
        vtk_ids = {
            "point": 1,
            "line": 3,
            "tri": 5,
            "quad": 9,
            "tet": 10,
            "hex": 12,
            "wedge": 13,
            "pyramid": 14,
        }
        conn, shapes, sizes = [], [], []
        for s, c in blocks:
            n = _SHAPE_NODES[s]
            c = c.reshape(-1, n)
            conn.append(c.reshape(-1))
            shapes.append(np.full(len(c), vtk_ids[s], dtype=np.int32))
            sizes.append(np.full(len(c), n, dtype=np.int64))
        size_arr = np.concatenate(sizes)
        elements = {
            "shape": "mixed",
            "shape_map": {s: vtk_ids[s] for s in names},
            "shapes": np.concatenate(shapes),
            "sizes": size_arr,
            "offsets": np.concatenate([[0], np.cumsum(size_arr)[:-1]]).astype(np.int64),
            "connectivity": np.concatenate(conn),
        }
    return {
        "coordsets": {
            "coords": {
                "type": "explicit",
                "values": {a: pts[:, i].copy() for i, a in enumerate(_AXES[: pts.shape[1]])},
            }
        },
        "topologies": {
            topology: {"type": "unstructured", "coordset": "coords", "elements": elements}
        },
        "fields": _fields(point_fields, cell_fields, units, topology),
    }


def _grid_values(values: Any, dims: Sequence[int]) -> Any:
    """Bring a grid field into Blueprint order (first axis fastest). Accepts flat arrays (taken
    as already ordered), ``dims``-shaped arrays and ``(*dims, C)`` arrays."""
    if isinstance(values, Mapping):
        return {k: _grid_values(v, dims) for k, v in values.items()}
    a = np.asarray(values)
    dims = tuple(int(d) for d in dims)
    n = int(np.prod(dims))
    if a.shape == dims:
        return a.reshape(-1, order="F")
    if a.ndim == len(dims) + 1 and a.shape[:-1] == dims:
        return a.reshape(n, a.shape[-1], order="F")
    if a.shape[0] != n:
        raise ValueError("grid field of shape %s does not fit %s points" % (a.shape, dims))
    return a
