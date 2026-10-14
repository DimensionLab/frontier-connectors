# SPDX-License-Identifier: Apache-2.0
"""Reader for Conduit relay Blueprint files (``<name>.cycle_<n>.root`` + domain files) that does
not need Conduit.

``conduit.relay.io.blueprint.save_mesh(mesh, path, protocol)`` (also used by Ascent's relay
extract and usable from a Catalyst Python pipeline) writes a *root* file holding
``blueprint_index``, ``protocol/name``, ``number_of_files``, ``number_of_trees``,
``file_pattern`` and ``tree_pattern``, and either the mesh itself (one file) or a directory
``<name>.cycle_<n>/domain_<i>.<ext>``. The layout below was read off files written by Conduit
0.9.5 on 2026-10-10 (protocols ``hdf5``, ``json``, ``yaml``, ``conduit_json``,
``conduit_base64_json``, ``conduit_bin``); documentation:
https://llnl-conduit.readthedocs.io/en/latest/relay_io.html and
https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#mesh-index-protocol.

* ``hdf5``: groups are objects, datasets are leaves, strings are ``|S1`` character arrays
  (read with ``h5py``, BSD-3, optional);
* ``json`` / ``yaml``: plain values (``yaml`` needs PyYAML, MIT, optional);
* ``conduit_json``: every leaf is ``{"dtype", "number_of_elements", ..., "value"}``;
* ``conduit_base64_json``: ``{"schema": ..., "data": {"base64": ...}}``;
* ``conduit_bin``: raw bytes in the file, the schema next to it in ``<file>_json``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors.errors import (
    ConnectorError,
    MissingDependencyError,
    UnsupportedSourceError,
)
from frontier_connectors.paths import h5_confine_external_links, resolve_in_case

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ROOT_SERIES = re.compile(r"^(?P<base>.+?)\.cycle_(?P<cycle>\d+)\.root$")

_DTYPES = {
    "int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
    "uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8",
    "float32": "f4", "float64": "f8",
}  # fmt: skip


class IncompleteFileError(Exception):
    """The file is (still) unreadable: truncated, half written or a referenced file is missing."""


def _is_schema_leaf(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and isinstance(node.get("dtype"), str)
        and ("value" in node or "number_of_elements" in node or "offset" in node)
    )


def _np_dtype(name: str, endianness: str | None) -> np.dtype:
    if name not in _DTYPES:
        raise UnsupportedSourceError(f"conduit dtype {name!r} is not supported")
    order = ">" if endianness == "big" else "<"
    return np.dtype(order + _DTYPES[name])


def _from_schema(schema: Any, buf: bytes | None) -> Any:
    """Decode a Conduit schema (``conduit_json`` values or offsets into ``buf``)."""
    if _is_schema_leaf(schema):
        dtype = schema["dtype"]
        n = int(schema.get("number_of_elements", 1))
        if dtype == "empty":
            return None
        if "value" in schema:
            v = schema["value"]
            if dtype == "char8_str":
                return str(v)
            return np.asarray(v, dtype=_np_dtype(dtype, schema.get("endianness"))).reshape(-1)
        if buf is None:
            raise UnsupportedSourceError("conduit schema without data")
        off = int(schema.get("offset", 0))
        if dtype == "char8_str":
            return bytes(buf[off : off + n]).split(b"\x00", 1)[0].decode("utf-8", "replace")
        dt = _np_dtype(dtype, schema.get("endianness"))
        stride = int(schema.get("stride", dt.itemsize))
        if off + (n - 1) * stride + dt.itemsize > len(buf):
            raise IncompleteFileError("conduit data buffer is shorter than its schema")
        arr = np.ndarray((n,), dtype=dt, buffer=buf, offset=off, strides=(stride,))
        return np.array(arr, dtype=dt.newbyteorder("="), copy=True)
    if isinstance(schema, dict):
        return {str(k): _from_schema(v, buf) for k, v in schema.items()}
    if isinstance(schema, list):
        return {f"domain_{i:06d}": _from_schema(v, buf) for i, v in enumerate(schema)}
    return _plain(schema)


def _plain(node: Any) -> Any:
    """Plain JSON / YAML values -> dict / ndarray / str."""
    if isinstance(node, dict):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, list):
        if node and all(isinstance(v, dict) for v in node):
            return {f"domain_{i:06d}": _plain(v) for i, v in enumerate(node)}
        try:
            arr = np.asarray(node)
        except ValueError as e:
            raise UnsupportedSourceError(f"ragged list in a relay file: {e}") from e
        if arr.dtype.kind not in "fiub":
            raise UnsupportedSourceError("non-numeric list in a relay file")
        return arr
    return node


def _h5_tree(group: Any) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            out[name] = _h5_tree(item)
            continue
        if item.shape is None:  # empty dataset = empty conduit node
            out[name] = None
            continue
        value = item[()]
        if isinstance(value, bytes):
            out[name] = value.decode("utf-8", "replace").rstrip("\x00")
        elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
            out[name] = (
                b"".join(value.reshape(-1).tolist()).decode("utf-8", "replace").rstrip("\x00")
            )
        elif isinstance(value, str):
            out[name] = value
        else:
            out[name] = np.asarray(value)
    return out


def load_file(path: Path) -> dict[str, Any]:
    """Decode one relay file of any supported protocol into a plain tree."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as e:
        raise IncompleteFileError(str(e)) from e
    if head == HDF5_MAGIC:
        try:
            import h5py
        except ImportError as e:
            raise MissingDependencyError(
                "h5py", purpose="read Conduit relay HDF5 files", license="BSD-3-Clause"
            ) from e
        try:
            with h5py.File(path, "r") as f:
                h5_confine_external_links(f, path=path)
                return _h5_tree(f)
        except (OSError, KeyError, RuntimeError) as e:
            raise IncompleteFileError(f"{path.name}: {e}") from e
    sidecar = path.with_name(path.name + "_json")
    if sidecar.exists():  # conduit_bin
        try:
            schema = json.loads(sidecar.read_text())
            return _from_schema(schema, path.read_bytes())
        except (OSError, ValueError) as e:
            raise IncompleteFileError(f"{path.name}: {e}") from e
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise IncompleteFileError(str(e)) from e
    if not raw.strip():
        raise IncompleteFileError(f"{path.name} is empty")
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise UnsupportedSourceError(
            f"{path} is not a Conduit relay file (binary, not HDF5)",
            hint="supported relay protocols: hdf5, json, yaml, conduit_json, "
            "conduit_base64_json, conduit_bin (needs the <file>_json schema next to it)",
        ) from e
    if txt.lstrip().startswith("{"):
        try:
            doc = json.loads(txt)
        except ValueError as e:
            raise IncompleteFileError(f"{path.name}: {e}") from e
        if isinstance(doc, dict) and set(doc) == {"schema", "data"}:
            data = doc["data"]
            if not isinstance(data, dict) or "base64" not in data:
                raise UnsupportedSourceError(f"{path}: unknown conduit data encoding")
            return _from_schema(doc["schema"], base64.b64decode(data["base64"]))
        if _has_schema_leaves(doc):
            return _from_schema(doc, None)
        return _plain(doc)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise MissingDependencyError(
            "pyyaml", purpose="read Conduit relay YAML files", license="MIT"
        ) from e
    try:
        doc = yaml.safe_load(txt)
    except yaml.YAMLError as e:
        raise IncompleteFileError(f"{path.name}: {e}") from e
    if not isinstance(doc, dict):
        raise IncompleteFileError(f"{path.name}: not a YAML mapping (yet)")
    return _plain(doc)


def _has_schema_leaves(doc: Any) -> bool:
    if _is_schema_leaf(doc):
        return True
    if isinstance(doc, dict):
        return any(_has_schema_leaves(v) for v in doc.values())
    return False


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v
    raise ConnectorError(f"relay root: expected a string, got {type(v).__name__}")


def _int(v: Any, default: int) -> int:
    if v is None:
        return default
    return int(np.asarray(v).reshape(-1)[0])


def _walk(tree: dict[str, Any], path: str) -> Any:
    cur: Any = tree
    for part in [p for p in path.split("/") if p]:
        if not isinstance(cur, dict) or part not in cur:
            raise IncompleteFileError(f"path {path!r} not found in a relay file")
        cur = cur[part]
    return cur


def _format(pattern: str, i: int) -> str:
    return pattern % i if "%" in pattern else pattern


def is_blueprint_root(path: Path) -> bool:
    """Cheap check that a ``.root`` file is a Conduit Blueprint root (and not e.g. a CERN ROOT
    file): looks for the index keys without decoding the arrays."""
    try:
        with open(path, "rb") as f:
            head = f.read(1 << 16)
    except OSError:
        return False
    if head[:8] == HDF5_MAGIC:
        try:
            import h5py

            with h5py.File(path, "r") as f:
                return "blueprint_index" in f
        except Exception:  # noqa: BLE001 - not ours, or h5py missing: fall back to a byte scan
            return b"blueprint_index" in head
    if b"blueprint_index" in head or path.with_name(path.name + "_json").exists():
        return True
    try:
        return b"blueprint_index" in path.read_bytes()
    except OSError:
        return False


def root_files(root: Path) -> list[Path]:
    """Files a root refers to (best effort, for completeness checks); raises
    :class:`IncompleteFileError` when the root itself cannot be decoded yet."""
    doc = load_file(root)
    if not isinstance(doc, dict) or "file_pattern" not in doc:
        return [root]
    pattern = _text(doc["file_pattern"])
    n = _int(doc.get("number_of_files"), 1)
    return [
        resolve_in_case(
            root.parent.resolve(), _format(pattern, i), what="file_pattern", referrer=root
        )
        for i in range(n if "%" in pattern else 1)
    ]


def load_root(root: Path) -> dict[str, Any]:
    """Load a Blueprint root file and everything it refers to into one mesh tree.

    One tree gives ``{<mesh name>: domain}``; several give
    ``{"domain_000000": domain, ...}`` (prefixed with the mesh name when the file holds more
    than one mesh). Raises :class:`IncompleteFileError` while files are missing or truncated.
    """
    root = Path(root)
    doc = load_file(root)
    if not isinstance(doc, dict):
        raise UnsupportedSourceError(f"{root} is not a Conduit relay root file")
    index = doc.get("blueprint_index")
    if not isinstance(index, dict) or not index:
        if any(k in doc for k in ("coordsets", "topologies")):
            return doc  # a bare mesh saved with relay.io.save
        raise UnsupportedSourceError(
            f"{root} has no blueprint_index",
            hint="write it with conduit.relay.io.blueprint.save_mesh (or Ascent's relay extract)",
        )
    mesh_names = list(index)
    n_files = _int(doc.get("number_of_files"), 1)
    n_trees = _int(doc.get("number_of_trees"), 1)
    file_pattern = _text(doc.get("file_pattern", root.name))
    tree_pattern = _text(doc.get("tree_pattern", "/"))
    if n_files not in (1, n_trees):
        raise UnsupportedSourceError(
            f"{root}: {n_trees} domains in {n_files} files (partition map layout)",
            hint="save with file_style 'multi_file' (one domain per file) or 'root_only'",
        )
    cache: dict[Path, dict[str, Any]] = {root.resolve(): doc}
    out: dict[str, Any] = {}
    for i in range(n_trees):
        fpath = resolve_in_case(
            root.parent.resolve(),
            _format(file_pattern, i if n_files > 1 else 0),
            what="file_pattern",
            referrer=root,
        )
        if fpath not in cache:
            if not fpath.exists():
                raise IncompleteFileError(
                    f"{fpath.name} (referenced by {root.name}) does not exist"
                )
            loaded = load_file(fpath)
            if not isinstance(loaded, dict):
                raise IncompleteFileError(f"{fpath.name} holds no tree")
            cache[fpath] = loaded
        base = _walk(cache[fpath], _format(tree_pattern, i))
        if not isinstance(base, dict):
            raise IncompleteFileError(f"{fpath.name}: tree {i} is not an object")
        for mesh in mesh_names:
            if mesh not in base:
                # a domain may legitimately lack one of several meshes, but not all of them
                continue
            if n_trees == 1:
                key = mesh
            elif len(mesh_names) == 1:
                key = f"domain_{i:06d}"
            else:
                key = f"{mesh}/domain_{i:06d}"
            _insert(out, key, base[mesh])
    if not out:
        raise IncompleteFileError(
            f"{root.name}: none of the meshes {mesh_names} found in its files"
        )
    return out


def _insert(out: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split("/")
    cur = out
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def root_state(root_doc: dict[str, Any]) -> tuple[float | None, int | None]:
    """``(time, cycle)`` recorded in the ``blueprint_index`` of a decoded root."""
    index = root_doc.get("blueprint_index")
    if not isinstance(index, dict):
        return None, None
    for mesh in index.values():
        state = mesh.get("state") if isinstance(mesh, dict) else None
        if isinstance(state, dict):
            t = state.get("time")
            c = state.get("cycle")
            return (
                float(np.asarray(t).reshape(-1)[0]) if t is not None else None,
                int(np.asarray(c).reshape(-1)[0]) if c is not None else None,
            )
    return None, None
