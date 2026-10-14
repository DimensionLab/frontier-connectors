# SPDX-License-Identifier: Apache-2.0
"""PyFR native files: ``.pyfrm`` meshes and ``.pyfrs`` solutions (HDF5).

Two on-disk layouts exist and both are handled (format facts from the BSD-3 PyFR sources and
documentation; no PyFR code is used or needed):

* **legacy** (PyFR <= 2.1): mesh datasets ``spt_<etype>_p<part>`` shaped ``(nspts, neles,
  ndims)`` plus ``con_p*``/``bcon_*``/``mesh_uuid``; solution datasets ``soln_<etype>_p<part>``
  shaped ``(nupts, nvars, neles)`` plus the strings ``config``, ``stats``, ``mesh_uuid``.
* **v3** (PyFR 3.x, ``version = 1``): mesh ``nodes`` (compound, ``location``), ``eles/<etype>``
  (compound, ``nodes``/``curved``/``faces``), ``codec``, ``creator``, ``mesh-uuid``; solution
  group ``<prefix>/p<order>-<etype>`` shaped ``(neles, nvars, nupts)`` with the reference
  solution points in the attribute ``pts``, plus ``config``, ``stats``, ``mesh-uuid``.

A flux-reconstruction solution lives at the solution points inside every high-order element, so
a ``.pyfrs`` becomes a ``point_cloud`` of those points (physical positions are interpolated from
the element shape points). A ``.pyfrm`` alone becomes a linear ``mesh`` of the element corners.
For cell-connected, subdivided output run ``pyfr export`` and read the VTU with the ``vtk``
connector.
"""

from __future__ import annotations

import configparser
import itertools
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import meshutil
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
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
from frontier_connectors.paths import h5_open
from frontier_connectors.select import match_names, select_time_indices, wants

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZONE_SOLN = "solution_points"
ZONE_MESH = "mesh"
ETYPES = ("quad", "tri", "hex", "tet", "pri", "pyr")
ETYPE_DIM = {"quad": 2, "tri": 2, "hex": 3, "tet": 3, "pri": 3, "pyr": 3}
ETYPE_VTK = {
    "quad": meshutil.VTK_QUAD,
    "tri": meshutil.VTK_TRIANGLE,
    "hex": meshutil.VTK_HEXAHEDRON,
    "tet": meshutil.VTK_TETRA,
    "pri": meshutil.VTK_WEDGE,
    "pyr": meshutil.VTK_PYRAMID,
}
EXPORT_HINT = (
    "run `pyfr export mesh.pyfrm soln.pyfrs out.vtu` (add `-d <n>` to subdivide) and read the "
    ".vtu with the vtk connector"
)
CONSERVATIVE_SYSTEMS = ("euler", "navier-stokes")

_LEGACY_DS = re.compile(r"^(?P<prefix>[A-Za-z0-9-]+)_(?P<etype>[a-z]+)_p(?P<part>\d+)$")
_V3_DS = re.compile(r"^p(?P<order>\d+)-(?P<etype>[a-z]+)$")
_NUMBERED = re.compile(r"^(?P<stem>.*?)[-_]?(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")


def _h5(path: Path) -> Any:
    import h5py  # noqa: F401 - availability check; the file is opened by h5_open

    try:
        return h5_open(path)
    except OSError as e:
        raise ConnectorError(
            f"{path} is not a readable HDF5 file: {e}",
            hint="PyFR .pyfrm/.pyfrs files are HDF5; the file is truncated or not a PyFR file",
        ) from e


def _text(node: Any) -> str:
    v = node[()]
    if isinstance(v, np.ndarray):
        v = v.item() if v.shape == () else v.tobytes()
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _ini(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str  # type: ignore[assignment]  # PyFR constants are case-sensitive
    try:
        cp.read_string(text)
    except configparser.Error as e:
        raise ConnectorError(
            f"embedded PyFR config/stats record is not valid INI: {e}",
            hint="the solution file is damaged; re-run or re-copy it",
        ) from e
    return cp


# -- reference elements -------------------------------------------------------------------------


def std_element(etype: str, order: int) -> np.ndarray:
    """PyFR's equispaced shape points of an element of polynomial ``order`` on the reference
    element (the order in which ``.pyfrm`` lists an element's nodes)."""
    n = order + 1
    p = np.linspace(-1.0, 1.0, n)
    if etype in ("quad", "hex"):
        d = ETYPE_DIM[etype]
        return np.array([q[::-1] for q in itertools.product(p, repeat=d)])
    if etype == "tri":
        return np.array([(a, b) for i, b in enumerate(p) for a in p[: n - i]])
    if etype == "tet":
        return np.array(
            [
                (a, b, c)
                for i, c in enumerate(p)
                for j, b in enumerate(p[: n - i])
                for a in p[: n - i - j]
            ]
        )
    if etype == "pri":
        return np.array([(a, b, c) for c in p for i, b in enumerate(p) for a in p[: n - i]])
    if etype == "pyr":
        m = 2 * order + 1
        q = np.linspace(-1.0, 1.0, m)
        return np.array(
            [
                (a, b, c)
                for i, c in enumerate(q[::2])
                for b in q[i : m - i : 2]
                for a in q[i : m - i : 2]
            ]
        )
    raise UnsupportedSourceError(f"unknown PyFR element type {etype!r}", hint=EXPORT_HINT)


def element_order(etype: str, nspts: int) -> int:
    """Polynomial order of the geometry from the number of shape points."""
    for k in range(1, 12):
        if len(std_element(etype, k)) == nspts:
            return k
    raise ConnectorError(
        f"{nspts} shape points is not a valid {etype} element", hint="the mesh file is damaged"
    )


def corner_indices(etype: str, order: int) -> list[int]:
    """Indices of the corner nodes inside :func:`std_element`, in VTK linear-cell order."""
    n = order + 1
    tri = n * (n + 1) // 2
    if etype == "quad":
        return [0, n - 1, n * n - 1, n * (n - 1)]
    if etype == "hex":
        top = n * n * (n - 1)
        base = [0, n - 1, n * n - 1, n * (n - 1)]
        return base + [top + b for b in base]
    if etype == "tri":
        return [0, n - 1, tri - 1]
    if etype == "tet":
        return [0, n - 1, tri - 1, n * (n + 1) * (n + 2) // 6 - 1]
    if etype == "pri":
        base = [0, n - 1, tri - 1]
        return base + [tri * (n - 1) + b for b in base]
    if etype == "pyr":
        return [0, n - 1, n * n - 1, n * (n - 1), len(std_element("pyr", order)) - 1]
    raise UnsupportedSourceError(f"unknown PyFR element type {etype!r}", hint=EXPORT_HINT)


def _exponents(etype: str, order: int) -> np.ndarray:
    """Monomial exponents spanning the nodal polynomial space of the element."""
    r = range(order + 1)
    if etype == "quad":
        e = [(i, j) for j in r for i in r]
    elif etype == "hex":
        e = [(i, j, k) for k in r for j in r for i in r]
    elif etype == "tri":
        e = [(i, j) for j in r for i in r if i + j <= order]
    elif etype == "tet":
        e = [(i, j, k) for k in r for j in r for i in r if i + j + k <= order]
    elif etype == "pri":
        e = [(i, j, k) for k in r for j in r for i in r if i + j <= order]
    else:
        raise UnsupportedSourceError(
            f"solution-point positions of PyFR {etype!r} elements are not implemented "
            "(their shape functions are rational)",
            hint=EXPORT_HINT,
        )
    return np.array(e, dtype=np.int64)


def _vandermonde(pts: np.ndarray, exps: np.ndarray) -> np.ndarray:
    return np.prod(pts[:, None, :] ** exps[None, :, :], axis=2)


def shape_interpolation(etype: str, nspts: int, upts: np.ndarray) -> np.ndarray:
    """Matrix ``(nupts, nspts)`` evaluating the element's nodal shape functions at the
    reference points ``upts``; multiply by an element's shape points to get positions."""
    order = element_order(etype, nspts)
    exps = _exponents(etype, order)
    vs = _vandermonde(std_element(etype, order), exps)
    vu = _vandermonde(np.asarray(upts, dtype=np.float64), exps)
    return np.linalg.solve(vs.T, vu.T).T


def tensor_points(etype: str, nupts: int, rule: str) -> np.ndarray:
    """Tensor-product Gauss-Legendre(-Lobatto) points of a quad/hex, first coordinate fastest
    (the order PyFR tabulates them in). Legacy files do not store the solution points."""
    d = ETYPE_DIM[etype]
    n = round(nupts ** (1.0 / d))
    if etype not in ("quad", "hex") or n**d != nupts:
        raise UnsupportedSourceError(
            f"legacy PyFR files do not record the solution points and those of {etype!r} "
            "elements come from PyFR's tabulated rules, which this reader does not ship",
            hint=EXPORT_HINT,
        )
    key = rule.strip().lower()
    if key == "gauss-legendre":
        x = np.polynomial.legendre.leggauss(n)[0]
    elif key == "gauss-legendre-lobatto":
        if n < 2:
            raise ConnectorError("a Lobatto rule needs at least 2 points", hint=EXPORT_HINT)
        inner = np.polynomial.legendre.Legendre.basis(n - 1).deriv().roots()
        x = np.concatenate([[-1.0], np.sort(inner.real), [1.0]])
    else:
        raise UnsupportedSourceError(
            f"solution-point rule {rule!r} of legacy PyFR files is not supported "
            "(only gauss-legendre and gauss-legendre-lobatto on quad/hex)",
            hint=EXPORT_HINT,
        )
    return np.array([q[::-1] for q in itertools.product(x, repeat=d)])


# -- file model ---------------------------------------------------------------------------------


@dataclass
class SolnInfo:
    """Cheap description of one ``.pyfrs`` (no heavy arrays read)."""

    path: Path
    layout: str  # "legacy" | "v3"
    uuid: str
    config: configparser.ConfigParser
    stats: configparser.ConfigParser
    prefix: str
    fields: list[str]
    #: etype -> list of dataset paths (one per partition for legacy files)
    datasets: dict[str, list[str]] = field(default_factory=dict)
    #: etype -> (neles, nvars, nupts)
    shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    subset: list[str] = field(default_factory=list)
    creator: str | None = None

    @property
    def time(self) -> float | None:
        try:
            return self.stats.getfloat("solver-time-integrator", "tcurr")
        except (configparser.Error, ValueError):
            return None

    @property
    def ndims(self) -> int:
        return ETYPE_DIM[next(iter(self.shapes))]

    @property
    def system(self) -> str:
        return self.config.get("solver", "system", fallback="").strip().lower()

    @property
    def order(self) -> int | None:
        try:
            return self.config.getint("solver", "order")
        except (configparser.Error, ValueError):
            return None


def _layout(f: Any) -> tuple[str, str] | None:
    """``(kind, layout)`` with kind ``mesh``/``soln``; ``None`` when not a PyFR file."""
    keys = set(f.keys())
    if "mesh-uuid" in keys:
        if "eles" in keys and "nodes" in keys:
            return "mesh", "v3"
        if "stats" in keys:
            return "soln", "v3"
    if "mesh_uuid" in keys:
        if any(k.startswith("spt_") for k in keys):
            return "mesh", "legacy"
        if "stats" in keys:
            return "soln", "legacy"
    return None


def _bad_layout(path: Path) -> ConnectorError:
    return ConnectorError(
        f"{path} is HDF5 but has neither the PyFR legacy (mesh_uuid, spt_*/soln_*) nor the "
        "PyFR 3 (mesh-uuid, eles/nodes or stats) layout",
        hint="supported: .pyfrm/.pyfrs written by PyFR 1.x-2.1 and 3.x; otherwise " + EXPORT_HINT,
    )


def soln_info(path: Path) -> SolnInfo:
    path = Path(path)
    with _h5(path) as f:
        found = _layout(f)
        if found is None or found[0] != "soln":
            raise _bad_layout(path)
        layout = found[1]
        try:
            config = _ini(_text(f["config"]))
            stats = _ini(_text(f["stats"]))
            uuid = _text(f["mesh-uuid" if layout == "v3" else "mesh_uuid"])
        except KeyError as e:
            raise ConnectorError(
                f"{path} lacks the PyFR record {e}", hint="the solution file is incomplete"
            ) from e
        prefix = stats.get("data", "prefix", fallback="soln")
        names = [n.strip() for n in stats.get("data", "fields", fallback="").split(",")]
        info = SolnInfo(path, layout, uuid, config, stats, prefix, [n for n in names if n])
        if layout == "v3":
            info.creator = _text(f["creator"]) if "creator" in f else None
            group: Any = f[prefix] if prefix in f else {}
            for name in sorted(group.keys()):
                m = _V3_DS.match(name)
                if not m or m.group("etype") not in ETYPES:
                    continue
                et = m.group("etype")
                ne, nv, nu = (int(s) for s in group[name].shape)
                info.datasets[et] = [f"{prefix}/{name}"]
                info.shapes[et] = (ne, nv, nu)
                if f"{name}-idxs" in group:
                    info.subset.append(et)
        else:
            parts: dict[str, list[tuple[int, str]]] = {}
            for name in f.keys():
                m = _LEGACY_DS.match(name)
                if m and m.group("prefix") == prefix and m.group("etype") in ETYPES:
                    parts.setdefault(m.group("etype"), []).append((int(m.group("part")), name))
                elif re.match(rf"^{re.escape(prefix)}_[a-z]+_idxs_p\d+$", name):
                    info.subset.append(name.split("_")[1])
            for et, items in sorted(parts.items()):
                items.sort()
                info.datasets[et] = [n for _, n in items]
                shapes = [f[n].shape for n in info.datasets[et]]
                if any(len(s) != 3 or s[:2] != shapes[0][:2] for s in shapes):
                    raise ConnectorError(
                        f"{path}: partitions of {et} elements disagree on shape {shapes}",
                        hint="the solution file is damaged",
                    )
                nu, nv = int(shapes[0][0]), int(shapes[0][1])
                info.shapes[et] = (sum(int(s[2]) for s in shapes), nv, nu)
        if not info.shapes:
            raise ConnectorError(
                f"{path} holds no '{prefix}' element data",
                hint="the file has only metadata; check the writer plugin of the run",
            )
        nvars = {s[1] for s in info.shapes.values()}
        if len(nvars) != 1:
            raise ConnectorError(
                f"{path}: element types carry different variable counts {sorted(nvars)}",
                hint="the solution file is damaged",
            )
        nv = nvars.pop()
        if len(info.fields) != nv:
            info.fields = (info.fields + [f"var_{i}" for i in range(nv)])[:nv]
            if len(set(info.fields)) != nv:
                info.fields = [f"var_{i}" for i in range(nv)]
    return info


def read_soln(info: SolnInfo, etype: str) -> np.ndarray:
    """Solution of one element type as ``(neles, nupts, nvars)`` float array."""
    with _h5(info.path) as f:
        if info.layout == "v3":
            return np.asarray(f[info.datasets[etype][0]][()]).transpose(0, 2, 1)
        arrs = [np.asarray(f[n][()]).transpose(2, 0, 1) for n in info.datasets[etype]]
        return np.concatenate(arrs, axis=0)


@dataclass
class MeshInfo:
    path: Path
    layout: str
    uuid: str
    ndims: int
    #: etype -> (neles, nspts)
    shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    boundaries: list[str] = field(default_factory=list)
    creator: str | None = None
    partitions: int = 1


def mesh_info(path: Path) -> MeshInfo:
    path = Path(path)
    with _h5(path) as f:
        found = _layout(f)
        if found is None or found[0] != "mesh":
            raise _bad_layout(path)
        layout = found[1]
        if layout == "v3":
            try:
                ndims = int(f["nodes"].dtype["location"].shape[0])
            except (KeyError, IndexError, TypeError) as e:
                raise ConnectorError(
                    f"{path}: 'nodes' has no 'location' member", hint="the mesh file is damaged"
                ) from e
            info = MeshInfo(path, layout, _text(f["mesh-uuid"]), ndims)
            info.creator = _text(f["creator"]) if "creator" in f else None
            for et in sorted(f["eles"].keys()):
                if et in ETYPES:
                    ds = f["eles"][et]
                    info.shapes[et] = (int(ds.shape[0]), int(ds.dtype["nodes"].shape[0]))
            if "codec" in f:
                codec = [c.decode() if isinstance(c, bytes) else str(c) for c in f["codec"][()]]
                info.boundaries = [c[3:] for c in codec if c.startswith("bc/")]
        else:
            parts: set[int] = set()
            shapes: dict[str, tuple[int, int]] = {}
            ndims = 0
            bcs: set[str] = set()
            for name in f.keys():
                m = _LEGACY_DS.match(name)
                if m and m.group("prefix") == "spt" and m.group("etype") in ETYPES:
                    nspts, neles, ndims = (int(s) for s in f[name].shape)
                    et = m.group("etype")
                    shapes[et] = (shapes.get(et, (0, nspts))[0] + neles, nspts)
                    parts.add(int(m.group("part")))
                elif name.startswith("bcon_"):
                    bcs.add(re.sub(r"_p\d+$", "", name[5:]))
            info = MeshInfo(path, layout, _text(f["mesh_uuid"]), ndims, shapes, sorted(bcs))
            info.partitions = len(parts) or 1
        if not info.shapes:
            raise ConnectorError(f"{path} holds no elements", hint="the mesh file is empty")
    return info


def read_shape_points(info: MeshInfo, etype: str) -> np.ndarray:
    """Shape points of every element of a type as ``(neles, nspts, ndims)`` float64."""
    with _h5(info.path) as f:
        if info.layout == "v3":
            nodes = np.asarray(f["eles"][etype].fields("nodes")[()], dtype=np.int64)
            loc = np.asarray(f["nodes"].fields("location")[()], dtype=np.float64)
            return loc[nodes]
        names = sorted(
            (n for n in f.keys() if re.match(rf"^spt_{etype}_p\d+$", n)),
            key=lambda n: int(n.rsplit("p", 1)[1]),
        )
        arrs = [np.asarray(f[n][()], dtype=np.float64).transpose(1, 0, 2) for n in names]
        return np.concatenate(arrs, axis=0)


def linear_mesh(info: MeshInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner-node skeleton of the mesh: ``(points, cells, cell_types)``."""
    blocks: list[tuple[int, np.ndarray]] = []
    if info.layout == "v3":
        with _h5(info.path) as f:
            points = np.asarray(f["nodes"].fields("location")[()], dtype=np.float64)
            for et, (_, nspts) in info.shapes.items():
                nodes = np.asarray(f["eles"][et].fields("nodes")[()], dtype=np.int64)
                blocks.append(
                    (ETYPE_VTK[et], nodes[:, corner_indices(et, element_order(et, nspts))])
                )
        cells, types = meshutil.pad_cell_blocks(blocks)
        points, cells, _ = meshutil.compact_points(points, cells)
        return points, cells, types
    chunks: list[np.ndarray] = []
    offset = 0
    for et, (_, nspts) in info.shapes.items():
        corners = read_shape_points(info, et)[:, corner_indices(et, element_order(et, nspts))]
        ne, nc, _ = corners.shape
        chunks.append(corners.reshape(ne * nc, -1))
        blocks.append((ETYPE_VTK[et], offset + np.arange(ne * nc).reshape(ne, nc)))
        offset += ne * nc
    cells, types = meshutil.pad_cell_blocks(blocks)
    return np.concatenate(chunks), cells, types


def solution_points(mesh: MeshInfo, soln: SolnInfo, etype: str) -> np.ndarray:
    """Physical positions ``(neles * nupts, ndims)`` of the solution points of one type."""
    ne, _, nu = soln.shapes[etype]
    if etype not in mesh.shapes:
        raise ConnectorError(
            f"{soln.path} has {etype} elements but mesh {mesh.path} has none",
            hint="pass the mesh the solution was computed on (options={'mesh': ...})",
        )
    spts = read_shape_points(mesh, etype)
    if etype in soln.subset:
        raise UnsupportedSourceError(
            f"{soln.path} stores a region subset of the {etype} elements, which is not read",
            hint="write the full solution (no `region` in [soln-plugin-writer]) or " + EXPORT_HINT,
        )
    if len(spts) != ne:
        raise ConnectorError(
            f"{soln.path} has {ne} {etype} elements, mesh {mesh.path} has {len(spts)}",
            hint="mesh and solution must come from the same run and the same partitioning",
        )
    if soln.layout == "v3":
        with _h5(soln.path) as f:
            ds = f[soln.datasets[etype][0]]
            if "pts" not in ds.attrs:
                raise ConnectorError(
                    f"{soln.path}: dataset {ds.name} lacks the 'pts' attribute",
                    hint=EXPORT_HINT,
                )
            upts = np.asarray(ds.attrs["pts"], dtype=np.float64).reshape(nu, -1)
    else:
        rule = soln.config.get(f"solver-elements-{etype}", "soln-pts", fallback="")
        upts = tensor_points(etype, nu, rule)
    m = shape_interpolation(etype, spts.shape[1], upts)
    return np.einsum("us,esd->eud", m, spts).reshape(ne * nu, -1)


# -- series -------------------------------------------------------------------------------------


def _family_key(path: Path) -> str:
    m = _NUMBERED.match(path.stem)
    return m.group("stem") if m else path.stem


def find_solutions(path: Path, *, single: bool = False) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.pyfrs"))
        keys = {_family_key(f) for f in files}
        if len(keys) > 1:
            raise UnsupportedSourceError(
                f"{path} holds several solution families ({', '.join(sorted(keys))})",
                hint="point at one .pyfrs file of the series you want",
            )
        return files
    if single:
        return [path]
    key = _family_key(path)
    return [f for f in sorted(path.parent.glob("*.pyfrs")) if _family_key(f) == key] or [path]


def find_mesh(soln: SolnInfo, options: Mapping[str, Any]) -> MeshInfo | None:
    if options.get("mesh"):
        info = mesh_info(Path(str(options["mesh"])))
        if info.uuid != soln.uuid:
            raise ConnectorError(
                f"mesh {info.path} (uuid {info.uuid}) does not belong to {soln.path} "
                f"(uuid {soln.uuid})",
                hint="pass the .pyfrm the run was started with",
            )
        return info
    for d in (soln.path.parent, soln.path.parent.parent):
        for cand in sorted(d.glob("*.pyfrm")):
            try:
                info = mesh_info(cand)
            except ConnectorError:
                continue
            if info.uuid == soln.uuid:
                return info
    return None


def _version(creator: str | None) -> str | None:
    m = re.search(r"(\d+(?:\.\d+)+\S*)", creator or "")
    return m.group(1) if m else None


def _parameters(config: configparser.ConfigParser) -> dict[str, float]:
    out: dict[str, float] = {}
    if config.has_section("constants"):
        for k, v in config.items("constants"):
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _physics(info: SolnInfo) -> dict[str, Any]:
    cfg = info.config
    phys: dict[str, Any] = {"method": "flux-reconstruction"}
    if info.system:
        phys["system"] = info.system
        phys["compressible"] = info.system in CONSERVATIVE_SYSTEMS
        phys["viscous"] = "navier-stokes" in info.system
    if info.order is not None:
        phys["polynomial_order"] = info.order
    for key, sect, opt in [
        ("time_scheme", "solver-time-integrator", "scheme"),
        ("riemann_solver", "solver-interfaces", "riemann-solver"),
        ("shock_capturing", "solver", "shock-capturing"),
    ]:
        if cfg.has_option(sect, opt):
            phys[key] = cfg.get(sect, opt)
    return phys


def _boundary_conditions(config: configparser.ConfigParser) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sect in config.sections():
        if sect.startswith("soln-bcs-"):
            out[sect[len("soln-bcs-") :]] = dict(config.items(sect))
    return out


def primitive_fields(names: list[str], ndims: int, system: str, gamma: float | None) -> bool:
    cons = ["rho", "rhou", "rhov", "rhow", "E"] if ndims == 3 else ["rho", "rhou", "rhov", "E"]
    return system in CONSERVATIVE_SYSTEMS and gamma is not None and names[: len(cons)] == cons


class PyfrConnector(BaseConnector):
    id = "pyfr"
    name = "PyFR native mesh and solution (.pyfrm, .pyfrs)"
    formats = ["pyfr-v3", "pyfr-legacy"]
    vendor = "PyFR (open source, Imperial College London / Texas A&M)"
    license = "Apache-2.0 (own reader; PyFR itself is BSD-3)"
    extensions = [".pyfrs", ".pyfrm"]
    maturity = "verified-synthetic"
    temporal = ["steady", "transient"]
    kinds = ["point_cloud", "mesh"]
    requires: list[str] = []
    export_help = (
        "PyFR writes soln-<t>.pyfrs next to the .pyfrm mesh; read them directly as a cloud of "
        "solution points, or run `pyfr export mesh.pyfrm soln.pyfrs out.vtu` and read the VTU "
        "with the vtk connector when cell connectivity is needed."
    )
    status_note = (
        "Reads PyFR <= 2.1 files (spt_* / soln_* datasets) and PyFR 3.x files (eles / nodes, "
        "<prefix>/p<order>-<etype> with 'pts'). A solution becomes a point cloud of solution "
        "points (quad, hex, tri, tet, pri; legacy files only quad and hex with Gauss-Legendre or "
        "Lobatto points; no pyramids, no region subsets); a mesh alone becomes a linear corner "
        "mesh. Validated on generated files and on PyFR 3.1 and 2.1 output."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        try:
            if path.is_dir():
                files = list(path.glob("*.pyfrs"))
                if files and self._magic(files[0]):
                    return Detection(
                        self.id, path, 0.85, "pyfr", [f"{len(files)} .pyfrs solution files"]
                    )
                return None
            if not path.is_file() or path.suffix.lower() not in (".pyfrs", ".pyfrm"):
                return None
            if not self._magic(path):
                return None
            with _h5(path) as f:
                found = _layout(f)
        except (OSError, ConnectorError):
            return None
        if found is None:
            return Detection(self.id, path, 0.4, "pyfr", ["HDF5 with a PyFR extension"])
        kind, layout = found
        return Detection(self.id, path, 0.97, f"pyfr-{layout}", [f"PyFR {layout} {kind} layout"])

    @staticmethod
    def _magic(path: Path) -> bool:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_MAGIC

    # -- plan ---------------------------------------------------------------------------------
    def _plan(
        self, path: Path, options: Mapping[str, Any]
    ) -> tuple[list[SolnInfo], list[float], list[str]]:
        files = find_solutions(path, single=options.get("series") is False)
        if not files:
            raise UnsupportedSourceError(
                f"no .pyfrs files in {path}", hint="point at a .pyfrs or .pyfrm file"
            )
        warnings: list[str] = []
        anchor = soln_info(files[0] if path.is_dir() else path)
        infos: list[SolnInfo] = []
        for f in files:
            info = anchor if f == anchor.path else soln_info(f)
            if info.uuid != anchor.uuid or info.shapes != anchor.shapes:
                warnings.append(f"{f.name} belongs to another mesh and is skipped")
                continue
            infos.append(info)
        if any(i.time is None for i in infos) and len(infos) > 1:
            raise ConnectorError(
                "a .pyfrs of the series has no [solver-time-integrator] tcurr in its stats",
                hint='read the files one by one with options={"series": false}',
            )
        infos.sort(key=lambda i: i.time or 0.0)
        times: list[float] = []
        kept: list[SolnInfo] = []
        for i in infos:
            t = float(i.time or 0.0)
            if times and t <= times[-1]:
                warnings.append(f"{i.path.name} repeats time {t} and is skipped")
                continue
            times.append(t)
            kept.append(i)
        return kept, times, warnings

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        opts = dict(options or {})
        if path.is_file() and path.suffix.lower() == ".pyfrm":
            return self._describe_mesh(path)
        infos, times, warnings = self._plan(path, opts)
        info = infos[0]
        gamma = _parameters(info.config).get("gamma")
        primitive = opts.get("primitive", True) and primitive_fields(
            info.fields, info.ndims, info.system, gamma
        )
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format=f"pyfr-{info.layout}",
            solver="PyFR",
            solver_version=_version(info.creator),
            domain="cfd",
            physics=_physics(info),
            time=TimeInfo(values=times if len(times) > 1 else []),
            boundary_conditions=_boundary_conditions(info.config),
            parameters=_parameters(info.config),
            warnings=warnings,
        )
        n_points = sum(ne * nu for ne, _, nu in info.shapes.values())
        desc.zones.append(
            ZoneInfo(
                name=ZONE_SOLN,
                kind="point_cloud",
                role="volume",
                n_points=n_points,
                n_cells=0,
                dim=info.ndims,
                cell_types={et: s[0] for et, s in info.shapes.items()},
            )
        )
        for name, comps in self._output_fields(info, primitive):
            desc.fields.append(FieldInfo(name=name, location="point", components=comps))
        desc.metadata = {
            "layout": info.layout,
            "mesh_uuid": info.uuid,
            "elements": {et: s[0] for et, s in info.shapes.items()},
            "solution_points_per_element": {et: s[2] for et, s in info.shapes.items()},
            "stored_fields": info.fields,
            "primitive": bool(primitive),
        }
        if len(times) == 1:
            desc.metadata["time_value"] = times[0]
        if info.subset:
            desc.warnings.append(
                f"region-subset solution ({', '.join(info.subset)}): read() will refuse it"
            )
        if find_mesh(info, opts) is None:
            desc.warnings.append(
                "no matching .pyfrm found next to the solution: read() needs "
                "options={'mesh': '<file>.pyfrm'}"
            )
        return desc

    @staticmethod
    def _output_fields(info: SolnInfo, primitive: bool) -> list[tuple[str, int]]:
        if not primitive:
            return [(n, 1) for n in info.fields]
        prim = ["rho", "u", "v", "w", "p"] if info.ndims == 3 else ["rho", "u", "v", "p"]
        return [(n, 1) for n in prim + info.fields[len(prim) :]]

    def _describe_mesh(self, path: Path) -> SourceDescription:
        info = mesh_info(path)
        orders = {et: element_order(et, s[1]) for et, s in info.shapes.items()}
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format=f"pyfr-{info.layout}",
            solver="PyFR",
            solver_version=_version(info.creator),
            domain="cfd",
        )
        desc.zones.append(
            ZoneInfo(
                name=ZONE_MESH,
                kind="mesh",
                role="volume",
                n_cells=sum(s[0] for s in info.shapes.values()),
                dim=info.ndims,
                cell_types={
                    meshutil.VTK_TYPE_NAMES[ETYPE_VTK[et]]: s[0] for et, s in info.shapes.items()
                },
            )
        )
        desc.boundary_conditions = {b: {} for b in info.boundaries}
        desc.metadata = {
            "layout": info.layout,
            "mesh_uuid": info.uuid,
            "geometry_order": orders,
            "partitions": info.partitions,
        }
        if any(o > 1 for o in orders.values()):
            desc.warnings.append(
                "curved (high-order) elements are reduced to their corner nodes; " + EXPORT_HINT
            )
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
        opts = dict(options or {})
        if path.is_file() and path.suffix.lower() == ".pyfrm":
            yield from self._read_mesh(path, zones)
            return
        infos, all_times, _ = self._plan(path, opts)
        match_names([ZONE_SOLN], zones, what="zone")
        transient = len(all_times) > 1
        indices = select_time_indices(all_times if transient else [], times)
        first_info = infos[0]
        mesh = find_mesh(first_info, opts)
        if mesh is None:
            raise ConnectorError(
                f"no .pyfrm with mesh uuid {first_info.uuid} found next to {first_info.path}",
                hint="pass options={'mesh': '/path/to/mesh.pyfrm'} (CLI: --option mesh=...)",
            )
        gamma = _parameters(first_info.config).get("gamma")
        primitive = bool(opts.get("primitive", True)) and primitive_fields(
            first_info.fields, first_info.ndims, first_info.system, gamma
        )
        sample_id = meshutil.default_sample_id(
            path if path.is_dir() or not transient else path.with_name(_family_key(path))
        )
        metadata: dict[str, Any] = {
            "solver": "PyFR",
            "solver_version": _version(first_info.creator),
            "source": str(path),
            "format": f"pyfr-{first_info.layout}",
            "physics": _physics(first_info),
            "parameters": _parameters(first_info.config),
            "boundary_conditions": _boundary_conditions(first_info.config),
            "mesh_uuid": first_info.uuid,
            "primitive": primitive,
        }
        etypes = list(first_info.shapes)
        first = True
        for ti in indices:
            info = infos[ti]
            sample = RawSample(
                id=sample_id,
                kind="point_cloud",
                zone=ZONE_SOLN,
                time=float(all_times[ti]) if transient else None,
                time_index=ti if transient else None,
                metadata=dict(metadata),
            )
            if first:
                sample.points = np.concatenate(
                    [solution_points(mesh, info, et) for et in etypes]
                ).astype(np.float64)
                counts = [info.shapes[et][0] * info.shapes[et][2] for et in etypes]
                if len(etypes) > 1:
                    codes = np.repeat([ETYPE_VTK[et] for et in etypes], counts).astype(np.int32)
                    sample.static["element_type"] = as_field(codes, "point")
                first = False
            data = np.concatenate(
                [read_soln(info, et).reshape(-1, info.shapes[et][1]) for et in etypes]
            )
            names = list(info.fields)
            if primitive:
                data, names = _to_primitive(data, names, info.ndims, float(gamma or 1.4))
            for i, name in enumerate(names):
                if wants(name, fields):
                    sample.fields[name] = as_field(data[:, i], "point")
            yield sample

    def _read_mesh(self, path: Path, zones: Sequence[str] | None) -> Iterator[RawSample]:
        match_names([ZONE_MESH], zones, what="zone")
        info = mesh_info(path)
        points, cells, types = linear_mesh(info)
        yield RawSample(
            id=meshutil.default_sample_id(path),
            kind="mesh",
            zone=ZONE_MESH,
            points=points,
            cells=cells,
            cell_types=types,
            metadata={
                "solver": "PyFR",
                "solver_version": _version(info.creator),
                "source": str(path),
                "format": f"pyfr-{info.layout}",
                "mesh_uuid": info.uuid,
            },
        )


def _to_primitive(
    data: np.ndarray, names: list[str], ndims: int, gamma: float
) -> tuple[np.ndarray, list[str]]:
    """Conservative ``rho, rho*u.., E`` -> ``rho, u.., p`` (ideal gas); extra columns kept."""
    out = np.array(data, dtype=np.float64, copy=True)
    rho = out[:, 0]
    vel = out[:, 1 : 1 + ndims] / rho[:, None]
    out[:, 1 : 1 + ndims] = vel
    out[:, 1 + ndims] = (gamma - 1.0) * (data[:, 1 + ndims] - 0.5 * rho * (vel**2).sum(axis=1))
    prim = ["rho", "u", "v", "w", "p"] if ndims == 3 else ["rho", "u", "v", "p"]
    return out, prim + names[len(prim) :]
