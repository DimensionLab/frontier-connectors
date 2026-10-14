# SPDX-License-Identifier: Apache-2.0
"""ParaView Catalyst 2 / in-situ streaming: Blueprint bridge, spool protocol, live follower,
Conduit relay files, ADIOS2 and the solver-side bundle.

The "solver" is a small Python program run in a subprocess (2-D explicit heat equation on a
grid, and a moving pulse on a triangle mesh) that emits steps through ``frontier_emit.py``
while this process follows the stream and builds the dataset.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from frontier_connector_catalyst import blueprint, relay, spool
from frontier_connector_catalyst.blueprint import StepConverter, mesh_to_samples
from frontier_connector_catalyst.connector import POINTER_NAME, CatalystConnector
from frontier_connector_catalyst.stream import FollowOptions, StreamTimeoutError

from frontier_connectors import importers, registry
from frontier_connectors.cli import main as cli_main
from frontier_connectors.convert import convert, frontier_data_available
from frontier_connectors.errors import (
    ConnectorError,
    SelectionError,
    UnsupportedSourceError,
)
from frontier_connectors.plugins import exporter_dir
from frontier_connectors.testing import (
    check_connector,
    hex_grid,
    open_store,
    quad_grid,
    tet_grid,
    tri_grid,
    validate_with_frontier_data,
)

FILES = exporter_dir("catalyst")


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, FILES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fe = _load("frontier_emit")


@pytest.fixture
def connector() -> CatalystConnector:
    return CatalystConnector()


# -- the tiny solver ---------------------------------------------------------------------------

NX, NY, ALPHA, DT, LX, LY = 16, 12, 0.05, 0.002, 1.0, 0.75

SOLVER = r"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import frontier_emit as fe

cfg = json.loads(sys.argv[2])
spool, kind, n_steps, pause = cfg["spool"], cfg["kind"], cfg["n_steps"], cfg["pause"]
time.sleep(cfg.get("start_delay", 0.0))

if kind == "heat":
    # explicit 2-D heat equation, periodic box, one Fourier mode as initial state
    nx, ny, alpha, dt, lx, ly = cfg["nx"], cfg["ny"], cfg["alpha"], cfg["dt"], cfg["lx"], cfg["ly"]
    dx, dy = lx / nx, ly / ny
    x, y = np.arange(nx) * dx, np.arange(ny) * dy
    T = np.sin(2 * np.pi * x / lx)[:, None] * np.cos(2 * np.pi * y / ly)[None, :]
    def lap(a):
        return ((np.roll(a, 1, 0) - 2 * a + np.roll(a, -1, 0)) / dx**2
                + (np.roll(a, 1, 1) - 2 * a + np.roll(a, -1, 1)) / dy**2)
    em = fe.Emitter(spool, solver="tiny-heat", solver_version="1.0", domain="thermal",
                    parameters={"alpha": alpha}, units={"length": "m"},
                    restart_from=cfg.get("restart_from"))
    first = cfg.get("first_step", 0)
    for _ in range(first):
        T = T + alpha * dt * lap(T)
    for step in range(first, n_steps):
        gx, gy = np.gradient(T, dx, dy)
        em.write_step(step, step * dt, fe.uniform_mesh(
            (nx, ny), spacing=(dx, dy),
            point_fields={"T": T, "gradT": np.stack([gx, gy], axis=-1)}, units={"T": "K"}))
        if cfg.get("die_after") is not None and step + 1 >= cfg["die_after"]:
            os._exit(3)          # crash: no end marker
        time.sleep(pause)
        T = T + alpha * dt * lap(T)
    em.close()
else:
    # triangle mesh of the unit square with a pulse moving in +x
    n = cfg["n"]
    xs = np.linspace(0.0, 1.0, n + 1)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = (i + (n + 1) * j).ravel()
    tris = np.concatenate(
        [np.stack([a, a + 1, a + n + 2], 1), np.stack([a, a + n + 2, a + n + 1], 1)])
    cen = pts[tris].mean(axis=1)
    def pulse(p, t):
        return np.exp(-((p[:, 0] - 0.2 - 0.5 * t) ** 2 + (p[:, 1] - 0.5) ** 2) / 0.02)
    with fe.Emitter(spool, solver="tiny-pulse", static_mesh=cfg.get("static", True),
                    domain="cfd", time_units="s") as em:
        for step in range(n_steps):
            t = 0.1 * step
            p = pulse(pts, t)
            moved = pts + (np.array([0.0, 0.05 * t, 0.0]) if cfg.get("moving") else 0.0)
            em.write_step(step * 5, t, fe.unstructured_mesh(
                moved, tris, "tri",
                point_fields={"p": p, "U": {"u": np.full(len(pts), 0.5), "v": np.zeros(len(pts)),
                                            "w": np.zeros(len(pts))}},
                cell_fields={"pc": pulse(cen, t)}, units={"p": "Pa", "U": "m/s"}))
            time.sleep(pause)
"""


def start_solver(tmp_path: Path, **cfg: Any) -> subprocess.Popen[bytes]:
    script = tmp_path / "tiny_solver.py"
    script.write_text(SOLVER)
    cfg.setdefault("kind", "heat")
    cfg.setdefault("n_steps", 6)
    cfg.setdefault("pause", 0.05)
    cfg.update(nx=NX, ny=NY, alpha=ALPHA, dt=DT, lx=LX, ly=LY)
    cfg["spool"] = str(cfg["spool"])
    return subprocess.Popen([sys.executable, str(script), str(FILES), json.dumps(cfg)])


def heat_exact(step: int) -> np.ndarray:
    """State of the discrete solver after ``step`` updates: the initial mode times g**step."""
    dx, dy = LX / NX, LY / NY
    x, y = np.arange(NX) * dx, np.arange(NY) * dy
    g = 1.0 + ALPHA * DT * (
        (2 * np.cos(2 * np.pi / NX) - 2) / dx**2 + (2 * np.cos(2 * np.pi / NY) - 2) / dy**2
    )
    return g**step * np.sin(2 * np.pi * x / LX)[:, None] * np.cos(2 * np.pi * y / LY)[None, :]


def pulse(p: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-((p[:, 0] - 0.2 - 0.5 * t) ** 2 + (p[:, 1] - 0.5) ** 2) / 0.02)


LIVE = {"follow": True, "timeout": 20, "poll": 0.01}


# -- 1. Blueprint bridge -------------------------------------------------------------------------


def _detect(connector: CatalystConnector, path: Path) -> Any:
    det = connector.detect(path)
    assert det is not None, f"{path} was not detected"
    return det


def _samples(tree: Any, *args: Any, **kw: Any) -> list[Any]:
    """``mesh_to_samples`` typed loosely: tests index optional arrays freely."""
    return list(mesh_to_samples(tree, *args, **kw))


def _one(tree: Any, **kw: Any) -> Any:
    samples = mesh_to_samples(tree, "s", **kw)
    assert len(samples) == 1
    samples[0].validate()
    return samples[0]


def test_uniform_grid_point_fields_and_units() -> None:
    nx, ny, nz = 4, 3, 2
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    f = 100 * i + 10 * j + k
    mesh = fe.uniform_mesh(
        (nx, ny, nz),
        origin=(1.0, 2.0, 3.0),
        spacing=(0.5, 0.25, 2.0),
        point_fields={"f": f, "v": np.stack([i, j, k], axis=-1)},
        units={"f": "K"},
    )
    s = _one(mesh)
    assert s.kind == "grid" and list(s.coords) == ["x", "y", "z"]
    np.testing.assert_allclose(s.coords["x"], 1.0 + 0.5 * np.arange(nx))
    np.testing.assert_allclose(s.coords["z"], [3.0, 5.0])
    assert s.fields["f"].dims == ["x", "y", "z", "component"]
    np.testing.assert_array_equal(s.fields["f"].data[..., 0], f)
    np.testing.assert_array_equal(s.fields["v"].data[2, 1, 1], [2, 1, 1])
    assert s.fields["f"].units == "K" and s.fields["f"].data.dtype != object
    assert s.metadata["spacing"] == [0.5, 0.25, 2.0]


def test_grid_element_fields_cell_centres_and_averaging() -> None:
    x, y = np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0])
    cell = np.array([[10.0], [20.0]])  # (2, 1) cells
    s = _one(fe.rectilinear_mesh((x, y), cell_fields={"c": cell}))
    assert s.metadata["grid_location"] == "cell_centers"
    np.testing.assert_allclose(s.coords["x"], [0.5, 2.0])
    np.testing.assert_allclose(s.fields["c"].data[..., 0], cell)
    both = _one(
        fe.rectilinear_mesh((x, y), point_fields={"p": np.zeros((3, 2))}, cell_fields={"c": cell})
    )
    np.testing.assert_allclose(both.coords["x"], x)
    # nodes take the mean of the cells that touch them
    np.testing.assert_allclose(both.fields["c"].data[:, 0, 0], [10.0, 15.0, 20.0])
    assert both.metadata["cell_fields_averaged_to_nodes"] == ["c"]


def test_structured_curvilinear_3d_and_2d() -> None:
    pts, hexes = hex_grid(3, 2, 2)
    warped = pts + 0.1 * np.sin(pts[:, [1, 2, 0]])
    s = _one(
        fe.structured_mesh(
            warped, (4, 3, 3), point_fields={"p": pts[:, 0]}, cell_fields={"c": np.arange(12.0)}
        )
    )
    assert s.kind == "mesh" and s.metadata["structured_shape"] == [4, 3, 3]
    np.testing.assert_array_equal(s.cells, hexes)
    assert set(s.cell_types.tolist()) == {12}
    np.testing.assert_allclose(s.points, warped)
    qp, quads = quad_grid(3, 2)
    surf = _one(fe.structured_mesh(qp[:, :2], (4, 3), cell_fields={"c": np.arange(6.0)}))
    assert surf.kind == "surface" and surf.points.shape == (12, 3)
    assert len(surf.faces) == 12
    parent = surf.static["face_parent"].data[:, 0]
    np.testing.assert_array_equal(surf.fields["c"].data[:, 0], np.arange(6.0)[parent])
    np.testing.assert_array_equal(np.sort(np.unique(surf.faces)), np.arange(12))
    del quads


WEDGE = (
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1.0]]),
    [[0, 1, 2, 3, 4, 5]],
)
PYRAMID = (
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1.0]]),
    [[0, 1, 2, 3, 4]],
)


@pytest.mark.parametrize(
    ("shape", "vtk", "kind"),
    [
        ("tet", 10, "mesh"),
        ("hex", 12, "mesh"),
        ("wedge", 13, "mesh"),
        ("pyramid", 14, "mesh"),
        ("line", 3, "mesh"),
        ("tri", 5, "surface"),
        ("quad", 9, "surface"),
        ("point", 1, "point_cloud"),
    ],
)
def test_unstructured_single_shapes(shape: str, vtk: int, kind: str) -> None:
    if shape == "tet":
        pts, cells = tet_grid(2, 1, 1)
    elif shape == "hex":
        pts, cells = hex_grid(2, 2, 1)
    elif shape == "wedge":
        pts, cells = WEDGE[0], np.array(WEDGE[1])
    elif shape == "pyramid":
        pts, cells = PYRAMID[0], np.array(PYRAMID[1])
    elif shape == "tri":
        pts, cells = tri_grid(2, 2)
    elif shape == "quad":
        pts, cells = quad_grid(2, 2)
    elif shape == "line":
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])
        cells = np.array([[0, 1], [1, 2]])
    else:
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])
        cells = np.array([[0], [1], [2]])
    cell_vals = np.arange(len(cells), dtype=float)
    s = _one(
        fe.unstructured_mesh(
            pts, cells, shape, point_fields={"p": pts[:, 0]}, cell_fields={"c": cell_vals}
        )
    )
    assert s.kind == kind
    np.testing.assert_allclose(s.points, pts)
    np.testing.assert_allclose(s.fields["p"].data[:, 0], pts[:, 0])
    if kind == "mesh":
        np.testing.assert_array_equal(s.cells, cells)
        assert set(s.cell_types.tolist()) == {vtk}
        assert s.fields["c"].location == "cell"
        np.testing.assert_allclose(s.fields["c"].data[:, 0], cell_vals)
    elif kind == "surface":
        n_tri = len(cells) * (1 if shape == "tri" else 2)
        assert s.faces.shape == (n_tri, 3)
        assert len(s.fields["c"].data) == n_tri
        if shape == "quad":
            parent = s.static["face_parent"].data[:, 0]
            np.testing.assert_allclose(s.fields["c"].data[:, 0], cell_vals[parent])
        else:
            assert "face_parent" not in s.static
    else:
        assert s.cells is None and s.fields["c"].location == "point"


def test_mixed_shapes_keep_order_types_and_padding() -> None:
    hp, hexes = hex_grid(1, 1, 1)
    pts = np.concatenate([hp, [[0.5, 0.5, 2.0]], [[2.0, 0.0, 0.0]]])
    blocks = [("hex", hexes), ("pyramid", [[4, 5, 6, 7, 8]]), ("tet", [[1, 9, 2, 5]])]
    mesh = fe.unstructured_mesh(pts, blocks, cell_fields={"id": [0.0, 1.0, 2.0]})
    el = mesh["topologies"]["mesh"]["elements"]
    assert el["shape"] == "mixed" and el["shape_map"] == {"hex": 12, "pyramid": 14, "tet": 10}
    s = _one(mesh)
    assert s.kind == "mesh" and s.cell_types.tolist() == [12, 14, 10]
    assert s.cells.shape == (3, 8)
    np.testing.assert_array_equal(s.cells[1], [4, 5, 6, 7, 8, -1, -1, -1])
    np.testing.assert_array_equal(s.cells[2], [1, 9, 2, 5, -1, -1, -1, -1])
    # arbitrary shape codes and shuffled offsets are honoured through shape_map / offsets
    el2 = dict(el)
    el2["shape_map"] = {"hex": 7, "pyramid": 3, "tet": 99}
    el2["shapes"] = np.array([7, 3, 99])
    mesh["topologies"]["mesh"]["elements"] = el2
    assert _one(mesh).cell_types.tolist() == [12, 14, 10]
    el2["shapes"] = np.array([7, 3, 5])
    with pytest.raises(ConnectorError, match="not in shape_map"):
        _one(mesh)


def test_polygons_and_polyhedra() -> None:
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1.5, 1, 0], [0.5, 2, 0], [-0.5, 1, 0], [3, 0, 0], [3, 1, 0.0]]
    )
    poly = {
        "coordsets": {"c": {"type": "explicit", "values": {"x": pts[:, 0], "y": pts[:, 1]}}},
        "topologies": {
            "t": {
                "type": "unstructured",
                "coordset": "c",
                "elements": {
                    "shape": "polygonal",
                    "sizes": [5, 4],
                    "offsets": [0, 5],
                    "connectivity": [0, 1, 2, 3, 4, 1, 5, 6, 2],
                },
            }
        },
        "fields": {"a": {"association": "element", "topology": "t", "values": [7.0, 9.0]}},
    }
    s = _one(poly)
    assert s.kind == "surface" and s.points.shape == (7, 3) and len(s.faces) == 3 + 2
    np.testing.assert_allclose(s.fields["a"].data[:, 0], [7, 7, 7, 9, 9])
    # one hexahedron and one tetrahedron as polyhedra sharing nothing
    hp, _ = hex_grid(1, 1, 1)
    tp = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1.0]])
    hex_faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    tet_faces = [[8, 10, 9], [8, 9, 11], [9, 10, 11], [10, 8, 11]]
    faces = hex_faces + tet_faces
    sizes = [len(f) for f in faces]
    ph = {
        "coordsets": {"c": {"type": "explicit", "values": np.concatenate([hp, tp])}},
        "topologies": {
            "t": {
                "type": "unstructured",
                "coordset": "c",
                "elements": {
                    "shape": "polyhedral",
                    "sizes": [6, 4],
                    "offsets": [0, 6],
                    "connectivity": list(range(10)),
                },
                "subelements": {
                    "shape": "polygonal",
                    "sizes": sizes,
                    "offsets": np.concatenate([[0], np.cumsum(sizes)[:-1]]),
                    "connectivity": np.concatenate(faces),
                },
            }
        },
        "fields": {"v": {"association": "element", "topology": "t", "values": [1.0, 2.0]}},
    }
    s = _one(ph)
    assert s.kind == "mesh" and s.cell_types.tolist() == [42, 42] and s.metadata["polyhedral"]
    assert sorted(v for v in s.cells[0] if v >= 0) == list(range(8))
    assert sorted(v for v in s.cells[1] if v >= 0) == [8, 9, 10, 11]


def test_points_topology_multi_domain_and_catalyst_node() -> None:
    pts, tets = tet_grid(1, 1, 1)
    qp, quads = quad_grid(2, 1)
    vol = fe.unstructured_mesh(pts, tets, "tet", point_fields={"p": pts[:, 0]})
    vol["state"] = {"time": 0.25, "cycle": 12, "domain_id": 3}
    wall = fe.unstructured_mesh(qp, quads, "quad", cell_fields={"q": [1.0, 2.0]})
    probes = {
        "coordsets": {
            "c": {"type": "explicit", "values": {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}}
        },
        "topologies": {"t": {"type": "points", "coordset": "c"}},
        "fields": {"T": {"association": "vertex", "topology": "t", "values": [3.0, 4.0]}},
    }
    samples = mesh_to_samples(
        {"vol": vol, "wall": wall, "probes": probes}, "case", time=0.25, time_index=0
    )
    assert [s.id for s in samples] == ["case__vol", "case__wall", "case__probes"]
    assert [s.kind for s in samples] == ["mesh", "surface", "point_cloud"]
    assert samples[0].metadata["blueprint"] == {"topology": "mesh", "domain": "vol", "domain_id": 3}
    assert blueprint.tree_time(blueprint.to_tree({"vol": vol})) == (0.25, 12)
    # zone and field selection
    only = mesh_to_samples({"vol": vol, "wall": wall}, "case", zones=["wall"], fields=["q"])
    assert [s.zone for s in only] == ["wall"] and list(only[0].fields) == ["q"]
    # a Catalyst execute node: one "mesh" channel and one "multimesh" channel
    node = {
        "catalyst/state/timestep": 4,
        "catalyst/state/time": 0.5,
        "catalyst/channels/grid/type": "mesh",
        "catalyst/channels/grid/data": vol,
        "catalyst/channels/parts/type": "multimesh",
        "catalyst/channels/parts/data": {"wall": wall, "probes": probes},
    }
    tree = blueprint.to_tree(node)
    assert blueprint.catalyst_state(tree) == {"timestep": 4.0, "time": 0.5}
    assert [z.name for z in blueprint.tree_zones(tree)] == ["grid", "parts/wall", "parts/probes"]
    assert blueprint.tree_time(tree) == (0.5, 4)
    bad = dict(node)
    bad["catalyst/channels/grid/type"] = "amrmesh"
    with pytest.raises(UnsupportedSourceError, match="amrmesh"):
        mesh_to_samples(bad)
    # several topologies in one domain become zones "<topology>"
    two = fe.unstructured_mesh(pts, tets, "tet", point_fields={"p": pts[:, 0]})
    two["topologies"]["skin"] = {
        "type": "unstructured",
        "coordset": "coords",
        "elements": {"shape": "tri", "connectivity": [0, 1, 2, 1, 2, 3]},
    }
    two["fields"]["s"] = {"association": "element", "topology": "skin", "values": [5.0, 6.0]}
    zs = mesh_to_samples(two, "x")
    assert [(s.zone, s.kind, sorted(s.fields)) for s in zs] == [
        ("mesh", "mesh", ["p"]),
        ("skin", "surface", ["s"]),
    ]


def test_interleaved_mcarray_and_component_names() -> None:
    pts, tets = tet_grid(1, 1, 1)
    mesh = fe.unstructured_mesh(pts, tets, "tet")
    mesh["fields"] = {
        "U": {
            "association": "vertex",
            "topology": "mesh",
            "values": {"u": pts[:, 0], "v": pts[:, 1], "w": pts[:, 2]},
        },
        "S": {
            "association": "element",
            "topology": "mesh",
            "values": np.arange(len(tets) * 2.0).reshape(-1, 2),
        },
        "n": {"association": "element", "topology": "mesh", "values": np.arange(len(tets))},
    }
    s = _one(mesh)
    assert s.fields["U"].component_names == ["u", "v", "w"]
    np.testing.assert_allclose(s.fields["U"].data, pts)
    assert s.fields["S"].data.shape == (len(tets), 2) and s.fields["S"].data.dtype == np.float32
    assert s.fields["n"].data.dtype.kind == "i"


def test_unsupported_pieces_raise_or_warn_and_never_drop_geometry() -> None:
    pts, tets = tet_grid(1, 1, 1)
    base = fe.unstructured_mesh(pts, tets, "tet", point_fields={"p": pts[:, 0]})

    def variant(**changes: Any) -> dict[str, Any]:
        tree = blueprint.to_tree(base)
        for path, value in changes.items():
            spool._set(tree, path.replace("__", "/"), value)
        return tree

    with pytest.raises(UnsupportedSourceError, match="not Cartesian"):
        _one(
            {
                "coordsets": {
                    "c": {"type": "explicit", "values": {"z": [0.0, 1.0], "r": [0.0, 1.0]}}
                },
                "topologies": {"t": {"type": "points", "coordset": "c"}},
            }
        )
    with pytest.raises(UnsupportedSourceError, match="unsupported element shape"):
        _one(variant(topologies__mesh__elements__shape="nurbs"))
    with pytest.raises(UnsupportedSourceError, match="grid_function"):
        _one(variant(topologies__mesh__grid_function="nodes"))
    with pytest.raises(ConnectorError, match="not a multiple of 4"):
        _one(variant(topologies__mesh__elements__connectivity=np.arange(7)))
    with pytest.raises(ConnectorError, match="outside"):
        _one(variant(topologies__mesh__elements__connectivity=np.array([0, 1, 2, 99])))
    with pytest.raises(ConnectorError, match="values but the topology"):
        _one(variant(fields__p__values=np.zeros(3)))
    with pytest.raises(UnsupportedSourceError, match="association"):
        _one(variant(fields__p__association="face"))
    with pytest.raises(UnsupportedSourceError, match="unknown topology type"):
        _one(variant(topologies__mesh__type="amr"))
    with pytest.raises(UnsupportedSourceError, match="no Blueprint mesh"):
        mesh_to_samples({"a": {"b": 1}})
    with pytest.raises(UnsupportedSourceError, match="strided"):
        _one(
            {
                "coordsets": {
                    "c": {"type": "explicit", "values": {"x": np.zeros(4), "y": np.zeros(4)}}
                },
                "topologies": {
                    "t": {
                        "type": "structured",
                        "coordset": "c",
                        "elements": {"dims": {"i": 1, "j": 1, "offsets": [0, 0]}},
                    }
                },
            }
        )
    # material data is skipped with a warning; the mesh and plain fields survive
    tree = variant(fields__vf__matset="m", fields__vf__matset_values__a=np.zeros(len(tets)))
    tree["matsets"] = {"m": {"topology": "mesh", "volume_fractions": {"a": np.ones(len(tets))}}}
    conv = StepConverter("s")
    (s,) = conv.convert(tree)
    assert list(s.fields) == ["p"] and s.cells is not None
    assert any("matset_values" in w for w in conv.warnings)
    assert any("matsets" in w for w in conv.warnings)
    zones, fields, warnings = blueprint.describe_tree(tree)
    assert zones[0].cell_types == {"tetra": len(tets)} and [f.name for f in fields] == ["p"]
    assert warnings


def test_step_converter_static_moving_and_changing_meshes() -> None:
    pts, tets = tet_grid(1, 1, 1)

    def step(points: np.ndarray, cells: np.ndarray, with_mesh: bool = True) -> dict[str, Any]:
        m = fe.unstructured_mesh(points, cells, "tet", point_fields={"p": points[:, 0]})
        if not with_mesh:
            del m["coordsets"], m["topologies"]
        return m

    conv = StepConverter("s")
    s0 = conv.convert(step(pts, tets), time=0.0, time_index=0)[0]
    s1 = conv.convert(step(pts, tets), time=1.0, time_index=1)[0]  # identical mesh resent
    s2 = conv.convert(step(pts, tets, with_mesh=False), time=2.0, time_index=2)[0]  # omitted
    s3 = conv.convert(step(pts * 1.1, tets), time=3.0, time_index=3)[0]  # moved
    s4 = conv.convert(step(pts, tets[:-1]), time=4.0, time_index=4)[0]  # topology changed
    assert s0.points is not None and s0.cells is not None
    assert s1.points is None and s1.cells is None
    assert s2.points is None and len(s2.fields["p"].data) == len(pts)
    assert s3.points is not None and s3.cells is None
    assert s4.points is not None and s4.cells is not None and len(s4.cells) == len(tets) - 1
    always = StepConverter("s", geometry="always")
    always.convert(step(pts, tets), time=0.0)
    assert always.convert(step(pts, tets, with_mesh=False), time=1.0)[0].cells is not None
    with pytest.raises(UnsupportedSourceError, match="no topologies"):
        StepConverter("s").convert(step(pts, tets, with_mesh=False), time=0.0)


# -- 2. spool protocol, finite read ---------------------------------------------------------------


def write_heat(spool_dir: Path, n: int, **kw: Any) -> None:
    dx, dy = LX / NX, LY / NY
    with fe.Emitter(
        spool_dir,
        solver="tiny-heat",
        solver_version="1.0",
        domain="thermal",
        parameters={"alpha": ALPHA},
        units={"length": "m"},
        **kw,
    ) as em:
        for step in range(n):
            em.write_step(
                step,
                step * DT,
                fe.uniform_mesh(
                    (NX, NY),
                    spacing=(dx, dy),
                    point_fields={"T": heat_exact(step)},
                    units={"T": "K"},
                ),
            )


def test_emitter_protocol_files_are_atomic_and_self_describing(tmp_path: Path) -> None:
    sp = tmp_path / "spool"
    write_heat(sp, 3)
    names = sorted(p.name for p in sp.iterdir())
    assert names == [
        "frontier-stream.done",
        "frontier-stream.json",
        "step_00000000.npz",
        "step_00000001.npz",
        "step_00000002.npz",
    ]
    header = json.loads((sp / "frontier-stream.json").read_text())
    assert header["protocol"] == "frontier-spool" and header["version"] == 1
    assert header["solver"] == "tiny-heat" and header["parameters"] == {"alpha": ALPHA}
    done = json.loads((sp / "frontier-stream.done").read_text())
    assert done["status"] == "complete" and done["n_steps"] == 3 and done["last_step"] == 2
    with np.load(sp / "step_00000001.npz", allow_pickle=False) as z:
        assert "fields/T/values" in z.files and "coordsets/coords/dims/i" in z.files
        meta = json.loads(str(z["__frontier_step__"][()]))
    assert meta["step"] == 1 and meta["prev"] == 0 and meta["time"] == pytest.approx(DT)
    assert meta["strings"]["topologies/mesh/type"] == "uniform"
    assert meta["run_id"] == header["run_id"]
    # the emitter is safe to vendor: numpy and the standard library only
    imported = set()
    for node in ast.walk(ast.parse((FILES / "frontier_emit.py").read_text())):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported - set(sys.stdlib_module_names) == {"numpy"}
    with pytest.raises(FileExistsError, match="restart_from"):
        fe.Emitter(sp)
    em = fe.Emitter(tmp_path / "other")
    em.write_step(5, 0.0, fe.uniform_mesh((2, 2)))
    with pytest.raises(ValueError, match="not after"):
        em.write_step(5, 1.0, fe.uniform_mesh((2, 2)))
    with pytest.raises(RuntimeError), fe.Emitter(tmp_path / "failing") as failing:
        failing.write_step(0, 0.0, fe.uniform_mesh((2, 2)))
        raise RuntimeError("solver blew up")
    failed = json.loads((tmp_path / "failing" / "frontier-stream.done").read_text())
    assert failed["status"] == "failed" and "solver blew up" in failed["message"]


def test_finite_spool_conformance_grid(tmp_path: Path, connector: CatalystConnector) -> None:
    sp = tmp_path / "heat"
    write_heat(sp, 5)
    report = check_connector(connector, sp, tmp_path, min_confidence=1.0)
    desc = report.description
    assert desc.format == "frontier-spool" and desc.solver == "tiny-heat"
    assert desc.solver_version == "1.0" and desc.domain == "thermal"
    assert desc.parameters == {"alpha": ALPHA} and desc.units == {"length": "m"}
    assert desc.time.values == pytest.approx([i * DT for i in range(5)])
    assert desc.zones[0].kind == "grid" and desc.zones[0].shape == [NX, NY]
    assert desc.fields[0].name == "T" and desc.fields[0].units == "K"
    assert desc.metadata["done"] is True and desc.metadata["end"]["status"] == "complete"
    assert report.sample_ids == ["heat"] and report.n_steps == 5
    for i, s in enumerate(report.samples["heat"]):
        assert s.time == pytest.approx(i * DT) and s.time_index == i
        np.testing.assert_allclose(s.fields["T"].data[..., 0], heat_exact(i), rtol=1e-6, atol=1e-7)
    store = open_store(tmp_path / "dataset-catalyst-zarr" / "samples" / "heat" / "grid.zarr")
    assert store["fields"]["T"].shape == (5, NX, NY, 1)
    # time selection on a finished stream; a single selected step stays a step of a transient
    last = list(connector.read(sp, times="last"))
    assert len(last) == 1 and last[0].time_index == 4
    np.testing.assert_allclose(last[0].fields["T"].data[..., 0], heat_exact(4), atol=1e-7)
    assert [s.time_index for s in connector.read(sp, times=slice(0, None, 2))] == [0, 2, 4]


def test_finite_spool_static_mesh_single_step_and_step_file(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    pts, tris = tri_grid(4, 4)
    sp = tmp_path / "pulse"
    with fe.Emitter(sp, solver="tiny-pulse", static_mesh=True) as em:
        for i in range(4):
            em.write_step(
                10 * i,
                0.1 * i,
                fe.unstructured_mesh(pts, tris, "tri", point_fields={"p": pulse(pts, 0.1 * i)}),
            )
    with np.load(sp / "step_00000020.npz") as z:
        assert not any(k.startswith(("coordsets", "topologies")) for k in z.files)
    report = check_connector(connector, sp, tmp_path)
    steps: list[Any] = list(report.samples["pulse"])
    assert [s.points is not None for s in steps] == [True, False, False, False]
    assert steps[0].kind == "surface" and steps[0].faces.shape == (len(tris), 3)
    # reading only a late step still finds the mesh stored with the first one
    (late,) = list(connector.read(sp, times=[3]))
    assert late.points is not None and late.time == pytest.approx(0.3)
    np.testing.assert_allclose(late.fields["p"].data[:, 0], pulse(pts, 0.3), atol=1e-6)
    # one step file on its own is a steady sample
    one = tmp_path / "one"
    with fe.Emitter(one) as em:
        em.write_step(7, 1.5, fe.unstructured_mesh(pts, tris, "tri", point_fields={"p": pts[:, 0]}))
    rep = check_connector(connector, one, tmp_path / "o1")
    assert rep.description.time.steady and rep.first().time is None
    assert _detect(connector, one / "step_00000007.npz").confidence == 0.9
    (single,) = list(connector.read(one / "step_00000007.npz"))
    assert single.id == "one" and single.time is None
    # an empty spool explains itself
    (tmp_path / "empty").mkdir()
    assert connector.detect(tmp_path / "empty") is None
    with pytest.raises(UnsupportedSourceError, match="no steps yet") as err:
        list(connector.read(tmp_path / "empty"))
    assert "follow=true" in str(err.value)


def test_truncated_step_in_a_finished_spool_is_an_error(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    sp = tmp_path / "heat"
    write_heat(sp, 3)
    victim = sp / "step_00000001.npz"
    victim.write_bytes(victim.read_bytes()[:200])
    with pytest.raises(ConnectorError, match="truncated or corrupt"):
        list(connector.read(sp))
    with pytest.raises(ConnectorError, match="truncated or corrupt"):
        list(connector.read(sp, options={"follow": True, "timeout": 5, "poll": 0.01}))


# -- 3. the follower, scripted (no threads, fake clock) -------------------------------------------


class Script:
    """Drives a follower deterministically: every poll advances a fake clock and runs the next
    scripted action (what the "solver" does between two looks of the reader)."""

    def __init__(self, *actions: Any, dt: float = 0.1) -> None:
        self.actions = list(actions)
        self.now = 0.0
        self.dt = dt
        self.polls = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, _seconds: float) -> None:
        self.now += self.dt
        self.polls += 1
        if self.actions:
            action = self.actions.pop(0)
            if action is not None:
                action()

    def options(self, **kw: Any) -> FollowOptions:
        fo = FollowOptions.from_options({"follow": True, "timeout": 5.0, **kw})
        fo.clock, fo.sleep = self.clock, self.sleep
        return fo


def staged_steps(tmp_path: Path, n: int, **kw: Any) -> tuple[Path, list[Path], Any]:
    """Steps written into a staging directory, to be moved into the spool one by one."""
    stage = tmp_path / "stage"
    em = fe.Emitter(stage, **kw)
    files = [
        em.write_step(i, 0.5 * i, fe.uniform_mesh((3, 2), point_fields={"f": np.full(6, float(i))}))
        for i in range(n)
    ]
    sp = tmp_path / "spool"
    sp.mkdir()
    (stage / "frontier-stream.json").rename(sp / "frontier-stream.json")
    return sp, files, em


def test_follower_yields_in_order_and_stops_at_done_marker(tmp_path: Path) -> None:
    sp, files, _ = staged_steps(tmp_path, 4)
    files[0].rename(sp / files[0].name)

    def arrive(i: int) -> Any:
        return lambda: files[i].rename(sp / files[i].name)

    def finish() -> None:
        (sp / "frontier-stream.done").write_text(json.dumps({"status": "complete"}))

    script = Script(None, arrive(1), None, None, lambda: (arrive(2)(), arrive(3)()), None, finish)
    src = spool.SpoolSource(sp)
    seen = []
    for step in src.follow(script.options()):
        seen.append((step.index, script.polls, float(step.tree["fields"]["f"]["values"][0])))
    assert [s[0] for s in seen] == [0, 1, 2, 3]
    assert [s[2] for s in seen] == [0.0, 1.0, 2.0, 3.0]
    # each step was delivered as soon as it was complete, not at the end
    assert [s[1] for s in seen] == [0, 2, 5, 5]
    assert src.end["reason"] == "done" and src.end["status"] == "complete" and script.polls == 7
    assert src.events == []


def test_follower_never_reads_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sp, files, _ = staged_steps(tmp_path, 3)
    payload = {i: f.read_bytes() for i, f in enumerate(files)}
    opened: list[str] = []
    real_load = spool.load_entry

    def spy(entry: spool.Entry) -> Any:
        opened.append(entry.path.name)
        return real_load(entry)

    monkeypatch.setattr(spool, "load_entry", spy)
    tmp1 = sp / ".step_00000001.npz.tmp-999"

    def begin_atomic() -> None:  # a slow writer following the protocol: hidden temporary file
        tmp1.write_bytes(payload[1][: len(payload[1]) // 2])

    def end_atomic() -> None:
        tmp1.write_bytes(payload[1])
        tmp1.rename(sp / "step_00000001.npz")

    def begin_in_place() -> None:  # a writer that does NOT rename: partial file is visible
        (sp / "step_00000002.npz").write_bytes(payload[2][:300])

    def more_in_place() -> None:
        (sp / "step_00000002.npz").write_bytes(payload[2][: len(payload[2]) - 40])

    def end_in_place() -> None:
        (sp / "step_00000002.npz").write_bytes(payload[2])
        (sp / "frontier-stream.done").write_text("")

    files[0].rename(sp / files[0].name)
    script = Script(
        begin_atomic, None, end_atomic, begin_in_place, None, more_in_place, None, end_in_place
    )
    src = spool.SpoolSource(sp)
    got = [
        (s.index, float(s.tree["fields"]["f"]["values"][0]), s.tree["fields"]["f"]["values"].shape)
        for s in src.follow(script.options(partial_grace=10))
    ]
    assert got == [(0, 0.0, (6,)), (1, 1.0, (6,)), (2, 2.0, (6,))]
    # the hidden temporary file was never opened; the in-place file was retried until whole
    assert not any(name.startswith(".") for name in opened)
    assert opened.count("step_00000001.npz") == 1
    assert opened.count("step_00000002.npz") >= 3
    assert src.end["reason"] == "done"


def test_follower_gives_up_on_a_file_that_stays_broken(tmp_path: Path) -> None:
    sp, files, _ = staged_steps(tmp_path, 2)
    files[0].rename(sp / files[0].name)
    (sp / "step_00000001.npz").write_bytes(files[1].read_bytes()[:100])
    script = Script()
    src = spool.SpoolSource(sp)
    it = src.follow(script.options(partial_grace=1.0, timeout=60))
    assert next(it).index == 0
    with pytest.raises(ConnectorError, match="truncated or corrupt"):
        next(it)
    assert 1.0 < script.now < 2.0  # after the grace period, long before the idle timeout


def test_follower_handles_out_of_order_arrival_and_lost_steps(tmp_path: Path) -> None:
    sp, files, _ = staged_steps(tmp_path, 5)

    def arrive(i: int) -> Any:
        return lambda: files[i].rename(sp / files[i].name)

    def finish() -> None:
        (sp / "frontier-stream.done").write_text("{}")

    files[0].rename(sp / files[0].name)
    # step 2 shows up before step 1 (network file system); step 3 is lost for good
    script = Script(arrive(2), None, arrive(1), None, arrive(4), *([None] * 30), finish)
    src = spool.SpoolSource(sp)
    seen = [(s.index, round(script.now, 1)) for s in src.follow(script.options(order_grace=1.0))]
    assert [i for i, _ in seen] == [0, 1, 2, 4]
    assert seen[1][1] == seen[2][1] == 0.3  # 2 was held back until 1 arrived
    assert 1.4 < seen[3][1] < 1.8  # 4 was released after order_grace without 3
    assert src.events == ["step 3 (before 4) never arrived; continuing without it"]


def test_follower_timeout_semantics(tmp_path: Path) -> None:
    sp, files, _ = staged_steps(tmp_path, 2)
    for f in files:
        f.rename(sp / f.name)
    # the solver died after two steps: no end marker ever comes
    script = Script()
    src = spool.SpoolSource(sp)
    got = [s.index for s in src.follow(script.options(timeout=3.0))]
    assert got == [0, 1] and src.end == {"reason": "timeout", "n_steps": 2, "last_step": 1}
    assert 3.0 < script.now < 3.3 and "no new complete step" in src.events[0]
    strict = spool.SpoolSource(sp).follow(Script().options(timeout=3.0, on_timeout="error"))
    assert [next(strict).index, next(strict).index] == [0, 1]
    with pytest.raises(StreamTimeoutError, match="no new complete step") as err:
        next(strict)
    assert "on_timeout=stop" in str(err.value)
    # until=timeout ignores the end marker (writers that never write one) and ends quietly
    (sp / "frontier-stream.done").write_text("{}")
    quiet = spool.SpoolSource(sp)
    assert [s.index for s in quiet.follow(Script().options(timeout=1.0, until="timeout"))] == [0, 1]
    assert quiet.end["reason"] == "timeout" and quiet.events == []
    assert [s.index for s in spool.SpoolSource(sp).follow(Script().options(max_steps=1))] == [0]
    # nothing ever arrives
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(StreamTimeoutError, match="no step appeared"):
        spool.SpoolSource(empty).wait_for_first(Script().options(timeout=2.0))
    with pytest.raises(StreamTimeoutError):
        list(spool.SpoolSource(empty).follow(Script().options(timeout=2.0)))
    # a failed run is reported, the steps are kept
    (sp / "frontier-stream.done").write_text(json.dumps({"status": "failed", "message": "NaN"}))
    failed = spool.SpoolSource(sp)
    assert len(list(failed.follow(Script().options()))) == 2
    assert failed.end["status"] == "failed" and "NaN" in failed.events[0]
    for bad in ({"until": "never"}, {"on_timeout": "ignore"}, {"timeout": "soon"}, {"poll": -1}):
        with pytest.raises(ConnectorError, match="bad option"):
            FollowOptions.from_options(bad)
    assert FollowOptions.from_options({"until": "done-marker", "follow": "true"}).follow


def test_follower_and_emitter_handle_a_solver_restart(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    sp = tmp_path / "run"

    def mesh(value: float) -> dict[str, Any]:
        return fe.uniform_mesh((3, 2), point_fields={"f": np.full(6, value)})

    first = fe.Emitter(sp, stream_id="run")
    for i in range(5):
        first.write_step(i, float(i), mesh(float(i)))
    # ... the solver crashes; it restarts from the checkpoint of step 2 and recomputes 3, 4, 5

    def restart() -> None:
        again = fe.Emitter(sp, restart_from=2)
        assert again.prev == 2 and sorted(p.name for p in sp.glob("step_*")) == [
            "step_00000000.npz",
            "step_00000001.npz",
            "step_00000002.npz",
        ]
        for i in range(3, 7):
            again.write_step(i, float(i), mesh(100.0 + i))
        again.close()

    script = Script(None, restart)
    src = spool.SpoolSource(sp)
    got = [
        (s.index, float(s.tree["fields"]["f"]["values"][0])) for s in src.follow(script.options())
    ]
    # steps 0..4 of the first run were already consumed; only what is new is appended
    assert got == [(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 105.0), (6, 106.0)]
    assert len(src.events) == 1 and "restarted" in src.events[0]
    header = json.loads((sp / "frontier-stream.json").read_text())
    assert header["stream_id"] == "run" and header["restarts"][0]["from_step"] == 2
    # a reader that starts after the restart sees one consistent run
    values = [float(s.fields["f"].data[0, 0, 0]) for s in connector.read(sp)]
    assert values == [0.0, 1.0, 2.0, 103.0, 104.0, 105.0, 106.0]
    assert connector.describe(sp).metadata["restarts"][0]["from_step"] == 2
    # on_restart=error refuses to continue
    sp2 = tmp_path / "run2"
    em = fe.Emitter(sp2)
    em.write_step(0, 0.0, mesh(0.0))
    strict = spool.SpoolSource(sp2).follow(
        Script(lambda: fe.Emitter(sp2, restart_from=0)).options(on_restart="error")
    )
    assert next(strict).index == 0
    with pytest.raises(ConnectorError, match="restarted"):
        next(strict)


# -- 4. live: a solver subprocess followed while it runs ------------------------------------------


def test_live_follow_grid_solver_builds_dataset_while_it_runs(tmp_path: Path) -> None:
    sp, out = tmp_path / "heat-run", tmp_path / "dataset"
    n = 8
    # the reader starts first: the solver creates the spool half a second later
    proc = start_solver(tmp_path, spool=sp, n_steps=n, pause=0.04, start_delay=0.3)
    try:
        t0 = time.monotonic()
        result = importers.run_import(
            "catalyst",
            str(sp),
            out,
            download_dir=tmp_path / "work",
            options={"timeout": 30, "poll": 0.01},
            overwrite=True,
        )
        elapsed = time.monotonic() - t0
    finally:
        assert proc.wait(timeout=30) == 0
    ds = result["dataset"]
    assert ds["sample_ids"] == ["heat-run"] and ds["n_steps"] == {"heat-run": n}
    assert (tmp_path / "work" / POINTER_NAME).is_file()
    assert elapsed > 0.3 + (n - 1) * 0.04  # it really waited for the solver
    store = open_store(out / "samples" / "heat-run" / "grid.zarr")
    temp = np.asarray(store["fields"]["T"])
    assert temp.shape == (n, NX, NY, 1)
    for step in range(n):  # every step, in order, equal to the solver state
        np.testing.assert_allclose(temp[step, ..., 0], heat_exact(step), rtol=1e-5, atol=1e-6)
    assert np.asarray(store["fields"]["gradT"]).shape == (n, NX, NY, 2)
    np.testing.assert_allclose(np.asarray(store["coords"]["x"]), np.arange(NX) * LX / NX)
    meta = json.loads((out / "samples" / "heat-run" / "meta.json").read_text())
    assert meta["solver"] == "tiny-heat" and meta["solver_version"] == "1.0"
    times = json.loads((out / "manifest.json").read_text())["samples"][0]
    assert times["n_steps"] == n
    if frontier_data_available():
        validate_with_frontier_data(out)


def test_live_follow_unstructured_solver_with_moving_field(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    sp, out = tmp_path / "pulse-run", tmp_path / "dataset"
    n, cells = 6, 6
    proc = start_solver(tmp_path, spool=sp, kind="pulse", n=cells, n_steps=n, pause=0.04)
    try:
        # describe waits for the stream, then the dataset is written while following
        desc = connector.describe(sp, options={**LIVE, "timeout": 30})
        from frontier_connectors.convert import to_frontier_dataset

        res = to_frontier_dataset(
            connector.read(sp, options={**LIVE, "timeout": 30}),
            out,
            description=desc,
            backend="zarr",
        )
    finally:
        assert proc.wait(timeout=30) == 0
    assert res.n_steps == {"pulse-run": n} and desc.metadata["live"] is True
    assert desc.zones[0].kind == "surface" and desc.zones[0].n_points == (cells + 1) ** 2
    assert {f.name: (f.location, f.components, f.units) for f in desc.fields} == {
        "p": ("point", 1, "Pa"),
        "U": ("point", 3, "m/s"),
        "pc": ("cell", 1, ""),
    }
    store = open_store(out / "samples" / "pulse-run" / "surface.zarr")
    pts = np.asarray(store["points"])
    faces = np.asarray(store["faces"])
    assert pts.shape == ((cells + 1) ** 2, 3) and faces.shape == (2 * cells * cells, 3)
    p = np.asarray(store["point_fields"]["p"]) if "point_fields" in store else None
    groups = {name: store[name] for name in store.group_keys()}
    fields = {k: np.asarray(a) for g in groups.values() for k, a in g.arrays()}
    assert fields["p"].shape == (n, len(pts), 1) and fields["pc"].shape == (n, len(faces), 1)
    cen = pts[faces].mean(axis=1)
    for step in range(n):
        t = 0.1 * step
        np.testing.assert_allclose(fields["p"][step, :, 0], pulse(pts, t), atol=1e-6)
        np.testing.assert_allclose(fields["pc"][step, :, 0], pulse(cen, t), atol=1e-6)
    # the pulse really moves: its maximum travels in +x
    peak_x = [float(pts[np.argmax(fields["p"][s, :, 0]), 0]) for s in (0, n - 1)]
    assert peak_x[1] > peak_x[0]
    np.testing.assert_allclose(fields["U"][0, 0], [0.5, 0.0, 0.0])
    assert "points_t" not in store  # static mesh: stored once
    if frontier_data_available():
        validate_with_frontier_data(out)
    del p


def test_live_follow_moving_mesh_stores_points_per_step(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    sp, out = tmp_path / "moving", tmp_path / "dataset"
    proc = start_solver(
        tmp_path, spool=sp, kind="pulse", n=3, n_steps=4, pause=0.02, static=False, moving=True
    )
    try:
        res = importers.run_import(
            "catalyst",
            str(sp),
            out,
            download_dir=tmp_path / "work",
            options={"timeout": 30, "poll": 0.01},
            backend="zarr",
        )["dataset"]
    finally:
        assert proc.wait(timeout=30) == 0
    assert res["n_steps"] == {"moving": 4}
    store = open_store(out / "samples" / "moving" / "surface.zarr")
    pts_t = np.asarray(store["points_t"])
    assert pts_t.shape == (4, 16, 3)
    np.testing.assert_allclose(pts_t[3, :, 1] - pts_t[0, :, 1], 0.05 * 0.3, atol=1e-6)


def test_live_solver_dies_follower_times_out_cleanly(tmp_path: Path) -> None:
    sp, out = tmp_path / "crash", tmp_path / "dataset"
    proc = start_solver(tmp_path, spool=sp, n_steps=50, pause=0.02, die_after=3)
    # the solver crashes (exit code 3) after three steps, without an end marker; the follower
    # is started on what it left behind so that the outcome does not depend on machine load
    assert proc.wait(timeout=60) == 3
    t0 = time.monotonic()
    res = convert(
        sp,
        out,
        connector="catalyst",
        options={"follow": True, "timeout": 0.5, "poll": 0.01},
        backend="zarr",
    )
    assert 0.5 < time.monotonic() - t0 < 30  # it waited for the timeout, then stopped
    assert any("no new complete step" in w and "on_timeout=stop" in w for w in res.warnings)
    assert not (sp / "frontier-stream.done").exists()
    assert res.n_steps == {"crash": 3}  # what arrived before the crash is kept and valid
    temp = np.asarray(open_store(out / "samples" / "crash" / "grid.zarr")["fields"]["T"])
    np.testing.assert_allclose(temp[2, ..., 0], heat_exact(2), rtol=1e-5, atol=1e-6)
    if frontier_data_available():
        validate_with_frontier_data(out)
    # on_timeout=error: a clear error and no half-written dataset left behind
    with pytest.raises(StreamTimeoutError, match="no new complete step") as err:
        convert(
            sp,
            tmp_path / "strict",
            connector="catalyst",
            backend="zarr",
            options={"follow": True, "timeout": 0.2, "poll": 0.01, "on_timeout": "error"},
        )
    assert "end marker" in str(err.value) and not (tmp_path / "strict").exists()
    # a restarted solver continues the same stream; a late follower gets one coherent run
    proc = start_solver(tmp_path, spool=sp, n_steps=6, pause=0.01, restart_from=1, first_step=2)
    try:
        res = convert(
            sp,
            tmp_path / "resumed",
            connector="catalyst",
            backend="zarr",
            options={"follow": True, "timeout": 30, "poll": 0.01},
        )
    finally:
        assert proc.wait(timeout=30) == 0
    assert res.n_steps == {"crash": 6}
    temp = np.asarray(
        open_store(tmp_path / "resumed" / "samples" / "crash" / "grid.zarr")["fields"]["T"]
    )
    for step in range(6):
        np.testing.assert_allclose(temp[step, ..., 0], heat_exact(step), rtol=1e-5, atol=1e-6)


def test_live_slow_writer_thread_is_never_read_half_written(
    tmp_path: Path, connector: CatalystConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real concurrent writer that dribbles every step file out in small chunks."""
    sp = tmp_path / "slow"
    stage, files, em = staged_steps(tmp_path, 4)
    del stage
    (tmp_path / "spool").rename(sp)
    em.close()
    payloads = [f.read_bytes() for f in files]
    done_doc = (tmp_path / "stage" / "frontier-stream.done").read_bytes()

    def slow_writer() -> None:
        for i, data in enumerate(payloads):
            final = sp / f"step_{i:08d}.npz"
            # even steps: protocol (temporary + rename); odd steps: written in place
            target = sp / f".{final.name}.tmp-1" if i % 2 == 0 else final
            with open(target, "wb") as f:
                for k in range(0, len(data), max(len(data) // 6, 1)):
                    f.write(data[k : k + max(len(data) // 6, 1)])
                    f.flush()
                    time.sleep(0.01)
            if target != final:
                target.rename(final)
        (sp / "frontier-stream.done").write_bytes(done_doc)

    partial_seen = []
    real_load = spool.load_entry

    def spy(entry: spool.Entry) -> Any:
        try:
            return real_load(entry)
        except spool.PartialStepError:
            partial_seen.append(entry.path.name)
            raise

    monkeypatch.setattr(spool, "load_entry", spy)
    writer = threading.Thread(target=slow_writer)
    writer.start()
    try:
        steps = list(connector.read(sp, options={"follow": True, "timeout": 20, "poll": 0.002}))
    finally:
        writer.join(timeout=20)
    assert [s.time_index for s in steps] == [0, 1, 2, 3]
    for i, s in enumerate(steps):  # every array complete and correct
        assert s.fields["f"].data.shape == (3, 2, 1)
        np.testing.assert_array_equal(s.fields["f"].data, np.full((3, 2, 1), float(i)))
    # only files written in place can ever be seen partially; they were retried, never yielded
    assert set(partial_seen) <= {"step_00000001.npz", "step_00000003.npz"}


def test_live_time_selection_rules(tmp_path: Path, connector: CatalystConnector) -> None:
    sp = tmp_path / "heat"
    write_heat(sp, 6)
    live = {"follow": True, "timeout": 5, "poll": 0.01}
    assert [s.time_index for s in connector.read(sp, times=slice(1, 5, 2), options=live)] == [1, 3]
    assert len(list(connector.read(sp, times="all", options=live))) == 6
    with pytest.raises(SelectionError, match="live stream"):
        list(connector.read(sp, times="last", options=live))
    with pytest.raises(ConnectorError, match="geometry"):
        list(connector.read(sp, options={"geometry": "sometimes"}))


def test_changing_topology_needs_time_as_samples(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    sp = tmp_path / "amr"
    with fe.Emitter(sp) as em:
        for i, n in enumerate((2, 3, 4)):
            pts, tris = tri_grid(n, n)
            em.write_step(
                i, 0.1 * i, fe.unstructured_mesh(pts, tris, "tri", point_fields={"p": pts[:, 0]})
            )
    with pytest.raises(ConnectorError, match="time_as_samples"):
        convert(sp, tmp_path / "bad", backend="zarr")
    res = convert(
        sp, tmp_path / "ok", backend="zarr", time_as_samples=True, options={"geometry": "always"}
    )
    assert res.sample_ids == ["amr_t00000", "amr_t00001", "amr_t00002"]
    if frontier_data_available():
        validate_with_frontier_data(tmp_path / "ok")


# -- 5. importer and CLI --------------------------------------------------------------------------


def test_importer_is_registered_plans_and_runs_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = {r["id"]: r for r in importers.list_importers()}
    assert rows["catalyst"]["status"] == "available" and rows["catalyst"]["credentials_env"] == []
    assert registry.get("catalyst").maturity == "verified-synthetic"
    info = registry.info(registry.get("catalyst"))
    assert info.status == "available" and "vtk" in info.export_help
    sp = tmp_path / "spool"
    plan = importers.run_import("catalyst", str(sp), dry_run=True, options={"timeout": 7})
    calls = plan["calls"]
    assert plan["dry_run"] and not sp.exists()  # a dry run touches nothing
    assert [c["method"] for c in calls] == ["FILE", "FILE", "FILE"]
    assert calls[1]["params"] == {
        "follow": True,
        "timeout_s": 7.0,
        "poll_s": 0.2,
        "until": "done-marker",
        "on_timeout": "stop",
    }
    assert calls[2]["target"].endswith("frontier-stream.done")
    sst = importers.run_import("catalyst", str(tmp_path / "run.sst"), dry_run=True)["calls"]
    assert sst[0]["method"] == "SDK" and sst[0]["params"]["engine"] == "SST"
    assert sst[0]["params"]["path"].endswith("/run")
    with pytest.raises(StreamTimeoutError, match="did not appear"):
        importers.run_import(
            "catalyst",
            str(sp),
            tmp_path / "x",
            download_dir=tmp_path / "w0",
            options={"timeout": 0.05, "poll": 0.01},
        )
    write_heat(sp, 3)
    out = tmp_path / "ds"
    code = cli_main(
        [
            "import",
            "catalyst",
            str(sp),
            "--out",
            str(out),
            "--download-dir",
            str(tmp_path / "work"),
            "--option",
            "follow=true",
            "--option",
            "timeout=5",
            "--option",
            "poll=0.01",
            "--json",
        ]
    )
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["dataset"]["n_steps"] == {"spool": 3}
    pointer = json.loads((tmp_path / "work" / POINTER_NAME).read_text())
    assert pointer["source"] == str(sp.resolve()) and pointer["options"]["follow"] is True
    det = registry.detect(tmp_path / "work")
    assert det[0].connector_id == "catalyst" and det[0].confidence == 1.0
    code = cli_main(["import", "catalyst", str(sp), "--dry-run"])
    assert code == 0 and "would make 3 call(s)" in capsys.readouterr().out
    assert registry.detect(sp)[0].connector_id == "catalyst"


def test_exporter_bundle_is_shipped() -> None:
    names = {p.name for p in FILES.iterdir()}
    assert {
        "frontier_emit.py",
        "frontier_catalyst_pipeline.py",
        "adaptor_example.txt",
        "README.txt",
    } <= names
    pipeline = (FILES / "frontier_catalyst_pipeline.py").read_text()
    assert pipeline.startswith("# SPDX-License-Identifier: Apache-2.0")
    assert "PVTrivialProducer" in pipeline and "catalyst_execute" in pipeline
    notes = (FILES / "adaptor_example.txt").read_text()
    for needle in (
        "CATALYST_IMPLEMENTATION_NAME=paraview",
        "CATALYST_IMPLEMENTATION_PATHS",
        "catalyst/channels/grid",
        "catalyst/state/timestep",
    ):
        assert needle in notes


def test_pipeline_vtk_to_blueprint_round_trip() -> None:
    """The VTK -> Blueprint half of the Catalyst pipeline script, with plain VTK (the ParaView
    half cannot run here)."""
    pytest.importorskip("vtkmodules")
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkImageData,
        vtkMultiBlockDataSet,
        vtkPolyData,
        vtkUnstructuredGrid,
    )

    from frontier_connectors import _vtk
    from frontier_connectors.testing import analytic_fields, make_ugrid

    pipe = _load("frontier_catalyst_pipeline")
    assert pipe._HAVE_PARAVIEW is False
    pts, tets = tet_grid(2, 1, 1)
    f = analytic_fields(pts)
    ug = make_ugrid(pts, tets, 10, point_data=f, cell_data={"c": np.arange(len(tets), dtype=float)})
    got = _one(pipe.vtk_to_blueprint(ug))
    want = _vtk.dataset_to_sample(ug, "s")
    assert got.kind == want.kind == "mesh"
    np.testing.assert_array_equal(got.cells, want.cells)
    np.testing.assert_array_equal(got.cell_types, want.cell_types)
    for name in ("p", "U", "c"):
        np.testing.assert_allclose(got.fields[name].data, want.fields[name].data)
        assert got.fields[name].location == want.fields[name].location
    # mixed cells
    hp, hexes = hex_grid(1, 1, 1)
    mixed = vtkUnstructuredGrid()
    vp = vtkPoints()
    vp.SetData(numpy_to_vtk(np.concatenate([hp, [[0.5, 0.5, 2.0]]]), deep=True))
    mixed.SetPoints(vp)
    mixed.InsertNextCell(12, 8, [int(v) for v in hexes[0]])
    mixed.InsertNextCell(14, 5, [4, 5, 6, 7, 8])
    m = _one(pipe.vtk_to_blueprint(mixed))
    assert m.cell_types.tolist() == [12, 14] and m.cells[1].tolist() == [4, 5, 6, 7, 8, -1, -1, -1]
    # image data
    img = vtkImageData()
    img.SetDimensions(3, 2, 2)
    img.SetOrigin(1.0, 0.0, 0.0)
    img.SetSpacing(0.5, 1.0, 2.0)
    arr = numpy_to_vtk(np.arange(12, dtype=float), deep=True)
    arr.SetName("T")
    img.GetPointData().AddArray(arr)
    g, w = _one(pipe.vtk_to_blueprint(img)), _vtk.dataset_to_sample(img, "s")
    assert g.kind == "grid"
    np.testing.assert_allclose(g.fields["T"].data, w.fields["T"].data)
    for ax in "xyz":
        np.testing.assert_allclose(g.coords[ax], w.coords[ax])
    # polydata with triangles and a quad, inside a multiblock
    qp, quads = quad_grid(2, 1)
    pd = vtkPolyData()
    pp = vtkPoints()
    pp.SetData(numpy_to_vtk(qp, deep=True))
    pd.SetPoints(pp)
    polys = vtkCellArray()
    polys.InsertNextCell(4, [int(v) for v in quads[0]])
    polys.InsertNextCell(3, [int(v) for v in quads[1][:3]])
    pd.SetPolys(polys)
    mb = vtkMultiBlockDataSet()
    mb.SetBlock(0, ug)
    mb.SetBlock(1, pd)
    tree = pipe.data_object_to_blueprint(mb)
    samples = _samples(tree, "mb")
    assert [(s.zone, s.kind) for s in samples] == [("block0", "mesh"), ("block1", "surface")]
    assert samples[1].faces.shape == (3, 3)
    # and the emitter accepts what the pipeline produces
    arrays, strings = fe.flatten(tree)
    assert (
        "block0/fields/U/values" in arrays
        and strings["block1/topologies/mesh/type"] == "unstructured"
    )


# -- 6. Conduit relay files -----------------------------------------------------------------------


def _char(text: str) -> np.ndarray:
    return np.frombuffer(text.encode() + b"\x00", dtype="S1")


def write_relay_hdf5(
    directory: Path, base: str, cycle: int, t: float, domains: list[dict[str, Any]]
) -> Path:
    """Write the layout of ``conduit.relay.io.blueprint.save_mesh(..., "hdf5")`` with h5py
    (strings as NUL-terminated ``|S1`` arrays, scalars as arrays of one element)."""
    import h5py

    def put(group: Any, tree: dict[str, Any]) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                put(group.create_group(key), value)
            elif isinstance(value, str):
                group.create_dataset(key, data=_char(value))
            else:
                group.create_dataset(key, data=np.atleast_1d(np.asarray(value)))

    root = directory / f"{base}.cycle_{cycle:06d}.root"
    multi = len(domains) > 1
    index = {"mesh": {"state": {"cycle": cycle, "time": t, "number_of_domains": len(domains)}}}
    with h5py.File(root, "w") as f:
        put(
            f,
            {
                "blueprint_index": index,
                "protocol": {"name": "hdf5", "version": "0.9.5"},
                "number_of_files": len(domains),
                "number_of_trees": len(domains),
                "file_pattern": f"{root.stem}/domain_%06d.hdf5" if multi else root.name,
                "tree_pattern": "/",
            },
        )
        if not multi:
            put(f, {"mesh": domains[0]})
    if multi:
        sub = directory / root.stem
        sub.mkdir()
        for i, dom in enumerate(domains):
            with h5py.File(sub / f"domain_{i:06d}.hdf5", "w") as f:
                put(f, {"mesh": dom})
    return root


def test_relay_hdf5_series_finite_and_followed(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    pytest.importorskip("h5py")
    sp = tmp_path / "relay"
    sp.mkdir()
    pts, tets = tet_grid(1, 1, 1)
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for i in range(3):
        left = fe.unstructured_mesh(pts, tets, "tet", point_fields={"p": pts[:, 0] + i})
        right = fe.unstructured_mesh(
            pts + [1.0, 0, 0], tets, "tet", point_fields={"p": pts[:, 0] + 10 + i}
        )
        for d, dom in enumerate((left, right)):
            dom["state"] = {"cycle": 10 * i, "time": 0.5 * i, "domain_id": d}
        write_relay_hdf5(sp, "run", 10 * i, 0.5 * i, [left, right])
    assert _detect(connector, sp).format == "conduit-relay"
    report = check_connector(connector, sp, tmp_path, min_confidence=0.8)
    assert report.description.format == "conduit-relay"
    assert report.description.time.values == [0.0, 0.5, 1.0]
    assert report.sample_ids == ["run__domain_000000", "run__domain_000001"]
    right_steps = report.samples["run__domain_000001"]
    assert right_steps[0].metadata["blueprint"]["domain_id"] == 1
    np.testing.assert_allclose(right_steps[2].fields["p"].data[:, 0], pts[:, 0] + 12)
    assert [s.points is not None for s in right_steps] == [True, False, False]
    # a single root file is a steady sample
    one = connector.describe(sp / "run.cycle_000010.root")
    assert one.time.steady and len(one.zones) == 2
    # followed: relay writers do not rename, so a root is read once it has settled; there is
    # no end marker in this layout, the follower ends by timeout (or a touched marker)
    live = list(
        connector.read(
            sp,
            zones=["domain_000000"],
            options={
                "follow": True,
                "until": "timeout",
                "timeout": 0.3,
                "poll": 0.01,
                "settle": 0.05,
            },
        )
    )
    assert [s.time for s in live] == [0.0, 0.5, 1.0]
    (sp / "frontier-stream.done").touch()
    assert (
        len(list(connector.read(sp, zones=["domain_000000"], options={**LIVE, "timeout": 5}))) == 3
    )
    # a root that refers to a domain file which is not there yet is not read
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    write_relay_hdf5(orphan, "run", 0, 0.0, [left, right])
    (orphan / "run.cycle_000000" / "domain_000001.hdf5").unlink()
    with pytest.raises(relay.IncompleteFileError, match="does not exist"):
        relay.load_root(orphan / "run.cycle_000000.root")
    with pytest.raises(ConnectorError, match="truncated or corrupt"):
        list(connector.read(orphan))


def test_relay_json_protocols_without_conduit(tmp_path: Path, connector: CatalystConnector) -> None:
    pts, tets = tet_grid(1, 1, 1)
    mesh = {
        "state": {"time": 2.0, "cycle": 4},
        "coordsets": {
            "coords": {
                "type": "explicit",
                "values": {a: pts[:, i].tolist() for i, a in enumerate("xyz")},
            }
        },
        "topologies": {
            "mesh": {
                "type": "unstructured",
                "coordset": "coords",
                "elements": {"shape": "tet", "connectivity": tets.reshape(-1).tolist()},
            }
        },
        "fields": {
            "p": {"association": "vertex", "topology": "mesh", "values": pts[:, 0].tolist()}
        },
    }
    doc = {
        "mesh": mesh,
        "blueprint_index": {"mesh": {"state": {"cycle": 4, "time": 2.0}}},
        "protocol": {"name": "json"},
        "number_of_files": 1,
        "number_of_trees": 1,
        "file_pattern": "one.cycle_000004.root",
        "tree_pattern": "/",
    }
    root = tmp_path / "one.cycle_000004.root"
    root.write_text(json.dumps(doc))
    assert relay.is_blueprint_root(root) and _detect(connector, root).confidence == 0.85
    (s,) = list(connector.read(root))
    assert s.id == "one" and s.kind == "mesh"
    np.testing.assert_array_equal(s.cells, tets)

    # conduit_json: every leaf carries its dtype
    def leaf(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: leaf(x) for k, x in v.items()}
        if isinstance(v, str):
            return {"dtype": "char8_str", "number_of_elements": len(v) + 1, "value": v}
        a = np.atleast_1d(np.asarray(v))
        return {
            "dtype": "float64" if a.dtype.kind == "f" else "int64",
            "number_of_elements": int(a.size),
            "value": a.tolist() if a.size > 1 else a[0].item(),
        }

    typed = tmp_path / "typed.root"
    typed.write_text(json.dumps(leaf(doc)))
    (t,) = list(connector.read(typed))
    np.testing.assert_array_equal(t.cells, tets)
    np.testing.assert_allclose(t.fields["p"].data[:, 0], pts[:, 0])
    # not ours: a CERN ROOT file is not claimed
    cern = tmp_path / "events.root"
    cern.write_bytes(b"root\x00\x00\xf3\x8c" + bytes(200))
    assert connector.detect(cern) is None
    half = tmp_path / "half.cycle_000001.root"
    half.write_text(json.dumps(doc)[:400])
    with pytest.raises(relay.IncompleteFileError):
        relay.load_root(half)


# -- 7. real Conduit (LLNL; not on PyPI: built from source in a scratch environment) --------------


def _conduit() -> Any:
    conduit = pytest.importorskip("conduit")
    if not hasattr(conduit, "Node"):
        pytest.skip("the installed `conduit` is not LLNL Conduit")
    pytest.importorskip("conduit.blueprint")
    return conduit  # conduit.blueprint is now loaded as an attribute


BRAID = {
    "uniform": "grid",
    "rectilinear": "grid",
    "structured": "mesh",
    "tris": "surface",
    "quads": "surface",
    "tets": "mesh",
    "hexs": "mesh",
    "wedges": "mesh",
    "pyramids": "mesh",
    "points": "point_cloud",
    "lines": "mesh",
    "mixed": "mesh",
    "mixed_2d": "surface",
}


@pytest.mark.parametrize("braid", sorted(BRAID))
def test_real_conduit_nodes_through_the_bridge(braid: str) -> None:
    conduit = _conduit()

    node = conduit.Node()
    nz = 0 if braid in ("tris", "quads", "mixed_2d") else 4
    conduit.blueprint.mesh.examples.braid(braid, 5, 4, nz, node)
    info = conduit.Node()
    assert conduit.blueprint.mesh.verify(node, info)
    node["state/time"] = 1.5
    s = _one(node, time=1.5, time_index=0)
    assert s.kind == BRAID[braid]
    tree = blueprint.to_tree(node)
    assert blueprint.tree_time(tree)[0] == 1.5
    braid_vals = np.asarray(node["fields/braid/values"]).reshape(-1)
    if s.kind == "grid":
        shape = (5, 4, nz)
        want = braid_vals.reshape(shape[::-1]).transpose(2, 1, 0)
        np.testing.assert_allclose(s.fields["braid"].data[..., 0], want, rtol=1e-6)
        assert s.fields["vel"].data.shape == (*shape, 3)
    else:
        assert len(s.points) == len(np.asarray(node["coordsets/coords/values/x"]))
        np.testing.assert_allclose(s.fields["braid"].data[:, 0], braid_vals, rtol=1e-6)
        np.testing.assert_allclose(s.points[:, 0], np.asarray(node["coordsets/coords/values/x"]))
        assert s.fields["vel"].component_names == ["u", "v", "w"][: s.fields["vel"].data.shape[1]]
    if braid == "hexs":
        assert s.cells.shape == (4 * 3 * 3, 8) and set(s.cell_types.tolist()) == {12}
        conn = np.asarray(node["topologies/mesh/elements/connectivity"]).reshape(-1, 8)
        np.testing.assert_array_equal(s.cells, conn)
        np.testing.assert_allclose(
            s.fields["radial"].data[:, 0], np.asarray(node["fields/radial/values"]), rtol=1e-6
        )
    if braid == "mixed":
        # braid "mixed" holds tets, hexes and polyhedra whose faces are mixed tris/quads
        assert set(s.cell_types.tolist()) == {10, 12, 42} and s.metadata["polyhedral"]
        sizes = (s.cells >= 0).sum(axis=1)
        assert set(sizes[s.cell_types == 12].tolist()) == {8}
        assert set(sizes[s.cell_types == 10].tolist()) == {4}
        assert set(sizes[s.cell_types == 42].tolist()) <= {4, 5, 6, 8}
    if braid == "structured":
        assert s.metadata["structured_shape"] == [5, 4, 4] and set(s.cell_types.tolist()) == {12}
    # the emitter's converter gives the same tree
    again = _one(fe.from_conduit(node))
    assert again.kind == s.kind and sorted(again.fields) == sorted(s.fields)


def test_real_conduit_polytopes_and_multi_domain() -> None:
    conduit = _conduit()

    tess = conduit.Node()
    conduit.blueprint.mesh.examples.polytess(3, 1, tess)
    s = _one(tess)
    assert s.kind == "surface" and s.static["face_parent"].data.max() + 1 == len(
        np.asarray(tess["topologies/topo/elements/sizes"])
    )
    tess3 = conduit.Node()
    conduit.blueprint.mesh.examples.polytess(3, 3, tess3)
    p = _one(tess3)
    assert p.kind == "mesh" and set(p.cell_types.tolist()) == {42} and p.metadata["polyhedral"]
    assert len(p.cells) == len(np.asarray(tess3["topologies/topo/elements/sizes"]))
    assert int(p.cells.max()) < len(p.points)
    spiral = conduit.Node()
    conduit.blueprint.mesh.examples.spiral(4, spiral)
    zones = mesh_to_samples(spiral, "spiral")
    assert len(zones) == 4 and all(z.kind == "grid" for z in zones)
    assert zones[2].metadata["blueprint"]["domain_id"] == 2
    cat = conduit.Node()
    cat["catalyst/state/timestep"] = 3
    cat["catalyst/state/time"] = 0.75
    cat["catalyst/channels/grid/type"] = "mesh"
    conduit.blueprint.mesh.examples.braid("tets", 3, 3, 3, cat["catalyst/channels/grid/data"])
    tree = blueprint.to_tree(cat)
    assert blueprint.tree_time(tree) == (0.75, 3)
    (c,) = _samples(cat, "c")
    assert c.zone == "mesh" and set(c.cell_types.tolist()) == {10}


CONDUIT_WRITER = r"""
import sys, time, json, pathlib
import numpy as np
import conduit, conduit.blueprint, conduit.relay.io
import conduit.relay.io.blueprint as rb
cfg = json.loads(sys.argv[1])
about = conduit.relay.io.about()   # keep the node alive while its children are read
if cfg["protocol"] == "hdf5" and about.fetch("protocols/hdf5").value() != "enabled":
    sys.exit(77)
sp = pathlib.Path(cfg["spool"])
if cfg["kind"] == "braid":
    for i in range(3):
        node = conduit.Node()
        conduit.blueprint.mesh.examples.braid("hexs", 4, 3, 3, node)
        node["state/cycle"] = 5 * i
        node["state/time"] = 0.1 * i
        node["fields/braid/values"] = np.asarray(node["fields/braid/values"]) + i
        rb.save_mesh(node, str(sp / "braid"), cfg["protocol"])
else:
    opts = conduit.Node()
    opts["file_style"] = "multi_file"
    time.sleep(0.2)
    for i in range(4):
        mesh = conduit.Node()
        conduit.blueprint.mesh.examples.spiral(3, mesh)
        for d in range(3):
            dom = mesh.child(d)
            dom["state/cycle"] = i
            dom["state/time"] = 0.25 * i
            dom["fields/dist/values"] = np.asarray(dom["fields/dist/values"]) + i
        rb.save_mesh(mesh, str(sp / "spiral"), cfg["protocol"], opts)
        time.sleep(0.03)
    (sp / "frontier-stream.done").touch()
"""


def conduit_writer(tmp_path: Path, wait: bool = True, **cfg: Any) -> subprocess.Popen[bytes]:
    """Real ``conduit.relay.io.blueprint.save_mesh`` in its own process (as in production; it
    also keeps Conduit's HDF5 library apart from the one bundled with h5py)."""
    script = tmp_path / "conduit_writer.py"
    script.write_text(CONDUIT_WRITER)
    cfg["spool"] = str(cfg["spool"])
    proc = subprocess.Popen([sys.executable, str(script), json.dumps(cfg)])
    if wait:
        code = proc.wait(timeout=60)
        if code == 77:
            pytest.skip("this Conduit build has no HDF5")
        assert code == 0
    return proc


@pytest.mark.parametrize(
    "protocol", ["hdf5", "json", "yaml", "conduit_json", "conduit_base64_json", "conduit_bin"]
)
def test_real_conduit_save_mesh_output_is_read_without_conduit(
    protocol: str, tmp_path: Path, connector: CatalystConnector
) -> None:
    conduit = _conduit()

    if protocol == "hdf5":
        pytest.importorskip("h5py")
    if protocol == "yaml":
        pytest.importorskip("yaml")
    sp = tmp_path / "relay"
    sp.mkdir()
    conduit_writer(tmp_path, spool=sp, protocol=protocol, kind="braid")
    report = check_connector(connector, sp, tmp_path, min_confidence=0.8)
    assert report.description.time.values == pytest.approx([0.0, 0.1, 0.2])
    steps: list[Any] = list(report.samples["braid"])
    assert len(steps) == 3 and steps[0].kind == "mesh"
    node = conduit.Node()
    conduit.blueprint.mesh.examples.braid("hexs", 4, 3, 3, node)
    direct = _one(node)
    np.testing.assert_array_equal(steps[0].cells, direct.cells)
    np.testing.assert_allclose(
        steps[2].fields["braid"].data, direct.fields["braid"].data + 2, rtol=1e-6
    )
    np.testing.assert_allclose(steps[2].fields["vel"].data, direct.fields["vel"].data, rtol=1e-6)
    np.testing.assert_allclose(steps[0].points, direct.points)


def test_real_conduit_multi_file_layout_followed_live(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    conduit = _conduit()

    pytest.importorskip("h5py")
    sp = tmp_path / "relay"
    sp.mkdir()
    ref = conduit.Node()
    conduit.blueprint.mesh.examples.spiral(3, ref)
    proc = conduit_writer(tmp_path, wait=False, spool=sp, protocol="hdf5", kind="spiral")
    try:
        steps = list(
            connector.read(
                sp, options={"follow": True, "timeout": 30, "poll": 0.01, "settle": 0.02}
            )
        )
    finally:
        code = proc.wait(timeout=60)
    if code == 77:
        pytest.skip("this Conduit build has no HDF5")
    assert code == 0
    by_zone: dict[str, list[Any]] = {}
    for s in steps:
        by_zone.setdefault(str(s.zone), []).append(s)
    assert sorted(by_zone) == ["domain_000000", "domain_000001", "domain_000002"]
    for d in range(3):
        zone = by_zone[f"domain_{d:06d}"]
        assert [s.time for s in zone] == [0.0, 0.25, 0.5, 0.75]
        base = _one(ref.child(d)).fields["dist"].data
        np.testing.assert_allclose(zone[3].fields["dist"].data, base + 3, rtol=1e-6)


# -- 8. ADIOS2 (optional; exercised for real in a scratch environment) ----------------------------

ADIOS_WRITER = r"""
import sys, time, json
import numpy as np
import adios2
cfg = json.loads(sys.argv[1])
ad = adios2.Adios()
io = ad.declare_io("w")
io.set_engine(cfg["engine"])
if cfg["engine"] == "SST":
    io.set_parameters({"RendezvousReaderCount": "1", "QueueLimit": "0", "OpenTimeoutSecs": "30"})
time.sleep(cfg.get("start_delay", 0.0))
n = cfg["n"]
xs = np.linspace(0.0, 1.0, n + 1)
yy, xx = np.meshgrid(xs, xs, indexing="ij")
pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
a = (i + (n + 1) * j).ravel()
quads = np.stack([a, a + 1, a + n + 2, a + n + 1], 1).astype(np.int64)
with adios2.Stream(io, cfg["path"], "w") as s:
    for step in range(cfg["n_steps"]):
        s.begin_step()
        if step == 0:   # static mesh: written once
            for k, ax in enumerate("xyz"):
                s.write("coordsets/coords/values/" + ax, np.ascontiguousarray(pts[:, k]))
            s.write("topologies/mesh/elements/connectivity", quads.reshape(-1))
            s.write_attribute("coordsets/coords/type", "explicit")
            s.write_attribute("topologies/mesh/type", "unstructured")
            s.write_attribute("topologies/mesh/coordset", "coords")
            s.write_attribute("topologies/mesh/elements/shape", "quad")
            s.write_attribute("fields/p/association", "vertex")
            s.write_attribute("fields/p/topology", "mesh")
            s.write_attribute("fields/p/units", "Pa")
            s.write_attribute("fields/U/association", "element")
            s.write_attribute("fields/U/topology", "mesh")
            s.write_attribute("frontier/solver", "tiny-adios")
        t = 0.1 * step
        s.write("fields/p/values", np.sin(pts[:, 0] + t))
        U = np.stack([np.full(len(quads), t), np.arange(len(quads), dtype=float)], axis=1)
        s.write("fields/U/values", U, shape=list(U.shape), start=[0, 0], count=list(U.shape))
        s.write("state/time", np.array(t))
        s.write("state/cycle", np.array(step * 2, dtype=np.int64))
        s.end_step()
        time.sleep(cfg.get("pause", 0.0))
"""


def run_adios_writer(tmp_path: Path, wait: bool = True, **cfg: Any) -> subprocess.Popen[bytes]:
    script = tmp_path / "adios_writer.py"
    script.write_text(ADIOS_WRITER)
    cfg.setdefault("n", 3)
    cfg.setdefault("n_steps", 4)
    cfg["path"] = str(cfg["path"])
    proc = subprocess.Popen([sys.executable, str(script), json.dumps(cfg)])
    if wait:
        assert proc.wait(timeout=60) == 0
    return proc


def _check_adios_steps(steps: list[Any], n: int, n_steps: int) -> None:
    assert [s.time_index for s in steps] == list(range(n_steps))
    assert [s.time for s in steps] == pytest.approx([0.1 * i for i in range(n_steps)])
    first = steps[0]
    assert first.kind == "surface" and first.points.shape == ((n + 1) ** 2, 3)
    assert [s.points is not None for s in steps] == [True] + [False] * (n_steps - 1)
    parent = first.static["face_parent"].data[:, 0]
    for i, s in enumerate(steps):
        np.testing.assert_allclose(
            s.fields["p"].data[:, 0], np.sin(first.points[:, 0] + 0.1 * i), atol=1e-6
        )
        np.testing.assert_allclose(s.fields["U"].data[:, 0], 0.1 * i, atol=1e-6)
        np.testing.assert_allclose(s.fields["U"].data[:, 1], parent)
    assert first.fields["p"].units == "Pa" and first.metadata["solver"] == "tiny-adios"


@pytest.mark.parametrize("engine", ["BP5", "BP4"])
def test_adios2_bp_file_blueprint_convention(
    engine: str, tmp_path: Path, connector: CatalystConnector
) -> None:
    pytest.importorskip("adios2")
    bp = tmp_path / "run.bp"
    run_adios_writer(tmp_path, path=bp, engine=engine)
    det = connector.detect(bp)
    assert det is not None and det.format == "adios2-bp" and det.needs == []
    report = check_connector(connector, bp, tmp_path, min_confidence=0.8)
    assert report.description.format == "adios2-bp" and report.description.solver == "tiny-adios"
    assert report.description.time.values == pytest.approx([0.0, 0.1, 0.2, 0.3])
    _check_adios_steps(report.samples["run"], 3, 4)
    (last,) = list(connector.read(bp, times="last"))
    assert last.time_index == 3 and last.points is not None  # mesh of step 0 carried along
    if frontier_data_available():
        validate_with_frontier_data(tmp_path / "dataset-catalyst-zarr")


def test_adios2_bp5_followed_while_another_process_writes(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    pytest.importorskip("adios2")
    bp = tmp_path / "live.bp"
    proc = run_adios_writer(
        tmp_path, wait=False, path=bp, engine="BP5", n_steps=6, pause=0.05, start_delay=0.3
    )
    try:
        t0 = time.monotonic()
        res = importers.run_import(
            "catalyst",
            str(bp),
            tmp_path / "ds",
            download_dir=tmp_path / "w",
            options={"timeout": 30, "poll": 0.02},
            backend="zarr",
        )
        elapsed = time.monotonic() - t0
    finally:
        assert proc.wait(timeout=60) == 0
    assert res["dataset"]["n_steps"] == {"live": 6} and elapsed > 0.3 + 4 * 0.05
    store = open_store(tmp_path / "ds" / "samples" / "live" / "surface.zarr")
    fields = {k: np.asarray(a) for g in store.group_keys() for k, a in store[g].arrays()}
    pts = np.asarray(store["points"])
    for i in range(6):
        np.testing.assert_allclose(fields["p"][i, :, 0], np.sin(pts[:, 0] + 0.1 * i), atol=1e-6)


def test_adios2_sst_stream_between_two_processes(
    tmp_path: Path, connector: CatalystConnector
) -> None:
    pytest.importorskip("adios2")
    name = tmp_path / "sim"
    proc = run_adios_writer(tmp_path, wait=False, path=name, engine="SST", n_steps=5, pause=0.03)
    try:
        res = importers.run_import(
            "catalyst",
            str(name) + ".sst",
            tmp_path / "ds",
            download_dir=tmp_path / "w",
            options={"timeout": 30, "poll": 0.02},
            backend="zarr",
        )
    finally:
        assert proc.wait(timeout=60) == 0
    assert res["dataset"]["n_steps"] == {"sim": 5}
    store = open_store(tmp_path / "ds" / "samples" / "sim" / "surface.zarr")
    fields = {k: np.asarray(a) for g in store.group_keys() for k, a in store[g].arrays()}
    pts = np.asarray(store["points"])
    for i in range(5):
        np.testing.assert_allclose(fields["p"][i, :, 0], np.sin(pts[:, 0] + 0.1 * i), atol=1e-6)
    meta = json.loads((tmp_path / "ds" / "samples" / "sim" / "meta.json").read_text())
    assert meta["solver"] == "tiny-adios"
    # nobody writes: opening times out with a clear error
    with pytest.raises(StreamTimeoutError, match="did not appear"):
        importers.run_import(
            "catalyst",
            str(tmp_path / "nobody.sst"),
            tmp_path / "x",
            download_dir=tmp_path / "w2",
            options={"timeout": 0.2, "poll": 0.01},
        )


def test_adios2_fides_predefined_models(tmp_path: Path, connector: CatalystConnector) -> None:
    adios2 = pytest.importorskip("adios2")
    nx, ny, nz = 4, 3, 2
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    uni = tmp_path / "uniform.bp"
    with adios2.Stream(str(uni), "w") as s:
        for step in range(3):
            s.begin_step()
            if step == 0:
                s.write_attribute("Fides_Data_Model", "uniform")
                s.write_attribute("Fides_Origin", np.array([1.0, 0.0, 0.0]))
                s.write_attribute("Fides_Spacing", np.array([0.5, 1.0, 2.0]))
                s.write_attribute("Fides_Dimension_Variable", "density")
                s.write_attribute("Fides_Time_Variable", "simtime")
            rho = (100.0 * i + 10 * j + k + step).astype(np.float64)  # shape (nz, ny, nx)
            s.write("density", rho, shape=list(rho.shape), start=[0, 0, 0], count=list(rho.shape))
            s.write("simtime", np.array(2.0 * step))
            s.end_step()
    steps = list(connector.read(uni))
    assert [s.time for s in steps] == [0.0, 2.0, 4.0] and steps[0].kind == "grid"
    np.testing.assert_allclose(steps[0].coords["x"], 1.0 + 0.5 * np.arange(nx))
    assert steps[1].fields["density"].data.shape == (nx, ny, nz, 1)
    assert steps[1].fields["density"].data[3, 2, 1, 0] == 300 + 20 + 1 + 1
    assert connector.describe(uni).time.values == [0.0, 2.0, 4.0]
    # the same data stored (nx, ny, nz): option fides_order=xyz
    xyz = tmp_path / "uniform_xyz.bp"
    with adios2.Stream(str(xyz), "w") as s:
        s.begin_step()
        s.write_attribute("Fides_Data_Model", "uniform")
        s.write_attribute("Fides_Dimension_Variable", "density")
        rho = np.ascontiguousarray((100.0 * i + 10 * j + k).transpose(2, 1, 0))
        s.write("density", rho, shape=list(rho.shape), start=[0, 0, 0], count=list(rho.shape))
        s.end_step()
    (g,) = list(connector.read(xyz, options={"fides_order": "xyz"}))
    assert g.fields["density"].data.shape == (nx, ny, nz, 1)
    assert (
        g.fields["density"].data[3, 2, 1, 0] == 321 and g.fields["density"].data[1, 0, 1, 0] == 101
    )
    pts, tets = tet_grid(1, 1, 1)
    uns = tmp_path / "tets.bp"
    with adios2.Stream(str(uns), "w") as s:
        s.begin_step()
        s.write_attribute("Fides_Data_Model", "unstructured_single")
        s.write_attribute("Fides_Cell_Type", "tetrahedron")
        s.write("points", pts, shape=list(pts.shape), start=[0, 0], count=list(pts.shape))
        s.write("connectivity", tets.reshape(-1))
        s.write("pressure", pts[:, 0].copy())
        s.write("cell_id", np.arange(len(tets), dtype=np.float64))
        s.end_step()
    (t,) = list(connector.read(uns))
    assert t.kind == "mesh" and t.time is None
    np.testing.assert_array_equal(t.cells, tets)
    assert t.fields["pressure"].location == "point" and t.fields["cell_id"].location == "cell"
    with adios2.Stream(str(tmp_path / "xgc.bp"), "w") as s:
        s.begin_step()
        s.write_attribute("Fides_Data_Model", "xgc")
        s.write("x", np.zeros(3))
        s.end_step()
    with pytest.raises(UnsupportedSourceError, match="xgc"):
        list(connector.read(tmp_path / "xgc.bp"))


def test_adios2_missing_package_is_explained(
    tmp_path: Path, connector: CatalystConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    bp = tmp_path / "fake.bp"
    bp.mkdir()
    (bp / "md.idx").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "adios2", None)
    det = connector.detect(bp)
    assert det is not None and det.needs == ["pip install adios2 (Apache-2.0)"]
    from frontier_connectors.errors import MissingDependencyError

    with pytest.raises(MissingDependencyError, match="adios2"):
        connector.describe(bp)
