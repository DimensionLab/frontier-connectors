# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false
# ruff: noqa: E501, UP006, UP007, UP031, UP035, UP037, UP045, B905
# (old-style typing and % formatting on purpose: this file runs under the solver's Python, 3.8+)
"""Catalyst 2 pipeline script (ParaView implementation) that feeds DimensionLab Frontier.

NOT RUN AGAINST PARAVIEW by the Frontier test-suite (no ParaView in CI): the ParaView calls
follow the documented ``paraview.catalyst`` script layout (``PVTrivialProducer`` named after the
channel, ``CreateExtractor``, ``catalyst.Options``, ``catalyst_execute(info)``,
``catalyst.get_args()``; https://docs.paraview.org/en/latest/Catalyst/blueprints.html and the
generated scripts of ParaView 5.11+ / 6.x, read 2026-10-10). The VTK -> Blueprint conversion
below (:func:`vtk_to_blueprint`) *is* tested, with plain VTK.

How the solver uses it (adaptor side, see ``adaptor_example.txt``)::

    node["catalyst/scripts/frontier/filename"] = "/path/to/frontier_catalyst_pipeline.py"
    node["catalyst/scripts/frontier/args"]     = ["--spool", "run/spool",
                                                  "--channel", "grid", "--every", "10"]
    catalyst_initialize(node)

    export CATALYST_IMPLEMENTATION_NAME=paraview
    export CATALYST_IMPLEMENTATION_PATHS=<paraview install>/lib/catalyst

Arguments (also read from the environment as ``FRONTIER_<NAME>``):

``--channel NAME``    Catalyst channel to export (default ``grid``)
``--spool DIR``       spool directory of the ``frontier-spool`` protocol (default ``frontier-spool``)
``--every N``         export every N-th time step (default 1)
``--mode MODE``       ``spool`` (default): convert the channel's VTK data to a Blueprint tree
                      and write it with ``frontier_emit`` (read it live with the ``catalyst``
                      importer). Serial runs only: with several MPI ranks every rank holds a
                      piece and this script refuses rather than write one of them.
                      ``vtk``: add a ParaView extractor that writes ``<channel>_<step>.vtpd``
                      (parallel safe; read the files with Frontier's ``vtk`` connector).
                      ``both``: do both.
``--solver NAME``     recorded as the solver name of the stream
``--static-mesh``     store the mesh with the first step only

``frontier_emit.py`` must be importable: keep it next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: VTK linear cell type id -> Blueprint shape name
VTK_SHAPES = {
    1: "point",
    3: "line",
    5: "tri",
    9: "quad",
    10: "tet",
    12: "hex",
    13: "wedge",
    14: "pyramid",
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
_SKIP_ARRAYS = ("vtkGhostType", "vtkOriginalPointIds", "vtkOriginalCellIds", "vtkValidPointMask")


# -- VTK -> Blueprint (plain VTK, no ParaView needed) -------------------------------------------


def _np(arr):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return np.array(vtk_to_numpy(arr), copy=True)


def _fields(ds, topology):
    out = {}
    for assoc, attrs in (("vertex", ds.GetPointData()), ("element", ds.GetCellData())):
        for i in range(attrs.GetNumberOfArrays()):
            arr = attrs.GetArray(i)  # None for string arrays
            if arr is None or not arr.GetName() or arr.GetName() in _SKIP_ARRAYS:
                continue
            name = arr.GetName()
            if name in out:
                name = name + "__cell"
            values = _np(arr)
            if values.ndim == 2 and arr.HasAComponentName():
                values = {
                    (arr.GetComponentName(c) or str(c)): values[:, c].copy()
                    for c in range(values.shape[1])
                }
            out[name] = {"association": assoc, "topology": topology, "values": values}
    return out


def _explicit(ds):
    pts = _np(ds.GetPoints().GetData()).astype(np.float64)
    return {"type": "explicit", "values": {a: pts[:, i].copy() for i, a in enumerate("xyz")}}


def _cell_array(ca):
    return _np(ca.GetConnectivityArray()).astype(np.int64), _np(ca.GetOffsetsArray()).astype(
        np.int64
    )


def _elements(types, conn, offsets):
    """Blueprint ``elements`` from VTK cell types + CSR connectivity."""
    sizes = np.diff(offsets)
    kinds = sorted(set(int(t) for t in np.unique(types)))
    unknown = [t for t in kinds if t not in VTK_SHAPES and t != 7]
    if unknown:
        raise ValueError(
            "VTK cell type(s) %s cannot be exported (linear cells and polygons only): "
            "use --mode vtk and Frontier's vtk connector instead" % unknown
        )
    if len(kinds) == 1 and kinds[0] in VTK_SHAPES:
        return {"shape": VTK_SHAPES[kinds[0]], "connectivity": conn}
    if kinds == [7]:
        return {"shape": "polygonal", "sizes": sizes, "offsets": offsets[:-1], "connectivity": conn}
    shape_map = {("polygonal" if t == 7 else VTK_SHAPES[t]): t for t in kinds}
    return {
        "shape": "mixed",
        "shape_map": shape_map,
        "shapes": np.asarray(types, dtype=np.int32),
        "sizes": sizes,
        "offsets": offsets[:-1],
        "connectivity": conn,
    }


def vtk_to_blueprint(ds):
    """One ``vtkDataSet`` -> Blueprint domain (nested dict with numpy leaves)."""
    cls = ds.GetClassName()
    if cls in ("vtkImageData", "vtkUniformGrid", "vtkStructuredPoints"):
        dims = [0, 0, 0]
        ds.GetDimensions(dims)
        origin, spacing = ds.GetOrigin(), ds.GetSpacing()
        coordset = {
            "type": "uniform",
            "dims": dict(zip("ijk", [int(d) for d in dims])),
            "origin": dict(zip("xyz", [float(v) for v in origin])),
            "spacing": dict(zip(("dx", "dy", "dz"), [float(v) for v in spacing])),
        }
        topo = {"type": "uniform", "coordset": "coords"}
    elif cls == "vtkRectilinearGrid":
        coordset = {
            "type": "rectilinear",
            "values": {
                "x": _np(ds.GetXCoordinates()).astype(np.float64),
                "y": _np(ds.GetYCoordinates()).astype(np.float64),
                "z": _np(ds.GetZCoordinates()).astype(np.float64),
            },
        }
        topo = {"type": "rectilinear", "coordset": "coords"}
    elif cls == "vtkStructuredGrid":
        dims = [0, 0, 0]
        ds.GetDimensions(dims)
        coordset = _explicit(ds)
        topo = {
            "type": "structured",
            "coordset": "coords",
            "elements": {"dims": dict(zip("ijk", [int(d) - 1 for d in dims]))},
        }
    elif cls == "vtkUnstructuredGrid":
        coordset = _explicit(ds)
        conn, offsets = _cell_array(ds.GetCells())
        n = int(ds.GetNumberOfCells())
        types = np.fromiter((ds.GetCellType(i) for i in range(n)), dtype=np.int32, count=n)
        topo = {
            "type": "unstructured",
            "coordset": "coords",
            "elements": _elements(types, conn, offsets),
        }
    elif cls == "vtkPolyData":
        coordset = _explicit(ds)
        if ds.GetNumberOfStrips() > 0:
            raise ValueError("triangle strips cannot be exported: apply a Triangulate filter")
        conns, offs, types = [], [np.zeros(1, dtype=np.int64)], []
        base = 0
        for ca, kind in ((ds.GetVerts(), 1), (ds.GetLines(), 3), (ds.GetPolys(), 7)):
            if ca is None or ca.GetNumberOfCells() == 0:
                continue
            c, o = _cell_array(ca)
            sizes = np.diff(o)
            if kind == 1 and (sizes != 1).any():
                raise ValueError("poly-vertex cells cannot be exported")
            if kind == 3 and (sizes != 2).any():
                raise ValueError("polylines cannot be exported: split them into segments")
            t = np.full(len(sizes), kind, dtype=np.int32)
            if kind == 7:
                t = np.where(sizes == 3, 5, np.where(sizes == 4, 9, 7)).astype(np.int32)
            conns.append(c)
            offs.append(o[1:] + base)
            types.append(t)
            base += len(c)
        if not conns:
            topo = {"type": "points", "coordset": "coords"}
        else:
            topo = {
                "type": "unstructured",
                "coordset": "coords",
                "elements": _elements(
                    np.concatenate(types), np.concatenate(conns), np.concatenate(offs)
                ),
            }
    else:
        raise ValueError("cannot export a %s: use --mode vtk" % cls)
    return {
        "coordsets": {"coords": coordset},
        "topologies": {"mesh": topo},
        "fields": _fields(ds, "mesh"),
    }


def _leaves(obj, prefix=""):
    """``(name, vtkDataSet)`` for every non-empty leaf of a composite data object."""
    if obj is None:
        return
    if obj.IsA("vtkPartitionedDataSetCollection"):
        for i in range(obj.GetNumberOfPartitionedDataSets()):
            name = None
            if obj.HasMetaData(i):
                name = obj.GetMetaData(i).Get(obj.NAME())
            yield from _leaves(obj.GetPartitionedDataSet(i), prefix + (name or "block%d" % i))
    elif obj.IsA("vtkPartitionedDataSet"):
        parts = [obj.GetPartition(i) for i in range(obj.GetNumberOfPartitions())]
        parts = [p for p in parts if p is not None and p.GetNumberOfPoints() > 0]
        for i, p in enumerate(parts):
            yield (prefix or "mesh") + ("" if len(parts) == 1 else "_part%d" % i), p
    elif obj.IsA("vtkMultiBlockDataSet"):
        for i in range(obj.GetNumberOfBlocks()):
            name = None
            if obj.HasMetaData(i):
                name = obj.GetMetaData(i).Get(obj.NAME())
            yield from _leaves(obj.GetBlock(i), prefix + (name or "block%d" % i) + "/")
    elif obj.IsA("vtkDataSet"):
        if obj.GetNumberOfPoints() > 0:
            yield prefix.rstrip("/") or "mesh", obj


def data_object_to_blueprint(obj):
    """Any VTK data object -> one Blueprint domain or ``{name: domain}`` for composites."""
    leaves = list(_leaves(obj))
    if not leaves:
        raise ValueError("the channel holds no data")
    if len(leaves) == 1 and not obj.IsA("vtkCompositeDataSet"):
        return vtk_to_blueprint(leaves[0][1])
    return {name.replace("/", "_"): vtk_to_blueprint(ds) for name, ds in leaves}


# -- ParaView Catalyst part -------------------------------------------------------------------


def _parse_args(argv):
    import argparse

    env = os.environ.get
    p = argparse.ArgumentParser(prog="frontier_catalyst_pipeline")
    p.add_argument("--channel", default=env("FRONTIER_CHANNEL", "grid"))
    p.add_argument("--spool", default=env("FRONTIER_SPOOL", "frontier-spool"))
    p.add_argument("--every", type=int, default=int(env("FRONTIER_EVERY", "1")))
    p.add_argument(
        "--mode", choices=("spool", "vtk", "both"), default=env("FRONTIER_MODE", "spool")
    )
    p.add_argument("--solver", default=env("FRONTIER_SOLVER", "catalyst"))
    p.add_argument("--static-mesh", action="store_true", default=bool(env("FRONTIER_STATIC_MESH")))
    args, _unknown = p.parse_known_args(list(argv))
    return args


try:  # everything below needs ParaView; importing this file for vtk_to_blueprint must not
    from paraview import catalyst
    from paraview import simple as pvs

    _HAVE_PARAVIEW = True
except ImportError:
    _HAVE_PARAVIEW = False

if _HAVE_PARAVIEW:
    ARGS = _parse_args(catalyst.get_args())
    _emitter = None

    # the producer is matched to the Catalyst channel by its registration name
    producer = pvs.PVTrivialProducer(registrationName=ARGS.channel)

    options = catalyst.Options()
    options.GlobalTrigger = "Time Step"
    options.EnableCatalystLive = 0
    options.ExtractsOutputDirectory = os.path.join(ARGS.spool, "vtk")

    if ARGS.mode in ("vtk", "both"):
        extractor = pvs.CreateExtractor("VTPD", producer, registrationName="frontier_vtpd")
        extractor.Trigger = "Time Step"
        extractor.Trigger.Frequency = max(ARGS.every, 1)
        extractor.Writer.FileName = ARGS.channel + "_{timestep:06d}.vtpd"

    def _ranks():
        try:
            from paraview import servermanager

            return int(
                servermanager.vtkProcessModule.GetProcessModule().GetNumberOfLocalPartitions()
            )
        except Exception:
            return 1

    def catalyst_initialize():
        global _emitter
        if ARGS.mode in ("spool", "both"):
            if _ranks() > 1:
                raise RuntimeError(
                    "frontier_catalyst_pipeline: --mode spool supports serial runs only; "
                    "use --mode vtk with %d MPI ranks" % _ranks()
                )
            import frontier_emit

            _emitter = frontier_emit.Emitter(
                ARGS.spool, solver=ARGS.solver, static_mesh=ARGS.static_mesh, overwrite=True
            )

    def catalyst_execute(info):
        if _emitter is None or info.timestep % max(ARGS.every, 1):
            return
        producer.UpdatePipeline(info.time)
        data = producer.GetClientSideObject().GetOutputDataObject(0)
        _emitter.write_step(int(info.timestep), float(info.time), data_object_to_blueprint(data))

    def catalyst_finalize():
        if _emitter is not None:
            _emitter.close()

    if __name__ == "__main__":
        # pvbatch frontier_catalyst_pipeline.py: replay extractors without a simulation
        pvs.SaveExtractsUsingCatalystOptions(options)
