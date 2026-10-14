# SPDX-License-Identifier: Apache-2.0
"""VTK family: XML (``.vtu .vtp .vts .vtr .vti .vtm .vtmb .vtpc .vthb``), parallel
(``.pvtu .pvtp .pvts .pvtr .pvti``), legacy ``.vtk``, VTKHDF (``.vtkhdf``/``.hdf``) and time
series (ParaView ``.pvd`` collections and ``*.series`` JSON files).

Each time step is read from its own file(s), so only one step is in memory at a time.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET  # noqa: N817
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frontier_connectors import _vtk, meshutil
from frontier_connectors.errors import ConnectorError, UnsupportedSourceError
from frontier_connectors.model import (
    BaseConnector,
    Detection,
    RawSample,
    SourceDescription,
    TimeInfo,
    TimeSelection,
    match_extension,
)
from frontier_connectors.paths import case_root_of, h5_check_file, resolve_in_case
from frontier_connectors.select import select_time_indices

XML_EXTENSIONS = [
    ".vtu", ".vtp", ".vts", ".vtr", ".vti", ".vtm", ".vtmb", ".vtpc", ".vtpd", ".vthb",
    ".pvtu", ".pvtp", ".pvts", ".pvtr", ".pvti",
]  # fmt: skip
SERIES_EXTENSIONS = [".pvd", ".series"]
OTHER_EXTENSIONS = [".vtk", ".vtkhdf"]


#: XML file types that hold their data themselves; every other type names further files
_LEAF_TYPES = frozenset({
    "UnstructuredGrid", "PolyData", "StructuredGrid", "RectilinearGrid", "ImageData",
    "HyperTreeGrid", "Table",
})  # fmt: skip
_VTKFILE_TYPE = re.compile(rb"<VTKFile\b[^>]*?\btype\s*=\s*[\"']([^\"']+)[\"']", re.DOTALL)
#: attributes that name another file in parallel (`Piece Source`) and composite
#: (`DataSet file`, `Partition file`, `Block file`) VTK XML files
_REFERENCE_ATTRIBUTES = ("Source", "file")
#: composite and parallel files are indexes: small. A larger one is refused, not parsed.
MAX_INDEX_FILE_BYTES = 256 << 20
MAX_REFERENCE_DEPTH = 8


def _xml_type(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            m = _VTKFILE_TYPE.search(f.read(1 << 16))
    except OSError:
        return None
    return m.group(1).decode("ascii", "replace") if m else None


def confine_vtk_references(
    path: Path, case_root: Path, *, _depth: int = 0, _seen: set[Path] | None = None
) -> None:
    """Check every file a parallel or composite VTK XML file names (and what those name)
    against the case root. VTK's readers open the pieces themselves, so this runs first."""
    seen = _seen if _seen is not None else set()
    if path in seen:
        return
    seen.add(path)
    kind = _xml_type(path)
    if kind is None or kind in _LEAF_TYPES:
        return  # not VTK XML (legacy, VTKHDF), or a file that holds its own data
    if path.stat().st_size > MAX_INDEX_FILE_BYTES:
        raise ConnectorError(
            f"{path} is a {kind} index file larger than {MAX_INDEX_FILE_BYTES} bytes",
            hint="index files only list pieces; this one is not a plausible VTK collection",
        )
    if _depth > MAX_REFERENCE_DEPTH:
        raise ConnectorError(
            f"{path}: VTK files reference each other more than {MAX_REFERENCE_DEPTH} deep",
            hint="flatten the collection",
        )
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        raise ConnectorError(f"{path} is not valid VTK XML: {e}") from e
    for el in root.iter():
        for attr in _REFERENCE_ATTRIBUTES:
            ref = el.get(attr)
            if not ref:
                continue
            target = resolve_in_case(case_root, ref, what=f"VTK <{el.tag} {attr}=>", referrer=path)
            if target.is_file():
                confine_vtk_references(target, case_root, _depth=_depth + 1, _seen=seen)


def read_vtk_file(path: Path, case_root: Path | None = None) -> Any:
    """Read one VTK file (any XML flavour, legacy or VTKHDF) and return the VTK data object.

    ``case_root`` is the directory the files it references must stay in (default: the
    directory of ``path``); see ``frontier_connectors.paths``."""
    path = Path(path)
    if not path.is_file():
        raise UnsupportedSourceError(f"{path} does not exist")
    confine_vtk_references(path, case_root if case_root is not None else case_root_of(path))
    ext = path.suffix.lower()
    if ext == ".vtk":
        from vtkmodules.vtkIOLegacy import vtkGenericDataObjectReader

        reader = vtkGenericDataObjectReader()
        reader.SetFileName(str(path))
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.ReadAllFieldsOn()
        reader.ReadAllNormalsOn()
    elif ext in (".vtkhdf", ".hdf"):
        from vtkmodules.vtkIOHDF import vtkHDFReader

        h5_check_file(path, what="VTKHDF link")
        reader = vtkHDFReader()
        reader.SetFileName(str(path))
    else:
        from vtkmodules.vtkIOXML import vtkXMLGenericDataObjectReader

        reader = vtkXMLGenericDataObjectReader()
        reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutputDataObject(0)
    if out is None or (hasattr(out, "GetNumberOfPoints") and out.GetNumberOfPoints() == 0):
        raise ConnectorError(
            f"VTK could not read any points from {path}",
            hint="the file may be truncated, or reference missing piece files next to it",
        )
    return out


def _time_value(obj: Any) -> float | None:
    """ParaView stores the step time as a ``TimeValue`` field-data array."""
    fd = obj.GetFieldData() if hasattr(obj, "GetFieldData") else None
    arr = fd.GetArray("TimeValue") if fd is not None else None
    if arr is not None and arr.GetNumberOfTuples() >= 1:
        return float(arr.GetTuple1(0))
    return None


class _Series:
    """Resolved time series: ``times[i]`` -> ``[(part_name | None, file), ...]``."""

    def __init__(self, times: list[float], steps: list[list[tuple[str | None, Path]]]) -> None:
        self.times = times
        self.steps = steps
        #: case root the pieces (and what they reference) are confined to
        self.root: Path | None = None

    @property
    def transient(self) -> bool:
        return len(self.times) > 1


def parse_pvd(path: Path) -> _Series:
    """Parse a ParaView ``.pvd`` collection (``<DataSet timestep= part= file=/>``)."""
    root_dir = case_root_of(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ConnectorError(f"{path} is not valid PVD XML: {e}") from e
    by_time: dict[float, list[tuple[str | None, Path]]] = {}
    datasets = root.findall(".//DataSet")
    if not datasets:
        raise ConnectorError(
            f"{path} lists no <DataSet> entries", hint="the collection file is empty"
        )
    parts = {d.get("part", "0") for d in datasets}
    for d in datasets:
        file = d.get("file")
        if not file:
            continue
        t = float(d.get("timestep", "0") or 0.0)
        part = d.get("name") or (f"part{d.get('part', '0')}" if len(parts) > 1 else None)
        piece = resolve_in_case(root_dir, file, what="PVD DataSet file", referrer=path)
        by_time.setdefault(t, []).append((part, piece))
    times = sorted(by_time)
    return _Series(times, [by_time[t] for t in times])


def parse_series(path: Path) -> _Series:
    """Parse a ``*.vtu.series`` JSON file (``{"files": [{"name": ..., "time": ...}]}``)."""
    root_dir = case_root_of(path)
    try:
        doc = json.loads(path.read_text())
        files = sorted(doc["files"], key=lambda f: float(f["time"]))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ConnectorError(
            f"{path} is not a VTK file-series JSON: {e}",
            hint='expected {"file-series-version": "1.0", "files": [{"name": ..., "time": ...}]}',
        ) from e
    return _Series(
        [float(f["time"]) for f in files],
        [
            [
                (
                    None,
                    resolve_in_case(
                        root_dir, str(f["name"]), what="file-series entry", referrer=path
                    ),
                )
            ]
            for f in files
        ],  # fmt: skip
    )


_NUMBERED = re.compile(r"^(.*?)[._-]?(\d+)$")


def directory_series(path: Path) -> _Series | None:
    """A directory holding one numbered family (``flow_0000.vtu``, ``flow_0001.vtu``, ...)."""
    files = [
        f for f in sorted(path.iterdir())
        if f.is_file() and match_extension(f, XML_EXTENSIONS + [".vtk"])
    ]  # fmt: skip
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for f in files:
        m = _NUMBERED.match(f.stem)
        if m:
            groups.setdefault((m.group(1), f.suffix.lower()), []).append((int(m.group(2)), f))
    if len(groups) != 1:
        return None
    items = sorted(next(iter(groups.values())))
    if len(items) < 2:
        return None
    return _Series([float(i) for i, _ in items], [[(None, f)] for _, f in items])


class VtkConnector(BaseConnector):
    id = "vtk"
    name = "VTK (XML, legacy, VTKHDF, PVD/series time series)"
    formats = ["vtk-xml", "vtk-legacy", "vtkhdf", "pvd"]
    vendor = "Kitware (open format)"
    license = "Apache-2.0 (reader: VTK, BSD-3)"
    extensions = XML_EXTENSIONS + SERIES_EXTENSIONS + OTHER_EXTENSIONS
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["surface", "mesh", "grid", "point_cloud"]
    export_help = (
        "ParaView: File > Save Data (.vtu/.vtm, or .pvd with 'Write time series'). "
        "OpenFOAM: foamToVTK. SU2/Palabos/Flow360/Luminary/SimScale write VTU directly."
    )

    def detect(self, path: Path) -> Detection | None:
        path = Path(path)
        if path.is_dir():
            if (path / "system" / "controlDict").exists():
                return None
            series = directory_series(path)
            if series is None:
                return None
            return Detection(
                self.id,
                path,
                0.55,
                "vtk-series-dir",
                [f"directory with {len(series.times)} numbered VTK files"],
            )
        ext = match_extension(path, self.extensions)
        if ext is None:
            return None
        head = path.read_bytes()[:256] if path.is_file() else b""
        notes = [f"extension {ext}"]
        conf = 0.6
        if ext == ".vtk" and head.startswith(b"# vtk DataFile"):
            conf, notes = 0.98, notes + ["legacy header '# vtk DataFile'"]
        elif ext == ".series":
            if b"file-series-version" not in head and b'"files"' not in head:
                return None
            conf, notes = 0.95, notes + ["VTK file-series JSON"]
        elif ext == ".pvd":
            conf = 0.95 if b"Collection" in head else 0.5
        elif ext == ".vtkhdf":
            conf = 0.95 if head.startswith(b"\x89HDF") else 0.3
        elif b"<VTKFile" in head:
            conf, notes = 0.98, notes + ["<VTKFile> root element"]
        return Detection(self.id, path, conf, "vtk" + ext, notes)

    # -- plan ---------------------------------------------------------------------------
    def _series(self, path: Path) -> _Series:
        series = self._plan(Path(path))
        #: case root: the directory given, or the directory of the file given
        series.root = case_root_of(Path(path))
        return series

    def _plan(self, path: Path) -> _Series:
        if path.is_dir():
            series = directory_series(path)
            if series is None:
                raise UnsupportedSourceError(
                    f"{path} is not a single numbered VTK file family",
                    hint="point at one file, a .pvd, or a *.series file",
                )
            return series
        ext = path.suffix.lower()
        if ext == ".pvd":
            return parse_pvd(path)
        if ext == ".series":
            return parse_series(path)
        return _Series([], [[(None, path)]])

    def _load_step(self, series: _Series, index: int) -> list[tuple[str, Any]]:
        """Leaves ``(zone, dataset)`` of one time step."""
        leaves: list[tuple[str, Any]] = []
        parts = series.steps[index]
        for part, file in parts:
            if not file.is_file():
                raise ConnectorError(
                    f"time step {index} references missing file {file}",
                    hint="copy the piece files next to the collection file",
                )
            obj = read_vtk_file(file, series.root)
            prefix = part or ""
            leaves.extend(_vtk.iter_leaves(obj, prefix))
        return leaves

    def describe(
        self, path: Path, *, options: Mapping[str, Any] | None = None
    ) -> SourceDescription:
        path = Path(path)
        series = self._series(path)
        desc = SourceDescription(
            connector_id=self.id,
            path=path,
            format="vtk" + (path.suffix.lower() or "-series-dir"),
            solver="unknown",
            time=TimeInfo(values=list(series.times)),
        )
        first = series.steps[0]
        for part, file in first:
            obj = read_vtk_file(file, series.root)
            sub = SourceDescription(connector_id=self.id, path=path)
            _vtk.describe_object(obj, sub)
            for z in sub.zones:
                if part:
                    z.name = f"{part}/{z.name}" if z.name != "main" else part
                desc.zones.append(z)
            for f in sub.fields:
                if not any(g.name == f.name and g.location == f.location for g in desc.fields):
                    f.zones = None
                    desc.fields.append(f)
            tv = _time_value(obj)
            if tv is not None and not series.transient:
                desc.metadata["time_value"] = tv
        if path.is_dir():
            desc.warnings.append(
                "times are the file numbers; write a .pvd or .series file to carry physical time"
            )
            desc.time.quantity = "iteration"
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
        series = self._series(path)
        base = meshutil.default_sample_id(path)
        selected: list[str] | None = None
        metadata = {"solver": "unknown", "source": str(path), "format": "vtk"}
        for ti in select_time_indices(series.times, times):
            leaves = self._load_step(series, ti)
            names = [n for n, _ in leaves]
            if selected is None:
                selected = _vtk.select_zones(names, zones)
            for name, ds in leaves:
                if name not in selected:
                    continue
                yield _vtk.dataset_to_sample(
                    ds,
                    meshutil.zone_sample_id(base, name, len(selected)),
                    zone=name,
                    fields=fields,
                    time=float(series.times[ti]) if series.transient else None,
                    time_index=ti if series.transient else None,
                    metadata=dict(metadata),
                )


def write_test_vtu(path: Path, points: np.ndarray, cells: np.ndarray, cell_type: int) -> None:
    """Tiny helper for docs/tests: write an unstructured grid with VTK (not used by readers)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    grid = vtkUnstructuredGrid()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.asarray(points, dtype=np.float64), deep=True))
    grid.SetPoints(pts)
    for row in np.asarray(cells):
        grid.InsertNextCell(cell_type, len(row), [int(i) for i in row])
    w = vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(path))
    w.SetInputData(grid)
    w.Write()
