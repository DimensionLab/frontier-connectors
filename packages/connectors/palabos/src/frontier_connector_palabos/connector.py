# SPDX-License-Identifier: Apache-2.0
"""Palabos connector."""

from __future__ import annotations

import re
from pathlib import Path

from frontier_connectors.model import (
    SourceDescription,
)
from frontier_connectors.routing import Target, VendorRouteConnector, _listdir


class PalabosConnector(VendorRouteConnector):
    id = "palabos"
    name = "Palabos (writes VTK .vti/.vtk directly)"
    formats = ["palabos-vtk", "vtk-xml"]
    vendor = "University of Geneva / FlowKit (open source, AGPL solver; open output format)"
    solver = "Palabos"
    routes = ["vtk"]
    extensions = []
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["grid"]
    export_help = (
        "No export step: Palabos writes VTK image data (VtkImageOutput2D/3D -> tmp/vtk*.vti, "
        "VtkStructuredImageOutput -> .vts). Point at the output directory (the one holding "
        "plbLog.dat) or at a single .vti file."
    )
    status_note = (
        "Reads Palabos output directories (.vti series and .vts states) with Re, resolution, "
        "relaxation frequency, dx and dt from plbLog.dat. An output directory is recognised by "
        "plbLog.dat next to its VTK files; file numbers are iterations, converted to time with the "
        "log's dt, and values are in the units the application wrote. The XML parameter file is "
        "not read. 3-D volume series with a log, parallel .pvti and sparse multi-block output are "
        "not validated."
    )

    def find_export(self, path: Path) -> Target | None:
        if path.is_dir() and (path / "plbLog.dat").is_file():
            files = [f for f in _listdir(path) if f.suffix.lower() in VTK_SUFFIXES]
            if len(files) == 1:  # a run that wrote one state: not a numbered family
                return "vtk", files[0], "Palabos output directory (plbLog.dat + one VTK file)"
            if files:
                return "vtk", path, "Palabos output directory (plbLog.dat + VTK files)"
        return None

    def enrich(self, desc: SourceDescription, target: Target) -> None:
        desc.physics.setdefault("method", "lattice Boltzmann")
        if not desc.time.steady:
            desc.time.quantity = "iteration"
        routed = target[1]
        title, values = read_plb_log((routed if routed.is_dir() else routed.parent) / "plbLog.dat")
        if title:
            desc.metadata["title"] = title
        for key, name in PLB_LOG_KEYS.items():
            if key in values:
                desc.parameters[name] = values[key]
        extra = {k: v for k, v in values.items() if k not in PLB_LOG_KEYS}
        if extra:
            desc.metadata["plb_log"] = extra
        dt = values.get("dt")
        if dt and not desc.time.steady:
            # the iteration axis stays (it is what the file names carry); the log gives the
            # factor to the time of the application's unit system
            desc.metadata["time_per_iteration"] = dt
            desc.warnings = [w for w in desc.warnings if "file numbers" not in w]
            desc.warnings.append(
                f"time axis is the iteration number; plbLog.dat gives dt={dt:g} per iteration "
                "in the dimensionless units of the application"
            )


#: file types of the Palabos VTK writers: VtkImageOutput2D/3D (.vti), VtkStructuredImageOutput
#: (.vts), legacy output (.vtk)
VTK_SUFFIXES = (".vti", ".vts", ".vtk")
#: symbol in plbLog.dat (``writeLogFile`` of Palabos' IncomprFlowParam) -> parameter name
PLB_LOG_KEYS = {
    "Re": "Re",
    "u": "lattice_velocity",
    "N": "resolution",
    "omega": "relaxation_frequency",
    "dx": "dx",
    "dt": "dt",
    "lx": "lx",
    "ly": "ly",
    "lz": "lz",
}
_LOG_LINE = re.compile(r"^.*?:\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)\s*$")


def read_plb_log(path: Path) -> tuple[str, dict[str, float]]:
    """``(title, {symbol: value})`` of a ``plbLog.dat``.

    Palabos applications write it with ``writeLogFile(parameters, title)``: the title, then
    lines such as ``Reynolds number:           Re=300`` and ``Time step deltaT:   dt=0.0001``.
    Applications may also write free text through ``plbLogFile``, which is ignored."""
    try:
        with open(path, "rb") as f:
            lines = f.read(65536).decode("utf-8", "replace").splitlines()
    except OSError:
        return "", {}
    title = ""
    values: dict[str, float] = {}
    for line in lines:
        m = _LOG_LINE.match(line)
        if m is None:
            if not title and not values and line.strip():
                title = line.strip()[:200]
            continue
        try:
            values.setdefault(m.group(1), float(m.group(2)))
        except ValueError:
            continue
    return title, values
