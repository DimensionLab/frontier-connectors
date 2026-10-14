# SPDX-License-Identifier: Apache-2.0
"""code_saturne connector."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path
from typing import Any

from frontier_connectors.model import (
    SourceDescription,
    match_extension,
)
from frontier_connectors.routing import Target, VendorRouteConnector, _case_with_marker, _listdir


class CodeSaturneConnector(VendorRouteConnector):
    id = "code-saturne"
    name = "code_saturne (EnSight Gold / MED in RESU/<run>/postprocessing)"
    formats = ["code-saturne-resu", "ensight-gold", "med"]
    vendor = "EDF (open source, GPL solver; open output formats)"
    solver = "Code_Saturne"
    routes = ["ensight", "med", "cgns"]
    extensions = []
    maturity = "verified-public-sample"
    temporal = ["steady", "transient"]
    kinds = ["surface", "mesh"]
    export_help = (
        "No export step: code_saturne writes EnSight Gold by default into "
        "RESU/<run>/postprocessing/ (RESULTS_FLUID_DOMAIN.case; MED or CGNS when selected in "
        "the writer settings). Point at the case directory, the run directory or the .case; "
        "setup.xml and run_solver.log next to postprocessing/ supply the physics."
    )
    status_note = (
        "Reads RESU/<run>/postprocessing output (code_saturne 9.1, FVM 0.3): mesh including "
        "polyhedra, cell fields, solver version, turbulence model, fluid properties, notebook "
        "variables and boundary natures. Transient EnSight output and the MED and CGNS writers are "
        "validated on generated files only."
    )

    @staticmethod
    def _postprocessing(path: Path) -> Path | None:
        """The ``postprocessing`` directory behind a case / RESU / run / postprocessing path."""
        if not path.is_dir():
            return None
        if path.name == "postprocessing" and path.parent.parent.name == "RESU":
            return path
        if (path / "postprocessing").is_dir() and path.parent.name == "RESU":
            return path / "postprocessing"
        resu = path if path.name == "RESU" else path / "RESU"
        if resu.is_dir() and (path.name == "RESU" or (path / "DATA").is_dir()):
            runs = [r / "postprocessing" for r in _listdir(resu) if r.is_dir()]
            runs = [r for r in runs if r.is_dir()]
            return runs[-1] if runs else None  # run ids are timestamps: the last is the latest
        return None

    def find_export(self, path: Path) -> Target | None:
        from frontier_connector_ensight.connector import find_cases

        post = self._postprocessing(path)
        if post is not None:
            cases = find_cases(post)
            if cases:
                # the fluid domain is the volume export; boundary and probe writers come after
                cases.sort(key=lambda c: ("FLUID_DOMAIN" not in c.name.upper(), c.name))
                return "ensight", cases[0], f"code_saturne layout RESU/{post.parent.name}"
            meds = [f for f in _listdir(post) if f.suffix.lower() == ".med"]
            if meds:
                return "med", meds[0], f"code_saturne layout RESU/{post.parent.name} (MED)"
            return None
        if path.is_file() and path.parent.name == "postprocessing":
            if path.parent.parent.parent.name == "RESU":
                if match_extension(path, [".case"]):
                    return "ensight", path, "code_saturne postprocessing directory"
                if path.suffix.lower() == ".med":
                    return "med", path, "code_saturne postprocessing directory (MED)"
        return _case_with_marker(path, ["fvm library", "code_saturne", "code saturne"], depth=0)

    def enrich(self, desc: SourceDescription, target: Target) -> None:
        _, routed, _ = target
        version = _version_from_description(desc.metadata.get("geometry_description"))
        if version:
            desc.solver_version = version
        run = routed.parent.parent if routed.parent.name == "postprocessing" else None
        if run is None:
            return
        desc.metadata["run"] = run.name
        banner = _log_version(run)
        if banner:
            desc.solver_version = banner
        setup = run / "setup.xml"
        if not setup.is_file():
            return
        try:
            root = ET.parse(setup).getroot()
        except (ET.ParseError, OSError):
            desc.warnings.append(f"{setup.name} could not be parsed; physics left empty")
            return
        if not desc.solver_version and root.get("solver_version"):
            desc.solver_version = root.get("solver_version")
        _apply_setup(desc, root)


_VERSION = re.compile(r"code[_ ]saturne\s+version\s+([0-9][0-9A-Za-z.\-]*)", re.IGNORECASE)
#: ``<time_passing>`` of the GUI: time step option (``idtvar``)
_TIME_STEPPING = {
    "0": "constant time step",
    "1": "time-varying time step",
    "2": "local time step (steady, pseudo-time)",
    "-1": "steady (SIMPLE)",
}


def _version_from_description(lines: Any) -> str | None:
    """``'Output by code_saturne version 9.1.0'`` (second description line of the geometry
    file) -> ``'9.1.0'``."""
    for line in lines or []:
        m = _VERSION.search(str(line))
        if m:
            return m.group(1)
    return None


def _log_version(run: Path) -> str | None:
    """Version of the solver banner (``Version 9.1.0`` under the ``code_saturne`` logo) at the
    top of ``run_solver.log`` (``listing`` before version 6)."""
    for name in ("run_solver.log", "listing"):
        log = run / name
        if not log.is_file():
            continue
        try:
            with open(log, "rb") as f:
                head = f.read(4096).decode("utf-8", "replace")
        except OSError:
            continue
        if "saturne" not in head.lower():
            continue
        m = re.search(r"^\s*Version\s+([0-9][0-9A-Za-z.\-]*)\s*$", head, re.MULTILINE)
        if m:
            return m.group(1)
    return None


def _number(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _apply_setup(desc: SourceDescription, root: ET.Element) -> None:
    """Physics, fluid properties, notebook variables and boundary conditions of the GUI case
    file ``setup.xml`` (element names as written by code_saturne 8 and 9)."""
    physics = desc.physics
    # './/turbulence' would also match the inlet's <turbulence choice="..."> element
    turbulence = root.find("thermophysical_models/turbulence")
    if turbulence is not None and turbulence.get("model"):
        physics["turbulence_model"] = turbulence.get("model")
    for tag in ("thermal_scalar", "compressible_model", "gas_combustion", "radiative_transfer"):
        node = root.find(f"thermophysical_models/{tag}")
        if node is not None and node.get("model") not in (None, "off"):
            physics[tag] = node.get("model")
    algo = root.find("numerical_parameters/velocity_pressure_algo")
    if algo is not None and algo.get("choice"):
        physics["velocity_pressure_algorithm"] = algo.get("choice")
    passing = root.findtext("analysis_control/time_parameters/time_passing")
    if passing is not None and passing.strip() in _TIME_STEPPING:
        physics["time_stepping"] = _TIME_STEPPING[passing.strip()]
    for tag, key in (("time_step_ref", "time_step_ref"), ("iterations", "iterations")):
        value = _number(root.findtext(f"analysis_control/time_parameters/{tag}"))
        if value is not None:
            desc.parameters[key] = value
    fluid = root.find("physical_properties/fluid_properties")
    if fluid is not None:
        for prop in fluid.findall("property"):
            value = _number(prop.findtext("initial_value"))
            if prop.get("choice") == "constant" and value is not None and prop.get("name"):
                desc.parameters[str(prop.get("name"))] = value
        for tag in ("reference_pressure", "reference_temperature"):
            value = _number(fluid.findtext(tag))
            if value is not None:
                desc.parameters[tag] = value
    notebook: dict[str, Any] = {}
    for var in root.findall("physical_properties/notebook/var"):
        value = _number(var.get("value"))
        if var.get("name") and value is not None:
            notebook[str(var.get("name"))] = value
            desc.parameters[str(var.get("name"))] = value
    if notebook:
        desc.metadata["notebook"] = notebook
    meshes = [m.get("name") for m in root.findall("solution_domain/meshes_list/mesh")]
    if any(meshes):
        desc.metadata["meshes"] = [m for m in meshes if m]
    bcs = root.find("boundary_conditions")
    if bcs is None:
        return
    for b in bcs.findall("boundary"):
        label, nature = b.get("label"), b.get("nature")
        if not label or not nature:
            continue
        bc: dict[str, Any] = {"type": nature, "selection": (b.text or "").strip()}
        if (b.get("name") or "").isdigit():
            # the value of the boundary_zone_id field written with the boundary mesh
            bc["zone_id"] = int(str(b.get("name")))
        detail = next((d for d in bcs.findall(nature) if d.get("label") == label), None)
        vp = detail.find("velocity_pressure") if detail is not None else None
        if vp is not None:
            if vp.get("choice") and vp.get("choice") != "off":
                bc["velocity_pressure"] = vp.get("choice")
            for tag in ("norm", "norm_formula", "flow1", "flow2"):
                text = vp.findtext(tag)
                if text:
                    number = _number(text)
                    bc[tag] = number if number is not None else text.strip()
            direction = [_number(vp.findtext(f"direction_{a}")) for a in "xyz"]
            if all(d is not None for d in direction):
                bc["direction"] = direction
        desc.boundary_conditions[label] = bc
        for z in desc.zones:
            if z.name.lower() == label.lower():
                z.bc_type = nature
