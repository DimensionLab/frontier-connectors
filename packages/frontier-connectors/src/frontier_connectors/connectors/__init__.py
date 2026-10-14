# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for the import paths used before connectors became plugins.

Until 0.1 every connector lived in ``frontier_connectors.connectors.<module>``. Each family is
now its own distribution with its own import package (``frontier_connector_fluent`` ...). Old
imports keep working, with a :class:`DeprecationWarning`, as long as the plugin is installed:

.. code-block:: python

    from frontier_connectors.connectors.fluent import FluentCffConnector   # old, still works
    from frontier_connector_fluent.connector import FluentCffConnector     # new

Looking connectors up by id (``frontier_connectors.get("fluent-cff")``) needs no change.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from collections.abc import Sequence
from types import ModuleType
from typing import Any

#: old submodule name -> module that replaced it
LEGACY_MODULES: dict[str, str] = {
    "_arraymap": "frontier_connectors._arraymap",
    "_matlab_io": "frontier_connector_matlab._matlab_io",
    "_matlab_layout": "frontier_connector_matlab._matlab_layout",
    "_meshio": "frontier_connectors._meshio",
    "abaqus": "frontier_connector_abaqus.connector",
    "ansys": "frontier_connector_ansys.connector",
    "cgns": "frontier_connector_cgns.connector",
    "cloud": "frontier_connectors.cloud",
    "comsol": "frontier_connector_comsol.connector",
    "ensight": "frontier_connector_ensight.connector",
    "fluent": "frontier_connector_fluent.connector",
    "geometry": "frontier_connector_geometry.connector",
    "grid": "frontier_connector_arrays.connector",
    "lsdyna": "frontier_connector_lsdyna.connector",
    "matlab": "frontier_connector_matlab.connector",
    "meshio_formats": "frontier_connector_meshio.connector",
    "nastran": "frontier_connector_nastran.connector",
    "nek5000": "frontier_connector_nek5000.connector",
    "openfoam": "frontier_connector_openfoam.connector",
    "pyfr": "frontier_connector_pyfr.connector",
    "starccm": "frontier_connector_starccm.connector",
    "tabular": "frontier_connector_tabular.connector",
    "tecplot": "frontier_connector_tecplot.connector",
    "vtk": "frontier_connector_vtk.connector",
}

#: classes of the former ``connectors.vendors`` module -> where each one lives now
LEGACY_VENDOR_CLASSES: dict[str, str] = {
    "AcuSolveConnector": "frontier_connector_acusolve.connector",
    "CatalystConnector": "frontier_connector_catalyst.connector",
    "CodeSaturneConnector": "frontier_connector_code_saturne.connector",
    "ConvergeConnector": "frontier_connector_converge.connector",
    "FidelityConnector": "frontier_connector_fidelity.connector",
    "OptiStructConnector": "frontier_connector_optistruct.connector",
    "PalabosConnector": "frontier_connector_palabos.connector",
    "PowerFlowConnector": "frontier_connector_powerflow.connector",
    "SimulinkFmuConnector": "frontier_connector_simulink.connector",
    "VendorRouteConnector": "frontier_connectors.routing",
    "converge_post_info": "frontier_connector_converge.connector",
}

_PREFIX = __name__ + "."


def _warn(old: str, new: str) -> None:
    warnings.warn(
        f"{_PREFIX}{old} moved to {new} (connectors are plugins now); update the import",
        DeprecationWarning,
        stacklevel=3,
    )


def _vendors_module() -> ModuleType:
    """A stand-in for the former ``vendors`` module that resolves each class lazily."""
    mod = ModuleType(_PREFIX + "vendors", "Former home of the vendor route connectors.")

    def _getattr(name: str) -> Any:
        target = LEGACY_VENDOR_CLASSES.get(name)
        if target is None:
            raise AttributeError(f"module {mod.__name__!r} has no attribute {name!r}")
        return getattr(importlib.import_module(target), name)

    mod.__getattr__ = _getattr  # type: ignore[method-assign]
    return mod


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:  # already executed under its real name
        return None


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_PREFIX):
            return None
        old = fullname[len(_PREFIX) :]
        if old == "vendors":
            _warn(old, "one plugin per vendor (frontier_connector_<vendor>.connector)")
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(_vendors_module()))
        new = LEGACY_MODULES.get(old)
        if new is None:
            return None
        try:
            module = importlib.import_module(new)
        except ModuleNotFoundError as e:
            if e.name and new.startswith(e.name):
                raise ModuleNotFoundError(
                    f"{fullname} moved to {new}, whose plugin is not installed: run "
                    f"`python -m frontier_connectors plugins install "
                    f"{new.split('.')[0].removeprefix('frontier_connector_').replace('_', '-')}`",
                    name=fullname,
                ) from e
            raise
        _warn(old, new)
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(module))


if not any(isinstance(f, _LegacyFinder) for f in sys.meta_path):
    sys.meta_path.append(_LegacyFinder())


def __getattr__(name: str) -> Any:
    if name in LEGACY_MODULES or name == "vendors":
        return importlib.import_module(_PREFIX + name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
