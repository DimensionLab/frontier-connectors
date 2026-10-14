# SPDX-License-Identifier: Apache-2.0
"""Small unit handling layer: parse unit strings, convert to SI, name common CAE unit systems.

Connectors report the units the *source* uses (``FieldData.units``, ``SourceDescription.units``).
Conversion is explicit: ``convert_sample_units(sample, length="m")`` or ``convert --si``; nothing
is rescaled silently, because most solver files do not record units at all.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# name -> (SI factor, SI offset, SI unit)
_UNITS: dict[str, tuple[float, float, str]] = {
    # length
    "m": (1.0, 0.0, "m"),
    "mm": (1e-3, 0.0, "m"),
    "cm": (1e-2, 0.0, "m"),
    "um": (1e-6, 0.0, "m"),
    "km": (1e3, 0.0, "m"),
    "in": (0.0254, 0.0, "m"),
    "ft": (0.3048, 0.0, "m"),
    # time
    "s": (1.0, 0.0, "s"),
    "ms": (1e-3, 0.0, "s"),
    "us": (1e-6, 0.0, "s"),
    "min": (60.0, 0.0, "s"),
    "h": (3600.0, 0.0, "s"),
    # mass
    "kg": (1.0, 0.0, "kg"),
    "g": (1e-3, 0.0, "kg"),
    "t": (1e3, 0.0, "kg"),
    "tonne": (1e3, 0.0, "kg"),
    "lbm": (0.45359237, 0.0, "kg"),
    "slug": (14.5939029, 0.0, "kg"),
    # temperature
    "K": (1.0, 0.0, "K"),
    "degC": (1.0, 273.15, "K"),
    "C": (1.0, 273.15, "K"),
    "degF": (5.0 / 9.0, 255.3722222222222, "K"),
    "R": (5.0 / 9.0, 0.0, "K"),
    # pressure / stress
    "Pa": (1.0, 0.0, "Pa"),
    "kPa": (1e3, 0.0, "Pa"),
    "MPa": (1e6, 0.0, "Pa"),
    "GPa": (1e9, 0.0, "Pa"),
    "bar": (1e5, 0.0, "Pa"),
    "mbar": (1e2, 0.0, "Pa"),
    "atm": (101325.0, 0.0, "Pa"),
    "psi": (6894.757293168, 0.0, "Pa"),
    "ksi": (6894757.293168, 0.0, "Pa"),
    # force / energy / power
    "N": (1.0, 0.0, "N"),
    "kN": (1e3, 0.0, "N"),
    "lbf": (4.4482216152605, 0.0, "N"),
    "J": (1.0, 0.0, "J"),
    "mJ": (1e-3, 0.0, "J"),
    "kJ": (1e3, 0.0, "J"),
    "W": (1.0, 0.0, "W"),
    "kW": (1e3, 0.0, "W"),
    # velocity
    "m/s": (1.0, 0.0, "m/s"),
    "mm/s": (1e-3, 0.0, "m/s"),
    "km/h": (1.0 / 3.6, 0.0, "m/s"),
    "ft/s": (0.3048, 0.0, "m/s"),
    "kn": (0.514444, 0.0, "m/s"),
    # angle
    "rad": (1.0, 0.0, "rad"),
    "deg": (np.pi / 180.0, 0.0, "rad"),
    # dimensionless
    "": (1.0, 0.0, ""),
    "-": (1.0, 0.0, ""),
    "1": (1.0, 0.0, ""),
}

_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "millimeter": "mm",
    "millimeters": "mm",
    "centimeter": "cm",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "millisecond": "ms",
    "kilogram": "kg",
    "gram": "g",
    "kelvin": "K",
    "celsius": "degC",
    "°c": "degC",
    "degc": "degC",
    "fahrenheit": "degF",
    "°f": "degF",
    "degf": "degF",
    "rankine": "R",
    "pascal": "Pa",
    "pa": "Pa",
    "mpa": "MPa",
    "gpa": "GPa",
    "kpa": "kPa",
    "n": "N",
    "newton": "N",
    "degree": "deg",
    "degrees": "deg",
    "radian": "rad",
    "radians": "rad",
    "none": "",
    "dimensionless": "",
    "nondimensional": "",
}

#: Consistent unit systems used by structural codes (LS-DYNA, Abaqus, Nastran do not store units).
UNIT_SYSTEMS: dict[str, dict[str, str]] = {
    "si": {
        "length": "m", "mass": "kg", "time": "s", "force": "N", "stress": "Pa", "energy": "J"
    },
    "mm-t-s": {
        "length": "mm", "mass": "t", "time": "s", "force": "N", "stress": "MPa", "energy": "mJ"
    },
    "mm-kg-ms": {
        "length": "mm", "mass": "kg", "time": "ms", "force": "kN", "stress": "GPa", "energy": "J"
    },
    "mm-g-ms": {
        "length": "mm", "mass": "g", "time": "ms", "force": "N", "stress": "MPa", "energy": "mJ"
    },
    "in-lbf-s": {
        "length": "in", "mass": "lbf*s^2/in", "time": "s", "force": "lbf", "stress": "psi",
        "energy": "lbf*in",
    },
}  # fmt: skip


def normalize_unit(unit: str | None) -> str:
    """Canonical spelling of a unit string (``"Meter"`` -> ``"m"``); unknown ones pass through."""
    if unit is None:
        return ""
    u = unit.strip().strip("[]")
    if u in _UNITS:
        return u
    return _ALIASES.get(u.lower(), u)


def si_factor(unit: str | None) -> tuple[float, float, str]:
    """``(factor, offset, si_unit)`` such that ``si = value * factor + offset``.

    Raises ``KeyError`` with the list of known units for anything it cannot parse; compound
    units of the form ``a*b/c^n`` built from known atoms are supported (no offsets).
    """
    u = normalize_unit(unit)
    if u in _UNITS:
        return _UNITS[u]
    factor = 1.0
    parts: list[str] = []
    for sign, token in re.findall(r"([*/]?)\s*([^*/\s]+)", u):
        m = re.match(r"^(.+?)(?:\^|\*\*)?(-?\d+)?$", token)
        if not m:
            raise KeyError(unit)
        atom, power = normalize_unit(m.group(1)), int(m.group(2) or 1)
        if atom not in _UNITS or _UNITS[atom][1] != 0.0:
            raise KeyError(
                f"unknown unit {unit!r}; known atoms: {', '.join(sorted(k for k in _UNITS if k))}"
            )
        f, _, si = _UNITS[atom]
        p = -power if sign == "/" else power
        factor *= f**p
        if si:
            parts.append(si if p == 1 else f"{si}^{p}")
    return factor, 0.0, "*".join(parts)


def to_si(values: Any, unit: str | None) -> tuple[np.ndarray, str]:
    """Convert ``values`` in ``unit`` to SI. Returns ``(array, si_unit)``."""
    f, off, si = si_factor(unit)
    arr = np.asarray(values, dtype=np.float64)
    return arr * f + off, si


def length_scale(from_unit: str | None, to_unit: str = "m") -> float:
    """Multiplier turning lengths in ``from_unit`` into ``to_unit``."""
    return si_factor(from_unit)[0] / si_factor(to_unit)[0]


_FOAM_BASE = ("kg", "m", "s", "K", "mol", "A", "cd")


def foam_dimensions_to_unit(dims: list[float] | tuple[float, ...]) -> str:
    """OpenFOAM ``dimensions [kg m s K mol A cd]`` -> unit string (``[0 2 -2 0 0 0 0]`` ->
    ``"m^2/s^2"``)."""
    num, den = [], []
    for base, p in zip(_FOAM_BASE, dims, strict=False):
        if p == 0:
            continue
        mag = abs(p)
        txt = base if mag == 1 else f"{base}^{mag:g}"
        (num if p > 0 else den).append(txt)
    if not num and not den:
        return ""
    head = "*".join(num) if num else "1"
    return head + ("/" + "/".join(den) if den else "")


def convert_sample_units(
    sample: Any, *, length: str | None = None, source_length: str | None = None
) -> Any:
    """Rescale ``sample.points`` (and grid ``coords``) from ``source_length`` (default: the
    sample's ``metadata["units"]["length"]``) to ``length``. Returns the same object."""
    if length is None:
        return sample
    units = sample.metadata.setdefault("units", {})
    src = source_length or units.get("length")
    if not src:
        raise KeyError(
            "source length unit unknown: pass source_length=... (CLI: --source-length mm)"
        )
    k = length_scale(src, length)
    if k != 1.0:
        if sample.points is not None:
            sample.points = (np.asarray(sample.points, dtype=np.float64) * k).astype(
                sample.points.dtype
            )
        for name in list(sample.coords):
            sample.coords[name] = np.asarray(sample.coords[name], dtype=np.float64) * k
    units["length"] = length
    units["length_converted_from"] = src
    return sample
