# SPDX-License-Identifier: Apache-2.0
"""Connector errors. Every error carries a ``hint`` that tells the user what to do next."""

from __future__ import annotations

from typing import Any


class ConnectorError(Exception):
    """Base class. ``str(err)`` is ``"<message>\\n  hint: <hint>"`` when a hint is set."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  hint: {self.hint}" if self.hint else self.message

    def to_dict(self) -> dict[str, Any]:
        return {"error": type(self).__name__, "message": self.message, "hint": self.hint}


class PolicyError(ConnectorError):
    """The organisation policy forbids the action. ``to_dict()`` adds ``status``
    (``"blocked_by_policy"``), the ``policy`` setting, where it was set and whether an
    administrator locked it."""

    def __init__(self, violation: object) -> None:
        self.policy = str(getattr(violation, "policy", ""))
        self.origin = str(getattr(violation, "origin", ""))
        self.locked = bool(getattr(violation, "locked", False))
        self.reason = getattr(violation, "reason", None)
        message = str(getattr(violation, "message", violation))
        admin = self.locked or self.origin.startswith("set by your administrator")
        super().__init__(
            f"{message} (policy {self.policy}, {self.origin})",
            hint="ask your Frontier administrator"
            if admin
            else f"see `frontier config explain {self.policy}`",
        )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "status": "blocked_by_policy",
            "policy": self.policy,
            "origin": self.origin,
            "locked": self.locked,
            "reason": self.reason,
        }


class UnsupportedSourceError(ConnectorError):
    """The path is not something this connector (or any connector) can read."""


class MissingDependencyError(ConnectorError):
    """An optional Python package is needed for this source."""

    def __init__(
        self, package: str, *, extra: str | None = None, purpose: str = "", license: str = ""
    ) -> None:
        self.package = package
        self.extra = extra
        what = f" to {purpose}" if purpose else ""
        lic = f" ({license})" if license else ""
        # `extra` is "<distribution>[<extra>]" of the plugin that declares it; a bare name is
        # an extra of the framework itself
        if extra and "[" in extra:
            install = f"pip install '{extra}'"
        elif extra:
            install = f"pip install 'frontier-connectors[{extra}]'"
        else:
            install = f"pip install {package}"
        super().__init__(
            f"optional dependency '{package}'{lic} is required{what}", hint=f"run: {install}"
        )


class NeedsExportError(ConnectorError):
    """The native file is proprietary; run the shipped exporter inside the vendor tool first."""


class NeedsCredentialsError(ConnectorError):
    """A cloud connector was used without credentials in the environment."""


class SelectionError(ConnectorError):
    """A zone/field/time selection matched nothing."""


class MappingError(ConnectorError):
    """A mapping config (MATLAB/HDF5/NetCDF/CSV) is missing, ambiguous or inconsistent."""
