# SPDX-License-Identifier: Apache-2.0
"""The offline guard of the connector framework.

Every place in ``frontier_connectors`` and its plugins that opens an outbound connection calls
:func:`require_online` first: the cloud transports and vendor SDK calls, the plugin index
fetch, the package installer and the public-sample download of the test helpers.

With the Frontier SDK installed the decision is ``frontier.config.require_online`` (setting
``network.offline``, from every configuration layer). Without it (``frontier`` is an optional
dependency of the framework) the variables the Frontier launcher exports to child processes
are honoured: ``FRONTIER_OFFLINE`` and ``FRONTIER_NETWORK__OFFLINE``.

The guard raises before any socket is opened, and before credentials are read.
"""

from __future__ import annotations

import os

from frontier_connectors import credentials
from frontier_connectors.errors import ConnectorError

__all__ = ["OFFLINE_ENV", "OfflineError", "offline", "require_online"]

#: environment variables that switch offline mode on when the SDK is not installed
OFFLINE_ENV = ("FRONTIER_OFFLINE", "FRONTIER_NETWORK__OFFLINE")
_TRUE = ("1", "true", "yes", "on")


class OfflineError(ConnectorError):
    """An outbound connection was refused because ``network.offline`` is set. Raised from
    ``frontier.config.OfflineError`` (its ``__cause__``) when the SDK made the decision."""


def _env_offline() -> bool:
    return any(os.environ.get(v, "").strip().lower() in _TRUE for v in OFFLINE_ENV)


def _refuse(what: str, cause: BaseException | None = None) -> OfflineError:
    err = OfflineError(
        f"offline mode is on (setting network.offline), so {what} is not possible",
        hint="already downloaded results convert without a connection: point at the result "
        "directory; or switch the setting off (`frontier config set network.offline false`)",
    )
    err.__cause__ = cause
    return err


def require_online(what: str, *, url: str | None = None) -> None:
    """Raise :class:`OfflineError` when outbound connections are switched off. ``what`` reads
    as an activity ("contacting the Flow360 API"). A loopback ``url`` is not outbound."""
    module = credentials.sdk("config")
    guard = getattr(module, "require_online", None) if module is not None else None
    if guard is not None:
        sdk_error = getattr(module, "OfflineError", RuntimeError)
        try:
            guard(what, url=url)
        except sdk_error as e:
            raise _refuse(what, e) from e
        except Exception:  # noqa: BLE001 - a damaged configuration must not open the network
            if _env_offline():
                raise _refuse(what) from None
        return
    if not _env_offline():
        return
    if url is not None and _loopback(url):
        return
    raise _refuse(what)


def offline() -> bool:
    """Is offline mode on? (For planning; call :func:`require_online` at the call site.)"""
    try:
        require_online("a connection")
    except OfflineError:
        return True
    return False


def _loopback(url: str) -> bool:
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
