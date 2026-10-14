# SPDX-License-Identifier: Apache-2.0
"""Credentials and settings of the connector framework, through the Frontier SDK when it is
installed.

The framework keeps its hard dependencies minimal, so the SDK (distribution ``frontier``,
modules ``frontier.auth`` and ``frontier.config``; extra ``frontier-connectors[config]``) is
imported lazily:

* **with the SDK**, a credential named by its environment variable (``FLOW360_APIKEY``) is
  mapped to the provider and field that declare it (``[[auth]]`` in a ``plugin.toml``, or a
  built-in provider) and resolved in the SDK's order: job binding, the local secret store
  (``frontier auth login flow360``), the environment, and last the legacy keyring entry;
  settings (``plugins.*``, ``storage.cache_dir``, ``policy.plugin_sources``) come from the
  layered configuration;
* **without it**, behaviour is what it always was: the environment variable, then (when the
  optional ``keyring`` package is installed) the keyring entry ``frontier-connectors`` /
  ``<variable>``; no configuration files and no policy.

Values are read at call time and never stored, logged or put in messages.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

#: service name of the legacy keyring entries (account = the environment variable name)
KEYRING_SERVICE = "frontier-connectors"


def sdk(name: str) -> Any | None:
    """``frontier.<name>`` (``auth`` or ``config``) when the SDK is installed, else ``None``.
    A broken SDK counts as absent: connectors must keep reading files."""
    try:
        return importlib.import_module(f"frontier.{name}")
    except Exception:  # noqa: BLE001 - ImportError, or anything a damaged install raises
        return None


def config() -> Any | None:
    """The process-wide ``frontier.config.Config``, or ``None`` without the SDK."""
    module = sdk("config")
    if module is None:
        return None
    try:
        return module.get_config()
    except Exception:  # noqa: BLE001 - loading is lenient; this is a last line of defence
        return None


def setting(key: str) -> Any | None:
    """The value of a setting when the SDK is installed **and** some layer sets it (a config
    file, the environment, a flag); ``None`` otherwise, so the caller's own default applies."""
    cfg = config()
    if cfg is None:
        return None
    try:
        return cfg.get(key) if cfg.is_set(key) else None
    except KeyError:
        return None


def _legacy(var: str) -> str | None:
    value = os.environ.get(var, "").strip()
    if value:
        return value
    try:
        keyring: Any = importlib.import_module("keyring")
    except ImportError:
        return None
    try:
        stored = keyring.get_password(KEYRING_SERVICE, var)
    except Exception:  # noqa: BLE001 - no keyring backend on this machine
        return None
    return stored.strip() if isinstance(stored, str) and stored.strip() else None


def keyring_installed() -> bool:
    """Whether the optional ``keyring`` package can be imported (nothing is asked of it)."""
    try:
        return importlib.import_module("keyring") is not None
    except ImportError:
        return False


def present(var: str) -> bool:
    """Whether the credential ``var`` is known to be configured, found **without** touching
    the OS keychain or the network (job bindings, the credentials file and the environment
    only). For listings; ``False`` also covers "only the keychain could tell"."""
    if os.environ.get(var, "").strip():
        return True
    auth = sdk("auth")
    found = provider_of(var) if auth is not None else None
    if auth is None or found is None:
        return False
    try:
        st = auth.quiet_status(found[0])
    except Exception:  # noqa: BLE001
        return False
    if st is None:
        return False
    return any(f["name"] == found[1] and f["configured"] for f in st["fields"])


def provider_of(var: str) -> tuple[str, str] | None:
    """``(provider id, field)`` of the credential that accepts the environment variable
    ``var``, when the SDK knows one."""
    auth = sdk("auth")
    if auth is None:
        return None
    try:
        return auth.provider_for_env(var)
    except Exception:  # noqa: BLE001
        return None


def secret(var: str) -> str | None:
    """Value of the credential known by the environment variable ``var``, or ``None``."""
    auth = sdk("auth")
    found = provider_of(var) if auth is not None else None
    if auth is not None and found is not None:
        try:
            resolved = auth.resolve(*found)
        except Exception:  # noqa: BLE001 - an unusable store: fall back to the old lookup
            return _legacy(var)
        return resolved.value if resolved is not None else None
    return _legacy(var)


def login_hint(var: str) -> str:
    """How to provide the credential ``var``: the login command when the SDK knows the
    provider, else the environment variable."""
    found = provider_of(var)
    if found is not None:
        return f"run `frontier auth login {found[0]}`, or `export {var}=...`"
    return f"`export {var}=...`"
