# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from frontier_connectors import registry
from frontier_connectors.plugins.state import HOME_ENV
from frontier_connectors.testing import NETWORK_ENV, network_enabled


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"network: downloads small public sample files (runs only with {NETWORK_ENV}=1)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if network_enabled():
        return
    skip = pytest.mark.skip(reason=f"network test: set {NETWORK_ENV}=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_frontier_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point ``FRONTIER_HOME`` at an empty directory: tests never read or write the developer's
    real ``~/.frontier/plugins.toml`` (a plugin disabled there must not fail the suite, and a
    test that disables one must not leak)."""
    home = tmp_path_factory.mktemp("frontier-home")
    monkeypatch.setenv(HOME_ENV, str(home))
    # ... nor the developer's `.env`, config files, secret store or network settings
    monkeypatch.setenv("FRONTIER_DEV", "0")
    monkeypatch.setenv("FRONTIER_SYSTEM_CONFIG", str(home / "system.toml"))
    # The built-in `policy.plugin_sources` is the official index plus the configured index
    # and wheel directory. These tests install from temporary directories and example URLs
    # named on the command line, which an administrator has to allow: this is that
    # administrator's system file. Tests about the policy itself write their own.
    (home / "system.toml").write_text('[policy]\nplugin_sources = ["any"]\n', encoding="utf-8")
    for var in ("FRONTIER_CREDENTIALS_BOUND", "FRONTIER_SECRETS__BACKEND"):
        monkeypatch.delenv(var, raising=False)
    registry.reload()
    yield home
    registry.reload()
