# SPDX-License-Identifier: Apache-2.0
"""``policy.allow_plugin_changes = false`` freezes the plugin set: install, uninstall,
enable, disable and exporter deployment are refused before anything is touched."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("frontier.config")

from frontier_connectors.errors import PolicyError
from frontier_connectors.plugins import exporters, installer, manage


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    system = tmp_path / "system.toml"
    system.write_text(
        "[policy]\nallow_plugin_changes = false\n"
        '[lock]\n"policy.allow_plugin_changes" = "Validated toolchain"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("FRONTIER_SYSTEM_CONFIG", str(system))
    import frontier.config as config

    config.reset()
    yield tmp_path
    config.reset()


def test_every_change_is_refused_with_the_reason(frozen: Path) -> None:
    calls = [
        lambda: installer.install(["fluent"]),
        lambda: installer.uninstall(["fluent"]),
        lambda: manage.set_enabled("fluent", False),
        lambda: manage.set_enabled("fluent", True),
        lambda: exporters.install_exporter("fluent", frozen / "out"),
        lambda: exporters.uninstall_exporter("fluent"),
    ]
    for call in calls:
        with pytest.raises(PolicyError) as err:
            call()
        assert err.value.policy == "policy.allow_plugin_changes"
        assert err.value.locked
        assert "managed by your administrator" in str(err.value)
        assert err.value.to_dict()["status"] == "blocked_by_policy"
    assert not (frozen / "out").exists()


def test_changes_are_allowed_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRONTIER_SYSTEM_CONFIG", str(tmp_path / "none.toml"))
    import frontier.config as config

    config.reset()
    installer.check_change("install", "fluent")
