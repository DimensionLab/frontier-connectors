# SPDX-License-Identifier: Apache-2.0
"""Plugin management: manifests, discovery, enable/disable state, install/uninstall.

``frontier-connectors`` itself is the framework only. Every connector, vendor-side exporter and
importer lives in its own distribution and is found through entry points:

==================== =========================== ====================================
entry-point group    value                       what it provides
==================== =========================== ====================================
frontier.plugins     import package              the package holding ``plugin.toml``
frontier.connectors  ``module:Class``            a :class:`~frontier_connectors.Connector`
frontier.importers   ``module:Class``            an :class:`~frontier_connectors.importers.Importer`
frontier.exporters   import package              vendor-side scripts under ``files/``
==================== =========================== ====================================

Command line: ``python -m frontier_connectors plugins list|search|info|install|uninstall|
enable|disable|doctor|install-exporter|uninstall-exporter``.
"""

from frontier_connectors.plugins.discovery import (
    PluginRecord,
    catalog,
    discover,
    find,
    missing_hint,
)
from frontier_connectors.plugins.exporters import (
    exporter_dir,
    exporter_files,
    install_exporter,
    list_exporters,
    uninstall_exporter,
)
from frontier_connectors.plugins.health import doctor
from frontier_connectors.plugins.installer import install, search, uninstall
from frontier_connectors.plugins.manage import set_enabled
from frontier_connectors.plugins.manifest import (
    DATA_MODEL_VERSION,
    GROUP_CONNECTORS,
    GROUP_EXPORTERS,
    GROUP_IMPORTERS,
    GROUP_PLUGINS,
    KINDS,
    ManifestError,
    PluginManifest,
    data_model_compatible,
    load_file,
    validate,
)
from frontier_connectors.plugins.state import PluginState, frontier_home, state_path

__all__ = [
    "DATA_MODEL_VERSION",
    "GROUP_CONNECTORS",
    "GROUP_EXPORTERS",
    "GROUP_IMPORTERS",
    "GROUP_PLUGINS",
    "KINDS",
    "ManifestError",
    "PluginManifest",
    "PluginRecord",
    "PluginState",
    "catalog",
    "data_model_compatible",
    "discover",
    "doctor",
    "exporter_dir",
    "exporter_files",
    "find",
    "frontier_home",
    "install",
    "install_exporter",
    "list_exporters",
    "load_file",
    "missing_hint",
    "search",
    "set_enabled",
    "state_path",
    "uninstall",
    "uninstall_exporter",
    "validate",
]
