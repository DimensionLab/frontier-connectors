# SPDX-License-Identifier: Apache-2.0
from frontier_connectors.testing import check_plugin_distribution


def test_installed_plugin_contract():
    check_plugin_distribution("frontier-connector-meshio")
