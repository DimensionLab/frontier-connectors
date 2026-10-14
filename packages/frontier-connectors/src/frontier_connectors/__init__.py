# SPDX-License-Identifier: Apache-2.0
"""frontier-connectors: plugins that read CFD/CAE solver output into ``frontier-dataset v1.1``.

.. code-block:: python

    import frontier_connectors as fc

    fc.detect("case/")                       # [Detection(connector_id="openfoam", ...)]
    desc = fc.describe("case/")              # zones, fields, time steps, solver metadata
    samples = fc.read("case/", zones=["internalMesh"], fields=["U", "p"], times="last")
    fc.to_frontier_dataset(samples, "datasets/case", description=desc)

Third-party connectors register under the entry-point group ``frontier.connectors``; see
the *Writing a connector* guide shipped with the package (``WRITING-A-CONNECTOR.md``).
"""

from frontier_connectors.convert import ConvertResult, convert, to_frontier_dataset
from frontier_connectors.errors import (
    ConnectorError,
    MappingError,
    MissingDependencyError,
    NeedsCredentialsError,
    NeedsExportError,
    SelectionError,
    UnsupportedSourceError,
)
from frontier_connectors.model import (
    MATURITIES,
    BaseConnector,
    Connector,
    Detection,
    FieldData,
    FieldInfo,
    RawSample,
    SourceDescription,
    TimeInfo,
    ZoneInfo,
    as_field,
)
from frontier_connectors.registry import (
    ENTRY_POINT_GROUP,
    ConnectorInfo,
    connectors,
    describe,
    detect,
    get,
    list_connectors,
    read,
    register,
)

__version__ = "0.9.0"

__all__ = [
    "ENTRY_POINT_GROUP",
    "MATURITIES",
    "BaseConnector",
    "Connector",
    "ConnectorError",
    "ConnectorInfo",
    "ConvertResult",
    "Detection",
    "FieldData",
    "FieldInfo",
    "MappingError",
    "MissingDependencyError",
    "NeedsCredentialsError",
    "NeedsExportError",
    "RawSample",
    "SelectionError",
    "SourceDescription",
    "TimeInfo",
    "UnsupportedSourceError",
    "ZoneInfo",
    "__version__",
    "as_field",
    "connectors",
    "convert",
    "describe",
    "detect",
    "get",
    "list_connectors",
    "read",
    "register",
    "to_frontier_dataset",
]
