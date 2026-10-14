// SPDX-License-Identifier: Apache-2.0
//! Connector plugin contract.
//!
//! A connector turns the on-disk output of a simulation tool (an OpenFOAM case, a
//! Fluent `.cas.h5/.dat.h5` pair, a STAR-CCM+ export, a VTK file tree, a CGNS file, ...)
//! into Frontier's canonical representation: a [`Sample`](SampleReader) with a mesh,
//! fields, and the simulation metadata needed to condition a model (boundary conditions,
//! physics model, Reynolds number, geometry provenance).
//!
//! Division of labour:
//!
//! * **Rust** owns *detection*: [`formats::detect`] recognises every supported format from
//!   magic bytes, extensions and directory layout without starting an interpreter, and the
//!   native connector crates (`frontier-connector-openfoam`, `frontier-connector-vtk`) can
//!   describe their formats without Python at all.
//! * **Python** (`packages/frontier-connectors`) owns the readers. [`python::PythonBridge`]
//!   runs `python -m frontier_connectors describe|convert --json` in a subprocess;
//!   [`ConnectorRegistry::open`] uses it for every format and falls back to a native
//!   connector when Python is unavailable.
//!
//! * **Plugins**: every connector, vendor-side exporter and importer is an installable Python
//!   distribution. [`plugins`] exposes the list, the enable/disable switches, install /
//!   uninstall and the health report of `python -m frontier_connectors plugins ...`.
//!
//! Connectors never load whole volumes eagerly; they expose a lazy [`SampleReader`]
//! so that the Rust data pipeline can stream terabyte-scale datasets.

pub mod formats;
pub mod plugins;
pub mod python;

use frontier_core::{Plugin, Result};
use frontier_data::{FieldDescriptor, MeshDescriptor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use formats::{Access, FORMATS, FormatSpec};
pub use plugins::{PluginInfo, PluginKind, PluginList, PluginStatus, PluginTarget};
pub use python::{ConvertOptions, ConvertReport, PythonBridge, SourceDescription};

/// Environment variable selecting how `open` describes a source: `auto` (default: Python
/// reader first, native connector as fallback), `native` (never start Python) or `python`
/// (never fall back).
pub const DESCRIBE_ENV: &str = "FRONTIER_CONNECTORS_DESCRIBE";

/// What a connector found at a path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub connector_id: String,
    pub root: PathBuf,
    pub confidence: f32,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Metadata of one simulation sample, independent of the solver that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub solver: String,
    pub solver_version: Option<String>,
    pub mesh: MeshDescriptor,
    pub fields: Vec<FieldDescriptor>,
    /// Available time steps (empty for steady-state).
    pub times: Vec<f64>,
    /// Free-form, solver-specific settings worth keeping for conditioning/lineage.
    #[serde(default)]
    pub attributes: BTreeMap<String, serde_json::Value>,
}

/// Lazy reader for a single sample.
pub trait SampleReader: Send {
    fn metadata(&self) -> &SampleMetadata;
}

/// A simulation-software connector.
pub trait Connector: Plugin {
    /// Cheap check whether `path` looks like something this connector can read.
    fn detect(&self, path: &Path) -> Option<Detection>;

    /// Open a sample rooted at `path`.
    fn open(&self, path: &Path) -> Result<Box<dyn SampleReader>>;
}

/// How [`ConnectorRegistry::open`] obtains metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescribeMode {
    /// Python reader first (richest metadata), native connector when Python is unavailable.
    Auto,
    /// Native connectors only; formats without one are unsupported.
    Native,
    /// Python reader only.
    Python,
}

impl DescribeMode {
    pub fn from_env() -> Self {
        match std::env::var(DESCRIBE_ENV).as_deref() {
            Ok("native") => Self::Native,
            Ok("python") => Self::Python,
            _ => Self::Auto,
        }
    }
}

/// Sample whose metadata came from the Python package.
struct DescribedSample {
    metadata: SampleMetadata,
}

impl SampleReader for DescribedSample {
    fn metadata(&self) -> &SampleMetadata {
        &self.metadata
    }
}

/// Registry of connectors: native in-process connectors plus the format catalogue that is
/// served by the Python package.
pub struct ConnectorRegistry {
    connectors: Vec<Box<dyn Connector>>,
    catalog: bool,
    mode: DescribeMode,
    bridge: Option<PythonBridge>,
}

impl Default for ConnectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectorRegistry {
    /// Registry that knows every format of [`formats::FORMATS`] (detection in Rust,
    /// `describe`/`convert` delegated to Python). Register native connectors on top.
    pub fn new() -> Self {
        Self {
            connectors: Vec::new(),
            catalog: true,
            mode: DescribeMode::from_env(),
            bridge: None,
        }
    }

    /// Registry with native connectors only: no format catalogue, never starts Python.
    pub fn native_only() -> Self {
        Self {
            connectors: Vec::new(),
            catalog: false,
            mode: DescribeMode::Native,
            bridge: None,
        }
    }

    pub fn with_mode(mut self, mode: DescribeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Use a specific Python command instead of resolving one from the environment.
    pub fn with_bridge(mut self, bridge: PythonBridge) -> Self {
        self.bridge = Some(bridge);
        self
    }

    pub fn register(&mut self, connector: Box<dyn Connector>) -> &mut Self {
        self.connectors.push(connector);
        self
    }

    /// Ids of the native connectors followed by the catalogue ids served by Python.
    pub fn ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .connectors
            .iter()
            .map(|c| c.manifest().id.as_str())
            .collect();
        if self.catalog {
            ids.extend(FORMATS.iter().map(|f| f.id));
        }
        ids
    }

    fn bridge(&self) -> PythonBridge {
        self.bridge.clone().unwrap_or_else(PythonBridge::from_env)
    }

    /// Run every native connector's detection and the fast catalogue detection against
    /// `path`, best match first.
    pub fn detect(&self, path: &Path) -> Vec<Detection> {
        let mut found: Vec<Detection> = self
            .connectors
            .iter()
            .filter_map(|c| c.detect(path))
            .collect();
        if self.catalog {
            found.extend(formats::detect(path));
        }
        found.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        found
    }

    fn native(&self, id: &str) -> Option<&dyn Connector> {
        self.connectors
            .iter()
            .find(|c| c.manifest().id == id)
            .map(|c| c.as_ref())
    }

    /// Describe `path` with the Python package (zones, fields, times, solver metadata).
    pub fn describe(&self, path: &Path, connector: Option<&str>) -> Result<SourceDescription> {
        self.bridge().describe(path, connector)
    }

    /// Convert `path` into a `frontier-dataset v1.1` directory through the Python package.
    pub fn convert(&self, path: &Path, out: &Path, opts: &ConvertOptions) -> Result<ConvertReport> {
        self.bridge().convert(path, out, opts)
    }

    /// Open the sample at `path` with the best-matching connector.
    ///
    /// In [`DescribeMode::Auto`] the Python reader of the best catalogue match is tried
    /// first; when Python cannot be started (or the format has no Python reader) the best
    /// native connector is used. An error reported *by the reader* (corrupt file, missing
    /// optional dependency, proprietary file that needs the exporter) is returned as is
    /// when no native connector can take over.
    pub fn open(&self, path: &Path) -> Result<Box<dyn SampleReader>> {
        let found = self.detect(path);
        if found.is_empty() {
            return Err(frontier_core::Error::Unsupported(format!(
                "no connector recognises {}",
                path.display()
            )));
        }
        let native = found
            .iter()
            .find_map(|d| self.native(&d.connector_id).map(|c| (c, d)));
        let python = found.iter().find(|d| {
            self.native(&d.connector_id).is_none() && formats::spec(&d.connector_id).is_some()
        });

        let mut python_err = None;
        if self.mode != DescribeMode::Native
            && let Some(det) = python
        {
            match self.bridge().describe(path, Some(&det.connector_id)) {
                Ok(desc) => {
                    return Ok(Box::new(DescribedSample {
                        metadata: desc.into_sample_metadata(),
                    }));
                }
                Err(e) => python_err = Some(e),
            }
        }
        if self.mode != DescribeMode::Python
            && let Some((connector, _)) = native
        {
            return connector.open(path);
        }
        Err(python_err.unwrap_or_else(|| {
            frontier_core::Error::Unsupported(format!(
                "{} is recognised as `{}` but no native reader exists and the Python reader \
                 is disabled ({DESCRIBE_ENV}=native)",
                path.display(),
                found[0].connector_id
            ))
        }))
    }
}

#[cfg(test)]
mod tests;
