// SPDX-License-Identifier: Apache-2.0
//! Plugin management through the Python package.
//!
//! Every connector, vendor-side exporter and importer is a Python distribution found through
//! entry points; which of them are loaded is decided by the Python registry from the plugin
//! manifests and from `~/.frontier/plugins.toml` (`FRONTIER_HOME` moves it). Rust does not
//! re-implement that: the types here mirror the JSON documents printed by
//! `python -m frontier_connectors plugins <command> --json`, so that `frontier plugins ...`
//! and the application's Plugins page show exactly what the Python side will do.
//!
//! | Rust | Python command |
//! |------|----------------|
//! | [`PythonBridge::plugins`] | `plugins list [--available]` |
//! | [`PythonBridge::search_plugins`] | `plugins search <query> [--find-links DIR]` |
//! | [`PythonBridge::plugin_info`] | `plugins info <id>` |
//! | [`PythonBridge::set_plugin_enabled`] | `plugins enable <id>` / `plugins disable <id>` |
//! | [`PythonBridge::install_plugins`] | `plugins install <spec>... [--find-links DIR] [--no-index]` |
//! | [`PythonBridge::uninstall_plugins`] | `plugins uninstall <id>...` |
//! | [`PythonBridge::plugin_doctor`] | `plugins doctor` |
//! | [`PythonBridge::exporters`] | `plugins exporters` |
//! | [`PythonBridge::install_exporter`] | `plugins install-exporter <vendor> --to DIR` |
//! | [`PythonBridge::uninstall_exporter`] | `plugins uninstall-exporter <vendor> [--from DIR]` |

use crate::python::PythonBridge;
use frontier_core::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// What a plugin is: `connector`, `exporter` or `importer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginKind {
    #[default]
    Connector,
    Exporter,
    Importer,
}

/// Load state of a plugin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginStatus {
    /// Installed and loaded by the registry.
    #[default]
    Enabled,
    /// Installed, switched off in `plugins.toml`.
    Disabled,
    /// Written against a frontier-dataset version this framework cannot serve.
    Incompatible,
    /// `plugin.toml` is missing or does not validate.
    Invalid,
    /// Known to the catalog (or a wheel directory) but not installed.
    NotInstalled,
}

/// An optional, lazily imported dependency of a plugin (an SDK, an alternative reader).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OptionalDependency {
    pub package: String,
    pub licence: String,
    pub purpose: String,
    pub module: String,
    /// Copyleft / proprietary / needs-a-vendor-licence note.
    pub flag: String,
}

/// One row of `plugins list --json` / `plugins search --json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginInfo {
    pub id: String,
    pub kind: PluginKind,
    pub name: String,
    pub summary: String,
    pub version: String,
    /// Python distribution name (`frontier-connector-fluent`).
    pub dist: String,
    pub status: PluginStatus,
    pub enabled: bool,
    pub installed: bool,
    /// Why the plugin is `invalid` / `incompatible`; notes otherwise.
    pub problems: Vec<String>,
    pub data_model_version: String,
    pub vendor: String,
    pub formats: Vec<String>,
    pub maturity: String,
    pub licence: String,
    /// pip requirements installed together with the plugin.
    pub requires: Vec<String>,
    /// Other plugins (by id) this one delegates to.
    pub plugins: Vec<String>,
    /// Plugins installed alongside by default (vendor-side exporter scripts).
    pub suggests: Vec<String>,
    pub optional: Vec<OptionalDependency>,
    /// Connector ids provided.
    pub connectors: Vec<String>,
    /// Importer ids provided.
    pub importers: Vec<String>,
    /// Exporter vendor keys provided.
    pub exporters: Vec<String>,
    /// Members switched off individually while the plugin itself stays enabled.
    pub disabled_connectors: Vec<String>,
    pub disabled_importers: Vec<String>,
    pub disabled_exporters: Vec<String>,
    pub first_party: bool,
    pub has_manifest: bool,
    pub location: String,
}

/// `plugins list --json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginList {
    /// frontier-dataset version of the installed framework, e.g. `1.1`.
    pub data_model_version: String,
    /// Path of the state file holding the enable/disable switches.
    pub state_file: PathBuf,
    pub plugins: Vec<PluginInfo>,
}

impl PluginList {
    pub fn get(&self, id: &str) -> Option<&PluginInfo> {
        self.plugins.iter().find(|p| p.id == id)
    }

    /// The plugin that provides connector `connector_id`, installed or not.
    pub fn provider_of(&self, connector_id: &str) -> Option<&PluginInfo> {
        self.plugins
            .iter()
            .find(|p| p.connectors.iter().any(|c| c == connector_id))
    }
}

/// What `enable` / `disable` act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginTarget {
    Plugin(String),
    Connector(String),
    Importer(String),
    Exporter(String),
}

impl PluginTarget {
    /// The spelling the Python CLI takes: `fluent`, `connector:fluent-live`, ...
    pub fn to_arg(&self) -> String {
        match self {
            Self::Plugin(id) => format!("plugin:{id}"),
            Self::Connector(id) => format!("connector:{id}"),
            Self::Importer(id) => format!("importer:{id}"),
            Self::Exporter(id) => format!("exporter:{id}"),
        }
    }
}

/// One entry of the result of `enable` / `disable`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EnableResult {
    /// `plugins` | `connectors` | `importers` | `exporters`
    pub scope: String,
    pub id: String,
    pub enabled: bool,
    /// `false` when the switch already had that position.
    pub changed: bool,
    pub state: PathBuf,
}

/// Where `install` looks for distributions.
#[derive(Debug, Clone, Default)]
pub struct InstallOptions {
    /// Directories of wheels (`--find-links`), e.g. an air-gapped bundle.
    pub find_links: Vec<PathBuf>,
    pub index_url: Option<String>,
    pub extra_index_url: Option<String>,
    /// Use only `find_links` (no network).
    pub no_index: bool,
    /// Skip the vendor-side exporter plugins a connector plugin suggests.
    pub no_suggested: bool,
    pub upgrade: bool,
    /// Print the installer command only.
    pub dry_run: bool,
}

/// Result of `install` / `uninstall`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InstallReport {
    /// The `pip` / `uv pip` command that ran (or would run).
    pub command: Vec<String>,
    pub returncode: Option<i32>,
    pub output: String,
    pub dry_run: bool,
    /// State of the plugins after an install.
    pub plugins: Vec<PluginInfo>,
    /// Ids removed by an uninstall.
    pub uninstalled: Vec<String>,
}

/// Health of one plugin (`plugins doctor`, `plugins info`): the [`PluginInfo`] fields plus
/// the checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginHealth {
    #[serde(flatten)]
    pub info: PluginInfo,
    pub ok: bool,
    pub requires_status: Vec<RequirementStatus>,
    pub plugins_status: Vec<PluginDependencyStatus>,
    pub optional_status: Vec<OptionalStatus>,
    pub loads: Vec<LoadCheck>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RequirementStatus {
    pub requirement: String,
    /// Installed version, `None` when missing.
    pub installed: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginDependencyStatus {
    pub plugin: String,
    pub status: String,
    /// Command that fixes it, `None` when nothing is wrong.
    pub fix: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OptionalStatus {
    pub package: String,
    pub installed: Option<String>,
    pub licence: String,
    pub purpose: String,
    pub flag: String,
    pub fix: Option<String>,
}

/// Result of importing one connector / importer, or of locating an exporter's scripts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadCheck {
    pub kind: String,
    pub id: String,
    pub ok: bool,
    pub error: Option<String>,
    pub disabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameworkInfo {
    pub dist: String,
    pub version: String,
    pub data_model_version: String,
    pub python: String,
    pub executable: String,
}

/// `plugins doctor --json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginDoctor {
    pub framework: FrameworkInfo,
    pub plugins: Vec<PluginHealth>,
    /// Copies of exporter scripts that are missing or older than the installed plugin.
    pub deployed_exporters: Vec<DeployedProblem>,
    pub ok: bool,
    pub frontier_home: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DeployedProblem {
    pub vendor: String,
    pub path: PathBuf,
    pub problem: String,
}

/// One vendor-side script bundle (`plugins exporters --json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExporterInfo {
    pub vendor: String,
    pub plugin: String,
    pub version: String,
    pub tool: String,
    pub usage: String,
    pub enabled: bool,
    pub path: PathBuf,
    pub files: Vec<String>,
}

/// Result of `install-exporter`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExporterInstall {
    pub vendor: String,
    pub plugin: String,
    pub version: String,
    pub path: PathBuf,
    pub files: Vec<String>,
    pub replaced_version: Option<String>,
    pub usage: String,
}

/// Result of `uninstall-exporter`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExporterUninstall {
    pub vendor: String,
    pub removed: Vec<PathBuf>,
    /// Files the user changed after the copy was made; they are left in place.
    pub kept_modified: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct Rows<T> {
    #[serde(alias = "plugins", alias = "results", alias = "exporters")]
    rows: Vec<T>,
}

fn s(text: &str) -> String {
    text.to_string()
}

fn path(p: &Path) -> String {
    std::path::absolute(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

impl PythonBridge {
    /// Installed plugins with their enable state; with `available` also the first-party
    /// plugins that are not installed (status `not-installed`).
    pub fn plugins(&self, available: bool) -> Result<PluginList> {
        let mut args = vec![s("plugins"), s("list")];
        if available {
            args.push(s("--available"));
        }
        self.typed(&args)
    }

    /// Search the catalog shipped with the framework and the wheels in `find_links`.
    pub fn search_plugins(&self, query: &str, find_links: &[PathBuf]) -> Result<Vec<PluginInfo>> {
        let mut args = vec![s("plugins"), s("search")];
        args.extend(query.split_whitespace().map(s));
        for dir in find_links {
            args.extend([s("--find-links"), path(dir)]);
        }
        Ok(self.typed::<Rows<PluginInfo>>(&args)?.rows)
    }

    /// Manifest and health of one plugin (installed or known to the catalog).
    pub fn plugin_info(&self, id: &str) -> Result<PluginHealth> {
        self.typed(&[s("plugins"), s("info"), s(id)])
    }

    /// Switch plugins (or single connectors / importers / exporters) on or off. Nothing is
    /// installed or removed; the switches live in `plugins.toml`.
    pub fn set_plugin_enabled(
        &self,
        targets: &[PluginTarget],
        enabled: bool,
    ) -> Result<Vec<EnableResult>> {
        let mut args = vec![s("plugins"), s(if enabled { "enable" } else { "disable" })];
        args.extend(targets.iter().map(PluginTarget::to_arg));
        Ok(self.typed::<Rows<EnableResult>>(&args)?.rows)
    }

    /// Install plugins by id, distribution name, path, wheel or `git+https://...` URL.
    pub fn install_plugins(
        &self,
        specs: &[String],
        opts: &InstallOptions,
    ) -> Result<InstallReport> {
        let mut args = vec![s("plugins"), s("install")];
        args.extend(specs.iter().cloned());
        for dir in &opts.find_links {
            args.extend([s("--find-links"), path(dir)]);
        }
        if let Some(url) = &opts.index_url {
            args.extend([s("--index-url"), url.clone()]);
        }
        if let Some(url) = &opts.extra_index_url {
            args.extend([s("--extra-index-url"), url.clone()]);
        }
        for (flag, on) in [
            ("--no-index", opts.no_index),
            ("--no-suggested", opts.no_suggested),
            ("--upgrade", opts.upgrade),
            ("--dry-run", opts.dry_run),
        ] {
            if on {
                args.push(s(flag));
            }
        }
        self.typed(&args)
    }

    /// Uninstall plugins by id. The framework itself is refused by the Python side.
    pub fn uninstall_plugins(&self, ids: &[String], dry_run: bool) -> Result<InstallReport> {
        let mut args = vec![s("plugins"), s("uninstall")];
        args.extend(ids.iter().cloned());
        if dry_run {
            args.push(s("--dry-run"));
        }
        self.typed(&args)
    }

    /// Per-plugin health: manifest, compatibility, missing requirements, optional packages
    /// with their licences, import errors. `ok == false` is a normal answer, not an error.
    pub fn plugin_doctor(&self) -> Result<PluginDoctor> {
        self.typed(&[s("plugins"), s("doctor")])
    }

    /// Installed vendor-side script bundles.
    pub fn exporters(&self) -> Result<Vec<ExporterInfo>> {
        Ok(self
            .typed::<Rows<ExporterInfo>>(&[s("plugins"), s("exporters")])?
            .rows)
    }

    /// Copy the scripts of `vendor` to `<to>/<vendor>/` with a version stamp.
    pub fn install_exporter(
        &self,
        vendor: &str,
        to: &Path,
        force: bool,
    ) -> Result<ExporterInstall> {
        let mut args = vec![
            s("plugins"),
            s("install-exporter"),
            s(vendor),
            s("--to"),
            path(to),
        ];
        if force {
            args.push(s("--force"));
        }
        self.typed(&args)
    }

    /// Remove copies made by [`install_exporter`](Self::install_exporter): the one under
    /// `from`, or every recorded copy.
    pub fn uninstall_exporter(
        &self,
        vendor: &str,
        from: Option<&Path>,
        force: bool,
    ) -> Result<ExporterUninstall> {
        let mut args = vec![s("plugins"), s("uninstall-exporter"), s(vendor)];
        if let Some(dir) = from {
            args.extend([s("--from"), path(dir)]);
        }
        if force {
            args.push(s("--force"));
        }
        self.typed(&args)
    }
}
