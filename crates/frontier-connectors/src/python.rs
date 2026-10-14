// SPDX-License-Identifier: Apache-2.0
//! Subprocess bridge to the Python `frontier_connectors` package.
//!
//! Rust keeps detection fast and dependency-free; everything that needs a real reader
//! (`describe`, `convert`) runs `python -m frontier_connectors <command> --json` and parses
//! the single JSON document it prints. The command is resolved once:
//!
//! 1. `FRONTIER_CONNECTORS_CMD` (whitespace separated, e.g. `python3 -m frontier_connectors`
//!    or `/opt/frontier/venv/bin/python -m frontier_connectors`),
//! 2. `uv run python -m frontier_connectors` when `uv` is on `PATH` and a uv workspace
//!    containing `packages/frontier-connectors` is found above the current directory,
//! 3. `python3 -m frontier_connectors`.

use crate::{Detection, SampleMetadata};
use frontier_core::{Error, Result};
use frontier_data::{FieldDescriptor, FieldLocation, MeshDescriptor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Environment variable overriding the command used to reach the Python package.
pub const CMD_ENV: &str = "FRONTIER_CONNECTORS_CMD";
/// Environment variable with the subprocess timeout in seconds (default 3600).
pub const TIMEOUT_ENV: &str = "FRONTIER_CONNECTORS_TIMEOUT_SECS";

/// One zone (cell zone, boundary patch, part, block) of a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ZoneInfo {
    pub name: String,
    /// `surface` | `mesh` | `grid` | `point_cloud`
    pub kind: String,
    pub role: String,
    pub n_points: Option<u64>,
    pub n_cells: Option<u64>,
    pub dim: u8,
    pub cell_types: BTreeMap<String, u64>,
    pub shape: Option<Vec<u64>>,
    /// `[xmin, xmax, ymin, ymax, zmin, zmax]` (the Python/VTK order).
    pub bounds: Option<Vec<f64>>,
    pub bc_type: Option<String>,
    #[serde(rename = "default", default = "yes")]
    pub read_by_default: bool,
}

fn yes() -> bool {
    true
}

/// One field of a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldInfo {
    pub name: String,
    /// `point` | `cell` | `grid` | `global`
    pub location: String,
    pub components: u32,
    pub units: String,
    pub component_names: Option<Vec<String>>,
    pub dtype: String,
    pub description: String,
    pub time_dependent: bool,
    pub zones: Option<Vec<String>>,
}

/// Time axis of a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TimeInfo {
    pub values: Vec<f64>,
    pub n_steps: u64,
    pub dt: Option<f64>,
    pub units: String,
    pub uniform: bool,
    pub quantity: String,
}

/// `python -m frontier_connectors describe --json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceDescription {
    pub connector_id: String,
    pub path: PathBuf,
    pub format: String,
    pub solver: String,
    pub solver_version: Option<String>,
    pub domain: String,
    pub temporal: String,
    pub physics: BTreeMap<String, serde_json::Value>,
    pub zones: Vec<ZoneInfo>,
    pub fields: Vec<FieldInfo>,
    pub time: TimeInfo,
    pub boundary_conditions: BTreeMap<String, serde_json::Value>,
    pub units: BTreeMap<String, serde_json::Value>,
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub warnings: Vec<String>,
}

impl SourceDescription {
    /// Collapse the description into the solver-independent [`SampleMetadata`] the rest of
    /// the Rust code base uses. Counts and bounds cover the zones read by default; the full
    /// zone table, physics and boundary conditions are kept under `attributes`.
    pub fn into_sample_metadata(self) -> SampleMetadata {
        let default_zones: Vec<&ZoneInfo> = {
            let d: Vec<&ZoneInfo> = self.zones.iter().filter(|z| z.read_by_default).collect();
            if d.is_empty() {
                self.zones.iter().collect()
            } else {
                d
            }
        };
        let n_points = default_zones.iter().filter_map(|z| z.n_points).sum();
        let n_cells = default_zones.iter().filter_map(|z| z.n_cells).sum();
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut any = false;
        for z in &default_zones {
            if let Some(b) = &z.bounds
                && b.len() == 6
            {
                any = true;
                for axis in 0..3 {
                    bounds[axis] = bounds[axis].min(b[2 * axis]);
                    bounds[axis + 3] = bounds[axis + 3].max(b[2 * axis + 1]);
                }
            }
        }
        if !any {
            bounds = [0.0; 6];
        }
        let dimension = default_zones
            .iter()
            .map(|z| z.dim)
            .max()
            .filter(|d| *d > 0)
            .unwrap_or(3);
        let fields = self
            .fields
            .iter()
            .filter(|f| f.location != "global")
            .map(|f| FieldDescriptor {
                name: f.name.clone(),
                location: if f.location == "cell" {
                    FieldLocation::Cell
                } else {
                    FieldLocation::Point
                },
                components: u8::try_from(f.components).unwrap_or(u8::MAX),
                unit: f.units.clone(),
                time: None,
            })
            .collect();
        let mut attributes: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut put = |k: &str, v: serde_json::Value| {
            if !v.is_null() && v != serde_json::json!({}) && v != serde_json::json!([]) {
                attributes.insert(k.to_string(), v);
            }
        };
        put("connector", serde_json::json!(self.connector_id));
        put("format", serde_json::json!(self.format));
        put("domain", serde_json::json!(self.domain));
        put("temporal", serde_json::json!(self.temporal));
        put("time_units", serde_json::json!(self.time.units));
        put("time_quantity", serde_json::json!(self.time.quantity));
        put("physics", serde_json::json!(self.physics));
        put("zones", serde_json::json!(self.zones));
        put(
            "boundary_conditions",
            serde_json::json!(self.boundary_conditions),
        );
        put("units", serde_json::json!(self.units));
        put("parameters", serde_json::json!(self.parameters));
        put("metadata", serde_json::json!(self.metadata));
        put("warnings", serde_json::json!(self.warnings));
        SampleMetadata {
            solver: self.solver,
            solver_version: self.solver_version,
            mesh: MeshDescriptor {
                n_points,
                n_cells,
                n_faces: None,
                dimension,
                bounds,
            },
            fields,
            times: if self.time.values.len() > 1 {
                self.time.values
            } else {
                Vec::new()
            },
            attributes,
        }
    }
}

/// One row of `python -m frontier_connectors list --json`: a connector of an installed,
/// enabled plugin. Connectors of disabled or uninstalled plugins are not listed here; see
/// [`PythonBridge::plugins`](crate::plugins).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ConnectorInfo {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub formats: Vec<String>,
    pub extensions: Vec<String>,
    pub license: String,
    /// `verified-public-sample` | `verified-synthetic` | `exporter+reader` |
    /// `implemented-unverified-live` | `planned`
    pub maturity: String,
    pub temporal: Vec<String>,
    pub kinds: Vec<String>,
    pub requires: Vec<String>,
    pub missing: Vec<String>,
    /// `available` | `missing-deps` | `planned` | `broken`
    pub status: String,
    /// `plugin:<plugin id>` or `runtime`
    pub source: String,
    /// Id of the plugin that provides the connector (see [`crate::plugins`]).
    pub plugin: String,
    pub export_help: String,
    pub note: String,
}

/// Options of `convert`.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    pub connector: Option<String>,
    pub zones: Vec<String>,
    pub fields: Vec<String>,
    /// CLI time selection: `all`, `last`, `0:100:10`, `t=0.1,0.2`, ...
    pub times: Option<String>,
    pub name: Option<String>,
    pub domain: Option<String>,
    pub mapping: Option<PathBuf>,
    pub overwrite: bool,
    pub time_as_samples: bool,
}

/// Result of `convert`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvertReport {
    pub out: PathBuf,
    pub name: String,
    pub n_samples: u64,
    pub sample_ids: Vec<String>,
    pub backends: BTreeMap<String, String>,
    pub n_steps: BTreeMap<String, u64>,
    pub warnings: Vec<String>,
    pub seconds: f64,
}

/// How to reach `python -m frontier_connectors`.
#[derive(Debug, Clone)]
pub struct PythonBridge {
    program: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
    timeout: Duration,
    env: Vec<(String, String)>,
}

impl PythonBridge {
    /// Explicit command, e.g. `PythonBridge::new("python3", ["-m", "frontier_connectors"])`.
    pub fn new<I, S>(program: impl Into<String>, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            cwd: None,
            timeout: Duration::from_secs(3600),
            env: Vec::new(),
        }
    }

    /// Set an environment variable for the subprocess (e.g. `FRONTIER_HOME` to point the
    /// plugin state at another directory).
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Run the subprocess from `dir` (needed for `uv run` to find the workspace).
    pub fn in_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Resolve the command from the environment (see the module docs).
    pub fn from_env() -> Self {
        let timeout = std::env::var(TIMEOUT_ENV)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(3600));
        if let Ok(cmd) = std::env::var(CMD_ENV) {
            let mut parts = cmd.split_whitespace().map(str::to_owned);
            if let Some(program) = parts.next() {
                return Self::new(program, parts).with_timeout(timeout);
            }
        }
        if on_path("uv")
            && let Some(root) = std::env::current_dir()
                .ok()
                .and_then(|d| workspace_root(&d))
        {
            return Self::new("uv", ["run", "python", "-m", "frontier_connectors"])
                .in_dir(root)
                .with_timeout(timeout);
        }
        Self::new("python3", ["-m", "frontier_connectors"]).with_timeout(timeout)
    }

    /// The command line, for logs and error messages.
    pub fn command_line(&self) -> String {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Run `<command> <args...> --json --compact` and parse stdout as JSON. A document of
    /// the form `{"error", "message", "hint"}` (exit code 2) becomes [`Error::Invalid`].
    pub fn run_json(&self, args: &[String]) -> Result<serde_json::Value> {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .args(args)
            .arg("--json")
            .arg("--compact")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = &self.cwd {
            cmd.current_dir(dir);
        }
        cmd.envs(self.env.iter().map(|(k, v)| (k, v)));
        let mut child = cmd.spawn().map_err(|e| {
            Error::Unsupported(format!(
                "cannot start `{}`: {e}. Install the frontier-connectors Python package \
                 (uv sync) or set {CMD_ENV}",
                self.command_line()
            ))
        })?;
        // drain the pipes on threads so a chatty child cannot block on a full pipe
        let mut stdout = child.stdout.take();
        let mut stderr = child.stderr.take();
        let out_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(s) = stdout.as_mut() {
                let _ = s.read_to_end(&mut buf);
            }
            buf
        });
        let err_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(s) = stderr.as_mut() {
                let _ = s.read_to_end(&mut buf);
            }
            buf
        });
        let started = Instant::now();
        let status = loop {
            match child.try_wait()? {
                Some(status) => break status,
                None if started.elapsed() > self.timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::Other(format!(
                        "`{}` timed out after {}s (raise {TIMEOUT_ENV})",
                        self.command_line(),
                        self.timeout.as_secs()
                    )));
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        let stdout = out_thread.join().unwrap_or_default();
        let stderr = err_thread.join().unwrap_or_default();
        let text = String::from_utf8_lossy(&stdout);
        // the JSON document is the last non-empty line (uv may print progress before it)
        let doc = text
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        match serde_json::from_str::<serde_json::Value>(doc) {
            Ok(value) => {
                if let Some(message) = value.get("message").and_then(|m| m.as_str())
                    && value.get("error").is_some()
                {
                    let hint = value.get("hint").and_then(|h| h.as_str());
                    return Err(Error::Invalid(match hint {
                        Some(h) => format!("{message} (hint: {h})"),
                        None => message.to_string(),
                    }));
                }
                Ok(value)
            }
            Err(_) => {
                let tail: String = String::from_utf8_lossy(&stderr)
                    .lines()
                    .rev()
                    .take(6)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                Err(Error::Other(format!(
                    "`{}` exited with {status} without a JSON document: {tail}",
                    self.command_line()
                )))
            }
        }
    }

    pub(crate) fn typed<T: for<'de> Deserialize<'de>>(&self, args: &[String]) -> Result<T> {
        Ok(serde_json::from_value(self.run_json(args)?)?)
    }

    /// The Python registry table.
    pub fn list(&self) -> Result<Vec<ConnectorInfo>> {
        self.typed(&["list".to_string()])
    }

    /// Full (content-aware) detection by the Python connectors.
    pub fn detect(&self, path: &Path) -> Result<Vec<Detection>> {
        self.typed(&["detect".to_string(), path_arg(path)])
    }

    /// Zones, fields, time steps and solver metadata of `path`.
    pub fn describe(&self, path: &Path, connector: Option<&str>) -> Result<SourceDescription> {
        let mut args = vec!["describe".to_string(), path_arg(path)];
        if let Some(c) = connector {
            args.extend(["--connector".to_string(), c.to_string()]);
        }
        self.typed(&args)
    }

    /// Convert `path` into a `frontier-dataset v1.1` directory at `out`.
    pub fn convert(&self, path: &Path, out: &Path, opts: &ConvertOptions) -> Result<ConvertReport> {
        let mut args = vec![
            "convert".to_string(),
            path_arg(path),
            "--out".to_string(),
            path_arg(out),
        ];
        let mut opt = |flag: &str, value: Option<String>| {
            if let Some(v) = value {
                args.extend([flag.to_string(), v]);
            }
        };
        opt("--connector", opts.connector.clone());
        opt(
            "--zones",
            (!opts.zones.is_empty()).then(|| opts.zones.join(",")),
        );
        opt(
            "--fields",
            (!opts.fields.is_empty()).then(|| opts.fields.join(",")),
        );
        opt("--times", opts.times.clone());
        opt("--name", opts.name.clone());
        opt("--domain", opts.domain.clone());
        opt("--mapping", opts.mapping.as_deref().map(path_arg));
        if opts.overwrite {
            args.push("--overwrite".to_string());
        }
        if opts.time_as_samples {
            args.push("--time-as-samples".to_string());
        }
        self.typed(&args)
    }
}

/// Absolute path as a string, so the subprocess may run from another directory.
fn path_arg(path: &Path) -> String {
    std::path::absolute(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths)
            .any(|dir| dir.join(program).is_file() || dir.join(format!("{program}.exe")).is_file())
    })
}

/// Nearest ancestor of `start` that is the Frontier uv workspace.
pub fn workspace_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|d| {
            d.join("pyproject.toml").is_file()
                && d.join("packages/frontier-connectors/pyproject.toml")
                    .is_file()
        })
        .map(Path::to_path_buf)
}
