// SPDX-License-Identifier: Apache-2.0
//! OpenFOAM case-directory connector.
//!
//! Detection is based on the canonical case layout (`system/controlDict`,
//! `constant/polyMesh`, time directories). Field parsing (ASCII/binary FoamFile
//! format, `decomposePar` processor directories, `.foam` stubs) will be implemented
//! incrementally; the first milestone is correct detection and metadata extraction.

use frontier_connectors::{Connector, Detection, SampleMetadata, SampleReader};
use frontier_core::{DATA_MODEL_VERSION, Plugin, PluginKind, PluginManifest, Result};
use frontier_data::MeshDescriptor;
use std::path::Path;

pub struct OpenFoamConnector {
    manifest: PluginManifest,
}

impl Default for OpenFoamConnector {
    fn default() -> Self {
        Self {
            manifest: PluginManifest {
                id: "connector-openfoam".into(),
                name: "OpenFOAM".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                kind: PluginKind::Connector,
                data_model_version: DATA_MODEL_VERSION,
                description: "Imports OpenFOAM cases (ESI and Foundation variants)".into(),
            },
        }
    }
}

impl Plugin for OpenFoamConnector {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }
}

/// Parse the numeric time directories of a case root (`0`, `0.5`, `1000`, ...).
pub fn time_directories(root: &Path) -> std::io::Result<Vec<f64>> {
    let mut times = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str()
            && let Ok(t) = name.parse::<f64>()
        {
            times.push(t);
        }
    }
    times.sort_by(f64::total_cmp);
    Ok(times)
}

/// Resolve a path to the case root: the directory itself, the parent of a `.foam` stub or
/// the grandparent of `system/controlDict`.
pub fn case_root(path: &Path) -> Option<&Path> {
    if path.is_dir() {
        return Some(path);
    }
    let name = path.file_name()?.to_str()?;
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".foam") || lower.ends_with(".openfoam") {
        return path.parent();
    }
    if name == "controlDict" {
        return path.parent()?.parent();
    }
    None
}

/// Value of a top-level `key value;` entry of an OpenFOAM dictionary (comments ignored).
pub fn dict_entry(text: &str, key: &str) -> Option<String> {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or("").trim())
        .find_map(|l| {
            let rest = l.strip_prefix(key)?;
            if !rest.starts_with(char::is_whitespace) {
                return None;
            }
            Some(
                rest.trim()
                    .trim_end_matches(';')
                    .trim()
                    .trim_matches('"')
                    .to_string(),
            )
        })
        .filter(|v| !v.is_empty())
}

/// `(name, type)` of every patch in a `polyMesh/boundary` file (ASCII part only).
pub fn boundary_patches(text: &str) -> Vec<(String, String)> {
    let mut patches = Vec::new();
    let mut depth = 0usize;
    let mut pending: Option<String> = None;
    let mut current: Option<String> = None;
    let body = text.split_once('(').map(|(_, b)| b).unwrap_or("");
    for raw in body.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "{" => {
                depth += 1;
                if depth == 1 {
                    current = pending.take();
                }
            }
            "}" => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    current = None;
                }
            }
            ")" if depth == 0 => break,
            _ if depth == 0 => pending = Some(line.to_string()),
            _ if depth == 1 => {
                if let (Some(name), Some(kind)) = (&current, line.strip_prefix("type")) {
                    let kind = kind.trim().trim_end_matches(';').trim();
                    if !kind.is_empty() && !patches.iter().any(|(n, _)| n == name) {
                        patches.push((name.clone(), kind.to_string()));
                    }
                }
            }
            _ => {}
        }
    }
    patches
}

/// Number of `processor<N>` directories of a decomposed case.
fn processor_count(root: &Path) -> usize {
    std::fs::read_dir(root)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .and_then(|n| n.strip_prefix("processor"))
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                })
                .count()
        })
        .unwrap_or(0)
}

/// Field files (`U`, `p`, `nut.gz`, ...) of one time directory.
fn field_names(time_dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(time_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .filter_map(|e| e.file_name().to_str().map(str::to_owned))
                .map(|n| n.trim_end_matches(".gz").to_string())
                .filter(|n| !n.starts_with('.') && !n.ends_with(".orig"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names.dedup();
    names
}

/// Component count from the `class` of a field file header (0 when it cannot be read, e.g.
/// for gzip-compressed fields).
fn field_components(file: &Path) -> u8 {
    use std::io::Read;
    let mut head = Vec::new();
    if let Ok(f) = std::fs::File::open(file) {
        let _ = f.take(2048).read_to_end(&mut head);
    }
    match dict_entry(&String::from_utf8_lossy(&head), "class").as_deref() {
        Some(c) if c.ends_with("ScalarField") => 1,
        Some(c) if c.ends_with("SymmTensorField") => 6,
        Some(c) if c.ends_with("TensorField") => 9,
        Some(c) if c.ends_with("VectorField") => 3,
        _ => 0,
    }
}

impl Connector for OpenFoamConnector {
    fn detect(&self, path: &Path) -> Option<Detection> {
        let root = case_root(path)?;
        let has_control_dict = root.join("system/controlDict").is_file();
        let processors = processor_count(root);
        let has_mesh = root.join("constant/polyMesh").is_dir()
            || root.join("processor0/constant/polyMesh").is_dir();
        if !has_control_dict && !has_mesh {
            return None;
        }
        let mut confidence = 0.0;
        let mut notes = Vec::new();
        if has_control_dict {
            confidence += 0.6;
            notes.push("system/controlDict present".into());
        }
        if has_mesh {
            confidence += 0.4;
            notes.push("constant/polyMesh present".into());
        }
        if processors > 0 {
            notes.push(format!(
                "decomposed into {processors} processor directories"
            ));
        }
        Some(Detection {
            connector_id: self.manifest.id.clone(),
            root: root.to_path_buf(),
            confidence,
            notes,
        })
    }

    /// Cheap native description: time directories, application, patch table and the field
    /// names of the latest time. Mesh sizes and field data come from the Python reader
    /// (`frontier_connectors::ConnectorRegistry::describe`).
    fn open(&self, path: &Path) -> Result<Box<dyn SampleReader>> {
        let root = case_root(path).ok_or_else(|| {
            frontier_core::Error::Unsupported(format!("{} is not an OpenFOAM case", path.display()))
        })?;
        let processors = processor_count(root);
        let data_root = if root.join("constant/polyMesh").is_dir() || processors == 0 {
            root.to_path_buf()
        } else {
            root.join("processor0")
        };
        let times = time_directories(&data_root)?;
        let mut attributes = std::collections::BTreeMap::new();
        if let Ok(text) = std::fs::read_to_string(root.join("system/controlDict")) {
            for key in ["application", "deltaT", "endTime", "writeFormat"] {
                if let Some(v) = dict_entry(&text, key) {
                    attributes.insert(key.to_string(), serde_json::Value::String(v));
                }
            }
        }
        if let Ok(text) = std::fs::read_to_string(data_root.join("constant/polyMesh/boundary")) {
            let patches: serde_json::Map<String, serde_json::Value> = boundary_patches(&text)
                .into_iter()
                .map(|(name, kind)| (name, serde_json::Value::String(kind)))
                .collect();
            if !patches.is_empty() {
                attributes.insert("patches".into(), serde_json::Value::Object(patches));
            }
        }
        if processors > 0 {
            attributes.insert("processors".into(), serde_json::json!(processors));
        }
        let fields = std::fs::read_dir(&data_root)
            .ok()
            .and_then(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        let t = e.file_name().to_str()?.parse::<f64>().ok()?;
                        Some((t, e.path()))
                    })
                    .max_by(|a, b| a.0.total_cmp(&b.0))
            })
            .map(|(t, dir)| {
                field_names(&dir)
                    .into_iter()
                    .map(|name| frontier_data::FieldDescriptor {
                        components: field_components(&dir.join(&name)),
                        name,
                        location: frontier_data::FieldLocation::Cell,
                        unit: String::new(),
                        time: Some(t),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let metadata = SampleMetadata {
            solver: "openfoam".into(),
            solver_version: None,
            mesh: MeshDescriptor {
                n_points: 0,
                n_cells: 0,
                n_faces: None,
                dimension: 3,
                bounds: [0.0; 6],
            },
            fields,
            times,
            attributes,
        };
        Ok(Box::new(OpenFoamSample { metadata }))
    }
}

struct OpenFoamSample {
    metadata: SampleMetadata,
}

impl SampleReader for OpenFoamSample {
    fn metadata(&self) -> &SampleMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_case_layout_and_times() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("system")).unwrap();
        std::fs::write(root.join("system/controlDict"), "FoamFile {}").unwrap();
        std::fs::create_dir_all(root.join("constant/polyMesh")).unwrap();
        for t in ["0", "0.5", "100"] {
            std::fs::create_dir(root.join(t)).unwrap();
        }

        let c = OpenFoamConnector::default();
        let d = c.detect(root).expect("detected");
        assert!((d.confidence - 1.0).abs() < f32::EPSILON);
        let sample = c.open(root).unwrap();
        assert_eq!(sample.metadata().times, vec![0.0, 0.5, 100.0]);
    }

    #[test]
    fn describes_patches_fields_and_decomposition() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("system")).unwrap();
        std::fs::write(
            root.join("system/controlDict"),
            "FoamFile\n{\n    class dictionary;\n}\napplication     simpleFoam; // steady\ndeltaT 1;\n",
        )
        .unwrap();
        for p in ["processor0", "processor1"] {
            std::fs::create_dir_all(root.join(p).join("constant/polyMesh")).unwrap();
            std::fs::create_dir_all(root.join(p).join("100")).unwrap();
            std::fs::create_dir_all(root.join(p).join("0")).unwrap();
        }
        std::fs::write(
            root.join("processor0/constant/polyMesh/boundary"),
            "FoamFile\n{\n    object boundary;\n}\n2\n(\n    inlet\n    {\n        type            patch;\n        nFaces 4;\n    }\n    walls\n    {\n        type wall;\n        inGroups 1(wall);\n    }\n)\n",
        )
        .unwrap();
        std::fs::write(
            root.join("processor0/100/U"),
            "FoamFile\n{\n    version 2.0;\n    class       volVectorField;\n    object U;\n}\n",
        )
        .unwrap();
        std::fs::write(root.join("processor0/100/p.gz"), "").unwrap();
        std::fs::write(root.join("case.foam"), "").unwrap();

        let c = OpenFoamConnector::default();
        let d = c
            .detect(&root.join("case.foam"))
            .expect("detected through the .foam stub");
        assert_eq!(d.root, root);
        assert!(d.notes.iter().any(|n| n.contains("2 processor")));
        let sample = c.open(&root.join("case.foam")).unwrap();
        let m = sample.metadata();
        assert_eq!(m.times, vec![0.0, 100.0]);
        assert_eq!(m.attributes["application"], "simpleFoam");
        assert_eq!(m.attributes["patches"]["inlet"], "patch");
        assert_eq!(m.attributes["patches"]["walls"], "wall");
        assert_eq!(m.attributes["processors"], 2);
        let names: Vec<&str> = m.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["U", "p"]);
        assert_eq!(m.fields[0].time, Some(100.0));
        assert_eq!((m.fields[0].components, m.fields[1].components), (3, 0));
    }

    #[test]
    fn ignores_random_directories() {
        let dir = tempfile::tempdir().unwrap();
        assert!(OpenFoamConnector::default().detect(dir.path()).is_none());
    }
}
