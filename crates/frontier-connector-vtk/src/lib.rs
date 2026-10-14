// SPDX-License-Identifier: Apache-2.0
//! Legacy VTK (`.vtk`) connector built on the pure-Rust `vtkio` crate.
//!
//! XML VTK files (`.vtu`, `.vtp`, `.vts`, `.vtr`, `.vti`) are deliberately not handled here:
//! `vtkio`'s XML and compression features depend on unmaintained parser versions with open
//! advisories, and these files come from outside. The Python connector
//! (`frontier-connector-vtk`, on the VTK library) reads every VTK format.

use frontier_connectors::{Connector, Detection, SampleMetadata, SampleReader};
use frontier_core::{DATA_MODEL_VERSION, Error, Plugin, PluginKind, PluginManifest, Result};
use frontier_data::{FieldDescriptor, FieldLocation, MeshDescriptor};
use std::path::Path;
use vtkio::Vtk;
use vtkio::model::{Attribute, DataSet, ElementType};

pub struct VtkConnector {
    manifest: PluginManifest,
}

impl Default for VtkConnector {
    fn default() -> Self {
        Self {
            manifest: PluginManifest {
                id: "connector-vtk".into(),
                name: "VTK".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                kind: PluginKind::Connector,
                data_model_version: DATA_MODEL_VERSION,
                description: "Describes legacy VTK datasets (.vtk)".into(),
            },
        }
    }
}

impl Plugin for VtkConnector {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }
}

const EXTENSIONS: &[&str] = &["vtk"];

impl Connector for VtkConnector {
    fn detect(&self, path: &Path) -> Option<Detection> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        if !EXTENSIONS.contains(&ext.as_str()) || !path.is_file() {
            return None;
        }
        Some(Detection {
            connector_id: self.manifest.id.clone(),
            root: path.to_path_buf(),
            confidence: 0.9,
            notes: vec![format!("extension .{ext}")],
        })
    }

    fn open(&self, path: &Path) -> Result<Box<dyn SampleReader>> {
        let vtk = Vtk::import(path).map_err(|e| Error::Invalid(format!("vtk: {e:?}")))?;
        let metadata = describe(&vtk);
        Ok(Box::new(VtkSample { metadata }))
    }
}

fn describe(vtk: &Vtk) -> SampleMetadata {
    let mut fields = Vec::new();
    let mut n_points = 0u64;
    let mut n_cells = 0u64;

    let mut push_attrs = |attrs: &[Attribute], location: FieldLocation| {
        for a in attrs {
            if let Attribute::DataArray(da) = a {
                let components = match da.elem {
                    ElementType::Scalars { num_comp, .. } => num_comp as u8,
                    ElementType::Vectors | ElementType::Normals => 3,
                    ElementType::Tensors => 9,
                    ElementType::TCoords(n) => n as u8,
                    ElementType::Generic(n) => n as u8,
                    ElementType::ColorScalars(n) => n as u8,
                    ElementType::LookupTable => 1,
                };
                fields.push(FieldDescriptor {
                    name: da.name.clone(),
                    location,
                    components,
                    unit: String::new(),
                    time: None,
                });
            }
        }
    };

    match &vtk.data {
        DataSet::UnstructuredGrid { pieces, .. } => {
            for piece in pieces {
                if let vtkio::model::Piece::Inline(p) = piece {
                    n_points += p.points.len() as u64 / 3;
                    n_cells += p.cells.cell_verts.num_cells() as u64;
                    push_attrs(&p.data.point, FieldLocation::Point);
                    push_attrs(&p.data.cell, FieldLocation::Cell);
                }
            }
        }
        DataSet::PolyData { pieces, .. } => {
            for piece in pieces {
                if let vtkio::model::Piece::Inline(p) = piece {
                    n_points += p.points.len() as u64 / 3;
                    push_attrs(&p.data.point, FieldLocation::Point);
                    push_attrs(&p.data.cell, FieldLocation::Cell);
                }
            }
        }
        _ => {}
    }

    SampleMetadata {
        solver: "vtk".into(),
        solver_version: Some(format!("{}.{}", vtk.version.major, vtk.version.minor)),
        mesh: MeshDescriptor {
            n_points,
            n_cells,
            n_faces: None,
            dimension: 3,
            bounds: [0.0; 6],
        },
        fields,
        times: Vec::new(),
        attributes: Default::default(),
    }
}

struct VtkSample {
    metadata: SampleMetadata,
}

impl SampleReader for VtkSample {
    fn metadata(&self) -> &SampleMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY: &str = "# vtk DataFile Version 3.0
tet
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0 1 0 0 0 1 0 0 0 1
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
10
POINT_DATA 4
SCALARS p float 1
LOOKUP_TABLE default
1 2 3 4
VECTORS U float
1 0 0 1 0 0 1 0 0 1 0 0
";

    #[test]
    fn describes_a_legacy_unstructured_grid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tet.vtk");
        std::fs::write(&path, LEGACY).unwrap();
        let c = VtkConnector::default();
        let d = c.detect(&path).expect("detected by extension");
        assert_eq!(d.connector_id, "connector-vtk");
        let sample = c.open(&path).unwrap();
        let m = sample.metadata();
        assert_eq!((m.mesh.n_points, m.mesh.n_cells), (4, 1));
        let fields: Vec<(&str, u8)> = m
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.components))
            .collect();
        assert_eq!(fields, vec![("p", 1), ("U", 3)]);
        assert!(m.fields.iter().all(|f| f.location == FieldLocation::Point));
    }

    #[test]
    fn rejects_other_extensions_and_reports_broken_files() {
        let dir = tempfile::tempdir().unwrap();
        let c = VtkConnector::default();
        let other = dir.path().join("mesh.cgns");
        std::fs::write(&other, "x").unwrap();
        assert!(c.detect(&other).is_none());
        assert!(c.detect(&dir.path().join("missing.vtu")).is_none());
        let broken = dir.path().join("broken.vtk");
        std::fs::write(&broken, "not a vtk file").unwrap();
        assert!(c.detect(&broken).is_some());
        assert!(c.open(&broken).is_err());
    }
}
