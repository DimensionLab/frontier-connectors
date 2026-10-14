// SPDX-License-Identifier: Apache-2.0
//! Fast, dependency-free format detection for every connector of the Python
//! `frontier_connectors` package.
//!
//! Detection here never parses a file: it looks at extensions, a few magic bytes, the
//! first kilobytes of text and the directory layout. It answers "which connector should
//! read this?" in microseconds so the CLI, the server and the app can route a path
//! without starting Python. Anything deeper (zones, fields, time steps) is delegated to
//! Python through [`crate::python::PythonBridge`].
//!
//! The table mirrors the connectors of the first-party Python plugins; the test
//! `catalog_matches_python_registry` keeps the two in sync. Which plugin provides a
//! connector (and whether it is installed and enabled) is answered by [`crate::plugins`].

use crate::Detection;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// How far a connector has been verified (same vocabulary as the Python registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// The native file is read directly.
    Direct,
    /// The native file is proprietary: a shipped exporter script must be run inside the
    /// vendor tool first; the connector reads what the exporter writes.
    NeedsExport,
    /// A cloud API: needs credentials in the environment.
    NeedsCredentials,
    /// Registry entry that documents an export route only.
    RouteOnly,
}

/// One row of the format catalogue.
#[derive(Debug, Clone, Copy)]
pub struct FormatSpec {
    /// Connector id in the Python registry (`python -m frontier_connectors list`).
    pub id: &'static str,
    pub name: &'static str,
    pub vendor: &'static str,
    /// Lower-case extensions including the dot; compound suffixes (`.cas.h5`) allowed.
    pub extensions: &'static [&'static str],
    pub access: Access,
}

/// Every connector the Python package registers, in the same order.
pub const FORMATS: &[FormatSpec] = &[
    FormatSpec {
        id: "openfoam",
        name: "OpenFOAM case",
        vendor: "OpenFOAM Foundation / ESI",
        extensions: &[".foam", ".openfoam"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "vtk",
        name: "VTK XML / legacy / VTKHDF / PVD",
        vendor: "Kitware",
        extensions: &[
            ".vtu", ".vtp", ".vts", ".vtr", ".vti", ".vtm", ".vtmb", ".vtpc", ".vtpd", ".vthb",
            ".pvtu", ".pvtp", ".pvts", ".pvtr", ".pvti", ".pvd", ".series", ".vtk", ".vtkhdf",
        ],
        access: Access::Direct,
    },
    FormatSpec {
        id: "ensight",
        name: "EnSight Gold",
        vendor: "Ansys (CEI)",
        extensions: &[".case", ".encas"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "cgns",
        name: "CGNS",
        vendor: "CGNS Steering Committee",
        extensions: &[".cgns"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "fluent-cff",
        name: "Ansys Fluent CFF",
        vendor: "Ansys",
        extensions: &[".cas.h5", ".dat.h5", ".msh.h5"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "fluent-legacy",
        name: "Ansys Fluent legacy case/data/mesh",
        vendor: "Ansys",
        extensions: &[".cas", ".dat", ".msh", ".cas.gz", ".dat.gz", ".msh.gz"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "fluent-live",
        name: "Ansys Fluent live session (PyFluent)",
        vendor: "Ansys",
        extensions: &[],
        access: Access::RouteOnly,
    },
    FormatSpec {
        id: "cfx",
        name: "Ansys CFX results",
        vendor: "Ansys",
        extensions: &[".res", ".def", ".trn", ".cfx"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "ansys-rst",
        name: "Ansys Mechanical / MAPDL results",
        vendor: "Ansys",
        extensions: &[".rst", ".rth", ".rmg"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "starccm",
        name: "Simcenter STAR-CCM+",
        vendor: "Siemens",
        extensions: &[".sim", ".ccm", ".ccmg", ".ccmp", ".ccmt"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "comsol",
        name: "COMSOL Multiphysics",
        vendor: "COMSOL",
        extensions: &[".mph"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "abaqus",
        name: "Abaqus",
        vendor: "Dassault Systemes",
        extensions: &[".odb", ".inp", ".frd", ".fil"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "lsdyna",
        name: "LS-DYNA d3plot / keyword",
        vendor: "Ansys (LST)",
        extensions: &[".d3plot", ".k", ".key", ".dyn", ".kw", ".k.gz", ".key.gz"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "nastran",
        name: "Nastran BDF / OP2",
        vendor: "MSC / Siemens / Altair",
        extensions: &[".bdf", ".nas", ".op2", ".fem"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "matlab",
        name: "MATLAB MAT-file",
        vendor: "MathWorks",
        extensions: &[".mat"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "octave",
        name: "GNU Octave save file (text, binary, HDF5)",
        vendor: "GNU Octave",
        extensions: &[],
        access: Access::Direct,
    },
    FormatSpec {
        id: "tecplot",
        name: "Tecplot",
        vendor: "Tecplot",
        extensions: &[".plt", ".szplt", ".tec", ".tp"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "exodus",
        name: "Exodus II",
        vendor: "Sandia",
        extensions: &[".e", ".exo", ".ex2", ".exii", ".g", ".gen"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "xdmf",
        name: "XDMF",
        vendor: "open format",
        extensions: &[".xdmf", ".xmf"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "gmsh",
        name: "Gmsh MSH",
        vendor: "Gmsh",
        extensions: &[".msh"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "med",
        name: "MED (Salome)",
        vendor: "EDF / CEA",
        extensions: &[".med", ".rmed", ".mmed"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "su2",
        name: "SU2 mesh",
        vendor: "SU2 Foundation",
        extensions: &[".su2"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "meshio",
        name: "Other meshio formats",
        vendor: "open formats",
        extensions: &[
            ".mesh", ".meshb", ".vol", ".ugrid", ".h5m", ".f3grid", ".node", ".ele", ".avs",
        ],
        access: Access::Direct,
    },
    FormatSpec {
        id: "netcdf",
        name: "NetCDF",
        vendor: "Unidata",
        extensions: &[".nc", ".nc4", ".cdf", ".netcdf"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "hdf5",
        name: "HDF5 (generic, mapping config)",
        vendor: "The HDF Group",
        extensions: &[".h5", ".hdf5", ".hdf", ".he5"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "zarr",
        name: "Zarr",
        vendor: "open format",
        extensions: &[".zarr"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "tabular",
        name: "CSV / Parquet point probes",
        vendor: "open formats",
        extensions: &[".csv", ".tsv", ".parquet", ".pq"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "geometry",
        name: "Geometry only (STL / OBJ / PLY / STEP)",
        vendor: "open formats",
        extensions: &[
            ".stl", ".obj", ".ply", ".off", ".step", ".stp", ".iges", ".igs", ".brep", ".brp",
        ],
        access: Access::Direct,
    },
    FormatSpec {
        id: "nek5000",
        name: "Nek5000 / NekRS field files",
        vendor: "Argonne",
        extensions: &[".nek5000"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "pyfr",
        name: "PyFR",
        vendor: "PyFR",
        extensions: &[".pyfrs", ".pyfrm"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "flow360",
        name: "Flexcompute Flow360",
        vendor: "Flexcompute",
        extensions: &[".flow360.json", ".flow360.toml"],
        access: Access::NeedsCredentials,
    },
    FormatSpec {
        id: "luminary",
        name: "Luminary Cloud",
        vendor: "Luminary Cloud",
        extensions: &[".luminary.json", ".luminary.toml"],
        access: Access::NeedsCredentials,
    },
    FormatSpec {
        id: "simscale",
        name: "SimScale",
        vendor: "SimScale",
        extensions: &[".simscale.json", ".simscale.toml"],
        access: Access::NeedsCredentials,
    },
    FormatSpec {
        id: "rescale",
        name: "Rescale",
        vendor: "Rescale",
        extensions: &[".rescale.json", ".rescale.toml"],
        access: Access::NeedsCredentials,
    },
    FormatSpec {
        id: "converge",
        name: "CONVERGE CFD",
        vendor: "Convergent Science",
        extensions: &[],
        access: Access::RouteOnly,
    },
    FormatSpec {
        id: "powerflow",
        name: "PowerFLOW",
        vendor: "Dassault Systemes",
        extensions: &[".snc", ".cdi", ".fnc", ".ckpt"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "fidelity",
        name: "Cadence Fidelity / FINE",
        vendor: "Cadence",
        extensions: &[".iec", ".cfv", ".run"],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "acusolve",
        name: "Altair AcuSolve",
        vendor: "Altair",
        extensions: &[],
        access: Access::NeedsExport,
    },
    FormatSpec {
        id: "code-saturne",
        name: "code_saturne",
        vendor: "EDF",
        extensions: &[],
        access: Access::RouteOnly,
    },
    FormatSpec {
        id: "palabos",
        name: "Palabos",
        vendor: "University of Geneva",
        extensions: &[],
        access: Access::RouteOnly,
    },
    FormatSpec {
        id: "optistruct",
        name: "Altair OptiStruct (decks, OP2, punch; .h3d through an export)",
        vendor: "Altair",
        extensions: &[".h3d", ".pch", ".pun", ".punch"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "radioss",
        name: "Altair Radioss / OpenRadioss animation files and starter decks",
        vendor: "Altair",
        extensions: &[".rad"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "simulink",
        name: "Simulink logs, FMU results and catalogues",
        vendor: "MathWorks / Modelica Association",
        extensions: &[".slx", ".fmu", ".mdl"],
        access: Access::Direct,
    },
    FormatSpec {
        id: "catalyst",
        name: "ParaView Catalyst / in-situ streams (spool, Conduit relay, ADIOS2)",
        vendor: "Kitware",
        extensions: &[".root", ".bp", ".sst"],
        access: Access::Direct,
    },
];

/// Look up a catalogue row by connector id.
pub fn spec(id: &str) -> Option<&'static FormatSpec> {
    FORMATS.iter().find(|f| f.id == id)
}

const HDF5_MAGIC: &[u8] = b"\x89HDF\r\n\x1a\n";
const HEAD_BYTES: usize = 4096;

/// What a cheap look at a path reveals; computed once and shared by all rules.
struct Probe<'a> {
    path: &'a Path,
    is_dir: bool,
    /// Lower-cased file (or directory) name.
    name: String,
    head: Vec<u8>,
    /// `head` decoded lossily and upper-cased, for keyword checks.
    text: String,
}

impl<'a> Probe<'a> {
    fn new(path: &'a Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let mut head = Vec::new();
        if meta.is_file()
            && let Ok(f) = File::open(path)
        {
            let _ = f.take(HEAD_BYTES as u64).read_to_end(&mut head);
        }
        let text = String::from_utf8_lossy(&head).to_uppercase();
        Some(Self {
            path,
            is_dir: meta.is_dir(),
            name,
            head,
            text,
        })
    }

    /// Longest entry of `exts` the file name ends with.
    fn ext(&self, exts: &'static [&'static str]) -> Option<&'static str> {
        exts.iter()
            .copied()
            .filter(|e| self.name.ends_with(e))
            .max_by_key(|e| e.len())
    }

    fn has_ext(&self, e: &str) -> bool {
        self.name.ends_with(e)
    }

    fn starts(&self, magic: &[u8]) -> bool {
        self.head.starts_with(magic)
    }

    fn hdf5(&self) -> bool {
        self.starts(HDF5_MAGIC) || self.head.get(512..520) == Some(HDF5_MAGIC)
    }

    fn netcdf3(&self) -> bool {
        self.head.len() > 4 && &self.head[..3] == b"CDF" && matches!(self.head[3], 1 | 2 | 5)
    }

    fn contains(&self, needle: &[u8]) -> bool {
        self.head.windows(needle.len()).any(|w| w == needle)
    }

    fn keyword(&self, kw: &str) -> bool {
        self.text.contains(kw)
    }

    /// A line of the head starts (after blanks) with `kw` (upper-case).
    fn line_starts(&self, kw: &str) -> bool {
        self.text.lines().any(|l| l.trim_start().starts_with(kw))
    }

    fn child(&self, rel: &str) -> bool {
        self.path.join(rel).exists()
    }

    /// First bytes of a small sidecar file inside a directory, upper-cased.
    fn child_text(&self, rel: &str) -> Option<String> {
        let mut buf = Vec::new();
        File::open(self.path.join(rel))
            .ok()?
            .take(HEAD_BYTES as u64)
            .read_to_end(&mut buf)
            .ok()?;
        Some(String::from_utf8_lossy(&buf).to_uppercase())
    }
}

fn hit(out: &mut Vec<Detection>, p: &Probe<'_>, id: &str, confidence: f32, note: &str) {
    let mut notes = vec![note.to_string()];
    if let Some(s) = spec(id) {
        match s.access {
            Access::NeedsExport => {
                notes.push(format!(
                    "native file is proprietary: see `python -m frontier_connectors plugins \
                     exporters` for the {} exporter scripts",
                    s.name
                ));
            }
            Access::NeedsCredentials => notes.push("cloud API: needs credentials".into()),
            _ => {}
        }
    }
    if let Some(existing) = out.iter_mut().find(|d| d.connector_id == id) {
        if confidence > existing.confidence {
            existing.confidence = confidence;
            existing.notes = notes;
        }
        return;
    }
    out.push(Detection {
        connector_id: id.to_string(),
        root: p.path.to_path_buf(),
        confidence,
        notes,
    });
}

fn detect_dir(p: &Probe<'_>, out: &mut Vec<Detection>) {
    let control = p.child("system/controlDict");
    let mesh = p.child("constant/polyMesh");
    let decomposed = p.child("processor0") || p.child("processors");
    if control || mesh || decomposed {
        let conf = match (control, mesh || decomposed) {
            (true, true) => 0.99,
            (true, false) => 0.8,
            _ => 0.6,
        };
        hit(out, p, "openfoam", conf, "OpenFOAM case layout");
    }
    if p.name.ends_with(".zarr")
        || p.child("zarr.json")
        || p.child(".zgroup")
        || p.child(".zarray")
        || p.child(".zattrs")
    {
        hit(out, p, "zarr", 0.9, "zarr store");
    }
    if let Some(manifest) = p.child_text("manifest.json")
        && manifest.contains("FRONTIER-DATASET")
    {
        hit(out, p, "zarr", 0.95, "frontier-dataset directory");
    }
    if let Some(sidecar) = p.child_text("frontier_export.json") {
        let (id, what) = if sidecar.contains("STAR-CCM") {
            ("starccm", "STAR-CCM+ export directory")
        } else if sidecar.contains("CFX") {
            ("cfx", "CFX export directory")
        } else if sidecar.contains("COMSOL") {
            ("comsol", "COMSOL export directory")
        } else if sidecar.contains("FLUENT") {
            ("fluent-live", "Fluent export directory")
        } else if sidecar.contains("POWERFLOW") {
            ("powerflow", "PowerFLOW export directory")
        } else if sidecar.contains("ACUSOLVE") {
            ("acusolve", "AcuSolve export directory")
        } else if sidecar.contains("OPTISTRUCT") || sidecar.contains("HYPERVIEW") {
            ("optistruct", "HyperView export directory")
        } else {
            ("", "")
        };
        if !id.is_empty() {
            hit(out, p, id, 0.95, what);
        }
    }
    if p.child("meta.json")
        && let Some(meta) = p.child_text("meta.json")
        && meta.contains("ABAQUS")
    {
        hit(out, p, "abaqus", 0.95, "Abaqus odb_export directory");
    }
    if p.child("ACUSIM.DIR") {
        hit(out, p, "acusolve", 0.9, "ACUSIM.DIR present");
    }
    if p.child("frontier-stream.json") {
        hit(out, p, "catalyst", 0.97, "frontier-spool stream directory");
    }
    if p.name.ends_with(".bp") && (p.child("md.idx") || p.child("md.0")) {
        hit(out, p, "catalyst", 0.9, "ADIOS2 BP directory");
    }
    if p.child("postprocessing") && (p.child("run_solver") || p.child("setup.xml")) {
        hit(out, p, "code-saturne", 0.8, "code_saturne RESU layout");
    }
    // one pass over the entries for file-family layouts
    let Ok(entries) = std::fs::read_dir(p.path) else {
        return;
    };
    let (mut d3plot, mut vtk, mut nek, mut case, mut stl, mut cff) = (false, 0u32, 0u32, 0, 0, 0);
    for e in entries.flatten().take(4096) {
        let n = e.file_name().to_string_lossy().to_lowercase();
        if n == "d3plot" {
            d3plot = true;
        } else if [".vtu", ".vtp", ".vti", ".vts", ".vtr", ".vtm", ".vtk"]
            .iter()
            .any(|x| n.ends_with(x))
        {
            vtk += 1;
        } else if nek_field_name(&n) {
            nek += 1;
        } else if n.ends_with(".case") || n.ends_with(".encas") {
            case += 1;
        } else if n.ends_with(".stl") {
            stl += 1;
        } else if n.ends_with(".dat.h5") || n.ends_with(".cas.h5") {
            cff += 1;
        }
    }
    if d3plot {
        hit(out, p, "lsdyna", 0.9, "directory contains d3plot");
    }
    if vtk >= 2 && !control {
        hit(out, p, "vtk", 0.55, "directory of numbered VTK files");
    }
    if nek >= 1 {
        hit(out, p, "nek5000", 0.7, "directory of Nek5000 field files");
    }
    if case >= 1 {
        hit(
            out,
            p,
            "ensight",
            0.6,
            "directory contains an EnSight case file",
        );
    }
    if stl >= 2 {
        hit(out, p, "geometry", 0.5, "directory of STL files");
    }
    if cff >= 2 {
        hit(out, p, "fluent-cff", 0.7, "directory of Fluent CFF files");
    }
}

/// `case0.f00001`-style Nek5000 field file name.
fn nek_field_name(lower_name: &str) -> bool {
    match lower_name.rsplit_once(".f") {
        Some((stem, digits)) => {
            !stem.is_empty() && digits.len() == 5 && digits.bytes().all(|b| b.is_ascii_digit())
        }
        None => false,
    }
}

fn detect_file(p: &Probe<'_>, out: &mut Vec<Detection>) {
    let hdf5 = p.hdf5();

    // -- unambiguous magic ---------------------------------------------------------------
    if p.contains(b"MATLAB 7.3 MAT-file") || p.starts(b"MATLAB 5.0 MAT-file") {
        hit(out, p, "matlab", 0.98, "MAT-file header");
    } else if p.has_ext(".mat") {
        hit(
            out,
            p,
            "matlab",
            0.6,
            "extension .mat (v4 or unknown header)",
        );
    }
    if p.starts(b"Octave-1-L") || p.starts(b"Octave-1-B") {
        hit(out, p, "octave", 0.98, "Octave binary save file");
    } else if p.starts(b"# Created by Octave") {
        hit(out, p, "octave", 0.97, "Octave text save file");
    }
    if p.starts(&[0x00, 0x00, 0x54, 0x2c]) {
        hit(
            out,
            p,
            "radioss",
            0.95,
            "Radioss animation file magic 0x542c",
        );
    }
    if p.has_ext(".fluent.json") {
        hit(out, p, "fluent-live", 0.9, "Fluent session spec");
    }
    if p.has_ext(".snc") && p.netcdf3() {
        hit(
            out,
            p,
            "powerflow",
            0.99,
            "PowerFLOW surface measurement (NetCDF)",
        );
    }
    if p.has_ext(".mat") && p.contains(b"Aclass") {
        hit(out, p, "simulink", 0.9, "Dymola / OpenModelica result file");
    }
    if p.starts(b"#!TDV") {
        hit(out, p, "tecplot", 0.98, "Tecplot binary magic #!TDV");
    } else if p.has_ext(".szplt") {
        hit(
            out,
            p,
            "tecplot",
            0.9,
            "Tecplot SZL (needs TecIO / pytecplot)",
        );
    }
    if p.starts(b"# vtk DataFile") {
        hit(out, p, "vtk", 0.98, "legacy VTK header");
    } else if p.contains(b"<VTKFile") {
        hit(out, p, "vtk", 0.98, "<VTKFile> root element");
    } else if let Some(e) = p.ext(spec("vtk").map(|s| s.extensions).unwrap_or(&[])) {
        let conf = if e == ".series" && !p.keyword("FILE-SERIES-VERSION") {
            0.0
        } else if e == ".vtkhdf" && !hdf5 {
            0.3
        } else {
            0.6
        };
        if conf > 0.0 {
            hit(out, p, "vtk", conf, &format!("extension {e}"));
        }
    }
    if p.starts(b"#std ") {
        hit(out, p, "nek5000", 0.98, "Nek5000 #std header");
    } else if p.has_ext(".nek5000") || nek_field_name(&p.name) {
        hit(out, p, "nek5000", 0.7, "Nek5000 file name");
    }
    if p.starts(b"PAR1") {
        hit(out, p, "tabular", 0.9, "Parquet magic");
    }
    if p.starts(b"$MeshFormat") {
        hit(out, p, "gmsh", 0.98, "$MeshFormat header");
    }
    if p.contains(b"<Xdmf") {
        hit(out, p, "xdmf", 0.98, "<Xdmf> root element");
    } else if p.has_ext(".xdmf") || p.has_ext(".xmf") {
        hit(out, p, "xdmf", 0.6, "extension");
    }
    // real ADF files set the high bit on the leading "@(#)", so only the rest is compared
    if p.head.get(4..16) == Some(b"ADF Database".as_slice()) && p.has_ext(".cgns") {
        hit(
            out,
            p,
            "cgns",
            0.9,
            "CGNS/ADF file (read through the VTK engine)",
        );
    }

    // -- HDF5 containers: the extension decides, content markers upgrade --------------------
    if hdf5 {
        if p.has_ext(".cas.h5") || p.has_ext(".dat.h5") || p.has_ext(".msh.h5") {
            hit(out, p, "fluent-cff", 0.97, "HDF5 + Fluent CFF suffix");
        } else if p.has_ext(".cgns") {
            hit(out, p, "cgns", 0.97, "HDF5 + .cgns");
        } else if p.has_ext(".med") || p.has_ext(".rmed") {
            hit(out, p, "med", 0.95, "HDF5 + .med");
        } else if p.has_ext(".pyfrs") || p.has_ext(".pyfrm") {
            hit(out, p, "pyfr", 0.95, "HDF5 + PyFR suffix");
        } else if p.name.starts_with("post") && p.has_ext(".h5") {
            hit(
                out,
                p,
                "converge",
                0.6,
                "CONVERGE post file name (post*.h5)",
            );
            hit(out, p, "hdf5", 0.3, "generic HDF5");
        } else if p.has_ext(".h5m") {
            hit(out, p, "meshio", 0.8, "MOAB h5m");
        } else if p.has_ext(".vtkhdf") {
            hit(out, p, "vtk", 0.95, "VTKHDF");
        } else if p
            .ext(spec("netcdf").map(|s| s.extensions).unwrap_or(&[]))
            .is_some()
        {
            hit(out, p, "netcdf", 0.9, "NetCDF-4 (HDF5)");
        } else if p
            .ext(spec("exodus").map(|s| s.extensions).unwrap_or(&[]))
            .is_some()
        {
            hit(out, p, "exodus", 0.85, "Exodus II on NetCDF-4");
        } else if !p.has_ext(".mat") {
            hit(
                out,
                p,
                "hdf5",
                0.3,
                "generic HDF5: Python detection inspects the tree (CGNS, CFF, MED, ...)",
            );
        }
    } else if p.netcdf3() {
        if p.contains(b"num_el_blk") || p.contains(b"coor_names") || p.contains(b"num_nodes") {
            hit(
                out,
                p,
                "exodus",
                0.95,
                "NetCDF classic with Exodus dimensions",
            );
        } else if p
            .ext(spec("exodus").map(|s| s.extensions).unwrap_or(&[]))
            .is_some()
        {
            hit(out, p, "exodus", 0.85, "NetCDF classic + Exodus extension");
        } else {
            hit(out, p, "netcdf", 0.95, "NetCDF classic magic");
        }
    }

    // -- text formats recognised by keywords -------------------------------------------------
    if (p.has_ext(".case") || p.has_ext(".encas")) && p.line_starts("FORMAT") {
        hit(out, p, "ensight", 0.97, "EnSight case file");
    } else if p.has_ext(".case") || p.has_ext(".encas") {
        hit(out, p, "ensight", 0.5, "extension");
    }
    if p.has_ext(".foam") || p.has_ext(".openfoam") {
        hit(out, p, "openfoam", 0.95, "ParaView OpenFOAM stub file");
    } else if p.name == "controldict" && p.keyword("FOAMFILE") {
        hit(out, p, "openfoam", 0.9, "system/controlDict");
    }
    if p.line_starts("*KEYWORD") {
        hit(out, p, "lsdyna", 0.97, "*KEYWORD deck");
    } else if [".k", ".key", ".dyn", ".kw", ".k.gz", ".key.gz"]
        .iter()
        .any(|e| p.has_ext(e))
    {
        hit(out, p, "lsdyna", 0.6, "keyword deck extension");
    }
    if p.name.starts_with("d3plot") || p.has_ext(".d3plot") {
        hit(out, p, "lsdyna", 0.9, "d3plot family file");
    }
    let abaqus_kw = ["*HEADING", "*NODE", "*PART", "*ELEMENT", "*PREPRINT"]
        .iter()
        .any(|k| p.line_starts(k));
    if p.has_ext(".inp") {
        hit(
            out,
            p,
            "abaqus",
            if abaqus_kw { 0.95 } else { 0.4 },
            "Abaqus input deck",
        );
    } else if p.has_ext(".odb") {
        hit(out, p, "abaqus", 0.9, "Abaqus output database");
    } else if p.has_ext(".frd") || p.has_ext(".fil") {
        hit(out, p, "abaqus", 0.7, "CalculiX .frd / Abaqus .fil results");
    }
    let bulk = p.line_starts("BEGIN BULK")
        || p.line_starts("GRID")
        || p.line_starts("CEND")
        || p.line_starts("SOL ");
    if [".bdf", ".nas", ".fem"].iter().any(|e| p.has_ext(e)) {
        hit(
            out,
            p,
            "nastran",
            if bulk { 0.95 } else { 0.7 },
            "Nastran bulk data",
        );
    } else if p.has_ext(".op2") {
        hit(
            out,
            p,
            "nastran",
            0.9,
            "Nastran OP2 results (reader: pyNastran)",
        );
    } else if p.has_ext(".dat") && (p.line_starts("BEGIN BULK") || p.line_starts("CEND")) {
        hit(out, p, "nastran", 0.85, "Nastran deck with .dat extension");
    }
    let tecplot_kw = p.line_starts("TITLE") || p.line_starts("VARIABLES") || p.line_starts("ZONE");
    if tecplot_kw && [".dat", ".tec", ".tp", ".plt"].iter().any(|e| p.has_ext(e)) {
        hit(out, p, "tecplot", 0.9, "Tecplot ASCII keywords");
    }
    if p.keyword("% VERSION:") && p.keyword("COMSOL") || p.line_starts("% MODEL:") {
        hit(out, p, "comsol", 0.95, "COMSOL export header");
    } else if p.has_ext(".mph") {
        hit(out, p, "comsol", 0.9, "COMSOL model file");
    }
    if p.has_ext(".su2") {
        hit(
            out,
            p,
            "su2",
            if p.line_starts("NDIME") || p.keyword("NDIME=") {
                0.97
            } else {
                0.6
            },
            "SU2 mesh",
        );
    }
    if [".cas", ".dat", ".msh", ".cas.gz", ".dat.gz", ".msh.gz"]
        .iter()
        .any(|e| p.has_ext(e))
        && !p.starts(b"$MeshFormat")
        && !tecplot_kw
        && !hdf5
    {
        let scheme = p.head.first() == Some(&b'(');
        let gz = p.starts(b"\x1f\x8b");
        let is_dat = p.has_ext(".dat");
        if scheme || gz || !is_dat {
            hit(
                out,
                p,
                "fluent-legacy",
                if scheme {
                    0.9
                } else if is_dat {
                    0.4
                } else {
                    0.6
                },
                "Fluent legacy case/data/mesh",
            );
        }
    }

    // -- extension only -------------------------------------------------------------------
    for (id, conf) in [
        ("starccm", 0.9),
        ("cfx", 0.6),
        ("ansys-rst", 0.8),
        ("geometry", 0.8),
        ("pyfr", 0.6),
        ("powerflow", 0.7),
        ("fidelity", 0.5),
        ("optistruct", 0.7),
        ("radioss", 0.8),
        ("simulink", 0.6),
        ("catalyst", 0.5),
        ("flow360", 0.9),
        ("luminary", 0.9),
        ("simscale", 0.9),
        ("rescale", 0.9),
        ("med", 0.5),
        ("meshio", 0.5),
        ("exodus", 0.4),
        ("netcdf", 0.4),
        ("cgns", 0.5),
        ("zarr", 0.5),
    ] {
        if let Some(s) = spec(id)
            && let Some(e) = p.ext(s.extensions)
            && !out.iter().any(|d| d.connector_id == id)
        {
            hit(out, p, id, conf, &format!("extension {e}"));
        }
    }
    if [".csv", ".tsv"].iter().any(|e| p.has_ext(e)) && !out.iter().any(|d| d.confidence > 0.9) {
        hit(out, p, "tabular", 0.6, "delimited text table");
    } else if [".txt", ".dat"].iter().any(|e| p.has_ext(e)) && out.is_empty() {
        hit(
            out,
            p,
            "tabular",
            0.3,
            "plain text table (needs numeric columns)",
        );
    }
}

/// Fast detection over the whole catalogue; best match first. Returns an empty vector
/// for paths that do not exist or that nothing recognises.
pub fn detect(path: &Path) -> Vec<Detection> {
    let Some(probe) = Probe::new(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if probe.is_dir {
        detect_dir(&probe, &mut out);
    } else {
        detect_file(&probe, &mut out);
    }
    out.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    out
}
