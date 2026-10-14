// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::fs;
use std::path::Path;

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&p, bytes).unwrap();
    p
}

fn best(path: &Path) -> (String, f32) {
    let found = formats::detect(path);
    assert!(!found.is_empty(), "nothing detected for {}", path.display());
    (found[0].connector_id.clone(), found[0].confidence)
}

#[cfg(unix)]
fn open_err(reg: &ConnectorRegistry, path: &Path) -> String {
    match reg.open(path) {
        Ok(_) => panic!("expected an error opening {}", path.display()),
        Err(e) => e.to_string(),
    }
}

const HDF5: &[u8] = b"\x89HDF\r\n\x1a\n\0\0\0\0";

#[test]
fn detects_files_by_magic_and_extension() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let mut mat73 = b"MATLAB 7.3 MAT-file, Platform: MACI64".to_vec();
    mat73.resize(512, b' ');
    mat73.extend_from_slice(HDF5);
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        (
            "a.vtu",
            b"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\">".to_vec(),
            "vtk",
        ),
        ("a.vtk", b"# vtk DataFile Version 3.0\n".to_vec(), "vtk"),
        ("t.pvd", b"<VTKFile type=\"Collection\">".to_vec(), "vtk"),
        (
            "f.vtu.series",
            b"{\"file-series-version\": \"1.0\"}".to_vec(),
            "vtk",
        ),
        ("run.cas.h5", HDF5.to_vec(), "fluent-cff"),
        ("run-0010.dat.h5", HDF5.to_vec(), "fluent-cff"),
        ("wing.cgns", HDF5.to_vec(), "cgns"),
        (
            "old.cgns",
            b"\xc0\xa8\xa3\xa9ADF Database Version B02012".to_vec(),
            "cgns",
        ),
        ("mesh.med", HDF5.to_vec(), "med"),
        ("sol.pyfrs", HDF5.to_vec(), "pyfr"),
        ("era.nc", HDF5.to_vec(), "netcdf"),
        ("classic.nc", b"CDF\x01\0\0\0\0".to_vec(), "netcdf"),
        (
            "can.ex2",
            b"CDF\x02\0\0\0\0....num_el_blk....".to_vec(),
            "exodus",
        ),
        ("blob.h5", HDF5.to_vec(), "hdf5"),
        (
            "flow.case",
            b"FORMAT\ntype: ensight gold\nGEOMETRY\n".to_vec(),
            "ensight",
        ),
        ("case.foam", Vec::new(), "openfoam"),
        (
            "legacy.cas",
            b"(0 \"fluent19.2.0 build-id: 10194\")\n".to_vec(),
            "fluent-legacy",
        ),
        (
            "fluent.msh",
            b"(0 \"GAMBIT to Fluent File\")\n".to_vec(),
            "fluent-legacy",
        ),
        ("gmsh.msh", b"$MeshFormat\n4.1 0 8\n".to_vec(), "gmsh"),
        ("car.sim", b"\0\0binary".to_vec(), "starccm"),
        ("m.mph", b"PK\x03\x04".to_vec(), "comsol"),
        (
            "field.txt",
            b"% Model:              heat.mph\n% Version:            COMSOL 6.2.0.290\n".to_vec(),
            "comsol",
        ),
        ("job.odb", b"\0\0".to_vec(), "abaqus"),
        (
            "job.inp",
            b"*Heading\n** Job name\n*Node\n1, 0., 0., 0.\n".to_vec(),
            "abaqus",
        ),
        (
            "crash.k",
            b"$ comment\n*KEYWORD\n*NODE\n".to_vec(),
            "lsdyna",
        ),
        ("d3plot", vec![0u8; 256], "lsdyna"),
        ("d3plot01", vec![0u8; 256], "lsdyna"),
        (
            "wing.bdf",
            b"SOL 101\nCEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\n".to_vec(),
            "nastran",
        ),
        ("wing.op2", vec![4, 0, 0, 0, 3, 0, 0, 0], "nastran"),
        (
            "deck.dat",
            b"SOL 103\nCEND\nBEGIN BULK\n".to_vec(),
            "nastran",
        ),
        (
            "res.mat",
            b"MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Mon".to_vec(),
            "matlab",
        ),
        ("big.mat", mat73, "matlab"),
        ("flow.plt", b"#!TDV112\x01\0\0\0".to_vec(), "tecplot"),
        ("flow.szplt", b"SZPLT".to_vec(), "tecplot"),
        (
            "flow.dat",
            b"TITLE = \"x\"\nVARIABLES = \"x\" \"y\" \"p\"\nZONE T=\"a\", I=3\n".to_vec(),
            "tecplot",
        ),
        (
            "grid.xdmf",
            b"<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\">".to_vec(),
            "xdmf",
        ),
        ("naca.su2", b"NDIME= 2\nNELEM= 10\n".to_vec(), "su2"),
        (
            "probes.csv",
            b"time,x,y,z,p\n0,0,0,0,1\n".to_vec(),
            "tabular",
        ),
        ("table.parquet", b"PAR1\x15\x04".to_vec(), "tabular"),
        ("car.stl", b"solid car\n".to_vec(), "geometry"),
        ("part.step", b"ISO-10303-21;\n".to_vec(), "geometry"),
        (
            "chan0.f00001",
            b"#std 4  8  8  8        512        512  0.1E+01".to_vec(),
            "nek5000",
        ),
        ("beam.rst", vec![0u8; 64], "ansys-rst"),
        ("pump.res", vec![0u8; 64], "cfx"),
        (
            "case.flow360.json",
            b"{\"provider\": \"flow360\"}".to_vec(),
            "flow360",
        ),
        (
            "job.rescale.json",
            b"{\"provider\": \"rescale\"}".to_vec(),
            "rescale",
        ),
        ("flow.snc", vec![0u8; 8], "powerflow"),
        ("result.h3d", vec![0u8; 8], "optistruct"),
        ("beam.frd", b"    1C".to_vec(), "abaqus"),
        ("post000001_0.5.h5", HDF5.to_vec(), "converge"),
        ("deck.k.gz", b"\x1f\x8b\x08".to_vec(), "lsdyna"),
        (
            "case.luminary.toml",
            b"provider = \"luminary\"".to_vec(),
            "luminary",
        ),
    ];
    for (name, bytes, want) in cases {
        let p = write(d, name, &bytes);
        let (id, conf) = best(&p);
        assert_eq!(id, want, "{name} detected as {id}");
        assert!(conf > 0.0 && conf <= 1.0);
    }
}

#[test]
fn magic_beats_extension_and_unknown_files_are_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // a .dat file is Tecplot only with Tecplot keywords, Fluent only with a scheme header
    let plain = write(d, "numbers.dat", b"0.0 1.0 2.0\n1.0 2.0 3.0\n");
    assert_eq!(best(&plain).0, "tabular");
    assert!(best(&plain).1 <= 0.3);
    let fluent = write(d, "old.dat", b"(0 \"fluent data\")\n");
    assert_eq!(best(&fluent).0, "fluent-legacy");
    // HDF5 content with a strong header wins over a misleading generic extension
    let confident = best(&write(d, "x.cas.h5", HDF5));
    let generic = best(&write(d, "x.h5", HDF5));
    assert!(confident.1 > generic.1);
    assert!(formats::detect(&write(d, "notes.md", b"# hello")).is_empty());
    assert!(formats::detect(&d.join("missing.vtu")).is_empty());
    // an .inp without Abaqus keywords is only a weak match
    let weak = best(&write(d, "other.inp", b"some other tool\n"));
    assert_eq!(weak.0, "abaqus");
    assert!(weak.1 < 0.5);
}

#[test]
fn detects_directory_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();

    let foam = d.join("cavity");
    write(
        &foam,
        "system/controlDict",
        b"FoamFile {}\napplication icoFoam;\n",
    );
    write(&foam, "constant/polyMesh/points", b"FoamFile {}\n");
    assert_eq!(best(&foam), ("openfoam".to_string(), 0.99));

    let decomposed = d.join("decomposed");
    write(&decomposed, "system/controlDict", b"FoamFile {}\n");
    write(
        &decomposed,
        "processor0/constant/polyMesh/points",
        b"FoamFile {}\n",
    );
    assert_eq!(best(&decomposed).0, "openfoam");

    let star = d.join("star_export");
    write(
        &star,
        "frontier_export.json",
        b"{\"solver\": \"STAR-CCM+\", \"version\": \"2402\"}",
    );
    write(&star, "export.case", b"FORMAT\ntype: ensight gold\n");
    assert_eq!(best(&star).0, "starccm");

    let series = d.join("series");
    for i in 0..3 {
        write(&series, &format!("flow_{i:04}.vtu"), b"<VTKFile>");
    }
    assert_eq!(best(&series).0, "vtk");

    let dyna = d.join("crash");
    write(&dyna, "d3plot", &[0u8; 64]);
    write(&dyna, "d3plot01", &[0u8; 64]);
    assert_eq!(best(&dyna).0, "lsdyna");

    let store = d.join("fields.zarr");
    write(
        &store,
        "zarr.json",
        b"{\"zarr_format\": 3, \"node_type\": \"group\"}",
    );
    assert_eq!(best(&store).0, "zarr");

    let dataset = d.join("dataset");
    write(
        &dataset,
        "manifest.json",
        b"{\"format\": \"frontier-dataset\", \"version\": 1}",
    );
    assert_eq!(best(&dataset).0, "zarr");

    let nek = d.join("nek");
    write(&nek, "chan0.f00001", b"#std 4");
    assert_eq!(best(&nek).0, "nek5000");

    assert!(formats::detect(&tempfile::tempdir().unwrap().keep()).is_empty());
}

#[test]
fn detects_the_formats_added_with_the_plugin_split() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let cases: Vec<(PathBuf, &str)> = vec![
        (write(d, "zoo.bin", b"Octave-1-L\0\x05\0\0\0x"), "octave"),
        (
            write(
                d,
                "zoo.txt",
                b"# Created by Octave 11.3.0, Sat Oct 10\n# name: x\n",
            ),
            "octave",
        ),
        (
            write(d, "crashA001", &[0, 0, 0x54, 0x2c, 0, 0, 0, 0]),
            "radioss",
        ),
        (write(d, "crash_0000.rad", b"#RADIOSS STARTER\n"), "radioss"),
        (
            write(d, "bracket.pch", b"$TITLE   = BRACKET\n"),
            "optistruct",
        ),
        (
            write(d, "run.fluent.json", b"{\"case\": \"a.cas.h5\"}"),
            "fluent-live",
        ),
        (write(d, "wing.snc", b"CDF\x02\0\0\0\0"), "powerflow"),
        (write(d, "plant.fmu", b"PK\x03\x04"), "simulink"),
        (write(d, "plant.mdl", b"Model {\n"), "simulink"),
        (
            write(d, "plant_res.mat", b"\0\0\0\0Aclass\0Atrajectory"),
            "simulink",
        ),
        (write(d, "mesh.cycle_000001.root", b"{}"), "catalyst"),
    ];
    for (path, id) in &cases {
        assert_eq!(&best(path).0, id, "{}", path.display());
    }
    let spool = d.join("spool");
    fs::create_dir_all(&spool).unwrap();
    write(
        &spool,
        "frontier-stream.json",
        b"{\"protocol\": \"frontier-spool\"}",
    );
    assert_eq!(best(&spool).0, "catalyst");
    let bp = d.join("run.bp");
    fs::create_dir_all(&bp).unwrap();
    write(&bp, "md.idx", b"ADIOS-BP v005");
    assert_eq!(best(&bp).0, "catalyst");
}

#[test]
fn catalogue_is_consistent() {
    let mut ids: Vec<&str> = FORMATS.iter().map(|f| f.id).collect();
    let n = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), n, "duplicate connector ids in FORMATS");
    for f in FORMATS {
        assert!(!f.name.is_empty() && !f.vendor.is_empty());
        for e in f.extensions {
            assert!(
                e.starts_with('.') && *e == e.to_lowercase(),
                "bad extension {e}"
            );
        }
        assert!(formats::spec(f.id).is_some());
    }
}

/// Fake `python -m frontier_connectors`: a shell script that prints a canned document.
#[cfg(unix)]
fn fake_bridge(dir: &Path, body: &str) -> PythonBridge {
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("fake-connectors.sh");
    fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    PythonBridge::new(script.to_string_lossy().to_string(), Vec::<String>::new())
}

#[cfg(unix)]
#[test]
fn bridge_parses_describe_and_maps_to_sample_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let doc = r#"{"connector_id":"cgns","path":"/x/wing.cgns","format":"cgns-hdf5","solver":"elsA","solver_version":"5.1","domain":"cfd","temporal":"transient","physics":{"turbulence_model":"SA"},"zones":[{"name":"fluid","kind":"mesh","role":"volume","n_points":100,"n_cells":60,"dim":3,"cell_types":{"hexahedron":60},"shape":null,"bounds":[0,2,-1,1,0,0.5],"bc_type":null,"default":true},{"name":"fluid/wall","kind":"surface","role":"boundary","n_points":30,"n_cells":20,"dim":3,"cell_types":{},"shape":null,"bounds":[0,9,0,9,0,9],"bc_type":"BCWall","default":false}],"fields":[{"name":"Velocity","location":"point","components":3,"units":"m/s"},{"name":"Pressure","location":"cell","components":1,"units":"Pa"}],"time":{"values":[0.0,0.1,0.2],"n_steps":3,"dt":0.1,"units":"s","uniform":true,"quantity":"time"},"boundary_conditions":{"wall":{"type":"BCWall"}},"units":{"length":"m"},"parameters":{"Mach":0.3},"metadata":{},"warnings":[]}"#;
    let bridge = fake_bridge(
        dir.path(),
        &format!("echo 'progress noise that is not json'\necho '{doc}'"),
    );
    let desc = bridge
        .describe(Path::new("wing.cgns"), Some("cgns"))
        .unwrap();
    assert_eq!(desc.zones.len(), 2);
    assert!(desc.zones[0].read_by_default && !desc.zones[1].read_by_default);
    let meta = desc.into_sample_metadata();
    assert_eq!(meta.solver, "elsA");
    assert_eq!(meta.solver_version.as_deref(), Some("5.1"));
    // only the default (volume) zone is counted; bounds are reordered to min/max triples
    assert_eq!((meta.mesh.n_points, meta.mesh.n_cells), (100, 60));
    assert_eq!(meta.mesh.bounds, [0.0, -1.0, 0.0, 2.0, 1.0, 0.5]);
    assert_eq!(meta.times, vec![0.0, 0.1, 0.2]);
    assert_eq!(meta.fields.len(), 2);
    assert_eq!(meta.fields[0].components, 3);
    assert_eq!(meta.fields[1].unit, "Pa");
    assert_eq!(meta.attributes["connector"], "cgns");
    assert_eq!(meta.attributes["physics"]["turbulence_model"], "SA");
    assert_eq!(meta.attributes["zones"][1]["bc_type"], "BCWall");
}

#[cfg(unix)]
#[test]
fn bridge_surfaces_connector_errors_with_hints_and_timeouts() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = fake_bridge(
        dir.path(),
        r#"echo '{"error":"NeedsExportError","message":"car.sim is a proprietary STAR-CCM+ file","hint":"run python -m frontier_connectors exporters starccm"}'; exit 2"#,
    );
    let err = bridge
        .describe(Path::new("car.sim"), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("proprietary STAR-CCM+ file"), "{err}");
    assert!(
        err.contains("hint: run python -m frontier_connectors exporters starccm"),
        "{err}"
    );

    let crash = fake_bridge(dir.path(), "echo 'Traceback: boom' >&2; exit 1");
    let err = crash.list().unwrap_err().to_string();
    assert!(
        err.contains("without a JSON document") && err.contains("boom"),
        "{err}"
    );

    let slow =
        fake_bridge(dir.path(), "sleep 5").with_timeout(std::time::Duration::from_millis(200));
    assert!(slow.list().unwrap_err().to_string().contains("timed out"));

    let missing = PythonBridge::new(
        "/nonexistent/frontier-python",
        ["-m", "frontier_connectors"],
    );
    let err = missing.list().unwrap_err().to_string();
    assert!(
        err.contains("cannot start") && err.contains(python::CMD_ENV),
        "{err}"
    );
}

#[cfg(unix)]
#[test]
fn bridge_builds_convert_arguments() {
    let dir = tempfile::tempdir().unwrap();
    // echo the arguments back inside the report so the test can inspect them
    let bridge = fake_bridge(
        dir.path(),
        r#"printf '{"out":"/o","name":"%s","n_samples":1,"sample_ids":["a"],"backends":{"a":"zarr-stream"},"n_steps":{"a":3},"warnings":[],"seconds":0.1}\n' "$*""#,
    );
    let opts = ConvertOptions {
        connector: Some("openfoam".into()),
        zones: vec!["internalMesh".into(), "wall*".into()],
        fields: vec!["U".into(), "p".into()],
        times: Some("0:10:2".into()),
        overwrite: true,
        time_as_samples: true,
        ..Default::default()
    };
    let report = bridge
        .convert(Path::new("/data/case"), Path::new("/data/out"), &opts)
        .unwrap();
    assert_eq!(report.n_steps["a"], 3);
    let args = report.name;
    for want in [
        "convert /data/case --out /data/out",
        "--connector openfoam",
        "--zones internalMesh,wall*",
        "--fields U,p",
        "--times 0:10:2",
        "--overwrite",
        "--time-as-samples",
        "--json --compact",
    ] {
        assert!(args.contains(want), "missing `{want}` in `{args}`");
    }
}

#[cfg(unix)]
struct NativeStub {
    manifest: frontier_core::PluginManifest,
}

#[cfg(unix)]
impl Plugin for NativeStub {
    fn manifest(&self) -> &frontier_core::PluginManifest {
        &self.manifest
    }
}

#[cfg(unix)]
impl Connector for NativeStub {
    fn detect(&self, path: &Path) -> Option<Detection> {
        (path.extension()? == "vtu").then(|| Detection {
            connector_id: self.manifest.id.clone(),
            root: path.to_path_buf(),
            confidence: 0.9,
            notes: vec![],
        })
    }

    fn open(&self, _path: &Path) -> Result<Box<dyn SampleReader>> {
        Ok(Box::new(DescribedSample {
            metadata: SampleMetadata {
                solver: "native-stub".into(),
                solver_version: None,
                mesh: MeshDescriptor {
                    n_points: 1,
                    n_cells: 0,
                    n_faces: None,
                    dimension: 3,
                    bounds: [0.0; 6],
                },
                fields: vec![],
                times: vec![],
                attributes: Default::default(),
            },
        }))
    }
}

#[cfg(unix)]
fn stub() -> Box<NativeStub> {
    Box::new(NativeStub {
        manifest: frontier_core::PluginManifest {
            id: "connector-vtk".into(),
            name: "stub".into(),
            version: "0".into(),
            kind: frontier_core::PluginKind::Connector,
            data_model_version: frontier_core::DATA_MODEL_VERSION,
            description: String::new(),
        },
    })
}

#[cfg(unix)]
#[test]
fn registry_prefers_python_and_falls_back_to_native() {
    let dir = tempfile::tempdir().unwrap();
    let vtu = write(dir.path(), "a.vtu", b"<VTKFile type=\"UnstructuredGrid\">");

    // Python available: its description wins
    let ok = fake_bridge(
        dir.path(),
        r#"echo '{"connector_id":"vtk","solver":"from-python","zones":[],"fields":[],"time":{"values":[]}}'"#,
    );
    let mut reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Auto)
        .with_bridge(ok.clone());
    reg.register(stub());
    assert_eq!(reg.open(&vtu).unwrap().metadata().solver, "from-python");
    assert!(reg.ids().contains(&"connector-vtk") && reg.ids().contains(&"fluent-cff"));
    let found = reg.detect(&vtu);
    assert!(found.iter().any(|d| d.connector_id == "connector-vtk"));
    assert!(found.iter().any(|d| d.connector_id == "vtk"));

    // Python missing: the native connector takes over
    let gone = PythonBridge::new("/nonexistent/python", Vec::<String>::new());
    let mut reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Auto)
        .with_bridge(gone.clone());
    reg.register(stub());
    assert_eq!(reg.open(&vtu).unwrap().metadata().solver, "native-stub");

    // native mode never starts Python; python mode never falls back
    let mut reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Native)
        .with_bridge(ok);
    reg.register(stub());
    assert_eq!(reg.open(&vtu).unwrap().metadata().solver, "native-stub");
    let mut reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Python)
        .with_bridge(gone);
    reg.register(stub());
    assert!(reg.open(&vtu).is_err());

    // a format with no native reader reports the Python error; unknown paths are unsupported
    let cgns = write(dir.path(), "w.cgns", HDF5);
    let reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Auto)
        .with_bridge(PythonBridge::new(
            "/nonexistent/python",
            Vec::<String>::new(),
        ));
    assert!(open_err(&reg, &cgns).contains("cannot start"));
    let unknown = write(dir.path(), "readme.md", b"hi");
    assert!(open_err(&reg, &unknown).contains("no connector recognises"));

    let native = ConnectorRegistry::native_only();
    assert!(native.ids().is_empty() && native.detect(&cgns).is_empty());
}

#[cfg(unix)]
#[test]
fn bridge_parses_plugin_documents_and_builds_plugin_arguments() {
    use crate::plugins::{InstallOptions, PluginKind, PluginStatus, PluginTarget};
    let dir = tempfile::tempdir().unwrap();
    // echo the arguments on stderr-free stdout line 1 is not possible (one JSON document),
    // so the script records them in a file and answers by subcommand
    let log = dir.path().join("args.log");
    let body = format!(
        r#"echo "$@" >> {log}
case "$2" in
  list) echo '{{"data_model_version":"1.1","state_file":"/home/u/.frontier/plugins.toml","plugins":[{{"id":"fluent","kind":"connector","name":"Ansys Fluent","version":"0.2.0","dist":"frontier-connector-fluent","status":"enabled","enabled":true,"installed":true,"connectors":["fluent-cff","fluent-live"],"disabled_connectors":["fluent-live"],"optional":[{{"package":"ansys-fluent-core","licence":"MIT","flag":"needs a licensed Ansys Fluent installation"}}],"future_field":1}},{{"id":"exporter-matlab","kind":"exporter","status":"disabled","enabled":false,"installed":true,"exporters":["matlab"]}},{{"id":"flow360","kind":"connector","status":"not-installed","installed":false,"connectors":["flow360"]}}]}}' ;;
  enable|disable) echo '{{"results":[{{"scope":"connectors","id":"fluent-live","enabled":false,"changed":true,"state":"/home/u/.frontier/plugins.toml"}}]}}' ;;
  install) echo '{{"command":["uv","pip","install","frontier-connector-fluent"],"returncode":null,"dry_run":true,"plugins":[]}}' ;;
  doctor) echo '{{"framework":{{"version":"0.2.0","data_model_version":"1.1"}},"ok":false,"plugins":[{{"id":"nastran","kind":"connector","status":"enabled","ok":false,"problems":["connector nastran: ImportError: x"],"optional_status":[{{"package":"pyNastran","installed":null,"licence":"BSD-3-Clause","fix":"pip install pyNastran"}}],"loads":[{{"kind":"connector","id":"nastran","ok":false,"error":"ImportError: x"}}]}}],"deployed_exporters":[{{"vendor":"matlab","path":"/c/matlab","problem":"copy is gone"}}]}}' ;;
  install-exporter) echo '{{"vendor":"starccm","plugin":"exporter-starccm","version":"0.2.0","path":"/case/starccm","files":["ExportForFrontier.java"],"replaced_version":null,"usage":"starccm+ -batch ExportForFrontier.java case.sim"}}' ;;
esac"#,
        log = log.display()
    );
    let bridge = fake_bridge(dir.path(), &body);

    let list = bridge.plugins(true).unwrap();
    assert_eq!(list.data_model_version, "1.1");
    assert!(list.state_file.ends_with(".frontier/plugins.toml"));
    let fluent = list.get("fluent").unwrap();
    assert_eq!(
        (fluent.kind, fluent.status),
        (PluginKind::Connector, PluginStatus::Enabled)
    );
    assert!(fluent.enabled && fluent.disabled_connectors == ["fluent-live"]);
    assert_eq!(fluent.optional[0].licence, "MIT");
    let matlab = list.get("exporter-matlab").unwrap();
    assert_eq!(
        (matlab.kind, matlab.status),
        (PluginKind::Exporter, PluginStatus::Disabled)
    );
    assert_eq!(
        list.provider_of("flow360").unwrap().status,
        PluginStatus::NotInstalled
    );
    assert!(list.provider_of("fluent-live").is_some() && list.provider_of("nope").is_none());

    let done = bridge
        .set_plugin_enabled(
            &[
                PluginTarget::Connector("fluent-live".into()),
                PluginTarget::Plugin("luminary".into()),
            ],
            false,
        )
        .unwrap();
    assert!(done[0].changed && !done[0].enabled && done[0].scope == "connectors");

    let report = bridge
        .install_plugins(
            &["fluent".to_string()],
            &InstallOptions {
                find_links: vec![PathBuf::from("/mnt/usb/wheels")],
                no_index: true,
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.dry_run && report.returncode.is_none() && report.command[0] == "uv");

    let doctor = bridge.plugin_doctor().unwrap();
    assert!(!doctor.ok && doctor.framework.data_model_version == "1.1");
    let nastran = &doctor.plugins[0];
    assert_eq!(nastran.info.id, "nastran");
    assert!(!nastran.ok && nastran.loads[0].error.as_deref() == Some("ImportError: x"));
    assert_eq!(
        nastran.optional_status[0].fix.as_deref(),
        Some("pip install pyNastran")
    );
    assert_eq!(doctor.deployed_exporters[0].problem, "copy is gone");

    let copied = bridge
        .install_exporter("starccm", Path::new("/case"), false)
        .unwrap();
    assert_eq!(copied.files, ["ExportForFrontier.java"]);
    assert!(copied.usage.starts_with("starccm+ -batch"));

    let args = fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = args.lines().collect();
    assert_eq!(lines[0], "plugins list --available --json --compact");
    assert_eq!(
        lines[1],
        "plugins disable connector:fluent-live plugin:luminary --json --compact"
    );
    assert_eq!(
        lines[2],
        "plugins install fluent --find-links /mnt/usb/wheels --no-index --dry-run --json --compact"
    );
    assert_eq!(lines[3], "plugins doctor --json --compact");
    assert_eq!(
        lines[4],
        "plugins install-exporter starccm --to /case --json --compact"
    );
}

/// End-to-end against the real Python package. Runs when `uv` and the workspace are
/// available (the normal developer setup) and is skipped otherwise, e.g. in a Rust-only CI.
fn real_bridge() -> Option<PythonBridge> {
    if std::env::var_os("FRONTIER_SKIP_PYTHON_TESTS").is_some() {
        return None;
    }
    let root = python::workspace_root(Path::new(env!("CARGO_MANIFEST_DIR")))?;
    let bridge = PythonBridge::new(
        "uv",
        [
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-m",
            "frontier_connectors",
        ],
    )
    .in_dir(root)
    .with_timeout(std::time::Duration::from_secs(300));
    match bridge.list() {
        Ok(_) => Some(bridge),
        Err(e) => {
            eprintln!("skipping: Python frontier_connectors not runnable here: {e}");
            None
        }
    }
}

#[test]
fn catalog_matches_python_registry() {
    let Some(bridge) = real_bridge() else { return };
    let rows = bridge.list().unwrap();
    let rust_ids: Vec<&str> = FORMATS.iter().map(|f| f.id).collect();
    assert!(
        rows.iter().all(|row| rust_ids.contains(&row.id.as_str())),
        "installed Python connectors must be described by the Rust catalog"
    );
    for row in &rows {
        assert_ne!(row.status, "broken", "{}: {}", row.id, row.note);
        let spec = formats::spec(&row.id).unwrap();
        for ext in spec.extensions {
            assert!(
                row.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)),
                "{}: Rust lists {ext} but the Python connector does not ({:?})",
                row.id,
                row.extensions
            );
        }
    }
}

#[test]
fn describe_and_convert_through_the_real_python_package() {
    let Some(bridge) = real_bridge() else { return };
    let dir = tempfile::tempdir().unwrap();
    let vtu = write(
        dir.path(),
        "tet.vtu",
        br#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 1 0 0 0 1 0 0 0 1</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
      <PointData>
        <DataArray type="Float32" Name="U" NumberOfComponents="3" format="ascii">1 0 0 1 0 0 1 0 0 1 0 0</DataArray>
      </PointData>
      <CellData>
        <DataArray type="Float32" Name="p" format="ascii">101325</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#,
    );
    assert_eq!(best(&vtu).0, "vtk");
    let reg = ConnectorRegistry::new()
        .with_mode(DescribeMode::Python)
        .with_bridge(bridge.clone());
    let sample = reg.open(&vtu).unwrap();
    let meta = sample.metadata();
    assert_eq!((meta.mesh.n_points, meta.mesh.n_cells), (4, 1));
    assert_eq!(meta.mesh.bounds, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let names: Vec<&str> = meta.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"U") && names.contains(&"p"));
    assert_eq!(meta.attributes["zones"][0]["cell_types"]["tetra"], 1);

    let out = dir.path().join("dataset");
    let report = reg.convert(&vtu, &out, &ConvertOptions::default()).unwrap();
    assert_eq!(report.sample_ids, vec!["tet".to_string()]);
    assert!(out.join("manifest.json").is_file());
    assert!(out.join("samples/tet/mesh.zarr/zarr.json").is_file());

    let err = reg
        .describe(&dir.path().join("missing.vtu"), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not exist"), "{err}");
}

#[test]
fn plugin_state_round_trip_through_the_real_python_package() {
    use crate::plugins::{PluginStatus, PluginTarget};
    let Some(bridge) = real_bridge() else { return };
    let home = tempfile::tempdir().unwrap();
    let bridge = bridge.with_env("FRONTIER_HOME", home.path().to_string_lossy());

    let list = bridge.plugins(true).unwrap();
    assert_eq!(list.data_model_version, "1.1");
    assert!(list.state_file.starts_with(home.path()));
    let vtk = list.get("vtk").expect("the vtk plugin is installed");
    assert!(vtk.enabled && vtk.first_party && vtk.connectors == ["vtk"]);
    // every connector of the Python registry comes from a listed plugin
    for row in bridge.list().unwrap() {
        let provider = list.provider_of(&row.id);
        assert_eq!(provider.map(|p| p.id.as_str()), Some(row.plugin.as_str()));
    }

    let done = bridge
        .set_plugin_enabled(&[PluginTarget::Plugin("palabos".into())], false)
        .unwrap();
    assert!(done[0].changed && home.path().join("plugins.toml").is_file());
    let after = bridge.plugins(false).unwrap();
    assert_eq!(after.get("palabos").unwrap().status, PluginStatus::Disabled);
    assert!(bridge.list().unwrap().iter().all(|r| r.id != "palabos"));
    bridge
        .set_plugin_enabled(&[PluginTarget::Plugin("palabos".into())], true)
        .unwrap();
    assert!(bridge.list().unwrap().iter().any(|r| r.id == "palabos"));

    let doctor = bridge.plugin_doctor().unwrap();
    assert!(doctor.plugins.iter().any(|p| p.info.id == "vtk" && p.ok));
    let exporters = bridge.exporters().unwrap();
    assert!(exporters.iter().any(|e| e.vendor == "catalyst"));
    let copied = bridge
        .install_exporter("catalyst", home.path(), false)
        .unwrap();
    assert!(copied.path.join("frontier_emit.py").is_file());
    let removed = bridge.uninstall_exporter("catalyst", None, false).unwrap();
    assert!(removed.kept_modified.is_empty() && !copied.path.exists());
}
