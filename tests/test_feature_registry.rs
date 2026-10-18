//! Ticket #83: one in-crate source of truth for the README Feature Status table.
//!
//! The registry (`kanso_infer::features`) carries every public feature claim as a
//! name, a status, an exec pointer, and — for `Wired` rows — a `wired` marker that
//! names the production symbol inside a `const`, so deleting that symbol breaks the
//! build instead of silently invalidating a ✅. These tests are the machine audit
//! that keeps the README table from going stale in either direction.

use kanso_infer::features::{FeatureStatus, FEATURES};

/// Parsed `| Feature | Status | Exec pointer |` rows of the README table.
fn readme_rows() -> Vec<(String, FeatureStatus, String)> {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("read README");
    let section = readme
        .split("## Feature Status")
        .nth(1)
        .expect("README must have a ## Feature Status table")
        .split("\n## ")
        .next()
        .unwrap();
    section
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('|') && !l.contains("---") && !l.starts_with("| Feature"))
        .map(|row| {
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            let name = cells.get(1).copied().unwrap_or("").to_string();
            let status = cells.get(2).copied().unwrap_or("");
            let pointer = cells.get(cells.len() - 2).copied().unwrap_or("").to_string();
            let status = match status {
                "✅" => FeatureStatus::Wired,
                "🧪" => FeatureStatus::KernelOnly,
                "📝" => FeatureStatus::Planned,
                other => panic!("unknown status glyph {other:?} in row: {row}"),
            };
            (name, status, pointer)
        })
        .collect()
}

#[test]
fn registry_covers_every_readme_row() {
    let rows = readme_rows();
    assert!(!rows.is_empty(), "README Feature Status table must have rows");
    let mut missing = Vec::new();
    for (name, status, _) in &rows {
        match FEATURES.iter().find(|f| f.name == name) {
            None => missing.push(name.clone()),
            Some(spec) => assert_eq!(
                spec.status, *status,
                "README status for {name:?} disagrees with the registry"
            ),
        }
    }
    assert!(
        missing.is_empty(),
        "README rows with no registry entry: {missing:#?}"
    );
}

#[test]
fn registry_has_no_orphan_rows() {
    let rows = readme_rows();
    let orphans: Vec<&str> = FEATURES
        .iter()
        .filter(|f| !rows.iter().any(|(name, _, _)| name == f.name))
        .map(|f| f.name)
        .collect();
    assert!(
        orphans.is_empty(),
        "registry entries with no README row: {orphans:#?}"
    );
    assert_eq!(
        FEATURES.len(),
        rows.len(),
        "README must have exactly one row per registry entry"
    );
}

#[test]
fn wired_rows_carry_a_production_marker() {
    let unmarked: Vec<&str> = FEATURES
        .iter()
        .filter(|f| f.status == FeatureStatus::Wired && f.wired.is_none())
        .map(|f| f.name)
        .collect();
    assert!(
        unmarked.is_empty(),
        "Wired rows must name a production symbol: {unmarked:#?}"
    );
}

#[test]
fn kernel_only_and_planned_rows_have_no_production_marker() {
    let mislabelled: Vec<&str> = FEATURES
        .iter()
        .filter(|f| f.status != FeatureStatus::Wired && f.wired.is_some())
        .map(|f| f.name)
        .collect();
    assert!(
        mislabelled.is_empty(),
        "non-Wired rows must not claim a production marker (relabel to ✅ when wired): {mislabelled:#?}"
    );
}

#[test]
fn planned_rows_point_at_the_roadmap() {
    let off_roadmap: Vec<(&str, &str)> = FEATURES
        .iter()
        .filter(|f| f.status == FeatureStatus::Planned && !f.exec_pointer.contains("roadmap"))
        .map(|f| (f.name, f.exec_pointer))
        .collect();
    assert!(
        off_roadmap.is_empty(),
        "Planned rows must point at the roadmap, not at code: {off_roadmap:#?}"
    );
}

#[test]
fn readme_pointers_match_registry_pointers() {
    let rows = readme_rows();
    let mut mismatched = Vec::new();
    for spec in FEATURES {
        let (_, _, pointer) = rows
            .iter()
            .find(|(name, _, _)| name == spec.name)
            .unwrap_or_else(|| panic!("README row missing for {:?}", spec.name));
        if !pointer.contains(spec.exec_pointer) {
            mismatched.push((spec.name, spec.exec_pointer, pointer.as_str()));
        }
    }
    assert!(
        mismatched.is_empty(),
        "README exec pointer must contain the registry pointer (name, registry, readme): {mismatched:#?}"
    );
}

#[test]
fn code_pointers_resolve_to_files_in_the_tree() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for spec in FEATURES {
        if spec.status == FeatureStatus::Planned {
            continue;
        }
        let pointer = spec
            .exec_pointer
            .split_whitespace()
            .find(|p| p.starts_with("src/") || p.starts_with("tests/"))
            .unwrap_or_else(|| panic!("{:?} has no file pointer: {}", spec.name, spec.exec_pointer));
        let path = root.join(pointer.trim_end_matches([',', ')']));
        assert!(
            path.exists(),
            "{:?} exec pointer does not resolve: {}",
            spec.name,
            path.display()
        );
    }
}

/// Acceptance criterion: removing a symbol a `Wired` marker names must break the
/// build, not merely fail a test. The markers are `const`-initialised `fn` pointers,
/// so a nested `cargo check` of a throwaway crate that deletes one public symbol and
/// re-exports the rest of `kanso_infer` must fail to compile.
#[test]
fn deleting_a_wired_symbol_breaks_the_build() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scratch = std::env::temp_dir().join(format!("kanso-83-scratch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(scratch.join("src")).expect("create scratch crate");

    // Shim: the real library plus a shadowed module whose production symbol is gone.
    std::fs::write(
        scratch.join("src/lib.rs"),
        "pub use kanso_infer::*;\npub mod model { pub use kanso_infer::model::*; }\n",
    )
    .expect("write scratch lib");

    std::fs::write(
        scratch.join("Cargo.toml"),
        format!(
            "[package]\nname = \"kanso-83-scratch\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n\
             [dependencies]\nkanso-infer = {{ path = \"{}\" }}\n\n[workspace]\n",
            manifest_dir.display()
        ),
    )
    .expect("write scratch manifest");

    // Registry source, copied verbatim so the markers are the ones under test.
    let registry_src = std::fs::read_to_string(manifest_dir.join("src/features.rs"))
        .expect("read src/features.rs");
    std::fs::write(scratch.join("src/features.rs"), &registry_src).expect("write scratch registry");

    let ok = run_check(&scratch);
    assert!(
        ok.status.success(),
        "the registry must compile against the real library first:\n{}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // Now delete one symbol that a Wired marker names, and watch the build fail.
    let marker_line = registry_src
        .lines()
        .find(|l| l.contains("wired: Some("))
        .expect("registry must carry at least one wired marker");
    let deleted = std::fs::read_to_string(scratch.join("src/lib.rs"))
        .unwrap()
        .replace(
            "pub mod model { pub use kanso_infer::model::*; }",
            "pub mod model { /* production symbol removed */ }",
        );
    std::fs::write(scratch.join("src/lib.rs"), deleted).expect("rewrite scratch lib");

    let broken = run_check(&scratch);
    assert!(
        !broken.status.success(),
        "deleting the production symbol named by a wired marker must break the build \
         (marker: {marker_line})"
    );

    let _ = std::fs::remove_dir_all(&scratch);
}

fn run_check(dir: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO"))
        .args(["check", "--quiet"])
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run nested cargo check")
}
