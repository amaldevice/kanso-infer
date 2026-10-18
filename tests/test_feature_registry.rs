//! Ticket #83: one in-crate source of truth for the README Feature Status table.
//!
//! The registry (`kanso_infer::features`) carries every public feature claim as a
//! name, a status, an exec pointer, and — for `Wired` rows — a `wired` marker that
//! names the production symbol inside a `const`, so deleting that symbol breaks the
//! build instead of silently invalidating a ✅. These tests are the machine audit
//! that keeps the README table from going stale in either direction.

use kanso_infer::features::{FEATURES, FeatureStatus};

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
            let pointer = cells
                .get(cells.len() - 2)
                .copied()
                .unwrap_or("")
                .to_string();
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
    assert!(
        !rows.is_empty(),
        "README Feature Status table must have rows"
    );
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
    let mut mismatched: Vec<(&str, &str, &str, Vec<&str>)> = Vec::new();
    for spec in FEATURES {
        let (_, _, pointer) = rows
            .iter()
            .find(|(name, _, _)| name == spec.name)
            .unwrap_or_else(|| panic!("README row missing for {:?}", spec.name));
        let missing: Vec<&str> = spec
            .exec_pointer
            .split_whitespace()
            .filter(|part| !pointer.contains(part))
            .collect();
        if !missing.is_empty() {
            mismatched.push((spec.name, spec.exec_pointer, pointer.as_str(), missing));
        }
    }
    assert!(
        mismatched.is_empty(),
        "README exec pointer must name every registry pointer (name, registry, readme, missing): {mismatched:#?}"
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
            .unwrap_or_else(|| {
                panic!("{:?} has no file pointer: {}", spec.name, spec.exec_pointer)
            });
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
/// so a throwaway crate that re-exports only the symbols the markers name must stop
/// compiling once one of those symbols is taken away.
#[test]
fn deleting_a_wired_symbol_breaks_the_build() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let registry_src = std::fs::read_to_string(manifest_dir.join("src/features.rs"))
        .expect("read src/features.rs");
    let markers = wired_markers(&registry_src);
    assert!(
        markers.len() >= FEATURES.iter().filter(|f| f.wired.is_some()).count(),
        "every wired marker must be discoverable in the source: {markers:#?}"
    );

    let scratch = std::env::temp_dir().join("kanso-83-scratch");
    std::fs::create_dir_all(scratch.join("src")).expect("create scratch crate");
    std::fs::write(
        scratch.join("Cargo.toml"),
        format!(
            "[package]\nname = \"kanso-83-scratch\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n\
             [dependencies]\nkanso-infer = {{ path = \"{}\" }}\n\n[workspace]\n",
            manifest_dir.display()
        ),
    )
    .expect("write scratch manifest");
    std::fs::write(scratch.join("src/features.rs"), &registry_src).expect("write scratch registry");

    // The shim exposes exactly the symbols the markers name — nothing else.
    let full_shim = marker_shim(&markers);
    std::fs::write(scratch.join("src/lib.rs"), &full_shim).expect("write scratch lib");
    let ok = run_check(&scratch);
    assert!(
        ok.status.success(),
        "registry must compile against the real library:\n{}\nshim:\n{full_shim}",
        String::from_utf8_lossy(&ok.stderr)
    );

    // Delete one production symbol the registry points at.
    let (deleted_marker, deleted_line) = markers
        .iter()
        .map(|m| (m, reexport_line(m)))
        .find(|(_, line)| full_shim.contains(line.as_str()))
        .expect("a marker must have a matching re-export");
    let without_symbol = full_shim.replacen(deleted_line.as_str(), "", 1);
    assert_ne!(without_symbol, full_shim);
    std::fs::write(scratch.join("src/lib.rs"), &without_symbol).expect("rewrite scratch lib");

    let broken = run_check(&scratch);
    let stderr = String::from_utf8_lossy(&broken.stderr).to_string();
    assert!(
        !broken.status.success(),
        "deleting {deleted_marker} (named by a wired marker) must break the build"
    );
    assert!(
        stderr.contains("unresolved")
            || stderr.contains("not found")
            || stderr.contains("E0432")
            || stderr.contains("E0433"),
        "the failure must be a resolution error, not something unrelated:\n{stderr}"
    );
}

/// Marker paths (`crate::a::b::Item::method`) declared inside `wired!(...)`, deduped.
fn wired_markers(registry_src: &str) -> Vec<String> {
    let mut found = std::collections::BTreeSet::new();
    let mut cursor = 0;
    while let Some(offset) = registry_src[cursor..].find("crate::") {
        let start = cursor + offset;
        let end = registry_src[start..]
            .char_indices()
            .take_while(|(_, c)| c.is_alphanumeric() || *c == ':' || *c == '_')
            .map(|(i, c)| start + i + c.len_utf8())
            .last()
            .unwrap_or(start);
        found.insert(registry_src[start..end].to_string());
        cursor = end.max(start + 1);
    }
    found.into_iter().collect()
}

/// Splits a marker into the module chain and the item the shim must re-export.
///
/// Rust naming makes this unambiguous here: a type is CamelCase and a module is
/// snake_case, so `crate::a::b::Item::method` re-exports `Item` from `a::b` while
/// `crate::a::b::free_fn` re-exports `free_fn` from `a::b`.
fn marker_parts(marker: &str) -> (Vec<String>, String) {
    let segments: Vec<String> = marker.split("::").map(str::to_string).collect();
    assert!(segments.len() >= 3, "marker too short: {marker}");
    let is_type = |s: &String| s.chars().next().is_some_and(|c| c.is_uppercase());
    match segments.iter().rposition(is_type) {
        Some(i) => (segments[1..i].to_vec(), segments[i].clone()),
        None => (
            segments[1..segments.len() - 1].to_vec(),
            segments[segments.len() - 1].clone(),
        ),
    }
}

fn reexport_line(marker: &str) -> String {
    let (modules, item) = marker_parts(marker);
    format!("pub use kanso_infer::{}::{item};", modules.join("::"))
}

/// Nested `pub mod` chain exposing one re-export per marker, and nothing more.
fn marker_shim(markers: &[String]) -> String {
    // module chain -> the distinct items re-exported from it.
    let mut tree: std::collections::BTreeMap<Vec<String>, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for marker in markers {
        let (modules, _) = marker_parts(marker);
        tree.entry(modules)
            .or_default()
            .insert(reexport_line(marker));
    }
    let mut out = String::from("#![allow(unused_imports)]\n");
    let mut closed: Vec<String> = Vec::new();
    for (modules, items) in &tree {
        let shared = modules
            .iter()
            .zip(&closed)
            .take_while(|(a, b)| a == b)
            .count();
        for name in &closed[shared..] {
            out.push_str("} ");
            let _ = name;
        }
        closed.truncate(shared);
        for (depth, module) in modules.iter().enumerate().skip(shared) {
            out.push_str(&format!("{}pub mod {module} {{ ", "    ".repeat(depth)));
            closed.push(module.clone());
        }
        for item in items {
            out.push_str(item);
            out.push(' ');
        }
        if modules.is_empty() {
            out.push('\n');
        }
    }
    for _ in 0..closed.len() {
        out.push_str("}\n");
    }
    // The registry under test must actually be compiled by the scratch crate.
    out.push_str("pub mod features;\n");
    out
}

fn run_check(dir: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO"))
        .args(["check", "--quiet"])
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run nested cargo check")
}

/// The Roadmap section is where a `Planned` row's reason lives; every planned claim
/// must be explained there, so a demotion cannot be a bare glyph flip.
#[test]
fn every_planned_row_is_explained_in_the_roadmap() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("read README");
    let roadmap = readme
        .split("## Roadmap")
        .nth(1)
        .expect("README must have a Roadmap section")
        .split("\n## ")
        .next()
        .unwrap();
    let unexplained: Vec<&str> = FEATURES
        .iter()
        .filter(|f| {
            let label = f.name.split(" (").next().unwrap_or(f.name);
            f.status == FeatureStatus::Planned && !roadmap.contains(label)
        })
        .map(|f| f.name)
        .collect();
    assert!(
        unexplained.is_empty(),
        "planned rows need a roadmap bullet explaining why they are not code yet: {unexplained:#?}"
    );
}

/// Ticket #83 demotes two rows from 🧪 to 📝. Both demotions must say *why* in the
/// README, and the loader arch gate that rejects hybrid GGUF must still be in place.
#[test]
fn hybrid_stub_and_mock_bench_profiles_are_planned_with_a_reason() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("read README");
    let rows = readme_rows();
    let status_of = |needle: &str| {
        rows.iter()
            .find(|(name, _, _)| name.contains(needle))
            .unwrap_or_else(|| panic!("README must have a row containing {needle:?}"))
            .1
    };
    assert_eq!(
        status_of("Hybrid SSM/conv"),
        FeatureStatus::Planned,
        "hybrid SSM/conv is a stub with no forward-pass integration, not kernel-only"
    );
    assert_eq!(
        status_of("bench profiles"),
        FeatureStatus::Planned,
        "granite-h/lfm bench profiles are mock shapes, not loaded weights"
    );

    let roadmap = readme.split("## Roadmap").nth(1).expect("Roadmap section");
    for reason in ["no integration into the forward pass", "mock"] {
        assert!(
            roadmap.contains(reason),
            "demoted rows must state why in the roadmap, missing {reason:?}"
        );
    }
    assert!(
        roadmap.contains("loader arch gate still rejects hybrid GGUF"),
        "demoting hybrid must not lift the loader arch gate"
    );
}
