//! Ticket #49: compile the Metal kernels ahead of time so startup loads a
//! `.metallib` instead of compiling shaders from source.
//!
//! `xcrun metal` ships with a full Xcode install, not with the Command Line Tools,
//! so a machine without one simply gets no `metal_aot` cfg and the pipeline cache
//! falls back to compiling each shader file once (see `src/metal/pipeline.rs`).

use std::path::PathBuf;
use std::process::Command;

/// Kernel sources, parsed out of `KERNEL_SOURCES` in `src/metal/pipeline.rs`
/// (the `file:` lines) so the AOT list cannot drift from the runtime list.
/// A file listed here without a `file:` entry fails the build below.
const KERNEL_FILES: &[&str] = &[
    "rmsnorm.metal",
    "rope.metal",
    "silu_gate.metal",
    "paged_kv.metal",
    "paged_attention_v2.metal",
    "fused_llama_layer.metal",
];

/// Fails the build when `KERNEL_FILES` drifts from the `file:` lines of
/// `KERNEL_SOURCES` in `src/metal/pipeline.rs`.
fn check_kernel_list_in_sync() {
    let src = std::fs::read_to_string("src/metal/pipeline.rs")
        .expect("reading src/metal/pipeline.rs for the kernel list");
    let mut listed: Vec<&str> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("file: \"")
            && let Some(name) = rest.split('"').next()
        {
            listed.push(name);
        }
    }
    assert!(
        !listed.is_empty(),
        "no `file:` entries parsed out of src/metal/pipeline.rs"
    );
    assert_eq!(
        listed, KERNEL_FILES,
        "build.rs KERNEL_FILES drifted from src/metal/pipeline.rs KERNEL_SOURCES"
    );
}

/// Command sequences (`xcrun -sdk macosx` prefix aside) that turn a `.metal` file
/// into a `.metallib`, tried in order: explicit codegen flags first, then plain
/// defaults, then the documented compile-then-link pair. An older toolchain may
/// reject a flag or the single-step form; whichever sequence works, works, and if
/// none does the build still succeeds without AOT.
fn recipes<'a>(src: &'a str, air: &'a str, dst: &'a str) -> [Vec<Vec<&'a str>>; 3] {
    [
        vec![vec!["metal", "-O2", "-ffast-math", src, "-o", dst]],
        vec![vec!["metal", src, "-o", dst]],
        vec![
            vec!["metal", "-O2", "-ffast-math", "-c", src, "-o", air],
            vec!["metallib", air, "-o", dst],
        ],
    ]
}

fn main() {
    check_kernel_list_in_sync();
    println!("cargo::rustc-check-cfg=cfg(metal_aot)");
    println!("cargo:rerun-if-changed=build.rs");
    for file in KERNEL_FILES {
        println!("cargo:rerun-if-changed=src/metal/kernels/{file}");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let out_dir =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo")).join("kernels");
    std::fs::create_dir_all(&out_dir).expect("creating OUT_DIR/kernels");

    let mut compiled_all = true;
    for file in KERNEL_FILES {
        let name = file.trim_end_matches(".metal");
        let src = format!("src/metal/kernels/{file}");
        let dst = out_dir.join(format!("{name}.metallib"));
        let air = out_dir.join(format!("{name}.air"));
        let dst = dst.to_string_lossy();
        let air = air.to_string_lossy();

        let mut error = String::new();
        let mut compiled = false;
        'recipes: for recipe in recipes(&src, &air, &dst) {
            for argv in recipe {
                match Command::new("xcrun")
                    .args(["-sdk", "macosx"])
                    .args(argv)
                    .output()
                {
                    Ok(out) if out.status.success() => {}
                    Ok(out) => {
                        error = String::from_utf8_lossy(&out.stderr).trim().to_string();
                        continue 'recipes;
                    }
                    Err(e) => {
                        error = e.to_string();
                        continue 'recipes;
                    }
                }
            }
            compiled = true;
            break;
        }

        if !compiled {
            compiled_all = false;
            println!(
                "cargo:warning=metal AOT skipped for {file} (startup will compile it once): {error}"
            );
        }
    }

    if compiled_all {
        println!("cargo:rustc-cfg=metal_aot");
    }
}
