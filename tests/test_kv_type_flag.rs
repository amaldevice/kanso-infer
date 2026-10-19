//! Ticket #88 / spec #82: `--kv-type` must be a *user* choice, not an API only tests
//! can reach.
//!
//! The registry row for quantized KV was `Wired` while the only callers of
//! `Engine::new_with_budget_and_kv` were `Engine::new_with_budget` (always F32) and
//! tests — "no production caller", the exact meaning of 🧪. These tests pin the
//! user-facing path:
//!
//! 1. the CLI parses `--kv-type f32|int8|fp8` (and rejects anything else at parse time);
//! 2. the parsed value maps onto the storage mode the engine builds;
//! 3. `kanso --help` advertises the flag, so it is discoverable rather than secret.
//!
//! The end-to-end wiring itself (flag → `build_engine` → `Engine`) lives in `src/main.rs`
//! and is exercised by `kanso --kv-type int8 --backend cpu`; the binary is not spawned
//! here because that needs a real GGUF on disk.

use kanso_infer::kv_cache::KvCacheType;

#[test]
fn kv_type_parses_the_three_documented_values() {
    for (text, expected) in [
        ("f32", KvCacheType::F32),
        ("int8", KvCacheType::Int8),
        ("fp8", KvCacheType::Fp8),
        ("INT8", KvCacheType::Int8),
    ] {
        let parsed: KvCacheType = text.parse().expect("documented kv type parses");
        assert_eq!(parsed, expected, "'{text}' must parse to {expected:?}");
    }
}

#[test]
fn unknown_kv_type_is_rejected_instead_of_defaulting() {
    let err = "q4"
        .parse::<KvCacheType>()
        .expect_err("q4 is not a kv type");
    let msg = err.to_string();
    assert!(
        msg.contains("f32") && msg.contains("int8") && msg.contains("fp8"),
        "the error must name the accepted values, got: {msg}"
    );
}

#[test]
fn kv_type_is_f32_by_default() {
    assert_eq!(KvCacheType::default(), KvCacheType::F32);
    assert_eq!(KvCacheType::default().mode(), None);
}

#[test]
fn kv_type_maps_onto_the_storage_mode() {
    assert_eq!(KvCacheType::F32.mode(), None);
    assert_eq!(
        KvCacheType::Int8.mode(),
        Some(kanso_infer::kv_cache::QuantMode::Int8)
    );
    assert_eq!(
        KvCacheType::Fp8.mode(),
        Some(kanso_infer::kv_cache::QuantMode::Fp8)
    );
}

#[test]
fn kv_type_round_trips_through_its_wire_form() {
    for t in [KvCacheType::F32, KvCacheType::Int8, KvCacheType::Fp8] {
        let json = serde_json::to_string(&t).expect("serialize");
        assert_eq!(json, format!("\"{t}\""));
        let back: KvCacheType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, t);
    }
}

/// The flag must be advertised, not hidden: a user cannot select what `--help` omits.
#[test]
fn kanso_help_advertises_kv_type() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kanso"))
        .args(["serve", "--help"])
        .output()
        .expect("run kanso serve --help");
    assert!(out.status.success(), "`kanso serve --help` must succeed");
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        help.contains("--kv-type"),
        "`serve --help` must advertise --kv-type, got:\n{help}"
    );
    assert!(
        help.contains("int8") && help.contains("fp8"),
        "`serve --help` must name the quantized options, got:\n{help}"
    );
}
