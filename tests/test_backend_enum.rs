mod common;

use candle_core::Device;
use kanso_infer::hardware::Backend;
use kanso_infer::model::Model;

#[test]
fn test_backend_foo_rejected() {
    let err = "foo".parse::<Backend>().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("foo"),
        "error must name the bad value, got: {msg}"
    );
    for (s, want) in [
        ("cpu", Backend::Cpu),
        ("metal", Backend::Metal),
        ("neon", Backend::Neon),
        ("amx", Backend::Amx),
        ("CPU", Backend::Cpu),
        ("Metal", Backend::Metal),
    ] {
        assert_eq!(s.parse::<Backend>().unwrap(), want, "parse {s:?}");
    }
}

/// Ticket #89: `--backend neon` used to resolve to `Device::Cpu` and then run the
/// exact same candle CPU matmul as `--backend cpu`, while printing "neon" — the
/// `--backend metal`-before-#42 placebo in a different costume.
///
/// The hand-written kernels in `src/quant/neon.rs` cannot read the production
/// `LinearWeight` blocks: `src/quant::BlockQ8_0` is `{ d: f32, qs: [i8; 32] }`
/// (36 B) while candle stores `{ d: f16, qs: [i8; 32] }` (34 B), and
/// `src/quant::BlockQ4_K` keeps `d`/`dmin` as f32 where candle keeps f16. Reading
/// them in place is impossible and repacking every weight on every token would
/// cost more than it saves, so the kernels stay benchmark-only and the backend
/// must refuse instead of silently measuring the CPU path.
#[test]
fn test_neon_backend_refuses_placebo_cpu_run() {
    let err = Backend::Neon
        .device()
        .expect_err("--backend neon must not silently resolve to the CPU path");
    let msg = err.to_string();
    assert!(msg.contains("neon"), "error must name the backend: {msg}");
    assert!(
        msg.contains("kernel"),
        "error must say the NEON kernels are not wired: {msg}"
    );
    // The refusal is specific to neon: cpu/amx still resolve to CPU on purpose.
    assert!(matches!(Backend::Cpu.device().unwrap(), Device::Cpu));
    assert!(matches!(Backend::Amx.device().unwrap(), Device::Cpu));
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_device_is_honest_gpu() {
    let Some(device) = common::metal_device_or_skip() else {
        return;
    };
    assert!(
        matches!(device, Device::Metal(_)),
        "metal backend must resolve to a GPU device, got {device:?}"
    );
    assert_eq!(kanso_infer::hardware::device_label(&device), "Metal GPU");
    assert_eq!(kanso_infer::hardware::device_label(&Device::Cpu), "CPU");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn test_metal_device_unavailable_off_macos() {
    assert!(Backend::Metal.device().is_err());
}

/// Ticket #99: on the `macos-14` CI runner `Device::new_metal(0)` panics inside
/// objc2 (`.expect("class MTLResidencySetDescriptor could not be found")`) instead
/// of returning `Err`. `metal_device()` must report that runtime gap as a plain
/// error, and on a host that does have the class it must still hand back a GPU.
#[cfg(target_os = "macos")]
#[test]
fn test_metal_device_reports_ok_or_a_runtime_gap() {
    match kanso_infer::hardware::metal_device() {
        Ok(dev) => assert!(
            matches!(dev, Device::Metal(_)),
            "metal_device must return a GPU device, got {dev:?}"
        ),
        Err(err) => assert!(
            kanso_infer::hardware::is_metal_runtime_gap(&err),
            "a metal_device failure that is not a runtime gap must not be \
             reported as one: {err}"
        ),
    }
}

/// The classifier is what every test uses to decide "skip" instead of "fail",
/// so it has to match the capability prefix and nothing else.
#[test]
fn test_is_metal_runtime_gap_matches_only_the_capability_prefix() {
    use kanso_infer::error::EngineError;
    use kanso_infer::hardware::{METAL_RUNTIME_GAP_PREFIX, is_metal_runtime_gap};

    let gap = EngineError::Metal(format!(
        "{METAL_RUNTIME_GAP_PREFIX}: class MTLResidencySetDescriptor could not be found"
    ));
    assert!(is_metal_runtime_gap(&gap), "prefixed metal error is a gap");

    for other in [
        EngineError::Metal("metal command buffer failed".to_string()),
        EngineError::Config("unknown backend 'foo'".to_string()),
        EngineError::Model("weights are truncated".to_string()),
        EngineError::Candle(candle_core::Error::Msg("cup failed".to_string())),
    ] {
        assert!(
            !is_metal_runtime_gap(&other),
            "only the capability prefix counts as a gap, got: {other}"
        );
    }
}

/// Ticket #99: `metal_device()` gates on the Darwin release (MTLResidencySet
/// needs macOS 15+), so the parsing feeding that gate must be pinned down
/// without needing a macOS 14 host.
#[test]
fn test_parse_darwin_major_releases() {
    use kanso_infer::hardware::parse_darwin_major;

    for (release, want) in [
        ("14.2.1", 14),
        ("15.0", 15),
        ("23.6.0", 23),
        ("27.0", 27),
        ("15", 15),
    ] {
        assert_eq!(
            parse_darwin_major(release),
            Some(want),
            "parse release {release:?}"
        );
    }
    // Unparseable releases must fail closed: unknown version => no Metal.
    assert_eq!(parse_darwin_major("unknown"), None);
    assert_eq!(parse_darwin_major(""), None);
}

/// This host is macOS 27 (class present), so the version gate must admit it.
#[cfg(target_os = "macos")]
#[test]
fn test_macos_host_passes_the_residency_set_version_gate() {
    use kanso_infer::hardware::{macos_has_residency_set_class, macos_major_version};

    let major = macos_major_version().expect("uname release must parse on macOS");
    assert!(
        major >= 15,
        "this machine is expected to have the class (macOS 15+), got major {major}"
    );
    assert!(
        macos_has_residency_set_class(),
        "macOS {major} must pass the residency-set version gate"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_cpu_parity_mock() {
    let Some(dev) = common::metal_device_or_skip() else {
        return;
    };
    let cpu = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("cpu mock");
    let mut metal = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("metal mock");
    metal.to_device(&dev).expect("move to metal");
    metal.set_backend(Backend::Metal);
    assert!(matches!(metal.device, Device::Metal(_)));

    let prompt = vec![1u32, 5, 42];
    let l_cpu: Vec<f32> = cpu
        .forward_single_prompt(&prompt)
        .expect("cpu forward")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec");
    let l_metal: Vec<f32> = metal
        .forward_single_prompt(&prompt)
        .expect("metal forward")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec");
    assert_eq!(l_cpu.len(), l_metal.len());
    let max_diff = l_cpu
        .iter()
        .zip(&l_metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "metal/cpu logits diverged: {max_diff}");
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_cpu_parity_real_model() {
    let Ok(path) = kanso_infer::weights::resolve_model_path("smollm2-360m-instruct-q8_0.gguf")
    else {
        return;
    };
    let Some(dev) = common::metal_device_or_skip() else {
        return;
    };
    let cpu = Model::load_gguf(&path, Device::Cpu).expect("cpu load");
    let metal = Model::load_gguf(&path, dev).expect("metal load");
    assert!(matches!(metal.device, Device::Metal(_)));

    let prompt = vec![1u32, 5, 42];
    let l_cpu: Vec<f32> = cpu
        .forward_single_prompt(&prompt)
        .expect("cpu forward")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec");
    let l_metal: Vec<f32> = metal
        .forward_single_prompt(&prompt)
        .expect("metal forward")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("to_vec");
    assert_eq!(l_cpu.len(), l_metal.len());
    let max_diff = l_cpu
        .iter()
        .zip(&l_metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "real-model metal/cpu logits diverged: {max_diff}"
    );
}
