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

#[cfg(target_os = "macos")]
#[test]
fn test_metal_device_is_honest_gpu() {
    let device = Backend::Metal.device().expect("metal device");
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

#[cfg(target_os = "macos")]
#[test]
fn test_metal_cpu_parity_mock() {
    let cpu = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("cpu mock");
    let mut metal = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("metal mock");
    metal
        .to_device(&Backend::Metal.device().expect("metal device"))
        .expect("move to metal");
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
    let cpu = Model::load_gguf(&path, Device::Cpu).expect("cpu load");
    let metal = Model::load_gguf(&path, Backend::Metal.device().expect("metal device"))
        .expect("metal load");
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
