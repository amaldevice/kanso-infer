#[test]
fn test_rmsnorm_numerical_parity() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let eps = 1e-5f32;

    let result = kanso_infer::metal::ops::cpu_rmsnorm(&input, &weight, eps);
    assert_eq!(result.len(), 4);

    // Mean sq = (1 + 4 + 9 + 16)/4 = 7.5
    // rms = sqrt(7.5 + 1e-5) = 2.7386127
    let expected = vec![
        1.0 / 2.7386127,
        2.0 / 2.7386127,
        3.0 / 2.7386127,
        4.0 / 2.7386127,
    ];

    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_silu_gate_multiply_parity() {
    let gate = vec![0.0f32, 1.0];
    let up = vec![2.0f32, 3.0];

    let result = kanso_infer::metal::ops::cpu_silu_gate_multiply(&gate, &up);
    assert_eq!(result.len(), 2);

    // silu(0) * 2 = 0
    assert!((result[0] - 0.0).abs() < 1e-4);
    // silu(1) * 3 = (1 / (1 + exp(-1))) * 3 = 0.7310586 * 3 = 2.1931758
    assert!((result[1] - 2.1931758).abs() < 1e-4);
}

#[test]
fn test_metal_shader_sources_embedded() {
    let shaders = kanso_infer::metal::shaders::get_all_shaders();
    assert!(shaders.contains_key("rmsnorm.metal"));
    assert!(shaders.contains_key("rope.metal"));
    assert!(shaders.contains_key("silu_gate.metal"));
    assert!(shaders.contains_key("paged_kv.metal"));

    for (name, source) in shaders {
        assert!(!source.trim().is_empty(), "Shader {} is empty", name);
        assert!(source.contains("kernel void"), "Shader {} missing kernel definition", name);
    }
}
