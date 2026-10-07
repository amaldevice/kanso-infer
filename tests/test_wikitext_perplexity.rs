use candle_core::{Device, Tensor};
use kanso_infer::bench::perplexity::{
    calculate_perplexity, calculate_perplexity_quantized, compare_backends_perplexity,
    cross_entropy_loss, format_ascii_ppl_report, get_bundled_wikitext2_text,
    get_bundled_wikitext2_tokens, run_wikitext2_perplexity_eval, tokenize_wikitext2,
    PerplexityReport, QuantKernel, WIKITEXT2_FIXTURE_TEXT,
};
use kanso_infer::model::Model;

#[test]
fn test_cross_entropy_loss_numerical_properties() {
    let dev = Device::Cpu;

    // 1. Uniform logits over vocab size 4:
    // P(y) = 1/4 = 0.25 => NLL = -ln(0.25) = ln(4) = 1.38629436...
    let uniform = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &dev).unwrap();
    let loss_uniform = cross_entropy_loss(&uniform, 2);
    let expected_uniform = 4.0f64.ln();
    assert!(
        (loss_uniform - expected_uniform).abs() < 1e-6,
        "Expected uniform loss {}, got {}",
        expected_uniform,
        loss_uniform
    );

    // 2. High confidence target logit:
    // Logit for target is much higher than others => loss approaches 0.
    let confident = Tensor::new(&[0.0f32, 50.0, 0.0, 0.0], &dev).unwrap();
    let loss_confident = cross_entropy_loss(&confident, 1);
    assert!(
        loss_confident < 1e-4,
        "Expected near-zero loss for high confidence, got {}",
        loss_confident
    );

    // 3. Low confidence / wrong prediction:
    let wrong = Tensor::new(&[50.0f32, 0.0, 0.0, 0.0], &dev).unwrap();
    let loss_wrong = cross_entropy_loss(&wrong, 1);
    assert!(
        loss_wrong > 40.0,
        "Expected high loss for wrong prediction, got {}",
        loss_wrong
    );

    // 4. Numerical stability with large logits (no overflow / NaN)
    let large = Tensor::new(&[1000.0f32, 1000.0, 1000.0, 1000.0], &dev).unwrap();
    let loss_large = cross_entropy_loss(&large, 0);
    assert!(
        !loss_large.is_nan() && !loss_large.is_infinite(),
        "Loss must not be NaN or Inf"
    );
    assert!((loss_large - expected_uniform).abs() < 1e-5);

    // 5. 2D tensor shape [1, vocab_size]
    let two_d = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &dev).unwrap();
    let loss_2d = cross_entropy_loss(&two_d, 3);
    assert!((loss_2d - expected_uniform).abs() < 1e-6);
}

#[test]
fn test_calculate_perplexity_sliding_window_toy() {
    // Mock tiny model with vocab_size 50
    let model = Model::create_mock_tiny(
        /* hidden_size */ 16,
        /* num_attention_heads */ 4,
        /* num_kv_heads */ 2,
        /* num_layers */ 2,
        /* vocab_size */ 50,
    )
    .expect("create mock tiny model");

    let tokens = vec![1u32, 5, 12, 19, 42, 7, 33, 2];
    let ppl = calculate_perplexity(&model, &tokens, 4).expect("calculate perplexity");

    // For mock model with all ones weights, every vocab logit is identical,
    // so loss per token is ln(vocab_size) = ln(50) ~= 3.9120
    // perplexity = exp(ln(50)) = 50.0
    assert!(
        (ppl - 50.0).abs() < 1e-3,
        "Perplexity for mock model with uniform logits should be 50.0, got {}",
        ppl
    );

    // Perplexity on coherent language text must be in reasonable range
    let nll = ppl.ln();
    assert!(
        nll >= 3.0 && nll <= 6.0,
        "NLL {} must be within [3.0, 6.0]",
        nll
    );
}

#[test]
fn test_sliding_window_strides_coverage() {
    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("create mock model");
    // Long token sequence (e.g. 64 tokens) with max_seq_len 128
    let tokens: Vec<u32> = (0..64).map(|i| (i * 7) % 50).collect();

    // Test different strides: 8, 16, 32
    let ppl_s8 = calculate_perplexity(&model, &tokens, 8).expect("stride 8");
    let ppl_s16 = calculate_perplexity(&model, &tokens, 16).expect("stride 16");
    let ppl_s32 = calculate_perplexity(&model, &tokens, 32).expect("stride 32");

    assert!((ppl_s8 - 50.0).abs() < 1e-3);
    assert!((ppl_s16 - 50.0).abs() < 1e-3);
    assert!((ppl_s32 - 50.0).abs() < 1e-3);
}

#[test]
fn test_bundled_wikitext2_fixture_evaluation() {
    let fixture_text = get_bundled_wikitext2_text();
    assert!(!fixture_text.is_empty(), "Fixture text must not be empty");
    assert!(
        fixture_text.contains("Valkyria") || fixture_text.contains("WikiText"),
        "Fixture text should contain sample WikiText content"
    );

    let tokens = get_bundled_wikitext2_tokens();
    assert!(
        tokens.len() >= 128,
        "Bundled fixture must yield at least 128 tokens, got {}",
        tokens.len()
    );

    let model = Model::create_mock_tiny(16, 4, 2, 2, 50).expect("model");
    let ppl = calculate_perplexity(&model, &tokens[..128], 32).expect("eval ppl");
    assert!(ppl > 0.0);
    let nll = ppl.ln();
    assert!(nll >= 3.0 && nll <= 6.0, "NLL must be in [3.0, 6.0]");
}

#[test]
fn test_backend_perplexity_parity_cpu_amx_neon_metal() {
    let report = compare_backends_perplexity("mock", &get_bundled_wikitext2_tokens()[..64], 32)
        .expect("compare backends");

    assert_eq!(report.dataset, "wikitext2");
    assert!(report.baseline_ppl > 0.0);

    let backends = ["cpu", "amx", "neon", "metal"];
    for name in &backends {
        let b = report
            .backends
            .iter()
            .find(|x| x.backend.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("backend {} missing in report", name));

        assert!(
            b.delta_vs_fp32.abs() < 0.01,
            "Backend {} Delta PPL {} exceeds 0.01 parity threshold",
            name,
            b.delta_vs_fp32
        );
    }
}

#[test]
fn test_quantized_kernels_delta_ppl_bound() {
    let model = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("model");
    let tokens: Vec<u32> = (0..64).map(|i| (i * 11) % 256).collect();
    let stride = 16;

    let baseline_ppl = calculate_perplexity(&model, &tokens, stride).expect("baseline ppl");

    // 1. NEON Q8_0
    let q8_ppl = calculate_perplexity_quantized(&model, &tokens, stride, QuantKernel::NeonQ8_0)
        .expect("q8 ppl");
    let delta_q8 = (q8_ppl - baseline_ppl).abs();
    assert!(
        delta_q8 < 0.15,
        "NEON Q8_0 Delta PPL {} must be < 0.15",
        delta_q8
    );

    // 2. NEON Q4_K
    let q4_ppl = calculate_perplexity_quantized(&model, &tokens, stride, QuantKernel::NeonQ4_K)
        .expect("q4 ppl");
    let delta_q4 = (q4_ppl - baseline_ppl).abs();
    assert!(
        delta_q4 < 0.15,
        "NEON Q4_K Delta PPL {} must be < 0.15",
        delta_q4
    );

    // 3. INT8 KV Cache
    let int8_kv_ppl =
        calculate_perplexity_quantized(&model, &tokens, stride, QuantKernel::Int8Kv)
            .expect("int8 kv ppl");
    let delta_kv = (int8_kv_ppl - baseline_ppl).abs();
    assert!(
        delta_kv < 0.15,
        "INT8 KV Delta PPL {} must be < 0.15",
        delta_kv
    );
}

#[test]
fn test_perplexity_report_ascii_and_json_serialization() {
    let report = run_wikitext2_perplexity_eval("mock", 32).expect("run eval");

    // ASCII table formatting
    let ascii = format_ascii_ppl_report(&report);
    assert!(ascii.contains("WikiText-2"));
    assert!(ascii.contains("cpu"));
    assert!(ascii.contains("amx"));
    assert!(ascii.contains("neon"));
    assert!(ascii.contains("metal"));
    assert!(ascii.contains("NEON Q8_0"));
    assert!(ascii.contains("NEON Q4_K"));
    assert!(ascii.contains("INT8 KV"));
    assert!(ascii.contains("PASS"));

    // JSON serialization
    let json_str = serde_json::to_string_pretty(&report).expect("serialize json");
    assert!(json_str.contains("\"dataset\": \"wikitext2\""));
    assert!(json_str.contains("\"all_passed\": true"));

    let deserialized: PerplexityReport =
        serde_json::from_str(&json_str).expect("deserialize json");
    assert_eq!(deserialized.dataset, "wikitext2");
    assert_eq!(deserialized.all_passed, true);
    assert_eq!(deserialized.backends.len(), 4);
    assert_eq!(deserialized.quantized_evals.len(), 3);
}
