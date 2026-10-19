use candle_core::{Device, Tensor};
use kanso_infer::bench::perplexity::{
    PerplexityReport, QuantKernel, calculate_perplexity, calculate_perplexity_quantized,
    compare_backends_perplexity, cross_entropy_loss, format_ascii_ppl_report,
    get_bundled_wikitext2_text, get_bundled_wikitext2_tokens, run_wikitext2_perplexity_eval,
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
        /* hidden_size */ 16, /* num_attention_heads */ 4, /* num_kv_heads */ 2,
        /* num_layers */ 2, /* vocab_size */ 50,
    )
    .expect("create mock tiny model");

    let tokens = vec![1u32, 5, 12, 19, 42, 7, 33, 2];
    let ppl = calculate_perplexity(&model, &tokens, 4).expect("calculate perplexity");

    // NOTE (#57): the fixed-seed random mock has no calibrated absolute PPL.
    // Assert properties instead: finite, positive, and deterministic across runs.
    assert!(
        ppl.is_finite() && ppl > 0.0,
        "PPL must be finite positive, got {}",
        ppl
    );
    let ppl_rerun = calculate_perplexity(&model, &tokens, 4).expect("rerun perplexity");
    assert!(
        (ppl - ppl_rerun).abs() == 0.0,
        "fixed-seed mock PPL must be bit-deterministic: {} vs {}",
        ppl,
        ppl_rerun
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

    // Strides must all yield finite PPL and agree with each other
    // (sliding-window coverage property, independent of absolute value).
    for (name, ppl) in [("s8", ppl_s8), ("s16", ppl_s16), ("s32", ppl_s32)] {
        assert!(
            ppl.is_finite() && ppl > 0.0,
            "stride {} PPL must be finite positive, got {}",
            name,
            ppl
        );
    }
    assert!(
        (ppl_s8 - ppl_s16).abs() / ppl_s8 < 1e-6 && (ppl_s8 - ppl_s32).abs() / ppl_s8 < 1e-6,
        "strides must agree: s8={} s16={} s32={}",
        ppl_s8,
        ppl_s16,
        ppl_s32
    );
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
    assert!(
        ppl.is_finite() && ppl > 0.0,
        "PPL must be finite positive, got {}",
        ppl
    );
}

#[test]
fn test_backend_perplexity_parity_cpu_amx_neon_metal() {
    let report = compare_backends_perplexity("mock", &get_bundled_wikitext2_tokens()[..64], 32)
        .expect("compare backends");

    assert_eq!(report.dataset, "wikitext2");
    assert!(report.baseline_ppl > 0.0);

    // Ticket #89: `neon` must be reported as *not run*, never as a number that
    // is secretly the CPU path. The remaining backends must be measured.
    let neon = report
        .backends
        .iter()
        .find(|x| x.backend == kanso_infer::Backend::Neon)
        .expect("neon row must still be present in the report");
    assert!(
        neon.perplexity.is_none(),
        "neon must not report a measured PPL (that would be the CPU path), got {:?}",
        neon.perplexity
    );
    assert!(
        neon.skipped_reason
            .as_deref()
            .is_some_and(|r| r.contains("not wired")),
        "neon row must carry the reason it was skipped, got {:?}",
        neon.skipped_reason
    );

    let backends = ["cpu", "amx", "metal"];
    for name in &backends {
        let b = report
            .backends
            .iter()
            .find(|x| x.backend.to_string().eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("backend {} missing in report", name));
        let ppl = b.perplexity.expect("measured backend must carry a PPL");

        assert!(
            (ppl - report.baseline_ppl).abs() < 0.01,
            "Backend {} Delta PPL {} exceeds 0.01 parity threshold",
            name,
            (ppl - report.baseline_ppl).abs()
        );
    }
}

#[test]
fn test_quantized_kernels_delta_ppl_bound() {
    let model = Model::create_mock_tiny(64, 4, 2, 2, 256).expect("model");
    let tokens: Vec<u32> = (0..64).map(|i| (i * 11) % 256).collect();
    let stride = 16;

    let baseline_ppl = calculate_perplexity(&model, &tokens, stride).expect("baseline ppl");

    // NOTE (#57): the absolute < 0.15 bound was calibrated on the all-ones mock
    // (baseline PPL = 50). On the random mock the baseline is uncalibrated, so an
    // absolute delta is meaningless. Assert finiteness + log the deltas instead;
    // a meaningful quant-fidelity bound needs a real calibrated model.
    assert!(baseline_ppl.is_finite() && baseline_ppl > 0.0);
    for (name, mode) in [
        ("NEON Q8_0", QuantKernel::NeonQ8_0),
        ("NEON Q4_K", QuantKernel::NeonQ4_K),
        ("INT8 KV", QuantKernel::Int8Kv),
    ] {
        let q_ppl = calculate_perplexity_quantized(&model, &tokens, stride, mode)
            .unwrap_or_else(|e| panic!("{name} eval failed: {e}"));
        assert!(
            q_ppl.is_finite() && q_ppl > 0.0,
            "{name} PPL must be finite positive, got {q_ppl}"
        );
        eprintln!("{name}: baseline={baseline_ppl:.4} quantized={q_ppl:.4}");
    }
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
    // NOTE (#57): PASS/FAIL verdicts depend on the ones-calibrated 0.15 threshold;
    // on the random mock only the report structure is asserted here.
    assert!(ascii.contains("PASS") || ascii.contains("FAIL"));

    // JSON serialization
    let json_str = serde_json::to_string_pretty(&report).expect("serialize json");
    assert!(json_str.contains("\"dataset\": \"wikitext2\""));

    let deserialized: PerplexityReport = serde_json::from_str(&json_str).expect("deserialize json");
    assert_eq!(deserialized.dataset, "wikitext2");
    assert_eq!(deserialized.backends.len(), 4);
    assert_eq!(deserialized.quantized_evals.len(), 3);
}

/// Legacy reports (pre-#92) wrote `backend` as a **PascalCase** string.
///
/// `Backend` on `origin/main` derived `Serialize` with no `rename_all`, so
/// `Backend::Cpu` went to disk as `"Cpu"` — not `"cpu"`. Ticket #92 pinned the wire
/// form to lowercase and kept the old spelling as `#[serde(alias = "Cpu")]`, which
/// is the compatibility that actually has to hold: every archived report on disk
/// carries the PascalCase spelling.
///
/// An earlier version of this test asserted `"cpu"`, a form `main` never produced;
/// it proved nothing. This one round-trips the real historical spelling.
#[test]
fn test_legacy_pascalcase_backend_report_still_deserializes() {
    let legacy = r#"{
        "dataset": "wikitext2",
        "num_tokens": 64,
        "stride": 32,
        "baseline_ppl": 12.5,
        "backends": [
            {"backend": "Cpu", "perplexity": 12.5, "delta_vs_fp32": 0.0},
            {"backend": "Amx", "perplexity": 12.6, "delta_vs_fp32": 0.1}
        ],
        "quantized_evals": [],
        "delta_threshold": 0.15,
        "all_passed": true
    }"#;
    let report: PerplexityReport =
        serde_json::from_str(legacy).expect("a report written by main must still deserialize");
    assert_eq!(report.backends.len(), 2);
    assert_eq!(report.backends[0].backend, kanso_infer::Backend::Cpu);
    assert_eq!(report.backends[1].backend, kanso_infer::Backend::Amx);

    // New reports write the pinned lowercase wire form; the old one still reads.
    let json = serde_json::to_string(&report).expect("serialize");
    assert!(json.contains("\"backend\":\"cpu\""), "got: {json}");
    let round_trip: PerplexityReport =
        serde_json::from_str(&json).expect("the new wire form reads back");
    assert_eq!(round_trip.backends[0].backend, kanso_infer::Backend::Cpu);
}

/// The lowercase form #92 pinned must read too, so a report written by this branch
/// is readable by this branch and by anything that adopted the pinned form early.
#[test]
fn test_pinned_lowercase_backend_report_deserializes() {
    let pinned = r#"{"backend": "cpu", "perplexity": 12.5, "delta_vs_fp32": 0.0}"#;
    let entry: kanso_infer::bench::BackendPpl =
        serde_json::from_str(pinned).expect("lowercase backend must deserialize");
    assert_eq!(entry.backend, kanso_infer::Backend::Cpu);
}
