//! RED tests for ticket #100: the six F12/F15 budget rows that had no
//! instrument behind them.
//!
//! Each test hands the gate a *synthetic* report that carries one number and
//! then breaks the budget on purpose. If the row still comes back green (or
//! `skipped`), the gate is not reading the report — it is guessing. That is the
//! whole point: a row may only read `measured` when the number it judged came
//! from the report in hand.
//!
//! The real measurement is a separate concern (#95 runs it on a Mac with the
//! GGUF). What is tested here is the plumbing: report field -> gate reading.

use kanso_infer::bench::{
    BenchmarkReport, GateBudget, GateVerdict, KernelProbes, evaluate_gate, evaluate_gate_run,
    evaluate_run_with_timings,
};

/// Weighed synthetic report: 100 MB weighed per decode step, 20 steps. Weighing
/// it keeps the #91 unweighed-report guard from masking the row under test.
fn weighed_report(decode_ms_per_step: f64) -> BenchmarkReport {
    let steps = 20;
    let decode_duration_secs = decode_ms_per_step * (steps as f64) / 1000.0;
    evaluate_run_with_timings(
        100,
        1.0,
        1,
        &[10.0],
        &[5.0],
        10,
        0.1,
        steps,
        decode_duration_secs,
        0xDEAD_BEEF,
        100_000_000,
    )
}

/// A single-row budget, so the shipped file's own statuses cannot decide this
/// test: only the report can.
fn budget_for(metric: &str, bound: &str, value: f64) -> GateBudget {
    GateBudget::from_json_str(&format!(
        r#"{{"name":"probe-{metric}","rows":[
            {{"metric":"{metric}","bound":"{bound}","budget":{value}}}]}}"#
    ))
    .expect("inline budget parses")
}

fn verdict_of(budget: &GateBudget, report: &BenchmarkReport) -> (Option<f64>, GateVerdict) {
    let outcome = evaluate_gate(budget, report).expect("gate evaluates");
    let row = &outcome.rows[0];
    (row.measured, row.verdict.clone())
}

/// `peak_rss_mb` is plumbed from `get_peak_rss_mb()` into every report, and the
/// gate judges that number: an impossible budget (0 MB) must be red.
#[test]
fn gate_reads_peak_rss_from_the_report() {
    let report = weighed_report(10.0);
    let rss = report
        .peak_rss_mb
        .expect("a real process has resident memory; the report must carry it");
    assert!(rss > 0.0, "peak RSS must be a real figure, got {rss}");

    let (measured, verdict) = verdict_of(&budget_for("peak_rss_mb", "max", 0.0), &report);
    assert_eq!(
        measured,
        Some(rss),
        "the gate must judge the report's own RSS, not a constant"
    );
    assert_eq!(
        verdict,
        GateVerdict::Violated,
        "{rss} MB cannot fit a 0 MB budget"
    );

    let (_, generous) = verdict_of(&budget_for("peak_rss_mb", "max", 1e9), &report);
    assert_eq!(generous, GateVerdict::Passed);
}

/// D9: a report written before the field existed must not pass a `max` budget
/// on its default value. Zero is "never measured", not "nothing resident".
#[test]
fn a_legacy_report_cannot_pass_the_peak_rss_budget_on_a_zero() {
    let legacy = r#"{
        "total_tokens": 100, "total_duration_secs": 1.0, "num_requests": 1,
        "total_tokens_per_sec": 100.0, "tokens_per_sec_per_request": 100.0,
        "avg_ms_per_step": 10.0, "mean_ttft_ms": 10.0, "max_ttft_ms": 10.0,
        "ttft_p50_ms": 10.0, "ttft_p90_ms": 10.0, "itl_mean_ms": 5.0,
        "itl_p50_ms": 5.0, "itl_p90_ms": 5.0, "itl_p99_ms": 5.0,
        "prefill_tokens_per_sec": 100.0, "decode_ms_per_step": 10.0,
        "decode_ms_per_token": 10.0, "checksum": 0, "prefix_hit_rate": 0.0,
        "bytes_read_per_step": 100000000, "peak_bandwidth_gb_s": 20.0,
        "peak_bandwidth_source": "measured", "telemetry": "measured"
    }"#;
    let parsed: BenchmarkReport = serde_json::from_str(legacy).expect("legacy report parses");
    assert_eq!(parsed.peak_rss_mb, None, "an absent field is not a zero");

    let (measured, verdict) = verdict_of(&budget_for("peak_rss_mb", "max", 900.0), &parsed);
    assert_eq!(
        measured, None,
        "a report with no RSS has no number to judge"
    );
    assert!(
        matches!(verdict, GateVerdict::Skipped(_)),
        "a zero RSS must never satisfy a max budget, got {verdict:?}"
    );
}

/// A report that *did* measure RSS still round-trips as a number, not as absent.
#[test]
fn a_measured_peak_rss_survives_the_json_round_trip() {
    let report = weighed_report(10.0);
    let json = report.to_json().expect("report serializes");
    let parsed: BenchmarkReport = serde_json::from_str(&json).expect("report parses back");
    assert_eq!(parsed.peak_rss_mb, report.peak_rss_mb);
    assert!(parsed.peak_rss_mb.is_some());
}

/// `perplexity_wikitext2` reads `measured` only when the run actually evaluated
/// perplexity; a report without the number is skipped, never passed.
#[test]
fn gate_reads_perplexity_only_when_the_report_carries_it() {
    let with_ppl = weighed_report(10.0).with_perplexity_wikitext2(9.5);

    let (measured, verdict) =
        verdict_of(&budget_for("perplexity_wikitext2", "max", 12.0), &with_ppl);
    assert_eq!(
        measured,
        Some(9.5),
        "the gate must report 9.5, not a target"
    );
    assert_eq!(verdict, GateVerdict::Passed);

    let (measured, verdict) =
        verdict_of(&budget_for("perplexity_wikitext2", "max", 8.0), &with_ppl);
    assert_eq!(measured, Some(9.5));
    assert_eq!(
        verdict,
        GateVerdict::Violated,
        "9.5 perplexity breaks an 8.0 budget"
    );

    let without_ppl = weighed_report(10.0);
    let (measured, verdict) = verdict_of(
        &budget_for("perplexity_wikitext2", "max", 12.0),
        &without_ppl,
    );
    assert_eq!(measured, None, "no eval, no number");
    assert!(
        matches!(verdict, GateVerdict::Skipped(_)),
        "a run that never evaluated perplexity must skip the row, got {verdict:?}"
    );
}

/// The two per-kernel rows are numbers the report carries once the probe ran.
#[test]
fn gate_reads_the_kernel_probes_from_the_report() {
    let report = weighed_report(10.0).with_probes(KernelProbes {
        gemv_peak_pct: Some(71.5),
        paged_attn_ctx4096_b1_over_theoretical: Some(1.6),
        ..KernelProbes::default()
    });

    let (measured, verdict) = verdict_of(&budget_for("gemv_peak_pct", "min", 60.0), &report);
    assert_eq!(measured, Some(71.5));
    assert_eq!(verdict, GateVerdict::Passed);
    let (_, verdict) = verdict_of(&budget_for("gemv_peak_pct", "min", 80.0), &report);
    assert_eq!(
        verdict,
        GateVerdict::Violated,
        "71.5% of peak does not reach an 80% bar"
    );

    let (measured, verdict) = verdict_of(
        &budget_for("paged_attn_ctx4096_b1_over_theoretical", "max", 2.0),
        &report,
    );
    assert_eq!(measured, Some(1.6));
    assert_eq!(verdict, GateVerdict::Passed);
    let (_, verdict) = verdict_of(
        &budget_for("paged_attn_ctx4096_b1_over_theoretical", "max", 1.5),
        &report,
    );
    assert_eq!(verdict, GateVerdict::Violated);

    let bare = weighed_report(10.0);
    let (measured, verdict) = verdict_of(&budget_for("gemv_peak_pct", "min", 60.0), &bare);
    assert_eq!(measured, None, "no probe, no number");
    assert!(matches!(verdict, GateVerdict::Skipped(_)));
}

/// `idle_gap_pct` is a decode-step figure; a report that never measured one has
/// no number to judge.
#[test]
fn gate_reads_idle_gap_from_the_report() {
    let report = weighed_report(10.0).with_probes(KernelProbes {
        idle_gap_pct: Some(3.2),
        ..KernelProbes::default()
    });

    let (measured, verdict) = verdict_of(&budget_for("idle_gap_pct", "max", 5.0), &report);
    assert_eq!(measured, Some(3.2));
    assert_eq!(verdict, GateVerdict::Passed);
    let (_, verdict) = verdict_of(&budget_for("idle_gap_pct", "max", 2.0), &report);
    assert_eq!(verdict, GateVerdict::Violated);

    let (measured, verdict) = verdict_of(
        &budget_for("idle_gap_pct", "max", 5.0),
        &weighed_report(10.0),
    );
    assert_eq!(measured, None);
    assert!(matches!(verdict, GateVerdict::Skipped(_)));
}

/// `ctx_latency_ratio` needs the pair a ctx sweep produces: two reports that
/// each say which context length they ran.
#[test]
fn ctx_latency_ratio_comes_from_a_ctx_sweep_of_reports() {
    let short = weighed_report(5.0).with_context_len(128);
    let long = weighed_report(7.0).with_context_len(2048);
    assert_eq!(
        (short.context_len, long.context_len),
        (Some(128), Some(2048))
    );

    let budget = budget_for("ctx_latency_ratio", "max", 1.3);
    let outcome = evaluate_gate_run(&budget, &[short, long]).expect("gate evaluates");
    let row = &outcome.rows[0];
    assert_eq!(
        row.measured,
        Some(1.4),
        "ctx 2048 at 7 ms over ctx 128 at 5 ms is 1.4x, got {:?}",
        row.measured
    );
    assert_eq!(row.verdict, GateVerdict::Violated);
}

/// The SSE row is a functional 0/1 result: a stream the harness consumed reads
/// 1, one it did not reads 0 — and 0 breaks the budget.
#[test]
fn gate_reads_the_sse_stream_probe_from_the_report() {
    let consumed = weighed_report(10.0).with_probes(KernelProbes {
        sse_sdk_stream_consumed: Some(1.0),
        ..KernelProbes::default()
    });
    let (measured, verdict) = verdict_of(
        &budget_for("sse_sdk_stream_consumed", "min", 1.0),
        &consumed,
    );
    assert_eq!(measured, Some(1.0));
    assert_eq!(verdict, GateVerdict::Passed);

    let not_consumed = weighed_report(10.0).with_probes(KernelProbes {
        sse_sdk_stream_consumed: Some(0.0),
        ..KernelProbes::default()
    });
    let (measured, verdict) = verdict_of(
        &budget_for("sse_sdk_stream_consumed", "min", 1.0),
        &not_consumed,
    );
    assert_eq!(
        measured,
        Some(0.0),
        "a failed smoke run reports 0, not nothing"
    );
    assert_eq!(verdict, GateVerdict::Violated);

    let (measured, verdict) = verdict_of(
        &budget_for("sse_sdk_stream_consumed", "min", 1.0),
        &weighed_report(10.0),
    );
    assert_eq!(measured, None);
    assert!(matches!(verdict, GateVerdict::Skipped(_)));
}

/// The new fields must survive the round trip the gate actually uses: `bench`
/// writes JSON, `bench --gate --gate-report` reads it back.
#[test]
fn the_new_fields_survive_the_report_json_round_trip() {
    let report = weighed_report(10.0)
        .with_perplexity_wikitext2(11.25)
        .with_context_len(2048)
        .with_probes(KernelProbes {
            gemv_peak_pct: Some(64.0),
            paged_attn_ctx4096_b1_over_theoretical: Some(1.9),
            idle_gap_pct: Some(4.5),
            sse_sdk_stream_consumed: Some(1.0),
        });

    let json = report.to_json().expect("report serializes");
    let parsed: BenchmarkReport = serde_json::from_str(&json).expect("report parses back");

    assert_eq!(parsed.peak_rss_mb, report.peak_rss_mb);
    assert_eq!(parsed.perplexity_wikitext2, Some(11.25));
    assert_eq!(parsed.context_len, Some(2048));
    let probes = parsed.probes.expect("probes survive JSON");
    assert_eq!(probes.gemv_peak_pct, Some(64.0));
    assert_eq!(probes.paged_attn_ctx4096_b1_over_theoretical, Some(1.9));
    assert_eq!(probes.idle_gap_pct, Some(4.5));
    assert_eq!(probes.sse_sdk_stream_consumed, Some(1.0));
}

/// A report written before these fields existed must still parse, and its rows
/// must skip — not default to a passing zero.
#[test]
fn a_legacy_report_without_the_new_fields_still_skips_them() {
    let legacy = r#"{
        "total_tokens": 100, "total_duration_secs": 1.0, "num_requests": 1,
        "total_tokens_per_sec": 100.0, "tokens_per_sec_per_request": 100.0,
        "avg_ms_per_step": 10.0, "mean_ttft_ms": 10.0, "max_ttft_ms": 10.0,
        "ttft_p50_ms": 10.0, "ttft_p90_ms": 10.0, "itl_mean_ms": 5.0,
        "itl_p50_ms": 5.0, "itl_p90_ms": 5.0, "itl_p99_ms": 5.0,
        "prefill_tokens_per_sec": 100.0, "decode_ms_per_step": 10.0,
        "decode_ms_per_token": 10.0, "checksum": 0, "prefix_hit_rate": 0.0,
        "bytes_read_per_step": 100000000, "peak_bandwidth_gb_s": 20.0,
        "peak_bandwidth_source": "measured", "telemetry": "measured"
    }"#;
    let parsed: BenchmarkReport = serde_json::from_str(legacy).expect("legacy report parses");

    assert_eq!(parsed.perplexity_wikitext2, None);
    assert_eq!(parsed.probes, None);
    let (measured, verdict) = verdict_of(&budget_for("gemv_peak_pct", "min", 60.0), &parsed);
    assert_eq!(measured, None, "an absent probe is not a zero");
    assert!(matches!(verdict, GateVerdict::Skipped(_)));
}

/// B7: the probe itself must produce a number. The tests above inject literals
/// to prove the gate *reads* a field; this one proves the probe *fills* it.
#[test]
fn the_gemv_probe_produces_a_real_figure_from_the_kernel_it_names() {
    use kanso_infer::bench::{GEMV_PROBE_KERNEL, probe_gemv_peak_pct};

    // On aarch64 the probe must time the NEON kernel, not the scalar reference:
    // a scalar figure would never reach the 60% peak the budget asks for, so
    // naming it `gemv_peak_pct` would be a lie.
    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            GEMV_PROBE_KERNEL.contains("neon"),
            "on aarch64 the probe must name the NEON kernel it times, got '{GEMV_PROBE_KERNEL}'"
        );
        let peak = 100.0;
        let first = probe_gemv_peak_pct(peak).expect("the NEON probe runs on aarch64");
        let second = probe_gemv_peak_pct(peak).expect("the probe is repeatable");
        assert!(first > 0.0, "a timed kernel reads above zero, got {first}");
        assert!(
            first != second,
            "two timings of the same kernel cannot be identical ({first}); a constant would mean \
             nothing was timed"
        );
        assert!(
            probe_gemv_peak_pct(0.0).is_none(),
            "no measured peak bandwidth means no share of peak"
        );
    }

    // Everywhere else the NEON kernel does not exist, so the probe must decline
    // rather than time the scalar fallback under a NEON-shaped name.
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert!(
            probe_gemv_peak_pct(100.0).is_none(),
            "without NEON there is no kernel to time under this name"
        );
    }
}
