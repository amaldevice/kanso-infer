//! RED tests for ticket #94: the F12/F15 budgets live in `gates/*.json` and are
//! read by the same `bench --gate` mechanism as #84.
//!
//! The budgets are split into two honest classes and the gate must treat them
//! differently:
//!
//! - **measured** rows name a metric the report really carries (or a ratio
//!   derived from it) and are judged for real: `exit 1` when broken.
//! - **not yet instrumented** rows name a number nobody measures yet. They must
//!   come back as `skipped (…)`, never as `passed`, so a declared target can
//!   never be mistaken for a result.
//!
//! Every report here is synthetic — this ticket ships the mechanism and the
//! budget, not a measurement. The real run is #95.

use kanso_infer::bench::{
    BenchmarkReport, GateBudget, GateRowStatus, GateVerdict, evaluate_gate,
    evaluate_gate_run, evaluate_run_with_timings,
};
use kanso_infer::speculative::profiler::SpeculativeProfiler;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The GGUF the asset-backed rows point at. Absent on a CI box.
const GGUF: &str = "models/smollm2-360m-instruct-q8_0.gguf";

/// A repo-root path, so the tests read the budgets that CI reads.
fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn fase1() -> GateBudget {
    GateBudget::from_file(&repo_path("gates/fase1.json")).expect("gates/fase1.json parses")
}

fn fase2() -> GateBudget {
    GateBudget::from_file(&repo_path("gates/fase2.json")).expect("gates/fase2.json parses")
}

/// The named rows of a shipped budget with the asset requirement stripped, so a
/// box without the GGUF can still exercise the comparison. The budget numbers
/// come from the file, not from this test.
fn asset_free(budget: &GateBudget, metrics: &[&str]) -> GateBudget {
    let rows = budget
        .rows
        .iter()
        .filter(|r| metrics.contains(&r.metric.as_str()))
        .cloned()
        .map(|mut r| {
            r.requires = None;
            r.status = GateRowStatus::Measured;
            r
        })
        .collect();
    GateBudget {
        name: format!("{}-asset-free", budget.name),
        rows,
    }
}

/// Weighed synthetic report: 100 MB/step. At 10 ms/step that is 10 GB/s
/// effective, i.e. exactly 50% of a 20 GB/s peak.
fn weighed_report(decode_ms_per_step: f64, total_tokens_per_sec: f64) -> BenchmarkReport {
    let steps = 20;
    let decode_duration_secs = decode_ms_per_step * (steps as f64) / 1000.0;
    let mut report = evaluate_run_with_timings(
        100,
        1.0,
        1,
        &[10.0],
        &[5.0],
        10,
        0.1,
        steps,
        decode_duration_secs,
        42,
        100_000_000,
    );
    report.peak_bandwidth_gb_s = 20.0;
    report.peak_bandwidth_source = "measured: synthetic test peak".to_string();
    report.total_tokens_per_sec = total_tokens_per_sec;
    report
}

fn row<'a>(
    outcome: &'a kanso_infer::bench::GateOutcome,
    metric: &str,
) -> &'a kanso_infer::bench::GateRowOutcome {
    outcome
        .rows
        .iter()
        .find(|r| r.metric == metric)
        .unwrap_or_else(|| panic!("budget has no row '{metric}':\n{}", outcome.format_table()))
}

#[test]
fn fase1_budget_declares_the_f12_rows() {
    let metrics: Vec<String> = fase1().rows.iter().map(|r| r.metric.clone()).collect();
    for expected in [
        "gemv_peak_pct",
        "paged_attn_ctx4096_b1_over_theoretical",
        "idle_gap_pct",
        "peak_rss_mb",
        "perplexity_wikitext2",
        "bandwidth_util_pct",
    ] {
        assert!(
            metrics.iter().any(|m| m == expected),
            "F12 row '{expected}' missing from gates/fase1.json: {metrics:?}"
        );
    }
}

#[test]
fn fase2_budget_declares_the_f15_rows() {
    let metrics: Vec<String> = fase2().rows.iter().map(|r| r.metric.clone()).collect();
    for expected in [
        "batch_scale_ratio",
        "ctx_latency_ratio",
        "spec_speedup_ratio",
        "sse_sdk_stream_consumed",
    ] {
        assert!(
            metrics.iter().any(|m| m == expected),
            "F15 row '{expected}' missing from gates/fase2.json: {metrics:?}"
        );
    }
}

/// Every row that names a target nobody measures yet must say so, name who will
/// instrument it, and say what it needs — otherwise it is an untested number
/// pretending to be a budget.
#[test]
fn every_not_yet_instrumented_row_names_an_owner_and_a_note() {
    for budget in [fase1(), fase2()] {
        for r in &budget.rows {
            if r.status != GateRowStatus::NotYetInstrumented {
                continue;
            }
            assert!(
                r.owner.is_some(),
                "row '{}' in gate '{}' is not instrumented but names no owner",
                r.metric,
                budget.name
            );
            assert!(
                r.note.is_some(),
                "row '{}' in gate '{}' declares no note saying what it needs",
                r.metric,
                budget.name
            );
        }
    }
}

/// The whole point of the two classes: an uninstrumented target is never green.
#[test]
fn not_yet_instrumented_rows_report_skipped_never_passed() {
    let budget = asset_free(&fase1(), &["bandwidth_util_pct", "idle_gap_pct", "gemv_peak_pct"]);
    let outcome = evaluate_gate(&budget, &weighed_report(10.0, 200.0)).expect("gate evaluates");

    let idle = row(&outcome, "idle_gap_pct");
    match &idle.verdict {
        GateVerdict::Skipped(reason) => assert!(
            reason.contains("#95"),
            "the skip reason must point at the ticket that instruments it, got '{reason}'"
        ),
        other => panic!("idle_gap_pct must be skipped, got {other:?}"),
    }
    assert_eq!(idle.measured, None, "an unmeasured row has no number");
    assert_eq!(idle.ratio, None, "an unmeasured row has no ratio");
    assert!(
        matches!(row(&outcome, "gemv_peak_pct").verdict, GateVerdict::Skipped(_)),
        "gemv_peak_pct must be skipped until #95 instruments it:\n{}",
        outcome.format_table()
    );

    // The measured row in the same budget is still judged on its merits.
    assert_eq!(
        row(&outcome, "bandwidth_util_pct").verdict,
        GateVerdict::Violated
    );
    assert!(!outcome.passed(), "a skipped row is never a pass");
    assert_eq!(outcome.code(), 1, "a violation outranks the skips");
}

/// Every F12 target that has no instrumentation behind it is declared as such.
#[test]
fn fase1_declares_the_uninstrumented_f12_targets() {
    let budget = fase1();
    for metric in [
        "gemv_peak_pct",
        "paged_attn_ctx4096_b1_over_theoretical",
        "idle_gap_pct",
        "peak_rss_mb",
        "perplexity_wikitext2",
    ] {
        assert_eq!(
            row_in(&budget, metric).status,
            GateRowStatus::NotYetInstrumented,
            "'{metric}' must be declared not-yet-instrumented, not judged from a number nobody measures"
        );
    }
    assert_eq!(
        row_in(&budget, "bandwidth_util_pct").status,
        GateRowStatus::Measured,
        "bandwidth_util_pct is derived from the report's weighed bytes"
    );
}

fn row_in<'a>(budget: &'a GateBudget, metric: &str) -> &'a kanso_infer::bench::GateRow {
    budget
        .rows
        .iter()
        .find(|r| r.metric == metric)
        .unwrap_or_else(|| panic!("gate '{}' has no row '{metric}'", budget.name))
}

/// `bandwidth_util_pct` is derived from the report's own weighed bytes and
/// measured peak, so it needs no new instrumentation: 100 MB/step at 10 ms/step
/// is 10 GB/s, half of a 20 GB/s peak, under the 60% bar.
#[test]
fn bandwidth_util_pct_is_derived_from_weighed_bytes_and_measured_peak() {
    let budget = asset_free(&fase1(), &["bandwidth_util_pct"]);
    let outcome = evaluate_gate(&budget, &weighed_report(10.0, 200.0)).expect("gate evaluates");

    let util = row(&outcome, "bandwidth_util_pct");
    assert_eq!(util.measured.expect("measured"), 50.0);
    assert_eq!(util.verdict, GateVerdict::Violated, "50% is under the 60% bar");
}

/// A step that does reach the bar passes. 100 MB/step at 8 ms/step is 12.5 GB/s
/// of a 20 GB/s peak = 62.5%.
#[test]
fn bandwidth_util_pct_passes_a_step_that_reaches_the_bar() {
    let budget = asset_free(&fase1(), &["bandwidth_util_pct"]);
    let outcome = evaluate_gate(&budget, &weighed_report(8.0, 250.0)).expect("gate evaluates");

    let util = row(&outcome, "bandwidth_util_pct");
    assert!((util.measured.expect("measured") - 62.5).abs() < 1e-9);
    assert_eq!(util.verdict, GateVerdict::Passed);
}

/// Without the GGUF the asset-backed rows must report `skipped`, and the CI job
/// must be able to tell that apart from a pass (exit 2).
#[test]
fn asset_rows_are_skipped_not_passed_when_the_gguf_is_absent() {
    if repo_path(GGUF).is_file() {
        eprintln!("SKIP: {GGUF} is present, so the rows would be measured");
        return;
    }

    let outcome = evaluate_gate(&fase2(), &weighed_report(10.0, 200.0)).expect("gate evaluates");

    for metric in ["batch_scale_ratio", "ctx_latency_ratio", "spec_speedup_ratio"] {
        match &row(&outcome, metric).verdict {
            GateVerdict::Skipped(reason) => assert!(
                reason.contains("missing required asset"),
                "'{metric}' must be skipped for the missing GGUF, got '{reason}'"
            ),
            other => panic!("'{metric}' must not be {other:?} without the GGUF"),
        }
    }
    assert!(
        matches!(
            row(&outcome, "sse_sdk_stream_consumed").verdict,
            GateVerdict::Skipped(_)
        ),
        "the SDK stream row is not instrumented either"
    );
    assert_eq!(outcome.code(), 2, "no GGUF means exit 2, never 0");
    assert!(!outcome.passed());
}

/// F15 reports the real-pair speedup as it is, including below 1x — the number
/// #55 measured was 0.84x and that must show up as red, not be hidden.
#[test]
fn spec_speedup_below_one_is_reported_as_a_violation() {
    let mut profiler = SpeculativeProfiler::new();
    profiler.record_step(4, 3, Duration::from_millis(20), true);
    // 40 ms autoregressive over 47.6 ms speculative = 0.84x.
    let spec = profiler.report_with_speedup(Duration::from_millis(47.6));

    let budget = asset_free(&fase2(), &["spec_speedup_ratio"]);
    let report = weighed_report(10.0, 200.0).with_speculative(spec);
    let outcome = evaluate_gate(&budget, &report).expect("gate evaluates");

    let ratio_row = row(&outcome, "spec_speedup_ratio");
    let measured = ratio_row.measured.expect("speedup is a real number");
    assert!(
        (measured - 0.84).abs() < 0.01,
        "the ratio must be reported as measured, got {measured}"
    );
    assert_eq!(
        ratio_row.verdict,
        GateVerdict::Violated,
        "0.84x is under the 1.0x bar and must be red:\n{}",
        outcome.format_table()
    );
    assert!(
        outcome.format_table().contains("0.840x"),
        "the table must print the honest ratio:\n{}",
        outcome.format_table()
    );
}

/// A report with no speculative block has no speedup to judge — that is a skip,
/// not a pass.
#[test]
fn spec_speedup_is_skipped_when_the_report_carries_no_speculative_block() {
    let budget = asset_free(&fase2(), &["spec_speedup_ratio"]);
    let outcome = evaluate_gate(&budget, &weighed_report(10.0, 200.0)).expect("gate evaluates");

    match &row(&outcome, "spec_speedup_ratio").verdict {
        GateVerdict::Skipped(reason) => assert!(
            reason.contains("speculative"),
            "the skip reason must say the report carries no speculative block, got '{reason}'"
        ),
        other => panic!("expected a skip, got {other:?}"),
    }
}

/// Batch-8 vs batch-1 on the real GGUF: the last report of a `--batch-sizes 1,8`
/// run over the first.
#[test]
fn batch_scale_ratio_is_last_report_over_first() {
    let budget = asset_free(&fase2(), &["batch_scale_ratio"]);
    let run = vec![weighed_report(10.0, 50.0), weighed_report(10.0, 210.0)];
    let outcome = evaluate_gate_run(&budget, &run).expect("gate evaluates");

    let ratio = row(&outcome, "batch_scale_ratio");
    assert!(
        (ratio.measured.expect("ratio") - 4.2).abs() < 1e-9,
        "batch-8 210 tok/s over batch-1 50 tok/s is 4.2x, got {:?}",
        ratio.measured
    );
    assert_eq!(ratio.verdict, GateVerdict::Passed);
}

/// Context latency: ctx 2048 costs at most 1.3x ctx 128 on the same weights.
#[test]
fn ctx_latency_ratio_compares_the_last_step_to_the_first() {
    let budget = asset_free(&fase2(), &["ctx_latency_ratio"]);
    let run = vec![weighed_report(5.0, 100.0), weighed_report(7.0, 100.0)];
    let outcome = evaluate_gate_run(&budget, &run).expect("gate evaluates");

    let ratio = row(&outcome, "ctx_latency_ratio");
    assert!(
        (ratio.measured.expect("ratio") - 1.4).abs() < 1e-9,
        "7 ms over 5 ms is 1.4x, got {:?}",
        ratio.measured
    );
    assert_eq!(ratio.verdict, GateVerdict::Violated);
}

/// A ratio row needs two reports; one report alone cannot say 4x or 1.3x.
#[test]
fn ratio_rows_are_skipped_when_the_run_has_only_one_report() {
    let budget = asset_free(&fase2(), &["batch_scale_ratio", "ctx_latency_ratio"]);
    let run = vec![weighed_report(10.0, 200.0)];
    let outcome = evaluate_gate_run(&budget, &run).expect("gate evaluates");

    for metric in ["batch_scale_ratio", "ctx_latency_ratio"] {
        match &row(&outcome, metric).verdict {
            GateVerdict::Skipped(reason) => assert!(
                reason.contains("two reports"),
                "the skip reason must say a ratio needs two reports, got '{reason}'"
            ),
            other => panic!("'{metric}' must not be {other:?} from one report"),
        }
    }
}
