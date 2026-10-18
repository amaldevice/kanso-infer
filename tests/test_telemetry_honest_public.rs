//! RED tests for ticket #91: a zero-byte report must never look like a result.
//!
//! `evaluate_run()` and `evaluate_run_with_timings(..., 0)` are public and can
//! emit `bytes_read_per_step = 0` — "not weighed yet", which used to be
//! indistinguishable from a legitimately small number. Every such report must
//! carry an explicit marker, and the gate must never count it as a pass.

use kanso_infer::bench::{
    BenchmarkReport, GateBudget, GateVerdict, TelemetryStatus, evaluate_gate,
    evaluate_run_with_timings,
};

/// Same shape as the gate tests' synthetic report, but nothing was weighed.
fn unweighed_report() -> BenchmarkReport {
    evaluate_run_with_timings(100, 1.0, 2, &[10.0, 20.0], &[5.0, 6.0], 10, 0.1, 20, 0.5, 42, 0)
}

/// The same numbers, with bytes weighed from a real model.
fn weighed_report() -> BenchmarkReport {
    evaluate_run_with_timings(
        100,
        1.0,
        2,
        &[10.0, 20.0],
        &[5.0, 6.0],
        10,
        0.1,
        20,
        0.5,
        42,
        50_000_000,
    )
}

fn min_throughput_budget() -> GateBudget {
    GateBudget::from_json_str(
        r#"{"name":"honest","rows":[{"metric":"total_tokens_per_sec","bound":"min","budget":1.0}]}"#,
    )
    .expect("budget parses")
}

#[test]
fn timings_path_with_zero_bytes_is_marked_unmeasured() {
    let report = unweighed_report();

    assert_eq!(report.bytes_read_per_step, 0);
    assert_eq!(
        report.telemetry,
        TelemetryStatus::Unmeasured,
        "a zero-byte report from a public path must say it was not weighed"
    );
    let json = report.to_json().expect("serialize");
    assert!(
        json.contains("\"telemetry\":\"unmeasured\""),
        "the marker must survive into the JSON a run writes: {json}"
    );
}

#[test]
fn weighed_report_is_marked_measured() {
    let report = weighed_report();

    assert_eq!(report.telemetry, TelemetryStatus::Measured);
    let json = report.to_json().expect("serialize");
    assert!(
        json.contains("\"telemetry\":\"measured\""),
        "a weighed report must say so: {json}"
    );
}

#[test]
fn with_telemetry_zero_does_not_claim_a_measurement() {
    let report = unweighed_report().with_telemetry(0);

    assert_eq!(
        report.telemetry,
        TelemetryStatus::Unmeasured,
        "with_telemetry(0) weighs nothing, so it must not claim Measured"
    );
}

#[test]
fn legacy_json_without_the_marker_deserializes_as_unmeasured() {
    // A report written before the marker existed carries no evidence of a
    // weighing; the safe reading is "unmeasured", never "zero is fine".
    let json = unweighed_report().to_json().expect("serialize");
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse");
    value.as_object_mut().expect("object").remove("telemetry");

    let report: BenchmarkReport = serde_json::from_value(value).expect("deserialize legacy");
    assert_eq!(report.telemetry, TelemetryStatus::Unmeasured);
}

#[test]
fn gate_cannot_pass_an_unmeasured_report() {
    let outcome = evaluate_gate(&min_throughput_budget(), &unweighed_report())
        .expect("gate evaluates");

    assert!(
        !outcome.passed(),
        "a report that was never weighed must not be a pass:\n{}",
        outcome.format_table()
    );
    assert_eq!(
        outcome.code(),
        2,
        "unmeasured is 'inconclusive', not green:\n{}",
        outcome.format_table()
    );
    match &outcome.rows[0].verdict {
        GateVerdict::Skipped(reason) => assert!(
            reason.contains("telemetry"),
            "the skip reason must name the missing telemetry, got {reason:?}"
        ),
        other => panic!("expected Skipped, got {other:?}"),
    }
}

#[test]
fn gate_still_passes_a_weighed_report() {
    let outcome = evaluate_gate(&min_throughput_budget(), &weighed_report())
        .expect("gate evaluates");

    assert!(outcome.passed(), "{}", outcome.format_table());
    assert_eq!(outcome.code(), 0);
}
