//! RED tests for ticket #84: `bench --gate <budget.json>`.
//!
//! Seam S2: the gate is observed through `GateOutcome` (measured value, ratio
//! against the budget, verdict, exit code) and through the exit code of the real
//! `bench` process. No test reaches into the comparison internals.

use kanso_infer::bench::{
    BenchmarkReport, GateBudget, GateVerdict, evaluate_gate, evaluate_run_with_timings,
};
use std::path::PathBuf;

/// Synthetic report, no model and no hardware involved:
/// `decode_ms_per_step` = 0.5 s / 20 steps x 1000 = 25 ms, `total_tokens_per_sec` = 100.
fn sample_report() -> BenchmarkReport {
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
        1_000_000,
    )
}

fn budget(rows: &str) -> GateBudget {
    GateBudget::from_json_str(&format!(r#"{{"name":"fase1","rows":[{rows}]}}"#))
        .expect("budget JSON parses")
}

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("kanso-84-{}-{name}", std::process::id()))
}

fn write_tmp(name: &str, content: &str) -> PathBuf {
    let path = tmp_path(name);
    std::fs::write(&path, content).expect("write temp file");
    path
}

#[test]
fn gate_passes_when_every_row_is_within_budget() {
    let budget = budget(
        r#"{"metric":"decode_ms_per_step","bound":"max","budget":25.0},
           {"metric":"total_tokens_per_sec","bound":"min","budget":100.0}"#,
    );

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(
        outcome.code(),
        0,
        "a value exactly at the budget is within it:\n{}",
        outcome.format_table()
    );
    assert!(outcome.passed());
    for row in &outcome.rows {
        assert_eq!(row.verdict, GateVerdict::Passed, "metric {}", row.metric);
        assert_eq!(row.measured.expect("measured value"), row.budget);
        assert!((row.ratio.expect("ratio against budget") - 1.0).abs() < 1e-9);
    }
}

#[test]
fn gate_exits_1_when_a_max_budget_is_exceeded() {
    let budget = budget(
        r#"{"metric":"decode_ms_per_step","bound":"max","budget":20.0},
           {"metric":"total_tokens_per_sec","bound":"min","budget":1.0}"#,
    );

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(outcome.code(), 1, "{}", outcome.format_table());
    assert!(!outcome.passed());
    assert_eq!(outcome.rows[0].verdict, GateVerdict::Violated);
    assert!((outcome.rows[0].ratio.expect("ratio") - 1.25).abs() < 1e-9);
    assert_eq!(
        outcome.rows[1].verdict,
        GateVerdict::Passed,
        "one violated row must not stop the others from being reported"
    );
}

#[test]
fn gate_exits_1_when_a_min_budget_is_undershot() {
    let budget = budget(r#"{"metric":"total_tokens_per_sec","bound":"min","budget":250.0}"#);

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(outcome.code(), 1, "{}", outcome.format_table());
    assert_eq!(outcome.rows[0].verdict, GateVerdict::Violated);
    assert!((outcome.rows[0].ratio.expect("ratio") - 0.4).abs() < 1e-9);
}

#[test]
fn gate_reports_skipped_when_the_required_asset_is_missing() {
    let missing = tmp_path("does-not-exist.gguf");
    let _ = std::fs::remove_file(&missing);
    let budget = budget(&format!(
        r#"{{"metric":"total_tokens_per_sec","bound":"min","budget":1.0,
             "requires":{{"kind":"file","path":"{}"}}}}"#,
        missing.display()
    ));

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(outcome.code(), 2, "{}", outcome.format_table());
    assert!(!outcome.passed(), "skipped is never a pass");
    assert_eq!(
        outcome.rows[0].measured, None,
        "a skipped row carries no measurement"
    );
    match &outcome.rows[0].verdict {
        GateVerdict::Skipped(reason) => assert!(
            reason.contains("does-not-exist.gguf"),
            "the reason must name the missing asset, got {reason:?}"
        ),
        other => panic!("expected skipped, got {other:?}"),
    }
}

#[test]
fn gate_measures_a_row_whose_required_asset_is_present() {
    let present = write_tmp("present.gguf", "not a real model, only presence matters");
    let budget = budget(&format!(
        r#"{{"metric":"total_tokens_per_sec","bound":"min","budget":1.0,
             "requires":{{"kind":"file","path":"{}"}}}}"#,
        present.display()
    ));

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(outcome.code(), 0, "{}", outcome.format_table());
    assert_eq!(outcome.rows[0].verdict, GateVerdict::Passed);
    let _ = std::fs::remove_file(&present);
}

#[test]
fn gate_reports_skipped_when_the_required_tool_is_missing() {
    let budget = budget(
        r#"{"metric":"total_tokens_per_sec","bound":"min","budget":1.0,
            "requires":{"kind":"tool","name":"kanso-84-not-a-real-tool"}}"#,
    );

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(outcome.code(), 2, "{}", outcome.format_table());
    match &outcome.rows[0].verdict {
        GateVerdict::Skipped(reason) => assert!(
            reason.contains("kanso-84-not-a-real-tool"),
            "the reason must name the missing tool, got {reason:?}"
        ),
        other => panic!("expected skipped, got {other:?}"),
    }
}

#[test]
fn gate_exits_1_not_2_when_a_row_is_violated_and_another_is_skipped() {
    let missing = tmp_path("also-missing.gguf");
    let _ = std::fs::remove_file(&missing);
    let budget = budget(&format!(
        r#"{{"metric":"total_tokens_per_sec","bound":"min","budget":1.0,
             "requires":{{"kind":"file","path":"{}"}}}},
           {{"metric":"decode_ms_per_step","bound":"max","budget":1.0}}"#,
        missing.display()
    ));

    let outcome = evaluate_gate(&budget, &sample_report()).expect("gate evaluates");

    assert_eq!(
        outcome.code(),
        1,
        "a real violation outranks a skipped row:\n{}",
        outcome.format_table()
    );
}

#[test]
fn gate_table_prints_measured_budget_ratio_and_verdict() {
    let budget = budget(r#"{"metric":"decode_ms_per_step","bound":"max","budget":20.0}"#);

    let table = evaluate_gate(&budget, &sample_report())
        .expect("gate evaluates")
        .format_table();

    assert!(table.contains("decode_ms_per_step"), "{table}");
    assert!(table.contains("25.0000"), "measured value missing:\n{table}");
    assert!(table.contains("20.0000"), "budget missing:\n{table}");
    assert!(table.contains("1.250x"), "ratio missing:\n{table}");
    assert!(table.contains("violated"), "verdict missing:\n{table}");
    assert!(table.contains("exit 1"), "exit code missing:\n{table}");
}

#[test]
fn gate_rejects_an_unknown_metric_instead_of_passing_it() {
    let budget = budget(r#"{"metric":"gemv_peak_pct","bound":"min","budget":0.6}"#);

    let err = evaluate_gate(&budget, &sample_report())
        .expect_err("an unknown metric must not be silently treated as passing");

    assert!(
        err.to_string().contains("gemv_peak_pct"),
        "error must name the bad metric, got {err}"
    );
}

#[test]
fn budget_rejects_unknown_keys() {
    let err = GateBudget::from_json_str(
        r#"{"name":"typo","rows":[{"metric":"total_tokens_per_sec","bound":"max","budget":1.0,
             "require":{"kind":"file","path":"models/smollm2.gguf"}}]}"#,
    )
    .expect_err("a misspelled `requires` must not silently skip the asset check");

    assert!(err.to_string().contains("require"), "got {err}");
}

fn bench_bin() -> &'static str {
    env!("CARGO_BIN_EXE_bench")
}

fn run_cli(budget_path: &PathBuf, report_path: &PathBuf) -> std::process::Output {
    std::process::Command::new(bench_bin())
        .args([
            "--gate",
            budget_path.to_str().expect("budget path"),
            "--gate-report",
            report_path.to_str().expect("report path"),
        ])
        .output()
        .expect("run bench --gate")
}

fn cli_fixture(tag: &str, budget_rows: &str) -> (PathBuf, PathBuf) {
    let report = write_tmp(&format!("{tag}-report.json"), &sample_report().to_json_pretty().unwrap());
    let budget = write_tmp(
        &format!("{tag}-budget.json"),
        &format!(r#"{{"name":"cli-{tag}","rows":[{budget_rows}]}}"#),
    );
    (budget, report)
}

#[test]
fn cli_gate_exits_0_when_the_report_is_within_budget() {
    let (budget, report) = cli_fixture(
        "pass",
        r#"{"metric":"decode_ms_per_step","bound":"max","budget":30.0}"#,
    );

    let out = run_cli(&budget, &report);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(0), "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("decode_ms_per_step"), "{stdout}");
    assert!(
        stdout.contains(report.to_str().unwrap()),
        "the evidence report path must be printed so the verdict can be re-checked:\n{stdout}"
    );
    assert!(
        !stderr.contains("--model"),
        "gating an existing report must not require a model:\n{stderr}"
    );
    let _ = std::fs::remove_file(&budget);
    let _ = std::fs::remove_file(&report);
}

#[test]
fn cli_gate_exits_1_when_the_report_breaks_budget() {
    let (budget, report) = cli_fixture(
        "violated",
        r#"{"metric":"decode_ms_per_step","bound":"max","budget":20.0}"#,
    );

    let out = run_cli(&budget, &report);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(1), "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("violated"), "{stdout}");
    assert!(
        !stderr.contains("--model"),
        "gating an existing report must not require a model:\n{stderr}"
    );
    let _ = std::fs::remove_file(&budget);
    let _ = std::fs::remove_file(&report);
}

#[test]
fn cli_gate_exits_2_when_a_required_asset_is_missing() {
    let missing = tmp_path("cli-missing.gguf");
    let _ = std::fs::remove_file(&missing);
    let (budget, report) = cli_fixture(
        "skipped",
        &format!(
            r#"{{"metric":"decode_ms_per_step","bound":"max","budget":30.0,
                 "requires":{{"kind":"file","path":"{}"}}}}"#,
            missing.display()
        ),
    );

    let out = run_cli(&budget, &report);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(2), "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(stdout.contains("skipped"), "{stdout}");
    let _ = std::fs::remove_file(&budget);
    let _ = std::fs::remove_file(&report);
}
