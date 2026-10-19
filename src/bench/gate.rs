//! `bench --gate <budget.json>`: performance budgets as a reviewable diff.
//!
//! A budget file declares rows of `{metric, bound, budget, requires?, status?}`.
//! Each row is compared against the metrics of a `BenchmarkReport`; the gate
//! prints the measured value, the ratio against the budget, the budget and the
//! verdict, and exits with a code CI can read:
//!
//! - `0` — every row measured and within budget
//! - `1` — at least one row violated its budget
//! - `2` — no violation, but a row could not be measured because the asset it
//!   requires is missing, or because the report's telemetry was never weighed.
//!   A skipped row is never counted as a pass.

use serde::{Deserialize, Serialize};

use crate::bench::BenchmarkReport;
use crate::error::EngineError;

/// Direction of a budget: an upper bound (latency) or a lower bound (throughput).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateBound {
    Min,
    Max,
}

impl std::fmt::Display for GateBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// What a row needs before its number means anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GateRequirement {
    /// A file that must exist, e.g. the GGUF the run weighed.
    File { path: String },
    /// An executable that must be on `PATH`, e.g. `xctrace`.
    Tool { name: String },
}

impl GateRequirement {
    /// `None` when satisfied, otherwise the reason the row cannot be measured.
    fn missing(&self) -> Option<String> {
        match self {
            Self::File { path } => {
                if std::path::Path::new(path).is_file() {
                    None
                } else {
                    Some(format!("missing required asset: {path}"))
                }
            }
            Self::Tool { name } => {
                if tool_on_path(name) {
                    None
                } else {
                    Some(format!("missing required tool: {name}"))
                }
            }
        }
    }
}

fn tool_on_path(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path).any(|dir| {
                let candidate = dir.join(name);
                candidate.is_file()
            })
        })
        .unwrap_or(false)
}

/// Whether a row's number can be measured today (#94).
///
/// A budget that declares a target nobody instruments is still worth reviewing,
/// but it must never read as a pass: `NotYetInstrumented` rows are always
/// skipped, and the row has to name who will instrument it and what it needs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GateRowStatus {
    /// The report carries this number, so the gate judges it.
    #[default]
    Measured,
    /// Nothing measures this yet; the row is a declared target only.
    NotYetInstrumented,
}

/// One budget row as declared in the budget file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateRow {
    pub metric: String,
    pub bound: GateBound,
    pub budget: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<GateRequirement>,
    #[serde(default, skip_serializing_if = "GateRowStatus::is_measured")]
    pub status: GateRowStatus,
    /// The ticket that will instrument this row, e.g. `#95`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// What the row needs before its number means anything.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl GateRowStatus {
    fn is_measured(&self) -> bool {
        *self == Self::Measured
    }
}

/// A whole budget file: `{"name": "...", "rows": [...]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateBudget {
    pub name: String,
    pub rows: Vec<GateRow>,
}

impl GateBudget {
    /// Parses a budget file. Unknown keys are rejected so a misspelled
    /// `requires` cannot silently turn an asset check into a pass.
    pub fn from_json_str(json: &str) -> Result<Self, EngineError> {
        serde_json::from_str(json)
            .map_err(|e| EngineError::Config(format!("invalid gate budget: {e}")))
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self, EngineError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json_str(&content)
            .map_err(|e| EngineError::Config(format!("{}: {}", path.display(), e)))
    }
}

/// Verdict for one row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateVerdict {
    Passed,
    Violated,
    /// The row's number could not be measured; carries the reason.
    Skipped(String),
}

impl GateVerdict {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Passed => "passed",
            Self::Violated => "violated",
            Self::Skipped(_) => "skipped",
        }
    }
}

/// One evaluated row: what was measured against what budget.
#[derive(Debug, Clone, PartialEq)]
pub struct GateRowOutcome {
    pub metric: String,
    pub bound: GateBound,
    pub budget: f64,
    /// `None` when the row was skipped. For a sweep this is the worst reading,
    /// i.e. the one the verdict was decided on.
    pub measured: Option<f64>,
    /// `measured / budget`; `None` when the row was skipped.
    pub ratio: Option<f64>,
    pub verdict: GateVerdict,
    /// Every reading behind the verdict, in report order: a `--batch-sizes 1,8`
    /// run yields two. Ratio rows carry one entry — the ratio itself.
    pub readings: Vec<Option<f64>>,
}

/// Result of a whole budget, with the exit code CI reads.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    pub name: String,
    pub rows: Vec<GateRowOutcome>,
}

impl GateOutcome {
    /// True only when every row was measured and within budget. A skipped row
    /// and a declared-but-uninstrumented row are never a pass.
    pub fn passed(&self) -> bool {
        self.rows
            .iter()
            .all(|row| row.verdict == GateVerdict::Passed)
    }

    /// `0` all rows passed, `1` any row violated, `2` any row skipped or not
    /// measured. A violation outranks an inconclusive row: a gate that measured
    /// a regression is not "inconclusive", it is red.
    pub fn code(&self) -> i32 {
        if self
            .rows
            .iter()
            .any(|row| row.verdict == GateVerdict::Violated)
        {
            1
        } else if self
            .rows
            .iter()
            .any(|row| matches!(row.verdict, GateVerdict::Skipped(_)))
        {
            2
        } else {
            0
        }
    }

    fn verdict_word(&self) -> &'static str {
        match self.code() {
            0 => "passed",
            1 => "violated",
            _ => "not measured",
        }
    }

    /// Why this gate is not green, one line per offending row.
    pub fn reasons(&self) -> Vec<String> {
        self.rows
            .iter()
            .filter_map(|row| match &row.verdict {
                GateVerdict::Passed => None,
                GateVerdict::Violated => Some(format!(
                    "{}: violated (measured {:.4}, budget {:.4})",
                    row.metric,
                    row.measured.unwrap_or(f64::NAN),
                    row.budget
                )),
                GateVerdict::Skipped(reason) => Some(format!("{}: skipped ({reason})", row.metric)),
            })
            .collect()
    }

    /// Human- and CI-readable table: measured value, budget, ratio, verdict.
    pub fn format_table(&self) -> String {
        let mut out = format!(
            "gate '{}': {} (exit {})\n",
            self.name,
            self.verdict_word(),
            self.code()
        );
        out.push_str(&format!(
            "{:<28} {:<6} {:>12} {:>12} {:>10}  {}\n",
            "metric", "bound", "measured", "budget", "ratio", "verdict"
        ));
        for row in &self.rows {
            // A sweep gets one line per report, so batch-1 and batch-8 are both
            // visible and each is judged on its own; the header carries the
            // aggregate verdict. A ratio row has a single reading.
            if row.readings.len() > 1 {
                for value in &row.readings {
                    let (measured, ratio, verdict) = match value {
                        Some(value) => {
                            let ratio = if row.budget != 0.0 {
                                format!("{:.3}x", value / row.budget)
                            } else {
                                "-".to_string()
                            };
                            (
                                format!("{value:.4}"),
                                ratio,
                                judge(row.bound, row.budget, *value).as_str().to_string(),
                            )
                        }
                        None => ("-".to_string(), "-".to_string(), "skipped".to_string()),
                    };
                    out.push_str(&format!(
                        "{:<28} {:<6} {:>12} {:>12} {:>10}  {}\n",
                        row.metric,
                        row.bound,
                        measured,
                        format!("{:.4}", row.budget),
                        ratio,
                        verdict
                    ));
                }
                continue;
            }

            let measured = row
                .measured
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let ratio = row
                .ratio
                .map(|v| format!("{v:.3}x"))
                .unwrap_or_else(|| "-".to_string());
            let verdict = match &row.verdict {
                GateVerdict::Skipped(reason) => format!("skipped ({reason})"),
                other => other.as_str().to_string(),
            };
            out.push_str(&format!(
                "{:<28} {:<6} {:>12} {:>12} {:>10}  {}\n",
                row.metric,
                row.bound,
                measured,
                format!("{:.4}", row.budget),
                ratio,
                verdict
            ));
        }
        out
    }
}

/// Report metrics a budget row may name. The first block is carried verbatim by
/// [`BenchmarkReport`]; the second is derived from fields the report really has,
/// so it needs no new instrumentation. The third needs two reports and is only
/// resolvable through [`evaluate_gate_run`].
pub const GATE_METRICS: &[&str] = &[
    "total_tokens_per_sec",
    "tokens_per_sec_per_request",
    "decode_ms_per_step",
    "decode_ms_per_token",
    "prefill_tokens_per_sec",
    "mean_ttft_ms",
    "max_ttft_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p99_ms",
    "prefix_hit_rate",
    "bandwidth_util_pct",
    "spec_speedup_ratio",
    "batch_scale_ratio",
    "ctx_latency_ratio",
    "peak_rss_mb",
    "perplexity_wikitext2",
    "gemv_peak_pct",
    "paged_attn_ctx4096_b1_over_theoretical",
    "idle_gap_pct",
    "sse_sdk_stream_consumed",
];

/// Metrics that need a second report: `last / first` over the run's sweep.
pub const GATE_RATIO_METRICS: &[&str] = &["batch_scale_ratio", "ctx_latency_ratio"];

/// Metrics a dedicated probe produces; absent means the probe did not run (#100).
pub const PROBE_METRICS: &[&str] = &[
    "gemv_peak_pct",
    "paged_attn_ctx4096_b1_over_theoretical",
    "idle_gap_pct",
    "sse_sdk_stream_consumed",
];

/// A row's number, or why it has none. `Missing` means the metric exists but
/// this report does not carry it — a skip, never a pass.
enum Reading {
    Value(f64),
    Missing(String),
}

fn metric_value(report: &BenchmarkReport, metric: &str) -> Option<f64> {
    let value = match metric {
        "total_tokens_per_sec" => report.total_tokens_per_sec,
        "tokens_per_sec_per_request" => report.tokens_per_sec_per_request,
        "decode_ms_per_step" => report.decode_ms_per_step,
        "decode_ms_per_token" => report.decode_ms_per_token,
        "prefill_tokens_per_sec" => report.prefill_tokens_per_sec,
        "mean_ttft_ms" => report.mean_ttft_ms,
        "max_ttft_ms" => report.max_ttft_ms,
        "ttft_p50_ms" => report.ttft_p50_ms,
        "ttft_p90_ms" => report.ttft_p90_ms,
        "itl_p50_ms" => report.itl_p50_ms,
        "itl_p90_ms" => report.itl_p90_ms,
        "itl_p99_ms" => report.itl_p99_ms,
        "prefix_hit_rate" => report.prefix_hit_rate,
        // Weighed bytes over step time over the host's measured peak: the
        // whole-step share of peak bandwidth, F12's GEMV proxy (#94).
        "bandwidth_util_pct" => {
            let (_, util_pct) = crate::hardware::calculate_bandwidth_utilization(
                report.bytes_read_per_step,
                report.decode_ms_per_step,
                report.peak_bandwidth_gb_s,
            );
            util_pct
        }
        "spec_speedup_ratio" => report.speculative.as_ref()?.speedup_ratio?,
        // Peak RSS is a whole-process figure every fresh run has (#100); a
        // report from before the field carries no measurement, not a zero.
        "peak_rss_mb" => report.peak_rss_mb?,
        // Perplexity only exists when the run asked for it (#100).
        "perplexity_wikitext2" => report.perplexity_wikitext2?,
        // Probe figures: absent when the probe did not run (#100).
        "gemv_peak_pct" => report.probes.as_ref()?.gemv_peak_pct?,
        "paged_attn_ctx4096_b1_over_theoretical" => {
            report
                .probes
                .as_ref()?
                .paged_attn_ctx4096_b1_over_theoretical?
        }
        "idle_gap_pct" => report.probes.as_ref()?.idle_gap_pct?,
        "sse_sdk_stream_consumed" => report.probes.as_ref()?.sse_sdk_stream_consumed?,
        _ => return None,
    };
    Some(value)
}

fn read_metric(report: &BenchmarkReport, metric: &str) -> Reading {
    match metric_value(report, metric) {
        Some(value) => Reading::Value(value),
        None if metric == "spec_speedup_ratio" => match &report.speculative {
            None => Reading::Missing("report carries no speculative block".to_string()),
            Some(spec) => match spec.speedup_ratio {
                Some(ratio) => Reading::Value(ratio),
                None => Reading::Missing(
                    "speculative block carries no speedup ratio (no autoregressive baseline)"
                        .to_string(),
                ),
            },
        },
        None if GATE_RATIO_METRICS.contains(&metric) => Reading::Missing(format!(
            "'{metric}' needs two reports (first vs last of the run)"
        )),
        // A probe that never ran is a skip with a reason a reader can act on,
        // not a silent zero (#100).
        None if PROBE_METRICS.contains(&metric) => Reading::Missing(format!(
            "report carries no '{metric}': the probe that produces it did not run"
        )),
        None => Reading::Missing(format!("report carries no '{metric}'")),
    }
}

/// Compares every budget row against `report`.
///
/// A row declared `not-yet-instrumented` is reported as `NotMeasured` — the
/// budget is visible and reviewable, but nothing is claimed about the number.
/// An unknown metric is an error, not a pass: a typo in a budget file must not
/// turn the gate green. Rows that need two reports (see [`GATE_RATIO_METRICS`])
/// are skipped here; use [`evaluate_gate_run`] to resolve them.
pub fn evaluate_gate(
    budget: &GateBudget,
    report: &BenchmarkReport,
) -> Result<GateOutcome, EngineError> {
    evaluate_with(budget, Some(report), None)
}

/// Compares a budget against a whole run of reports — a `--batch-sizes` sweep or
/// a context-latency pair. Ratio rows read `last / first`; everything else is
/// judged per report and a violation anywhere wins.
///
/// A run with a single report cannot say "batch-8 is 4x batch-1", so its ratio
/// rows are skipped rather than guessed.
pub fn evaluate_gate_run(
    budget: &GateBudget,
    reports: &[BenchmarkReport],
) -> Result<GateOutcome, EngineError> {
    evaluate_with(budget, reports.first(), Some(reports))
}

fn evaluate_with(
    budget: &GateBudget,
    report: Option<&BenchmarkReport>,
    run: Option<&[BenchmarkReport]>,
) -> Result<GateOutcome, EngineError> {
    let mut rows = Vec::with_capacity(budget.rows.len());
    for row in &budget.rows {
        if row.status == GateRowStatus::NotYetInstrumented {
            rows.push(GateRowOutcome {
                metric: row.metric.clone(),
                bound: row.bound,
                budget: row.budget,
                measured: None,
                ratio: None,
                verdict: GateVerdict::Skipped(uninstrumented_reason(row)),
                readings: Vec::new(),
            });
            continue;
        }

        if !GATE_METRICS.contains(&row.metric.as_str()) {
            return Err(EngineError::Config(format!(
                "gate '{}': unknown metric '{}' (known: {})",
                budget.name,
                row.metric,
                GATE_METRICS.join(", ")
            )));
        }

        let report = report.ok_or_else(|| {
            EngineError::Config(format!(
                "gate '{}': row '{}' needs at least one report",
                budget.name, row.metric
            ))
        })?;

        // An asset that is not there, or a report that was never weighed, makes
        // the number meaningless even when the report carries one (#91).
        let skipped = row
            .requires
            .as_ref()
            .and_then(|requirement| requirement.missing())
            .or_else(|| {
                report
                    .is_unweighed()
                    .then(|| "unmeasured telemetry: report carries no weighed bytes".to_string())
            });

        // Ratio rows summarise the whole run; the rest are read per report so a
        // sweep shows every batch size, not just the worst one.
        let readings: Vec<Reading> = if GATE_RATIO_METRICS.contains(&row.metric.as_str()) {
            match run {
                Some(reports) if reports.len() >= 2 => match ratio_of(reports, &row.metric) {
                    Some(ratio) => vec![Reading::Value(ratio)],
                    None => vec![Reading::Missing(format!(
                        "'{}': the run's first report measured zero, so no ratio exists",
                        row.metric
                    ))],
                },
                _ => vec![Reading::Missing(format!(
                    "'{}' needs two reports (first vs last of the run)",
                    row.metric
                ))],
            }
        } else {
            match run {
                Some(reports) => reports
                    .iter()
                    .map(|report| read_metric(report, &row.metric))
                    .collect(),
                None => vec![read_metric(report, &row.metric)],
            }
        };

        let (measured, ratio, verdict) = match skipped {
            Some(reason) => (None, None, GateVerdict::Skipped(reason)),
            None => decide(row, &readings),
        };

        rows.push(GateRowOutcome {
            metric: row.metric.clone(),
            bound: row.bound,
            budget: row.budget,
            measured,
            ratio,
            verdict,
            readings: readings
                .iter()
                .map(|reading| match reading {
                    Reading::Value(value) => Some(*value),
                    Reading::Missing(_) => None,
                })
                .collect(),
        });
    }
    Ok(GateOutcome {
        name: budget.name.clone(),
        rows,
    })
}

/// One verdict for a row from its readings: a violation anywhere wins, then a
/// reading nobody produced (a skip), and only an all-pass is green.
fn decide(row: &GateRow, readings: &[Reading]) -> (Option<f64>, Option<f64>, GateVerdict) {
    let mut worst: Option<f64> = None;
    let mut missing: Option<String> = None;
    for reading in readings {
        match reading {
            Reading::Missing(reason) => {
                missing.get_or_insert_with(|| reason.clone());
            }
            Reading::Value(value) => {
                if judge(row.bound, row.budget, *value) == GateVerdict::Violated {
                    return (
                        Some(*value),
                        (row.budget != 0.0).then(|| *value / row.budget),
                        GateVerdict::Violated,
                    );
                }
                // Nothing violated, so the reported reading is the one furthest
                // from the budget — the number a reader should look at.
                worst = Some(match worst {
                    None => *value,
                    Some(current) => match row.bound {
                        GateBound::Max => current.max(*value),
                        GateBound::Min => current.min(*value),
                    },
                });
            }
        }
    }

    match (missing, worst) {
        (Some(reason), _) => (None, None, GateVerdict::Skipped(reason)),
        (None, Some(value)) => (
            Some(value),
            (row.budget != 0.0).then(|| value / row.budget),
            GateVerdict::Passed,
        ),
        (None, None) => (
            None,
            None,
            GateVerdict::Skipped(format!("report carries no '{}'", row.metric)),
        ),
    }
}

/// Why a declared-but-uninstrumented row has no number: the ticket that will
/// measure it and what it needs. The gate never guesses a value for it (#94).
fn uninstrumented_reason(row: &GateRow) -> String {
    let owner = row.owner.as_deref().unwrap_or("no owner declared");
    match &row.note {
        Some(note) => format!("not yet instrumented, owner {owner}: {note}"),
        None => format!("not yet instrumented, owner {owner}"),
    }
}

fn judge(bound: GateBound, budget: f64, measured: f64) -> GateVerdict {
    let within = match bound {
        GateBound::Max => measured <= budget,
        GateBound::Min => measured >= budget,
    };
    if within {
        GateVerdict::Passed
    } else {
        GateVerdict::Violated
    }
}

/// `last / first` for the two ratio metrics. A zero baseline cannot produce a
/// ratio, so it reads as missing instead of infinity.
fn ratio_of(reports: &[BenchmarkReport], metric: &str) -> Option<f64> {
    let first = &reports[0];
    let last = &reports[reports.len() - 1];
    let (base, top) = match metric {
        "batch_scale_ratio" => (first.total_tokens_per_sec, last.total_tokens_per_sec),
        "ctx_latency_ratio" => (first.decode_ms_per_step, last.decode_ms_per_step),
        _ => unreachable!("ratio_of is only called for GATE_RATIO_METRICS"),
    };
    (base > 0.0).then(|| top / base)
}
