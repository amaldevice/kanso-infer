//! `bench --gate <budget.json>`: performance budgets as a reviewable diff.
//!
//! A budget file declares rows of `{metric, bound, budget, requires?}`. Each row
//! is compared against the metrics of a `BenchmarkReport`; the gate prints the
//! measured value, the ratio against the budget, the budget and the verdict, and
//! exits with a code CI can read:
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

/// One budget row as declared in the budget file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateRow {
    pub metric: String,
    pub bound: GateBound,
    pub budget: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<GateRequirement>,
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
    /// `None` when the row was skipped.
    pub measured: Option<f64>,
    /// `measured / budget`; `None` when the row was skipped.
    pub ratio: Option<f64>,
    pub verdict: GateVerdict,
}

/// Result of a whole budget, with the exit code CI reads.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    pub name: String,
    pub rows: Vec<GateRowOutcome>,
}

impl GateOutcome {
    /// True only when every row was measured and within budget. A skipped row
    /// is never a pass.
    pub fn passed(&self) -> bool {
        self.rows
            .iter()
            .all(|row| row.verdict == GateVerdict::Passed)
    }

    /// `0` all rows passed, `1` any row violated, `2` any row skipped.
    /// A violation outranks a skip: a gate that measured a regression is not
    /// "inconclusive", it is red.
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
            _ => "skipped",
        }
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

/// Report metrics a budget row may name.
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
];

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
        _ => return None,
    };
    Some(value)
}

/// Compares every budget row against `report`.
///
/// An unknown metric is an error, not a pass: a typo in a budget file must not
/// turn the gate green.
pub fn evaluate_gate(
    budget: &GateBudget,
    report: &BenchmarkReport,
) -> Result<GateOutcome, EngineError> {
    let mut rows = Vec::with_capacity(budget.rows.len());
    for row in &budget.rows {
        let measured = metric_value(report, &row.metric).ok_or_else(|| {
            EngineError::Config(format!(
                "gate '{}': unknown metric '{}' (known: {})",
                budget.name,
                row.metric,
                GATE_METRICS.join(", ")
            ))
        })?;

        let skip_reason = row
            .requires
            .as_ref()
            .and_then(|requirement| requirement.missing())
            .or_else(|| {
                // #91: a report that was never weighed cannot back a verdict.
                report
                    .is_unweighed()
                    .then(|| "unmeasured telemetry: report carries no weighed bytes".to_string())
            });

        let (measured, ratio, verdict) = match skip_reason {
            Some(reason) => (None, None, GateVerdict::Skipped(reason)),
            None => {
                let within = match row.bound {
                    GateBound::Max => measured <= row.budget,
                    GateBound::Min => measured >= row.budget,
                };
                let ratio = if row.budget != 0.0 {
                    Some(measured / row.budget)
                } else {
                    None
                };
                let verdict = if within {
                    GateVerdict::Passed
                } else {
                    GateVerdict::Violated
                };
                (Some(measured), ratio, verdict)
            }
        };

        rows.push(GateRowOutcome {
            metric: row.metric.clone(),
            bound: row.bound,
            budget: row.budget,
            measured,
            ratio,
            verdict,
        });
    }
    Ok(GateOutcome {
        name: budget.name.clone(),
        rows,
    })
}
