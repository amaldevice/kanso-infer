use crate::speculative::verifier::VerificationResult;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Standalone metrics snapshot for speculative decoding performance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeReport {
    pub total_proposed: usize,
    pub total_accepted: usize,
    pub total_target_tokens: usize,
    pub total_steps: usize,
    pub rollback_events: usize,
    pub acceptance_rate: f64,
    pub mean_accepted_per_step: f64,
    pub total_duration_secs: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speedup_ratio: Option<f64>,
}

impl SpeculativeReport {
    pub fn average_accepted_tokens_per_step(&self) -> f64 {
        self.mean_accepted_per_step
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn format_report(&self) -> String {
        let mut s = format!(
            "Speculative Decoding Report:\n  Proposed Tokens (N_prop): {}\n  Accepted Tokens (N_acc): {}\n  Steps: {}\n  Rollback Events: {}\n  Acceptance Rate (alpha): {:.2}%\n  Mean Accepted/Step (tau_bar): {:.2}\n  Speculative Duration: {:.3}s",
            self.total_proposed,
            self.total_accepted,
            self.total_steps,
            self.rollback_events,
            self.acceptance_rate * 100.0,
            self.mean_accepted_per_step,
            self.total_duration_secs,
        );
        if let Some(s_ratio) = self.speedup_ratio {
            s.push_str(&format!("\n  Speedup Ratio (S): {:.2}x", s_ratio));
        }
        s
    }
}

impl std::fmt::Display for SpeculativeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Speculative Report [N_prop={}, N_acc={}, rollbacks={}, alpha={:.2}%, tau_bar={:.2}, duration={:.3}s",
            self.total_proposed,
            self.total_accepted,
            self.rollback_events,
            self.acceptance_rate * 100.0,
            self.mean_accepted_per_step,
            self.total_duration_secs
        )?;
        if let Some(s) = self.speedup_ratio {
            write!(f, ", S={:.2}x", s)?;
        }
        write!(f, "]")
    }
}

/// Automated efficiency profiler for speculative decoding.
///
/// Tracks proposed speculative tokens (N_prop), accepted speculative tokens (N_acc),
/// atomic KV cache rollback events, and wall-clock execution duration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeProfiler {
    pub total_proposed: usize,
    pub total_accepted: usize,
    pub total_target_tokens: usize,
    pub total_steps: usize,
    pub rollback_events: usize,
    pub total_duration: Duration,
}

impl SpeculativeProfiler {
    /// Creates a new empty `SpeculativeProfiler`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the outcome and duration of a speculative decoding verification step.
    pub fn record_step(
        &mut self,
        proposed: usize,
        accepted: usize,
        duration: Duration,
        had_rollback: bool,
    ) {
        self.record_step_with_target_tokens(proposed, accepted, 1, duration, had_rollback);
    }

    /// Records a step specifying the number of verified target tokens (minimum 1).
    pub fn record_step_with_target_tokens(
        &mut self,
        proposed: usize,
        accepted: usize,
        target_tokens: usize,
        duration: Duration,
        had_rollback: bool,
    ) {
        self.total_proposed += proposed;
        self.total_accepted += accepted;
        self.total_target_tokens += target_tokens;
        self.total_steps += 1;
        self.total_duration += duration;
        if had_rollback {
            self.rollback_events += 1;
        }
    }

    /// Records a step directly from a `VerificationResult`.
    pub fn record_verification(
        &mut self,
        proposed: usize,
        result: &VerificationResult,
        duration: Duration,
    ) {
        let had_rollback = result.num_rejected > 0 || !result.rolled_back_blocks.is_empty();
        self.record_step(proposed, result.num_accepted, duration, had_rollback);
    }

    /// Empirical acceptance rate: alpha = N_acc / N_prop.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_proposed as f64
        }
    }

    /// Alias for `acceptance_rate()`.
    pub fn alpha(&self) -> f64 {
        self.acceptance_rate()
    }

    /// Average accepted tokens per step: tau_bar = (total_accepted + total_target_tokens) / total_steps.
    /// Each target step contributes at least 1 verified token.
    pub fn average_accepted_tokens_per_step(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.total_accepted + self.total_target_tokens) as f64 / self.total_steps as f64
        }
    }

    /// Alias for `average_accepted_tokens_per_step()`.
    pub fn mean_accepted_per_step(&self) -> f64 {
        self.average_accepted_tokens_per_step()
    }

    /// Alias for `average_accepted_tokens_per_step()`.
    pub fn tau_bar(&self) -> f64 {
        self.average_accepted_tokens_per_step()
    }

    /// Measures wall-clock empirical speedup ratio: S = T_autoregressive / T_speculative.
    pub fn speedup_ratio(&self, autoregressive_time: Duration) -> f64 {
        let spec_secs = self.total_duration.as_secs_f64();
        if spec_secs <= 0.0 {
            0.0
        } else {
            autoregressive_time.as_secs_f64() / spec_secs
        }
    }

    /// Computes speedup ratio from autoregressive duration in seconds.
    pub fn speedup_ratio_secs(&self, autoregressive_secs: f64) -> f64 {
        self.speedup_ratio(Duration::from_secs_f64(autoregressive_secs))
    }

    pub fn n_prop(&self) -> usize {
        self.total_proposed
    }

    pub fn n_acc(&self) -> usize {
        self.total_accepted
    }

    pub fn rollback_count(&self) -> usize {
        self.rollback_events
    }

    pub fn rollback_events(&self) -> usize {
        self.rollback_events
    }

    pub fn step_count(&self) -> usize {
        self.total_steps
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    pub fn total_duration_secs(&self) -> f64 {
        self.total_duration.as_secs_f64()
    }

    /// Resets all counters and timers to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Generates a `SpeculativeReport` snapshot of current metrics.
    pub fn report(&self) -> SpeculativeReport {
        SpeculativeReport {
            total_proposed: self.total_proposed,
            total_accepted: self.total_accepted,
            total_target_tokens: self.total_target_tokens,
            total_steps: self.total_steps,
            rollback_events: self.rollback_events,
            acceptance_rate: self.acceptance_rate(),
            mean_accepted_per_step: self.average_accepted_tokens_per_step(),
            total_duration_secs: self.total_duration.as_secs_f64(),
            speedup_ratio: None,
        }
    }

    /// Generates a `SpeculativeReport` with speedup ratio against autoregressive baseline.
    pub fn report_with_speedup(&self, autoregressive_time: Duration) -> SpeculativeReport {
        let mut rep = self.report();
        rep.speedup_ratio = Some(self.speedup_ratio(autoregressive_time));
        rep
    }

    /// Serializes the report snapshot to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        self.report().to_json()
    }

    /// Serializes the report snapshot to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        self.report().to_json_pretty()
    }

    /// Serializes the report snapshot with speedup ratio to JSON.
    pub fn to_json_with_speedup(
        &self,
        autoregressive_time: Duration,
    ) -> Result<String, serde_json::Error> {
        self.report_with_speedup(autoregressive_time).to_json()
    }

    /// Serializes the report snapshot with speedup ratio to pretty-printed JSON.
    pub fn to_json_pretty_with_speedup(
        &self,
        autoregressive_time: Duration,
    ) -> Result<String, serde_json::Error> {
        self.report_with_speedup(autoregressive_time)
            .to_json_pretty()
    }

    /// Formats human-readable report.
    pub fn format_report(&self, autoregressive_duration: Option<Duration>) -> String {
        match autoregressive_duration {
            Some(ar) => self.report_with_speedup(ar).format_report(),
            None => self.report().format_report(),
        }
    }

    /// Prints human-readable report to stdout.
    pub fn print_report(&self, autoregressive_duration: Option<Duration>) {
        println!("{}", self.format_report(autoregressive_duration));
    }
}

impl std::fmt::Display for SpeculativeProfiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SpeculativeProfiler [N_prop={}, N_acc={}, rollbacks={}, alpha={:.2}%, tau_bar={:.2}, duration={:.3}s]",
            self.total_proposed,
            self.total_accepted,
            self.rollback_events,
            self.acceptance_rate() * 100.0,
            self.mean_accepted_per_step(),
            self.total_duration.as_secs_f64()
        )
    }
}
