use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::speculative::verifier::VerificationResult;

/// Trait abstracting over Duration and f64 seconds for speedup calculations.
pub trait SpeedupTime {
    fn to_duration(&self) -> Duration;
}

impl SpeedupTime for Duration {
    fn to_duration(&self) -> Duration {
        *self
    }
}

impl SpeedupTime for &Duration {
    fn to_duration(&self) -> Duration {
        **self
    }
}

impl SpeedupTime for f64 {
    fn to_duration(&self) -> Duration {
        Duration::from_secs_f64(*self)
    }
}

/// Standalone metrics snapshot for speculative decoding performance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeReport {
    pub total_proposed: usize,
    pub total_accepted: usize,
    pub total_steps: usize,
    pub rollback_events: usize,
    pub acceptance_rate: f64,
    pub mean_accepted_per_step: f64,
    pub total_duration_secs: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speedup_ratio: Option<f64>,
}

impl SpeculativeReport {
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
        self.total_proposed += proposed;
        self.total_accepted += accepted;
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

    /// Average accepted draft tokens per target verification step: tau_bar = N_acc / total_steps.
    pub fn mean_accepted_per_step(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_steps as f64
        }
    }

    /// Alias for `mean_accepted_per_step()`.
    pub fn tau_bar(&self) -> f64 {
        self.mean_accepted_per_step()
    }

    /// Measures wall-clock empirical speedup ratio: S = T_autoregressive / T_speculative.
    pub fn speedup_ratio<T: SpeedupTime>(&self, autoregressive_time: T) -> f64 {
        let spec_secs = self.total_duration.as_secs_f64();
        if spec_secs <= 0.0 {
            0.0
        } else {
            autoregressive_time.to_duration().as_secs_f64() / spec_secs
        }
    }

    /// Computes speedup ratio from autoregressive duration in seconds.
    pub fn speedup_ratio_secs(&self, autoregressive_secs: f64) -> f64 {
        self.speedup_ratio(autoregressive_secs)
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
            total_steps: self.total_steps,
            rollback_events: self.rollback_events,
            acceptance_rate: self.acceptance_rate(),
            mean_accepted_per_step: self.mean_accepted_per_step(),
            total_duration_secs: self.total_duration.as_secs_f64(),
            speedup_ratio: None,
        }
    }

    /// Generates a `SpeculativeReport` with speedup ratio against autoregressive baseline.
    pub fn report_with_speedup<T: SpeedupTime>(&self, autoregressive_time: T) -> SpeculativeReport {
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
    pub fn to_json_with_speedup<T: SpeedupTime>(
        &self,
        autoregressive_time: T,
    ) -> Result<String, serde_json::Error> {
        self.report_with_speedup(autoregressive_time).to_json()
    }

    /// Serializes the report snapshot with speedup ratio to pretty-printed JSON.
    pub fn to_json_pretty_with_speedup<T: SpeedupTime>(
        &self,
        autoregressive_time: T,
    ) -> Result<String, serde_json::Error> {
        self.report_with_speedup(autoregressive_time).to_json_pretty()
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
