//! Anomaly detection: NaN/Inf and loss spikes.
//!
//! [`AnomalyDetector`] watches one metric and emits [`AlertRecord`]s with gentle,
//! calm copy (brand §13):
//!
//! - **Non-finite** (`NaN`/`Inf`) → a [`Severity::Critical`] alert immediately.
//!   This is the counterpart to the "reject, don't alert" policy in the EMA/
//!   Welford processors: the statistic processors keep their state clean, and the
//!   anomaly detector is the single place that surfaces the bad value.
//! - **Loss spike** → a [`Severity::Warning`] when a finite value's rolling
//!   z-score exceeds a threshold (default 4).
//!
//! It produces [`AlertRecord`]s rather than `DerivedMetric`s, so it is its own
//! type rather than a [`Processor`](crate::Processor). Wiring alerts onto the
//! event bus as [`Event::Alert`](emry_core::Event) happens in the engine
//! assembly (EMRY-015).

use emry_core::{AlertRecord, Event, MetricId, Severity};
use std::collections::VecDeque;

/// Default rolling-window z-score threshold for spike detection.
pub const DEFAULT_Z_THRESHOLD: f64 = 4.0;

/// Default number of recent finite samples used to compute the z-score.
pub const DEFAULT_WINDOW: usize = 50;

/// Minimum history before spike detection activates (a z-score is meaningless
/// with too few samples, and noisy with a zero/near-zero std).
const MIN_SAMPLES: usize = 5;

/// Extracts `(step, value)` for `target` from a metric-bearing event.
fn step_value(event: &Event, target: MetricId) -> Option<(u64, f64)> {
    match event {
        Event::Metric {
            id, value, step, ..
        } if *id == target => Some((*step, *value)),
        Event::MetricsBatch { step, values, .. } => values
            .iter()
            .find_map(|(id, value)| (*id == target).then_some((*step, *value))),
        _ => None,
    }
}

/// Detects non-finite values and statistical spikes in a single metric.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    target: MetricId,
    label: String,
    window: VecDeque<f64>,
    capacity: usize,
    z_threshold: f64,
}

impl AnomalyDetector {
    /// Creates a detector for `target` using a human-readable `label` (e.g.
    /// `"Loss"`) in alert messages, with default window and threshold.
    #[must_use]
    pub fn new(target: MetricId, label: impl Into<String>) -> Self {
        Self::with_config(target, label, DEFAULT_WINDOW, DEFAULT_Z_THRESHOLD)
    }

    /// Creates a detector with an explicit rolling `window` and z-score
    /// `z_threshold`.
    ///
    /// # Panics
    ///
    /// Panics if `window < 2` or `z_threshold <= 0.0`.
    #[must_use]
    pub fn with_config(
        target: MetricId,
        label: impl Into<String>,
        window: usize,
        z_threshold: f64,
    ) -> Self {
        assert!(window >= 2, "window must be >= 2");
        assert!(z_threshold > 0.0, "z_threshold must be > 0");
        Self {
            target,
            label: label.into(),
            window: VecDeque::with_capacity(window),
            capacity: window,
            z_threshold,
        }
    }

    /// Inspects an event, returning any alerts it triggered.
    ///
    /// A non-finite value yields a critical alert and is **not** added to the
    /// window (so it cannot poison future z-scores). A finite value is checked
    /// for a spike against the prior window, then appended.
    pub fn detect(&mut self, event: &Event) -> Vec<AlertRecord> {
        let Some((step, value)) = step_value(event, self.target) else {
            return Vec::new();
        };

        if !value.is_finite() {
            let kind = if value.is_nan() { "NaN" } else { "infinite" };
            return vec![AlertRecord {
                severity: Severity::Critical,
                message: format!(
                    "{} became {kind} at step {step} — you may want to pause and check your data.",
                    self.label
                ),
                step: Some(step),
            }];
        }

        let mut alerts = Vec::new();
        if let Some(z) = self.z_score(value) {
            if z.abs() > self.z_threshold {
                alerts.push(AlertRecord {
                    severity: Severity::Warning,
                    message: format!(
                        "{} spiked at step {step} (z-score {z:.1}) — this may be a transient blip.",
                        self.label
                    ),
                    step: Some(step),
                });
            }
        }

        self.push(value);
        alerts
    }

    /// Z-score of `value` against the current window, or `None` if there is too
    /// little history or the window has no spread.
    fn z_score(&self, value: f64) -> Option<f64> {
        let n = self.window.len();
        if n < MIN_SAMPLES {
            return None;
        }
        #[allow(clippy::cast_precision_loss)] // window length is small and bounded
        let count = n as f64;
        let mean = self.window.iter().sum::<f64>() / count;
        let variance = self.window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        let std = variance.sqrt();
        if std <= 0.0 {
            return None;
        }
        Some((value - mean) / std)
    }

    fn push(&mut self, value: f64) {
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::Phase;

    const TARGET: MetricId = MetricId(0);

    fn metric(value: f64, step: u64) -> Event {
        Event::Metric {
            id: TARGET,
            value,
            step,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    /// Feed `n` lightly-jittered values around `value` so the window has a
    /// small but non-zero spread (a real series is never perfectly flat).
    fn prime(det: &mut AnomalyDetector, n: u64, value: f64) {
        for step in 0..n {
            let jitter = if step % 2 == 0 { 0.01 } else { -0.01 };
            assert!(det.detect(&metric(value + jitter, step)).is_empty());
        }
    }

    #[test]
    fn nan_triggers_critical_alert_immediately() {
        let mut det = AnomalyDetector::new(TARGET, "Loss");
        let alerts = det.detect(&metric(f64::NAN, 12_400));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, Severity::Critical);
        assert_eq!(alerts[0].step, Some(12_400));
        assert!(alerts[0].message.contains("NaN"));
        assert!(alerts[0].message.contains("12400"));
    }

    #[test]
    fn infinity_triggers_critical_alert() {
        let mut det = AnomalyDetector::new(TARGET, "Grad");
        let alerts = det.detect(&metric(f64::INFINITY, 7));
        assert_eq!(alerts[0].severity, Severity::Critical);
        assert!(alerts[0].message.contains("infinite"));
    }

    #[test]
    fn non_finite_does_not_enter_window() {
        let mut det = AnomalyDetector::with_config(TARGET, "Loss", 8, 4.0);
        prime(&mut det, 6, 1.0);
        det.detect(&metric(f64::NAN, 100)); // must not pollute the window
                                            // A real spike is still measured against the clean 1.0 history.
        let alerts = det.detect(&metric(50.0, 101));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, Severity::Warning);
    }

    #[test]
    fn loss_spike_triggers_warning() {
        let mut det = AnomalyDetector::with_config(TARGET, "Loss", 50, 4.0);
        prime(&mut det, 20, 2.0);
        // A small jitter would be within threshold; a huge jump is a spike.
        let alerts = det.detect(&metric(100.0, 21));
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, Severity::Warning);
        assert!(alerts[0].message.contains("z-score"));
    }

    #[test]
    fn no_spike_for_values_within_threshold() {
        let mut det = AnomalyDetector::with_config(TARGET, "Loss", 50, 4.0);
        // Mildly varying series; nothing exceeds z=4.
        for step in 0..30u64 {
            let v = 2.0 + f64::from(u32::try_from(step % 3).unwrap()) * 0.1;
            assert!(det.detect(&metric(v, step)).is_empty());
        }
    }

    #[test]
    fn no_spike_before_min_samples() {
        let mut det = AnomalyDetector::new(TARGET, "Loss");
        // Fewer than MIN_SAMPLES: even a wild value can't be judged a spike.
        det.detect(&metric(1.0, 0));
        det.detect(&metric(1.0, 1));
        let alerts = det.detect(&metric(1000.0, 2));
        assert!(alerts.is_empty());
    }

    #[test]
    fn flat_window_has_no_spread_so_no_false_spike() {
        // All identical values -> std 0 -> z_score returns None, no divide-by-0.
        let mut det = AnomalyDetector::with_config(TARGET, "Loss", 10, 4.0);
        prime(&mut det, 8, 5.0);
        // Next identical value: std still 0, no alert.
        assert!(det.detect(&metric(5.0, 8)).is_empty());
    }

    #[test]
    fn ignores_other_metrics_and_non_metric_events() {
        let mut det = AnomalyDetector::new(MetricId(9), "Loss");
        assert!(det.detect(&metric(f64::NAN, 0)).is_empty()); // wrong id
        assert!(det.detect(&Event::PhaseChange(Phase::Eval)).is_empty());
    }

    #[test]
    fn reads_from_metrics_batch() {
        let mut det = AnomalyDetector::new(MetricId(1), "LR");
        let batch = Event::MetricsBatch {
            step: 42,
            epoch: 0,
            phase: Phase::Train,
            values: vec![(MetricId(0), 1.0), (MetricId(1), f64::NAN)],
        };
        let alerts = det.detect(&batch);
        assert_eq!(alerts[0].severity, Severity::Critical);
        assert_eq!(alerts[0].step, Some(42));
    }

    #[test]
    #[should_panic(expected = "z_threshold must be > 0")]
    fn rejects_non_positive_threshold() {
        let _ = AnomalyDetector::with_config(TARGET, "Loss", 10, 0.0);
    }

    #[test]
    #[should_panic(expected = "window must be >= 2")]
    fn rejects_tiny_window() {
        let _ = AnomalyDetector::with_config(TARGET, "Loss", 1, 4.0);
    }
}
