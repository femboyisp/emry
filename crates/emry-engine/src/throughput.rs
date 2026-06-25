//! Throughput (steps/sec) and ETA estimation.
//!
//! [`Throughput`] watches the `step` of metric events and reports two derived
//! series: `"steps_per_sec"` and, when the total step count is known,
//! `"eta_secs"` (estimated seconds until completion).
//!
//! Events carry no wall-clock time, so the processor stamps arrival time from a
//! monotonic [`Instant`] on the live path. The rate math lives in the pure
//! [`Throughput::observe`] (time injected) so it is tested deterministically
//! without sleeping.

use crate::processor::{DerivedMetric, Processor};
use emry_core::Event;
use std::collections::VecDeque;
use std::time::Instant;

/// Pulls the `step` out of a metric-bearing event.
fn step_of(event: &Event) -> Option<u64> {
    match event {
        Event::Metric { step, .. } | Event::MetricsBatch { step, .. } => Some(*step),
        _ => None,
    }
}

/// Estimates training throughput and time-to-completion from observed steps.
#[derive(Debug)]
pub struct Throughput {
    total_steps: Option<u64>,
    /// Sliding window of `(time_secs, step)` samples, one per step advance.
    window: VecDeque<(f64, u64)>,
    capacity: usize,
    last_step: Option<u64>,
    start: Instant,
}

impl Throughput {
    /// Creates a throughput estimator.
    ///
    /// `total_steps` enables ETA when known. `window_capacity` is the number of
    /// step samples averaged for the rate (larger = smoother, laggier).
    ///
    /// # Panics
    ///
    /// Panics if `window_capacity < 2` (a rate needs at least two samples).
    #[must_use]
    pub fn new(total_steps: Option<u64>, window_capacity: usize) -> Self {
        assert!(window_capacity >= 2, "window_capacity must be >= 2");
        Self {
            total_steps,
            window: VecDeque::with_capacity(window_capacity),
            capacity: window_capacity,
            last_step: None,
            start: Instant::now(),
        }
    }

    /// Records a step observed at `now_secs` (monotonic seconds), returning the
    /// derived metrics. Pure: no clock access, so callers/tests control time.
    ///
    /// Only step *advances* are recorded; a repeated or out-of-order step is
    /// ignored so multiple metrics logged at the same step don't skew the rate.
    #[allow(clippy::cast_precision_loss)] // step counts and durations fit f64 exactly for any real run
    pub fn observe(&mut self, step: u64, now_secs: f64) -> Vec<DerivedMetric> {
        if self.last_step.is_some_and(|last| step <= last) {
            return Vec::new();
        }
        self.last_step = Some(step);
        self.window.push_back((now_secs, step));
        if self.window.len() > self.capacity {
            self.window.pop_front();
        }

        // Need two samples spanning a positive duration to define a rate.
        if self.window.len() < 2 {
            return Vec::new();
        }
        let (t0, s0) = self.window.front().copied().unwrap();
        let (t1, s1) = self.window.back().copied().unwrap();
        let dt = t1 - t0;
        if dt <= 0.0 {
            return Vec::new();
        }

        let steps_per_sec = (s1 - s0) as f64 / dt;
        let mut out = vec![DerivedMetric::new("steps_per_sec", steps_per_sec)];
        if let Some(total) = self.total_steps {
            if steps_per_sec > 0.0 {
                let remaining = total.saturating_sub(s1) as f64;
                out.push(DerivedMetric::new("eta_secs", remaining / steps_per_sec));
            }
        }
        out
    }
}

impl Processor for Throughput {
    fn on_event(&mut self, event: &Event) -> Vec<DerivedMetric> {
        let Some(step) = step_of(event) else {
            return Vec::new();
        };
        let now_secs = self.start.elapsed().as_secs_f64();
        self.observe(step, now_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::{MetricId, Phase};

    const EPS: f64 = 1e-9;

    fn metric_at(step: u64) -> Event {
        Event::Metric {
            id: MetricId(0),
            value: 0.0,
            step,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    fn named<'a>(out: &'a [DerivedMetric], name: &str) -> Option<&'a f64> {
        out.iter().find(|d| d.name == name).map(|d| &d.value)
    }

    #[test]
    fn first_sample_yields_no_rate() {
        let mut t = Throughput::new(None, 8);
        assert!(t.observe(0, 0.0).is_empty());
    }

    #[test]
    fn rate_and_eta_from_two_samples() {
        let mut t = Throughput::new(Some(100), 8);
        assert!(t.observe(0, 0.0).is_empty());
        let out = t.observe(10, 1.0); // 10 steps in 1s -> 10/s
        approx(*named(&out, "steps_per_sec").unwrap(), 10.0);
        // (100 - 10) remaining / 10 per sec = 9s
        approx(*named(&out, "eta_secs").unwrap(), 9.0);
    }

    #[test]
    fn no_eta_when_total_unknown() {
        let mut t = Throughput::new(None, 8);
        t.observe(0, 0.0);
        let out = t.observe(5, 1.0);
        approx(*named(&out, "steps_per_sec").unwrap(), 5.0);
        assert!(named(&out, "eta_secs").is_none());
    }

    #[test]
    fn non_advancing_steps_are_ignored() {
        let mut t = Throughput::new(None, 8);
        t.observe(5, 0.0);
        assert!(t.observe(5, 1.0).is_empty(), "same step ignored");
        assert!(t.observe(3, 2.0).is_empty(), "out-of-order step ignored");
    }

    #[test]
    fn zero_duration_window_yields_no_rate() {
        let mut t = Throughput::new(None, 8);
        t.observe(0, 5.0);
        // Advancing step but identical timestamp -> dt == 0, no rate.
        assert!(t.observe(1, 5.0).is_empty());
    }

    #[test]
    fn window_slides_to_recent_samples() {
        let mut t = Throughput::new(None, 2);
        t.observe(0, 0.0);
        t.observe(10, 1.0);
        // Window now holds the last two: (1.0, 10) and (2.0, 30) -> 20/s.
        let out = t.observe(30, 2.0);
        approx(*named(&out, "steps_per_sec").unwrap(), 20.0);
    }

    #[test]
    fn eta_zero_at_or_past_total() {
        let mut t = Throughput::new(Some(10), 8);
        t.observe(0, 0.0);
        let out = t.observe(20, 1.0); // past total
        approx(*named(&out, "eta_secs").unwrap(), 0.0);
    }

    #[test]
    #[should_panic(expected = "window_capacity must be >= 2")]
    fn rejects_tiny_window() {
        let _ = Throughput::new(None, 1);
    }

    #[test]
    fn live_processor_path_emits_rate() {
        let mut t = Throughput::new(None, 8);
        assert!(t.on_event(&metric_at(0)).is_empty());
        std::thread::sleep(std::time::Duration::from_millis(2));
        let out = t.on_event(&metric_at(100));
        // Real elapsed time is nonzero, so a positive rate is reported.
        assert!(*named(&out, "steps_per_sec").unwrap() > 0.0);
        // Non-metric events are ignored.
        assert!(t.on_event(&Event::PhaseChange(Phase::Eval)).is_empty());
    }
}
