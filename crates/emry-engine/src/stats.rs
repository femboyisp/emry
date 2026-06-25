//! Streaming statistic processors: [`Ema`] and [`Welford`].
//!
//! Each tracks a single registered metric (by [`MetricId`]) and emits named
//! [`DerivedMetric`]s as values arrive. Non-finite inputs (`NaN`/`Inf`) are
//! **rejected** — skipped without updating the running statistic — so a single
//! bad step never poisons the smoothed value or the variance. Surfacing the
//! anomaly itself is the anomaly processor's responsibility (EMRY-013).

use crate::processor::{DerivedMetric, Processor};
use emry_core::{Event, MetricId};

/// Extracts the value for `target` from an event, if present.
fn value_for(event: &Event, target: MetricId) -> Option<f64> {
    match event {
        Event::Metric { id, value, .. } if *id == target => Some(*value),
        Event::MetricsBatch { values, .. } => values
            .iter()
            .find_map(|(id, value)| (*id == target).then_some(*value)),
        _ => None,
    }
}

/// Exponential moving average of a single metric.
///
/// `smoothed = alpha * value + (1 - alpha) * previous`, seeded with the first
/// observed value. Larger `alpha` tracks the raw signal more closely.
#[derive(Debug, Clone)]
pub struct Ema {
    target: MetricId,
    out_name: String,
    alpha: f64,
    current: Option<f64>,
}

impl Ema {
    /// Creates an EMA over `target`, emitting under `out_name`.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the half-open range `(0.0, 1.0]`.
    #[must_use]
    pub fn new(target: MetricId, out_name: impl Into<String>, alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "EMA alpha must be in (0.0, 1.0], got {alpha}"
        );
        Self {
            target,
            out_name: out_name.into(),
            alpha,
            current: None,
        }
    }

    /// Current smoothed value, if any input has been observed.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        self.current
    }
}

impl Processor for Ema {
    fn on_event(&mut self, event: &Event) -> Vec<DerivedMetric> {
        let Some(value) = value_for(event, self.target) else {
            return Vec::new();
        };
        if !value.is_finite() {
            return Vec::new(); // reject; keep current state intact
        }
        let next = match self.current {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        };
        self.current = Some(next);
        vec![DerivedMetric::new(self.out_name.clone(), next)]
    }
}

/// Online mean and (sample) standard deviation of a single metric, via
/// Welford's algorithm — numerically stable and single-pass.
///
/// Emits two derived metrics per update: `"{base}_mean"` and `"{base}_std"`.
/// `std` is the sample standard deviation (divides by `n - 1`); it is `0.0`
/// until at least two values have been observed.
#[derive(Debug, Clone)]
pub struct Welford {
    target: MetricId,
    mean_name: String,
    std_name: String,
    count: f64,
    mean: f64,
    m2: f64,
}

impl Welford {
    /// Creates a Welford accumulator over `target`, emitting `"{base}_mean"`
    /// and `"{base}_std"`.
    #[must_use]
    pub fn new(target: MetricId, base: &str) -> Self {
        Self {
            target,
            mean_name: format!("{base}_mean"),
            std_name: format!("{base}_std"),
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Running mean, or `0.0` if no values have been observed.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample standard deviation, or `0.0` with fewer than two values.
    #[must_use]
    pub fn std(&self) -> f64 {
        if self.count > 1.0 {
            (self.m2 / (self.count - 1.0)).sqrt()
        } else {
            0.0
        }
    }
}

impl Processor for Welford {
    fn on_event(&mut self, event: &Event) -> Vec<DerivedMetric> {
        let Some(value) = value_for(event, self.target) else {
            return Vec::new();
        };
        if !value.is_finite() {
            return Vec::new(); // reject; keep accumulator intact
        }
        self.count += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        vec![
            DerivedMetric::new(self.mean_name.clone(), self.mean()),
            DerivedMetric::new(self.std_name.clone(), self.std()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::Phase;

    const TARGET: MetricId = MetricId(0);
    const EPS: f64 = 1e-9;

    fn metric(value: f64) -> Event {
        Event::Metric {
            id: TARGET,
            value,
            step: 0,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    #[test]
    fn ema_matches_reference_series() {
        // alpha=0.5 over [1,2,3,4]: 1 -> 1.5 -> 2.25 -> 3.125
        let mut ema = Ema::new(TARGET, "loss_ema", 0.5);
        let outputs: Vec<f64> = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(|v| ema.on_event(&metric(v))[0].value)
            .collect();
        approx(outputs[0], 1.0);
        approx(outputs[1], 1.5);
        approx(outputs[2], 2.25);
        approx(outputs[3], 3.125);
        approx(ema.value().unwrap(), 3.125);
    }

    #[test]
    fn ema_rejects_non_finite_without_corrupting_state() {
        let mut ema = Ema::new(TARGET, "loss_ema", 0.5);
        assert_eq!(ema.on_event(&metric(1.0)).len(), 1);
        assert!(ema.on_event(&metric(f64::NAN)).is_empty());
        assert!(ema.on_event(&metric(f64::INFINITY)).is_empty());
        // State preserved: next real value smooths from 1.0, not from NaN.
        approx(ema.on_event(&metric(3.0))[0].value, 2.0);
    }

    #[test]
    fn ema_ignores_other_metrics_and_non_metric_events() {
        let mut ema = Ema::new(MetricId(5), "x", 0.3);
        assert!(ema.on_event(&metric(1.0)).is_empty()); // wrong id
        assert!(ema.on_event(&Event::PhaseChange(Phase::Eval)).is_empty());
        assert_eq!(ema.value(), None);
    }

    #[test]
    fn ema_reads_from_metrics_batch() {
        let mut ema = Ema::new(MetricId(1), "lr_ema", 1.0);
        let batch = Event::MetricsBatch {
            step: 0,
            epoch: 0,
            phase: Phase::Train,
            values: vec![(MetricId(0), 9.0), (MetricId(1), 2.0)],
        };
        // alpha=1.0 tracks the raw value exactly.
        approx(ema.on_event(&batch)[0].value, 2.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in")]
    fn ema_rejects_out_of_range_alpha() {
        let _ = Ema::new(TARGET, "x", 0.0);
    }

    #[test]
    fn welford_matches_reference_series() {
        // [2,4,4,4,5,5,7,9]: mean 5, sum-sq-dev 32, sample var 32/7.
        let mut w = Welford::new(TARGET, "loss");
        let mut last = Vec::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            last = w.on_event(&metric(v));
        }
        approx(w.mean(), 5.0);
        approx(w.std(), (32.0_f64 / 7.0).sqrt());
        // Emits mean then std under the right names.
        assert_eq!(last[0].name, "loss_mean");
        assert_eq!(last[1].name, "loss_std");
        approx(last[0].value, 5.0);
    }

    #[test]
    fn welford_std_is_zero_before_two_samples() {
        let mut w = Welford::new(TARGET, "loss");
        let out = w.on_event(&metric(3.0));
        approx(out[0].value, 3.0); // mean
        approx(out[1].value, 0.0); // std
        approx(w.std(), 0.0);
    }

    #[test]
    fn welford_rejects_non_finite_without_corrupting_state() {
        let mut w = Welford::new(TARGET, "loss");
        w.on_event(&metric(2.0));
        w.on_event(&metric(8.0));
        assert!(w.on_event(&metric(f64::NAN)).is_empty());
        approx(w.mean(), 5.0); // unchanged by the rejected NaN
    }
}
