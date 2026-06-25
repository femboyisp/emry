//! Browser-facing dashboard state, reduced from the event stream.
//!
//! [`WebState`] is the web analogue of the TUI's `UiState`: a pure reducer over
//! [`Event`]s that serializes to JSON for the WebSocket. It is intentionally
//! separate from the ratatui-coupled TUI state (this crate must not pull in
//! ratatui); a shared reducer is a future refactor.

use emry_core::{Event, MetricId, Severity};
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};

const DEFAULT_HISTORY: usize = 2048;
const DEFAULT_ALERTS: usize = 16;
const DEFAULT_MARKERS: usize = 256;

/// A tracked metric and its recent history, as sent to the browser.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WebMetric {
    /// Metric id.
    pub id: u16,
    /// Human-readable label (falls back to `m{id}`).
    pub label: String,
    /// Most recent value.
    pub latest: f64,
    /// Recent values, oldest first (capped FIFO). Serializes as a JSON array.
    pub history: VecDeque<f64>,
    /// The step each `history` value was recorded at (parallel to `history`),
    /// so the chart x-axis is step-based for phase bands + checkpoint markers.
    pub steps: VecDeque<u64>,
}

/// A phase transition: the run entered `phase` at `step`. The chart shades the
/// background by phase between consecutive spans.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhaseSpan {
    /// Step at which this phase began.
    pub step: u64,
    /// Phase name (screaming-snake).
    pub phase: String,
}

/// An alert surfaced to the dashboard.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct WebAlert {
    /// Severity (`INFO` | `WARNING` | `CRITICAL`).
    pub severity: String,
    /// Calm alert copy.
    pub message: String,
    /// Step the alert refers to, if any.
    pub step: Option<u64>,
}

/// The full dashboard state serialized over the WebSocket.
#[derive(Debug, Clone, Serialize, Default)]
pub struct WebState {
    /// Project / experiment name.
    pub project: String,
    /// Latest step seen.
    pub step: u64,
    /// Current phase (screaming-snake string).
    pub phase: String,
    /// Whether the run has finished.
    pub finished: bool,
    /// Tracked metrics in first-seen order.
    pub metrics: Vec<WebMetric>,
    /// Recent alerts (most recent last, capped FIFO).
    pub alerts: VecDeque<WebAlert>,
    /// Phase transitions in step order (for background shading).
    pub phases: VecDeque<PhaseSpan>,
    /// Steps at which checkpoints were taken (for vertical markers).
    pub checkpoints: VecDeque<u64>,
    #[serde(skip)]
    labels: BTreeMap<u16, String>,
}

impl WebState {
    /// Creates an empty state, optionally seeded with metric labels.
    #[must_use]
    pub fn with_labels(labels: &[(MetricId, &str)]) -> Self {
        let mut state = Self::default();
        for (id, name) in labels {
            state.labels.insert(id.index(), (*name).to_owned());
        }
        state
    }

    /// Reduces one event into the state.
    pub fn apply(&mut self, event: &Event) {
        match event {
            Event::RunStarted(meta) => self.project.clone_from(&meta.project),
            Event::Metric {
                id, value, step, ..
            } => {
                self.step = *step;
                self.record(*id, *value, *step);
            }
            Event::MetricsBatch { step, values, .. } => {
                self.step = *step;
                for (id, value) in values {
                    self.record(*id, *value, *step);
                }
            }
            Event::PhaseChange(phase) => {
                self.phase = phase_str(*phase);
                self.phases.push_back(PhaseSpan {
                    step: self.step,
                    phase: self.phase.clone(),
                });
                cap(&mut self.phases, DEFAULT_MARKERS);
            }
            Event::Alert(alert) => {
                self.alerts.push_back(WebAlert {
                    severity: severity_str(alert.severity),
                    message: alert.message.clone(),
                    step: alert.step,
                });
                cap(&mut self.alerts, DEFAULT_ALERTS);
            }
            Event::Checkpoint { step, .. } => {
                self.checkpoints.push_back(*step);
                cap(&mut self.checkpoints, DEFAULT_MARKERS);
            }
            Event::RunFinished { .. } => self.finished = true,
            Event::ConfigPatch(_) => {}
        }
    }

    fn record(&mut self, id: MetricId, value: f64, step: u64) {
        let label = self.label_for(id);
        let view = if let Some(v) = self.metrics.iter_mut().find(|m| m.id == id.index()) {
            v
        } else {
            self.metrics.push(WebMetric {
                id: id.index(),
                label,
                latest: value,
                history: VecDeque::new(),
                steps: VecDeque::new(),
            });
            self.metrics.last_mut().expect("just pushed")
        };
        view.latest = value;
        view.history.push_back(value);
        view.steps.push_back(step);
        if view.history.len() > DEFAULT_HISTORY {
            view.history.pop_front();
            view.steps.pop_front();
        }
    }

    fn label_for(&self, id: MetricId) -> String {
        self.labels
            .get(&id.index())
            .cloned()
            .unwrap_or_else(|| format!("m{}", id.index()))
    }
}

fn cap<T>(deque: &mut VecDeque<T>, max: usize) {
    while deque.len() > max {
        deque.pop_front();
    }
}

fn phase_str(phase: emry_core::Phase) -> String {
    serde_json::to_value(phase)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn severity_str(severity: Severity) -> String {
    serde_json::to_value(severity)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use emry_core::{AlertRecord, FinishReason, Phase};

    fn batch(step: u64, pairs: &[(u16, f64)]) -> Event {
        Event::MetricsBatch {
            step,
            epoch: 0,
            phase: Phase::Train,
            values: pairs.iter().map(|(id, v)| (MetricId(*id), *v)).collect(),
        }
    }

    #[test]
    fn records_steps_phase_spans_and_checkpoints() {
        let mut s = WebState::default();
        s.apply(&batch(0, &[(0, 1.0)]));
        s.apply(&batch(1, &[(0, 0.9)]));
        s.apply(&Event::PhaseChange(Phase::Eval)); // transitions at step 1
        s.apply(&Event::Checkpoint {
            path: "/ckpt/2.pt".into(),
            step: 2,
        });
        // Each value carries its step (parallel to history).
        assert_eq!(
            s.metrics[0].steps.iter().copied().collect::<Vec<_>>(),
            vec![0, 1]
        );
        // The phase span records where EVAL began.
        assert_eq!(s.phases.len(), 1);
        assert_eq!(s.phases[0].step, 1);
        assert_eq!(s.phases[0].phase, "EVAL");
        // The checkpoint step is recorded for a marker.
        assert_eq!(s.checkpoints.iter().copied().collect::<Vec<_>>(), vec![2]);
        // They serialize for the browser.
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"phases\"") && json.contains("\"checkpoints\""));
        assert!(json.contains("\"steps\""));
    }

    #[test]
    fn metrics_accumulate_with_labels() {
        let mut s = WebState::with_labels(&[(MetricId(0), "loss")]);
        s.apply(&batch(0, &[(0, 1.0)]));
        s.apply(&batch(1, &[(0, 0.5)]));
        assert_eq!(s.step, 1);
        assert_eq!(s.metrics[0].label, "loss");
        assert_eq!(s.metrics[0].latest, 0.5);
        assert_eq!(
            s.metrics[0].history.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.5]
        );
    }

    #[test]
    fn unknown_metric_falls_back_to_m_id() {
        let mut s = WebState::default();
        s.apply(&batch(0, &[(7, 1.0)]));
        assert_eq!(s.metrics[0].label, "m7");
    }

    #[test]
    fn phase_and_finish_and_alerts() {
        let mut s = WebState::default();
        s.apply(&Event::PhaseChange(Phase::Eval));
        assert_eq!(s.phase, "EVAL");
        s.apply(&Event::Alert(AlertRecord {
            severity: Severity::Critical,
            message: "Loss became NaN".into(),
            step: Some(12),
        }));
        assert_eq!(s.alerts[0].severity, "CRITICAL");
        assert_eq!(s.alerts[0].step, Some(12));
        s.apply(&Event::RunFinished {
            duration_secs: 1.0,
            reason: FinishReason::Completed,
        });
        assert!(s.finished);
    }

    #[test]
    fn serializes_to_json_for_the_browser() {
        let mut s = WebState::with_labels(&[(MetricId(0), "loss")]);
        s.apply(&batch(0, &[(0, 0.25)]));
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"project\""));
        assert!(json.contains("\"loss\""));
        assert!(json.contains("0.25"));
        // The internal label map is not serialized.
        assert!(!json.contains("labels"));
    }

    #[test]
    fn alerts_are_capped() {
        let mut s = WebState::default();
        for i in 0..(DEFAULT_ALERTS + 5) {
            s.apply(&Event::Alert(AlertRecord {
                severity: Severity::Info,
                message: format!("a{i}"),
                step: None,
            }));
        }
        assert_eq!(s.alerts.len(), DEFAULT_ALERTS);
        assert_eq!(
            s.alerts.back().unwrap().message,
            format!("a{}", DEFAULT_ALERTS + 4)
        );
    }
}
