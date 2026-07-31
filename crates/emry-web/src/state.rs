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

/// A checkpoint marker: the step it was taken at and a short label derived from
/// its path, so the phase-aware chart can split the curve at checkpoint
/// boundaries and label each segment (`.../phase1-reasoning/x.pt` → `reasoning`).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WebCheckpoint {
    /// Step the checkpoint was taken at.
    pub step: u64,
    /// Short human label for the segment this checkpoint ends.
    pub label: String,
}

/// A named curriculum stage the run entered (from `run.stage(...)`). Like
/// [`WebCheckpoint`], a stage step is a phase-segment boundary, but the label is
/// the explicit stage name. When present these take precedence over checkpoints.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WebStage {
    /// Step the stage began at.
    pub step: u64,
    /// Explicit stage name (segment label).
    pub label: String,
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
    /// Checkpoints taken (vertical markers + phase-segment boundaries/labels).
    pub checkpoints: VecDeque<WebCheckpoint>,
    /// Named curriculum stages (explicit phase-segment boundaries); when
    /// non-empty these take precedence over checkpoints as chart segments.
    pub stages: VecDeque<WebStage>,
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
            Event::Checkpoint { path, step } => {
                self.checkpoints.push_back(WebCheckpoint {
                    step: *step,
                    label: checkpoint_label(path),
                });
                cap(&mut self.checkpoints, DEFAULT_MARKERS);
            }
            Event::StageChange { name, step } => {
                self.stages.push_back(WebStage {
                    step: *step,
                    label: name.clone(),
                });
                cap(&mut self.stages, DEFAULT_MARKERS);
            }
            Event::MetricsRegistered { names } => {
                for (id, name) in names {
                    self.labels.insert(id.index(), name.clone());
                    // Relabel any metric already tracked under the `m{id}` fallback.
                    if let Some(m) = self.metrics.iter_mut().find(|m| m.id == id.index()) {
                        m.label.clone_from(name);
                    }
                }
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

/// Derives a short segment label from a checkpoint path: the parent directory
/// name with a leading `phaseN-`/`phaseN_` ordering prefix stripped, else the
/// file stem. Kept in sync with `emry_tui::ui::checkpoint_label` (the reference
/// implementation); the paired unit tests below assert the same cases.
fn checkpoint_label(path: &str) -> String {
    let p = std::path::Path::new(path);
    let raw = p
        .parent()
        .and_then(std::path::Path::file_name)
        .or_else(|| p.file_stem())
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    raw.split_once(['-', '_'])
        .filter(|(head, _)| {
            head.starts_with("phase") && head[5..].chars().all(|c| c.is_ascii_digit())
        })
        .map_or(raw.clone(), |(_, rest)| rest.to_string())
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
            path: "/out/phase2-knowledge/adapters.pt".into(),
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
        // The checkpoint records its step (marker) and a phase-segment label
        // derived from the path (phaseN- prefix stripped).
        assert_eq!(s.checkpoints.len(), 1);
        assert_eq!(s.checkpoints[0].step, 2);
        assert_eq!(s.checkpoints[0].label, "knowledge");
        // They serialize for the browser.
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"phases\"") && json.contains("\"checkpoints\""));
        assert!(json.contains("\"steps\""));
    }

    #[test]
    fn stage_change_is_recorded_and_serialized() {
        let mut s = WebState::default();
        s.apply(&batch(0, &[(0, 1.0)]));
        s.apply(&Event::StageChange {
            name: "reasoning".into(),
            step: 0,
        });
        assert_eq!(s.stages.len(), 1);
        assert_eq!(s.stages[0].step, 0);
        assert_eq!(s.stages[0].label, "reasoning");
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"stages\"") && json.contains("reasoning"));
    }

    #[test]
    fn checkpoint_label_matches_tui_reference() {
        // Same cases the TUI's checkpoint_label test asserts, so the two copies
        // can't silently drift.
        assert_eq!(
            checkpoint_label("out/phase1-reasoning/adapters.safetensors"),
            "reasoning"
        );
        assert_eq!(checkpoint_label("runs/phase12_code/x.pt"), "code");
        assert_eq!(checkpoint_label("ckpts/warmup/x.pt"), "warmup"); // no phaseN prefix
        assert_eq!(checkpoint_label("step_200.pt"), "step_200"); // file stem fallback
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
    fn metrics_registered_seeds_and_relabels() {
        let mut s = WebState::default();
        s.apply(&batch(0, &[(0, 1.0)])); // first seen under the m{id} fallback
        assert_eq!(s.metrics[0].label, "m0");
        s.apply(&Event::MetricsRegistered {
            names: vec![(MetricId(0), "loss".into())],
        });
        assert_eq!(s.metrics[0].label, "loss"); // relabeled from the name table
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
