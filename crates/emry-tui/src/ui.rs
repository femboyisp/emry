//! Observable dashboard state and rendering.
//!
//! [`UiState`] is a pure reducer over the [`Event`] stream — feed it events with
//! [`UiState::apply`] and it tracks per-metric history, the current phase,
//! progress, and recent alerts. [`render`] draws it into any ratatui
//! [`Backend`](ratatui::backend::Backend), so the whole view is testable against
//! a `TestBackend` without a real terminal.
//!
//! Derived series (EMA, throughput, ETA) are not shown yet — wiring the
//! processors' `DerivedMetric`s into this state is EMRY-022.

use crate::chart::render_braille;
use emry_core::{AlertRecord, Event, MetricId, Phase, Severity};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use std::collections::BTreeMap;

/// Warm brand palette (design §13).
pub const CREAM: Color = Color::Rgb(0xF5, 0xF0, 0xE8);
/// Terracotta accent.
pub const TERRACOTTA: Color = Color::Rgb(0xC4, 0x71, 0x4A);
/// Warm gray for secondary text.
pub const WARM_GRAY: Color = Color::Rgb(0x6B, 0x65, 0x60);

const DEFAULT_HISTORY: usize = 4096;
const DEFAULT_ALERTS: usize = 5;

/// A single tracked metric and its recent history.
#[derive(Debug, Clone)]
pub struct MetricView {
    /// The metric's id.
    pub id: MetricId,
    /// Human-readable label (falls back to `m{id}` when unknown).
    pub label: String,
    /// Most recent value.
    pub latest: f64,
    /// Recent values, oldest first (capped).
    pub history: Vec<f64>,
}

/// The full dashboard state, reduced from the event stream.
#[derive(Debug, Clone)]
pub struct UiState {
    /// Project / experiment name.
    pub project: String,
    /// Latest step seen.
    pub step: u64,
    /// Total steps, if known (enables a progress ratio).
    pub total_steps: Option<u64>,
    /// Current training phase.
    pub phase: Phase,
    /// Tracked metrics, in first-seen order.
    pub metrics: Vec<MetricView>,
    /// Recent alerts (most recent last, capped).
    pub alerts: Vec<AlertRecord>,
    /// Index of the metric whose chart is shown.
    pub selected: usize,
    /// Whether rendering is paused (state still updates).
    pub paused: bool,
    /// Whether the run has finished.
    pub finished: bool,
    labels: BTreeMap<u16, String>,
    max_history: usize,
    max_alerts: usize,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            project: String::new(),
            step: 0,
            total_steps: None,
            phase: Phase::Train,
            metrics: Vec::new(),
            alerts: Vec::new(),
            selected: 0,
            paused: false,
            finished: false,
            labels: BTreeMap::new(),
            max_history: DEFAULT_HISTORY,
            max_alerts: DEFAULT_ALERTS,
        }
    }
}

impl UiState {
    /// Creates an empty state with metric-name labels seeded (id index → name).
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
                self.record(*id, *value);
            }
            Event::MetricsBatch { step, values, .. } => {
                self.step = *step;
                for (id, value) in values {
                    self.record(*id, *value);
                }
            }
            Event::PhaseChange(phase) => self.phase = *phase,
            Event::Alert(alert) => {
                self.alerts.push(alert.clone());
                if self.alerts.len() > self.max_alerts {
                    self.alerts.remove(0);
                }
            }
            Event::RunFinished { .. } => self.finished = true,
            Event::Checkpoint { .. } | Event::ConfigPatch(_) => {}
        }
    }

    /// Selects the metric chart by index, ignoring out-of-range requests.
    pub fn select(&mut self, index: usize) {
        if index < self.metrics.len() {
            self.selected = index;
        }
    }

    /// Toggles the paused flag.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    fn record(&mut self, id: MetricId, value: f64) {
        let label = self.label_for(id);
        let max_history = self.max_history;
        let view = if let Some(v) = self.metrics.iter_mut().find(|m| m.id == id) {
            v
        } else {
            self.metrics.push(MetricView {
                id,
                label,
                latest: value,
                history: Vec::new(),
            });
            self.metrics.last_mut().expect("just pushed")
        };
        view.latest = value;
        view.history.push(value);
        if view.history.len() > max_history {
            view.history.remove(0);
        }
    }

    fn label_for(&self, id: MetricId) -> String {
        self.labels
            .get(&id.index())
            .cloned()
            .unwrap_or_else(|| format!("m{}", id.index()))
    }
}

/// Draws the four-pane dashboard: header, metric cards, chart, alert strip.
pub fn render(frame: &mut Frame, state: &UiState) {
    let area = frame.area();
    let panes = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Length(3), // metric cards
            Constraint::Min(3),    // chart
            Constraint::Length(3), // alerts
        ])
        .split(area);

    render_header(frame, panes[0], state);
    render_cards(frame, panes[1], state);
    render_chart(frame, panes[2], state);
    render_alerts(frame, panes[3], state);
}

fn block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(WARM_GRAY))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(TERRACOTTA).add_modifier(Modifier::BOLD),
        ))
}

fn render_header(frame: &mut Frame, area: Rect, state: &UiState) {
    let progress = match state.total_steps {
        Some(total) if total > 0 => format!("step {}/{total}", state.step),
        _ => format!("step {}", state.step),
    };
    let status = if state.finished {
        "finished"
    } else if state.paused {
        "paused"
    } else {
        "running"
    };
    let line = Line::from(vec![
        Span::styled(
            state.project.clone(),
            Style::default().fg(TERRACOTTA).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(progress, Style::default().fg(CREAM)),
        Span::raw("  "),
        Span::styled(format!("{:?}", state.phase), Style::default().fg(WARM_GRAY)),
        Span::raw("  "),
        Span::styled(status, Style::default().fg(WARM_GRAY)),
    ]);
    frame.render_widget(Paragraph::new(line).block(block("Emry")), area);
}

fn render_cards(frame: &mut Frame, area: Rect, state: &UiState) {
    let mut spans = Vec::new();
    for (i, m) in state.metrics.iter().enumerate() {
        let style = if i == state.selected {
            Style::default().fg(TERRACOTTA).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(CREAM)
        };
        spans.push(Span::styled(format!("{}={:.4}", m.label, m.latest), style));
        spans.push(Span::raw("   "));
    }
    if spans.is_empty() {
        spans.push(Span::styled(
            "waiting for metrics…",
            Style::default().fg(WARM_GRAY),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(block("Metrics")),
        area,
    );
}

fn render_chart(frame: &mut Frame, area: Rect, state: &UiState) {
    let Some(metric) = state.metrics.get(state.selected) else {
        frame.render_widget(block("Chart"), area);
        return;
    };
    // Inner drawable area excludes the one-cell border on each side.
    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = render_braille(&metric.history, inner_w, inner_h)
        .into_iter()
        .map(|row| Line::from(Span::styled(row, Style::default().fg(TERRACOTTA))))
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(block(&format!(
            "{} (latest {:.4})",
            metric.label, metric.latest
        ))),
        area,
    );
}

fn render_alerts(frame: &mut Frame, area: Rect, state: &UiState) {
    let line = match state.alerts.last() {
        Some(alert) => {
            let color = match alert.severity {
                Severity::Critical => TERRACOTTA,
                Severity::Warning => Color::Rgb(0xD9, 0x9A, 0x4A),
                Severity::Info => WARM_GRAY,
            };
            Line::from(Span::styled(
                alert.message.clone(),
                Style::default().fg(color),
            ))
        }
        None => Line::from(Span::styled(
            "no alerts — all calm",
            Style::default().fg(WARM_GRAY),
        )),
    };
    frame.render_widget(Paragraph::new(line).block(block("Alerts")), area);
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::float_cmp,
        clippy::field_reassign_with_default
    )]
    use super::*;
    use emry_core::{FinishReason, RunMeta};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use uuid::Uuid;

    fn batch(step: u64, pairs: &[(u16, f64)]) -> Event {
        Event::MetricsBatch {
            step,
            epoch: 0,
            phase: Phase::Train,
            values: pairs.iter().map(|(id, v)| (MetricId(*id), *v)).collect(),
        }
    }

    #[test]
    fn run_started_sets_project() {
        let mut s = UiState::default();
        s.apply(&Event::RunStarted(RunMeta {
            run_id: Uuid::nil(),
            project: "llama".into(),
            config: serde_json::Value::Null,
            start_time_secs: 0.0,
        }));
        assert_eq!(s.project, "llama");
    }

    #[test]
    fn metrics_accumulate_history_and_latest() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        s.apply(&batch(0, &[(0, 1.0)]));
        s.apply(&batch(1, &[(0, 0.5)]));
        assert_eq!(s.step, 1);
        assert_eq!(s.metrics.len(), 1);
        assert_eq!(s.metrics[0].label, "loss");
        assert_eq!(s.metrics[0].latest, 0.5);
        assert_eq!(s.metrics[0].history, vec![1.0, 0.5]);
    }

    #[test]
    fn unknown_metric_gets_fallback_label() {
        let mut s = UiState::default();
        s.apply(&batch(0, &[(7, 1.0)]));
        assert_eq!(s.metrics[0].label, "m7");
    }

    #[test]
    fn history_is_capped() {
        let mut s = UiState::default();
        s.max_history = 3;
        for step in 0..10 {
            s.apply(&batch(step, &[(0, step as f64)]));
        }
        assert_eq!(s.metrics[0].history.len(), 3);
        assert_eq!(s.metrics[0].history, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn alerts_are_capped_keeping_most_recent() {
        let mut s = UiState::default();
        s.max_alerts = 2;
        for i in 0..5 {
            s.apply(&Event::Alert(AlertRecord {
                severity: Severity::Warning,
                message: format!("a{i}"),
                step: Some(i),
            }));
        }
        assert_eq!(s.alerts.len(), 2);
        assert_eq!(s.alerts[1].message, "a4");
    }

    #[test]
    fn selection_ignores_out_of_range() {
        let mut s = UiState::default();
        s.apply(&batch(0, &[(0, 1.0), (1, 2.0)]));
        s.select(1);
        assert_eq!(s.selected, 1);
        s.select(9); // ignored
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn pause_and_finish_flags() {
        let mut s = UiState::default();
        s.toggle_pause();
        assert!(s.paused);
        s.apply(&Event::RunFinished {
            duration_secs: 1.0,
            reason: FinishReason::Completed,
        });
        assert!(s.finished);
    }

    #[test]
    fn renders_into_test_backend_without_panicking() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        for step in 0..50 {
            s.apply(&batch(step, &[(0, 1.0 / (step as f64 + 1.0))]));
        }
        s.apply(&Event::Alert(AlertRecord {
            severity: Severity::Critical,
            message: "Loss became NaN".into(),
            step: Some(12),
        }));

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();

        let text = buffer_text(terminal.backend());
        assert!(text.contains("Emry"));
        assert!(text.contains("loss"));
        assert!(text.contains("Loss became NaN"));
    }

    #[test]
    fn renders_empty_state() {
        let s = UiState::default();
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();
        let text = buffer_text(terminal.backend());
        assert!(text.contains("waiting for metrics"));
    }

    fn buffer_text(backend: &TestBackend) -> String {
        let buffer = backend.buffer();
        buffer
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }
}
