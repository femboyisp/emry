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

use crate::chart::{
    adaptive_ema_span, ema_smooth, render_braille_segments, render_braille_steps_scaled,
    value_range, Segment, StepChart, BRAILLE_BLANK,
};
use emry_core::{AlertRecord, Event, MetricId, Phase, Severity};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use std::collections::{BTreeMap, VecDeque};

/// Warm brand palette (design §13).
pub const CREAM: Color = Color::Rgb(0xF5, 0xF0, 0xE8);
/// Terracotta accent.
pub const TERRACOTTA: Color = Color::Rgb(0xC4, 0x71, 0x4A);
/// Warm gray for secondary text.
pub const WARM_GRAY: Color = Color::Rgb(0x6B, 0x65, 0x60);
/// Amber accent — used for the comparison-baseline overlay.
pub const AMBER: Color = Color::Rgb(0xD9, 0x9A, 0x4A);

const DEFAULT_HISTORY: usize = 4096;
const DEFAULT_ALERTS: usize = 5;
const DEFAULT_MARKERS: usize = 512;

/// A single tracked metric and its recent history.
#[derive(Debug, Clone)]
pub struct MetricView {
    /// The metric's id.
    pub id: MetricId,
    /// Human-readable label (falls back to `m{id}` when unknown).
    pub label: String,
    /// Most recent value.
    pub latest: f64,
    /// Recent values, oldest first (capped FIFO).
    pub history: VecDeque<f64>,
    /// Step each `history` value was recorded at (parallel to `history`), so the
    /// chart x-axis is step-based for phase bands + checkpoint markers.
    pub steps: VecDeque<u64>,
}

/// A recorded checkpoint: the step it was taken at and a short label derived
/// from its path (e.g. `.../phase1-reasoning/adapters.safetensors` → `reasoning`).
/// Checkpoint steps double as segment boundaries for the phase-aware chart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    /// Step the checkpoint was taken at.
    pub step: u64,
    /// Short human label for the segment this checkpoint ends.
    pub label: String,
}

/// Derives a short segment label from a checkpoint path: the parent directory
/// name with a leading `phaseN-`/`phaseN_` stripped, else the file stem.
#[must_use]
pub fn checkpoint_label(path: &str) -> String {
    let p = std::path::Path::new(path);
    let raw = p
        .parent()
        .and_then(std::path::Path::file_name)
        .or_else(|| p.file_stem())
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    // Strip a leading "phase<N>-" / "phase<N>_" ordering prefix if present.
    raw.split_once(['-', '_'])
        .filter(|(head, _)| {
            head.starts_with("phase") && head[5..].chars().all(|c| c.is_ascii_digit())
        })
        .map_or(raw.clone(), |(_, rest)| rest.to_string())
}

/// A named curriculum stage the run entered at `step`. Like [`Checkpoint`],
/// stage steps are segment boundaries for the phase-aware chart, but the label
/// is the explicit name from `run.stage(...)` rather than derived from a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMark {
    /// Step the stage began at.
    pub step: u64,
    /// Explicit stage name (segment label).
    pub label: String,
}

/// A phase transition: the run entered `phase` at `step` (for chart shading).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseSpan {
    /// Step at which this phase began.
    pub step: u64,
    /// Phase entered.
    pub phase: Phase,
}

/// A prior run's metric series, overlaid on the live chart for comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineSeries {
    /// Metric name (matched against the selected live metric's label).
    pub label: String,
    /// Step of each value (parallel to `values`).
    pub steps: Vec<u64>,
    /// Values (parallel to `steps`).
    pub values: Vec<f64>,
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
    /// Phase transitions in step order (background shading; capped FIFO).
    pub phases: VecDeque<PhaseSpan>,
    /// Checkpoints taken (vertical markers + phase-segment boundaries; capped FIFO).
    pub checkpoints: VecDeque<Checkpoint>,
    /// Named curriculum stages (explicit phase-segment boundaries; capped FIFO).
    /// When non-empty these take precedence over checkpoints as chart segments.
    pub stages: VecDeque<StageMark>,
    /// Prior-run series overlaid on the chart for comparison (empty if none).
    pub baseline: Vec<BaselineSeries>,
    labels: BTreeMap<u16, String>,
    max_history: usize,
    max_alerts: usize,
    max_markers: usize,
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
            phases: VecDeque::new(),
            checkpoints: VecDeque::new(),
            stages: VecDeque::new(),
            baseline: Vec::new(),
            labels: BTreeMap::new(),
            max_history: DEFAULT_HISTORY,
            max_alerts: DEFAULT_ALERTS,
            max_markers: DEFAULT_MARKERS,
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

    /// Sets the comparison baseline overlaid on the chart (from a prior run).
    pub fn set_baseline(&mut self, baseline: Vec<BaselineSeries>) {
        self.baseline = baseline;
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
                self.phase = *phase;
                self.phases.push_back(PhaseSpan {
                    step: self.step,
                    phase: *phase,
                });
                cap_front(&mut self.phases, self.max_markers);
            }
            Event::Alert(alert) => {
                self.alerts.push(alert.clone());
                if self.alerts.len() > self.max_alerts {
                    self.alerts.remove(0);
                }
            }
            Event::Checkpoint { step, path } => {
                self.checkpoints.push_back(Checkpoint {
                    step: *step,
                    label: checkpoint_label(path),
                });
                cap_front(&mut self.checkpoints, self.max_markers);
            }
            Event::StageChange { name, step } => {
                self.stages.push_back(StageMark {
                    step: *step,
                    label: name.clone(),
                });
                cap_front(&mut self.stages, self.max_markers);
            }
            Event::MetricsRegistered { names } => {
                for (id, name) in names {
                    self.labels.insert(id.index(), name.clone());
                    // Relabel any metric already shown under the `m{id}` fallback.
                    if let Some(m) = self.metrics.iter_mut().find(|m| m.id == *id) {
                        m.label.clone_from(name);
                    }
                }
            }
            Event::RunFinished { .. } => self.finished = true,
            Event::ConfigPatch(_) => {}
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

    fn record(&mut self, id: MetricId, value: f64, step: u64) {
        let label = self.label_for(id);
        let max_history = self.max_history;
        let view = if let Some(v) = self.metrics.iter_mut().find(|m| m.id == id) {
            v
        } else {
            self.metrics.push(MetricView {
                id,
                label: label.clone(),
                latest: value,
                history: VecDeque::new(),
                steps: VecDeque::new(),
            });
            self.metrics.last_mut().expect("just pushed")
        };
        view.label = label; // refresh in case labels were seeded after first use
        view.latest = value;
        view.history.push_back(value);
        view.steps.push_back(step);
        if view.history.len() > max_history {
            view.history.pop_front(); // O(1) FIFO drop, not O(n) shift
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

/// Chooses the chart y-scale and filters the baseline to it.
///
/// The scale is the **live** curve's own range (falling back to `0..1` when the
/// live curve is empty), so the current run is always readable regardless of how
/// large the comparison baseline's range is. Baseline points outside the live
/// range are dropped (hidden) rather than clamped to the border.
///
/// Returns the `(min, max)` scale, the in-range baseline (`None` if the metric
/// has no baseline *or* none of its points fall in range), and whether a
/// baseline existed but was entirely off-scale (so the title can say so).
type Baseline = (Vec<u64>, Vec<f64>);
fn chart_yscale(history: &[f64], base: Option<Baseline>) -> ((f64, f64), Option<Baseline>, bool) {
    let (g_min, g_max) = value_range(history).unwrap_or((0.0, 1.0));
    let Some((bs, bv)) = base else {
        return ((g_min, g_max), None, false);
    };
    let (mut fs, mut fv) = (Vec::new(), Vec::new());
    for (&st, &v) in bs.iter().zip(&bv) {
        if v.is_finite() && v >= g_min && v <= g_max {
            fs.push(st);
            fv.push(v);
        }
    }
    let had_baseline = !bv.is_empty();
    if fv.is_empty() {
        ((g_min, g_max), None, had_baseline)
    } else {
        ((g_min, g_max), Some((fs, fv)), false)
    }
}

/// Splits `[s0, s1]` at `marks` (each a `(boundary step, label)`, ascending)
/// into per-phase [`Segment`]s each y-scaled to its own data. Each segment is
/// labeled by the mark that ends it; the trailing live segment is `current`.
fn build_segments(
    values: &[f64],
    steps: &[u64],
    s0: u64,
    s1: u64,
    marks: &[(u64, String)],
) -> (Vec<Segment>, Vec<String>) {
    let range_in = |a: u64, b: u64| {
        let vs: Vec<f64> = steps
            .iter()
            .zip(values)
            .filter(|(&st, _)| st >= a && st <= b)
            .map(|(_, &v)| v)
            .collect();
        value_range(&vs).unwrap_or((0.0, 1.0))
    };
    let (mut segs, mut labels) = (Vec::new(), Vec::new());
    let mut start = s0;
    for (b, label) in marks {
        let (min, max) = range_in(start, *b);
        segs.push(Segment {
            start,
            end: *b,
            min,
            max,
        });
        labels.push(label.clone());
        start = *b;
    }
    let (min, max) = range_in(start, s1);
    segs.push(Segment {
        start,
        end: s1,
        min,
        max,
    });
    labels.push("current".to_string());
    (segs, labels)
}

fn render_chart(frame: &mut Frame, area: Rect, state: &UiState) {
    let Some(metric) = state.metrics.get(state.selected) else {
        frame.render_widget(block("Chart"), area);
        return;
    };
    // Inner drawable area excludes the one-cell border on each side.
    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;

    let steps: Vec<u64> = metric.steps.iter().copied().collect();
    // Per-step metrics (loss especially) are noisy enough that the raw curve
    // renders as unreadable confetti; smooth it into an EMA trend line.
    let raw: Vec<f64> = metric.history.iter().copied().collect();
    let history = ema_smooth(&raw, adaptive_ema_span(raw.len()));
    let s0 = steps.first().copied().unwrap_or(0);
    let s1 = steps.last().copied().unwrap_or(0);

    // Phase-aware: if stage/checkpoint boundaries split the window, scale each
    // segment to its own range so differently-scaled phases each read clearly.
    // Explicit `run.stage(...)` marks take precedence over checkpoint-derived
    // labels. (The baseline overlay can't share per-segment scales, so it's
    // omitted in this mode.)
    let marks: Vec<(u64, String)> = if state.stages.is_empty() {
        state
            .checkpoints
            .iter()
            .map(|c| (c.step, c.label.clone()))
            .collect()
    } else {
        state
            .stages
            .iter()
            .map(|s| (s.step, s.label.clone()))
            .collect()
    };
    let marks: Vec<(u64, String)> = marks
        .into_iter()
        .filter(|&(st, _)| st > s0 && st < s1)
        .collect();
    if !marks.is_empty() {
        let (segments, labels) = build_segments(&history, &steps, s0, s1, &marks);
        let chart = render_braille_segments(&history, &steps, inner_w, inner_h, s0, s1, &segments);
        let title = format!(
            "{} (latest {:.4})  ·  {}",
            metric.label,
            metric.latest,
            labels.join(" │ ")
        );
        let lines = compose_chart_lines(&chart, None, &state.phases, &state.checkpoints);
        frame.render_widget(Paragraph::new(lines).block(block(&title)), area);
        return;
    }

    // The baseline series matching this metric, clipped to the live step window
    // and smoothed the same way so the comparison is like-for-like.
    let base = state
        .baseline
        .iter()
        .find(|b| b.label == metric.label)
        .map(|b| {
            let (mut bs, mut bv) = (Vec::new(), Vec::new());
            for (&st, &v) in b.steps.iter().zip(&b.values) {
                if st >= s0 && st <= s1 {
                    bs.push(st);
                    bv.push(v);
                }
            }
            let bv = ema_smooth(&bv, adaptive_ema_span(bv.len()));
            (bs, bv)
        });

    // Scale the y-axis to the LIVE curve only, and keep the baseline only where
    // it overlaps that range — a wide-range baseline must not crush the run.
    let ((g_min, g_max), base_in_range, base_off_scale) = chart_yscale(&history, base);

    let live =
        render_braille_steps_scaled(&history, &steps, inner_w, inner_h, s0, s1, g_min, g_max);
    let base_chart = base_in_range.as_ref().map(|(bs, bv)| {
        render_braille_steps_scaled(bv, bs, inner_w, inner_h, s0, s1, g_min, g_max)
    });

    let title = if base_chart.is_some() {
        format!(
            "{} (latest {:.4})  ·  ╌ baseline",
            metric.label, metric.latest
        )
    } else if base_off_scale {
        format!(
            "{} (latest {:.4})  ·  baseline off-scale",
            metric.label, metric.latest
        )
    } else {
        format!("{} (latest {:.4})", metric.label, metric.latest)
    };
    let lines = compose_chart_lines(
        &live,
        base_chart.as_ref(),
        &state.phases,
        &state.checkpoints,
    );
    frame.render_widget(Paragraph::new(lines).block(block(&title)), area);
}

/// Builds the styled chart rows: the terracotta live curve over phase-tinted
/// backgrounds, an amber baseline curve behind it (where the live curve doesn't
/// occupy the cell), and dim checkpoint markers through remaining blank cells.
fn compose_chart_lines<'a>(
    live: &StepChart,
    base: Option<&StepChart>,
    phases: &VecDeque<PhaseSpan>,
    checkpoints: &VecDeque<Checkpoint>,
) -> Vec<Line<'a>> {
    if live.rows.is_empty() {
        return Vec::new();
    }
    let width = live.col_steps.len();
    let (s0, s1) = (
        live.col_steps.first().copied().unwrap_or(0),
        live.col_steps.last().copied().unwrap_or(0),
    );

    // Per-column background tint (by the phase active at that column's step).
    let col_bg: Vec<Option<Color>> = live
        .col_steps
        .iter()
        .map(|&st| phase_bg(phase_at(phases, st)))
        .collect();

    // Columns carrying a checkpoint marker, mapped onto the chart's own step
    // axis (`col_steps`) so they stay aligned with the curve.
    let mut is_ckpt = vec![false; width];
    for c in checkpoints.iter().map(|c| c.step) {
        if c < s0 || c > s1 {
            continue;
        }
        let col = live
            .col_steps
            .partition_point(|&s| s < c)
            .min(width.saturating_sub(1));
        if let Some(slot) = is_ckpt.get_mut(col) {
            *slot = true;
        }
    }

    live.rows
        .iter()
        .enumerate()
        .map(|(r, row)| {
            let live_cells: Vec<char> = row.chars().collect();
            let base_cells: Vec<char> = base
                .and_then(|b| b.rows.get(r))
                .map(|s| s.chars().collect())
                .unwrap_or_default();
            let mut spans: Vec<Span<'a>> = Vec::new();
            let mut buf = String::new();
            let mut cur: Option<Style> = None;
            for (c, &ch) in live_cells.iter().enumerate().take(width) {
                let base_ch = base_cells.get(c).copied().unwrap_or(BRAILLE_BLANK);
                // Live curve wins a cell; else the baseline shows; else a
                // checkpoint marker through blank space; else empty.
                let (glyph, fg) = if ch != BRAILLE_BLANK {
                    (ch, TERRACOTTA)
                } else if base_ch != BRAILLE_BLANK {
                    (base_ch, AMBER)
                } else if is_ckpt[c] {
                    ('\u{250a}', WARM_GRAY) // ┊
                } else {
                    (BRAILLE_BLANK, TERRACOTTA)
                };
                let mut style = Style::default().fg(fg);
                if let Some(bg) = col_bg[c] {
                    style = style.bg(bg);
                }
                if cur == Some(style) {
                    buf.push(glyph);
                } else {
                    if let Some(prev) = cur.take() {
                        spans.push(Span::styled(std::mem::take(&mut buf), prev));
                    }
                    buf.push(glyph);
                    cur = Some(style);
                }
            }
            if let Some(prev) = cur {
                spans.push(Span::styled(buf, prev));
            }
            Line::from(spans)
        })
        .collect()
}

/// The phase active at `step`: the most recent span at or before it (default
/// [`Phase::Train`] before any transition).
fn phase_at(phases: &VecDeque<PhaseSpan>, step: u64) -> Phase {
    phases
        .iter()
        .rev()
        .find(|s| s.step <= step)
        .map_or(Phase::Train, |s| s.phase)
}

/// Subtle background tint for a phase band (TRAIN is untinted — the default
/// panel). Dark, warm tints keep the cream/terracotta foreground readable.
fn phase_bg(phase: Phase) -> Option<Color> {
    match phase {
        Phase::Train => None,
        Phase::Eval => Some(Color::Rgb(0x2C, 0x25, 0x18)),
        Phase::Test => Some(Color::Rgb(0x2C, 0x20, 0x18)),
        Phase::Warmup => Some(Color::Rgb(0x24, 0x22, 0x20)),
        Phase::Checkpoint => Some(Color::Rgb(0x26, 0x24, 0x22)),
    }
}

/// Drops oldest elements from the front of `q` until it fits `max` (O(1) FIFO).
fn cap_front<T>(q: &mut VecDeque<T>, max: usize) {
    while q.len() > max {
        q.pop_front();
    }
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
        assert_eq!(
            s.metrics[0].history.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.5]
        );
    }

    #[test]
    fn metrics_registered_seeds_and_relabels() {
        let mut s = UiState::default();
        s.apply(&batch(0, &[(0, 1.0)])); // first seen under the m{id} fallback
        assert_eq!(s.metrics[0].label, "m0");
        // A name table relabels an already-tracked metric…
        s.apply(&Event::MetricsRegistered {
            names: vec![(MetricId(0), "loss".into())],
        });
        assert_eq!(s.metrics[0].label, "loss");
        // …and seeds names for metrics not yet seen.
        s.apply(&Event::MetricsRegistered {
            names: vec![(MetricId(1), "lr".into())],
        });
        s.apply(&batch(1, &[(1, 0.1)]));
        assert_eq!(s.metrics[1].label, "lr");
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
        assert_eq!(
            s.metrics[0].history.iter().copied().collect::<Vec<_>>(),
            vec![7.0, 8.0, 9.0]
        );
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

    #[test]
    fn tracks_steps_phases_and_checkpoints() {
        let mut s = UiState::default();
        s.apply(&batch(0, &[(0, 1.0)]));
        s.apply(&batch(5, &[(0, 0.9)]));
        s.apply(&Event::PhaseChange(Phase::Eval)); // transitions at step 5
        s.apply(&Event::Checkpoint {
            path: "/ckpt/6.pt".into(),
            step: 6,
        });
        // Steps are recorded parallel to history.
        assert_eq!(
            s.metrics[0].steps.iter().copied().collect::<Vec<_>>(),
            vec![0, 5]
        );
        // The phase span captures where EVAL began.
        assert_eq!(
            s.phases.iter().copied().collect::<Vec<_>>(),
            vec![PhaseSpan {
                step: 5,
                phase: Phase::Eval
            }]
        );
        // Checkpoints are recorded (previously dropped), with a derived label.
        assert_eq!(
            s.checkpoints.iter().map(|c| c.step).collect::<Vec<_>>(),
            vec![6]
        );
    }

    #[test]
    fn phase_at_resolves_active_span() {
        let spans: VecDeque<PhaseSpan> = [
            PhaseSpan {
                step: 10,
                phase: Phase::Eval,
            },
            PhaseSpan {
                step: 20,
                phase: Phase::Train,
            },
        ]
        .into_iter()
        .collect();
        assert_eq!(phase_at(&VecDeque::new(), 5), Phase::Train); // default
        assert_eq!(phase_at(&spans, 5), Phase::Train);
        assert_eq!(phase_at(&spans, 15), Phase::Eval);
        assert_eq!(phase_at(&spans, 25), Phase::Train);
        assert!(phase_bg(Phase::Train).is_none());
        assert!(phase_bg(Phase::Eval).is_some());
    }

    #[test]
    fn chart_draws_phase_band_and_checkpoint_marker() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        for step in 0..40u64 {
            s.apply(&batch(step, &[(0, 1.0 / (step as f64 + 1.0))]));
        }
        s.apply(&Event::PhaseChange(Phase::Eval)); // shades from step 39 onward
        s.apply(&Event::Checkpoint {
            path: "/ckpt/20.pt".into(),
            step: 20,
        });

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();
        let buffer = terminal.backend().buffer();

        // A checkpoint marker glyph is drawn somewhere in the chart.
        let text = buffer_text(terminal.backend());
        assert!(text.contains('\u{250a}'), "checkpoint marker rendered");
        // At least one cell carries the EVAL phase background tint.
        let eval_bg = phase_bg(Phase::Eval).unwrap();
        assert!(
            buffer.content().iter().any(|c| c.bg == eval_bg),
            "EVAL phase band shaded a cell background"
        );
    }

    #[test]
    fn chart_overlays_baseline_in_amber() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        for step in 0..40u64 {
            s.apply(&batch(step, &[(0, 1.0 / (step as f64 + 1.0))]));
        }
        // A baseline within the live curve's y-range (live is ~0.025..1.0), so
        // it renders on the shared live scale in its own amber cells.
        s.set_baseline(vec![BaselineSeries {
            label: "loss".into(),
            steps: (0..40).collect(),
            values: (0..40).map(|i| 0.8 - f64::from(i) * 0.01).collect(),
        }]);

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();
        let buffer = terminal.backend().buffer();

        // The chart title advertises the overlay, and some cell is amber.
        assert!(buffer_text(terminal.backend()).contains("baseline"));
        assert!(
            buffer.content().iter().any(|c| c.fg == AMBER),
            "baseline rendered in amber"
        );
    }

    #[test]
    fn checkpoint_label_strips_phase_prefix() {
        assert_eq!(
            checkpoint_label("out/phase1-reasoning/adapters.safetensors"),
            "reasoning"
        );
        assert_eq!(checkpoint_label("runs/phase12_code/x.pt"), "code");
        assert_eq!(checkpoint_label("ckpts/warmup/x.pt"), "warmup"); // no phaseN prefix
        assert_eq!(checkpoint_label("step_200.pt"), "step_200"); // file stem fallback
    }

    #[test]
    fn build_segments_splits_and_labels_by_checkpoint() {
        let steps: Vec<u64> = (0..=9).collect();
        let values: Vec<f64> = vec![2.0, 1.8, 1.6, 1.4, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5];
        let marks = vec![(4u64, "reasoning".to_string())];
        let (segs, labels) = build_segments(&values, &steps, 0, 9, &marks);
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start, segs[0].end), (0, 4));
        assert_eq!((segs[1].start, segs[1].end), (4, 9));
        // Each segment scaled to its own data range.
        assert!((segs[0].max - 5.0).abs() < 1e-9); // includes the step-4 spike
        assert!(segs[1].max <= 5.0 && segs[1].min <= 1.0);
        assert_eq!(labels, vec!["reasoning".to_string(), "current".to_string()]);
    }

    #[test]
    fn segmented_chart_titles_phases_from_checkpoints() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        for step in 0..10u64 {
            s.apply(&batch(
                step,
                &[(0, 2.0 - f64::from(u32::try_from(step).unwrap()) * 0.1)],
            ));
        }
        s.apply(&Event::Checkpoint {
            path: "out/phase1-reasoning/adapters.safetensors".into(),
            step: 4,
        });
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();
        let text = buffer_text(terminal.backend());
        assert!(text.contains("reasoning"), "segment label in title");
        assert!(text.contains("current"), "live segment labeled");
    }

    #[test]
    fn chart_yscale_uses_live_range_and_hides_off_scale_baseline() {
        // Real-world shape: live loss 0.6..2.0, baseline loss all far above.
        let live = vec![2.0, 1.5, 1.0, 0.8, 0.6];
        let base = (vec![0, 1, 2, 3, 4], vec![5.5, 4.0, 3.0, 2.5, 2.2]);
        let ((lo, hi), filtered, off_scale) = chart_yscale(&live, Some(base));
        assert_eq!((lo, hi), (0.6, 2.0)); // scale = live range, NOT union up to 5.5
        assert!(filtered.is_none()); // baseline entirely above range -> hidden
        assert!(off_scale);
    }

    #[test]
    fn chart_yscale_keeps_in_range_baseline_points() {
        let live = vec![2.0, 1.0, 0.6];
        let base = (vec![0, 1, 2], vec![1.8, 1.2, 5.0]); // 5.0 is off-scale
        let ((lo, hi), filtered, off_scale) = chart_yscale(&live, Some(base));
        assert_eq!((lo, hi), (0.6, 2.0));
        let (fs, fv) = filtered.expect("in-range baseline points survive");
        assert_eq!(fv, vec![1.8, 1.2]); // 5.0 dropped
        assert_eq!(fs, vec![0, 1]);
        assert!(!off_scale);
    }

    #[test]
    fn baseline_only_overlays_when_label_matches() {
        let mut s = UiState::with_labels(&[(MetricId(0), "loss")]);
        for step in 0..20u64 {
            s.apply(&batch(step, &[(0, 1.0 / (step as f64 + 1.0))]));
        }
        // Baseline for a different metric — must not appear on the loss chart.
        s.set_baseline(vec![BaselineSeries {
            label: "accuracy".into(),
            steps: (0..20).collect(),
            values: (0..20).map(|_| 5.0).collect(),
        }]);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, &s)).unwrap();
        let buffer = terminal.backend().buffer();
        assert!(!buffer_text(terminal.backend()).contains("baseline"));
        assert!(buffer.content().iter().all(|c| c.fg != AMBER));
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
