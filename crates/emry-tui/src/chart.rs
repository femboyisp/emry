//! Spike-preserving chart primitives: min/max downsampling and a braille
//! renderer.
//!
//! Loss curves have sparse, important spikes; averaging hides them. [`downsample_minmax`]
//! keeps the min **and** max of each bucket so a single bad step survives heavy
//! compression. [`render_braille`] turns a series into a compact band chart using
//! Unicode braille (a 2×4 dot grid per character cell), which the TUI (EMRY-021)
//! wraps in a `ratatui` widget.

/// Compresses `data` into `buckets` (min, max) pairs, left to right.
///
/// Each bucket reports the minimum and maximum of its slice, so spikes are never
/// averaged away. If `data` has fewer points than `buckets`, each point becomes
/// its own `(v, v)` pair. Returns empty for empty input or zero buckets.
///
/// Non-finite values are ignored within a bucket; a bucket with no finite value
/// reports `(0.0, 0.0)`.
#[must_use]
pub fn downsample_minmax(data: &[f64], buckets: usize) -> Vec<(f64, f64)> {
    if buckets == 0 || data.is_empty() {
        return Vec::new();
    }
    if data.len() <= buckets {
        return data
            .iter()
            .map(|&v| if v.is_finite() { (v, v) } else { (0.0, 0.0) })
            .collect();
    }
    (0..buckets)
        .map(|b| {
            let start = b * data.len() / buckets;
            let end = (((b + 1) * data.len() / buckets).max(start + 1)).min(data.len());
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for &v in &data[start..end] {
                if v.is_finite() {
                    min = min.min(v);
                    max = max.max(v);
                }
            }
            if min.is_finite() {
                (min, max)
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
}

/// Braille dot bit for a sub-cell `(col, row)`, `col` in `0..2`, `row` in `0..4`
/// (row 0 = top). Returns 0 for out-of-range coordinates.
fn dot_bit(col: usize, row: usize) -> u8 {
    match (col, row) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (0, 3) => 0x40,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        (1, 3) => 0x80,
        _ => 0,
    }
}

/// Renders `data` as a braille band chart of `width` × `height` character cells.
///
/// Returns `height` strings (top row first), each `width` characters wide. Each
/// cell packs a 2×4 dot grid, so the effective resolution is `2*width` columns
/// by `4*height` rows. Within each column the dots are filled from the bucket's
/// min to its max, drawing a vertical band that makes spikes visible. Values are
/// normalised against the global min/max of `data`. Empty input yields blank rows.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // normalised values are small, non-negative, and bounded by the dot grid
pub fn render_braille(data: &[f64], width: usize, height: usize) -> Vec<String> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let blank = || vec![" ".repeat(width); height];

    let dot_w = width * 2;
    let dot_h = height * 4;
    let columns = downsample_minmax(data, dot_w);
    if columns.is_empty() {
        return blank();
    }

    // Normalise against the actual finite data range, not the bucket outputs:
    // an all-non-finite bucket reports a (0,0) sentinel that would otherwise
    // distort the scale.
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    for &v in data {
        if v.is_finite() {
            g_min = g_min.min(v);
            g_max = g_max.max(v);
        }
    }
    if !g_min.is_finite() {
        return blank(); // no finite data to plot
    }
    let span = if (g_max - g_min).abs() < f64::EPSILON {
        1.0
    } else {
        g_max - g_min
    };
    let top_dot = (dot_h - 1) as f64;

    let mut cells = vec![vec![0u8; width]; height];
    for (dot_col, &(min, max)) in columns.iter().enumerate() {
        let cell_col = dot_col / 2;
        let sub_col = dot_col % 2;
        // Map values to dot rows measured from the bottom (0 = lowest value).
        // Clamp to the grid: a (0,0) sentinel column (all-non-finite bucket) can
        // fall outside the data range, which would otherwise underflow from_top.
        let to_row = |v: f64| (((v - g_min) / span) * top_dot).round().clamp(0.0, top_dot) as usize;
        let y_lo = to_row(min);
        let y_hi = to_row(max);
        for y in y_lo..=y_hi {
            let from_top = (dot_h - 1) - y;
            cells[from_top / 4][cell_col] |= dot_bit(sub_col, from_top % 4);
        }
    }

    cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|bits| char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' '))
                .collect()
        })
        .collect()
}

/// The blank braille cell (`U+2800`), an "empty" column with no plotted dots.
pub const BRAILLE_BLANK: char = '\u{2800}';

/// A step-based braille chart: the rendered rows plus the step value at each
/// character column, so callers can align phase bands and checkpoint markers to
/// the same x-axis the curve uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepChart {
    /// Braille rows, top first (each `width` chars wide).
    pub rows: Vec<String>,
    /// Step value at each character column (length `width`).
    pub col_steps: Vec<u64>,
}

/// The finite min/max of `values`, or `None` if there are no finite values.
#[must_use]
pub fn value_range(values: &[f64]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    min.is_finite().then_some((min, max))
}

/// EMA smoothing span (in points) adaptive to a series' length: ~8% of the
/// points, clamped to `[3, 64]`. Short series get light smoothing, long ones
/// more, so both read as a clean trend rather than raw per-step noise.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn adaptive_ema_span(n: usize) -> usize {
    ((n as f64) * 0.08).round().clamp(3.0, 64.0) as usize
}

/// Exponential moving average over `values` with `span` (`alpha = 2/(span+1)`).
///
/// Returns a vec the same length as `values` (empty in → empty out). Non-finite
/// values carry the previous smoothed value forward, so a single NaN/Inf can't
/// wreck the line; leading non-finite entries stay non-finite (drawn blank).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ema_smooth(values: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span.max(1) as f64 + 1.0);
    let mut out = Vec::with_capacity(values.len());
    let mut acc = f64::NAN;
    for &v in values {
        if v.is_finite() {
            acc = if acc.is_finite() {
                alpha * v + (1.0 - alpha) * acc
            } else {
                v
            };
        }
        out.push(acc);
    }
    out
}

/// Linear interpolation of the value at step `st` over ascending `pts`
/// (step, value), clamped to the endpoints outside the point range.
#[allow(clippy::cast_precision_loss)]
fn interp_at(pts: &[(u64, f64)], st: u64) -> f64 {
    let (first_s, first_v) = pts[0];
    let (last_s, last_v) = pts[pts.len() - 1];
    if st <= first_s {
        return first_v;
    }
    if st >= last_s {
        return last_v;
    }
    let i = pts.partition_point(|&(s, _)| s <= st); // first index with s > st
    let (s_hi, v_hi) = pts[i];
    let (s_lo, v_lo) = pts[i - 1];
    if s_hi == s_lo {
        return v_hi;
    }
    let f = (st - s_lo) as f64 / (s_hi - s_lo) as f64;
    v_lo + (v_hi - v_lo) * f
}

/// Renders `values` (with their parallel `steps`, ascending) as a step-based
/// braille band chart, auto-scaling the y-axis to the data. See
/// [`render_braille_steps_scaled`] for the shared-scale variant used to overlay
/// two series.
#[must_use]
pub fn render_braille_steps(
    values: &[f64],
    steps: &[u64],
    width: usize,
    height: usize,
) -> StepChart {
    let s0 = steps.first().copied().unwrap_or(0);
    let s1 = steps.last().copied().unwrap_or(0);
    let (g_min, g_max) = value_range(values).unwrap_or((f64::NAN, f64::NAN));
    render_braille_steps_scaled(values, steps, width, height, s0, s1, g_min, g_max)
}

/// Like [`render_braille_steps`] but with an explicit step window `[s0, s1]` and
/// y-scale `[g_min, g_max]`, so two series can be drawn on the **same** axes and
/// overlaid (e.g. a live curve and a comparison baseline). Points are placed by
/// step within the window; empty columns carry the previous forward. A
/// non-finite scale yields blank rows.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)] // values are normalised/bounded; the args are a deliberate scale+window
pub fn render_braille_steps_scaled(
    values: &[f64],
    steps: &[u64],
    width: usize,
    height: usize,
    s0: u64,
    s1: u64,
    g_min: f64,
    g_max: f64,
) -> StepChart {
    if width == 0 || height == 0 {
        return StepChart {
            rows: Vec::new(),
            col_steps: Vec::new(),
        };
    }
    let step_span = s1.saturating_sub(s0).max(1) as f64;
    let col_steps: Vec<u64> = (0..width)
        .map(|c| {
            if width == 1 {
                s0
            } else {
                s0 + ((c as f64 / (width - 1) as f64) * step_span).round() as u64
            }
        })
        .collect();
    let blank = || StepChart {
        rows: vec![BRAILLE_BLANK.to_string().repeat(width); height],
        col_steps: col_steps.clone(),
    };
    if values.is_empty() || steps.len() != values.len() || !g_min.is_finite() || !g_max.is_finite()
    {
        return blank();
    }

    let dot_w = width * 2;
    let dot_h = height * 4;

    // Finite (step, value) points inside the window, ascending by step.
    let pts: Vec<(u64, f64)> = steps
        .iter()
        .zip(values)
        .filter(|(&st, &v)| v.is_finite() && st >= s0 && st <= s1)
        .map(|(&st, &v)| (st, v))
        .collect();
    if pts.is_empty() {
        return blank();
    }

    let span = if (g_max - g_min).abs() < f64::EPSILON {
        1.0
    } else {
        g_max - g_min
    };
    let top_dot = (dot_h - 1) as f64;
    let to_row = |v: f64| (((v - g_min) / span) * top_dot).round().clamp(0.0, top_dot) as usize;

    // Draw a *connected* polyline: interpolate the series at each dot column's
    // step, then bridge each column's row to the previous column's. A bare
    // scatter of points reads as noise at tall panel heights; joining them into
    // a line reads as a curve.
    let mut cells = vec![vec![0u8; width]; height];
    let mut prev_row: Option<usize> = None;
    for dot_col in 0..dot_w {
        let st = if dot_w == 1 {
            s0
        } else {
            s0 + ((dot_col as f64 / (dot_w - 1) as f64) * step_span).round() as u64
        };
        let y = to_row(interp_at(&pts, st));
        let (lo, hi) = prev_row.map_or((y, y), |p| (p.min(y), p.max(y)));
        let cell_col = dot_col / 2;
        let sub_col = dot_col % 2;
        for yy in lo..=hi {
            let from_top = (dot_h - 1) - yy;
            cells[from_top / 4][cell_col] |= dot_bit(sub_col, from_top % 4);
        }
        prev_row = Some(y);
    }

    let rows = cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|bits| char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' '))
                .collect()
        })
        .collect();

    StepChart { rows, col_steps }
}

/// One x-axis segment with its own y-scale, for the phase-aware chart. Steps in
/// `[start, end]` are scaled against this segment's `[min, max]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// First step of the segment (inclusive).
    pub start: u64,
    /// Last step of the segment (inclusive).
    pub end: u64,
    /// y-axis minimum for this segment.
    pub min: f64,
    /// y-axis maximum for this segment.
    pub max: f64,
}

/// Renders a connected polyline where each x-`segment` is scaled to its **own**
/// y-range, so phases with different loss scales are each readable. The line
/// breaks at segment boundaries (leaving a visual divider). `segments` must be
/// non-empty and cover `[s0, s1]` left to right.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
pub fn render_braille_segments(
    values: &[f64],
    steps: &[u64],
    width: usize,
    height: usize,
    s0: u64,
    s1: u64,
    segments: &[Segment],
) -> StepChart {
    let step_span = s1.saturating_sub(s0).max(1) as f64;
    let col_steps: Vec<u64> = (0..width)
        .map(|c| {
            if width <= 1 {
                s0
            } else {
                s0 + ((c as f64 / (width - 1) as f64) * step_span).round() as u64
            }
        })
        .collect();
    let blank = StepChart {
        rows: vec![BRAILLE_BLANK.to_string().repeat(width); height],
        col_steps: col_steps.clone(),
    };
    if width == 0 || height == 0 || segments.is_empty() || steps.len() != values.len() {
        return blank;
    }
    let pts: Vec<(u64, f64)> = steps
        .iter()
        .zip(values)
        .filter(|(&st, &v)| v.is_finite() && st >= s0 && st <= s1)
        .map(|(&st, &v)| (st, v))
        .collect();
    if pts.is_empty() {
        return blank;
    }

    let dot_w = width * 2;
    let dot_h = height * 4;
    let top_dot = (dot_h - 1) as f64;
    let seg_for = |st: u64| {
        segments
            .iter()
            .position(|s| st >= s.start && st <= s.end)
            .unwrap_or(segments.len() - 1)
    };

    let mut cells = vec![vec![0u8; width]; height];
    let mut prev: Option<(usize, usize)> = None; // (segment index, row)
    for dot_col in 0..dot_w {
        let st = if dot_w == 1 {
            s0
        } else {
            s0 + ((dot_col as f64 / (dot_w - 1) as f64) * step_span).round() as u64
        };
        let si = seg_for(st);
        let seg = &segments[si];
        let span = if (seg.max - seg.min).abs() < f64::EPSILON {
            1.0
        } else {
            seg.max - seg.min
        };
        let y = (((interp_at(&pts, st) - seg.min) / span) * top_dot)
            .round()
            .clamp(0.0, top_dot) as usize;
        // Connect within a segment; break (divider gap) across boundaries.
        let (lo, hi) = match prev {
            Some((ps, pr)) if ps == si => (pr.min(y), pr.max(y)),
            _ => (y, y),
        };
        let cell_col = dot_col / 2;
        let sub_col = dot_col % 2;
        for yy in lo..=hi {
            let from_top = (dot_h - 1) - yy;
            cells[from_top / 4][cell_col] |= dot_bit(sub_col, from_top % 4);
        }
        prev = Some((si, y));
    }

    let rows = cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|bits| char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' '))
                .collect()
        })
        .collect();
    StepChart { rows, col_steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_braille_segments_renders_and_handles_empty() {
        let steps: Vec<u64> = (0..10).collect();
        // Segment A: flat ~1.0; segment B: swings 0..5 — each autoscaled to itself.
        let values = vec![1.0, 1.1, 1.0, 1.1, 1.0, 0.0, 5.0, 0.0, 5.0, 0.0];
        let segs = vec![
            Segment {
                start: 0,
                end: 4,
                min: 1.0,
                max: 1.1,
            },
            Segment {
                start: 5,
                end: 9,
                min: 0.0,
                max: 5.0,
            },
        ];
        let chart = render_braille_segments(&values, &steps, 40, 8, 0, 9, &segs);
        assert_eq!(chart.rows.len(), 8);
        assert_eq!(chart.col_steps.len(), 40);
        let plotted = |cols: std::ops::Range<usize>| {
            chart.rows.iter().any(|r| {
                r.chars()
                    .skip(cols.start)
                    .take(cols.len())
                    .any(|c| c != BRAILLE_BLANK)
            })
        };
        assert!(plotted(0..20), "left segment drew something");
        assert!(plotted(20..40), "right segment drew something");
        // No segments -> blank.
        let blank = render_braille_segments(&values, &steps, 40, 8, 0, 9, &[]);
        assert!(blank
            .rows
            .iter()
            .all(|r| r.chars().all(|c| c == BRAILLE_BLANK)));
    }

    #[test]
    fn ema_smooth_reduces_step_to_step_variance() {
        // A series oscillating across its full range (like real per-step loss).
        let noisy = vec![1.0, 2.0, 0.5, 1.8, 0.6, 1.9, 0.7, 1.7];
        let sm = ema_smooth(&noisy, 5);
        assert_eq!(sm.len(), noisy.len());
        let total_var = |xs: &[f64]| xs.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>();
        assert!(
            total_var(&sm) < total_var(&noisy),
            "smoothed line jitters less than the raw"
        );
    }

    #[test]
    fn ema_smooth_handles_empty_and_nonfinite() {
        assert!(ema_smooth(&[], 5).is_empty());
        let sm = ema_smooth(&[f64::NAN, 1.0, f64::INFINITY, 2.0], 3);
        assert!(sm[0].is_nan(), "leading non-finite stays blank");
        assert!(sm[1].is_finite(), "first finite seeds the average");
        assert!(sm[2].is_finite(), "Inf is carried over, not propagated");
        assert!(sm[3].is_finite());
    }

    #[test]
    fn adaptive_ema_span_scales_and_clamps() {
        assert_eq!(adaptive_ema_span(0), 3); // clamp low
        assert_eq!(adaptive_ema_span(10), 3); // 0.8 -> clamp 3
        assert_eq!(adaptive_ema_span(500), 40); // ~8%
        assert_eq!(adaptive_ema_span(100_000), 64); // clamp high
    }

    #[test]
    fn downsample_empty_or_zero_buckets_is_empty() {
        assert!(downsample_minmax(&[], 10).is_empty());
        assert!(downsample_minmax(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn downsample_short_series_is_identity() {
        let out = downsample_minmax(&[1.0, 2.0, 3.0], 10);
        assert_eq!(out, vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
    }

    #[test]
    fn downsample_produces_requested_bucket_count() {
        let data: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(downsample_minmax(&data, 80).len(), 80);
    }

    #[test]
    fn spike_survives_1000_to_80_compression() {
        // A flat baseline with one tall spike.
        let mut data = vec![0.0_f64; 1000];
        data[503] = 100.0;
        let buckets = downsample_minmax(&data, 80);
        assert_eq!(buckets.len(), 80);
        // Exactly one bucket carries the spike as its max; the rest stay flat.
        let spiked: Vec<_> = buckets.iter().filter(|(_, max)| *max >= 100.0).collect();
        assert_eq!(spiked.len(), 1, "spike preserved in exactly one bucket");
        assert!(buckets.iter().all(|(min, _)| *min == 0.0));
    }

    #[test]
    fn downsample_ignores_non_finite() {
        let out = downsample_minmax(&[1.0, f64::NAN, 3.0, f64::INFINITY, 2.0], 2);
        // First bucket [1, NaN] -> min/max from 1 only; second [3, Inf, 2] -> 2..3.
        assert_eq!(out[0], (1.0, 1.0));
        assert_eq!(out[1], (2.0, 3.0));
    }

    #[test]
    fn render_dimensions_match_request() {
        let data: Vec<f64> = (0..200).map(f64::from).collect();
        let rows = render_braille(&data, 40, 6);
        assert_eq!(rows.len(), 6);
        assert!(rows.iter().all(|r| r.chars().count() == 40));
    }

    #[test]
    fn render_empty_or_zero_size_is_blank_or_empty() {
        assert!(render_braille(&[], 0, 5).is_empty());
        let blank = render_braille(&[], 10, 3);
        assert_eq!(blank.len(), 3);
        assert!(blank.iter().all(|r| r.chars().all(|c| c == ' ')));
    }

    #[test]
    fn rendered_chars_are_all_braille() {
        let data: Vec<f64> = (0..100).map(|i| f64::from(i).sin()).collect();
        for row in render_braille(&data, 20, 4) {
            assert!(row
                .chars()
                .all(|c| ('\u{2800}'..='\u{28FF}').contains(&c) || c == ' '));
        }
    }

    #[test]
    fn spike_renders_in_a_higher_row_than_baseline() {
        // Flat low baseline with one spike; the spike's column should light up
        // the top row while baseline columns only light the bottom row.
        let mut data = vec![0.0_f64; 200];
        data[100] = 50.0;
        let rows = render_braille(&data, 20, 4);
        // Top row has at least one non-blank cell (the spike reaches the top).
        assert!(rows[0].chars().any(|c| c != '\u{2800}' && c != ' '));
        // Bottom row is non-blank too (the baseline sits at the bottom).
        assert!(rows[3].chars().any(|c| c != '\u{2800}' && c != ' '));
    }

    #[test]
    fn short_series_filters_non_finite() {
        // len <= buckets short-circuit path must also reject non-finite.
        let out = downsample_minmax(&[1.0, f64::NAN, f64::INFINITY], 10);
        assert_eq!(out, vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)]);
    }

    #[test]
    fn render_normalizes_against_data_not_sentinels() {
        // Large finite values plus an all-NaN region must not be rescaled by the
        // (0,0) sentinel, and must not panic when 0 is outside the data range.
        let mut data = vec![1_000_000.0_f64; 200];
        for v in data.iter_mut().take(20) {
            *v = f64::NAN;
        }
        let rows = render_braille(&data, 20, 4);
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.chars().count() == 20));
    }

    #[test]
    fn render_all_non_finite_is_blank() {
        let rows = render_braille(&[f64::NAN, f64::INFINITY], 10, 3);
        assert!(rows.iter().all(|r| r.chars().all(|c| c == ' ')));
    }

    #[test]
    fn flat_series_renders_without_panicking() {
        // Zero span must not divide by zero.
        let rows = render_braille(&[5.0; 50], 10, 3);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn step_chart_dimensions_and_col_steps() {
        let values: Vec<f64> = (0..100).map(f64::from).collect();
        let steps: Vec<u64> = (0..100u64).map(|i| i * 10).collect(); // 0..990
        let chart = render_braille_steps(&values, &steps, 40, 6);
        assert_eq!(chart.rows.len(), 6);
        assert!(chart.rows.iter().all(|r| r.chars().count() == 40));
        assert_eq!(chart.col_steps.len(), 40);
        // x-axis spans [first step, last step], ascending.
        assert_eq!(chart.col_steps[0], 0);
        assert_eq!(*chart.col_steps.last().unwrap(), 990);
        assert!(chart.col_steps.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn step_chart_places_points_by_step_not_index() {
        // Two points far apart in step land at the left and right edges, with a
        // continuous (carried-forward) line between them.
        let chart = render_braille_steps(&[1.0, 2.0], &[0, 1000], 20, 4);
        assert_eq!(chart.rows.len(), 4);
        // Every column has some dot (carry-forward fills the gap), so no fully
        // blank column between the two endpoints.
        let has_dot = |c: usize| {
            chart
                .rows
                .iter()
                .any(|r| r.chars().nth(c) != Some(BRAILLE_BLANK))
        };
        assert!(has_dot(0) && has_dot(19));
    }

    #[test]
    fn step_chart_handles_edge_cases() {
        assert!(render_braille_steps(&[], &[], 10, 3)
            .rows
            .iter()
            .all(|r| r.chars().all(|c| c == BRAILLE_BLANK)));
        assert!(render_braille_steps(&[1.0], &[5], 0, 3).rows.is_empty());
        // Mismatched lengths fall back to blank rather than panicking.
        assert_eq!(render_braille_steps(&[1.0, 2.0], &[0], 8, 2).rows.len(), 2);
        // Single point / single column does not divide by zero.
        let one = render_braille_steps(&[3.0], &[7], 1, 2);
        assert_eq!(one.col_steps, vec![7]);
    }
}
