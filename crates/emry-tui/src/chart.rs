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

/// Renders `values` (with their parallel `steps`, ascending) as a braille band
/// chart whose x-axis is **step-based** — each point lands in the column for its
/// step, so the curve aligns with phase bands and checkpoint markers drawn at
/// the same steps. Empty columns carry the previous column forward so the line
/// stays continuous. `steps` must be the same length as `values`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // normalised values are small/bounded; step indices fit the grid
pub fn render_braille_steps(
    values: &[f64],
    steps: &[u64],
    width: usize,
    height: usize,
) -> StepChart {
    if width == 0 || height == 0 {
        return StepChart {
            rows: Vec::new(),
            col_steps: Vec::new(),
        };
    }
    let blank = StepChart {
        rows: vec![BRAILLE_BLANK.to_string().repeat(width); height],
        col_steps: vec![0; width],
    };
    if values.is_empty() || steps.len() != values.len() {
        return blank;
    }

    let s0 = steps[0];
    let s1 = steps[steps.len() - 1];
    let step_span = s1.saturating_sub(s0).max(1) as f64;

    let dot_w = width * 2;
    let dot_h = height * 4;

    // Accumulate (min, max) per dot column, keyed by each point's step.
    let mut cols: Vec<Option<(f64, f64)>> = vec![None; dot_w];
    for (&v, &st) in values.iter().zip(steps) {
        if !v.is_finite() {
            continue;
        }
        let frac = (st.saturating_sub(s0) as f64) / step_span;
        let col = (frac * (dot_w - 1) as f64)
            .round()
            .clamp(0.0, (dot_w - 1) as f64) as usize;
        if let Some(slot) = cols.get_mut(col) {
            *slot = Some(match *slot {
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
                None => (v, v),
            });
        }
    }
    // Carry the last seen column forward across gaps so the line is continuous.
    let mut last: Option<(f64, f64)> = None;
    for slot in &mut cols {
        if slot.is_none() {
            *slot = last;
        } else {
            last = *slot;
        }
    }

    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            g_min = g_min.min(v);
            g_max = g_max.max(v);
        }
    }
    if !g_min.is_finite() {
        return blank;
    }
    let span = if (g_max - g_min).abs() < f64::EPSILON {
        1.0
    } else {
        g_max - g_min
    };
    let top_dot = (dot_h - 1) as f64;

    let mut cells = vec![vec![0u8; width]; height];
    for (dot_col, slot) in cols.iter().enumerate() {
        let Some((min, max)) = *slot else { continue };
        let cell_col = dot_col / 2;
        let sub_col = dot_col % 2;
        let to_row = |v: f64| (((v - g_min) / span) * top_dot).round().clamp(0.0, top_dot) as usize;
        for y in to_row(min)..=to_row(max) {
            let from_top = (dot_h - 1) - y;
            cells[from_top / 4][cell_col] |= dot_bit(sub_col, from_top % 4);
        }
    }

    let rows = cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|bits| char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' '))
                .collect()
        })
        .collect();

    // Step at the centre of each character column.
    let col_steps = (0..width)
        .map(|c| {
            if width == 1 {
                s0
            } else {
                s0 + ((c as f64 / (width - 1) as f64) * step_span).round() as u64
            }
        })
        .collect();

    StepChart { rows, col_steps }
}

#[cfg(test)]
mod tests {
    use super::*;

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
