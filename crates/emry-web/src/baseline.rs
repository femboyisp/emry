//! Comparison-baseline types served to the dashboard.
//!
//! A prior run's metric series, fetched once by the browser and overlaid on the
//! live chart. The series are loaded by [`emry_store::load_baseline`] (the single
//! canonical `metrics.jsonl` reader) and mapped into these `Serialize` types at
//! the composition root (the CLI / the demo).

use serde::Serialize;

/// One metric's full series from the baseline run.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BaselineSeries {
    /// Metric name.
    pub label: String,
    /// Step of each value (parallel to `values`).
    pub steps: Vec<u64>,
    /// Values (parallel to `steps`).
    pub values: Vec<f64>,
}

/// A loaded baseline run: its metric series, in first-seen order.
#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct Baseline {
    /// Series in first-seen order.
    pub metrics: Vec<BaselineSeries>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_for_the_browser() {
        let baseline = Baseline {
            metrics: vec![BaselineSeries {
                label: "acc".into(),
                steps: vec![0, 5],
                values: vec![0.5, 0.9],
            }],
        };
        let json = serde_json::to_string(&baseline).unwrap();
        assert!(
            json.contains("\"acc\"") && json.contains("\"steps\"") && json.contains("\"values\"")
        );
    }

    #[test]
    fn default_is_empty() {
        assert!(Baseline::default().metrics.is_empty());
    }
}
