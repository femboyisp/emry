//! Baseline run loader for the comparison overlay.
//!
//! Reads a completed run's `metrics.jsonl` into per-metric `(steps, values)`
//! series. The dashboard fetches it once and overlays the matching metric as a
//! dashed line behind the live curve, so a run can be compared against a prior
//! one at the same training step.

use emry_core::MetricRecord;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

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

/// A loaded baseline run: its metric series, keyed by name.
#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct Baseline {
    /// Series in first-seen order.
    pub metrics: Vec<BaselineSeries>,
}

/// Loads a baseline from a run's `metrics.jsonl` (`path` is the file, or a run
/// directory containing it). Unparseable lines are skipped.
///
/// # Errors
///
/// Returns [`std::io::Error`] if the file cannot be read.
pub fn load_baseline(path: &Path) -> std::io::Result<Baseline> {
    let file = if path.is_dir() {
        path.join("metrics.jsonl")
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(file)?;

    let mut order: Vec<String> = Vec::new();
    let mut series: BTreeMap<String, (Vec<u64>, Vec<f64>)> = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<MetricRecord>(line) else {
            continue; // skip a malformed / third-party-incompatible row
        };
        for (name, value) in record.values {
            let entry = series.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                (Vec::new(), Vec::new())
            });
            entry.0.push(record.step);
            entry.1.push(value);
        }
    }

    let metrics = order
        .into_iter()
        .map(|label| {
            let (steps, values) = series.remove(&label).unwrap_or_default();
            BaselineSeries {
                label,
                steps,
                values,
            }
        })
        .collect();
    Ok(Baseline { metrics })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_metrics(lines: &[&str]) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("emry-baseline-{}-{n}.jsonl", std::process::id()));
        let mut f = std::fs::File::create(&p).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        p
    }

    #[test]
    fn loads_series_per_metric_in_order() {
        let p = temp_metrics(&[
            r#"{"step":0,"epoch":0,"phase":"TRAIN","values":{"loss":1.0,"lr":0.1}}"#,
            r#"{"step":1,"epoch":0,"phase":"TRAIN","values":{"loss":0.5,"lr":0.1}}"#,
            "",
            "garbage not json",
            r#"{"step":2,"epoch":0,"phase":"TRAIN","values":{"loss":0.25,"lr":0.1}}"#,
        ]);
        let b = load_baseline(&p).unwrap();
        std::fs::remove_file(&p).ok();
        let loss = b.metrics.iter().find(|m| m.label == "loss").unwrap();
        assert_eq!(loss.steps, vec![0, 1, 2]);
        assert_eq!(loss.values, vec![1.0, 0.5, 0.25]); // garbage/blank skipped
        assert!(b.metrics.iter().any(|m| m.label == "lr"));
    }

    #[test]
    fn missing_file_errors() {
        assert!(load_baseline(Path::new("/no/such/emry/metrics.jsonl")).is_err());
    }

    #[test]
    fn serializes_for_the_browser() {
        let p = temp_metrics(&[r#"{"step":5,"epoch":0,"phase":"TRAIN","values":{"acc":0.9}}"#]);
        let b = load_baseline(&p).unwrap();
        std::fs::remove_file(&p).ok();
        let json = serde_json::to_string(&b).unwrap();
        assert!(
            json.contains("\"acc\"") && json.contains("\"steps\"") && json.contains("\"values\"")
        );
    }
}
