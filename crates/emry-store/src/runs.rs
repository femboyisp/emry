//! Reading run directories for the `emry runs` and `emry compare` commands.
//!
//! [`list_runs`] scans a base log directory for run subdirectories (those with a
//! `run.meta`) and summarizes each from `run.meta` + the optional `summary.json`.
//! [`final_metrics`] reads a run's `metrics.jsonl` and returns the last value
//! seen for each metric, for side-by-side comparison.

use crate::meta::{RunMetaFile, Summary, RUN_META_FILE, SUMMARY_FILE};
use crate::writer::METRICS_FILE;
use emry_core::{EmryError, MetricRecord};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A summarized run, as shown by `emry runs`.
#[derive(Debug, Clone, PartialEq)]
pub struct RunInfo {
    /// Run-directory name (`{project}_{timestamp}`).
    pub dir_name: String,
    /// Project / experiment name.
    pub project: String,
    /// Unix start time (seconds); used for ordering.
    pub start_time_secs: f64,
    /// Steps recorded (from `summary.json`; `None` if the run has not finished).
    pub steps: Option<u64>,
    /// Wall-clock duration in seconds (`None` until finished).
    pub duration_secs: Option<f64>,
    /// Finish reason (screaming-snake; `None` until finished).
    pub reason: Option<String>,
}

impl RunInfo {
    /// Whether the run has a `summary.json` (i.e. finished cleanly).
    #[must_use]
    pub fn finished(&self) -> bool {
        self.reason.is_some()
    }
}

/// Lists the runs under `base` (e.g. `./logs`), newest first.
///
/// A subdirectory is a run only if it contains a `run.meta`; other directories
/// are ignored. A run missing or with an unreadable `summary.json` is still
/// listed (as unfinished) rather than dropped.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if `base` cannot be read, or [`EmryError::Json`] if
/// a `run.meta` is present but malformed.
pub fn list_runs(base: &Path) -> Result<Vec<RunInfo>, EmryError> {
    let mut runs = Vec::new();
    for entry in std::fs::read_dir(base)? {
        let dir = entry?.path();
        if !dir.is_dir() {
            continue;
        }
        let meta_path = dir.join(RUN_META_FILE);
        if !meta_path.is_file() {
            continue; // not a run directory
        }
        let meta: RunMetaFile = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        let summary = read_summary(&dir);
        runs.push(RunInfo {
            dir_name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            project: meta.project,
            start_time_secs: meta.start_time_secs,
            steps: summary.as_ref().map(|s| s.steps),
            duration_secs: summary.as_ref().map(|s| s.duration_secs),
            reason: summary.as_ref().map(|s| reason_str(s.reason)),
        });
    }
    // Newest first.
    runs.sort_by(|a, b| b.start_time_secs.total_cmp(&a.start_time_secs));
    Ok(runs)
}

/// Reads a run's `metrics.jsonl` (`path` is the run directory or the file) and
/// returns the last value seen for each metric.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the file cannot be read or [`EmryError::Json`]
/// if a line is not a valid [`MetricRecord`].
pub fn final_metrics(path: &Path) -> Result<BTreeMap<String, f64>, EmryError> {
    let file = if path.is_dir() {
        path.join(METRICS_FILE)
    } else {
        path.to_path_buf()
    };
    let mut latest: BTreeMap<String, f64> = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(file)?).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: MetricRecord = serde_json::from_str(trimmed)?;
        for (name, value) in record.values {
            latest.insert(name, value); // last row wins
        }
    }
    Ok(latest)
}

/// Reads `summary.json` from a run directory, returning `None` if it is absent
/// or unreadable (an unfinished or partially written run).
fn read_summary(dir: &Path) -> Option<Summary> {
    let text = std::fs::read_to_string(dir.join(SUMMARY_FILE)).ok()?;
    serde_json::from_str(&text).ok()
}

/// A [`FinishReason`](emry_core::FinishReason) as its screaming-snake string.
fn reason_str(reason: emry_core::FinishReason) -> String {
    serde_json::to_value(reason)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    use super::*;
    use crate::meta::write_json;
    use emry_core::FinishReason;
    use uuid::Uuid;

    fn make_run(base: &Path, project: &str, start: f64, summary: Option<(u64, f64)>) {
        let dir = base.join(format!("{project}_{}", start as u64));
        std::fs::create_dir_all(&dir).unwrap();
        write_json(
            &dir,
            RUN_META_FILE,
            &RunMetaFile {
                run_id: Uuid::nil(),
                project: project.to_string(),
                start_time_secs: start,
                mode: "file".into(),
            },
        )
        .unwrap();
        if let Some((steps, duration)) = summary {
            write_json(
                &dir,
                SUMMARY_FILE,
                &Summary {
                    run_id: Uuid::nil(),
                    project: project.to_string(),
                    duration_secs: duration,
                    reason: FinishReason::Completed,
                    steps,
                    dropped: 0,
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn lists_runs_newest_first_with_finished_status() {
        let base = crate::test_util::TempRunDir::new();
        make_run(base.path(), "old", 100.0, Some((50, 5.0)));
        make_run(base.path(), "new", 200.0, None); // unfinished
                                                   // A non-run directory is ignored.
        std::fs::create_dir_all(base.path().join("not-a-run")).unwrap();

        let runs = list_runs(base.path()).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].project, "new"); // newest first
        assert!(!runs[0].finished());
        assert_eq!(runs[1].project, "old");
        assert!(runs[1].finished());
        assert_eq!(runs[1].steps, Some(50));
        assert_eq!(runs[1].reason.as_deref(), Some("COMPLETED"));
    }

    #[test]
    fn list_runs_errors_on_missing_base() {
        assert!(matches!(
            list_runs(Path::new("/no/such/emry/logs")),
            Err(EmryError::Io(_))
        ));
    }

    #[test]
    fn final_metrics_keeps_last_value_per_metric() {
        let dir = crate::test_util::TempRunDir::new();
        std::fs::write(
            dir.path().join(METRICS_FILE),
            "{\"step\":0,\"epoch\":0,\"phase\":\"TRAIN\",\"values\":{\"loss\":1.0,\"lr\":0.1}}\n\
             {\"step\":1,\"epoch\":0,\"phase\":\"TRAIN\",\"values\":{\"loss\":0.4}}\n",
        )
        .unwrap();
        let final_m = final_metrics(dir.path()).unwrap();
        assert_eq!(final_m["loss"], 0.4); // last row wins
        assert_eq!(final_m["lr"], 0.1); // carried from the earlier row
    }

    #[test]
    fn final_metrics_errors_on_missing_file() {
        assert!(matches!(
            final_metrics(Path::new("/no/such/emry/metrics.jsonl")),
            Err(EmryError::Io(_))
        ));
    }
}
