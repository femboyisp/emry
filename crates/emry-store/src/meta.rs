//! Run-directory layout and metadata files.
//!
//! A run lives in `logs/{project}_{timestamp}/` alongside its JSONL logs:
//!
//! ```text
//! logs/{project}_{timestamp}/
//!   run.meta       # run_id, project, start_time, mode  (written at start)
//!   config.json    # hyperparameters                     (written at start)
//!   events.jsonl   # full audit stream                   (crate::writer)
//!   metrics.jsonl  # wide metric rows                     (crate::writer)
//!   summary.json   # duration, reason, counts            (written at finish)
//! ```

use emry_core::{EmryError, FinishReason};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// File name of the run-metadata file.
pub const RUN_META_FILE: &str = "run.meta";
/// File name of the persisted run configuration.
pub const CONFIG_FILE: &str = "config.json";
/// File name of the end-of-run summary.
pub const SUMMARY_FILE: &str = "summary.json";

/// Metadata captured when a run starts, written to `run.meta`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetaFile {
    /// UUID v4 identifying this run.
    pub run_id: Uuid,
    /// Human-readable project / experiment name.
    pub project: String,
    /// Unix timestamp (seconds) when the run started.
    pub start_time_secs: f64,
    /// Deploy mode in effect (`embedded` | `sidecar` | `file`).
    pub mode: String,
}

/// End-of-run summary, written to `summary.json` on finish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// UUID v4 identifying this run.
    pub run_id: Uuid,
    /// Human-readable project / experiment name.
    pub project: String,
    /// Wall-clock run duration in seconds.
    pub duration_secs: f64,
    /// Why the run ended.
    pub reason: FinishReason,
    /// Number of metric emissions (steps) recorded.
    pub steps: u64,
    /// Number of events dropped because the ring was full.
    pub dropped: u64,
}

/// Standard run-directory name: `{project}_{YYYYMMDD_HHMMSS}` (UTC).
#[must_use]
pub fn run_dir_name(project: &str, start_time_secs: f64) -> String {
    format!("{project}_{}", format_utc_timestamp(start_time_secs))
}

/// Creates `base/{project}_{timestamp}/`, returning its path. `base` (e.g.
/// `EMRY_LOG_DIR`) is created if absent.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the directory cannot be created.
pub fn create_run_dir(
    base: &Path,
    project: &str,
    start_time_secs: f64,
) -> Result<PathBuf, EmryError> {
    let dir = base.join(run_dir_name(project, start_time_secs));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Writes `value` to `dir/name` as pretty JSON (overwriting).
///
/// # Errors
///
/// Returns [`EmryError::Json`] on serialization failure or [`EmryError::Io`] on
/// write failure.
pub fn write_json<T: Serialize>(dir: &Path, name: &str, value: &T) -> Result<(), EmryError> {
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(dir.join(name), json)?;
    Ok(())
}

/// Formats a Unix timestamp (seconds) as `YYYYMMDD_HHMMSS` in UTC.
///
/// Uses Howard Hinnant's `civil_from_days` algorithm so no date crate is needed.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // intermediate civil-date values are small
fn format_utc_timestamp(secs: f64) -> String {
    let secs = secs as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (hour, min, sec) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    // civil_from_days: days since 1970-01-01 -> (year, month, day).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365; // [0, 399]
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); // [0, 365]
    let mp = (5 * day_of_year + 2) / 153; // [0, 11]
    let day = day_of_year - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}{month:02}{day:02}_{hour:02}{min:02}{sec:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRunDir;

    #[test]
    fn epoch_zero_is_unix_origin() {
        assert_eq!(format_utc_timestamp(0.0), "19700101_000000");
    }

    #[test]
    fn known_timestamp_formats_correctly() {
        // 2021-06-21 12:00:00 UTC = 1_624_276_800.
        assert_eq!(format_utc_timestamp(1_624_276_800.0), "20210621_120000");
    }

    #[test]
    fn run_dir_name_combines_project_and_timestamp() {
        assert_eq!(
            run_dir_name("llama-sft", 1_624_276_800.0),
            "llama-sft_20210621_120000"
        );
    }

    #[test]
    fn create_run_dir_makes_nested_path() {
        let base = TempRunDir::new();
        let dir = create_run_dir(base.path(), "proj", 0.0).unwrap();
        assert!(dir.is_dir());
        assert!(dir.ends_with("proj_19700101_000000"));
    }

    #[test]
    fn run_meta_roundtrips() {
        let base = TempRunDir::new();
        let meta = RunMetaFile {
            run_id: Uuid::nil(),
            project: "p".into(),
            start_time_secs: 1.0,
            mode: "file".into(),
        };
        write_json(base.path(), RUN_META_FILE, &meta).unwrap();
        let text = std::fs::read_to_string(base.path().join(RUN_META_FILE)).unwrap();
        let back: RunMetaFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn summary_roundtrips() {
        let base = TempRunDir::new();
        let summary = Summary {
            run_id: Uuid::nil(),
            project: "p".into(),
            duration_secs: 12.5,
            reason: FinishReason::Completed,
            steps: 100,
            dropped: 0,
        };
        write_json(base.path(), SUMMARY_FILE, &summary).unwrap();
        let text = std::fs::read_to_string(base.path().join(SUMMARY_FILE)).unwrap();
        let back: Summary = serde_json::from_str(&text).unwrap();
        assert_eq!(back, summary);
        assert!(text.contains("\"reason\": \"COMPLETED\""));
    }

    #[test]
    fn write_json_errors_on_missing_dir() {
        let err = write_json(Path::new("/no/such/emry/dir"), "x.json", &1).unwrap_err();
        assert!(matches!(err, EmryError::Io(_)));
    }
}
