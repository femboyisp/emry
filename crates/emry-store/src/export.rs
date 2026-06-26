//! CSV export of a run's `metrics.jsonl`.
//!
//! Turns the wide [`MetricRecord`] rows into a flat CSV with a stable column
//! layout — `step,epoch,phase` followed by one column per metric name — so a run
//! can be loaded into pandas, a spreadsheet, or the prior generation's CSV
//! tooling. Metric columns appear in first-seen order across the file; a row
//! missing a given metric leaves that cell empty.

use emry_core::{EmryError, MetricRecord};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use crate::writer::METRICS_FILE;

/// Reads a run's `metrics.jsonl` (`path` is the file, or a run directory
/// containing it) and writes it to `out` as CSV. Returns the number of data
/// rows written (excluding the header).
///
/// Blank lines are skipped. Unparseable lines are an error, so a corrupt log is
/// surfaced rather than silently dropped.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the file cannot be read or `out` cannot be
/// written, or [`EmryError::Json`] if a line is not a valid [`MetricRecord`].
pub fn export_csv<W: Write>(path: &Path, out: &mut W) -> Result<usize, EmryError> {
    let file = if path.is_dir() {
        path.join(METRICS_FILE)
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(file)?;

    // Pass 1: parse rows and learn the metric columns in first-seen order.
    let mut records: Vec<MetricRecord> = Vec::new();
    let mut columns: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: MetricRecord = serde_json::from_str(line)?;
        for name in record.values.keys() {
            if seen.insert(name.clone()) {
                columns.push(name.clone());
            }
        }
        records.push(record);
    }

    // Header.
    let mut header = vec!["step".to_string(), "epoch".to_string(), "phase".to_string()];
    header.extend(columns.iter().cloned());
    write_row(out, &header)?;

    // Pass 2: one CSV row per record, blank cells for absent metrics.
    for record in &records {
        let mut fields = vec![
            record.step.to_string(),
            record.epoch.to_string(),
            phase_str(record.phase),
        ];
        for col in &columns {
            match record.values.get(col) {
                Some(value) => fields.push(value.to_string()),
                None => fields.push(String::new()),
            }
        }
        write_row(out, &fields)?;
    }
    Ok(records.len())
}

/// Phase as its screaming-snake wire string (e.g. `TRAIN`).
fn phase_str(phase: emry_core::Phase) -> String {
    serde_json::to_value(phase)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Writes one CSV record (RFC 4180: comma-separated, `\n`-terminated, fields
/// quoted only when they contain a comma, quote, CR, or LF).
fn write_row<W: Write>(out: &mut W, fields: &[String]) -> Result<(), EmryError> {
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.write_all(b",")?;
        }
        out.write_all(escape(field).as_bytes())?;
    }
    out.write_all(b"\n")?;
    Ok(())
}

/// Quotes and escapes a CSV field if it contains a delimiter, quote, or newline.
fn escape(field: &str) -> std::borrow::Cow<'_, str> {
    if field.contains([',', '"', '\n', '\r']) {
        std::borrow::Cow::Owned(format!("\"{}\"", field.replace('"', "\"\"")))
    } else {
        std::borrow::Cow::Borrowed(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRunDir;
    use emry_core::Phase;
    use std::collections::BTreeMap;

    fn write_metrics(dir: &Path, lines: &[&str]) {
        let mut text = String::new();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        std::fs::write(dir.join(METRICS_FILE), text).unwrap();
    }

    fn csv_of(path: &Path) -> (usize, String) {
        let mut buf = Vec::new();
        let rows = export_csv(path, &mut buf).unwrap();
        (rows, String::from_utf8(buf).unwrap())
    }

    #[test]
    fn exports_wide_rows_with_first_seen_columns() {
        let dir = TempRunDir::new();
        write_metrics(
            dir.path(),
            &[
                r#"{"step":0,"epoch":0,"phase":"TRAIN","values":{"loss":1.0,"lr":0.1}}"#,
                "",
                r#"{"step":1,"epoch":0,"phase":"EVAL","values":{"loss":0.5,"acc":0.9}}"#,
            ],
        );
        let (rows, csv) = csv_of(dir.path());
        assert_eq!(rows, 2);
        let lines: Vec<&str> = csv.lines().collect();
        // Columns: step,epoch,phase + first-seen metric order (loss, lr, then acc).
        assert_eq!(lines[0], "step,epoch,phase,loss,lr,acc");
        assert_eq!(lines[1], "0,0,TRAIN,1,0.1,");
        // Second row has no lr → empty cell; acc fills its column.
        assert_eq!(lines[2], "1,0,EVAL,0.5,,0.9");
    }

    #[test]
    fn accepts_a_file_path_directly() {
        let dir = TempRunDir::new();
        write_metrics(
            dir.path(),
            &[r#"{"step":7,"epoch":1,"phase":"TRAIN","values":{"loss":0.25}}"#],
        );
        let (rows, csv) = csv_of(&dir.path().join(METRICS_FILE));
        assert_eq!(rows, 1);
        assert!(csv.contains("7,1,TRAIN,0.25"));
    }

    #[test]
    fn quotes_fields_needing_escaping() {
        let mut values = BTreeMap::new();
        values.insert("weird,name".to_string(), 1.5);
        let record = MetricRecord {
            step: 0,
            epoch: 0,
            phase: Phase::Train,
            values,
        };
        let dir = TempRunDir::new();
        std::fs::write(
            dir.path().join(METRICS_FILE),
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .unwrap();
        let (_, csv) = csv_of(dir.path());
        assert!(csv.lines().next().unwrap().contains("\"weird,name\""));
    }

    #[test]
    fn empty_file_yields_header_only() {
        let dir = TempRunDir::new();
        write_metrics(dir.path(), &[]);
        let (rows, csv) = csv_of(dir.path());
        assert_eq!(rows, 0);
        assert_eq!(csv, "step,epoch,phase\n");
    }

    #[test]
    fn corrupt_line_is_an_error() {
        let dir = TempRunDir::new();
        write_metrics(dir.path(), &["not json"]);
        let mut buf = Vec::new();
        assert!(matches!(
            export_csv(dir.path(), &mut buf),
            Err(EmryError::Json(_))
        ));
    }

    #[test]
    fn missing_file_is_io_error() {
        let mut buf = Vec::new();
        assert!(matches!(
            export_csv(Path::new("/no/such/emry/metrics.jsonl"), &mut buf),
            Err(EmryError::Io(_))
        ));
    }
}
