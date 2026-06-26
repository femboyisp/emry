//! CSV export of a run's `metrics.jsonl`.
//!
//! Turns the wide [`MetricRecord`] rows into a flat CSV with a stable column
//! layout — `step,epoch,phase` followed by one column per metric name — so a run
//! can be loaded into pandas, a spreadsheet, or the prior generation's CSV
//! tooling. Metric columns appear in first-seen order across the file; a row
//! missing a given metric leaves that cell empty.
//!
//! Export streams the file in two passes (one to learn the columns, one to emit
//! rows) so peak memory is proportional to the number of distinct metric names,
//! not the row count — long runs have millions of rows.

use emry_core::{EmryError, MetricRecord};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Write};
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
    let file = resolve_metrics(path);
    let columns = scan_columns(&file)?;

    // Header.
    let mut header = vec!["step".to_string(), "epoch".to_string(), "phase".to_string()];
    header.extend(columns.iter().cloned());
    write_row(out, &header)?;

    // Pass 2: re-stream the file, one CSV row per record, blank cells for
    // metrics absent from a row.
    let mut rows = 0;
    for line in open_lines(&file)? {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: MetricRecord = serde_json::from_str(trimmed)?;
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
        rows += 1;
    }
    Ok(rows)
}

/// Resolves `path` to a `metrics.jsonl` file (joining it if `path` is a run dir).
fn resolve_metrics(path: &Path) -> std::path::PathBuf {
    if path.is_dir() {
        path.join(METRICS_FILE)
    } else {
        path.to_path_buf()
    }
}

/// Streams `file` once to learn the metric column names in first-seen order,
/// retaining only the names (not the rows) so memory stays bounded.
fn scan_columns(file: &Path) -> Result<Vec<String>, EmryError> {
    let mut columns: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in open_lines(file)? {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: MetricRecord = serde_json::from_str(trimmed)?;
        for name in record.values.keys() {
            if seen.insert(name.clone()) {
                columns.push(name.clone());
            }
        }
    }
    Ok(columns)
}

/// Opens `path` for buffered line-by-line reading.
fn open_lines(path: &Path) -> Result<Lines<BufReader<File>>, EmryError> {
    Ok(BufReader::new(File::open(path)?).lines())
}

/// Wraps a parquet/arrow error as an [`EmryError::Io`] (the parquet write path
/// is effectively an I/O-layer failure; this avoids a core API change).
#[cfg(feature = "parquet")]
fn to_io<E: std::error::Error + Send + Sync + 'static>(err: E) -> EmryError {
    EmryError::Io(std::io::Error::other(err))
}

/// Reads a run's `metrics.jsonl` (`path` is the file, or a run directory
/// containing it) and writes it to `out_path` as a Parquet file. Returns the
/// number of data rows written.
///
/// Schema: `step` (u64), `epoch` (u32), `phase` (utf8), then one nullable
/// `float64` column per metric (null where a row omits that metric). Rows are
/// written in batches so memory stays bounded for long runs.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the file cannot be read/written (parquet and
/// arrow failures are surfaced here too) or [`EmryError::Json`] if a line is not
/// a valid [`MetricRecord`].
#[cfg(feature = "parquet")]
pub fn export_parquet(path: &Path, out_path: &Path) -> Result<usize, EmryError> {
    // A failure mid-stream (disk-full write, input I/O error) would otherwise
    // leave a truncated, footerless Parquet file that no reader can open. Remove
    // the output on any error so the user is not left with a corrupt artifact.
    let result = write_parquet(path, out_path);
    if result.is_err() {
        let _ = std::fs::remove_file(out_path);
    }
    result
}

#[cfg(feature = "parquet")]
fn write_parquet(path: &Path, out_path: &Path) -> Result<usize, EmryError> {
    use arrow_array::builder::{Float64Builder, StringBuilder, UInt32Builder, UInt64Builder};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    /// Rows per Arrow record batch.
    const BATCH: usize = 8192;

    let file = resolve_metrics(path);
    let columns = scan_columns(&file)?;

    let mut fields = vec![
        Field::new("step", DataType::UInt64, false),
        Field::new("epoch", DataType::UInt32, false),
        Field::new("phase", DataType::Utf8, false),
    ];
    for col in &columns {
        fields.push(Field::new(col, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    let out = File::create(out_path)?;
    let mut writer = ArrowWriter::try_new(out, Arc::clone(&schema), None).map_err(to_io)?;

    let mut step_b = UInt64Builder::new();
    let mut epoch_b = UInt32Builder::new();
    let mut phase_b = StringBuilder::new();
    let mut metric_b: Vec<Float64Builder> = columns.iter().map(|_| Float64Builder::new()).collect();

    // Finishes the current builders into a batch and writes it.
    let flush = |writer: &mut ArrowWriter<File>,
                 step_b: &mut UInt64Builder,
                 epoch_b: &mut UInt32Builder,
                 phase_b: &mut StringBuilder,
                 metric_b: &mut [Float64Builder]|
     -> Result<(), EmryError> {
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(step_b.finish()),
            Arc::new(epoch_b.finish()),
            Arc::new(phase_b.finish()),
        ];
        for builder in metric_b.iter_mut() {
            arrays.push(Arc::new(builder.finish()));
        }
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(to_io)?;
        writer.write(&batch).map_err(to_io)?;
        Ok(())
    };

    let mut rows = 0usize;
    let mut in_batch = 0usize;
    for line in open_lines(&file)? {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: MetricRecord = serde_json::from_str(trimmed)?;
        step_b.append_value(record.step);
        epoch_b.append_value(record.epoch);
        phase_b.append_value(phase_str(record.phase));
        for (builder, col) in metric_b.iter_mut().zip(&columns) {
            builder.append_option(record.values.get(col).copied());
        }
        rows += 1;
        in_batch += 1;
        if in_batch >= BATCH {
            flush(
                &mut writer,
                &mut step_b,
                &mut epoch_b,
                &mut phase_b,
                &mut metric_b,
            )?;
            in_batch = 0;
        }
    }
    if in_batch > 0 {
        flush(
            &mut writer,
            &mut step_b,
            &mut epoch_b,
            &mut phase_b,
            &mut metric_b,
        )?;
    }
    writer.close().map_err(to_io)?;
    Ok(rows)
}

/// Phase as its screaming-snake wire string (e.g. `TRAIN`).
fn phase_str(phase: emry_core::Phase) -> String {
    serde_json::to_value(phase)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Writes one CSV record: comma-separated, LF-terminated, fields quoted (per
/// RFC 4180 escaping) only when they contain a comma, quote, CR, or LF.
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

    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_roundtrips_rows_and_schema() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let dir = TempRunDir::new();
        write_metrics(
            dir.path(),
            &[
                r#"{"step":0,"epoch":0,"phase":"TRAIN","values":{"loss":1.0,"lr":0.1}}"#,
                r#"{"step":1,"epoch":0,"phase":"EVAL","values":{"loss":0.5,"acc":0.9}}"#,
            ],
        );
        let out = dir.path().join("metrics.parquet");
        let rows = export_parquet(dir.path(), &out).unwrap();
        assert_eq!(rows, 2);

        // Read it back: 2 rows, schema = step,epoch,phase + first-seen metrics.
        let file = std::fs::File::open(&out).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut total = 0;
        let mut names: Vec<String> = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            total += batch.num_rows();
            if names.is_empty() {
                names = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
            }
        }
        assert_eq!(total, 2);
        assert_eq!(names, vec!["step", "epoch", "phase", "loss", "lr", "acc"]);
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_corrupt_line_is_an_error() {
        let dir = TempRunDir::new();
        write_metrics(dir.path(), &["not json"]);
        let out = dir.path().join("bad.parquet");
        assert!(matches!(
            export_parquet(dir.path(), &out),
            Err(EmryError::Json(_))
        ));
    }
}
