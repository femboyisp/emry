//! Tail a `metrics.jsonl` file and parse appended rows into [`Event`]s.
//!
//! [`JsonlTailer`] tracks a byte offset and, on each [`poll`](JsonlTailer::poll),
//! reads only the lines appended since the last call, parses each as a wide
//! metric row, and yields [`Event::MetricsBatch`] events. This backs `emry watch`
//! on both Emry's own `metrics.jsonl` and third-party / v1 JSONL files.
//!
//! # Why polling, not inotify
//!
//! The primary target is HPC (SLURM / shared filesystems), where inotify-style
//! events are unreliable over NFS. Offset polling is robust there and needs no
//! extra dependency. The trade-off is latency bounded by the poll interval,
//! which is fine for a human-watched dashboard.

use emry_core::{Event, MetricId, MetricRegistry, Phase};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A lenient wide metric row. `step` and `values` are required; `epoch`/`phase`
/// default so minimal third-party files (just `step` + `values`) still parse.
#[derive(Debug, Deserialize)]
struct WatchedRow {
    step: u64,
    #[serde(default)]
    epoch: u32,
    #[serde(default = "default_phase")]
    phase: Phase,
    values: BTreeMap<String, f64>,
}

fn default_phase() -> Phase {
    Phase::Train
}

/// Tails a JSONL metrics file, parsing appended rows into events.
#[derive(Debug)]
pub struct JsonlTailer {
    path: PathBuf,
    offset: u64,
    registry: MetricRegistry,
    skipped: u64,
}

impl JsonlTailer {
    /// Creates a tailer for `path` starting at the beginning of the file.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            offset: 0,
            registry: MetricRegistry::new(),
            skipped: 0,
        }
    }

    /// Reads any rows appended since the last poll and returns their events.
    ///
    /// Only complete (newline-terminated) lines are consumed; a partial trailing
    /// line is left for a later poll. If the file shrank (truncation / rotation),
    /// the tailer restarts from the beginning. Unparseable lines are skipped and
    /// counted ([`JsonlTailer::skipped`]) rather than aborting the stream.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be opened or read. A missing
    /// file yields `Ok(vec![])` so polling can begin before the run writes.
    pub fn poll(&mut self) -> std::io::Result<Vec<Event>> {
        let mut file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        if len < self.offset {
            // File was truncated or rotated; restart from the top.
            self.offset = 0;
        }
        file.seek(SeekFrom::Start(self.offset))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // Consume only up to the last newline; keep any partial trailing line.
        let consumed = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        self.offset += consumed as u64;

        // Decode per line: a line with invalid UTF-8 (e.g. a stray binary write)
        // is skipped and counted rather than aborting the whole poll — third-party
        // files must not take down the watcher.
        let mut events = Vec::new();
        for line_bytes in bytes[..consumed].split(|&b| b == b'\n') {
            let Ok(line) = std::str::from_utf8(line_bytes) else {
                self.skipped += 1;
                continue;
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match self.parse_line(line) {
                Some(event) => events.push(event),
                None => self.skipped += 1,
            }
        }
        Ok(events)
    }

    /// Number of lines skipped because they failed to parse.
    #[must_use]
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// The metric labels discovered so far, as `(id, name)` pairs — used to seed
    /// an observer (e.g. the TUI) with names resolved from the file.
    #[must_use]
    pub fn labels(&self) -> Vec<(MetricId, String)> {
        (0..self.registry.len())
            .filter_map(|i| {
                let id = MetricId(u16::try_from(i).ok()?);
                self.registry.name(id).map(|n| (id, n.to_owned()))
            })
            .collect()
    }

    fn parse_line(&mut self, line: &str) -> Option<Event> {
        let row: WatchedRow = serde_json::from_str(line).ok()?;
        let values = row
            .values
            .into_iter()
            .map(|(name, value)| (self.registry.register(&name), value))
            .collect();
        Some(Event::MetricsBatch {
            step: row.step,
            epoch: row.epoch,
            phase: row.phase,
            values,
        })
    }
}

/// Polls `path` until `stop` is set, passing each non-empty batch of parsed
/// events to `on_events`. The thin live driver over [`JsonlTailer`].
///
/// Transient poll errors are swallowed and retried on the next tick rather than
/// terminating the loop — over NFS on HPC a brief I/O hiccup must not kill a
/// long-running watch; the dashboard simply shows no new data until it recovers.
pub fn run_watch<F: FnMut(&[Event])>(
    path: impl Into<PathBuf>,
    poll_interval: Duration,
    stop: &AtomicBool,
    mut on_events: F,
) {
    let mut tailer = JsonlTailer::new(path);
    while !stop.load(Ordering::Acquire) {
        let events = tailer.poll().unwrap_or_default();
        if !events.is_empty() {
            on_events(&events);
        }
        if !poll_interval.is_zero() {
            std::thread::sleep(poll_interval);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU32;

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TempFile(PathBuf);
    impl TempFile {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let p =
                std::env::temp_dir().join(format!("emry-watch-{}-{n}.jsonl", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Self(p)
        }
        fn append(&self, line: &str) {
            self.append_bytes(line.as_bytes());
        }
        fn append_bytes(&self, bytes: &[u8]) {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.0)
                .unwrap();
            f.write_all(bytes).unwrap();
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn step_values(event: &Event) -> (u64, Vec<(MetricId, f64)>) {
        match event {
            Event::MetricsBatch { step, values, .. } => (*step, values.clone()),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn missing_file_polls_empty() {
        let mut t = JsonlTailer::new("/no/such/emry/file.jsonl");
        assert!(t.poll().unwrap().is_empty());
    }

    #[test]
    fn parses_emry_metrics_rows() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"epoch\":0,\"phase\":\"TRAIN\",\"values\":{\"loss\":1.0}}\n");
        f.append("{\"step\":1,\"epoch\":0,\"phase\":\"EVAL\",\"values\":{\"loss\":0.5}}\n");
        let mut t = JsonlTailer::new(f.path());
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 2);
        let (step0, vals0) = step_values(&events[0]);
        assert_eq!(step0, 0);
        assert_eq!(vals0[0].1, 1.0);
    }

    #[test]
    fn parses_minimal_third_party_rows() {
        // No epoch/phase — a v1/third-party logger that only writes step+values.
        let f = TempFile::new();
        f.append("{\"step\":5,\"values\":{\"accuracy\":0.9}}\n");
        let mut t = JsonlTailer::new(f.path());
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::MetricsBatch {
                step, epoch, phase, ..
            } => {
                assert_eq!(*step, 5);
                assert_eq!(*epoch, 0); // defaulted
                assert_eq!(*phase, Phase::Train); // defaulted
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(t.labels(), vec![(MetricId(0), "accuracy".to_string())]);
    }

    #[test]
    fn only_appended_lines_are_returned_across_polls() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        let mut t = JsonlTailer::new(f.path());
        assert_eq!(t.poll().unwrap().len(), 1);
        // Nothing new yet.
        assert!(t.poll().unwrap().is_empty());
        // Append more; only the new rows come back.
        f.append("{\"step\":1,\"values\":{\"loss\":0.9}}\n");
        f.append("{\"step\":2,\"values\":{\"loss\":0.8}}\n");
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(step_values(&events[0]).0, 1);
    }

    #[test]
    fn partial_trailing_line_is_held_until_complete() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        f.append("{\"step\":1,\"values\":{\"loss\":0.9}}"); // no newline yet
        let mut t = JsonlTailer::new(f.path());
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 1, "partial line not consumed");
        // Finish the line; now it parses.
        f.append("\n");
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(step_values(&events[0]).0, 1);
    }

    #[test]
    fn blank_and_garbage_lines_are_skipped_and_counted() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        f.append("\n");
        f.append("not json at all\n");
        f.append("{\"step\":1,\"values\":{\"loss\":0.9}}\n");
        let mut t = JsonlTailer::new(f.path());
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 2, "two valid rows");
        assert_eq!(t.skipped(), 1, "one garbage line counted (blank ignored)");
    }

    #[test]
    fn truncation_restarts_from_top() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        let mut t = JsonlTailer::new(f.path());
        assert_eq!(t.poll().unwrap().len(), 1);
        // Truncate and rewrite with a genuinely shorter file so len < offset.
        std::fs::write(f.path(), "{\"step\":9,\"values\":{\"a\":0.1}}\n").unwrap();
        let events = t.poll().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(step_values(&events[0]).0, 9);
    }

    #[test]
    fn invalid_utf8_line_is_skipped_not_fatal() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        f.append_bytes(&[0xff, 0xfe, b'\n']); // invalid UTF-8 line
        f.append("{\"step\":1,\"values\":{\"loss\":0.9}}\n");
        let mut t = JsonlTailer::new(f.path());
        let events = t.poll().unwrap();
        assert_eq!(
            events.len(),
            2,
            "valid rows still parse around the bad line"
        );
        assert_eq!(t.skipped(), 1);
    }

    #[test]
    fn run_watch_stops_on_flag_and_delivers_events() {
        let f = TempFile::new();
        f.append("{\"step\":0,\"values\":{\"loss\":1.0}}\n");
        let stop = AtomicBool::new(false);
        let mut total = 0usize;
        // Callback sets stop after the first delivery, ending the loop.
        run_watch(f.path(), Duration::ZERO, &stop, |events| {
            total += events.len();
            stop.store(true, Ordering::Release);
        });
        assert_eq!(total, 1);
    }
}
