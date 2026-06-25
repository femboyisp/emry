//! Background JSONL sink: a writer thread fed by a bounded channel.
//!
//! [`JsonlSink::spawn`] starts a thread that owns a [`JsonlWriter`] and drains a
//! bounded channel, batching writes and flushing every [`FLUSH_EVERY`] records
//! or [`FLUSH_INTERVAL`], whichever comes first. The producer side never blocks:
//! a full channel drops the record and counts it ([`JsonlSink::dropped`]),
//! mirroring the ring/bus backpressure policy. [`JsonlSink::finish`] flushes all
//! pending records before returning.

use crate::writer::JsonlWriter;
use crossbeam_channel::{bounded, RecvTimeoutError, Sender};
use emry_core::{EmryError, Event, MetricRecord};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Flush after this many buffered records.
pub const FLUSH_EVERY: usize = 64;
/// Flush at least this often even when fewer than [`FLUSH_EVERY`] records arrive.
pub const FLUSH_INTERVAL: Duration = Duration::from_millis(100);
/// Default channel capacity between producer and writer thread.
pub const DEFAULT_CAPACITY: usize = 8192;

/// A record queued for persistence.
#[derive(Debug, Clone)]
enum Record {
    Event(Event),
    Metric(MetricRecord),
}

/// Handle to a background JSONL writer thread.
#[derive(Debug)]
pub struct JsonlSink {
    tx: Option<Sender<Record>>,
    handle: Option<JoinHandle<Result<(), EmryError>>>,
    dropped: Arc<AtomicU64>,
}

impl JsonlSink {
    /// Spawns a writer thread for the run directory `dir` (which must exist),
    /// using [`DEFAULT_CAPACITY`].
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Io`] if the JSONL files cannot be created.
    pub fn spawn(dir: &Path) -> Result<Self, EmryError> {
        Self::spawn_with_capacity(dir, DEFAULT_CAPACITY)
    }

    /// Spawns a writer thread with an explicit channel `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`EmryError::Io`] if the JSONL files cannot be created.
    pub fn spawn_with_capacity(dir: &Path, capacity: usize) -> Result<Self, EmryError> {
        // Create the writer on the calling thread so file-creation errors are
        // reported synchronously rather than swallowed by the worker.
        let writer = JsonlWriter::create(dir)?;
        let (tx, rx) = bounded::<Record>(capacity);

        let handle = std::thread::spawn(move || -> Result<(), EmryError> {
            let mut writer = writer;
            let mut since_flush = 0usize;
            loop {
                match rx.recv_timeout(FLUSH_INTERVAL) {
                    Ok(record) => {
                        write_record(&mut writer, &record)?;
                        since_flush += 1;
                        if since_flush >= FLUSH_EVERY {
                            writer.flush()?;
                            since_flush = 0;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if since_flush > 0 {
                            writer.flush()?;
                            since_flush = 0;
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            // Drain anything still queued, then flush everything.
            for record in rx.try_iter() {
                write_record(&mut writer, &record)?;
            }
            writer.flush()?;
            Ok(())
        });

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            dropped: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Queues an event for the audit log. Never blocks; drops and counts if the
    /// channel is full.
    pub fn write_event(&self, event: Event) {
        self.send(Record::Event(event));
    }

    /// Queues a wide metric row. Never blocks; drops and counts if full.
    pub fn write_metric(&self, record: MetricRecord) {
        self.send(Record::Metric(record));
    }

    fn send(&self, record: Record) {
        if let Some(tx) = &self.tx {
            if tx.try_send(record).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Number of records dropped because the channel was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Flushes all pending records and joins the writer thread.
    ///
    /// # Errors
    ///
    /// Returns any [`EmryError`] the writer thread hit while writing or flushing.
    ///
    /// # Panics
    ///
    /// Panics if the writer thread panicked.
    pub fn finish(mut self) -> Result<(), EmryError> {
        self.shutdown()
    }

    /// Disconnects the channel and joins the thread, surfacing its result.
    fn shutdown(&mut self) -> Result<(), EmryError> {
        // Dropping the sender disconnects the channel, ending the worker loop.
        self.tx = None;
        match self.handle.take() {
            Some(handle) => handle.join().expect("jsonl writer thread panicked"),
            None => Ok(()),
        }
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        // Best-effort flush on drop; errors are ignored (finish() surfaces them).
        let _ = self.shutdown();
    }
}

fn write_record(writer: &mut JsonlWriter, record: &Record) -> Result<(), EmryError> {
    match record {
        Record::Event(event) => writer.write_event(event),
        Record::Metric(metric) => writer.write_metric(metric),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRunDir;
    use crate::writer::{EVENTS_FILE, METRICS_FILE};
    use emry_core::{FinishReason, Phase};
    use std::collections::BTreeMap;
    use std::fs;

    fn record(step: u64) -> MetricRecord {
        let mut values = BTreeMap::new();
        values.insert(
            "loss".to_string(),
            1.0 / f64::from(u32::try_from(step + 1).unwrap()),
        );
        MetricRecord {
            step,
            epoch: 0,
            phase: Phase::Train,
            values,
        }
    }

    #[test]
    fn finish_flushes_all_pending_records() {
        let dir = TempRunDir::new();
        let sink = JsonlSink::spawn(dir.path()).unwrap();
        for step in 0..200 {
            sink.write_metric(record(step));
        }
        sink.write_event(Event::RunFinished {
            duration_secs: 1.0,
            reason: FinishReason::Completed,
        });
        let dropped = sink.dropped();
        sink.finish().unwrap();

        let metrics = fs::read_to_string(dir.path().join(METRICS_FILE)).unwrap();
        let events = fs::read_to_string(dir.path().join(EVENTS_FILE)).unwrap();
        assert_eq!(
            metrics.lines().count(),
            200,
            "all metrics flushed on finish"
        );
        assert_eq!(events.lines().count(), 1);
        // Ample capacity (8192) for 201 records: nothing dropped.
        assert_eq!(dropped, 0);
    }

    #[test]
    fn full_channel_drops_and_counts() {
        let dir = TempRunDir::new();
        // Capacity 1 and no time for the worker to drain: most sends overflow.
        let sink = JsonlSink::spawn_with_capacity(dir.path(), 1).unwrap();
        for step in 0..1000 {
            sink.write_metric(record(step));
        }
        let dropped = sink.dropped();
        sink.finish().unwrap();
        // We can't assert an exact count (timing-dependent), but with capacity 1
        // and 1000 rapid sends, some must have been dropped.
        assert!(dropped > 0, "expected some drops, got {dropped}");
    }

    #[test]
    fn drop_without_finish_still_flushes() {
        let dir = TempRunDir::new();
        {
            let sink = JsonlSink::spawn(dir.path()).unwrap();
            sink.write_metric(record(0));
            // No finish(): Drop must flush and join.
        }
        let metrics = fs::read_to_string(dir.path().join(METRICS_FILE)).unwrap();
        assert_eq!(metrics.lines().count(), 1);
    }

    #[test]
    fn spawn_fails_for_missing_directory() {
        let err = JsonlSink::spawn(Path::new("/no/such/emry/dir")).unwrap_err();
        assert!(matches!(err, EmryError::Io(_)));
    }
}
