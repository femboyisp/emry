//! The processing pipeline thread.
//!
//! [`Pipeline::spawn`] starts a single thread that:
//! 1. drains the event ring ([`EventConsumer`]),
//! 2. forwards each raw [`Event`] to the [`EventBus`] for observers, and
//! 3. runs every [`Processor`], emitting their [`DerivedMetric`]s on a channel.
//!
//! It shuts down gracefully when it sees [`Event::RunFinished`] or when
//! [`Pipeline::stop`] is called (or the [`Pipeline`] is dropped). Because the
//! loop only exits while the ring is empty, all queued events are drained first.

use crate::processor::{DerivedMetric, Processor};
use crossbeam_channel::{unbounded, Receiver};
use emry_core::{Event, EventBus, EventConsumer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// How long the thread sleeps when the ring is momentarily empty, balancing
/// latency against idle CPU.
const POLL_INTERVAL: Duration = Duration::from_micros(200);

/// Counts of work done by the pipeline, returned when it finishes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineStats {
    /// Number of events drained from the ring and processed.
    pub events_processed: u64,
    /// Number of derived metrics emitted by processors.
    pub derived_emitted: u64,
}

/// A running processing pipeline. Drop or [`join`](Pipeline::join) to finish.
pub struct Pipeline {
    handle: Option<JoinHandle<PipelineStats>>,
    stop: Arc<AtomicBool>,
}

impl Pipeline {
    /// Spawns the pipeline thread, returning the handle and the receiver for
    /// derived metrics emitted by `processors`.
    #[must_use]
    pub fn spawn(
        mut consumer: EventConsumer,
        mut processors: Vec<Box<dyn Processor>>,
        bus: Arc<EventBus>,
    ) -> (Self, Receiver<DerivedMetric>) {
        let (derived_tx, derived_rx) = unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);

        let handle = std::thread::spawn(move || {
            let mut stats = PipelineStats::default();
            loop {
                if let Some(event) = consumer.pop() {
                    bus.publish(&event);
                    stats.events_processed += 1;
                    for processor in &mut processors {
                        for derived in processor.on_event(&event) {
                            stats.derived_emitted += 1;
                            // Ignore send errors: a dropped receiver just means
                            // nobody is consuming derived metrics.
                            let _ = derived_tx.send(derived);
                        }
                    }
                    if matches!(event, Event::RunFinished { .. }) {
                        break;
                    }
                } else {
                    // Only stop when the ring is drained, so pending events are
                    // never abandoned.
                    if stop_thread.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
            }
            stats
        });

        (
            Self {
                handle: Some(handle),
                stop,
            },
            derived_rx,
        )
    }

    /// Requests a graceful shutdown: the thread drains the ring, then exits.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Waits for the pipeline thread to finish and returns its stats.
    ///
    /// # Panics
    ///
    /// Panics if the pipeline thread panicked.
    #[must_use]
    pub fn join(mut self) -> PipelineStats {
        self.handle
            .take()
            .expect("pipeline handle present until join/drop")
            .join()
            .expect("pipeline thread panicked")
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::{event_ring_with_capacity, EventProducer, FinishReason, MetricId, Phase};

    fn metric(step: u64) -> Event {
        Event::Metric {
            id: MetricId(0),
            value: 0.0,
            step,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    fn finished() -> Event {
        Event::RunFinished {
            duration_secs: 1.0,
            reason: FinishReason::Completed,
        }
    }

    /// A processor that records nothing and emits nothing.
    struct NoOp;
    impl Processor for NoOp {
        fn on_event(&mut self, _event: &Event) -> Vec<DerivedMetric> {
            Vec::new()
        }
    }

    /// Emits one derived metric per `Metric` event.
    struct CountMetrics {
        seen: u32,
    }
    impl Processor for CountMetrics {
        fn on_event(&mut self, event: &Event) -> Vec<DerivedMetric> {
            if matches!(event, Event::Metric { .. }) {
                self.seen += 1;
                vec![DerivedMetric::new("metric_count", f64::from(self.seen))]
            } else {
                Vec::new()
            }
        }
    }

    fn push_all(producer: &mut EventProducer, events: impl IntoIterator<Item = Event>) {
        for event in events {
            producer.push(event).expect("ring has capacity");
        }
    }

    #[test]
    fn noop_pipeline_drains_and_shuts_down_on_run_finished() {
        let (mut producer, consumer) = event_ring_with_capacity(256);
        let bus = Arc::new(EventBus::new());
        let subscriber = bus.subscribe();

        push_all(&mut producer, (0..10).map(metric));
        producer.push(finished()).unwrap();

        let (pipeline, derived_rx) =
            Pipeline::spawn(consumer, vec![Box::new(NoOp)], Arc::clone(&bus));
        let stats = pipeline.join();

        assert_eq!(stats.events_processed, 11, "10 metrics + RunFinished");
        assert_eq!(stats.derived_emitted, 0);
        assert!(derived_rx.try_recv().is_err());
        // Every event was forwarded to the bus subscriber.
        let forwarded: Vec<_> = subscriber.try_iter().collect();
        assert_eq!(forwarded.len(), 11);
    }

    #[test]
    fn processors_emit_derived_metrics() {
        let (mut producer, consumer) = event_ring_with_capacity(256);
        let bus = Arc::new(EventBus::new());

        push_all(&mut producer, (0..5).map(metric));
        producer.push(finished()).unwrap();

        let (pipeline, derived_rx) = Pipeline::spawn(
            consumer,
            vec![Box::new(CountMetrics { seen: 0 })],
            Arc::clone(&bus),
        );
        let stats = pipeline.join();

        assert_eq!(stats.derived_emitted, 5);
        let derived: Vec<_> = derived_rx.try_iter().collect();
        assert_eq!(derived.len(), 5);
        assert_eq!(
            derived.last().unwrap(),
            &DerivedMetric::new("metric_count", 5.0)
        );
    }

    #[test]
    fn stop_drains_remaining_events_without_run_finished() {
        let (mut producer, consumer) = event_ring_with_capacity(256);
        let bus = Arc::new(EventBus::new());

        push_all(&mut producer, (0..20).map(metric)); // no RunFinished

        let (pipeline, _derived_rx) =
            Pipeline::spawn(consumer, vec![Box::new(NoOp)], Arc::clone(&bus));
        pipeline.stop();
        let stats = pipeline.join();

        assert_eq!(stats.events_processed, 20, "all pending events drained");
    }

    #[test]
    fn drop_without_join_shuts_down_cleanly() {
        let (mut producer, consumer) = event_ring_with_capacity(256);
        let bus = Arc::new(EventBus::new());

        push_all(&mut producer, (0..3).map(metric));
        producer.push(finished()).unwrap();

        let (pipeline, _derived_rx) =
            Pipeline::spawn(consumer, vec![Box::new(NoOp)], Arc::clone(&bus));
        // Dropping joins the thread via Drop; must not hang or panic.
        drop(pipeline);
    }
}
