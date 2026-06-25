//! Single-producer, single-consumer event ring with drop-and-count backpressure.
//!
//! The training thread pushes [`Event`]s; the engine's pipeline thread pops
//! them. The ring wraps [`rtrb`], a lock-free SPSC queue, so emission never
//! takes a lock.
//!
//! # Backpressure policy
//!
//! Observability must never block or slow training. When the ring is full,
//! [`EventProducer::push`] **drops** the event and increments a shared dropped
//! counter, returning [`RingFull`] — it never blocks and never grows unbounded.
//! Callers on the hot path are expected to ignore the error (fire-and-forget);
//! the dropped count is surfaced to observers so the loss of events is visible
//! rather than silent. A full ring means the consumer is behind, not that
//! training should wait.

use crate::types::Event;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Default ring capacity (number of in-flight events).
pub const DEFAULT_CAPACITY: usize = 65_536;

/// Returned by [`EventProducer::push`] when the ring is full and the event was
/// dropped. The drop has already been counted; the hot path may ignore this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingFull;

impl std::fmt::Display for RingFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("event ring is full; event dropped")
    }
}

impl std::error::Error for RingFull {}

/// Producer half of the event ring, held by the producing (training) thread.
pub struct EventProducer {
    inner: rtrb::Producer<Event>,
    dropped: Arc<AtomicU64>,
}

/// Consumer half of the event ring, held by the consuming (pipeline) thread.
pub struct EventConsumer {
    inner: rtrb::Consumer<Event>,
    dropped: Arc<AtomicU64>,
}

/// Creates an event ring with [`DEFAULT_CAPACITY`].
#[must_use]
pub fn event_ring() -> (EventProducer, EventConsumer) {
    event_ring_with_capacity(DEFAULT_CAPACITY)
}

/// Creates an event ring with an explicit `capacity` (number of slots).
#[must_use]
pub fn event_ring_with_capacity(capacity: usize) -> (EventProducer, EventConsumer) {
    let (producer, consumer) = rtrb::RingBuffer::new(capacity);
    let dropped = Arc::new(AtomicU64::new(0));
    (
        EventProducer {
            inner: producer,
            dropped: Arc::clone(&dropped),
        },
        EventConsumer {
            inner: consumer,
            dropped,
        },
    )
}

impl EventProducer {
    /// Pushes an event, or drops it and counts the drop if the ring is full.
    ///
    /// Never blocks. On a full ring the event is discarded, the dropped counter
    /// is incremented, and [`RingFull`] is returned.
    ///
    /// # Errors
    ///
    /// Returns [`RingFull`] if the ring had no free slot; the event is dropped.
    pub fn push(&mut self, event: Event) -> Result<(), RingFull> {
        match self.inner.push(event) {
            Ok(()) => Ok(()),
            Err(_full) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Err(RingFull)
            }
        }
    }

    /// Number of events dropped so far because the ring was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Number of free slots currently available for pushing.
    #[must_use]
    pub fn free_slots(&self) -> usize {
        self.inner.slots()
    }
}

impl EventConsumer {
    /// Pops the oldest queued event, or `None` if the ring is empty.
    #[must_use]
    pub fn pop(&mut self) -> Option<Event> {
        self.inner.pop().ok()
    }

    /// Number of events dropped so far because the ring was full.
    ///
    /// Shares the producer's counter so observers can surface dropped events.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Number of events currently queued and readable.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.inner.slots()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Phase;

    fn metric(step: u64) -> Event {
        Event::Metric {
            id: crate::types::MetricId(0),
            value: 0.0,
            step,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    #[test]
    fn default_ring_uses_default_capacity() {
        let (producer, _consumer) = event_ring();
        assert_eq!(producer.free_slots(), DEFAULT_CAPACITY);
    }

    #[test]
    fn push_then_pop_preserves_fifo_order() {
        let (mut producer, mut consumer) = event_ring_with_capacity(8);
        for step in 0..5 {
            producer.push(metric(step)).unwrap();
        }
        assert_eq!(consumer.pending(), 5);
        for step in 0..5 {
            match consumer.pop() {
                Some(Event::Metric { step: s, .. }) => assert_eq!(s, step),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(consumer.pop().is_none());
    }

    #[test]
    fn overflow_drops_and_counts_without_blocking() {
        let (mut producer, _consumer) = event_ring_with_capacity(2);
        assert!(producer.push(metric(0)).is_ok());
        assert!(producer.push(metric(1)).is_ok());
        // Third push has no slot: dropped, counted, never blocks.
        assert_eq!(producer.push(metric(2)), Err(RingFull));
        assert_eq!(producer.push(metric(3)), Err(RingFull));
        assert_eq!(producer.dropped(), 2);
    }

    #[test]
    fn dropped_counter_is_shared_with_consumer() {
        let (mut producer, consumer) = event_ring_with_capacity(1);
        producer.push(metric(0)).unwrap();
        let _ = producer.push(metric(1));
        assert_eq!(consumer.dropped(), 1);
    }

    /// `10_000` events cross a thread boundary without deadlock or loss. The
    /// producer spins on a full ring (test-only) so this exercises concurrency,
    /// not the drop policy.
    #[test]
    fn ten_thousand_push_pop_across_threads() {
        const N: u64 = 10_000;
        let (mut producer, mut consumer) = event_ring_with_capacity(64);

        let producer_thread = std::thread::spawn(move || {
            for step in 0..N {
                while producer.push(metric(step)).is_err() {
                    std::thread::yield_now();
                }
            }
        });

        let mut received = 0u64;
        while received < N {
            if let Some(Event::Metric { step, .. }) = consumer.pop() {
                assert_eq!(step, received);
                received += 1;
            } else {
                std::thread::yield_now();
            }
        }

        producer_thread.join().unwrap();
        assert_eq!(received, N, "every event crossed the thread boundary");
        // Note: dropped() counts every full-ring rejection, so a spin-retry
        // producer inflates it — those events were re-pushed, not lost. The
        // drop counter measures rejections, which is what the hot path (which
        // does not retry) cares about.
    }
}
