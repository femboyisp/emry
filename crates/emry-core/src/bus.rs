//! Pub/sub event fan-out.
//!
//! The engine pipeline [`publish`](EventBus::publish)es events and derived state;
//! observers ([`subscribe`](EventBus::subscribe)rs such as the TUI and web
//! dashboard) each receive their own copy.
//!
//! # Backpressure policy
//!
//! Fan-out uses one **bounded** `crossbeam_channel` per subscriber, mirroring the
//! event ring ([`crate::ring`]): publishing never blocks on a slow observer. If a
//! subscriber stops draining (e.g. a stalled TUI render), its queue fills and
//! further events for it are **dropped and counted** rather than buffered without
//! limit — a laggy observer must never grow memory unbounded on a long run.
//! Dropped-event counts are exposed via [`EventBus::dropped`]. A subscriber whose
//! receiver is dropped entirely is pruned on the next publish.
//!
//! An [`EventBus`] is `Send + Sync`; share it across threads behind an `Arc`.

use crate::types::Event;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Default per-subscriber queue capacity.
///
/// Generous enough to absorb normal render/render-resize hitches at typical
/// logging rates, while still bounding memory for a wedged subscriber.
pub const DEFAULT_SUBSCRIBER_CAPACITY: usize = 16_384;

/// A multi-subscriber event fan-out channel.
#[derive(Debug, Default)]
pub struct EventBus {
    subscribers: Mutex<Vec<Sender<Event>>>,
    // Standalone statistic; see `crate::ring` for why `Relaxed` is sufficient.
    dropped: AtomicU64,
}

impl EventBus {
    /// Creates an empty bus with no subscribers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a subscriber with [`DEFAULT_SUBSCRIBER_CAPACITY`].
    #[must_use]
    pub fn subscribe(&self) -> Receiver<Event> {
        self.subscribe_with_capacity(DEFAULT_SUBSCRIBER_CAPACITY)
    }

    /// Registers a subscriber with an explicit queue `capacity`, returning its
    /// receiver. Events published after this call are delivered to it until its
    /// queue is full (then dropped) or its receiver is dropped (then pruned).
    #[must_use]
    pub fn subscribe_with_capacity(&self, capacity: usize) -> Receiver<Event> {
        let (tx, rx) = bounded(capacity);
        self.subscribers
            .lock()
            .expect("event bus mutex poisoned")
            .push(tx);
        rx
    }

    /// Delivers a clone of `event` to every live subscriber, returning the number
    /// of live subscribers afterwards.
    ///
    /// Never blocks. A subscriber whose queue is full has the event dropped and
    /// counted ([`EventBus::dropped`]) but stays subscribed; a subscriber whose
    /// receiver was dropped is pruned. The lock is held across the fan-out, which
    /// briefly serializes concurrent publishers — acceptable since each send is
    /// wait-free `try_send`, and the engine drains the ring on a single thread.
    pub fn publish(&self, event: &Event) -> usize {
        let mut subscribers = self.subscribers.lock().expect("event bus mutex poisoned");
        let mut dropped_now: u64 = 0;
        subscribers.retain(|tx| match tx.try_send(event.clone()) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                dropped_now += 1;
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        });
        if dropped_now > 0 {
            self.dropped.fetch_add(dropped_now, Ordering::Relaxed);
        }
        subscribers.len()
    }

    /// Number of events dropped because a subscriber's queue was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Current number of live subscribers.
    ///
    /// Subscribers whose receivers were dropped are only pruned on the next
    /// [`publish`](EventBus::publish), so this may briefly overcount.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.subscribers
            .lock()
            .expect("event bus mutex poisoned")
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MetricId, Phase};

    fn metric(step: u64) -> Event {
        Event::Metric {
            id: MetricId(0),
            value: 0.0,
            step,
            epoch: 0,
            phase: Phase::Train,
        }
    }

    fn step_of(event: &Event) -> u64 {
        match event {
            Event::Metric { step, .. } => *step,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn subscribe_increments_count() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count(), 0);
        let _a = bus.subscribe();
        let _b = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn publish_with_no_subscribers_is_a_noop() {
        let bus = EventBus::new();
        assert_eq!(bus.publish(&metric(0)), 0);
        assert_eq!(bus.dropped(), 0);
    }

    #[test]
    fn two_subscribers_each_receive_all_100_events_in_order() {
        let bus = EventBus::new();
        let a = bus.subscribe();
        let b = bus.subscribe();

        for step in 0..100 {
            assert_eq!(bus.publish(&metric(step)), 2);
        }

        for rx in [&a, &b] {
            for expected in 0..100 {
                let event = rx.try_recv().expect("event available");
                assert_eq!(step_of(&event), expected);
            }
            assert!(rx.try_recv().is_err(), "exactly 100 events per subscriber");
        }
        assert_eq!(bus.dropped(), 0);
    }

    #[test]
    fn full_subscriber_queue_drops_and_counts_without_blocking() {
        let bus = EventBus::new();
        // Tiny queue; never drained.
        let _slow = bus.subscribe_with_capacity(2);
        assert_eq!(bus.publish(&metric(0)), 1);
        assert_eq!(bus.publish(&metric(1)), 1);
        // Queue full from here: events are dropped, subscriber stays live.
        assert_eq!(bus.publish(&metric(2)), 1);
        assert_eq!(bus.publish(&metric(3)), 1);
        assert_eq!(bus.dropped(), 2);
        assert_eq!(bus.subscriber_count(), 1);
    }

    #[test]
    fn dropped_subscriber_is_pruned_on_next_publish() {
        let bus = EventBus::new();
        let live = bus.subscribe();
        let dead = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);

        drop(dead);
        // First publish prunes the dead subscriber; the live one still receives.
        assert_eq!(bus.publish(&metric(7)), 1);
        assert_eq!(bus.subscriber_count(), 1);
        assert_eq!(step_of(&live.try_recv().unwrap()), 7);
    }

    #[test]
    fn delivers_across_threads() {
        use std::sync::Arc;

        let bus = Arc::new(EventBus::new());
        let rx = bus.subscribe();

        let producer = {
            let bus = Arc::clone(&bus);
            std::thread::spawn(move || {
                for step in 0..100 {
                    bus.publish(&metric(step));
                }
            })
        };

        let mut received = 0u64;
        while received < 100 {
            if let Ok(event) = rx.recv() {
                assert_eq!(step_of(&event), received);
                received += 1;
            }
        }
        producer.join().unwrap();
        assert_eq!(received, 100);
    }
}
