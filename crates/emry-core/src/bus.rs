//! Pub/sub event fan-out.
//!
//! The engine pipeline [`publish`](EventBus::publish)es events and derived state;
//! observers ([`subscribe`](EventBus::subscribe)rs such as the TUI and web
//! dashboard) each receive their own copy. Fan-out uses one unbounded
//! `crossbeam_channel` per subscriber, so publishing never blocks on a slow or
//! stalled observer — backpressure against *training* is handled upstream by the
//! event ring ([`crate::ring`]), not here.
//!
//! An [`EventBus`] is `Send + Sync`; share it across threads behind an `Arc`.

use crate::types::Event;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::sync::Mutex;

/// A multi-subscriber event fan-out channel.
#[derive(Debug, Default)]
pub struct EventBus {
    subscribers: Mutex<Vec<Sender<Event>>>,
}

impl EventBus {
    /// Creates an empty bus with no subscribers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new subscriber and returns its receiver.
    ///
    /// Each subscriber gets an independent unbounded queue; events published
    /// after this call are delivered to it.
    #[must_use]
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = unbounded();
        self.subscribers
            .lock()
            .expect("event bus mutex poisoned")
            .push(tx);
        rx
    }

    /// Delivers a clone of `event` to every live subscriber.
    ///
    /// Never blocks: unbounded queues mean a send always succeeds unless the
    /// subscriber's receiver has been dropped, in which case that subscriber is
    /// pruned. Returns the number of subscribers the event was delivered to.
    pub fn publish(&self, event: &Event) -> usize {
        let mut subscribers = self.subscribers.lock().expect("event bus mutex poisoned");
        subscribers.retain(|tx| tx.send(event.clone()).is_ok());
        subscribers.len()
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
