//! axum HTTP + WebSocket server for the live web dashboard.
//!
//! Routes:
//! - `GET /` — the dashboard page (static; uPlot charts land in EMRY-041).
//! - `GET /healthz` — liveness probe.
//! - `GET /ws` — WebSocket streaming the [`WebState`] JSON at ≤10 Hz.
//!
//! A background task drains the event bus into a shared [`WebState`]; each WS
//! connection ticks at 10 Hz and sends the current snapshot. Throttling here (not
//! per-event) keeps the browser update rate bounded regardless of emit rate.

use crate::state::WebState;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use crossbeam_channel::Receiver;
use emry_core::{Event, MetricId};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// WebSocket push interval (10 Hz).
pub const PUSH_INTERVAL: Duration = Duration::from_millis(100);

/// Shared dashboard state behind a mutex (read by WS connections, written by the
/// event-draining task).
pub type SharedState = Arc<Mutex<WebState>>;

const INDEX_HTML: &str = include_str!("index.html");

/// Builds the router over a shared state (no event source wired — for testing
/// routes and for [`serve`] to drive).
pub fn app(state: SharedState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/ws", get(ws_handler))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_loop(socket, state))
}

async fn ws_loop(mut socket: WebSocket, state: SharedState) {
    let mut ticker = tokio::time::interval(PUSH_INTERVAL);
    // Skip (not burst) missed ticks: if a snapshot serialization runs long, we
    // resume at the next interval instead of firing a rapid catch-up burst —
    // keeping the push rate at ≤10 Hz as intended.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let snapshot = snapshot_json(&state);
        if socket.send(Message::Text(snapshot)).await.is_err() {
            break; // client disconnected
        }
    }
}

/// Serializes the current shared state to JSON. Falls back to `{}` if a prior
/// panic poisoned the lock, so a WS connection degrades rather than crashes.
fn snapshot_json(state: &SharedState) -> String {
    match state.lock() {
        Ok(guard) => serde_json::to_string(&*guard).unwrap_or_else(|_| "{}".to_string()),
        Err(_) => "{}".to_string(),
    }
}

/// Spawns a task draining `events` into a fresh shared [`WebState`] and returns
/// it. The caller serves the [`app`] over the returned state.
///
/// The drain thread blocks on `recv()` and exits when the bus `Sender` is
/// dropped (i.e. the run ends). For `emry web`, the server runs for the process
/// lifetime, so this is the natural shutdown; there is no separate stop signal.
#[must_use]
pub fn spawn_state(events: Receiver<Event>) -> SharedState {
    spawn_state_with_labels(events, &[])
}

/// Like [`spawn_state`], but seeds metric `labels` (id → name) so the dashboard
/// shows real names instead of the `m{id}` fallback. The live bus carries only
/// [`MetricId`]s, so callers that know the names (e.g. the embedded SDK) pass
/// them here; the file-tail path resolves names from `metrics.jsonl`.
#[must_use]
pub fn spawn_state_with_labels(
    events: Receiver<Event>,
    labels: &[(MetricId, &str)],
) -> SharedState {
    let state: SharedState = Arc::new(Mutex::new(WebState::with_labels(labels)));
    let drain = Arc::clone(&state);
    std::thread::spawn(move || {
        while let Ok(event) = events.recv() {
            if let Ok(mut guard) = drain.lock() {
                guard.apply(&event);
            }
        }
    });
    state
}

/// Binds `addr`, drains `events` into the dashboard state, and serves the web UI.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the address cannot be bound or serving fails.
pub async fn serve(addr: SocketAddr, events: Receiver<Event>) -> std::io::Result<()> {
    serve_with_labels(addr, events, &[]).await
}

/// Like [`serve`], but seeds metric `labels` for real metric names.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the address cannot be bound or serving fails.
pub async fn serve_with_labels(
    addr: SocketAddr,
    events: Receiver<Event>,
    labels: &[(MetricId, &str)],
) -> std::io::Result<()> {
    let state = spawn_state_with_labels(events, labels);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(state)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use emry_core::{MetricId, Phase};
    use http_body_util::BodyExt;
    use tower::ServiceExt; // for `oneshot`

    fn shared() -> SharedState {
        Arc::new(Mutex::new(WebState::default()))
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let resp = app(shared())
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn index_serves_html() {
        let resp = app(shared())
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(html.starts_with("<!DOCTYPE html>"));
        // The dashboard ships its chart canvas, the WS wiring, and the brand
        // accent — and stays self-hosted (no external script/style URLs).
        assert!(html.contains("<canvas"));
        assert!(html.contains("/ws"));
        assert!(html.contains("#c4714a")); // terracotta
        assert!(!html.contains("http://") || html.contains("ws://"));
        assert!(!html.contains("https://"), "no CDN — air-gap friendly");
    }

    #[tokio::test]
    async fn unknown_route_404s() {
        let resp = app(shared())
            .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn snapshot_reflects_drained_events() {
        let (tx, rx) = crossbeam_channel::bounded(256);
        let state = spawn_state(rx);
        tx.send(Event::MetricsBatch {
            step: 3,
            epoch: 0,
            phase: Phase::Train,
            values: vec![(MetricId(0), 0.5)],
        })
        .unwrap();
        // Wait for the drain task to apply it.
        let mut json = String::new();
        for _ in 0..100 {
            json = snapshot_json(&state);
            if json.contains("\"step\":3") {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            json.contains("\"step\":3"),
            "drain task applied the event: {json}"
        );
    }
}
