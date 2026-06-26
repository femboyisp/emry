//! Web dashboard (axum + WebSocket + uPlot).

pub mod baseline;
pub mod server;
pub mod state;

pub use baseline::{load_baseline, Baseline, BaselineSeries};
pub use server::{
    app, app_with_baseline, serve, serve_with_baseline, serve_with_labels, spawn_state,
    spawn_state_with_labels, AppState, SharedState, PUSH_INTERVAL,
};
pub use state::{WebAlert, WebMetric, WebState};
