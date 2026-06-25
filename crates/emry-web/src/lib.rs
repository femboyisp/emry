//! Web dashboard (axum + WebSocket + uPlot).

pub mod server;
pub mod state;

pub use server::{
    app, serve, serve_with_labels, spawn_state, spawn_state_with_labels, SharedState, PUSH_INTERVAL,
};
pub use state::{WebAlert, WebMetric, WebState};
