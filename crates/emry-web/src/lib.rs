//! Web dashboard (axum + WebSocket + uPlot).

pub mod server;
pub mod state;

pub use server::{app, serve, spawn_state, SharedState, PUSH_INTERVAL};
pub use state::{WebAlert, WebMetric, WebState};
