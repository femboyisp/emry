//! Web dashboard (axum + WebSocket + uPlot).

pub mod baseline;
pub mod project;
pub mod security;
pub mod server;
pub mod state;

pub use baseline::{Baseline, BaselineSeries};
pub use project::{app_project, serve_project, Project, ProjectRun, ProjectSeries};
pub use security::{TlsConfig, WebSecurity};
pub use server::{
    app, app_with_baseline, serve, serve_with_baseline, serve_with_labels, spawn_state,
    spawn_state_with_labels, AppState, SharedState, PUSH_INTERVAL,
};
pub use state::{WebAlert, WebMetric, WebState};
