//! Terminal dashboard (ratatui).

pub mod app;
pub mod chart;
pub mod ui;

pub use app::{map_key, run, run_terminal, Action, RunLimit};
pub use chart::{downsample_minmax, render_braille};
pub use emry_core::Phase;
pub use ui::{render, MetricView, UiState};
