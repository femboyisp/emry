//! Terminal dashboard (ratatui).

pub mod chart;

pub use chart::{downsample_minmax, render_braille};
pub use emry_core::Phase;
