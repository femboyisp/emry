//! Terminal dashboard (ratatui).

pub use emry_core::Phase;

#[cfg(test)]
mod tests {
    use super::Phase;

    #[test]
    fn phase_reexport_matches_core() {
        assert!(matches!(Phase::Eval, Phase::Eval));
    }
}
