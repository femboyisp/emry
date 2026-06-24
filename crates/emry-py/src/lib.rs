//! `PyO3` bindings for the Python `emry` package.

/// Package version string exposed to Python.
#[must_use]
pub fn emry_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_package() {
        assert_eq!(emry_version(), "0.1.0");
    }
}
