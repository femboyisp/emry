//! Run engine: processors, pipeline, and `RunHandle`.

pub use emry_core::{EmryError, Phase, RunMeta};

#[cfg(test)]
mod tests {
    use super::Phase;

    #[test]
    fn reexports_are_usable() {
        assert_eq!(format!("{:?}", Phase::Train), "Train");
    }
}
