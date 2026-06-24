//! Persistence: JSONL, Parquet, rotation.

pub use emry_core::EmryError;

#[cfg(test)]
mod tests {
    use super::EmryError;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn error_type_is_shared() {
        let err = EmryError::Io(IoError::new(ErrorKind::Other, "disk"));
        assert!(err.to_string().contains("disk"));
    }
}
