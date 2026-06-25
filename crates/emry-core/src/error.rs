//! Error types for Emry.

use thiserror::Error;

/// Top-level error type for the Emry library.
#[derive(Debug, Error)]
pub enum EmryError {
    /// I/O failure while reading or writing run data.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Wire-protocol framing or msgpack (de)serialization failed.
    #[error("protocol error: {0}")]
    Protocol(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn io_error_display() {
        let err = EmryError::Io(IoError::new(ErrorKind::NotFound, "missing"));
        assert_eq!(err.to_string(), "io error: missing");
    }

    #[test]
    fn json_error_from_invalid_value() {
        let err = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let wrapped = EmryError::Json(err);
        assert!(wrapped.to_string().starts_with("json error:"));
    }

    #[test]
    fn protocol_error_display() {
        let err = EmryError::Protocol("frame too large".to_string());
        assert_eq!(err.to_string(), "protocol error: frame too large");
    }
}
