//! Ingest: Unix socket, file tail, wire protocol.

pub use emry_core::EmryError;

#[cfg(test)]
mod tests {
    use super::EmryError;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn ingest_uses_core_errors() {
        let err = EmryError::Io(IoError::new(ErrorKind::BrokenPipe, "socket"));
        assert!(err.to_string().contains("socket"));
    }
}
