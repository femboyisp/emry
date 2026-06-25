//! Ingest: Unix socket, file tail, wire protocol.

pub mod watch;
pub mod wire;

#[cfg(unix)]
pub mod socket;

pub use emry_core::EmryError;
pub use watch::{run_watch, JsonlTailer};
pub use wire::{read_frame, write_frame, MAX_FRAME_BYTES};

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
