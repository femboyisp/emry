//! Sidecar wire protocol: length-prefixed msgpack frames.
//!
//! # Frame format
//!
//! Each [`Event`] is sent as one frame:
//!
//! ```text
//! ┌────────────────────┬──────────────────────────┐
//! │ u32 length (LE)    │ msgpack-encoded Event      │
//! │ 4 bytes            │ `length` bytes             │
//! └────────────────────┴──────────────────────────┘
//! ```
//!
//! The length prefix is the byte count of the msgpack payload, little-endian.
//! Frames are written back-to-back on the stream; a clean close (EOF on a frame
//! boundary) ends the stream.
//!
//! # msgpack encoding
//!
//! [`Event`] is an adjacently-tagged enum, which only roundtrips through msgpack
//! when structs are encoded **as maps** — so the encoder uses
//! `rmp_serde::Serializer::with_struct_map` (the default compact/sequence
//! encoding would fail to deserialize). This was discovered in EMRY-002.
//!
//! Frames larger than [`MAX_FRAME_BYTES`] are rejected on read to bound memory
//! against a corrupt or hostile length prefix.

use emry_core::{EmryError, Event};
use serde::Serialize;
use std::io::{ErrorKind, Read, Write};

/// Maximum accepted frame payload size (16 MiB). A larger length prefix is
/// treated as a protocol error rather than allocating it.
pub const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;

/// Encodes `event` to its msgpack payload (map-encoded structs; no length
/// prefix).
///
/// # Errors
///
/// Returns [`EmryError::Protocol`] if msgpack serialization fails.
pub fn encode(event: &Event) -> Result<Vec<u8>, EmryError> {
    let mut payload = Vec::new();
    event
        .serialize(&mut rmp_serde::Serializer::new(&mut payload).with_struct_map())
        .map_err(|e| EmryError::Protocol(e.to_string()))?;
    Ok(payload)
}

/// Writes one length-prefixed frame for `event` to `w`.
///
/// # Errors
///
/// Returns [`EmryError::Protocol`] on encode failure or if the payload exceeds
/// [`MAX_FRAME_BYTES`], or [`EmryError::Io`] on write failure.
pub fn write_frame<W: Write>(w: &mut W, event: &Event) -> Result<(), EmryError> {
    let payload = encode(event)?;
    let len = u32::try_from(payload.len())
        .ok()
        .filter(|&n| n <= MAX_FRAME_BYTES)
        .ok_or_else(|| {
            EmryError::Protocol(format!("frame of {} bytes too large", payload.len()))
        })?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&payload)?;
    Ok(())
}

/// Reads one frame from `r`, returning `Ok(None)` on a clean EOF at a frame
/// boundary (the stream ended).
///
/// # Errors
///
/// Returns [`EmryError::Protocol`] if the length prefix exceeds
/// [`MAX_FRAME_BYTES`] or the payload is not valid msgpack, or [`EmryError::Io`]
/// on a short read (a truncated frame).
pub fn read_frame<R: Read>(r: &mut R) -> Result<Option<Event>, EmryError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        // EOF exactly at a frame boundary: the peer closed cleanly.
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(EmryError::Protocol(format!(
            "frame length {len} exceeds cap {MAX_FRAME_BYTES}"
        )));
    }
    let mut payload = vec![0u8; len as usize];
    r.read_exact(&mut payload)?; // a short read here = truncated frame = Io error
    let event = rmp_serde::from_slice(&payload).map_err(|e| EmryError::Protocol(e.to_string()))?;
    Ok(Some(event))
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::{FinishReason, MetricId, Phase};
    use std::io::Cursor;

    fn sample_events() -> Vec<Event> {
        vec![
            Event::MetricsBatch {
                step: 7,
                epoch: 1,
                phase: Phase::Train,
                values: vec![(MetricId(0), 0.5), (MetricId(1), 1e-3)],
            },
            Event::PhaseChange(Phase::Eval),
            Event::RunFinished {
                duration_secs: 12.0,
                reason: FinishReason::Completed,
            },
        ]
    }

    #[test]
    fn frame_roundtrips_through_msgpack() {
        for event in sample_events() {
            let mut buf = Vec::new();
            write_frame(&mut buf, &event).unwrap();
            let mut cursor = Cursor::new(buf);
            let back = read_frame(&mut cursor).unwrap();
            assert_eq!(back, Some(event));
        }
    }

    #[test]
    fn multiple_frames_read_in_order_then_eof() {
        let events = sample_events();
        let mut buf = Vec::new();
        for event in &events {
            write_frame(&mut buf, event).unwrap();
        }
        let mut cursor = Cursor::new(buf);
        for event in &events {
            assert_eq!(read_frame(&mut cursor).unwrap().as_ref(), Some(event));
        }
        // Clean EOF at the boundary.
        assert_eq!(read_frame(&mut cursor).unwrap(), None);
    }

    #[test]
    fn empty_stream_is_clean_eof() {
        let mut cursor = Cursor::new(Vec::new());
        assert_eq!(read_frame(&mut cursor).unwrap(), None);
    }

    #[test]
    fn oversized_length_prefix_is_rejected() {
        let mut buf = (MAX_FRAME_BYTES + 1).to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 8]);
        let mut cursor = Cursor::new(buf);
        let err = read_frame(&mut cursor).unwrap_err();
        assert!(matches!(err, EmryError::Protocol(_)));
    }

    #[test]
    fn truncated_payload_is_io_error() {
        // Claims 100 bytes but provides only 2.
        let mut buf = 100u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 2]);
        let mut cursor = Cursor::new(buf);
        let err = read_frame(&mut cursor).unwrap_err();
        assert!(matches!(err, EmryError::Io(_)));
    }

    #[test]
    fn garbage_payload_is_protocol_error() {
        // Valid length, but the payload is not valid msgpack for an Event.
        let mut buf = 3u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0xff, 0xff, 0xff]);
        let mut cursor = Cursor::new(buf);
        let err = read_frame(&mut cursor).unwrap_err();
        assert!(matches!(err, EmryError::Protocol(_)));
    }
}
