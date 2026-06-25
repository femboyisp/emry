//! Unix-socket sidecar transport (Unix only).
//!
//! The engine binds a Unix socket on a shared/login node ([`bind`], perms
//! `0600`); a training process [`connect`]s and streams length-prefixed msgpack
//! frames ([`crate::wire`]). [`serve`] accepts a connection and reads its frames
//! until the peer disconnects, then waits for the next.
//!
//! Connections are handled one at a time. This matches Emry's model — distributed
//! ranks `all_reduce` in Python and only rank 0 emits — so a single streaming
//! sender is the expected case. While reading an open connection, the stop flag
//! is only re-checked once that connection closes.

use crate::wire::{read_frame, write_frame};
use emry_core::{EmryError, Event};
use std::io::{BufReader, BufWriter, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Permission bits for the socket file: owner read/write only.
pub const SOCKET_MODE: u32 = 0o600;

/// How long the accept loop sleeps between non-blocking accept attempts.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Binds a Unix socket at `path` with `0600` permissions, removing any stale
/// socket file first.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the path cannot be bound or its permissions set.
pub fn bind(path: &Path) -> Result<UnixListener, EmryError> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(SOCKET_MODE))?;
    Ok(listener)
}

/// Accepts connections on `listener` until `stop` is set, reading every framed
/// [`Event`] from each and passing it to `on_event`.
///
/// Returns when `stop` is observed between connections. A read error on a
/// connection ends that connection but not the loop.
///
/// # Errors
///
/// Returns [`EmryError::Io`] only for a fatal accept error.
pub fn serve<F: FnMut(Event)>(
    listener: &UnixListener,
    stop: &AtomicBool,
    mut on_event: F,
) -> Result<(), EmryError> {
    listener.set_nonblocking(true)?;
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                // Blocking reads for the lifetime of this connection.
                stream.set_nonblocking(false)?;
                let mut reader = BufReader::new(stream);
                // Read until the peer closes cleanly (Ok(None)) or errors; either
                // way we drop this connection and loop back to accept the next.
                while let Ok(Some(event)) = read_frame(&mut reader) {
                    on_event(event);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Connects to the engine socket at `path`.
///
/// # Errors
///
/// Returns [`EmryError::Io`] if the socket cannot be reached.
pub fn connect(path: &Path) -> Result<UnixStream, EmryError> {
    Ok(UnixStream::connect(path)?)
}

/// Sends `events` as frames over `stream` and flushes.
///
/// # Errors
///
/// Returns [`EmryError::Protocol`] / [`EmryError::Io`] on encode or write failure.
pub fn send_events(stream: &UnixStream, events: &[Event]) -> Result<(), EmryError> {
    let mut writer = BufWriter::new(stream);
    for event in events {
        write_frame(&mut writer, event)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use emry_core::{FinishReason, MetricId, Phase};
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Mutex};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_socket() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("emry-sock-{}-{n}.sock", std::process::id()))
    }

    fn events() -> Vec<Event> {
        vec![
            Event::MetricsBatch {
                step: 0,
                epoch: 0,
                phase: Phase::Train,
                values: vec![(MetricId(0), 1.0)],
            },
            Event::MetricsBatch {
                step: 1,
                epoch: 0,
                phase: Phase::Train,
                values: vec![(MetricId(0), 0.5)],
            },
            Event::RunFinished {
                duration_secs: 1.0,
                reason: FinishReason::Completed,
            },
        ]
    }

    #[test]
    fn bind_sets_0600_permissions() {
        let path = temp_socket();
        let _listener = bind(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, SOCKET_MODE);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bind_replaces_stale_socket_file() {
        let path = temp_socket();
        std::fs::write(&path, b"stale").unwrap();
        let _listener = bind(&path).unwrap(); // must not fail on the existing file
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn client_send_is_received_by_server() {
        let path = temp_socket();
        let listener = bind(&path).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let received = Arc::new(Mutex::new(Vec::new()));

        let server = {
            let stop = Arc::clone(&stop);
            let received = Arc::clone(&received);
            std::thread::spawn(move || {
                serve(&listener, &stop, |event| {
                    received.lock().unwrap().push(event);
                })
                .unwrap();
            })
        };

        // Client streams events, then disconnects (drop = clean EOF).
        let stream = connect(&path).unwrap();
        send_events(&stream, &events()).unwrap();
        drop(stream);

        // Give the server a moment to drain the connection, then stop it.
        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Release);
        server.join().unwrap();

        let got = received.lock().unwrap();
        assert_eq!(got.len(), 3);
        assert!(matches!(got[2], Event::RunFinished { .. }));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn serve_returns_when_stopped_with_no_clients() {
        let path = temp_socket();
        let listener = bind(&path).unwrap();
        let stop = AtomicBool::new(true); // already stopped
        serve(&listener, &stop, |_| panic!("no events expected")).unwrap();
        std::fs::remove_file(&path).ok();
    }
}
