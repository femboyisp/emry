"""Tests for the sidecar socket backend and wire encoding."""

import socket
import struct
import threading

import pytest

msgpack = pytest.importorskip("msgpack")

from emry import wire
from emry.phase import Phase
from emry.run import run
from emry.socket_backend import SocketBackend


def decode_frames(data: bytes):
    """Splits a byte stream of `[u32 LE len][msgpack]` frames into events."""
    events, off = [], 0
    while off + 4 <= len(data):
        (length,) = struct.unpack_from("<I", data, off)
        off += 4
        events.append(msgpack.unpackb(data[off : off + length], raw=False))
        off += length
    return events


def test_metrics_batch_wire_shape():
    ev = wire.metrics_batch(7, 1, "TRAIN", [(0, 0.5), (1, 1e-3)])
    assert ev == {
        "type": "METRICS_BATCH",
        "data": {"step": 7, "epoch": 1, "phase": "TRAIN", "values": [[0, 0.5], [1, 0.001]]},
    }


def test_frame_roundtrips_through_msgpack():
    ev = wire.run_finished(12.0, "COMPLETED")
    framed = wire.frame(ev)
    (length,) = struct.unpack_from("<I", framed, 0)
    assert length == len(framed) - 4
    assert msgpack.unpackb(framed[4:], raw=False) == ev


def test_oversize_frame_rejected(monkeypatch):
    monkeypatch.setattr(wire, "MAX_FRAME_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds"):
        wire.frame(wire.phase_change("TRAIN"))


def test_socket_backend_streams_framed_events():
    # A real connected socket pair stands in for the engine; the test reads the
    # frames the backend sends and decodes them.
    server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    received = bytearray()

    def reader():
        while True:
            chunk = server.recv(65536)
            if not chunk:
                break
            received.extend(chunk)

    t = threading.Thread(target=reader)
    t.start()

    backend = SocketBackend(client, start_secs=0.0)
    backend._ids = {"loss": 0}
    backend.emit(0, 0, Phase.TRAIN, {"loss": 1.0})
    backend.emit(1, 0, Phase.EVAL, {"loss": 0.5, "acc": 0.9})
    backend.finish(steps=2, reason="COMPLETED")
    t.join()
    server.close()

    events = decode_frames(bytes(received))
    types = [e["type"] for e in events]
    # Initial PhaseChange(TRAIN), TRAIN batch, PhaseChange(EVAL), EVAL batch,
    # RunFinished — the first phase is signalled so observers don't read stale.
    assert types == [
        "PHASE_CHANGE",
        "METRICS_BATCH",
        "PHASE_CHANGE",
        "METRICS_BATCH",
        "RUN_FINISHED",
    ]
    assert events[0]["data"] == "TRAIN"
    assert events[1]["data"]["values"] == [[0, 1.0]]
    assert events[2]["data"] == "EVAL"
    # A new metric name gets the next client-local id.
    assert events[3]["data"]["values"] == [[0, 0.5], [1, 0.9]]
    assert events[4]["data"]["reason"] == "COMPLETED"


def test_run_factory_streams_to_sidecar_engine(tmp_path):
    sock_path = str(tmp_path / "e.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    listener.listen(1)
    received = bytearray()

    def serve():
        conn, _ = listener.accept()
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
        conn.close()

    t = threading.Thread(target=serve)
    t.start()

    import os

    os.environ["EMRY_SOCKET"] = sock_path
    try:
        with run("sc", metrics=["loss"], mode="sidecar") as r:
            r.emit(loss=1.0)
    finally:
        del os.environ["EMRY_SOCKET"]
    t.join()
    listener.close()

    events = decode_frames(bytes(received))
    # The client streams metrics + finish; the engine owns run identity, so no
    # RunStarted is sent.
    assert all(e["type"] != "RUN_STARTED" for e in events)
    assert any(e["type"] == "METRICS_BATCH" for e in events)
    assert events[-1]["type"] == "RUN_FINISHED"


def test_sidecar_falls_back_to_jsonl_when_no_engine(tmp_path, monkeypatch):
    # No engine listening: run() should fall back to the file backend (with a
    # warning), not raise.
    monkeypatch.setenv("EMRY_SOCKET", str(tmp_path / "absent.sock"))
    with pytest.warns(UserWarning, match="not reachable"):
        with run("sc", metrics=["loss"], mode="sidecar", log_dir=str(tmp_path)) as r:
            r.emit(loss=1.0)
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert (run_dir / "metrics.jsonl").exists()


def test_sidecar_without_socket_warns_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("EMRY_SOCKET", raising=False)
    with pytest.warns(UserWarning, match="EMRY_SOCKET is unset"):
        with run("sc", metrics=["loss"], mode="sidecar", log_dir=str(tmp_path)) as r:
            r.emit(loss=1.0)
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert (run_dir / "metrics.jsonl").exists()
