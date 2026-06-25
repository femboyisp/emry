"""Tests for the PyO3 native backend (EMRY-033).

The direct-native test is skipped when the extension isn't built (the pure-Python
CI job); it runs locally after ``maturin develop`` and in the wheel-build job.
The embedded-mode test works either way — natively when built, via the JSONL
fallback otherwise.
"""

import json

import pytest

from emry import run


def test_native_extension_drives_the_engine(tmp_path):
    native = pytest.importorskip("emry._native")
    handle = native.RunHandle("nativetest", str(tmp_path), ["loss", "lr"], 100)
    loss = handle.register("loss")
    lr = handle.register("lr")
    handle.set_phase("TRAIN")
    for i in range(20):
        handle.emit([(loss, 1.0 / (i + 1)), (lr, 1e-3)])
    assert handle.dropped() == 0
    handle.finish()
    handle.finish()  # idempotent

    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 20
    first = json.loads(rows[0])
    assert first["step"] == 0 and first["values"]["loss"] == 1.0
    assert (tmp_path / "run.meta").exists()


def test_native_set_phase_rejects_unknown(tmp_path):
    native = pytest.importorskip("emry._native")
    handle = native.RunHandle("p", str(tmp_path), [], None)
    with pytest.raises(ValueError):
        handle.set_phase("BOGUS")
    handle.finish()


def test_embedded_mode_produces_run_dir(tmp_path):
    # Native when the extension is built; JSONL fallback otherwise. Either path
    # must yield a run directory with the expected number of metric rows.
    with run("emb", metrics=["loss"], mode="embedded", log_dir=str(tmp_path)) as r:
        for _ in r.steps(5):
            r.emit(loss=1.0 / (r.step + 1))
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    rows = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 5
