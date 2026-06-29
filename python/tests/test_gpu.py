"""Tests for the nvidia-smi GPU sampler (subprocess + clock injected)."""

import pytest

from emry import run
from emry.gpu import GpuSampler

# One GPU at 73% util, 4096/16384 MiB, 61°C; a second idle GPU.
SAMPLE_CSV = "0, 73, 4096, 16384, 61\n1, 0, 12, 16384, 35\n"


def test_parses_nvidia_smi_into_per_gpu_metrics():
    s = GpuSampler(runner=lambda: SAMPLE_CSV, clock=lambda: 0.0)
    out = s.sample()
    assert out["gpu0_util"] == 73.0
    assert out["gpu0_mem_mb"] == 4096.0
    assert out["gpu0_mem_pct"] == pytest.approx(25.0)
    assert out["gpu0_temp_c"] == 61.0
    assert out["gpu1_util"] == 0.0  # second GPU present


def test_throttles_to_one_query_per_interval():
    calls = {"n": 0}

    def runner():
        calls["n"] += 1
        return SAMPLE_CSV

    now = {"t": 100.0}
    s = GpuSampler(interval=1.0, runner=runner, clock=lambda: now["t"])
    assert s.sample()  # first call queries
    now["t"] = 100.5
    assert s.sample() == {}  # within interval -> skipped, no query
    assert calls["n"] == 1
    now["t"] = 101.5
    assert s.sample()  # interval elapsed -> queries again
    assert calls["n"] == 2


def test_failure_disables_sampling_silently():
    def boom():
        raise OSError("nvidia-smi exploded")

    s = GpuSampler(runner=boom, clock=lambda: 0.0)
    assert s.sample() == {}  # no raise
    # Disabled thereafter even once time advances.
    s._last = float("-inf")
    assert s.sample() == {}


def test_malformed_rows_are_skipped():
    s = GpuSampler(runner=lambda: "garbage\n0, 50, 1024, 2048, 40\n", clock=lambda: 0.0)
    out = s.sample()
    assert out == {
        "gpu0_util": 50.0,
        "gpu0_mem_mb": 1024.0,
        "gpu0_mem_pct": pytest.approx(50.0),
        "gpu0_temp_c": 40.0,
    }


def test_run_merges_gpu_metrics_into_emit(tmp_path):
    # Inject a sampler so the run emits GPU stats alongside the metric.
    sampler = GpuSampler(runner=lambda: "0, 90, 8192, 16384, 70\n", clock=lambda: 0.0)
    sampler._interval = 0.0  # always fresh for the test
    with run(
        "gpu-demo", metrics=["loss"], mode="file", log_dir=str(tmp_path), gpu=sampler
    ) as r:
        r.emit(loss=1.0)

    import json

    run_dir = next(tmp_path.iterdir())
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["values"]["loss"] == 1.0
    assert rows[0]["values"]["gpu0_util"] == 90.0
    assert rows[0]["values"]["gpu0_mem_pct"] == pytest.approx(50.0)
