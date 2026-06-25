"""Tests for the emry.run() API and the pure-Python JSONL backend."""

import json

import pytest

import emry
from emry import Phase, run
from emry.run import NullBackend, Run


class FakeBackend:
    """Records emit/finish calls for assertions."""

    def __init__(self):
        self.emitted = []  # (step, epoch, phase, values)
        self.finished_at = None

    def emit(self, step, epoch, phase, values):
        self.emitted.append((step, epoch, phase, values))

    def finish(self, *, steps):
        self.finished_at = steps


def test_emit_coerces_and_forwards_with_advancing_step():
    fake = FakeBackend()
    r = Run("p", fake)
    r.emit(loss=1.0, lr=2)
    r.emit(loss=0.5)
    assert [e[0] for e in fake.emitted] == [0, 1]  # step advances
    assert fake.emitted[0][3] == {"loss": 1.0, "lr": 2.0}  # int coerced to float


def test_phase_and_epoch_are_stamped():
    fake = FakeBackend()
    r = Run("p", fake)
    for epoch in r.epochs(2):
        r.phase = Phase.EVAL if epoch == 1 else Phase.TRAIN
        r.emit(loss=0.1)
    assert fake.emitted[0][1] == 0 and fake.emitted[0][2] == Phase.TRAIN
    assert fake.emitted[1][1] == 1 and fake.emitted[1][2] == Phase.EVAL


def test_steps_and_track_are_iteration_sugar():
    fake = FakeBackend()
    r = Run("p", fake)
    assert list(r.steps(3)) == [0, 1, 2]
    assert list(r.track(["a", "b"])) == ["a", "b"]


def test_context_manager_finishes_on_exit():
    fake = FakeBackend()
    with run("p", backend=fake) as r:
        r.emit(loss=1.0)
        r.emit(loss=0.9)
    assert fake.finished_at == 2


def test_finish_is_idempotent_and_blocks_further_emit():
    fake = FakeBackend()
    r = Run("p", fake)
    r.finish()
    r.finish()  # no-op, no error
    with pytest.raises(RuntimeError, match="after the run finished"):
        r.emit(loss=1.0)


def test_run_factory_uses_injected_backend():
    fake = FakeBackend()
    r = run("p", config={"lr": 1e-3}, metrics=["loss"], backend=fake)
    assert isinstance(r, Run)
    assert r.config == {"lr": 1e-3}
    assert r.metrics == ["loss"]


def test_null_backend_runs_clean():
    with run("p", backend=NullBackend()) as r:
        r.emit(loss=1.0)
    assert r.step == 1


def test_jsonl_backend_writes_wire_compatible_run_dir(tmp_path):
    with run("demo", config={"lr": 2e-5}, metrics=["loss"], log_dir=str(tmp_path), mode="file") as r:
        for _ in r.steps(3):
            r.emit(loss=1.0 / (r.step + 1))

    run_dir = next(tmp_path.iterdir())
    assert run_dir.name.startswith("demo_")

    # run.meta carries run_id / project / mode.
    meta = json.loads((run_dir / "run.meta").read_text())
    assert meta["project"] == "demo" and meta["mode"] == "file"
    assert len(meta["run_id"]) == 36  # uuid4

    # config.json holds the hyperparameters.
    assert json.loads((run_dir / "config.json").read_text())["lr"] == 2e-5

    # metrics.jsonl is the wide-row schema emry watch reads.
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0] == {"step": 0, "epoch": 0, "phase": "TRAIN", "values": {"loss": 1.0}}

    # summary.json records the step count.
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["steps"] == 3 and summary["reason"] == "COMPLETED"


def test_package_exports():
    assert emry.Phase is Phase
    assert callable(emry.run)
    assert emry.Phase.TRAIN.value == "TRAIN"
