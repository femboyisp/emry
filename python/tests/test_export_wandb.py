"""Tests for the post-hoc Weights & Biases exporter (fake `wandb` injected)."""

import json
import sys
import types

import pytest

from emry.export_wandb import _main, to_wandb


class _FakeRun:
    """Records log() calls and exposes a dict-like summary."""

    def __init__(self):
        self.logged = []  # list of (values, step)
        self.summary = {}

    def log(self, values, step=None):
        self.logged.append((values, step))


def _fake_wandb():
    """A stand-in `wandb` module recording init/log/finish."""
    module = types.ModuleType("wandb")
    calls = {"init": None, "finished": 0, "run": None}

    def init(*, project=None, entity=None, name=None, config=None):
        run = _FakeRun()
        calls["init"] = {
            "project": project,
            "entity": entity,
            "name": name,
            "config": config,
        }
        calls["run"] = run
        return run

    def finish():
        calls["finished"] += 1

    module.init = init
    module.finish = finish
    module._calls = calls
    return module, calls


def _write_run(run_dir):
    """Writes a minimal finished run directory in the on-disk format."""
    (run_dir / "run.meta").write_text(
        json.dumps({"run_id": "r1", "project": "demo", "mode": "file"})
    )
    (run_dir / "config.json").write_text(json.dumps({"lr": 0.1, "model": "x"}))
    rows = [
        {"step": 0, "epoch": 0, "phase": "TRAIN", "values": {"loss": 2.0}},
        # deliberately out of order to prove step-sorting
        {"step": 2, "epoch": 1, "phase": "EVAL", "values": {"loss": 1.0, "acc": 0.6}},
        {"step": 1, "epoch": 0, "phase": "TRAIN", "values": {"loss": 1.5}},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"run_id": "r1", "project": "demo", "steps": 3, "reason": "COMPLETED"})
    )


def test_replays_run_into_wandb(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _write_run(tmp_path)

    run = to_wandb(tmp_path, entity="team", name="myrun")

    # init got the run's own project (from run.meta), passed entity/name/config.
    assert calls["init"]["project"] == "demo"
    assert calls["init"]["entity"] == "team"
    assert calls["init"]["name"] == "myrun"
    assert calls["init"]["config"] == {"lr": 0.1, "model": "x"}

    # one log() per row, in ascending step order, with the right step= and values.
    assert run.logged == [
        ({"loss": 2.0}, 0),
        ({"loss": 1.5}, 1),
        ({"loss": 1.0, "acc": 0.6}, 2),
    ]

    # summary values copied over, and finish() called exactly once.
    assert run.summary["steps"] == 3
    assert run.summary["reason"] == "COMPLETED"
    assert calls["finished"] == 1


def test_explicit_project_and_config_override(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _write_run(tmp_path)

    to_wandb(tmp_path, project="override", config={"lr": 0.2, "extra": 1})

    assert calls["init"]["project"] == "override"
    # config override merges onto (and wins over) the run's stored config.
    assert calls["init"]["config"] == {"lr": 0.2, "model": "x", "extra": 1}


def test_missing_files_are_tolerated(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    # Empty run dir: no meta/config/metrics/summary.
    run = to_wandb(tmp_path, project="p")

    assert calls["init"]["project"] == "p"
    assert calls["init"]["config"] == {}
    assert run.logged == []
    assert run.summary == {}
    assert calls["finished"] == 1


def test_cli_invokes_to_wandb(tmp_path, monkeypatch):
    fake, calls = _fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    _write_run(tmp_path)

    _main([str(tmp_path), "--project", "cli-proj", "--entity", "team", "--name", "n"])

    assert calls["init"] == {
        "project": "cli-proj",
        "entity": "team",
        "name": "n",
        "config": {"lr": 0.1, "model": "x"},
    }
    assert calls["finished"] == 1


def test_missing_dependency_raises_clear_error(tmp_path, monkeypatch):
    # Ensure importing `wandb` fails even if it is installed in the env.
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"pip install emry\[wandb\]"):
        to_wandb(tmp_path)
