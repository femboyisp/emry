"""Tests for live observer resolution and spawning (EMRY-035)."""

from pathlib import Path

import pytest

from emry import live, run


@pytest.mark.parametrize(
    "value,ssh,tty,force,expected",
    [
        (False, False, True, False, []),
        (None, False, True, False, []),
        ("off", False, True, False, []),
        ("none", True, True, True, []),
        ("tui", False, False, False, ["tui"]),
        ("web", False, False, False, ["web"]),
        ("both", False, False, False, ["tui", "web"]),
        # auto: SSH -> web, local TTY -> tui, headless -> nothing
        ("auto", True, False, False, ["web"]),
        ("auto", False, True, False, ["tui"]),
        ("auto", False, False, False, []),
        # EMRY_LIVE_FORCE flips SSH back to TUI
        ("auto", True, False, True, ["tui"]),
        # SSH + local TTY still defaults to web (force flips it to tui)
        ("auto", True, True, False, ["web"]),
        ("auto", True, True, True, ["tui"]),
        ("bogus", False, True, False, []),
    ],
)
def test_resolve_live(value, ssh, tty, force, expected):
    assert live.resolve_live(value, ssh=ssh, tty=tty, force_tui=force) == expected


def test_observer_command_tui_targets():
    assert live.observer_command("tui", run_dir=Path("/logs/r"), socket_path=None) == [
        "emry",
        "tui",
        "--run-dir",
        "/logs/r",
    ]
    # Socket takes precedence over run_dir.
    assert live.observer_command("tui", run_dir=Path("/logs/r"), socket_path="/s.sock") == [
        "emry",
        "tui",
        "--socket",
        "/s.sock",
    ]
    # No target -> can't run.
    assert live.observer_command("tui", run_dir=None, socket_path=None) is None


def test_observer_command_web_not_available_yet():
    assert live.observer_command("web", run_dir=Path("/logs/r"), socket_path=None) is None


def test_spawn_observers_builds_argv_without_launching(monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))

    monkeypatch.setattr(live.subprocess, "Popen", FakePopen)
    procs = live.spawn_observers(["tui"], run_dir=Path("/logs/r"), socket_path=None)
    assert len(procs) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["emry", "tui", "--run-dir", "/logs/r"]
    assert kwargs.get("start_new_session") is True


def test_spawn_observers_warns_on_web(monkeypatch):
    monkeypatch.setattr(live.subprocess, "Popen", lambda *a, **k: None)
    with pytest.warns(UserWarning, match="not available"):
        procs = live.spawn_observers(["web"], run_dir=Path("/logs/r"), socket_path=None)
    assert procs == []


def test_spawn_observers_warns_on_missing_binary(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError

    monkeypatch.setattr(live.subprocess, "Popen", boom)
    with pytest.warns(UserWarning, match="emry"):
        live.spawn_observers(["tui"], run_dir=Path("/logs/r"), socket_path=None)


def test_run_auto_does_not_spawn_in_non_interactive(tmp_path, monkeypatch):
    # Default live="auto" with no TTY/SSH must not launch anything.
    launched = []
    monkeypatch.setattr(
        "emry.live.subprocess.Popen", lambda *a, **k: launched.append(a)
    )
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    with run("p", metrics=["loss"], log_dir=str(tmp_path)) as r:
        r.emit(loss=1.0)
    assert launched == []


def test_run_tui_spawns(tmp_path, monkeypatch):
    launched = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            launched.append(cmd)

    monkeypatch.setattr("emry.live.subprocess.Popen", FakePopen)
    with run("p", metrics=["loss"], log_dir=str(tmp_path), live="tui") as r:
        r.emit(loss=1.0)
    assert len(launched) == 1
    assert launched[0][:2] == ["emry", "tui"]
    assert launched[0][2] == "--run-dir"
