"""Tests for emry.mode — must mirror the Rust DeployMode (EMRY-005)."""

import pytest

from emry.mode import DeployEnv, DeployMode, detect, resolve


@pytest.mark.parametrize(
    "text,expected",
    [
        ("embedded", DeployMode.EMBEDDED),
        ("EMBEDDED", DeployMode.EMBEDDED),
        ("  Sidecar ", DeployMode.SIDECAR),
        ("file", DeployMode.FILE),
        ("nonsense", None),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_parse(text, expected):
    assert DeployMode.parse(text) == expected


@pytest.mark.parametrize(
    "name,api,emry_mode,ssh,slurm,tty,expected",
    [
        ("api overrides everything", DeployMode.FILE, "embedded", True, True, True, DeployMode.FILE),
        ("EMRY_MODE beats auto", None, "sidecar", False, False, True, DeployMode.SIDECAR),
        ("invalid EMRY_MODE falls through", None, "nonsense", False, False, True, DeployMode.EMBEDDED),
        ("empty EMRY_MODE falls through", None, "", False, False, False, DeployMode.FILE),
        ("whitespace EMRY_MODE falls through", None, "   ", True, False, False, DeployMode.SIDECAR),
        ("SSH -> sidecar", None, None, True, False, True, DeployMode.SIDECAR),
        ("SLURM -> sidecar", None, None, False, True, False, DeployMode.SIDECAR),
        ("TTY -> embedded", None, None, False, False, True, DeployMode.EMBEDDED),
        ("no signals -> file", None, None, False, False, False, DeployMode.FILE),
    ],
)
def test_resolve_precedence(name, api, emry_mode, ssh, slurm, tty, expected):
    env = DeployEnv(
        emry_mode=emry_mode,
        ssh_connection=ssh,
        slurm_job_id=slurm,
        stdout_is_tty=tty,
    )
    assert resolve(api, env) == expected, name


def test_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("EMRY_MODE", "sidecar")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    env = DeployEnv.from_env()
    assert env.emry_mode == "sidecar"
    assert env.slurm_job_id is True
    assert env.ssh_connection is False


def test_from_env_handles_unset(monkeypatch):
    monkeypatch.delenv("EMRY_MODE", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    env = DeployEnv.from_env()
    assert env.emry_mode is None
    assert env.ssh_connection is False
    assert env.slurm_job_id is False


def test_detect_against_real_env_returns_valid_mode():
    assert detect() in (DeployMode.EMBEDDED, DeployMode.SIDECAR, DeployMode.FILE)
    # An explicit override is honoured regardless of environment.
    assert detect(DeployMode.FILE) is DeployMode.FILE


def test_mode_values_match_rust_display():
    # The Rust Display strings these must agree with.
    assert DeployMode.EMBEDDED.value == "embedded"
    assert DeployMode.SIDECAR.value == "sidecar"
    assert DeployMode.FILE.value == "file"
