"""Tests for the bundled-CLI shim (``emry._cli``)."""

import os
import sys
from pathlib import Path

import pytest

from emry import _cli


def test_binary_path_present(monkeypatch, tmp_path):
    (tmp_path / "_bin").mkdir()
    exe = tmp_path / "_bin" / _cli._EXE
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(_cli, "__file__", str(tmp_path / "_cli.py"))
    assert _cli.binary_path() == exe


def test_binary_path_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(_cli, "__file__", str(tmp_path / "_cli.py"))
    assert _cli.binary_path() is None


def test_main_execs_bundled_binary(monkeypatch):
    called = {}
    monkeypatch.setattr(_cli, "binary_path", lambda: Path("/opt/emry/_bin/emry"))
    monkeypatch.setattr(sys, "argv", ["emry", "runs", "--log-dir", "x"])
    monkeypatch.setattr(os, "execv", lambda path, argv: called.update(path=path, argv=argv))
    _cli.main()
    assert called["path"] == "/opt/emry/_bin/emry"
    assert called["argv"] == ["/opt/emry/_bin/emry", "runs", "--log-dir", "x"]


def test_main_errors_when_binary_missing(monkeypatch):
    monkeypatch.setattr(_cli, "binary_path", lambda: None)
    monkeypatch.setattr(sys, "argv", ["emry"])
    with pytest.raises(SystemExit) as exc:
        _cli.main()
    assert "bundled CLI binary is missing" in str(exc.value)
