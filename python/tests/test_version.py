"""Tests for the emry Python package."""

from __future__ import annotations

import emry


def test_version_is_string() -> None:
    assert isinstance(emry.__version__, str)
    assert emry.__version__ == "0.1.0"
