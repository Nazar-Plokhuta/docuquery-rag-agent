"""Unit tests for boot-time environment checks in ``src.main``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from src.main import _warn_if_not_writable


def test_warns_with_remedy_when_data_directory_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(os, "access", lambda *_args, **_kwargs: False)

    with caplog.at_level(logging.WARNING, logger="src.main"):
        _warn_if_not_writable(tmp_path)

    assert "not writable" in caplog.text
    assert "read-only database" in caplog.text
    assert "APP_UID" in caplog.text


def test_stays_silent_when_data_directory_is_writable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.main"):
        _warn_if_not_writable(tmp_path)

    assert caplog.text == ""
