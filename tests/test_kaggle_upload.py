"""RED: kaggle upload script contract (loud misuse, silent skip)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("kaggle_upload", Path("scripts/kaggle_upload.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_when_bad_slug_then_value_error() -> None:
    mod = _load()
    with pytest.raises(ValueError):
        mod.upload_dataset("a/b")
    with pytest.raises(ValueError):
        mod.upload_dataset("")


def test_when_no_credentials_then_skip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    mod = _load()
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert mod.upload_dataset("anyslug", _skip=True) is None


def test_when_username_env_then_resolved(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    mod = _load()
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    assert mod._resolve_username() == "someone"
