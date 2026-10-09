"""RED: resolve_class_names shared by CLI prepare and build_run_plan (DRY)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surfacedeformdetectionpoc import main
from surfacedeformdetectionpoc.train import resolve_class_names
from tests.helpers import write_image


def test_when_meta_and_classes_both_absent_then_value_error() -> None:
    with pytest.raises(ValueError, match="class names"):
        resolve_class_names(meta=None, classes=None)


def test_when_both_present_then_meta_wins(tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"classes": [{"title": "from_meta"}]}))
    assert resolve_class_names(meta=str(meta), classes=["from_cli"]) == ["from_meta"]


def test_when_cli_prepare_lacks_both_then_raises(tmp_path: Path) -> None:
    src = tmp_path / "data"
    write_image(src / "s.jpg", 10, 10)
    with pytest.raises(ValueError, match="class names"):
        main(
            [
                "prepare",
                "--data",
                str(src),
                "--out",
                str(tmp_path / "yolo"),
                "--classes",
            ]
        )
