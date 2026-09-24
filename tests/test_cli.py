"""Slice 6 RED: CLI prepare path + meta class names."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def test_when_meta_then_titles_in_order(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import class_names_from_meta

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"classes": [{"title": "defect_2"}, {"title": "defect_1"}]}))
    assert class_names_from_meta(meta) == ["defect_2", "defect_1"]


def test_when_prepare_then_dataset_built(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc import main

    src = tmp_path / "data"
    src.mkdir()
    cv2.imwrite(str(src / "s.jpg"), np.full((10, 10, 3), 128, dtype=np.uint8))
    (src / "s.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "yolo"
    assert main(["prepare", "--data", str(src), "--out", str(out), "--classes", "scratch"]) == 0
    assert (out / "dataset.yaml").is_file()
    assert (out / "labels" / "train" / "s.txt").is_file()
