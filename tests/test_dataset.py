"""Slice 2a RED: dataset discovery + YOLO-txt passthrough + dataset.yaml."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _img(path: Path, w: int = 100, h: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, dtype=np.uint8))


def test_when_nested_subfolders_then_all_images_found() -> None:
    import tempfile

    from surfacedeformdetectionpoc.dataset import discover_images

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _img(root / "a.jpg")
        _img(root / "sub" / "deep" / "b.png")
        (root / "notes.txt").write_text("not an image")
        found = discover_images(root)
        assert sorted(p.name for p in found) == ["a.jpg", "b.png"]


def test_when_yolo_txt_present_then_labels_copied_and_yaml_valid() -> None:
    import tempfile

    import yaml

    from surfacedeformdetectionpoc.dataset import build_yolo_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _img(root / "train" / "img" / "s1.jpg")
        lbl = root / "train" / "ann" / "s1.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.2 0.1\n")
        _img(root / "extra" / "bg.bmp")
        out = Path(tmp) / "yolo"
        yaml_path = build_yolo_dataset(root, out, class_names=["scratch", "dent"], val_fraction=0.0)
        cfg = yaml.safe_load(yaml_path.read_text())
        assert cfg["names"] == {0: "scratch", 1: "dent"}
        assert (out / "labels" / "train" / "s1.txt").read_text() == "0 0.5 0.5 0.2 0.1\n"
        assert (out / "labels" / "train" / "bg.txt").read_text() == ""
        assert (out / "images" / "train" / "s1.jpg").exists()
