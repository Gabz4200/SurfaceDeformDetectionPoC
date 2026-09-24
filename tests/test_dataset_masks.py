"""Slice 2b RED: DatasetNinja bitmap-mask -> YOLO bbox lines."""

from __future__ import annotations

import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np


def _encode_mask(mask: np.ndarray) -> str:
    from PIL import Image

    pil = Image.fromarray(mask.astype(np.uint8))
    pil.putpalette([0, 0, 0, 255, 255, 255])
    buf = io.BytesIO()
    pil.save(buf, format="PNG", transparency=0, optimize=0)
    return base64.b64encode(zlib.compress(buf.getvalue())).decode()


def _ann(mask: np.ndarray, origin: list[int], cls: str = "scratch") -> dict:
    return {
        "size": {"height": 50, "width": 100},
        "objects": [
            {
                "classTitle": cls,
                "geometryType": "bitmap",
                "bitmap": {"data": _encode_mask(mask), "origin": origin},
            }
        ],
    }


def _parse(line: str) -> tuple[int, float, float, float, float]:
    c, xc, yc, w, h = line.split()
    return int(c), float(xc), float(yc), float(w), float(h)


def test_when_real_bitmap_then_mask_inside_image() -> None:
    from surfacedeformdetectionpoc.dataset import decode_bitmap

    ann = json.loads(Path("data/severstal-DatasetNinja/train/ann/b595566b3.jpg.json").read_text())
    obj = ann["objects"][0]
    mask = decode_bitmap(obj["bitmap"]["data"])
    assert mask.ndim == 2 and mask.dtype == bool and bool(mask.any())
    ox, oy = obj["bitmap"]["origin"]
    assert ox + mask.shape[1] <= 1600
    assert oy + mask.shape[0] <= 256


def test_when_full_mask_at_origin_then_hand_computed_line() -> None:
    from surfacedeformdetectionpoc.dataset import annotation_to_yolo_lines

    lines = annotation_to_yolo_lines(
        _ann(np.ones((4, 4), bool), [10, 20]), 100, 50, ["scratch", "dent"]
    )
    assert _parse(lines[0]) == (0, 0.12, 0.44, 0.04, 0.08)


def test_when_empty_ann_then_no_lines() -> None:
    from surfacedeformdetectionpoc.dataset import annotation_to_yolo_lines

    assert annotation_to_yolo_lines({"objects": []}, 100, 50, ["scratch"]) == []


def test_when_datasetninja_layout_then_labels_generated() -> None:
    import tempfile

    import cv2

    from surfacedeformdetectionpoc.dataset import build_yolo_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        img_dir = root / "train" / "img"
        ann_dir = root / "train" / "ann"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        cv2.imwrite(str(img_dir / "s1.jpg"), np.full((50, 100, 3), 128, dtype=np.uint8))
        (ann_dir / "s1.jpg.json").write_text(json.dumps(_ann(np.ones((4, 4), bool), [10, 20])))
        out = Path(tmp) / "yolo"
        build_yolo_dataset(root, out, class_names=["scratch"], val_fraction=0.0)
        assert _parse((out / "labels" / "train" / "s1.txt").read_text().strip()) == (
            0,
            0.12,
            0.44,
            0.04,
            0.08,
        )
