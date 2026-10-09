"""RED: collision-safe dataset output (C1) and sampler/label contracts (M3, M5, n2)."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import write_image as _img


def test_when_stems_collide_across_folders_then_all_labels_preserved(
    tmp_path: Path,
) -> None:
    from surfacedeformdetectionpoc.dataset import build_yolo_dataset

    root = tmp_path / "data"
    _img(root / "a" / "s1.jpg")
    _img(root / "b" / "s1.jpg")
    (root / "a" / "s1.txt").write_text("0 0.1 0.1 0.1 0.1\n")
    (root / "b" / "s1.txt").write_text("1 0.9 0.9 0.2 0.2\n")
    out = tmp_path / "yolo"
    build_yolo_dataset(root, out, class_names=["scratch", "dent"], val_fraction=0.0)
    labels = sorted(p.read_text() for p in (out / "labels" / "train").glob("*.txt"))
    images = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))
    assert len(labels) == 2 and labels[0] != labels[1]
    assert len(images) == 2 and len(set(images)) == 2
    assert all(p.suffix == ".jpg" for p in (out / "images" / "train").glob("*"))


def test_when_n_filled_by_quotas_then_n_returned_and_no_truncation(
    tmp_path: Path,
) -> None:
    from surfacedeformdetectionpoc.dataset import sample_stratified

    root = tmp_path / "data"
    labeled = {
        "a1.jpg": "0\n",
        "a2.jpg": "0\n",
        "a3.jpg": "0\n",
        "b1.jpg": "1\n",
        "b2.jpg": "1\n",
        "b3.jpg": "1\n",
    }
    for name, txt in labeled.items():
        _img(root / name)
        (root / f"{Path(name).stem}.txt").write_text(txt)
    for name in ("bg1.jpg", "bg2.jpg", "bg3.jpg"):
        _img(root / name)
    got = sample_stratified(root, 6, ["aa", "bb"], min_per_class=3, max_bg_fraction=0.25, seed=0)
    assert len(got) == 6


def test_when_label_line_malformed_then_value_error_names_file(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import image_classes

    root = tmp_path / "data"
    _img(root / "bad.jpg")
    (root / "bad.txt").write_text("not a label line\n")
    with pytest.raises(ValueError, match="bad.txt"):
        image_classes(root / "bad.jpg", ["aa", "bb"])


def test_when_label_index_negative_then_value_error(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import image_classes

    root = tmp_path / "data"
    _img(root / "neg.jpg")
    (root / "neg.txt").write_text("-1 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="neg.txt"):
        image_classes(root / "neg.jpg", ["aa", "bb"])


def test_when_ann_object_missing_title_then_ignored(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import image_classes

    root = tmp_path / "data"
    _img(root / "notitle.jpg")
    ann = {"objects": [{"geometryType": "rectangle"}, {"classTitle": "aa"}]}
    (root / "notitle.jpg.json").write_text(json.dumps(ann))
    assert image_classes(root / "notitle.jpg", ["aa"]) == {"aa"}


def test_when_bitmap_garbage_then_value_error() -> None:
    from surfacedeformdetectionpoc.dataset import decode_bitmap

    with pytest.raises(ValueError, match="Undecodable bitmap mask"):
        decode_bitmap("!!!not-base64!!!")


def test_when_bitmap_zlib_garbage_then_value_error() -> None:
    import binascii

    from surfacedeformdetectionpoc.dataset import decode_bitmap

    payload = base64.b64encode(b"definitely-not-zlib").decode()
    with pytest.raises(ValueError, match="Undecodable bitmap mask"):
        decode_bitmap(payload)
    assert binascii.Error is not None


def test_when_bitmap_valid_then_decodes() -> None:
    import io

    from PIL import Image

    from surfacedeformdetectionpoc.dataset import decode_bitmap

    pil = Image.fromarray(np.ones((4, 4), dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    payload = base64.b64encode(zlib.compress(buf.getvalue())).decode()
    assert decode_bitmap(payload).any()
