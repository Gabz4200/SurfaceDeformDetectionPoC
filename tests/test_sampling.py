"""RED: stratified subset sampler seam."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((10, 10, 3), 128, dtype=np.uint8))


def _layout(root: Path) -> None:
    labeled = {
        "a1.jpg": "0\n",
        "a2.jpg": "0\n",
        "a3.jpg": "0\n",
        "b1.jpg": "1\n",
        "b2.jpg": "1\n",
        "ab1.jpg": "0\n1\n",
    }
    for name, txt in labeled.items():
        _img(root / "img" / name)
        (root / "ann" / f"{Path(name).stem}.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / "ann" / f"{Path(name).stem}.txt").write_text(txt)
    for name in ("bg1.jpg", "bg2.jpg", "bg3.jpg"):
        _img(root / "img" / name)


def test_when_sample_then_quotas_met_and_bg_capped(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import sample_stratified

    root = tmp_path / "data"
    _layout(root)
    got = sample_stratified(root, 6, ["aa", "bb"], min_per_class=2, max_bg_fraction=0.25, seed=0)
    names = sorted(p.name for p in got)
    assert len(got) == 6
    joined = " ".join(names)
    assert "b1.jpg" in names or "b2.jpg" in names or "ab1.jpg" in names
    assert sum(1 for n in names if n.startswith("bg")) <= 1
    assert joined.count("a") >= 1


def test_when_sample_twice_then_deterministic(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import sample_stratified

    root = tmp_path / "data"
    _layout(root)
    kwargs: dict = {"class_names": ["aa", "bb"], "min_per_class": 1, "seed": 7}
    first = sample_stratified(root, 5, **kwargs)  # type: ignore[arg-type]
    second = sample_stratified(root, 5, **kwargs)  # type: ignore[arg-type]
    assert [p.name for p in first] == [p.name for p in second]


def test_when_too_few_images_then_raises(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.dataset import sample_stratified

    root = tmp_path / "data"
    _layout(root)
    with pytest.raises(ValueError):
        sample_stratified(root, 999, ["aa", "bb"], seed=0)
