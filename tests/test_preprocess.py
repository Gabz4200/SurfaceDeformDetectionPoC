"""Slice 1 RED: preprocess seam contract (PRD FR-2.1..FR-2.4)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def _gradient(h: int = 64, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(row, (h, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_when_color_input_then_bgr_uint8_out() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    out = preprocess_image(_gradient())
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_when_gray_input_then_bgr_out() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    gray = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    out = preprocess_image(gray)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_when_flat_input_then_flat_out() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = preprocess_image(flat)
    assert out.shape == (32, 32, 3)
    assert bool((out[:, :, 0] == out[0, 0, 0]).all())


def test_when_gradient_input_then_contrast_enhanced() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    img = _gradient()
    out = preprocess_image(img)
    plain = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert float(np.mean(cv2.absdiff(out, plain))) > 0.5


def test_when_clip_limit_changes_then_output_changes() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    img = _gradient()
    default = preprocess_image(img)
    strong = preprocess_image(img, clip_limit=8.0)
    assert bool((default != strong).any())


def test_when_bad_input_then_value_error() -> None:
    from surfacedeformdetectionpoc.preprocess import preprocess_image

    with pytest.raises(ValueError):
        preprocess_image(np.zeros((8, 8, 2), dtype=np.uint8))
