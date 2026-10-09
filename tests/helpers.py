"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, w: int = 100, h: int = 50) -> None:
    """Write a solid-gray image at ``path``, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, dtype=np.uint8))
