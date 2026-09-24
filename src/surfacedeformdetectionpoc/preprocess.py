"""Optical preprocessing: grayscale + CLAHE glare normalization (PRD FR-2.1..FR-2.4)."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Equalize a metal-surface image and return 3-channel BGR uint8.

    Args:
        image: HxW grayscale or HxWx3 BGR uint8 array.
        clip_limit: CLAHE clipping threshold (default 2.0).
        tile_grid_size: CLAHE contextual tile grid (default 8x8).

    Returns:
        HxWx3 BGR uint8 array ready for YOLO backends.

    Raises:
        ValueError: If input shape or dtype is unsupported.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 ndarray, got {type(image)}.")
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Expected HxW or HxWx3 array, got shape {image.shape}.")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
