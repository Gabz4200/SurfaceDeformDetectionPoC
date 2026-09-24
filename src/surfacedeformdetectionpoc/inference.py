"""Deep learning inference: YOLO detection with PRD thresholds (FR-3.1..FR-3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """One defect detection in absolute pixel coordinates."""

    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]


def results_to_detections(results: Any, conf: float) -> list[Detection]:
    """Flatten Ultralytics results to typed detections, dropping below ``conf``.

    Args:
        results: Ultralytics ``Results`` list (or duck-typed equivalent).
        conf: Minimum confidence kept.

    Returns:
        Detections in input order.
    """
    dets: list[Detection] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        names: dict[int, str] = getattr(r, "names", {})
        xyxy = np.asarray(boxes.xyxy).tolist()
        confs = np.asarray(boxes.conf).tolist()
        clses = np.asarray(boxes.cls).tolist()
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clses, strict=True):
            if float(c) < conf:
                continue
            idx = int(k)
            dets.append(
                Detection(
                    label=names.get(idx, str(idx)),
                    class_id=idx,
                    confidence=float(c),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
    return dets


class Detector:
    """YOLO defect detector with configurable confidence and NMS IoU."""

    DEFAULT_CONF = 0.25
    DEFAULT_IOU = 0.45

    def __init__(
        self,
        weights: str | Path,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        device: str = "cpu",
    ) -> None:
        """Load detection weights.

        Args:
            weights: Path to a YOLO ``.pt`` checkpoint.
            conf: Minimum detection confidence (default 0.25).
            iou: NMS IoU threshold (default 0.45).
            device: Inference device.

        Raises:
            FileNotFoundError: If ``weights`` does not exist.
        """
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Weights not found: {weights}.")
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(weights))

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a BGR image.

        Args:
            image: HxWx3 BGR uint8 array.

        Returns:
            Detections at or above the configured confidence.
        """
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )
        return results_to_detections(results, self.conf)
