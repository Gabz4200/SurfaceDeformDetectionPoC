"""Slice 3 RED: inference seam contract (PRD FR-3.1..FR-3.3)."""

from __future__ import annotations

import numpy as np
import pytest


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
        self.conf = np.array([0.9, 0.1])
        self.cls = np.array([1.0, 0.0])


class _Result:
    names = {0: "scratch", 1: "dent"}
    boxes = _Boxes()


def test_when_constructed_then_prd_default_thresholds() -> None:
    from surfacedeformdetectionpoc.inference import Detector

    d = Detector.__new__(Detector)
    assert Detector.DEFAULT_CONF == 0.25
    assert Detector.DEFAULT_IOU == 0.45
    assert d is not None


def test_when_low_conf_then_filtered_with_names_and_boxes() -> None:
    from surfacedeformdetectionpoc.inference import results_to_detections

    dets = results_to_detections([_Result()], conf=0.25)  # type: ignore[arg-type]
    assert len(dets) == 1
    assert dets[0].label == "dent"
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[0].bbox == (10.0, 20.0, 30.0, 40.0)


def test_when_missing_weights_then_raises() -> None:
    from surfacedeformdetectionpoc.inference import Detector

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError, OSError)):
        Detector("/nonexistent/weights.pt")
