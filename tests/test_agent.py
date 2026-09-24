"""Slice 5 RED: agent orchestration seam (PRD FR-4.1/FR-4.4)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


class _HitDetector:
    def predict(self, image: np.ndarray):  # type: ignore[no-untyped-def]
        from surfacedeformdetectionpoc.inference import Detection

        return [Detection(label="dent", class_id=1, confidence=0.9, bbox=(5, 5, 20, 20))]


class _CleanDetector:
    def predict(self, image: np.ndarray):  # type: ignore[no-untyped-def]
        return []


def _agent(detector, tmp: str):  # type: ignore[no-untyped-def]
    from surfacedeformdetectionpoc.agent import AutomotiveInspectionAgent
    from surfacedeformdetectionpoc.database import InspectionDB

    return AutomotiveInspectionAgent(
        detector=detector,
        db=InspectionDB(Path(tmp) / "inspections.db"),
        results_dir=Path(tmp) / "results",
    )


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 40, 3), 128, dtype=np.uint8))


def test_when_hit_then_logged_and_artifact_saved() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "strip.jpg"
        _image(img)
        agent = _agent(_HitDetector(), tmp)
        out = agent.inspect_image(img)
        assert out["defect_detected"] is True
        assert out["confidence"] == 0.9
        assert out["class_label"] == "dent"
        assert (Path(tmp) / "results" / "strip.jpg").is_file()
        row = agent.db.get(out["row_id"])
        assert row["image_name"] == "strip.jpg" and row["defect_detected"] == 1


def test_when_clean_then_logged_without_artifact() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "clean.jpg"
        _image(img)
        agent = _agent(_CleanDetector(), tmp)
        out = agent.inspect_image(img)
        assert out["defect_detected"] is False
        assert out["confidence"] == 0.0
        assert out["class_label"] == ""
        assert not (Path(tmp) / "results" / "clean.jpg").exists()
        assert agent.db.get(out["row_id"])["defect_detected"] == 0


def test_when_missing_file_then_raises() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent = _agent(_CleanDetector(), tmp)
        with pytest.raises((FileNotFoundError, ValueError)):
            agent.inspect_image(Path(tmp) / "ghost.jpg")


def test_when_directory_then_each_image_inspected() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "incoming"
        _image(src / "a.jpg")
        _image(src / "sub" / "b.png")
        (src / "notes.txt").write_text("skip me")
        agent = _agent(_CleanDetector(), tmp)
        outs = agent.inspect_directory(src)
        assert sorted(o["image_name"] for o in outs) == ["a.jpg", "b.png"]
