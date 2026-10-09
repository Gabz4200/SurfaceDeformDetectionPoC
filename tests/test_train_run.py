"""RED: train_model returns the actual run dir (M1) and validates weights (m3)."""

from __future__ import annotations

from pathlib import Path

import pytest


class _FakeYOLO:
    """Fake trainer whose save_dir increments like Ultralytics does."""

    counter = 0

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.trainer = None

    def train(self, **kwargs):
        project, name = Path(kwargs["project"]), kwargs["name"]
        _FakeYOLO.counter += 1
        save_dir = project / (name if _FakeYOLO.counter == 1 else f"{name}{_FakeYOLO.counter}")
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"fake")
        self.trainer = type("T", (), {"save_dir": save_dir})()
        return None


def test_when_retrain_same_name_then_returns_incremented_run(tmp_path: Path) -> None:
    import surfacedeformdetectionpoc.train as train_mod

    train_mod.YOLO = _FakeYOLO
    weights = tmp_path / "base.pt"
    weights.write_bytes(b"fake")
    data = tmp_path / "dataset.yaml"
    data.write_text("train: images/train")
    first = train_mod.train_model(data, weights=str(weights), project=tmp_path / "runs")
    second = train_mod.train_model(data, weights=str(weights), project=tmp_path / "runs")
    assert first != second
    assert second.is_file() and second.parent.parent.name == "sdd2"
    assert "weights" in second.parts


def test_when_weights_missing_then_raises_file_not_found(tmp_path: Path) -> None:
    import surfacedeformdetectionpoc.train as train_mod

    with pytest.raises(FileNotFoundError):
        train_mod.train_model(
            tmp_path / "dataset.yaml",
            weights=str(tmp_path / "nonexistent.pt"),
            project=tmp_path / "runs",
        )
