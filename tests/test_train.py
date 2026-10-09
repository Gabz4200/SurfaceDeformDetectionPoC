"""Slice 6 RED: YOLO training seam."""

from __future__ import annotations

from pathlib import Path


class _FakeYOLO:
    seen: dict = {}

    def __init__(self, weights: str) -> None:
        type(self).seen["weights"] = weights
        self.trainer = None

    def train(self, **kwargs):
        type(self).seen.update(kwargs)
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        best = save_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"fake-weights")
        self.trainer = type("T", (), {"save_dir": save_dir})()
        return None


def test_when_train_then_kwargs_forwarded_and_best_returned(monkeypatch, tmp_path: Path) -> None:
    import surfacedeformdetectionpoc.train as train_mod

    monkeypatch.setattr(train_mod, "YOLO", _FakeYOLO)
    weights = tmp_path / "base.pt"
    weights.write_bytes(b"fake")
    data = tmp_path / "dataset.yaml"
    data.write_text("train: images/train")
    best = train_mod.train_model(
        data, weights=str(weights), epochs=3, imgsz=640, project=tmp_path / "runs"
    )
    assert best.is_file()
    assert _FakeYOLO.seen["data"] == str(data)
    assert _FakeYOLO.seen["epochs"] == 3
    assert _FakeYOLO.seen["imgsz"] == 640
