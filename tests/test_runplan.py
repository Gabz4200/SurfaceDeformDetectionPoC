"""RED: Hydra run plan + yolo26n default."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _cfg(tmp: Path, **over: object) -> dict:
    base: dict = {
        "data": {
            "root": str(tmp / "data"),
            "out": str(tmp / "yolo"),
            "classes": ["aa", "bb"],
            "meta": None,
            "val_fraction": 0.2,
            "seed": 42,
            "subset_n": 4,
            "min_per_class": 1,
            "max_bg_fraction": 0.5,
        },
        "model": {"weights": "yolo26n.pt"},
        "device": {"device": "cpu"},
        "training": {
            "epochs": 5,
            "imgsz": 640,
            "project": str(tmp / "runs"),
            "name": "t",
        },
    }
    base.update(over)
    return base


def test_when_plan_then_typed_fields(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.train import build_run_plan

    plan = build_run_plan(_cfg(tmp_path))
    assert plan.weights == "yolo26n.pt"
    assert plan.device == "cpu"
    assert plan.epochs == 5
    assert plan.subset_n == 4
    assert plan.class_names == ["aa", "bb"]


def test_when_meta_set_then_names_from_meta(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.train import build_run_plan

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"classes": [{"title": "bb"}, {"title": "aa"}]}))
    cfg = _cfg(tmp_path)
    cfg["data"]["meta"] = str(meta)
    cfg["data"]["classes"] = None
    assert build_run_plan(cfg).class_names == ["bb", "aa"]


def test_when_key_missing_then_raises(tmp_path: Path) -> None:
    from surfacedeformdetectionpoc.train import build_run_plan

    cfg = _cfg(tmp_path)
    del cfg["training"]
    with pytest.raises(KeyError):
        build_run_plan(cfg)


def test_when_compose_defaults_then_yolo26n_on_cpu() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path("configs").resolve()),
    ):
        cfg = compose(config_name="config")
        assert cfg.model.weights == "yolo26n.pt"
        assert cfg.device.device == "cpu"
        cuda = compose(config_name="config", overrides=["device=cuda"])
        assert cuda.device.device == "0"


def test_when_train_default_then_yolo26n() -> None:
    import inspect

    from surfacedeformdetectionpoc.train import train_model

    assert inspect.signature(train_model).parameters["weights"].default == "yolo26n.pt"
