"""YOLO training entry point (PRD FR-3.1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from surfacedeformdetectionpoc.dataset import class_names_from_meta


@dataclass(frozen=True)
class RunPlan:
    """Typed training run resolved from config (never a DictConfig downstream)."""

    data_root: Path
    out_dir: Path
    class_names: list[str]
    val_fraction: float
    seed: int
    subset_n: int | None
    min_per_class: int
    max_bg_fraction: float
    weights: str
    epochs: int
    imgsz: int
    device: str
    project: Path
    name: str


def resolve_class_names(meta: str | Path | None, classes: list[str] | None) -> list[str]:
    """Resolve ordered class names from a DatasetNinja ``meta.json`` or a list.

    Args:
        meta: Path to ``meta.json``; takes precedence when both are given.
        classes: Fallback class names.

    Returns:
        Class names in file or list order.

    Raises:
        ValueError: When neither source yields a name.
    """
    names = class_names_from_meta(meta) if meta else [str(c) for c in (classes or [])]
    if not names:
        raise ValueError("No class names: set data.classes or data.meta.")
    return names


def build_run_plan(cfg: Mapping[str, Any]) -> RunPlan:
    """Map a nested config mapping to a validated :class:`RunPlan`.

    Args:
        cfg: ``data`` / ``model`` / ``device`` / ``training`` sections.

    Returns:
        Fully typed run plan.

    Raises:
        KeyError: On missing sections or keys.
        ValueError: When no class names resolve.
    """
    data, model, device, training = cfg["data"], cfg["model"], cfg["device"], cfg["training"]
    names = resolve_class_names(data["meta"], data["classes"])
    return RunPlan(
        data_root=Path(data["root"]),
        out_dir=Path(data["out"]),
        class_names=names,
        val_fraction=float(data["val_fraction"]),
        seed=int(data["seed"]),
        subset_n=int(data["subset_n"]) if data["subset_n"] is not None else None,
        min_per_class=int(data["min_per_class"]),
        max_bg_fraction=float(data["max_bg_fraction"]),
        weights=str(model["weights"]),
        epochs=int(training["epochs"]),
        imgsz=int(training["imgsz"]),
        device=str(device["device"]),
        project=Path(training["project"]),
        name=str(training["name"]),
    )


def train_model(
    dataset_yaml: str | Path,
    weights: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
    project: str | Path = "models/runs",
    name: str = "sdd",
) -> Path:
    """Fine-tune a YOLO checkpoint on a prepared ``dataset.yaml``.

    Args:
        dataset_yaml: Dataset descriptor from :func:`build_yolo_dataset`.
        weights: Base checkpoint (``.pt`` path or Ultralytics model name).
        epochs: Training epochs.
        imgsz: Training image size.
        device: Training device.
        project: Runs directory.
        name: Run name under ``project``.

    Returns:
        Path to the resulting ``best.pt`` weights.

    Raises:
        FileNotFoundError: If ``best.pt`` is missing after training.
    """
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Weights not found: {weights}.")
    model = YOLO(str(weights))
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        verbose=False,
    )
    # Ultralytics increments the run directory (sdd, sdd2, ...) when it
    # already exists, so return the trainer's resolved save_dir rather than
    # the requested name, which would point at a stale first run.
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Trainer did not expose a save_dir after training.")
    best = Path(save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training produced no weights at {best}.")
    return best
