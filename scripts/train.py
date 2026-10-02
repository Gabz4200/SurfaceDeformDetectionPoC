"""Declarative training entry point: config in, weights out (Hydra shell)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from surfacedeformdetectionpoc.dataset import build_yolo_dataset, sample_stratified
from surfacedeformdetectionpoc.train import build_run_plan, train_model

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the run plan from config and execute prepare + train."""
    plan = build_run_plan(OmegaConf.to_container(cfg, resolve=True))
    plan = replace(
        plan,
        data_root=Path(to_absolute_path(str(plan.data_root))),
        out_dir=Path(to_absolute_path(str(plan.out_dir))),
        project=Path(to_absolute_path(str(plan.project))),
    )
    select = (
        sample_stratified(
            data_root=plan.data_root,
            n=plan.subset_n,
            class_names=plan.class_names,
            min_per_class=plan.min_per_class,
            max_bg_fraction=plan.max_bg_fraction,
            seed=plan.seed,
        )
        if plan.subset_n is not None
        else None
    )
    yaml_path = build_yolo_dataset(
        data_root=plan.data_root,
        out_dir=plan.out_dir,
        class_names=plan.class_names,
        val_fraction=plan.val_fraction,
        seed=plan.seed,
        select=select,
    )
    best = train_model(
        dataset_yaml=yaml_path,
        weights=plan.weights,
        epochs=plan.epochs,
        imgsz=plan.imgsz,
        device=plan.device,
        project=str(plan.project),
        name=plan.name,
    )
    logger.info("Best weights: %s", best)


if __name__ == "__main__":
    main()
