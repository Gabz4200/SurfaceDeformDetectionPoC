"""Declarative training entry point: config in, weights out (Hydra shell)."""

from __future__ import annotations

import logging
from dataclasses import replace

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
        data_root=to_absolute_path(str(plan.data_root)),
        out_dir=to_absolute_path(str(plan.out_dir)),
        project=to_absolute_path(str(plan.project)),
    )
    select = (
        sample_stratified(
            plan.data_root,
            plan.subset_n,
            plan.class_names,
            plan.min_per_class,
            plan.max_bg_fraction,
            plan.seed,
        )
        if plan.subset_n is not None
        else None
    )
    yaml_path = build_yolo_dataset(
        plan.data_root,
        plan.out_dir,
        plan.class_names,
        plan.val_fraction,
        plan.seed,
        select,
    )
    best = train_model(
        yaml_path,
        plan.weights,
        plan.epochs,
        plan.imgsz,
        plan.device,
        str(plan.project),
        plan.name,
    )
    logger.info("Best weights: %s", best)


if __name__ == "__main__":
    main()
