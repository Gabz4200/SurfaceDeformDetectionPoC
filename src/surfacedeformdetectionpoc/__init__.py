"""SurfaceDeformDetectionPoC: steel surface deformation inspection pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _prepare(args: argparse.Namespace) -> int:
    from surfacedeformdetectionpoc.dataset import build_yolo_dataset, class_names_from_meta

    names = class_names_from_meta(args.meta) if args.meta else args.classes
    yaml_path = build_yolo_dataset(
        Path(args.data), Path(args.out), names, args.val_fraction, args.seed
    )
    logger.info("Dataset ready: %s", yaml_path)
    return 0


def _train(args: argparse.Namespace) -> int:
    from surfacedeformdetectionpoc.train import train_model

    best = train_model(
        args.data_yaml,
        args.weights,
        args.epochs,
        args.imgsz,
        args.device,
        args.project,
        args.name,
    )
    logger.info("Best weights: %s", best)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from surfacedeformdetectionpoc.agent import AutomotiveInspectionAgent
    from surfacedeformdetectionpoc.database import InspectionDB
    from surfacedeformdetectionpoc.inference import Detector

    agent = AutomotiveInspectionAgent(
        Detector(args.weights, args.conf, args.iou, args.device),
        InspectionDB(args.db),
        args.results_dir,
    )
    outs = (
        agent.inspect_directory(args.path)
        if Path(args.path).is_dir()
        else [agent.inspect_image(args.path)]
    )
    hits = sum(1 for o in outs if o["defect_detected"])
    logger.info("Inspected %d images, %d with defects.", len(outs), hits)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``prepare`` data, ``train`` weights, or ``inspect`` images."""
    parser = argparse.ArgumentParser(prog="surfacedeformdetectionpoc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Build a YOLO dataset from ./data.")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="models/yolo_dataset")
    p.add_argument("--classes", nargs="*", default=["defect_1", "defect_2", "defect_3", "defect_4"])
    p.add_argument("--meta", default=None, help="DatasetNinja meta.json for class names.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_prepare)

    t = sub.add_parser("train", help="Fine-tune YOLO on a prepared dataset.")
    t.add_argument("--data-yaml", required=True)
    t.add_argument("--weights", default="yolo26n.pt")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--device", default="cpu")
    t.add_argument("--project", default="models/runs")
    t.add_argument("--name", default="sdd")
    t.set_defaults(func=_train)

    i = sub.add_parser("inspect", help="Inspect an image or directory.")
    i.add_argument("--path", required=True)
    i.add_argument("--weights", required=True)
    i.add_argument("--db", default="inspections.db")
    i.add_argument("--results-dir", default="results")
    i.add_argument("--conf", type=float, default=0.25)
    i.add_argument("--iou", type=float, default=0.45)
    i.add_argument("--device", default="cpu")
    i.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))
