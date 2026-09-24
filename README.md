# SurfaceDeformDetectionPoC

Automated optical inspection of surface deformations (scratches, dents, porosity, cracks) on automotive sheet metal. Pipeline: OpenCV CLAHE preprocessing → YOLO detection → SQLite logging → overlay artifacts.

## Setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra cpu        # local CPU box
uv sync --extra dev --extra cuda       # GPU machine (Kaggle: use cuda)
```

> [!NOTE]
> Unset `OMP_NUM_THREADS` / `MKL_NUM_THREADS` caps so torch sees all cores.

## Data

Drop images anywhere under `data/` — discovery is recursive, so new subfolders are picked up with no code changes. Supported:

- Images: `.jpg` / `.png` / `.bmp`
- Labels next to images: YOLO `.txt` (CVAT export), or DatasetNinja `ann/<img>.json` bitmap masks (converted to boxes automatically). Label-free images train as background.

## Usage

```bash
# 1. Build YOLO dataset (default: stratified 200-image PoC subset)
uv run python scripts/train.py
# Full dataset, GPU, larger model — config only, no code changes
uv run python scripts/train.py device=cuda data.subset_n=null model=yolo26s

# 2. Inspect images with trained weights
uv run surfacedeformdetectionpoc inspect --path data --weights models/runs/sdd/weights/best.pt
```

Artifacts: dataset in `models/yolo_dataset/`, weights in `models/runs/`, overlays in `results/`, logs in `inspections.db`.

## Config

Hydra groups in `configs/`: `data`, `model` (`yolo26n` default, `yolo26s` available), `device` (`cpu`/`cuda`), `training`. Override any key on the command line (see above).

## Checks

```bash
uv run ruff check src tests scripts
uv run pyrefly check
uv run pytest
uv run aislop scan .
```

Pre-commit runs all of the above on every commit (`pre-commit install` — already installed). Set `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) and each commit also uploads the repo as a Kaggle dataset for GPU training; without credentials that step skips silently. `data/` (gigabytes of steel) and `models/` (weights) are excluded — upload data to Kaggle once, separately.
