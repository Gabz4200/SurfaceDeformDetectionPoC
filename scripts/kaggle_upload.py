"""Upload this repository to Kaggle as a dataset (code only; data travels separately).

Used both as a pre-commit hook and manually. Skips silently (exit 0) when
Kaggle credentials or kagglehub are absent, so commits never block on setup.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SLUG = "surfacedeformdetectionpoc"

# Mirrors .gitignore, plus training-scale dirs kagglehub would otherwise pack:
# data/ holds 1.7GB+ of steel images and models/ holds weights and datasets.
_IGNORE_PATTERNS = [
    "__pycache__/",
    "*.py[oc]",
    "*.egg-info/",
    ".venv/",
    ".ipynb_checkpoints/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".opencode/",
    ".aislop/",
    "outputs/",
    ".multirun/",
    "data/",
    "*.tar",
    "models/",
    "results/",
    "*.db",
    "*.pt",
    "htmlcov/",
    "coverage.xml",
]


def _resolve_username() -> str:
    """Kaggle username from env, else via locals creds; "" when unresolvable."""
    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    has_credentials = (
        (Path.home() / ".kaggle" / "kaggle.json").is_file()
        or os.environ.get("KAGGLE_KEY") is not None
        or os.environ.get("KAGGLE_API_TOKEN") is not None
    )
    if not has_credentials:
        return ""
    try:
        import kagglehub
    except ImportError:
        return ""
    return str(kagglehub.whoami()["username"])


def upload_dataset(
    slug: str,
    version_notes: str = "",
    repo_root: Path = REPO_ROOT,
    _skip: bool = False,
) -> str | None:
    """Upload repo as ``<KAGGLE_USERNAME>/<slug>``; None when skipped.

    Args:
        slug: Kaggle dataset slug (single name).
        version_notes: Notes for this dataset version.
        repo_root: Repository root uploaded.
        _skip: Return None instead of raising on missing setup.

    Raises:
        ValueError: On a bad slug, or on missing setup when not skipping.
    """
    if not slug or "/" in slug or "\\" in slug:
        raise ValueError(f"Dataset slug must be a single name, got {slug!r}.")
    username = _resolve_username()
    if not username:
        if _skip:
            logger.info("No Kaggle credentials; skipping upload.")
            return None
        raise ValueError("Set KAGGLE_USERNAME (and KAGGLE_KEY, or ~/.kaggle/kaggle.json).")
    try:
        import kagglehub
    except ImportError:
        if _skip:
            logger.info("kagglehub not installed; skipping upload.")
            return None
        raise
    handle = f"{username}/{slug}"
    kagglehub.dataset_upload(
        handle,
        str(repo_root),
        version_notes=version_notes,
        ignore_patterns=_IGNORE_PATTERNS,
    )
    return handle


def main(argv: list[str] | None = None) -> int:
    """Hook/CLI entry: upload a commit-stamped version, or skip silently."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args(argv)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        sha = "manual"
    handle = upload_dataset(args.slug, version_notes=f"auto-upload {sha}", _skip=True)
    if handle:
        logger.info("Uploaded https://www.kaggle.com/datasets/%s", handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
