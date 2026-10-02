"""YOLO dataset preparation: recursive discovery + label passthrough (PRD FR-1.1/FR-1.3)."""

from __future__ import annotations

import base64
import json
import random
import shutil
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(data_root: Path) -> list[Path]:
    """Recursively find supported images under ``data_root``.

    Args:
        data_root: Root directory scanned recursively.

    Returns:
        Sorted list of image paths.
    """
    return sorted(p for p in data_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def decode_bitmap(data_b64: str) -> np.ndarray:
    """Decode a Supervisely/DatasetNinja bitmap mask to a bool array.

    Args:
        data_b64: Base64 zlib-compressed PNG mask (cropped to defect).

    Returns:
        2D bool array, True on defect pixels.

    Raises:
        ValueError: If the payload cannot be decoded to a mask.
    """
    try:
        raw = zlib.decompress(base64.b64decode(data_b64))
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        raise ValueError(f"Undecodable bitmap mask: {e}") from e
    if img is None:
        raise ValueError("Undecodable bitmap mask.")
    if img.ndim == 3 and img.shape[2] >= 4:
        return img[:, :, 3].astype(bool)
    if img.ndim == 2:
        return img.astype(bool)
    raise ValueError("Undecodable bitmap mask.")


def annotation_to_yolo_lines(
    ann: dict[str, Any], img_w: int, img_h: int, class_names: list[str]
) -> list[str]:
    """Convert a DatasetNinja annotation dict to YOLO label lines.

    Args:
        ann: Parsed ``*.jpg.json`` annotation (``objects`` with bitmap geometry).
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        class_names: Ordered defect class names.

    Returns:
        YOLO ``"<idx> <xc> <yc> <w> <h>"`` lines in normalized coordinates.

    Raises:
        ValueError: On unknown geometry, unknown class, or empty mask.
    """
    lines: list[str] = []
    index = {name: i for i, name in enumerate(class_names)}
    for obj in ann.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            raise ValueError(f"Unsupported geometry: {obj.get('geometryType')}.")
        title = obj.get("classTitle")
        if title not in index:
            raise ValueError(f"Unknown class {title!r}, expected one of {class_names}.")
        mask = decode_bitmap(obj["bitmap"]["data"])
        if not mask.any():
            raise ValueError("Empty bitmap mask.")
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ys, xs = np.flatnonzero(rows), np.flatnonzero(cols)
        top, bottom, left, right = int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])
        ox, oy = obj["bitmap"]["origin"]
        x_min, y_min = ox + left, oy + top
        x_max, y_max = ox + right + 1, oy + bottom + 1
        xc = (x_min + x_max) / 2 / img_w
        yc = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        lines.append(f"{index[title]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def class_names_from_meta(meta_path: str | Path) -> list[str]:
    """Read ordered class titles from a DatasetNinja ``meta.json``.

    Args:
        meta_path: Path to ``meta.json``.

    Returns:
        Class titles in file order.
    """
    meta = json.loads(Path(meta_path).read_text())
    return [str(c["title"]) for c in meta["classes"]]


def image_classes(image: Path, class_names: list[str]) -> set[str]:
    """Name the defect classes present in one image's sidecar annotation.

    Args:
        image: Source image path.
        class_names: Ordered defect class names.

    Returns:
        Class titles found; empty when the image is background.

    Raises:
        ValueError: On unknown class index or title.
    """
    label = _sibling_label(image)
    if label is not None:
        found: set[str] = set()
        for line in label.read_text().splitlines():
            if line.strip():
                idx = int(line.split()[0])
                if idx >= len(class_names):
                    raise ValueError(f"Unknown class index {idx} in {label}.")
                found.add(class_names[idx])
        return found
    ann_path = _sibling_ann_json(image)
    if ann_path is not None:
        titles = {o.get("classTitle") for o in json.loads(ann_path.read_text()).get("objects", [])}
        unknown = titles - set(class_names)
        if unknown:
            raise ValueError(f"Unknown classes {sorted(unknown)} in {ann_path}.")
        return set(titles)
    return set()


def sample_stratified(
    data_root: str | Path,
    n: int,
    class_names: list[str],
    min_per_class: int = 15,
    max_bg_fraction: float = 0.25,
    seed: int = 42,
) -> list[Path]:
    """Pick ``n`` images with rare-class quotas and capped background share.

    Args:
        data_root: Scanned recursively for images.
        n: Subset size.
        class_names: Ordered defect class names.
        min_per_class: Best-effort images containing each class (rarest first).
        max_bg_fraction: Max share of label-free images.
        seed: Deterministic sampling seed.

    Returns:
        Sorted image paths.

    Raises:
        ValueError: If ``n`` exceeds available images or a class is absent.
    """
    images = discover_images(Path(data_root))
    if n > len(images):
        raise ValueError(f"Requested {n} images but found {len(images)}.")
    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = {c: [] for c in class_names}
    img_labels: dict[Path, set[str]] = {}
    bg: list[Path] = []
    for img in images:
        present = image_classes(img, class_names)
        img_labels[img] = present
        if not present:
            bg.append(img)
        for c in present:
            buckets[c].append(img)
    missing = [c for c in class_names if not buckets[c]]
    if missing:
        raise ValueError(f"Classes absent from data: {missing}.")
    taken: list[Path] = []
    taken_set: set[Path] = set()
    taken_counts: dict[str, int] = dict.fromkeys(class_names, 0)
    for c in sorted(class_names, key=lambda k: len(buckets[k])):
        cands = buckets[c][:]
        rng.shuffle(cands)
        need = min_per_class - taken_counts[c]
        for p in cands:
            if need <= 0:
                break
            if p in taken_set:
                continue
            taken.append(p)
            taken_set.add(p)
            for c2 in img_labels[p]:
                taken_counts[c2] += 1
            need -= 1
    bg_set = set(bg)
    labeled = [p for p in images if p not in bg_set and p not in taken_set]
    rng.shuffle(labeled)
    bg_shuffled = bg[:]
    rng.shuffle(bg_shuffled)
    chosen = (taken + labeled)[: n - min(len(bg_shuffled), int(n * max_bg_fraction))]
    chosen += bg_shuffled[: n - len(chosen)]
    return sorted(chosen)


def _first_existing(cands: list[Path]) -> Path | None:
    return next((c for c in cands if c.is_file()), None)


def _sibling_ann_json(image: Path) -> Path | None:
    names = [f"{image.name}.json", f"{image.stem}.json"]
    dirs = [image.parent / "ann", image.parent.parent / "ann", image.parent]
    return _first_existing([d / n for d in dirs for n in names])


def _sibling_label(image: Path) -> Path | None:
    return _first_existing(
        [
            image.with_suffix(".txt"),
            image.parent / "ann" / f"{image.stem}.txt",
            image.parent.parent / "ann" / f"{image.stem}.txt",
        ]
    )


def _write_label(src: Path, dest_lbl: Path, class_names: list[str]) -> None:
    label = _sibling_label(src)
    if label is not None:
        shutil.copyfile(label, dest_lbl)
        return
    ann_path = _sibling_ann_json(src)
    if ann_path is not None:
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unreadable image: {src}.")
        h, w = img.shape[:2]
        ann = json.loads(ann_path.read_text())
        dest_lbl.write_text(
            "\n".join(annotation_to_yolo_lines(ann, w, h, class_names))
            + ("\n" if ann.get("objects") else "")
        )
    else:
        dest_lbl.write_text("")


def build_yolo_dataset(
    data_root: Path,
    out_dir: Path,
    class_names: list[str],
    val_fraction: float = 0.2,
    seed: int = 42,
    select: list[Path] | None = None,
) -> Path:
    """Assemble a YOLO training tree from loose images under ``data_root``.

    Images keep their pixels untouched. Label resolution per image: YOLO-txt
    sibling passthrough, else DatasetNinja ``*.jpg.json`` bitmap conversion,
    else empty (background) file.

    Args:
        data_root: Scanned recursively for images (skipped when ``select`` given).
        out_dir: Destination ``images/{train,val}``, ``labels/{train,val}``.
        class_names: Ordered defect class names.
        val_fraction: Share held out for validation.
        seed: Deterministic split seed.
        select: Explicit image list (e.g. from :func:`sample_stratified`).

    Returns:
        Path to the written ``dataset.yaml``.
    """
    images = [Path(p) for p in select] if select is not None else discover_images(Path(data_root))
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}
    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, paths in splits.items():
        for src in paths:
            dest_img = out / "images" / split / f"{src.stem}{src.suffix.lower()}"
            shutil.copyfile(src, dest_img)
            _write_label(src, out / "labels" / split / f"{src.stem}.txt", class_names)
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": dict(enumerate(class_names)),
            }
        )
    )
    return yaml_path
