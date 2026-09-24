"""Autonomous inspection agent: fetch -> preprocess -> infer -> log (PRD FR-4.1/FR-4.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from surfacedeformdetectionpoc.database import InspectionDB
from surfacedeformdetectionpoc.dataset import discover_images
from surfacedeformdetectionpoc.inference import Detection, Detector
from surfacedeformdetectionpoc.preprocess import preprocess_image


class AutomotiveInspectionAgent:
    """Orchestrates the single-image inspection pipeline."""

    def __init__(
        self,
        detector: Detector,
        db: InspectionDB,
        results_dir: str | Path = "results",
    ) -> None:
        """Wire the pipeline collaborators.

        Args:
            detector: Configured YOLO defect detector.
            db: Inspection log database.
            results_dir: Overlay artifact destination.
        """
        self.detector = detector
        self.db = db
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def inspect_image(self, image_path: str | Path) -> dict[str, Any]:
        """Run the full pipeline on one image.

        Args:
            image_path: Source image file.

        Returns:
            Mapping with ``image_name``, ``defect_detected``, ``confidence``,
            ``class_label``, ``row_id``, and ``artifact`` (path or None).

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        src = Path(image_path)
        image = cv2.imread(str(src))
        if image is None:
            raise FileNotFoundError(f"Unreadable image: {src}.")
        dets: list[Detection] = self.detector.predict(preprocess_image(image))
        artifact: str | None = None
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            defect, confidence, label = True, best.confidence, best.label
            overlay = image.copy()
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    overlay,
                    f"{d.label} {d.confidence:.0%}",
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            dest = self.results_dir / src.name
            cv2.imwrite(str(dest), overlay)
            artifact = str(dest)
        else:
            defect, confidence, label = False, 0.0, ""
        row_id = self.db.log(src.name, defect, float(confidence), label)
        return {
            "image_name": src.name,
            "defect_detected": defect,
            "confidence": float(confidence),
            "class_label": label,
            "row_id": row_id,
            "artifact": artifact,
        }

    def inspect_directory(self, directory: str | Path) -> list[dict[str, Any]]:
        """Inspect every image under ``directory``, recursively.

        Args:
            directory: Root scanned for images.

        Returns:
            Per-image result mappings in sorted path order.
        """
        return [self.inspect_image(p) for p in discover_images(Path(directory))]
