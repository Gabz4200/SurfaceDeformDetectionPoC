"""Slice 4 RED: persistence seam contract (PRD FR-4.2/FR-4.3)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def test_when_init_then_schema_exists() -> None:
    import sqlite3
    import tempfile

    from surfacedeformdetectionpoc.database import InspectionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = InspectionDB(Path(tmp) / "inspections.db")
        cols = {
            r[1]: r[2]
            for r in sqlite3.connect(str(db.path)).execute("PRAGMA table_info(inspection_logs)")
        }
        assert cols == {
            "id": "INTEGER",
            "timestamp": "TEXT",
            "image_name": "TEXT",
            "defect_detected": "INTEGER",
            "confidence": "REAL",
            "class_label": "TEXT",
        }
        db.close()


def test_when_log_hit_then_row_matches_prd_schema() -> None:
    import tempfile

    from surfacedeformdetectionpoc.database import InspectionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = InspectionDB(Path(tmp) / "inspections.db")
        row_id = db.log("strip_001.jpg", True, 0.87, "dent")
        row = db.get(row_id)
        assert row["image_name"] == "strip_001.jpg"
        assert row["defect_detected"] == 1
        assert row["confidence"] == 0.87
        assert row["class_label"] == "dent"
        datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        db.close()


def test_when_log_clean_then_zero_flag_and_ids_increment() -> None:
    import tempfile

    from surfacedeformdetectionpoc.database import InspectionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = InspectionDB(Path(tmp) / "inspections.db")
        first = db.log("a.jpg", False, 0.0, "")
        second = db.log("b.jpg", True, 0.5, "scratch")
        assert second == first + 1
        assert db.get(first)["defect_detected"] == 0
        db.close()
