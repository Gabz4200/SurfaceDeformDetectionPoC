"""RED: InspectionDB context manager (M2)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


def test_when_used_as_context_manager_then_connection_closed_on_exit() -> None:
    from surfacedeformdetectionpoc.database import InspectionDB

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "inspections.db"
        with InspectionDB(db_path) as db:
            row_id = db.log("a.jpg", True, 0.5, "dent")
            assert db.get(row_id)["image_name"] == "a.jpg"
        with pytest.raises(sqlite3.ProgrammingError):
            db.log("b.jpg", False, 0.0, "")

        # Connection actually closed, not just logically unusable.
        fresh = sqlite3.connect(str(db_path))
        assert fresh.execute("SELECT COUNT(*) FROM inspection_logs").fetchone()[0] == 1
        fresh.close()


def test_when_closed_then_log_raises() -> None:
    from surfacedeformdetectionpoc.database import InspectionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = InspectionDB(Path(tmp) / "inspections.db")
        db.close()
        with pytest.raises(sqlite3.ProgrammingError):
            db.log("a.jpg", False, 0.0, "")
