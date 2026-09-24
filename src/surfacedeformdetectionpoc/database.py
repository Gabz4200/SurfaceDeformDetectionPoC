"""Inspection persistence: SQLite logging (PRD FR-4.2/FR-4.3)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class InspectionDB:
    """SQLite inspection log; creates ``inspection_logs`` on instantiation."""

    def __init__(self, path: str | Path = "inspections.db") -> None:
        """Open (creating parent dirs) and initialize the schema.

        Args:
            path: Database file path.
        """
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS inspection_logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT NOT NULL, "
            "image_name TEXT NOT NULL, "
            "defect_detected INTEGER NOT NULL, "
            "confidence REAL NOT NULL, "
            "class_label TEXT NOT NULL)"
        )
        self._conn.commit()

    def log(
        self, image_name: str, defect_detected: bool, confidence: float, class_label: str
    ) -> int:
        """Persist one inspection and return its row id.

        Args:
            image_name: Source filename.
            defect_detected: True if at least one box was detected.
            confidence: Highest detection confidence (0.00 to 1.00).
            class_label: Primary defect class ("" when clean).

        Returns:
            Auto-incremented row id.
        """
        cur = self._conn.execute(
            "INSERT INTO inspection_logs "
            "(timestamp, image_name, defect_detected, confidence, class_label) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_name,
                1 if defect_detected else 0,
                confidence,
                class_label,
            ),
        )
        self._conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("Insert returned no row id.")
        return cur.lastrowid

    def get(self, row_id: int) -> dict[str, Any]:
        """Fetch one log entry by id.

        Args:
            row_id: Value returned by :meth:`log`.

        Returns:
            Column-name to value mapping.

        Raises:
            KeyError: If no row has that id.
        """
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute("SELECT * FROM inspection_logs WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise KeyError(f"No inspection with id {row_id}.")
        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
