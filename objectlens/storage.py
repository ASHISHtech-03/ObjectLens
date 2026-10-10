"""Persistence layer - SQLite repository for analysis history."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .exceptions import StorageError
from .pipeline import AnalysisResult

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    image_name      TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    object_count    INTEGER NOT NULL,
    pixels_per_mm   REAL
);
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    obj_index       INTEGER NOT NULL,
    label           TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    color           TEXT    NOT NULL,
    area_px         REAL    NOT NULL,
    perimeter_px    REAL    NOT NULL,
    centroid_x      REAL    NOT NULL,
    centroid_y      REAL    NOT NULL,
    length_mm       REAL,
    width_mm        REAL,
    area_mm2        REAL,
    is_reference    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id);
"""


class Repository:
    """Small data-access object. Uses parameterised queries only (no SQL injection)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as conn, conn:
                conn.executescript(SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Cannot initialise database '{db_path}': {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_analysis(self, result: AnalysisResult, model_name: str) -> int:
        """Store one analysis (all-or-nothing transaction); returns the run id."""
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO runs (created_at, image_name, model, object_count, pixels_per_mm)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     result.image_name, model_name, result.count, result.pixels_per_mm))
                run_id = cur.lastrowid
                rows = []
                for o in result.objects:
                    m, f = o.measurement, o.features
                    rows.append((run_id, o.id, o.label, o.confidence, f.color_name, f.area,
                                 f.perimeter, f.centroid[0], f.centroid[1],
                                 m.length_mm if m else None, m.width_mm if m else None,
                                 m.area_mm2 if m else None, int(o.is_reference)))
                conn.executemany(
                    "INSERT INTO detections (run_id, obj_index, label, confidence, color,"
                    " area_px, perimeter_px, centroid_x, centroid_y, length_mm, width_mm,"
                    " area_mm2, is_reference) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            log.info("Saved run %d (%d detections)", run_id, len(rows))
            return int(run_id)
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save analysis: {exc}") from exc

    def list_runs(self, limit: int = 20) -> List[Dict]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                                    (int(limit),)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def get_detections(self, run_id: int) -> List[Dict]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT * FROM detections WHERE run_id = ? ORDER BY obj_index",
                                    (int(run_id),)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def delete_run(self, run_id: int) -> bool:
        try:
            with closing(self._connect()) as conn, conn:
                cur = conn.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc
