"""Simple SQLite persistence helper (local alternative noted in PRD)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def get_conn(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    return conn


def init_predictions_table(db_path: str | Path) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                recency_days REAL,
                frequency REAL,
                monetary REAL,
                churn_probability REAL,
                churn_label INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
