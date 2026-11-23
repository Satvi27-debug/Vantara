"""Predictions log endpoints (SQLite)."""

from pathlib import Path

from fastapi import APIRouter

from src.utils.db import get_conn, init_predictions_table

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/recent")
def recent_predictions(limit: int = 50):
    db_path = Path(__file__).resolve().parents[2] / "data" / "predictions.db"
    init_predictions_table(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT id, customer_id, recency_days, frequency, monetary, churn_probability, churn_label, created_at FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        cols = ["id", "customer_id", "recency_days", "frequency", "monetary", "churn_probability", "churn_label", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()
