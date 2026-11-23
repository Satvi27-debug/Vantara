"""Batch scoring endpoints (stub + basic real if model exists)."""

from pathlib import Path
import io

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from src.utils.db import get_conn, init_predictions_table

router = APIRouter(prefix="/batch", tags=["batch"])


def _compute_engagement_score_df(df: pd.DataFrame) -> pd.Series:
    r = df["RecencyDays"].astype(float).clip(lower=0)
    f = df["Frequency"].astype(float).clip(lower=0)
    m = df["Monetary"].astype(float).clip(lower=0)
    r_score = 1.0 / (1.0 + r / 30.0)
    f_score = (f / 10.0).clip(upper=1.0)
    m_score = (m / 500.0).clip(upper=1.0)
    return 0.4 * r_score + 0.3 * f_score + 0.3 * m_score


def _run_inference_if_available(df: pd.DataFrame):
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    model_file = artifacts_dir / "churn_logreg.joblib"
    if not model_file.exists():
        return None, "Model not trained. Train first via scripts/train_baseline_churn.py"

    required = ["RecencyDays", "Frequency", "Monetary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
    if "AvgOrderValue" not in df.columns:
        df["AvgOrderValue"] = df["Monetary"] / df["Frequency"].replace(0, 1)
    if "TotalQuantity" not in df.columns:
        df["TotalQuantity"] = df["Frequency"]
    if "EngagementScore" not in df.columns:
        df["EngagementScore"] = _compute_engagement_score_df(df)

    from src.models.infer import predict_churn

    out = predict_churn(df, artifacts_dir)
    return out, None


def _log_predictions_to_db(df_out: pd.DataFrame) -> None:
    try:
        db_path = Path(__file__).resolve().parents[2] / "data" / "predictions.db"
        init_predictions_table(db_path)
        conn = get_conn(db_path)
        try:
            for _, r in df_out.iterrows():
                cid = None
                if "Customer ID" in df_out.columns and pd.notna(r.get("Customer ID")):
                    try:
                        cid = int(r.get("Customer ID"))
                    except Exception:
                        cid = None
                conn.execute(
                    "INSERT INTO predictions (customer_id, recency_days, frequency, monetary, churn_probability, churn_label) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        cid,
                        float(r.get("RecencyDays", 0)),
                        float(r.get("Frequency", 0)),
                        float(r.get("Monetary", 0)),
                        float(r.get("churn_probability", 0.5)),
                        int(r.get("churn_label", 0)),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # best effort logging; do not fail the request
        pass


@router.post("/score")
async def batch_score(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    out, err = _run_inference_if_available(df.copy())
    if err:
        return {"rows": len(df), "predictions": [], "note": err}
    _log_predictions_to_db(out)
    return {"rows": len(out), "predictions": out.head(50).to_dict(orient="records"), "note": "model predictions (first 50 shown; logged to DB)"}


@router.post("/score.csv")
async def batch_score_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    out, err = _run_inference_if_available(df.copy())
    if err:
        raise HTTPException(status_code=400, detail=err)
    _log_predictions_to_db(out)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="scored_output.csv"'},
    )
