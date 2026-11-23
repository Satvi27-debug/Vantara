"""Metrics endpoints (read from artifacts)."""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/comparison")
def comparison():
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    path = artifacts_dir / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_comparison.csv not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@router.get("/clv")
def clv_metrics():
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    path = artifacts_dir / "clv_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="clv_metrics.json not found")
    return json.loads(path.read_text(encoding="utf-8"))
