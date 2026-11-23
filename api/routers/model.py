"""Model metadata endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata")
def model_metadata():
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    model_file = artifacts_dir / "churn_logreg.joblib"
    metrics_file = artifacts_dir / "metrics.json"
    metrics = None
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    if model_file.exists():
        return {
            "model_name": "churn_logreg_rfm_baseline",
            "version": "0.1.0",
            "status": "trained",
            "artifacts_dir": str(artifacts_dir),
            "metrics": metrics,
        }
    return {
        "model_name": "stub_churn_model",
        "version": "0.0.1",
        "status": "not_trained",
        "metrics": {},
    }
