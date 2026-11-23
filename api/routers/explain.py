"""Explainability endpoints (baseline model, SHAP if available)."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplainRequest(BaseModel):
    RecencyDays: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float | None = None
    TotalQuantity: float | None = None
    EngagementScore: float | None = None


def _engagement(recency: float, frequency: float, monetary: float) -> float:
    r = max(recency, 0.0)
    f = max(frequency, 0.0)
    m = max(monetary, 0.0)
    r_score = 1.0 / (1.0 + r / 30.0)
    f_score = min(f / 10.0, 1.0)
    m_score = min(m / 500.0, 1.0)
    return float(0.4 * r_score + 0.3 * f_score + 0.3 * m_score)


@router.post("/shap")
def explain_shap(payload: ExplainRequest) -> Dict[str, Any]:
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    model_path = artifacts_dir / "churn_logreg.joblib"
    scaler_path = artifacts_dir / "churn_scaler.joblib"
    feats_path = artifacts_dir / "churn_features.joblib"
    if not (model_path.exists() and scaler_path.exists() and feats_path.exists()):
        raise HTTPException(status_code=404, detail="Baseline model artifacts not found. Train first.")

    try:
        import shap  # type: ignore
    except Exception:
        raise HTTPException(status_code=503, detail="shap is not installed on the server.")

    features: List[str] = joblib.load(feats_path)
    row = {
        "RecencyDays": payload.RecencyDays,
        "Frequency": payload.Frequency,
        "Monetary": payload.Monetary,
        "AvgOrderValue": payload.AvgOrderValue if payload.AvgOrderValue is not None else (payload.Monetary / max(payload.Frequency, 1.0)),
        "TotalQuantity": payload.TotalQuantity if payload.TotalQuantity is not None else payload.Frequency,
        "EngagementScore": payload.EngagementScore if payload.EngagementScore is not None else _engagement(payload.RecencyDays, payload.Frequency, payload.Monetary),
    }
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    Xs = scaler.transform(X)

    try:
        explainer = shap.LinearExplainer(model, Xs, feature_perturbation="interventional")
        shap_values = explainer(Xs)
        vals = np.array(shap_values.values).ravel()
    except Exception:
        explainer = shap.Explainer(model, scaler.transform)
        shap_values = explainer(X)
        vals = np.array(shap_values.values).ravel()

    contribs = sorted(
        [{"feature": f, "shap_value": float(v)} for f, v in zip(features, vals)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )
    return {"features": row, "top_contributions": contribs[:10]}
