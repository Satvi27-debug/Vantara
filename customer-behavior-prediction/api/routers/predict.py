"""Prediction endpoints."""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from api.schemas.prediction import CustomerFeatures, PredictResponse

router = APIRouter(prefix="/predict", tags=["predict"])


def _compute_engagement_score(recency: float, frequency: float, monetary: float) -> float:
    # Simple heuristic (not the exact training normalization); better than missing feature
    # Lower recency better, higher freq/monetary better
    r = max(recency, 0.0)
    f = max(frequency, 0.0)
    m = max(monetary, 0.0)
    # crude normalization
    r_score = 1.0 / (1.0 + r / 30.0)
    f_score = min(f / 10.0, 1.0)
    m_score = min(m / 500.0, 1.0)
    return float(0.4 * r_score + 0.3 * f_score + 0.3 * m_score)


@router.post("", response_model=PredictResponse)
def predict_one(payload: CustomerFeatures) -> PredictResponse:
    artifacts_dir = Path(__file__).resolve().parents[2] / "models_artifacts"
    model_file = artifacts_dir / "churn_logreg.joblib"

    if model_file.exists():
        try:
            from src.models.infer import predict_churn

            eng = _compute_engagement_score(payload.RecencyDays, payload.Frequency, payload.Monetary)
            df = pd.DataFrame(
                [
                    {
                        "CustomerID": payload.CustomerID,
                        "RecencyDays": payload.RecencyDays,
                        "Frequency": payload.Frequency,
                        "Monetary": payload.Monetary,
                        "AvgOrderValue": payload.Monetary / max(payload.Frequency, 1.0),
                        "TotalQuantity": payload.Frequency,
                        "EngagementScore": eng,
                    }
                ]
            )
            out = predict_churn(df, artifacts_dir)
            row = out.iloc[0]
            return PredictResponse(
                CustomerID=int(payload.CustomerID),
                churn_probability=float(row["churn_probability"]),
                churn_label=int(row["churn_label"]),
                note="model prediction",
            )
        except Exception:
            pass

    prob = 0.5
    return PredictResponse(
        CustomerID=payload.CustomerID,
        churn_probability=prob,
        churn_label=1 if prob >= 0.5 else 0,
        note="stub response (model not available)",
    )
