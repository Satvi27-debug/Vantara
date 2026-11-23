"""Inference helpers for baseline churn model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd


def load_baseline_artifacts(artifacts_dir: str | Path) -> Tuple[object, object, List[str]]:
    artifacts_dir = Path(artifacts_dir)
    model = joblib.load(artifacts_dir / "churn_logreg.joblib")
    scaler = joblib.load(artifacts_dir / "churn_scaler.joblib")
    features = joblib.load(artifacts_dir / "churn_features.joblib")
    return model, scaler, features


def predict_churn(df_features: pd.DataFrame, artifacts_dir: str | Path) -> pd.DataFrame:
    model, scaler, features = load_baseline_artifacts(artifacts_dir)
    X = df_features[features].fillna(0.0)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1] if hasattr(model, "predict_proba") else None
    preds = model.predict(Xs)
    out = df_features.copy()
    if probs is not None:
        out["churn_probability"] = probs
    else:
        out["churn_probability"] = 0.0
    out["churn_label"] = preds.astype(int)
    return out
