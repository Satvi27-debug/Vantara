"""Minimal baseline churn model training (Logistic Regression on RFM + Engagement)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


FEATURE_COLS = ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def train_baseline_churn(processed_csv: str | Path, artifacts_dir: str | Path) -> Dict[str, Any]:
    processed_csv = Path(processed_csv)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv)
    for c in FEATURE_COLS + ["Churn"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X = df[FEATURE_COLS].fillna(0.0)
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1] if len(model.classes_) == 2 else model.predict_proba(X_test_s).max(axis=1)

    metrics = {
        "model": "LogisticRegression (baseline)",
        "features": FEATURE_COLS,
        "roc_auc": float(roc_auc_score(y_test, probs)) if y.nunique() > 1 else None,
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(model, artifacts_dir / "churn_logreg.joblib")
    joblib.dump(scaler, artifacts_dir / "churn_scaler.joblib")
    joblib.dump(FEATURE_COLS, artifacts_dir / "churn_features.joblib")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
