"""Gaussian Mixture Model segmentation (PRD Section 9 - alternative to DBSCAN)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def build_gmm_segments(processed_csv: str | Path, artifacts_dir: str | Path, n_components: int = 4) -> Dict:
    processed_csv = Path(processed_csv)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv)
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X = df[FEATURE_COLS].fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GaussianMixture(n_components=n_components, random_state=42)
    labels = model.fit_predict(Xs)

    out = df[["Customer ID"]].copy() if "Customer ID" in df.columns else pd.DataFrame({"row_id": range(len(df))})
    out["SegmentGMM"] = labels.astype(int)

    out_path = processed_csv.parent / "customer_segments_gmm.csv"
    out.to_csv(out_path, index=False)
    joblib.dump(model, artifacts_dir / "gmm_segments.joblib")
    joblib.dump(scaler, artifacts_dir / "gmm_segments_scaler.joblib")

    return {"segments_csv": str(out_path), "n_components": n_components}
