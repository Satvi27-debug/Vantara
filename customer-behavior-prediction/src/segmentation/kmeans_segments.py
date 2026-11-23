"""Simple KMeans segmentation on RFM + Engagement (PRD Section 9)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def build_kmeans_segments(processed_csv: str | Path, artifacts_dir: str | Path, k: int = 4) -> Dict:
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

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Xs)

    out = df[["Customer ID"]].copy() if "Customer ID" in df.columns else pd.DataFrame({"row_id": range(len(df))})
    out["Segment"] = labels.astype(int)

    # Simple business-readable naming by average Monetary/Recency profile
    profile = df.copy()
    profile["Segment"] = labels
    summary = profile.groupby("Segment").agg(
        RecencyDays=("RecencyDays", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Customers=("Segment", "count"),
    ).reset_index()

    # Name heuristic
    def name_row(r):
        if r["RecencyDays"] > profile["RecencyDays"].median() and r["Monetary"] < profile["Monetary"].median():
            return "At Risk Low Value"
        if r["RecencyDays"] <= profile["RecencyDays"].median() and r["Monetary"] >= profile["Monetary"].median():
            return "Loyal High Value"
        if r["RecencyDays"] <= profile["RecencyDays"].median() and r["Monetary"] < profile["Monetary"].median():
            return "Active Low Value"
        return "Lapsed Mid Value"

    summary["SegmentName"] = summary.apply(name_row, axis=1)
    name_map = dict(zip(summary["Segment"], summary["SegmentName"]))
    out["SegmentName"] = out["Segment"].map(name_map)

    out_path = processed_csv.parent / "customer_segments.csv"
    out.to_csv(out_path, index=False)
    joblib.dump(model, artifacts_dir / "kmeans_segments.joblib")
    joblib.dump(scaler, artifacts_dir / "kmeans_segments_scaler.joblib")
    summary.to_csv(processed_csv.parent / "segment_summary.csv", index=False)

    return {"segments_csv": str(out_path), "summary_csv": str(processed_csv.parent / "segment_summary.csv"), "k": k}
