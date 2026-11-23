"""Engagement score (composite) - PRD Section 7."""

from __future__ import annotations

import pandas as pd


def add_engagement_score(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Simple normalized composite engagement score:
    higher frequency/monetary and lower recency => higher score.
    """
    out = rfm.copy()
    # Normalize safely
    for col, invert in [("RecencyDays", True), ("Frequency", False), ("Monetary", False)]:
        if col not in out.columns:
            out[col] = 0.0
        vals = out[col].astype(float)
        rng = vals.max() - vals.min()
        if rng == 0:
            norm = 0.5
        else:
            norm = (vals - vals.min()) / rng
        out[f"_n_{col}"] = 1.0 - norm if invert else norm

    out["EngagementScore"] = (out["_n_RecencyDays"] * 0.4 + out["_n_Frequency"] * 0.3 + out["_n_Monetary"] * 0.3)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_n_")])
    return out
