"""Label builders (PRD Section 5.1)."""

from __future__ import annotations

import pandas as pd


def add_churn_label_rfm(rfm: pd.DataFrame, churn_days: int = 90) -> pd.DataFrame:
    """
    Simple churn label: churned if RecencyDays > churn_days (baseline definition).
    """
    out = rfm.copy()
    out["Churn"] = (out["RecencyDays"] > churn_days).astype(int)
    return out
