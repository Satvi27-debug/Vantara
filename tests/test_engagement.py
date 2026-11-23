"""Engagement score tests."""

import pandas as pd

from src.features.engagement import add_engagement_score


def test_add_engagement_score():
    df = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "RecencyDays": [10, 100],
            "Frequency": [5, 1],
            "Monetary": [500.0, 50.0],
        }
    )
    out = add_engagement_score(df)
    assert "EngagementScore" in out.columns
    assert out["EngagementScore"].between(0, 1).all()
