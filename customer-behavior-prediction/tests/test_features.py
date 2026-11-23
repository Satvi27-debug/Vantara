"""Unit tests for feature engineering (basic sanity)."""

import pandas as pd

from src.features.rfm import compute_rfm
from src.features.labels import add_churn_label_rfm


def test_compute_rfm_and_label():
    df = pd.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "Invoice": ["A", "A", "B"],
            "InvoiceDate": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-01"]),
            "Quantity": [1, 2, 1],
            "Price": [10.0, 10.0, 5.0],
        }
    )
    rfm = compute_rfm(df, snapshot_date=pd.Timestamp("2021-02-01"))
    assert "RecencyDays" in rfm.columns
    assert len(rfm) == 2
    labeled = add_churn_label_rfm(rfm, churn_days=90)
    assert "Churn" in labeled.columns
