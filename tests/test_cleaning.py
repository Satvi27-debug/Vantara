"""Basic cleaning tests."""

import pandas as pd

from src.data.cleaning import clean_transactions


def test_clean_transactions_basic():
    df = pd.DataFrame(
        {
            "Invoice": ["1", "1", "2"],
            "StockCode": ["A", "A", "POST"],
            "Description": ["x", "x", "postage"],
            "Quantity": [1, 1, 1],
            "InvoiceDate": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"]),
            "Price": [10.0, 10.0, 5.0],
            "Customer ID": [1, 1, 1],
            "Country": ["UK", "UK", "UK"],
        }
    )
    df_cust, df_all = clean_transactions(df)
    # duplicate removed, POST removed
    assert len(df_all) <= 2
    assert (df_cust["Customer ID"].notna()).all()
