"""RFM feature engineering (PRD Section 7)."""

from __future__ import annotations

import pandas as pd


def compute_rfm(df_customer: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute RFM at customer level.
    df_customer must include: Customer ID, Invoice, InvoiceDate, Quantity, Price.
    Returns one row per Customer ID with RecencyDays, Frequency, Monetary, HistoricalCLV.
    """
    if snapshot_date is None:
        snapshot_date = df_customer["InvoiceDate"].max() + pd.Timedelta(days=1)

    df = df_customer.copy()
    df["LineAmount"] = df["Quantity"] * df["Price"]

    sales = df[df["Quantity"] > 0].copy()

    agg = sales.groupby("Customer ID").agg(
        RecencyDays=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineAmount", "sum"),
        HistoricalCLV=("LineAmount", "sum"),
        AvgOrderValue=("LineAmount", "mean"),
        TotalQuantity=("Quantity", "sum"),
    ).reset_index()

    return agg
