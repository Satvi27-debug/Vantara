"""Basic data validation checks (PRD Section 6.2)."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_basic(df: pd.DataFrame, min_rows: int = 1000) -> dict:
    """
    Run basic checks. Raises on hard failures; returns summary dict for soft checks.
    """
    validate_schema(df)

    if len(df) < min_rows:
        raise ValueError(f"Too few rows: {len(df)} < {min_rows}")

    null_rates = df[REQUIRED_COLUMNS].isna().mean().to_dict()
    # Hard fail if InvoiceDate has too many nulls
    if null_rates.get("InvoiceDate", 0.0) > 0.01:
        raise ValueError(f"InvoiceDate null rate too high: {null_rates['InvoiceDate']:.3f}")

    summary = {
        "rows": len(df),
        "null_rates": null_rates,
        "min_invoice_date": str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
        "max_invoice_date": str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
    }
    return summary
