"""Data cleaning utilities (PRD Section 6.2)."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


NON_PRODUCT_STOCKCODES = {
    "POST",
    "D",
    "M",
    "BANK CHARGES",
    "PADS",
    "DOT",
    "CRUK",
}


def _normalize_stockcode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean raw-like transactions dataframe.
    Returns:
      - df_customer: rows with valid Customer ID (for customer-level modeling)
      - df_all: cleaned full view (including missing CustomerID) for reference
    """
    df = df.copy()
    # Standardize key columns
    if "Customer ID" in df.columns:
        df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    # Drop exact duplicate lines
    before = len(df)
    df = df.drop_duplicates()
    _ = before - len(df)  # could log

    # Filter obvious non-product stock codes (helps product affinity later)
    if "StockCode" in df.columns:
        sc = _normalize_stockcode(df["StockCode"])
        df = df[~sc.isin(NON_PRODUCT_STOCKCODES)]

    # Basic price/quantity sanity: keep rows but we will separate returns
    # Returns are negative Quantity
    # Remove rows with zero/negative Price as per PRD note (adjustments) — be explicit
    if "Price" in df.columns:
        df = df[df["Price"].notna()]
        df = df[df["Price"] > 0]

    if "Quantity" in df.columns:
        df = df[df["Quantity"].notna()]

    df_all = df.copy()

    # Customer-level table requires Customer ID
    df_customer = df_all[df_all.get("Customer ID").notna()].copy() if "Customer ID" in df_all.columns else df_all.iloc[0:0].copy()
    # Make Customer ID int-like
    if "Customer ID" in df_customer.columns:
        df_customer["Customer ID"] = df_customer["Customer ID"].astype("Int64")

    return df_customer, df_all
