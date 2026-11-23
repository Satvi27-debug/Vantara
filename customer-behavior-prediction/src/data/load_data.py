"""Data loading utilities for Online Retail II (PRD Section 5/6)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_from_excel(excel_path: str | Path) -> pd.DataFrame:
    """
    Load Online Retail II from Excel with sheets 'Year 2009-2010' and 'Year 2010-2011'.
    Returns a single concatenated DataFrame sorted by InvoiceDate if present.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Raw Excel not found: {excel_path}")

    df_1 = pd.read_excel(excel_path, sheet_name="Year 2009-2010")
    df_2 = pd.read_excel(excel_path, sheet_name="Year 2010-2011")
    df = pd.concat([df_1, df_2], ignore_index=True)

    # Normalize column names lightly (strip)
    df.columns = [str(c).strip() for c in df.columns]

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.sort_values("InvoiceDate").reset_index(drop=True)

    return df


def load_from_ucimlrepo(dataset_id: int = 502) -> pd.DataFrame:
    """
    Optional loader via ucimlrepo (may require network/package). PRD recommends this when available.
    """
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("ucimlrepo is not installed or failed to import.") from e

    ds = fetch_ucirepo(id=dataset_id)
    df = ds.data.features.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df
