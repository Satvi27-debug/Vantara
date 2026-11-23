"""Append model metrics to a comparison CSV (PRD Section 11)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def append_comparison_row(artifacts_dir: str | Path, row: dict) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "model_comparison.csv"
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    return path
