"""Example pipeline: raw -> cleaned -> RFM + engagement + churn label -> data/processed (run from repo root)."""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_from_excel
from src.data.cleaning import clean_transactions
from src.data.validate import validate_basic
from src.features.rfm import compute_rfm
from src.features.labels import add_churn_label_rfm
from src.features.engagement import add_engagement_score


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    excel_path = repo_root / "data" / "raw" / "online_retail_II.xlsx"
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_from_excel(excel_path)
    validate_basic(df)

    df_customer, _ = clean_transactions(df)
    rfm = compute_rfm(df_customer)
    rfm = add_engagement_score(rfm)
    rfm_labeled = add_churn_label_rfm(rfm, churn_days=90)

    out_path = out_dir / "customer_rfm_churn.csv"
    rfm_labeled.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows=", len(rfm_labeled))
    print(rfm_labeled.head())


if __name__ == "__main__":
    main()
