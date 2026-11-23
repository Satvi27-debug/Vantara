"""Create a sample batch CSV for /batch/score testing from processed RFM file."""

from pathlib import Path

import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    out = repo_root / "data" / "processed" / "sample_batch_for_scoring.csv"
    if not processed.exists():
        raise FileNotFoundError(f"Not found: {processed}. Run scripts/build_features_example.py first.")
    df = pd.read_csv(processed)
    # Keep feature columns only
    keep = ["Customer ID", "RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity"]
    keep = [c for c in keep if c in df.columns]
    sample = df[keep].head(200).copy()
    sample.to_csv(out, index=False)
    print("Wrote:", out, "rows=", len(sample))


if __name__ == "__main__":
    main()
