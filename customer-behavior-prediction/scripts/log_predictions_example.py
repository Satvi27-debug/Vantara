"""Example: run predictions on a small sample and persist to SQLite (docs/../data/predictions.db)."""

from pathlib import Path
import sqlite3
import pandas as pd
import joblib

from src.utils.db import init_predictions_table, get_conn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    db_path = repo_root / "data" / "predictions.db"

    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")

    init_predictions_table(db_path)

    df = pd.read_csv(processed_csv).head(50)
    # Try model if exists
    model_file = artifacts_dir / "churn_logreg.joblib"
    if model_file.exists():
        from src.models.infer import predict_churn
        out = predict_churn(df, artifacts_dir)
    else:
        out = df.copy()
        out["churn_probability"] = 0.5
        out["churn_label"] = 0

    conn = get_conn(db_path)
    try:
        for _, r in out.iterrows():
            conn.execute(
                "INSERT INTO predictions (customer_id, recency_days, frequency, monetary, churn_probability, churn_label) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    int(r.get("Customer ID")) if "Customer ID" in out.columns and pd.notna(r.get("Customer ID")) else None,
                    float(r.get("RecencyDays", 0)),
                    float(r.get("Frequency", 0)),
                    float(r.get("Monetary", 0)),
                    float(r.get("churn_probability", 0.5)),
                    int(r.get("churn_label", 0)),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    print("Wrote sample predictions to:", db_path)


if __name__ == "__main__":
    main()
