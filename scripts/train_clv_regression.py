"""Train a simple CLV regression baseline (RandomForest) and save metrics."""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLS = ["RecencyDays", "Frequency", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")

    df = pd.read_csv(processed_csv)
    target = "HistoricalCLV" if "HistoricalCLV" in df.columns else "Monetary"
    for c in FEATURE_COLS + [target]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X = df[FEATURE_COLS].fillna(0.0)
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    r2 = float(r2_score(y_test, preds))
    metrics = {"model": "RandomForestRegressor (CLV baseline)", "target": target, "mae": mae, "rmse": rmse, "r2": r2}
    joblib.dump(model, artifacts_dir / "clv_rf.joblib")
    joblib.dump(FEATURE_COLS, artifacts_dir / "clv_rf_features.joblib")
    (artifacts_dir / "clv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Done:", metrics)


if __name__ == "__main__":
    main()
