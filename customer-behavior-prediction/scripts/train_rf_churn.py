"""Train RandomForest baseline churn model and log metrics."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

from src.models.compare import append_comparison_row


FEATURE_COLS = ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")

    df = pd.read_csv(processed_csv)
    X = df[FEATURE_COLS].fillna(0.0)
    y = df["Churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    row = {
        "model": "RandomForest",
        "roc_auc": float(roc_auc_score(y_test, probs)) if probs is not None and y.nunique() > 1 else None,
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    joblib.dump(model, artifacts_dir / "churn_rf.joblib")
    joblib.dump(FEATURE_COLS, artifacts_dir / "churn_rf_features.joblib")
    append_comparison_row(artifacts_dir, row)
    print("Done:", row)


if __name__ == "__main__":
    main()
