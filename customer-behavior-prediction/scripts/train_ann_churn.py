"""ANN churn classifier scaffold (PRD Section 8.2). Requires tensorflow/keras installed.
This is a template; adapt preprocessing/scaling and training as needed.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


FEATURE_COLS = ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "EngagementScore"]


def main() -> None:
    try:
        from tensorflow import keras  # type: ignore
    except Exception:
        print("TensorFlow not installed. Install tensorflow to run this scaffold.")
        return

    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")

    df = pd.read_csv(processed_csv)
    X = df[FEATURE_COLS].fillna(0.0).values
    y = df["Churn"].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

    probs = model.predict(X_test_s, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "model": "ANN (scaffold)",
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(set(y)) > 1 else None,
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    model.save(artifacts_dir / "churn_ann.keras")
    joblib.dump(scaler, artifacts_dir / "churn_ann_scaler.joblib")
    joblib.dump(FEATURE_COLS, artifacts_dir / "churn_ann_features.joblib")
    (artifacts_dir / "ann_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Done:", metrics)


if __name__ == "__main__":
    main()
