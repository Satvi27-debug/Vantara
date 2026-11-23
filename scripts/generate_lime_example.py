"""Generate a simple LIME explanation for the baseline model (if lime installed)."""

from pathlib import Path
import json
import joblib
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")
    for name in ["churn_logreg.joblib", "churn_scaler.joblib", "churn_features.joblib"]:
        if not (artifacts_dir / name).exists():
            raise FileNotFoundError("Run scripts/train_baseline_churn.py first.")

    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception:
        print("lime not installed. pip install lime to enable.")
        return

    df = pd.read_csv(processed_csv)
    features = joblib.load(artifacts_dir / "churn_features.joblib")
    X = df[features].fillna(0.0)
    scaler = joblib.load(artifacts_dir / "churn_scaler.joblib")
    model = joblib.load(artifacts_dir / "churn_logreg.joblib")
    Xs = scaler.transform(X)

    explainer = LimeTabularExplainer(
        Xs,
        feature_names=features,
        class_names=["no_churn", "churn"],
        mode="classification",
        random_state=42,
    )
    i = 0
    exp = explainer.explain_instance(Xs[i], model.predict_proba, num_features=min(6, len(features)))
    out = {
        "instance_index": i,
        "explanation": exp.as_list(),
    }
    (docs_dir / "lime_example.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved docs/lime_example.json")


if __name__ == "__main__":
    main()
