"""Generate a SHAP summary plot for the baseline churn model (if shap is installed)."""

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    if not processed_csv.exists():
        raise FileNotFoundError("Run scripts/build_features_example.py first.")
    model_path = artifacts_dir / "churn_logreg.joblib"
    scaler_path = artifacts_dir / "churn_scaler.joblib"
    feats_path = artifacts_dir / "churn_features.joblib"
    if not (model_path.exists() and scaler_path.exists() and feats_path.exists()):
        raise FileNotFoundError("Run scripts/train_baseline_churn.py first.")

    try:
        import shap  # type: ignore
    except Exception as e:
        print("shap not installed; skipping. Install shap to enable this script.")
        return

    df = pd.read_csv(processed_csv)
    features = joblib.load(feats_path)
    X = df[features].fillna(0.0).sample(n=min(200, len(df)), random_state=42)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    Xs = scaler.transform(X)

    explainer = shap.LinearExplainer(model, Xs, feature_perturbation="interventional") if hasattr(shap, "LinearExplainer") else shap.Explainer(model, Xs)
    shap_values = explainer(Xs)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    out = docs_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()
