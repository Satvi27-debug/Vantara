"""Generate PDP plots for top features (best effort)."""

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
    if not (artifacts_dir / "churn_logreg.joblib").exists():
        raise FileNotFoundError("Run scripts/train_baseline_churn.py first.")

    try:
        from sklearn.inspection import PartialDependenceDisplay  # type: ignore
    except Exception:
        print("sklearn PartialDependenceDisplay not available.")
        return

    df = pd.read_csv(processed_csv)
    features = joblib.load(artifacts_dir / "churn_features.joblib")
    model = joblib.load(artifacts_dir / "churn_logreg.joblib")
    X = df[features].fillna(0.0).sample(n=min(300, len(df)), random_state=42)

    top = features[:2] if len(features) >= 2 else features
    try:
        PartialDependenceDisplay.from_estimator(model, X, top)
        out = docs_dir / "pdp_top_features.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)
    except Exception as e:
        print("PDP generation failed:", e)


if __name__ == "__main__":
    main()
