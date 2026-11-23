"""Train baseline churn model from processed CSV and save artifacts."""

from pathlib import Path

from src.models.train_baseline import train_baseline_churn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"

    if not processed_csv.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_csv}. Run scripts/build_features_example.py first.")

    metrics = train_baseline_churn(processed_csv, artifacts_dir)
    print("Training complete. Metrics:", metrics)
    print("Artifacts saved to:", artifacts_dir)


if __name__ == "__main__":
    main()
