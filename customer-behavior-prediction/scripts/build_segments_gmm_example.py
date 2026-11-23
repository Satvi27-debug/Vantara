"""Build GMM segments from processed RFM+engagement data."""

from pathlib import Path

from src.segmentation.gmm_segments import build_gmm_segments


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"

    if not processed_csv.exists():
        raise FileNotFoundError(f"Not found: {processed_csv}. Run scripts/build_features_example.py first.")

    result = build_gmm_segments(processed_csv, artifacts_dir, n_components=4)
    print("GMM segments built:", result)


if __name__ == "__main__":
    main()
