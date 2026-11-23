"""Build KMeans segments from processed RFM+engagement data."""

from pathlib import Path

from src.segmentation.kmeans_segments import build_kmeans_segments


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_csv = repo_root / "data" / "processed" / "customer_rfm_churn.csv"
    artifacts_dir = repo_root / "models_artifacts"

    if not processed_csv.exists():
        raise FileNotFoundError(f"Not found: {processed_csv}. Run scripts/build_features_example.py first.")

    result = build_kmeans_segments(processed_csv, artifacts_dir, k=4)
    print("Segments built:", result)


if __name__ == "__main__":
    main()
