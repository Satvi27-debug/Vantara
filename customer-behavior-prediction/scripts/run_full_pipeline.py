"""Run full local pipeline: build features, then train baseline model."""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    print("[1/2] Building features...")
    subprocess.check_call([py, str(repo_root / "scripts" / "build_features_example.py")], cwd=str(repo_root))

    print("[2/2] Training baseline model...")
    subprocess.check_call([py, str(repo_root / "scripts" / "train_baseline_churn.py")], cwd=str(repo_root))

    print("Done. Artifacts in models_artifacts/, processed data in data/processed/.")


if __name__ == "__main__":
    main()
