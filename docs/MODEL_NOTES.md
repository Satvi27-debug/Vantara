# Model notes (PRD-aligned)

## Implemented (runnable in this repo)
- Baseline churn: Logistic Regression (`scripts/train_baseline_churn.py`)
- Churn comparisons (scripts): RandomForest, XGBoost, LightGBM (append to `models_artifacts/model_comparison.csv`)
- CLV regression baseline: RandomForest (`scripts/train_clv_regression.py`)
- Segmentation: KMeans (`scripts/build_segments_example.py`), GMM (`scripts/build_segments_gmm_example.py`)
- XAI (basic): `scripts/generate_shap_summary.py` (saves `docs/shap_summary.png` if `shap` installed), `scripts/generate_lime_example.py` (placeholder)
- Local persistence example: `scripts/log_predictions_example.py` writes sample predictions to `data/predictions.db` (SQLite)
- DL scaffolds: `scripts/train_ann_churn.py` (runnable if TensorFlow installed), `scripts/train_lstm_placeholder.py`, `scripts/train_autoencoder_placeholder.py`

## Docs placeholders added
- `docs/ARCHITECTURE_NOTES.md`, `docs/ER_NOTES.md`, `docs/final_report_outline.md` (fill/export diagrams/report as needed)

## In PRD but still need your completion
- Full DL training runs and tuning, full XAI narratives in UI, full DB-backed API integration, final diagrams exported as images, final report PDF, coverage/leakage test hardening.

## How to compare churn models
Run training scripts, then open `models_artifacts/model_comparison.csv`.
