# Model notes (PRD-aligned)

## Implemented (runnable in this repo)
- Baseline churn: Logistic Regression (`scripts/train_baseline_churn.py`)
- Churn comparisons: RF/XGB/LGBM scripts append to `models_artifacts/model_comparison.csv`
- CLV regression baseline: `scripts/train_clv_regression.py` → `models_artifacts/clv_metrics.json`
- Segmentation: KMeans and GMM scripts + CSV outputs
- XAI: `scripts/generate_shap_summary.py`, dashboard local SHAP form (if artifacts+shap available), API `POST /explain/shap` (baseline model)
- Persistence: SQLite logging in batch scoring + `GET /predictions/recent`, example script `scripts/log_predictions_example.py`
- DL scaffolds: ANN script (if TensorFlow), LSTM/Autoencoder placeholders

## Still to complete for full PRD
- Full DL training/tuning and documented comparisons beyond scaffolds
- Broader XAI (LIME/PDP, plain-language template) fully integrated in UI/workflows
- Production DB choice hardening (Postgres option) and full persisted scoring views as specified
- Final exported architecture/ER diagrams (replace placeholders) and final report PDF
- Stricter leakage tests and coverage targets

## How to compare churn models
Run training scripts, then open `models_artifacts/model_comparison.csv` (also shown in dashboard).
