# Model notes (PRD-aligned)

## Implemented (runnable in this repo)
- Baseline churn: Logistic Regression (`scripts/train_baseline_churn.py`)
- Churn comparisons: RF/XGB/LGBM scripts append to `models_artifacts/model_comparison.csv`
- CLV regression baseline: `scripts/train_clv_regression.py` → `models_artifacts/clv_metrics.json`
- Segmentation: KMeans and GMM scripts + CSV outputs
- XAI: `scripts/generate_shap_summary.py`, `scripts/generate_lime_example.py`, `scripts/generate_pdp.py` (best effort), dashboard local SHAP form, API `POST /explain/shap`
- Metrics API: `GET /metrics/comparison`, `GET /metrics/clv`
- Persistence: SQLite logging in batch scoring + `GET /predictions/recent`, example script `scripts/log_predictions_example.py`
- DL scaffolds: ANN script (if TensorFlow), LSTM/Autoencoder placeholders

## Still requires manual/data execution (cannot be fully auto-completed here)
- Running full DL training/tuning on the real dataset and recording final metrics
- Exporting final architecture/ER diagrams as polished images (placeholders + notes provided) and final PDF report
- Expanding XAI narrative (plain-language template) and any stakeholder-specific dashboard polish beyond current views

## How to compare churn models
Run training scripts, then open `models_artifacts/model_comparison.csv` or call `GET /metrics/comparison`.
