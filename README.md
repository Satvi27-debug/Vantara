# Customer Behavior Prediction Platform (Vantara Retail Solutions)

PRD-driven prototype: churn risk (baseline), RFM + engagement features, FastAPI API, Streamlit dashboard.

## Prerequisites
- Python 3.10+
- Install deps: `pip install -r requirements.txt`
- Dataset: Online Retail II (UCI). Put the Excel file at:
  - `data/raw/online_retail_II.xlsx` (sheets: `Year 2009-2010`, `Year 2010-2011`)

## End-to-end (local)
From the `customer-behavior-prediction/` directory:

### Option A: one command (recommended after data is in place)
```bash
python scripts/run_full_pipeline.py
```
Then optionally:
```bash
python scripts/build_segments_example.py
python scripts/build_segments_gmm_example.py
python scripts/train_rf_churn.py
python scripts/train_xgb_churn.py
python scripts/train_lgbm_churn.py
python scripts/train_clv_regression.py
python scripts/generate_shap_summary.py
python scripts/log_predictions_example.py
```

See `docs/MODEL_NOTES.md` for details and remaining PRD items.

## Notes
- Repo includes runnable baselines, segmentation, extra model training scripts, a basic SHAP plot script, and a SQLite logging example. Remaining PRD items (full DL, full XAI in UI, full DB-backed API flows, full docs/diagrams/report, stricter tests/coverage) are listed in `docs/MODEL_NOTES.md`.
