# Final Report (Draft) — Customer Behavior Prediction Platform

## 1. Executive summary
This project delivers a PRD-aligned prototype for customer churn risk, RFM/engagement features, segmentation, baseline modeling, a FastAPI service, and a Streamlit dashboard for Vantara Retail Solutions (internal training/onboarding context).

## 2. Data and EDA (summary)
- Primary dataset: Online Retail II (UCI). Raw files expected at `data/raw/online_retail_II.xlsx`.
- Cleaning steps implemented: deduplication, removal/filtering of non-product stock codes, basic price/quantity sanity filters, customer-ID separation for customer-level tables (see `src/data/cleaning.py`).
- EDA notebooks: `notebooks/01_eda.ipynb` (to be expanded with full charts/findings).

## 3. Feature engineering
- Implemented: RFM (Recency, Frequency, Monetary, HistoricalCLV, AvgOrderValue, TotalQuantity), EngagementScore (composite), baseline churn label (RecencyDays > threshold).
- Leakage discipline: point-in-time intent documented in PRD; concrete leakage tests to be strengthened.

## 4. Modeling
- Baselines/scripts: Logistic Regression (`train_baseline_churn.py`), RandomForest/XGBoost/LightGBM (`train_rf_churn.py`, `train_xgb_churn.py`, `train_lgbm_churn.py`) with metrics logged to `models_artifacts/model_comparison.csv`.
- CLV: RandomForest regression script (`train_clv_regression.py`) with metrics in `models_artifacts/clv_metrics.json`.
- DL scaffolds: ANN script template; LSTM/Autoencoder placeholders (see `docs/MODEL_NOTES.md`).

## 5. Segmentation
- KMeans (`build_segments_example.py`) and GMM (`build_segments_gmm_example.py`) producing customer segment assignments and summaries.

## 6. Explainability
- Scripts: `generate_shap_summary.py`, `generate_lime_example.py`, `generate_pdp.py` (best effort).
- API: `POST /explain/shap` returns top contributions for baseline model inputs.
- Dashboard: summary image display + local SHAP form. LIME/PDP outputs can be generated to `docs/` and referenced; plain-language template is a next step.

## 7. Deployment
- FastAPI: health, model metadata, predict, batch scoring (logs to SQLite), `GET /predictions/recent`, `GET /metrics/comparison`, `GET /metrics/clv`, and explain endpoint. Run via `uvicorn api.main:app`.
- Streamlit dashboard: overview, segmentation, leaderboard, model comparison, recent predictions, trends, explainability, batch scoring helper.
- Dockerfiles/compose provided for local orchestration.

## 8. Results vs success metrics
Run the provided scripts on the real dataset, then record measured metrics in artifacts and compare to PRD targets (ROC-AUC ≥0.80, CLV R² ≥0.60, churn recall ≥0.70, latency/dashboard notes).

## 9. Risks and limitations
- Small customer base can limit DL performance; tree baselines preferred initially.
- Class imbalance requires careful metric reporting.
- Some PRD items require executing training on real data and manual artifact finalization (diagrams/PDF).

## 10. Next steps
- Execute full training runs, generate final XAI artifacts, export final diagrams, produce final PDF, and tighten tests/coverage.
