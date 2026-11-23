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
- SHAP summary script (`generate_shap_summary.py`) can produce `docs/shap_summary.png` if `shap` installed.
- API endpoint `POST /explain/shap` (baseline model) returns top SHAP contributions for provided feature values.
- Dashboard includes a local SHAP form and displays the summary image when available. LIME/PDP and a formal plain-language template remain next steps.

## 7. Deployment
- FastAPI app with health, model metadata, predict, batch scoring (CSV), recent predictions (`GET /predictions/recent`), and explain endpoint. Batch scoring logs to local SQLite (`data/predictions.db`).
- Streamlit dashboard: overview, segmentation, leaderboard, model comparison, recent predictions, trends, explainability, batch scoring helper.
- Dockerfiles/compose provided for local orchestration (review/harden as needed).

## 8. Results vs success metrics
Populate after running full experiments on the real dataset and recording metrics in `models_artifacts/` and comparison CSV. Compare against PRD targets (ROC-AUC, recall, CLV R², latency notes).

## 9. Risks and limitations
- Small customer base can limit DL performance; tree baselines preferred initially.
- Class imbalance requires careful metric reporting.
- Some PRD features remain scaffolded/documented for follow-up.

## 10. Next steps
- Complete DL training/tuning runs, expand XAI (LIME/PDP, stakeholder template), harden DB option, export final diagrams, and finalize measured results in this report.
