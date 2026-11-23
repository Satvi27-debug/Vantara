# Vantara Retail Solutions — Customer Behavior Prediction Platform

This repository contains a PRD-driven prototype for **Customer Behavior Prediction** (churn risk, RFM/engagement features, segmentation, baseline models, FastAPI API, and Streamlit dashboard) built as an internal analytics/training project.

## Contents
- `customer-behavior-prediction/` — main project (code, scripts, API, dashboard, docs, scaffolds)
- `9d043eea983979be328ec962c826cea6.docx` — Product Requirements Document (PRD)

## Quick start
See the detailed guide in:
- `customer-behavior-prediction/README.md`

### Minimal steps (after placing dataset)
```bash
cd customer-behavior-prediction
pip install -r requirements.txt
python scripts/run_full_pipeline.py
python scripts/build_segments_example.py
python scripts/build_segments_gmm_example.py
uvicorn api.main:app --reload
# in another terminal
streamlit run frontend/dashboard.py
```

## Dataset
Online Retail II (UCI). Place the Excel file at:
`customer-behavior-prediction/data/raw/online_retail_II.xlsx` (sheets: `Year 2009-2010`, `Year 2010-2011`).

## Notes
- This is a baseline/prototype implementation aligned to the PRD. Some advanced items (full DL tuning, full XAI UI integration, production DB wiring, final diagrams/report PDFs beyond placeholders) are documented as next steps in `customer-behavior-prediction/docs/MODEL_NOTES.md`.

## Author / Git identity (for this repo)
- Name: Satvi27-debug
- Email: satvendragirinihal@gmail.com
