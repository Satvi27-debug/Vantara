# Vantara Retail Solutions — Customer Behavior Prediction Platform

PRD-driven prototype for churn risk, RFM/engagement features, segmentation, baseline models, FastAPI API, and Streamlit dashboard.

## What’s in this repo (at root)
- `api/`, `src/`, `scripts/`, `frontend/`, `notebooks/`, `config/`, `tests/`, `docs/`, `data/` (create `data/raw/` and place dataset), `models_artifacts/`
- `PRD.md` — improved PRD (markdown)
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`
- `docs/` contains notes/placeholders: `MODEL_NOTES.md`, `ARCHITECTURE_NOTES.md`, `ER_NOTES.md`, `final_report_outline.md`, and placeholder diagrams

## Quick start
1. Create and activate a Python 3.10+ environment, then install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the dataset at `data/raw/online_retail_II.xlsx` (Online Retail II, sheets: `Year 2009-2010`, `Year 2010-2011`).
3. Run the baseline pipeline:
   ```bash
   python scripts/run_full_pipeline.py
   python scripts/build_segments_example.py
   python scripts/build_segments_gmm_example.py
   ```
4. Start API and dashboard:
   ```bash
   uvicorn api.main:app --reload
   streamlit run frontend/dashboard.py
   ```

## Key docs
- `PRD.md` — product requirements (improved)
- `docs/MODEL_NOTES.md` — implemented vs remaining model work
- `docs/ARCHITECTURE_NOTES.md`, `docs/ER_NOTES.md` — architecture/ER notes

## Notes
- Baselines and scaffolds are included; some advanced PRD items (full DL runs, full XAI UI integration, final diagrams/report PDFs) are documented as next steps in `docs/MODEL_NOTES.md`.
- Do not commit real secrets; use environment variables as needed.

## Git identity (this repo)
- Name: Satvi27-debug
- Email: satvendragirinihal@gmail.com
