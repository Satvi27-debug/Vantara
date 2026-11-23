# Architecture (PRD Section 14)

## Intended layers
- Data layer: `data/raw`, `data/interim`, `data/processed`, optional SQLite/Postgres for predictions/segments
- Processing layer: `src/data`, `src/features` (Pandas/NumPy pipelines)
- Modeling layer: `src/models`, `models_artifacts/` (serialized models/scalers/encoders)
- Serving layer: `api/` (FastAPI), `frontend/` (Streamlit)

## Data flow (describe)
Raw Excel/CSV → cleaning/validation → customer-level feature table → training/inference → predictions stored (CSV/DB) → API serves predictions → dashboard visualizes.

Export a diagram image to `docs/architecture_diagram.png` when ready (draw.io/Lucidchart).
