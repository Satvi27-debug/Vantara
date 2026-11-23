# Vantara Retail Solutions — Customer Behavior Prediction Platform (PRD)

**Document Owner:** Data Science & Analytics Division  
**Primary Contributor:** Single Data Scientist/ML Engineer (intern-to-associate level)  
**Timeline:** 4 weeks (20 working days), full-time effort  
**Version:** 1.1 (revised for clarity)  
**Status:** Approved for execution (internal training/onboarding)  
**Classification:** Internal

---

## 1. Executive summary
Vantara Retail Solutions needs a practical, end-to-end analytics product that turns historical omnichannel transaction data into **actionable customer intelligence**. The **Customer Behavior Prediction Platform** will ingest order history, engineer customer-level features, train and compare ML/DL models, produce interpretable churn and value insights, segment customers, and expose results through a **REST API** and a **Streamlit dashboard**.

This project is both (a) a usable internal prototype for Marketing/Retention workflows and (b) a reference implementation for onboarding and knowledge transfer. Delivery emphasizes **reproducibility**, **explainability**, and **clear handoff documentation**.

---

## 2. Business problem and motivation
### 2.1 Current state (pain points)
- Retention/marketing decisions rely on **rule-based segments** (e.g., “no order in 90 days”), which are reactive and treat customers within a segment uniformly.
- **High-value customers** showing early disengagement are often identified too late.
- Product recommendations rely on **generic best-seller logic** instead of individual behavior.

### 2.2 Why this matters
- Retaining customers is typically cheaper than acquiring new ones; accurate churn risk and CLV signals help prioritize spend.
- A single, maintained predictive layer can replace fragmented ad-hoc reports and support consistent campaign targeting.

### 2.3 Intended business use cases
- Prioritized retention campaigns for high-value, high-risk customers
- Smarter promotion allocation informed by risk/value (and future sensitivity analysis)
- Demand/assortment planning signals from predicted behavior
- Personalized recommendation inputs for email/site journeys
- Executive cohort/revenue health views (dashboard)

---

## 3. Goals, objectives, and success metrics
### 3.1 Objectives
1. Build a **reproducible pipeline**: raw transactions → validated cleaned data → customer-level feature tables.
2. Train and compare **classical ML** models for churn and at least **one CLV regression** model.
3. Train and compare **deep learning** baselines: ANN (tabular), LSTM (sequences), Autoencoder (anomaly detection) as specified.
4. Produce **customer segments** using ≥2 approaches with clear business profiles.
5. Provide **global and local explainability** (SHAP/LIME + plain-language summaries).
6. Deliver **FastAPI** services and a **Streamlit** dashboard for non-technical stakeholders.

### 3.2 Success metrics (targets)
| Metric | Target |
|--------|--------|
| Churn ROC-AUC (held-out test) | ≥ 0.80 |
| CLV regression R² (test) | ≥ 0.60 |
| Churn recall (churned class) | ≥ 0.70 (priority over accuracy) |
| API p95 single-customer prediction latency | < 400 ms |
| Dashboard full segment view load | < 3 s (local reference) |
| Reproducible pipeline | Single documented command/path from raw → processed where feasible |

**Important:** Churn is expected to be imbalanced. Report **Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix** together; do not optimize for accuracy alone.

---

## 4. Scope
### 4.1 In scope (MVP)
- End-to-end pipeline and versioned scripts/notebooks as defined in repo structure.
- Classical churn models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, plus one of KNN or SVM.
- ANN on same engineered features for comparison.
- LSTM on time-ordered purchase sequences for purchase-timing/behavior modeling (as specified).
- Autoencoder for unsupervised anomaly detection on spending patterns.
- Segmentation: K-Means (+ elbow/silhouette) and one of DBSCAN or GMM; optional hierarchical exploration on a sample.
- XAI: SHAP global + example force plots, ≥1 LIME example, PDPs for key features, plain-language “why flagged” template.
- FastAPI + Streamlit + Dockerized local run (`docker-compose up` target).

### 4.2 Out of scope (explicit)
- Transformer-based sequence models as required deliverables (document as future work).
- Real-time streaming ingestion (batch scoring in this phase).
- Production-grade multi-tenant auth, RBAC, full CI/CD hardening (note as future).
- Formal A/B testing framework for live campaign impact measurement.

---

## 5. Data
### 5.1 Primary dataset
**Online Retail II** (UCI Machine Learning Repository; Kaggle mirrors available). Typical scale: ~1.07M transaction line items across two sheets (2009–2011), line-item granularity, key fields include Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country. ~5.9k customers with valid IDs. Derived targets: churn label (e.g., no purchase in next 90 days in labeling strategy), CLV-related outputs, next-purchase/category signals as feasible.

### 5.2 Optional enrichment (stretch)
Customer Personality Analysis (Kaggle) as a separate module only (no shared keys) — not required for core delivery.

### 5.3 Data quality considerations (must be handled)
- Missing Customer ID: exclude from customer-level modeling (may be used only for product-level analysis if needed).
- Negative Quantity: returns/cancellations — handle explicitly (not silent drops) where they carry signal.
- Zero/negative or invalid prices: apply documented cleaning rules.
- Non-product StockCodes (e.g., postage/adjustments): filter for product-affinity features.
- Seasonality and calendar effects: account for in feature/EDA interpretation.

### 5.4 Access
Prefer UCI as authoritative source; `ucimlrepo` (id=502) recommended when available. Manual Excel download is fallback. Respect dataset license (CC BY 4.0) with proper attribution.

---

## 6. Functional requirements (summary)
### 6.1 Data pipeline
- Load both sheets, standardize columns/types, combine chronologically.
- Cleaning with documented rules, deduplication strategy, outlier handling (IQR + domain rules), validation checks that **fail loudly** before feature engineering.
- Wrangling in versioned, testable Python (Pandas/NumPy): encoding/scaling strategies per model family; **persist fitted preprocessors** for inference parity.

### 6.2 Feature engineering (examples)
Recency, Frequency, Monetary (and historical CLV), basket behavior, trends/variability, seasonality signals, category affinity, return rate, discount sensitivity proxies, engagement score. **Point-in-time cutoff discipline** is mandatory to avoid leakage; include leakage-focused tests where practical.

### 6.3 Modeling
- Classical + required DL models per Section 4.1, with fixed seeds, stratified splits, CV on train only, test used once for final reporting, and run logging for comparability.
- Recommended emphasis: tree ensembles (e.g., XGBoost/LightGBM) often outperform small-data DL for this dataset size; document trade-offs honestly.

### 6.4 Segmentation
K-Means (elbow + silhouette) plus DBSCAN or GMM; each segment needs **business-readable names and summary statistics**.

### 6.5 Explainability
SHAP (global + per-customer force plots for representative cases), ≥1 LIME comparison, PDPs for top features, and a plain-language explanation template for stakeholders.

### 6.6 Evaluation
Churn: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix (+ stratified CV). CLV: MAE, RMSE, R². Clustering: Silhouette, Davies–Bouldin. Produce a consolidated comparison table and a written model recommendation.

### 6.7 Serving & UI
- FastAPI: health, model metadata, single prediction, batch scoring (CSV), validated inputs (Pydantic).
- Streamlit dashboard: segmentation views, churn leaderboard, trends, per-customer explanations (when available), batch upload flow.
- Docker Compose for local stack (API, dashboard, DB as designed).

---

## 7. Non-functional requirements
- Code quality: type hints, docstrings, linting; structured logging; tests (including leakage intent) with a coverage goal aligned to PRD (target ≥70% on core package where stated).
- Config via YAML/env vars; no secrets in VCS; reproducible environment files.
- Documentation: README with clean setup/run steps, architecture/ER notes or diagrams, and clear handoff notes.

---

## 8. Deliverables (checklist)
- Cleaned, feature-engineered datasets and pipeline scripts
- Serialized model artifacts and comparison logs/metrics
- Segmentation outputs with business profiles
- Explainability artifacts/scripts and templates (as implemented)
- Working FastAPI + Streamlit (local)
- Docker Compose setup
- Documentation: README, model notes, architecture/ER notes, report outline (and exported diagrams/PDF when produced)

---

## 9. Risks and mitigations (high level)
- Small customer base limits DL gains → prefer strong tabular baselines, document clearly.
- Class imbalance → class weights/SMOTE on train only, report recall/F1/AUC.
- Leakage risk → point-in-time rules and tests.
- Time constraints → defer Transformers/CatBoost-style extras if needed; prioritize required core path.

---

## 10. Acceptance criteria (summary)
- Documented path to reproduce processed data and train baseline models.
- Required model families represented via scripts/notebooks as specified (or clearly documented stubs with next steps).
- Segments produced with interpretable profiles.
- API and dashboard runnable locally (Docker or documented local commands).
- Metrics and limitations discussed honestly relative to Section 3.2 targets.

---

*This PRD is the source of truth for scope. Implementation should not add unrelated features (“gold plating”) beyond what is needed to satisfy the objectives above.*