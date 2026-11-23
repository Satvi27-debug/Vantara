"""Streamlit dashboard (improved: comparisons, recent predictions, segments, basic SHAP)."""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_csv_rel(rel_parts: tuple[str, ...]) -> pd.DataFrame | None:
    for base in [Path.cwd(), _repo_root()]:
        p = base.joinpath(*rel_parts)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    return None


def _load_predictions_db() -> pd.DataFrame | None:
    for base in [Path.cwd(), _repo_root()]:
        p = base / "data" / "predictions.db"
        if p.exists():
            try:
                conn = sqlite3.connect(str(p))
                df = pd.read_sql_query(
                    "SELECT id, customer_id, recency_days, frequency, monetary, churn_probability, churn_label, created_at FROM predictions ORDER BY id DESC LIMIT 200",
                    conn,
                )
                conn.close()
                return df
            except Exception:
                return None
    return None


def main() -> None:
    st.set_page_config(page_title="Vantara CBP Dashboard", layout="wide", page_icon="📊")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 Customer Behavior Prediction Platform")
    st.caption("Vantara Retail Solutions — Churn • RFM • Segmentation (local prototype)")

    df = _load_csv_rel(("data", "processed", "customer_rfm_churn.csv"))
    seg_kmeans = _load_csv_rel(("data", "processed", "customer_segments.csv"))
    seg_gmm = _load_csv_rel(("data", "processed", "customer_segments_gmm.csv"))
    comp = _load_csv_rel(("models_artifacts", "model_comparison.csv"))

    tabs = st.tabs(
        [
            "Overview",
            "Segmentation",
            "Churn Leaderboard",
            "Model Comparison",
            "Recent Predictions",
            "Trends",
            "Explainability",
            "Batch Scoring",
        ]
    )

    with tabs[0]:
        st.subheader("Overview")
        if df is None:
            st.warning("No processed data found. Run: `python scripts/run_full_pipeline.py`")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers", f"{len(df):,}")
            if "Churn" in df.columns:
                c2.metric("Churn Rate (baseline label)", f"{float(df['Churn'].mean() * 100):.1f}%")
            if "Monetary" in df.columns:
                c3.metric("Avg Monetary", f"{df['Monetary'].mean():,.2f}")
            if "Frequency" in df.columns:
                c4.metric("Avg Frequency", f"{df['Frequency'].mean():,.2f}")
            st.markdown("#### Sample data")
            st.dataframe(df.head(200), use_container_width=True, height=320)

    with tabs[1]:
        st.subheader("Customer Segmentation")
        choice = st.radio("Method", ["KMeans", "GMM"], horizontal=True, key="seg_method")
        seg = seg_kmeans if choice == "KMeans" else seg_gmm
        if seg is None:
            st.info(f"No {choice} segments found. Run the corresponding build script.")
        else:
            if df is not None and "Customer ID" in df.columns and "Customer ID" in seg.columns:
                merged = df.merge(seg, on="Customer ID", how="left")
            else:
                merged = seg
            label_col = "SegmentName" if "SegmentName" in merged.columns else ("SegmentGMM" if "SegmentGMM" in merged.columns else None)
            if label_col and label_col in merged.columns:
                st.markdown("#### Segment counts")
                st.bar_chart(merged[label_col].value_counts())
            st.dataframe(merged.head(300), use_container_width=True, height=360)

    with tabs[2]:
        st.subheader("Churn Risk Leaderboard (baseline labels / RFM view)")
        if df is None:
            st.info("Load processed data first.")
        else:
            sort_cols = [c for c in ["Churn", "Monetary"] if c in df.columns]
            view = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else df
            st.dataframe(view.head(200), use_container_width=True, height=420)

    with tabs[3]:
        st.subheader("Model Comparison (from artifacts)")
        if comp is None or comp.empty:
            st.info("No `models_artifacts/model_comparison.csv` found. Run training scripts (RF/XGB/LGBM) to populate it.")
        else:
            st.dataframe(comp, use_container_width=True)
        clv_path = _repo_root() / "models_artifacts" / "clv_metrics.json"
        if not clv_path.exists():
            clv_path = Path.cwd() / "models_artifacts" / "clv_metrics.json"
        if clv_path.exists():
            try:
                import json

                st.markdown("#### CLV metrics (latest saved)")
                st.json(json.loads(clv_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    with tabs[4]:
        st.subheader("Recent Predictions (from local SQLite log)")
        recent_preds = _load_predictions_db()
        if recent_preds is None or recent_preds.empty:
            st.info("No predictions logged yet. Use API `POST /batch/score` to score a CSV (logs to `data/predictions.db`).")
        else:
            st.dataframe(recent_preds, use_container_width=True, height=420)

    with tabs[5]:
        st.subheader("Trends")
        if df is None:
            st.info("Load processed data first.")
        else:
            st.markdown("Basic distributions (proxy trends). For time-series trends, extend with order-level aggregates.")
            if "Monetary" in df.columns:
                st.line_chart(df["Monetary"].head(300).reset_index(drop=True))

    with tabs[6]:
        st.subheader("Explainability")
        st.markdown("**SHAP summary image (if generated):**")
        shap_img = _repo_root() / "docs" / "shap_summary.png"
        if not shap_img.exists():
            shap_img = Path.cwd() / "docs" / "shap_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="SHAP summary", use_container_width=True)
        else:
            st.info("No `docs/shap_summary.png`. Run `python scripts/generate_shap_summary.py` (needs shap).")

        st.markdown("**Quick local SHAP (baseline model):** enter values and compute.")
        c1, c2, c3 = st.columns(3)
        with c1:
            rec = st.number_input("RecencyDays", value=30.0, min_value=0.0)
            freq = st.number_input("Frequency", value=5.0, min_value=0.0)
        with c2:
            mon = st.number_input("Monetary", value=200.0, min_value=0.0)
            aov = st.number_input("AvgOrderValue (optional)", value=0.0, min_value=0.0)
        with c3:
            tq = st.number_input("TotalQuantity (optional)", value=0.0, min_value=0.0)
            eng = st.number_input("EngagementScore (optional)", value=0.0, min_value=0.0, max_value=1.0)

        if st.button("Compute SHAP (local)"):
            try:
                import joblib
                import numpy as np

                artifacts_dir = _repo_root() / "models_artifacts"
                if not (artifacts_dir / "churn_logreg.joblib").exists():
                    artifacts_dir = Path.cwd() / "models_artifacts"
                model_path = artifacts_dir / "churn_logreg.joblib"
                scaler_path = artifacts_dir / "churn_scaler.joblib"
                feats_path = artifacts_dir / "churn_features.joblib"
                if not (model_path.exists() and feats_path.exists() and scaler_path.exists()):
                    st.error("Baseline artifacts not found. Train model first.")
                else:
                    try:
                        import shap  # type: ignore
                    except Exception:
                        st.error("shap not installed.")
                    else:
                        features = joblib.load(feats_path)
                        aovv = aov if aov > 0 else (mon / max(freq, 1.0))
                        tqv = tq if tq > 0 else freq
                        if eng > 0:
                            engv = eng
                        else:
                            r = max(rec, 0.0)
                            f = max(freq, 0.0)
                            m = max(mon, 0.0)
                            r_score = 1.0 / (1.0 + r / 30.0)
                            f_score = min(f / 10.0, 1.0)
                            m_score = min(m / 500.0, 1.0)
                            engv = 0.4 * r_score + 0.3 * f_score + 0.3 * m_score
                        row = {
                            "RecencyDays": rec,
                            "Frequency": freq,
                            "Monetary": mon,
                            "AvgOrderValue": aovv,
                            "TotalQuantity": tqv,
                            "EngagementScore": engv,
                        }
                        X = pd.DataFrame([[row[f] for f in features]], columns=features)
                        scaler = joblib.load(scaler_path)
                        model = joblib.load(model_path)
                        Xs = scaler.transform(X)
                        try:
                            explainer = shap.LinearExplainer(model, Xs, feature_perturbation="interventional")
                            sv = explainer(Xs)
                            vals = np.array(sv.values).ravel()
                        except Exception:
                            explainer = shap.Explainer(model, scaler.transform)
                            sv = explainer(X)
                            vals = np.array(sv.values).ravel()
                        contribs = sorted(
                            [{"feature": f, "shap_value": float(v)} for f, v in zip(features, vals)],
                            key=lambda x: abs(x["shap_value"]),
                            reverse=True,
                        )
                        st.dataframe(pd.DataFrame(contribs), use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")

    with tabs[7]:
        st.subheader("Batch Scoring")
        st.markdown("Upload CSV with at least: `RecencyDays`, `Frequency`, `Monetary`.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_up")
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            st.dataframe(up_df.head(50), use_container_width=True)
            st.success("Call API: `POST /batch/score` or `/batch/score.csv` to score and log results.")


if __name__ == "__main__":
    main()
