"""Streamlit dashboard (improved UI)."""

from pathlib import Path

import pandas as pd
import streamlit as st


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_csv(name: str) -> pd.DataFrame | None:
    for p in [Path(f"data/processed/{name}"), _repo_root() / "data" / "processed" / name]:
        if p.exists():
            return pd.read_csv(p)
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

    df = _load_csv("customer_rfm_churn.csv")
    seg_kmeans = _load_csv("customer_segments.csv")
    seg_gmm = _load_csv("customer_segments_gmm.csv")

    tabs = st.tabs(["Overview", "Segmentation", "Churn Leaderboard", "Trends", "Explainability", "Batch Scoring"])

    with tabs[0]:
        st.subheader("Overview")
        if df is None:
            st.warning("No processed data found. Run: `python scripts/run_full_pipeline.py` (or build features first).")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers", f"{len(df):,}")
            if "Churn" in df.columns:
                churn_rate = float(df["Churn"].mean() * 100)
                c2.metric("Churn Rate (baseline)", f"{churn_rate:.1f}%")
            if "Monetary" in df.columns:
                c3.metric("Avg Monetary", f"{df['Monetary'].mean():,.2f}")
            if "Frequency" in df.columns:
                c4.metric("Avg Frequency", f"{df['Frequency'].mean():,.2f}")

            st.markdown("#### Sample data")
            st.dataframe(df.head(200), use_container_width=True, height=320)

            colA, colB = st.columns(2)
            with colA:
                if "RecencyDays" in df.columns:
                    st.markdown("#### Recency (days) distribution (sample)")
                    st.bar_chart(df["RecencyDays"].head(300))
            with colB:
                if "Churn" in df.columns:
                    st.markdown("#### Churn distribution")
                    st.bar_chart(df["Churn"].value_counts())

    with tabs[1]:
        st.subheader("Customer Segmentation")
        choice = st.radio("Segmentation method", ["KMeans", "GMM"], horizontal=True)
        seg = seg_kmeans if choice == "KMeans" else seg_gmm
        if seg is None:
            st.info(f"No {choice} segments found. Run the corresponding build script.")
        else:
            if df is not None and "Customer ID" in df.columns and "Customer ID" in seg.columns:
                merged = df.merge(seg, on="Customer ID", how="left")
            else:
                merged = seg
            label_col = "SegmentName" if "SegmentName" in merged.columns else ("SegmentGMM" if "SegmentGMM" in merged.columns else None)
            if label_col:
                st.markdown("#### Segment counts")
                st.bar_chart(merged[label_col].value_counts())
            st.dataframe(merged.head(300), use_container_width=True, height=360)

    with tabs[2]:
        st.subheader("Churn Risk Leaderboard (baseline labels)")
        if df is None:
            st.info("Load processed data first.")
        else:
            sort_cols = []
            if "Churn" in df.columns:
                sort_cols.append("Churn")
            if "Monetary" in df.columns:
                sort_cols.append("Monetary")
            view = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else df
            st.dataframe(view.head(150), use_container_width=True, height=420)

    with tabs[3]:
        st.subheader("Trends")
        st.info("Trend charts from raw/order-level time series can be added next (placeholder).")

    with tabs[4]:
        st.subheader("Explainability")
        st.text_input("Customer ID (for future SHAP/LIME lookup)", value="", key="xai_cust")
        st.info("Placeholder: integrate SHAP/LIME outputs and show per-customer explanation panel.")

    with tabs[5]:
        st.subheader("Batch Scoring")
        st.markdown("Upload a CSV with at least: `RecencyDays`, `Frequency`, `Monetary` (optional: `AvgOrderValue`, `TotalQuantity`, `EngagementScore`).")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            st.markdown("#### Preview")
            st.dataframe(up_df.head(50), use_container_width=True)
            st.success("Use the API: `POST /batch/score` or `POST /batch/score.csv` to score this file.")


if __name__ == "__main__":
    main()
