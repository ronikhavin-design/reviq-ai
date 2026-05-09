"""
RevIQ AI: Executive Revenue Intelligence Dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.config import REPORTS_DIR, PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR
from src.features.build_features import get_model_matrix, load_raw, build_churn_features
from src.explainability.shap_explainer import (
    compute_shap_values, top_drivers_for_customer, format_churn_explanation,
)

st.set_page_config(
    page_title="RevIQ AI: Revenue Intelligence",
    page_icon="",
    layout="wide",
)

# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    path = MODELS_DIR / "churn_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_risk_table() -> pd.DataFrame | None:
    path = REPORTS_DIR / "customer_risk_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_targets() -> pd.DataFrame:
    return pd.read_csv(SYNTHETIC_DIR / "targets.csv")

@st.cache_data
def load_churn_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "churn_features.csv")

def risk_badge(level: str) -> str:
    colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return colors.get(level, "⚪")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RevIQ AI")
    st.caption("SaaS Revenue Intelligence Platform")
    st.divider()
    page = st.radio("Navigate", ["Executive Summary", "Churn Risk", "ARR Forecast", "Customer Deep Dive"])


# ── Page: Executive Summary ──────────────────────────────────────────────────

if page == "Executive Summary":
    st.title("Executive Revenue Summary")

    targets = load_targets()
    risk = load_risk_table()

    latest = targets.iloc[-1]
    prev = targets.iloc[-2] if len(targets) > 1 else latest

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = latest["actual_arr"] - prev["actual_arr"]
        st.metric("Current ARR", f"${latest['actual_arr']/1e6:.1f}M", f"${delta/1e6:+.1f}M vs last month")

    with col2:
        gap = latest["forecast_arr"] - latest["target_arr"]
        st.metric("Forecast vs Target", f"${gap/1e6:+.1f}M", f"Target: ${latest['target_arr']/1e6:.1f}M")

    with col3:
        if risk is not None:
            arr_at_risk = risk[risk["risk_level"] == "High"]["arr_at_risk"].sum()
            st.metric("ARR at Risk", f"${arr_at_risk/1e6:.1f}M", "High-risk customers")
        else:
            st.metric("ARR at Risk", "Run models first")

    with col4:
        if risk is not None:
            high_count = (risk["risk_level"] == "High").sum()
            st.metric("High-Risk Customers", str(high_count), "Require intervention")
        else:
            st.metric("High-Risk Customers", "N/A")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ARR Trend: Actual vs Target vs Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=targets["month"], y=targets["actual_arr"] / 1e6,
                                  mode="lines+markers", name="Actual", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=targets["month"], y=targets["target_arr"] / 1e6,
                                  mode="lines", name="Target", line=dict(color="#4CAF50", dash="dash")))
        fig.add_trace(go.Scatter(x=targets["month"], y=targets["forecast_arr"] / 1e6,
                                  mode="lines", name="Forecast", line=dict(color="#FF9800", dash="dot")))
        fig.update_layout(yaxis_title="ARR ($M)", xaxis_title="Month", height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if risk is not None:
            st.subheader("Revenue Risk Distribution")
            risk_counts = risk["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Customers"]
            colors = {"High": "#f44336", "Medium": "#FF9800", "Low": "#4CAF50"}
            fig2 = px.pie(risk_counts, values="Customers", names="Risk Level",
                          color="Risk Level", color_discrete_map=colors, height=320)
            fig2.update_layout(margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train models to see risk distribution.")


# ── Page: Churn Risk ─────────────────────────────────────────────────────────

elif page == "Churn Risk":
    st.title("Customer Churn Risk")

    risk = load_risk_table()
    if risk is None:
        st.warning("No risk scores found. Run `python -m src.models.train_churn_model` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", f"{(risk['risk_level']=='High').sum()} customers")
    col2.metric("Medium Risk", f"{(risk['risk_level']=='Medium').sum()} customers")
    col3.metric("Low Risk", f"{(risk['risk_level']=='Low').sum()} customers")

    st.subheader("Top 20 Customers by Revenue Risk")
    top20 = risk.head(20).copy()
    top20["Risk"] = top20["risk_level"].apply(risk_badge) + " " + top20["risk_level"]
    top20["Churn %"] = (top20["churn_probability"] * 100).round(1).astype(str) + "%"
    top20["ARR"] = top20["arr"].apply(lambda x: f"${x:,.0f}")
    top20["ARR at Risk"] = top20["arr_at_risk"].apply(lambda x: f"${x:,.0f}")

    display_cols = ["customer_id", "ARR", "Churn %", "Risk", "ARR at Risk", "nps", "health_score", "last_touch_days"]
    available = [c for c in display_cols if c in top20.columns]
    st.dataframe(top20[available], use_container_width=True, hide_index=True)

    st.subheader("Churn Probability Distribution")
    fig = px.histogram(risk, x="churn_probability", color="risk_level",
                       color_discrete_map={"High": "#f44336", "Medium": "#FF9800", "Low": "#4CAF50"},
                       nbins=30, labels={"churn_probability": "Churn Probability"})
    st.plotly_chart(fig, use_container_width=True)


# ── Page: ARR Forecast ───────────────────────────────────────────────────────

elif page == "ARR Forecast":
    st.title("ARR Forecast")

    targets = load_targets()

    arr_model_path = MODELS_DIR / "arr_forecast_model.pkl"
    arr_features_path = MODELS_DIR / "arr_features.pkl"

    if not arr_model_path.exists():
        st.warning("No ARR model found. Run `python -m src.models.train_arr_forecast` first.")
        targets_display = targets.copy()
        targets_display["Actual ARR"] = targets_display["actual_arr"].apply(lambda x: f"${x/1e6:.2f}M")
        targets_display["Target ARR"] = targets_display["target_arr"].apply(lambda x: f"${x/1e6:.2f}M")
        targets_display["Gap"] = ((targets_display["actual_arr"] - targets_display["target_arr"]) / 1e6).apply(lambda x: f"${x:+.2f}M")
        st.dataframe(targets_display[["month", "Actual ARR", "Target ARR", "Gap"]], use_container_width=True)
    else:
        model = joblib.load(arr_model_path)
        features = joblib.load(arr_features_path)
        arr_df = pd.read_csv(PROCESSED_DIR / "arr_features.csv")
        X = arr_df[features].fillna(0)
        preds = model.predict(X)

        forecast_df = arr_df.copy()
        forecast_df["predicted_arr"] = preds

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df["month"], y=forecast_df["total_arr"] / 1e6,
                                  name="Actual", mode="lines+markers", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=forecast_df["month"], y=forecast_df["predicted_arr"] / 1e6,
                                  name="Predicted", mode="lines", line=dict(color="#FF9800", dash="dot")))
        if "target_arr" in forecast_df:
            fig.add_trace(go.Scatter(x=forecast_df["month"], y=forecast_df["target_arr"] / 1e6,
                                      name="Target", mode="lines", line=dict(color="#4CAF50", dash="dash")))
        fig.update_layout(yaxis_title="ARR ($M)", xaxis_title="Month", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Customer Deep Dive ─────────────────────────────────────────────────

elif page == "Customer Deep Dive":
    st.title("Customer Deep Dive")

    risk = load_risk_table()
    model = load_model()

    if risk is None or model is None:
        st.warning("Train the churn model first.")
        st.stop()

    customer_id = st.selectbox("Select Customer", risk["customer_id"].tolist())
    row = risk[risk["customer_id"] == customer_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Probability", f"{row['churn_probability']:.1%}")
    col2.metric("Risk Level", f"{risk_badge(row['risk_level'])} {row['risk_level']}")
    col3.metric("ARR", f"${row['arr']:,.0f}")
    col4.metric("ARR at Risk", f"${row['arr_at_risk']:,.0f}")

    if "nps" in row:
        col5, col6, col7 = st.columns(3)
        col5.metric("NPS", f"{int(row['nps'])}/10")
        col6.metric("Health Score", f"{row['health_score']:.2f}")
        col7.metric("Days Since CS Touch", f"{int(row['last_touch_days'])}")

    st.subheader("SHAP Risk Drivers")
    with st.spinner("Computing SHAP explanations..."):
        try:
            df = load_churn_features()
            X, _ = get_model_matrix(df)
            customer_rows = df[df["customer_id"] == customer_id] if "customer_id" in df.columns else df
            if len(customer_rows) == 0:
                st.warning("Customer not found in feature matrix.")
            else:
                last_idx = customer_rows.index[-1]
                X_single = X.loc[[last_idx]].fillna(0)
                sv = compute_shap_values(model, X_single)
                drivers = top_drivers_for_customer(sv, X.columns.tolist(), 0, top_n=7)

                for d in drivers:
                    icon = "🔺" if d["shap_value"] > 0 else "🔻"
                    impact = abs(d["shap_value"])
                    st.write(f"{icon} **{d['feature']}**: {d['direction']} (impact: {impact:.4f})")
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
