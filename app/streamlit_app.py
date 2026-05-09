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
from src.scenarios.scenario_simulator import ScenarioInput, run_scenario
from src.scenarios.retention_optimizer import OptimizerInput, run_optimizer

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
    page = st.radio("Navigate", ["Executive Summary", "Churn Risk", "ARR Forecast", "Customer Deep Dive", "Retention Planning"])


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


# ── Page: Retention Planning ─────────────────────────────────────────────────

elif page == "Retention Planning":
    st.title("Retention Planning")

    risk = load_risk_table()
    if risk is None:
        st.warning("No risk scores found. Run `python -m src.models.train_churn_model` first.")
        st.stop()

    st.info(
        "Use the Scenario Simulator to model campaign-level ARR impact, or the Retention Optimizer "
        "to generate a ranked per-customer call list. Both tools read from the same churn risk scores."
    )

    with st.expander("Campaign Settings", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Resources**")
            discount_budget = st.number_input(
                "Total Discount Budget ($)", min_value=0, value=50_000, step=1_000,
            )
            cs_hours = st.number_input(
                "CS Team Hours Available", min_value=0.0, value=150.0, step=10.0,
            )
            avg_discount = st.number_input(
                "Avg Discount per Customer ($)", min_value=0, value=1_000, step=100,
            )
            avg_hours = st.number_input(
                "Avg CS Hours per Customer", min_value=0.1, value=2.0, step=0.5,
            )
        with col_b:
            st.markdown("**Campaign Assumptions**")
            target_levels = st.multiselect(
                "Target Risk Levels",
                options=["High", "Medium", "Low"],
                default=["High", "Medium"],
            )
            retention_rate = st.slider(
                "Retention Success Rate (binary model)", 0.0, 1.0, 0.30, 0.05,
            )
            churn_reduction = st.slider(
                "Churn Reduction Rate (continuous model)", 0.0, 1.0, 0.40, 0.05,
            )
            use_min_roi = st.checkbox("Apply Minimum ROI Filter (Optimizer only)")
            minimum_roi = None
            if use_min_roi:
                minimum_roi = st.number_input(
                    "Minimum ROI (x)", min_value=0.1, value=2.0, step=0.5,
                )

    tab_sim, tab_opt = st.tabs(["Scenario Simulator", "Retention Optimizer"])

    # ── Tab: Scenario Simulator ──────────────────────────────────────────────

    with tab_sim:
        sim_inputs = ScenarioInput(
            retention_success_rate=retention_rate,
            churn_reduction_rate=churn_reduction,
            cs_capacity_hours=float(cs_hours),
            avg_hours_per_customer=float(avg_hours),
            discount_budget=float(discount_budget),
            avg_discount_per_customer=float(avg_discount),
            target_risk_levels=target_levels if target_levels else ["High", "Medium"],
        )
        try:
            sim = run_scenario(sim_inputs, risk_scores=risk)
        except Exception as e:
            st.error(f"Scenario simulation failed: {e}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eligible Customers", sim.customers_eligible)
        c2.metric("Targeted Customers", sim.customers_targeted)
        c3.metric("Excluded", sim.customers_excluded)
        c4.metric("Binding Constraint", sim.binding_constraint.replace("_", " ").title())

        c5, c6, c7 = st.columns(3)
        c5.metric("ARR at Risk (before)", f"${sim.original_arr_at_risk/1e6:.2f}M")
        c6.metric("Expected ARR Saved (binary)", f"${sim.expected_arr_saved/1e6:.2f}M",
                  f"ROI {sim.roi_multiple:.1f}x")
        c7.metric("Expected ARR Saved (continuous)", f"${sim.expected_arr_saved_by_reduction/1e6:.2f}M",
                  f"ROI {sim.roi_multiple_by_reduction:.1f}x")

        st.subheader("Binary vs Continuous ARR Impact")
        bar_data = pd.DataFrame({
            "Model": ["Binary (retention_success_rate)", "Continuous (churn_reduction_rate)"],
            "ARR Saved ($M)": [
                sim.expected_arr_saved / 1e6,
                sim.expected_arr_saved_by_reduction / 1e6,
            ],
        })
        fig_bar = px.bar(
            bar_data, x="ARR Saved ($M)", y="Model", orientation="h",
            color="Model",
            color_discrete_sequence=["#2196F3", "#FF9800"],
            height=220,
        )
        fig_bar.update_layout(
            showlegend=False, margin=dict(t=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if sim.impact_by_risk_level:
            st.subheader("Impact by Risk Level")
            impact_rows = [
                {
                    "Risk Level": r.risk_level,
                    "Targeted": r.customers_targeted,
                    "Saved (binary)": r.customers_saved,
                    "ARR Saved ($)": f"${r.arr_saved:,.0f}",
                }
                for r in sim.impact_by_risk_level
            ]
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

    # ── Tab: Retention Optimizer ─────────────────────────────────────────────

    with tab_opt:
        opt_inputs = OptimizerInput(
            total_discount_budget=float(discount_budget),
            total_cs_hours=float(cs_hours),
            target_risk_levels=target_levels if target_levels else ["High", "Medium"],
            minimum_roi=minimum_roi,
            avg_discount_per_customer=float(avg_discount),
            avg_cs_hours_per_customer=float(avg_hours),
        )
        try:
            opt = run_optimizer(opt_inputs, risk_scores=risk)
        except Exception as e:
            st.error(f"Optimizer failed: {e}")
            st.stop()

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Eligible Customers", opt.n_eligible)
        o2.metric("Selected", opt.n_selected)
        o3.metric("Excluded", opt.n_excluded)
        o4.metric("Expected ROI", f"{opt.expected_roi:.1f}x")

        o5, o6, o7 = st.columns(3)
        o5.metric("Expected ARR Saved", f"${opt.total_expected_saved_arr/1e6:.2f}M")
        o6.metric("Remaining Budget", f"${opt.remaining_budget:,.0f}")
        o7.metric("Remaining CS Hours", f"{opt.remaining_cs_hours:.1f}h")

        if not opt.selected_customers.empty:
            st.subheader("Top Selected Customers by Expected ARR Saved")
            top_n = min(20, len(opt.selected_customers))
            chart_df = opt.selected_customers.head(top_n).copy()
            fig_opt = px.bar(
                chart_df,
                x="expected_saved_arr",
                y="customer_id",
                orientation="h",
                color="risk_level",
                color_discrete_map={"High": "#f44336", "Medium": "#FF9800", "Low": "#4CAF50"},
                labels={"expected_saved_arr": "Expected ARR Saved ($)", "customer_id": "Customer"},
                height=max(300, top_n * 22),
            )
            fig_opt.update_layout(
                margin=dict(t=10),
                yaxis=dict(autorange="reversed"),
                showlegend=True,
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            st.subheader("Selected Customer Call List")
            call_list = opt.selected_customers.copy()
            display_cols = [
                "customer_id", "risk_level", "arr", "arr_at_risk",
                "estimated_save_probability", "expected_saved_arr", "expected_roi", "selection_reason",
            ]
            available = [c for c in display_cols if c in call_list.columns]
            call_list["arr"] = call_list["arr"].apply(lambda x: f"${x:,.0f}")
            call_list["arr_at_risk"] = call_list["arr_at_risk"].apply(lambda x: f"${x:,.0f}")
            call_list["expected_saved_arr"] = call_list["expected_saved_arr"].apply(lambda x: f"${x:,.0f}")
            call_list["estimated_save_probability"] = call_list["estimated_save_probability"].apply(
                lambda x: f"{x:.0%}"
            )
            call_list["expected_roi"] = call_list["expected_roi"].apply(lambda x: f"{x:.1f}x")
            st.dataframe(call_list[available], use_container_width=True, hide_index=True)

            csv_bytes = opt.selected_customers.to_csv(index=False).encode()
            st.download_button(
                "Download Call List (CSV)", csv_bytes, "cs_call_list.csv", "text/csv",
            )

        if opt.exclusion_counts:
            st.subheader("Exclusion Summary")
            excl_df = pd.DataFrame([
                {"Reason": k.replace("_", " ").title(), "Customers": v}
                for k, v in opt.exclusion_counts.items()
            ])
            st.dataframe(excl_df, use_container_width=True, hide_index=True)
