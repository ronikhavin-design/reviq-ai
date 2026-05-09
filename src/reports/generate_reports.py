"""
Report Generation Module: Phase 4a of the RevIQ AI RAG Copilot.

Reads existing pipeline outputs (risk scores, ARR targets, churn features,
trained models) and writes clean, business-language Markdown reports into
data/reports/markdown/.

These reports are the knowledge base for the Phase 4b RAG retrieval layer.
Each document uses consistent ## headings so it can be chunked by section
for vector embedding.

No LLM, no vector database, and no network calls are made here.
This module is purely deterministic: same inputs, same outputs.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SYNTHETIC_DIR
from src.scenarios.retention_optimizer import (
    DEFAULT_SAVE_PROBABILITY_BY_LEVEL,
    OptimizerInput,
    run_optimizer,
)

MARKDOWN_DIR = REPORTS_DIR / "markdown"
RISK_SCORES_PATH = REPORTS_DIR / "customer_risk_scores.csv"
TARGETS_PATH = SYNTHETIC_DIR / "targets.csv"


# ── Formatting helpers ────────────────────────────────────────────────────────

def _header(title: str, source_files: list[str]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    sources = ", ".join(source_files)
    return (
        f"# {title}\n\n"
        f"*Generated: {date_str}*  \n"
        f"*Source files: {sources}*\n\n"
        "---\n\n"
    )


def _section(heading: str, body: str) -> str:
    return f"## {heading}\n\n{body}\n\n"


# ── Report generators ─────────────────────────────────────────────────────────

def generate_executive_summary(
    targets: pd.DataFrame,
    risk: pd.DataFrame,
) -> str:
    """
    Generate the Executive Revenue Summary report.

    Covers ARR trend, forecast vs. target, and total revenue at risk.
    Intended audience: CEO, CFO, VP Sales.
    """
    latest = targets.iloc[-1]
    prev = targets.iloc[-2] if len(targets) > 1 else latest

    arr_delta = latest["actual_arr"] - prev["actual_arr"]
    forecast_gap = latest["forecast_arr"] - latest["target_arr"]
    total_arr_at_risk = risk["arr_at_risk"].sum()
    high_risk_arr = risk[risk["risk_level"] == "High"]["arr_at_risk"].sum()
    high_risk_count = int((risk["risk_level"] == "High").sum())
    total_customers = len(risk)

    trend_word = "growing" if arr_delta > 0 else "declining"
    gap_word = "above" if forecast_gap >= 0 else "below"

    content = _header(
        "Executive Revenue Summary",
        ["targets.csv", "customer_risk_scores.csv"],
    )

    content += _section("Key Metrics", (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Current ARR | ${latest['actual_arr'] / 1e6:.2f}M |\n"
        f"| ARR vs Prior Month | ${arr_delta / 1e6:+.2f}M |\n"
        f"| ARR Forecast (next period) | ${latest['forecast_arr'] / 1e6:.2f}M |\n"
        f"| Board ARR Target | ${latest['target_arr'] / 1e6:.2f}M |\n"
        f"| Forecast vs Target | ${forecast_gap / 1e6:+.2f}M ({gap_word} target) |\n"
        f"| Total ARR at Risk | ${total_arr_at_risk / 1e6:.2f}M |\n"
        f"| High-Risk ARR | ${high_risk_arr / 1e6:.2f}M |\n"
        f"| High-Risk Customers | {high_risk_count} of {total_customers} |\n"
    ))

    content += _section("Interpretation", (
        f"Revenue is {trend_word} month-over-month. "
        f"The current forecast of ${latest['forecast_arr'] / 1e6:.2f}M sits "
        f"${abs(forecast_gap / 1e6):.2f}M {gap_word} the board-level target of "
        f"${latest['target_arr'] / 1e6:.2f}M. "
        f"The churn model has flagged ${total_arr_at_risk / 1e6:.2f}M in total ARR at risk "
        f"across {total_customers} scored customers. "
        f"Of this, ${high_risk_arr / 1e6:.2f}M is concentrated in {high_risk_count} "
        f"High-risk accounts that require immediate Customer Success intervention."
    ))

    content += _section("Top Risks", (
        f"- **Revenue gap**: Forecast is ${abs(forecast_gap / 1e6):.2f}M {gap_word} target, "
        f"signaling execution or market headwinds that need to be addressed in the current quarter.\n"
        f"- **Churn concentration**: {high_risk_count} High-risk customers hold "
        f"${high_risk_arr / 1e6:.2f}M in ARR at risk, representing "
        f"{high_risk_arr / total_arr_at_risk:.0%} of total portfolio churn exposure.\n"
        f"- **Forecast credibility**: A forecast that is {gap_word} target by more than 5% "
        f"warrants a pipeline coverage review with the Sales and Finance teams."
    ))

    content += _section("Recommended Actions", (
        "1. Authorize Customer Success outreach to all High-risk accounts this week.\n"
        "2. Schedule a pipeline review with Sales and Finance to address the forecast gap.\n"
        "3. Use the Retention Planning dashboard to model campaign ROI before committing budget.\n"
        "4. Track ARR trend weekly and update the board forecast if the gap widens."
    ))

    return content


def generate_churn_risk_summary(
    risk: pd.DataFrame,
    feature_importances: Optional[dict] = None,
) -> str:
    """
    Generate the Churn Risk Summary report.

    Covers risk tier distribution, ARR concentration by tier, and model
    feature importances when available.
    Intended audience: VP Customer Success, CS Operations.
    """
    counts = risk["risk_level"].value_counts()
    arr_by_tier = risk.groupby("risk_level")["arr_at_risk"].sum()
    total_arr_at_risk = float(risk["arr_at_risk"].sum())
    avg_churn_prob = float(risk["churn_probability"].mean())

    high_count = int(counts.get("High", 0))
    medium_count = int(counts.get("Medium", 0))
    low_count = int(counts.get("Low", 0))

    tier_rows = ""
    for level in ["High", "Medium", "Low"]:
        n = int(counts.get(level, 0))
        arr = float(arr_by_tier.get(level, 0.0))
        pct = arr / total_arr_at_risk * 100 if total_arr_at_risk > 0 else 0.0
        tier_rows += f"| {level} | {n} | ${arr / 1e6:.2f}M | {pct:.1f}% |\n"

    content = _header("Churn Risk Summary", ["customer_risk_scores.csv"])

    content += _section("Risk Distribution", (
        "| Risk Level | Customers | ARR at Risk | % of Total Risk |\n"
        "|---|---|---|---|\n"
        f"{tier_rows}\n"
        f"Average churn probability across all customers: **{avg_churn_prob:.1%}**"
    ))

    content += _section("Interpretation", (
        f"The churn model has scored {len(risk)} customers. "
        f"{high_count} are classified as High-risk ({high_count / len(risk):.1%} of the portfolio) "
        f"and hold ${arr_by_tier.get('High', 0.0) / 1e6:.2f}M in ARR at risk. "
        f"{medium_count} Medium-risk customers represent an additional "
        f"${arr_by_tier.get('Medium', 0.0) / 1e6:.2f}M that is still recoverable with early intervention. "
        f"{low_count} Low-risk customers account for the remainder of the portfolio."
    ))

    if feature_importances:
        top_features = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:7]
        feat_rows = "\n".join(
            f"| {i + 1} | {feat.replace('_', ' ').title()} | {imp:.3f} |"
            for i, (feat, imp) in enumerate(top_features)
        )
        content += _section("Top Churn Risk Drivers", (
            "The following features are the strongest predictors of churn in the trained model:\n\n"
            "| Rank | Feature | Importance Score |\n"
            "|---|---|---|\n"
            f"{feat_rows}\n\n"
            "Health score and login activity are the two leading early warning signals. "
            "A customer who reduces platform engagement in the 90 days before renewal "
            "is significantly more likely to churn."
        ))

    content += _section("Top Risks", (
        f"- **High-risk concentration**: {high_count} accounts hold "
        f"{arr_by_tier.get('High', 0.0) / total_arr_at_risk:.0%} of total portfolio ARR at risk.\n"
        f"- **Medium-risk window**: {medium_count} Medium-risk customers are still recoverable "
        f"but will degrade to High if not contacted within the next 30 days.\n"
        f"- **Engagement drop**: Declining login frequency and health score are the top "
        f"leading indicators. These should be monitored on a weekly basis."
    ))

    content += _section("Recommended Actions", (
        "1. Trigger immediate CS outreach for all High-risk customers.\n"
        "2. Set automated health score alerts to catch Medium-risk accounts before they deteriorate.\n"
        "3. Review NPS feedback from High-risk accounts for qualitative intervention signals.\n"
        "4. Use the Retention Optimizer to generate a prioritized call list within budget."
    ))

    return content


def generate_top_customers_at_risk(
    risk: pd.DataFrame,
    top_n: int = 20,
) -> str:
    """
    Generate the Top Customers at Risk report.

    Lists the highest-revenue-risk accounts with their key health indicators.
    Intended audience: CS team, Account Executives.
    """
    top = risk.head(top_n).copy().reset_index(drop=True)

    content = _header(
        f"Top {top_n} Customers at Risk",
        ["customer_risk_scores.csv"],
    )

    content += _section("Overview", (
        f"The following {top_n} customers represent the highest revenue risk in the portfolio, "
        f"ranked by ARR at risk (churn probability multiplied by ARR). "
        f"Total ARR at risk across these {top_n} accounts: "
        f"**${top['arr_at_risk'].sum() / 1e6:.2f}M**."
    ))

    header_row = (
        "| # | Customer | ARR | Churn Prob | Risk Level | ARR at Risk"
        " | NPS | Health Score | Days Since CS Touch |"
    )
    sep_row = "|---|---|---|---|---|---|---|---|---|"
    data_rows = []
    for i, row in top.iterrows():
        nps_str = str(int(row["nps"])) if "nps" in row.index and pd.notna(row["nps"]) else "N/A"
        hs_str = f"{row['health_score']:.2f}" if "health_score" in row.index and pd.notna(row["health_score"]) else "N/A"
        touch_str = str(int(row["last_touch_days"])) if "last_touch_days" in row.index and pd.notna(row["last_touch_days"]) else "N/A"
        data_rows.append(
            f"| {i + 1} | {row['customer_id']} | ${row['arr']:,.0f}"
            f" | {row['churn_probability']:.1%} | {row['risk_level']}"
            f" | ${row['arr_at_risk']:,.0f} | {nps_str} | {hs_str} | {touch_str} |"
        )

    content += _section("Customer Risk Table", (
        f"{header_row}\n{sep_row}\n" + "\n".join(data_rows)
    ))

    high_in_top = int((top["risk_level"] == "High").sum())
    med_in_top = int((top["risk_level"] == "Medium").sum())

    content += _section("Key Observations", (
        f"- {high_in_top} of the top {top_n} accounts are classified as High-risk.\n"
        f"- {med_in_top} are Medium-risk and still have a higher save probability with early action.\n"
        f"- Customers with health scores below 0.40 and NPS below 5 are the highest priority "
        f"for immediate escalation.\n"
        f"- Days since last CS touch is a leading indicator: accounts with no recent contact "
        f"show significantly elevated churn probability."
    ))

    content += _section("Recommended Actions", (
        "1. Assign a dedicated CSM to each High-risk account on this list immediately.\n"
        "2. Schedule executive business reviews for accounts with ARR above $500K at risk.\n"
        "3. For accounts with NPS below 5, prioritize a product feedback call before the renewal date.\n"
        "4. Export this list to the CS platform as the campaign call list."
    ))

    return content


def generate_retention_planning_summary(risk: pd.DataFrame) -> str:
    """
    Generate the Retention Planning Summary report.

    Models three budget scenarios using the Retention Optimizer and reports
    expected ARR saved, ROI, and constraints for each.
    Intended audience: VP Customer Success, CFO.
    """
    scenarios = [
        ("Conservative", 25_000.0, 75.0),
        ("Standard", 50_000.0, 150.0),
        ("Aggressive", 100_000.0, 300.0),
    ]

    high_save_prob = DEFAULT_SAVE_PROBABILITY_BY_LEVEL["High"]
    med_save_prob = DEFAULT_SAVE_PROBABILITY_BY_LEVEL["Medium"]

    content = _header("Retention Planning Summary", ["customer_risk_scores.csv"])

    content += _section("Overview", (
        "This report models three retention campaign scenarios to show expected ARR saved "
        "at different budget and CS capacity levels. The Retention Optimizer selects customers "
        "greedily by expected ROI (ARR saved per dollar spent), using per-tier save probabilities "
        f"of {high_save_prob:.0%} for High-risk and {med_save_prob:.0%} for Medium-risk accounts. "
        "All scenarios assume a $1,000 average discount and 2.0 CS hours per customer."
    ))

    scenario_rows = ""
    scenario_results = []
    for name, budget, hours in scenarios:
        opt_in = OptimizerInput(
            total_discount_budget=budget,
            total_cs_hours=hours,
            target_risk_levels=["High", "Medium"],
            avg_discount_per_customer=1_000.0,
            avg_cs_hours_per_customer=2.0,
        )
        result = run_optimizer(opt_in, risk_scores=risk)
        roi_str = f"{result.expected_roi:.1f}x" if result.total_cost > 0 else "N/A"
        scenario_rows += (
            f"| {name} | ${budget:,.0f} | {hours:.0f}h | "
            f"{result.n_selected} | ${result.total_expected_saved_arr / 1e6:.2f}M | {roi_str} |\n"
        )
        scenario_results.append((name, result))

    content += _section("Scenario Comparison", (
        "| Scenario | Budget | CS Hours | Customers Selected | Expected ARR Saved | Expected ROI |\n"
        "|---|---|---|---|---|---|\n"
        f"{scenario_rows}"
    ))

    _, std = scenario_results[1]
    content += _section("Standard Scenario Detail", (
        f"With a $50,000 budget and 150 CS hours, the optimizer selects "
        f"**{std.n_selected} customers** for outreach, projecting "
        f"**${std.total_expected_saved_arr / 1e6:.2f}M in ARR saved**.\n\n"
        f"- Customers eligible (High and Medium risk): {std.n_eligible}\n"
        f"- Customers excluded: {std.n_excluded}\n"
        f"- Budget remaining after campaign: ${std.remaining_budget:,.0f}\n"
        f"- CS hours remaining after campaign: {std.remaining_cs_hours:.1f}h"
    ))

    content += _section("Top Risks", (
        "- **Save probability uncertainty**: Per-tier save probabilities are model defaults. "
        "Actual CS win rates may vary. Calibrate with historical outcome data as it accumulates.\n"
        "- **Budget concentration**: Focusing the entire budget on the highest-ROI accounts "
        "maximizes expected return but leaves some Medium-risk accounts unaddressed.\n"
        "- **Renewal timing**: CS interventions are most effective before the final 30 days "
        "of a renewal window. Accounts within 30 days of renewal should be flagged as urgent."
    ))

    content += _section("Recommended Actions", (
        "1. Approve the Standard scenario ($50K, 150 hours) as the default campaign starting point.\n"
        "2. Use the Retention Optimizer on the dashboard to generate the exact prioritized call list.\n"
        "3. Download the call list CSV and load it into the CS platform.\n"
        "4. Record intervention outcomes per customer to calibrate save probabilities over time.\n"
        "5. Re-run the optimizer monthly as churn risk scores are refreshed."
    ))

    return content


def generate_model_performance_summary(
    churn_model_info: Optional[dict] = None,
    arr_model_info: Optional[dict] = None,
) -> str:
    """
    Generate the Model Performance Summary report.

    Documents model types, top predictive features, and error metrics.
    Intended audience: Data team, Engineering, Finance (for forecast accuracy).
    """
    content = _header(
        "Model Performance Summary",
        ["churn_model.pkl", "arr_forecast_model.pkl", "churn_features.pkl", "arr_features.pkl"],
    )

    content += _section("Overview", (
        "This report documents the two predictive models powering the RevIQ AI platform: "
        "the churn classifier and the ARR forecaster. Both models were trained on SaaS customer "
        "data and produce the scores that drive the Churn Risk and Retention Planning dashboards."
    ))

    if churn_model_info:
        model_type = churn_model_info.get("model_type", "Unknown")
        top_features = churn_model_info.get("top_features", {})
        feat_rows = ""
        for i, (feat, imp) in enumerate(list(top_features.items())[:7]):
            feat_rows += f"| {i + 1} | {feat.replace('_', ' ').title()} | {imp:.4f} |\n"

        churn_body = (
            f"**Model type**: {model_type}  \n"
            f"**Task**: Binary classification (churn vs. retained)  \n"
            f"**Output**: Churn probability (0.0 to 1.0) and risk tier (High, Medium, Low)  \n\n"
        )
        if feat_rows:
            churn_body += (
                "**Top predictive features**:\n\n"
                "| Rank | Feature | Importance Score |\n"
                "|---|---|---|\n"
                f"{feat_rows}\n"
                "Health score and login activity are the two strongest early warning signals. "
                "A customer who reduces platform engagement in the 90 days before renewal "
                "is significantly more likely to churn."
            )
        else:
            churn_body += "Feature importance data not available for this model type."
        content += _section("Churn Model", churn_body)

    if arr_model_info:
        model_type = arr_model_info.get("model_type", "Unknown")
        mae = arr_model_info.get("mae", None)
        features = arr_model_info.get("features", [])
        mae_str = f"${mae / 1e6:.3f}M" if mae is not None else "Not computed"
        feat_list = (
            ", ".join(f.replace("_", " ") for f in features)
            if features else "Not available"
        )
        content += _section("ARR Forecast Model", (
            f"**Model type**: {model_type}  \n"
            f"**Task**: Monthly ARR regression  \n"
            f"**Mean Absolute Error (in-sample)**: {mae_str}  \n"
            f"**Input features**: {feat_list}  \n\n"
            "The forecast model uses lagged ARR values and rolling averages to project "
            "next-month revenue. A MAE below $0.1M on a portfolio of this size indicates "
            "good short-term accuracy for planning purposes."
        ))

    content += _section("Model Limitations", (
        "- Both models were trained on synthetic data. Real-world performance should be "
        "benchmarked against live customer outcomes before relying on the scores for "
        "large budget decisions.\n"
        "- The churn model does not incorporate external signals such as competitor activity "
        "or macroeconomic conditions.\n"
        "- Save probabilities used in the Retention Optimizer are assumed defaults (0.20 for "
        "High, 0.35 for Medium, 0.50 for Low). These are not calibrated from historical "
        "intervention outcomes."
    ))

    content += _section("Recommended Actions", (
        "1. Establish a ground-truth tracking process: log actual churn outcomes per customer "
        "to enable model retraining on live data.\n"
        "2. Set a retraining schedule (monthly or quarterly) as new outcome data accumulates.\n"
        "3. Add prediction confidence intervals to the ARR forecast to communicate uncertainty "
        "to the Finance team.\n"
        "4. Validate churn model predictions against CRM renewal data on a rolling 90-day basis."
    ))

    return content


# ── File writing ──────────────────────────────────────────────────────────────

def write_report(content: str, filename: str, output_dir: Path) -> Path:
    """Write a report string to a Markdown file and return the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


# ── Model metadata loaders ────────────────────────────────────────────────────

def load_churn_model_info() -> Optional[dict]:
    """Load churn model type and feature importances from disk."""
    import joblib

    model_path = MODELS_DIR / "churn_model.pkl"
    features_path = MODELS_DIR / "churn_features.pkl"
    if not model_path.exists() or not features_path.exists():
        return None
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    top_features: dict = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(features, model.feature_importances_))
        top_features = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        )
    return {"model_type": type(model).__name__, "top_features": top_features}


def load_arr_model_info() -> Optional[dict]:
    """Load ARR forecast model type, features, and in-sample MAE from disk."""
    import joblib

    model_path = MODELS_DIR / "arr_forecast_model.pkl"
    features_path = MODELS_DIR / "arr_features.pkl"
    arr_features_csv = PROCESSED_DIR / "arr_features.csv"
    if not model_path.exists() or not features_path.exists():
        return None
    model = joblib.load(model_path)
    features: list = joblib.load(features_path)
    mae: Optional[float] = None
    if arr_features_csv.exists():
        arr_df = pd.read_csv(arr_features_csv)
        if all(f in arr_df.columns for f in features) and "total_arr" in arr_df.columns:
            preds = model.predict(arr_df[features].fillna(0))
            mae = float(np.mean(np.abs(preds - arr_df["total_arr"].values)))
    return {"model_type": type(model).__name__, "features": features, "mae": mae}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def generate_all_reports(
    risk_scores: Optional[pd.DataFrame] = None,
    targets: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    churn_model_info: Optional[dict] = "load_from_disk",
    arr_model_info: Optional[dict] = "load_from_disk",
) -> dict[str, Path]:
    """
    Generate all Markdown reports and write them to output_dir.

    Pass DataFrames directly to skip file I/O in tests.
    Pass churn_model_info / arr_model_info dicts directly to skip model loading in tests.
    Pass None for either model info dict to omit model_performance_summary.md.
    Leave as default ("load_from_disk") to load models from the standard paths.

    Returns a dict mapping filename to the Path written.
    """
    if risk_scores is None:
        risk_scores = pd.read_csv(RISK_SCORES_PATH)
    if targets is None:
        targets = pd.read_csv(TARGETS_PATH)
    if output_dir is None:
        output_dir = MARKDOWN_DIR

    if churn_model_info == "load_from_disk":
        churn_model_info = load_churn_model_info()
    if arr_model_info == "load_from_disk":
        arr_model_info = load_arr_model_info()

    fi = churn_model_info.get("top_features") if churn_model_info else None

    reports: dict[str, str] = {
        "executive_summary.md": generate_executive_summary(targets, risk_scores),
        "churn_risk_summary.md": generate_churn_risk_summary(risk_scores, fi),
        "top_customers_at_risk.md": generate_top_customers_at_risk(risk_scores, top_n=20),
        "retention_planning_summary.md": generate_retention_planning_summary(risk_scores),
    }

    if churn_model_info is not None or arr_model_info is not None:
        reports["model_performance_summary.md"] = generate_model_performance_summary(
            churn_model_info, arr_model_info
        )

    written: dict[str, Path] = {}
    for filename, content in reports.items():
        written[filename] = write_report(content, filename, output_dir)

    return written


if __name__ == "__main__":
    from loguru import logger

    logger.info("Generating Markdown reports for RAG knowledge base...")
    written = generate_all_reports()
    for name, path in written.items():
        logger.info(f"  Written: {path}")
    logger.info(f"Done. {len(written)} reports generated in {MARKDOWN_DIR}")
