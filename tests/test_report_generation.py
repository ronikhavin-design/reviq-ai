"""
Tests for the Report Generation module (Phase 4a RAG Copilot).

All report generator functions accept DataFrames directly, so no production
file system access is required. File writing tests use pytest's tmp_path fixture.
"""

import pandas as pd
import pytest
from pathlib import Path

from src.reports.generate_reports import (
    generate_churn_risk_summary,
    generate_executive_summary,
    generate_model_performance_summary,
    generate_retention_planning_summary,
    generate_top_customers_at_risk,
    generate_all_reports,
    write_report,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_risk_scores(
    n_high: int = 5,
    n_medium: int = 10,
    n_low: int = 20,
    high_arr: float = 100_000.0,
    medium_arr: float = 50_000.0,
    low_arr: float = 20_000.0,
) -> pd.DataFrame:
    rows = []
    cid = 1
    for level, count, arr, churn_prob in [
        ("High",   n_high,   high_arr,   0.70),
        ("Medium", n_medium, medium_arr, 0.35),
        ("Low",    n_low,    low_arr,    0.10),
    ]:
        for _ in range(count):
            rows.append({
                "customer_id":        f"C{cid:04d}",
                "arr":                arr,
                "mrr":                arr / 12,
                "churn_probability":  churn_prob,
                "arr_at_risk":        arr * churn_prob,
                "risk_level":         level,
                "revenue_risk_score": churn_prob,
                "nps":                5,
                "health_score":       0.50,
                "last_touch_days":    14,
                "months_to_renewal":  6,
            })
            cid += 1
    return pd.DataFrame(rows).sort_values("arr_at_risk", ascending=False).reset_index(drop=True)


def make_targets(n_months: int = 12) -> pd.DataFrame:
    rows = []
    base = 10_000_000.0
    for i in range(1, n_months + 1):
        rows.append({
            "month":        i,
            "target_arr":   base + i * 500_000,
            "actual_arr":   base + i * 450_000,
            "forecast_arr": base + i * 420_000,
        })
    return pd.DataFrame(rows)


def make_churn_model_info() -> dict:
    return {
        "model_type": "XGBClassifier",
        "top_features": {
            "health_score": 0.142,
            "logins_roll3m": 0.135,
            "health_score_lag1": 0.062,
            "segment_mid_market": 0.054,
            "feature_adoption_roll3m": 0.052,
            "logins": 0.041,
            "plan_professional": 0.038,
        },
    }


def make_arr_model_info() -> dict:
    return {
        "model_type": "XGBRegressor",
        "features": ["arr_lag1", "arr_lag2", "arr_roll3m", "arr_growth_3m"],
        "mae": 53_000.0,
    }


# ── Executive Summary tests ───────────────────────────────────────────────────

def test_executive_summary_returns_string():
    result = generate_executive_summary(make_targets(), make_risk_scores())
    assert isinstance(result, str)
    assert len(result) > 200


def test_executive_summary_contains_required_sections():
    result = generate_executive_summary(make_targets(), make_risk_scores())
    for section in ["Key Metrics", "Interpretation", "Top Risks", "Recommended Actions"]:
        assert f"## {section}" in result, f"Missing section: {section}"


def test_executive_summary_contains_source_files():
    result = generate_executive_summary(make_targets(), make_risk_scores())
    assert "targets.csv" in result
    assert "customer_risk_scores.csv" in result


def test_executive_summary_contains_arr_figures():
    result = generate_executive_summary(make_targets(), make_risk_scores())
    assert "$" in result
    assert "ARR" in result


def test_executive_summary_above_target_wording():
    targets = make_targets()
    targets.loc[targets.index[-1], "forecast_arr"] = 999_000_000.0
    result = generate_executive_summary(targets, make_risk_scores())
    assert "above" in result


def test_executive_summary_below_target_wording():
    targets = make_targets()
    targets.loc[targets.index[-1], "forecast_arr"] = 1.0
    result = generate_executive_summary(targets, make_risk_scores())
    assert "below" in result


# ── Churn Risk Summary tests ──────────────────────────────────────────────────

def test_churn_risk_summary_returns_string():
    result = generate_churn_risk_summary(make_risk_scores())
    assert isinstance(result, str)
    assert len(result) > 200


def test_churn_risk_summary_contains_required_sections():
    result = generate_churn_risk_summary(make_risk_scores())
    for section in ["Risk Distribution", "Interpretation", "Top Risks", "Recommended Actions"]:
        assert f"## {section}" in result, f"Missing section: {section}"


def test_churn_risk_summary_contains_all_tiers():
    result = generate_churn_risk_summary(make_risk_scores())
    for level in ["High", "Medium", "Low"]:
        assert level in result


def test_churn_risk_summary_includes_feature_importances_when_provided():
    fi = {"health_score": 0.14, "logins_roll3m": 0.13}
    result = generate_churn_risk_summary(make_risk_scores(), feature_importances=fi)
    assert "Top Churn Risk Drivers" in result
    assert "Health Score" in result


def test_churn_risk_summary_omits_drivers_section_when_no_importances():
    result = generate_churn_risk_summary(make_risk_scores(), feature_importances=None)
    assert "Top Churn Risk Drivers" not in result


def test_churn_risk_summary_source_file_present():
    result = generate_churn_risk_summary(make_risk_scores())
    assert "customer_risk_scores.csv" in result


# ── Top Customers at Risk tests ───────────────────────────────────────────────

def test_top_customers_returns_string():
    result = generate_top_customers_at_risk(make_risk_scores())
    assert isinstance(result, str)
    assert len(result) > 200


def test_top_customers_contains_required_sections():
    result = generate_top_customers_at_risk(make_risk_scores())
    for section in ["Overview", "Customer Risk Table", "Key Observations", "Recommended Actions"]:
        assert f"## {section}" in result, f"Missing section: {section}"


def test_top_customers_respects_top_n_parameter():
    risk = make_risk_scores(n_high=10, n_medium=20, n_low=30)
    result = generate_top_customers_at_risk(risk, top_n=5)
    assert "Top 5 Customers at Risk" in result
    # All selected customers should be in the table
    top5_ids = risk.head(5)["customer_id"].tolist()
    for cid in top5_ids:
        assert cid in result


def test_top_customers_does_not_exceed_top_n():
    risk = make_risk_scores(n_high=3, n_medium=5, n_low=5)
    result = generate_top_customers_at_risk(risk, top_n=20)
    # Should not error even if fewer than top_n customers exist
    assert isinstance(result, str)


def test_top_customers_source_file_present():
    result = generate_top_customers_at_risk(make_risk_scores())
    assert "customer_risk_scores.csv" in result


# ── Retention Planning Summary tests ─────────────────────────────────────────

def test_retention_planning_summary_returns_string():
    result = generate_retention_planning_summary(make_risk_scores())
    assert isinstance(result, str)
    assert len(result) > 200


def test_retention_planning_summary_contains_required_sections():
    result = generate_retention_planning_summary(make_risk_scores())
    for section in ["Overview", "Scenario Comparison", "Standard Scenario Detail",
                    "Top Risks", "Recommended Actions"]:
        assert f"## {section}" in result, f"Missing section: {section}"


def test_retention_planning_summary_contains_three_scenarios():
    result = generate_retention_planning_summary(make_risk_scores())
    for scenario in ["Conservative", "Standard", "Aggressive"]:
        assert scenario in result


def test_retention_planning_summary_source_file_present():
    result = generate_retention_planning_summary(make_risk_scores())
    assert "customer_risk_scores.csv" in result


def test_retention_planning_summary_contains_budget_figures():
    result = generate_retention_planning_summary(make_risk_scores())
    assert "$50,000" in result or "$50000" in result or "50,000" in result


# ── Model Performance Summary tests ──────────────────────────────────────────

def test_model_performance_summary_returns_string():
    result = generate_model_performance_summary(make_churn_model_info(), make_arr_model_info())
    assert isinstance(result, str)
    assert len(result) > 200


def test_model_performance_summary_contains_required_sections():
    result = generate_model_performance_summary(make_churn_model_info(), make_arr_model_info())
    for section in ["Overview", "Churn Model", "ARR Forecast Model",
                    "Model Limitations", "Recommended Actions"]:
        assert f"## {section}" in result, f"Missing section: {section}"


def test_model_performance_summary_includes_model_types():
    result = generate_model_performance_summary(make_churn_model_info(), make_arr_model_info())
    assert "XGBClassifier" in result
    assert "XGBRegressor" in result


def test_model_performance_summary_includes_mae():
    result = generate_model_performance_summary(None, make_arr_model_info())
    assert "MAE" in result or "Mean Absolute Error" in result


def test_model_performance_summary_with_no_info_still_returns_string():
    result = generate_model_performance_summary(None, None)
    assert isinstance(result, str)
    assert len(result) > 100


def test_model_performance_summary_source_files_present():
    result = generate_model_performance_summary(make_churn_model_info(), make_arr_model_info())
    assert "churn_model.pkl" in result
    assert "arr_forecast_model.pkl" in result


# ── write_report tests ────────────────────────────────────────────────────────

def test_write_report_creates_file(tmp_path):
    content = "# Test Report\n\nSome content."
    path = write_report(content, "test.md", tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == content


def test_write_report_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b" / "markdown"
    content = "# Report"
    path = write_report(content, "report.md", nested)
    assert path.exists()


def test_write_report_returns_correct_path(tmp_path):
    path = write_report("content", "my_report.md", tmp_path)
    assert path.name == "my_report.md"
    assert path.parent == tmp_path


# ── generate_all_reports tests ────────────────────────────────────────────────

CORE_REPORTS = [
    "executive_summary.md",
    "churn_risk_summary.md",
    "top_customers_at_risk.md",
    "retention_planning_summary.md",
]


def test_generate_all_reports_creates_core_files(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=None,
        arr_model_info=None,
    )
    for name in CORE_REPORTS:
        assert name in written, f"Missing report: {name}"
        assert written[name].exists(), f"File not written: {name}"


def test_generate_all_reports_creates_model_performance_when_info_provided(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=make_churn_model_info(),
        arr_model_info=make_arr_model_info(),
    )
    assert "model_performance_summary.md" in written
    assert written["model_performance_summary.md"].exists()


def test_generate_all_reports_omits_model_performance_when_info_is_none(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=None,
        arr_model_info=None,
    )
    assert "model_performance_summary.md" not in written


def test_generate_all_reports_files_are_nonempty(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=None,
        arr_model_info=None,
    )
    for name, path in written.items():
        content = path.read_text(encoding="utf-8")
        assert len(content) > 200, f"Report too short: {name}"


def test_generate_all_reports_all_files_contain_source_section(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=None,
        arr_model_info=None,
    )
    for name, path in written.items():
        content = path.read_text(encoding="utf-8")
        assert "Source files:" in content, f"No source files section in: {name}"


def test_generate_all_reports_no_em_dashes(tmp_path):
    written = generate_all_reports(
        risk_scores=make_risk_scores(),
        targets=make_targets(),
        output_dir=tmp_path,
        churn_model_info=make_churn_model_info(),
        arr_model_info=make_arr_model_info(),
    )
    for name, path in written.items():
        content = path.read_text(encoding="utf-8")
        assert "—" not in content, f"Em dash found in: {name}"
