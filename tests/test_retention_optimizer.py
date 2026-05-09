"""
Tests for the Retention Budget Optimizer.

All tests use synthetic DataFrames so no file system access is required.
Each test documents the business rule or constraint being verified.
"""

import pandas as pd
import pytest

from src.scenarios.retention_optimizer import (
    OptimizerInput,
    OptimizerResult,
    run_optimizer,
    format_optimizer_summary,
    DEFAULT_SAVE_PROBABILITY_BY_LEVEL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_risk_scores(
    n_high: int = 10,
    n_medium: int = 20,
    n_low: int = 0,
    high_arr: float = 100_000,
    medium_arr: float = 50_000,
    low_arr: float = 20_000,
) -> pd.DataFrame:
    """Build a synthetic risk scores DataFrame.

    Uses fixed churn probabilities per tier so arr_at_risk values are
    deterministic and easy to reason about in assertions.
    """
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
                "churn_probability":  churn_prob,
                "arr_at_risk":        arr * churn_prob,
                "risk_level":         level,
                "revenue_risk_score": churn_prob,
            })
            cid += 1
    return pd.DataFrame(rows)


def default_inputs(**overrides) -> OptimizerInput:
    """Return an OptimizerInput with sensible defaults, with optional overrides."""
    inputs = OptimizerInput(
        total_discount_budget=50_000.0,
        total_cs_hours=150.0,
        target_risk_levels=["High", "Medium"],
        max_customers=None,
        minimum_roi=None,
        avg_discount_per_customer=1_000.0,
        avg_cs_hours_per_customer=2.0,
    )
    for k, v in overrides.items():
        setattr(inputs, k, v)
    return inputs


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_eligible_filtered_by_risk_level():
    """Only customers in target_risk_levels are considered for selection.
    Low-risk customers must be excluded even when resources are plentiful."""
    scores = make_risk_scores(n_high=5, n_medium=5, n_low=20)
    result = run_optimizer(
        default_inputs(
            total_discount_budget=1_000_000.0,
            total_cs_hours=10_000.0,
            target_risk_levels=["High", "Medium"],
        ),
        risk_scores=scores,
    )
    assert result.n_eligible == 10  # only High + Medium
    if not result.selected_customers.empty:
        assert set(result.selected_customers["risk_level"].tolist()).issubset({"High", "Medium"})


def test_budget_constraint_respected():
    """Total cost of selected customers must never exceed total_discount_budget."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        total_discount_budget=5_000.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.total_cost <= inputs.total_discount_budget


def test_cs_hours_constraint_respected():
    """Total CS hours used must never exceed total_cs_hours."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        total_cs_hours=6.0,
        avg_cs_hours_per_customer=2.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.total_cs_hours_used <= inputs.total_cs_hours


def test_max_customers_constraint_respected():
    """n_selected must never exceed max_customers."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        total_discount_budget=1_000_000.0,
        total_cs_hours=10_000.0,
        max_customers=4,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.n_selected <= 4


def test_minimum_roi_filter_excludes_low_roi_customers():
    """Customers below minimum_roi are excluded even if budget remains.

    With default save probs: High ROI = 14x, Medium ROI = 6.125x.
    Setting minimum_roi=10.0 should exclude all Medium customers."""
    scores = make_risk_scores(n_high=5, n_medium=5)
    inputs = default_inputs(
        total_discount_budget=1_000_000.0,
        total_cs_hours=10_000.0,
        minimum_roi=10.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    if not result.selected_customers.empty:
        assert (result.selected_customers["expected_roi"] >= 10.0).all()
    assert result.exclusion_counts.get("roi_below_minimum", 0) == 5  # all Medium excluded


def test_customers_selected_by_roi_order():
    """Higher ROI customers are selected before lower ROI customers.

    High-tier save prob is 0.20, Medium is 0.35.
    High arr_at_risk = 70,000: ROI = 70,000 * 0.20 / 1,000 = 14.0x
    Medium arr_at_risk = 17,500: ROI = 17,500 * 0.35 / 1,000 = 6.125x
    With a 3-customer budget, all selected should be High."""
    scores = make_risk_scores(n_high=5, n_medium=10)
    inputs = default_inputs(
        total_discount_budget=3_000.0,
        total_cs_hours=10_000.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.n_selected == 3
    assert all(r == "High" for r in result.selected_customers["risk_level"].tolist())


def test_selected_plus_excluded_equals_eligible():
    """Every eligible customer must appear in exactly one of selected or excluded.
    No customers should be silently dropped."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_optimizer(default_inputs(), risk_scores=scores)
    assert result.n_selected + result.n_excluded == result.n_eligible


def test_expected_roi_formula():
    """expected_roi = total_expected_saved_arr / total_cost."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_optimizer(default_inputs(), risk_scores=scores)
    if result.total_cost > 0:
        expected = result.total_expected_saved_arr / result.total_cost
        assert abs(result.expected_roi - expected) < 0.001


def test_remaining_budget_accurate():
    """remaining_budget = total_discount_budget - total_cost."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(total_discount_budget=50_000.0)
    result = run_optimizer(inputs, risk_scores=scores)
    expected_remaining = inputs.total_discount_budget - result.total_cost
    assert abs(result.remaining_budget - expected_remaining) < 0.01


def test_remaining_cs_hours_accurate():
    """remaining_cs_hours = total_cs_hours - total_cs_hours_used."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(total_cs_hours=150.0)
    result = run_optimizer(inputs, risk_scores=scores)
    expected_remaining = inputs.total_cs_hours - result.total_cs_hours_used
    assert abs(result.remaining_cs_hours - expected_remaining) < 0.01


def test_zero_budget_selects_no_customers():
    """With $0 budget and non-zero cost per customer, no customer is selected."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        total_discount_budget=0.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.n_selected == 0
    assert result.total_cost == 0.0
    assert result.total_expected_saved_arr == 0.0


def test_all_selected_when_resources_unlimited():
    """With unlimited budget and hours, all eligible customers should be selected."""
    scores = make_risk_scores(n_high=5, n_medium=5)
    inputs = default_inputs(
        total_discount_budget=1_000_000.0,
        total_cs_hours=10_000.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.n_selected == result.n_eligible
    assert result.n_excluded == 0


def test_save_probability_varies_by_risk_level():
    """High-risk customers must have a lower save probability than Medium-risk.
    Reflects the real-world observation that customers already at High risk
    are harder to retain than those caught earlier at Medium risk."""
    scores = make_risk_scores(n_high=5, n_medium=5)
    inputs = default_inputs(
        total_discount_budget=1_000_000.0,
        total_cs_hours=10_000.0,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    highs = result.selected_customers[result.selected_customers["risk_level"] == "High"]
    meds = result.selected_customers[result.selected_customers["risk_level"] == "Medium"]
    if not highs.empty and not meds.empty:
        assert highs["estimated_save_probability"].iloc[0] < meds["estimated_save_probability"].iloc[0]


def test_selection_reason_present_for_every_customer():
    """Every customer (selected and excluded) must have a non-empty selection_reason.
    This is required for the CS team to understand and trust the optimizer's output."""
    scores = make_risk_scores(n_high=5, n_medium=10)
    inputs = default_inputs(total_discount_budget=5_000.0)
    result = run_optimizer(inputs, risk_scores=scores)
    for df, label in [
        (result.selected_customers, "selected"),
        (result.excluded_customers, "excluded"),
    ]:
        if not df.empty:
            assert "selection_reason" in df.columns, f"selection_reason missing from {label}"
            assert df["selection_reason"].notna().all(), f"Null reason in {label}"
            assert (df["selection_reason"].str.len() > 0).all(), f"Empty reason in {label}"


def test_exclusion_counts_sum_to_n_excluded():
    """The values in exclusion_counts must sum to n_excluded."""
    scores = make_risk_scores(n_high=5, n_medium=10)
    inputs = default_inputs(total_discount_budget=3_000.0)
    result = run_optimizer(inputs, risk_scores=scores)
    if result.exclusion_counts:
        assert sum(result.exclusion_counts.values()) == result.n_excluded


def test_max_customers_reason_logged():
    """Customers excluded due to max_customers limit should have the correct reason code."""
    scores = make_risk_scores(n_high=5, n_medium=5)
    inputs = default_inputs(
        total_discount_budget=1_000_000.0,
        total_cs_hours=10_000.0,
        max_customers=2,
    )
    result = run_optimizer(inputs, risk_scores=scores)
    assert result.n_selected == 2
    assert result.exclusion_counts.get("max_customers_reached", 0) == result.n_eligible - 2


def test_format_summary_returns_non_empty_string():
    """format_optimizer_summary must return a non-empty string with key labels."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_optimizer(default_inputs(), risk_scores=scores)
    summary = format_optimizer_summary(result)
    assert isinstance(summary, str)
    assert len(summary) > 100
    assert "ROI" in summary
    assert "ARR" in summary
