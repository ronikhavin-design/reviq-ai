"""
Tests for the Scenario Simulator.

All tests use synthetic DataFrames so no file system access is needed.
Each test documents the business rule it is verifying, not just the code path.
"""

import math
import pandas as pd
import pytest

from src.scenarios.scenario_simulator import (
    ScenarioInput,
    ScenarioResult,
    run_scenario,
    format_scenario_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_risk_scores(
    n_high: int = 10,
    n_medium: int = 20,
    n_low: int = 30,
    high_arr: float = 100_000,
    medium_arr: float = 50_000,
    low_arr: float = 20_000,
) -> pd.DataFrame:
    """Build a synthetic risk scores DataFrame for testing.

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


def default_inputs(**overrides) -> ScenarioInput:
    """Return a ScenarioInput with sensible defaults, with optional overrides."""
    inputs = ScenarioInput(
        retention_success_rate=0.30,
        churn_reduction_rate=0.40,
        cs_capacity_hours=150.0,
        avg_hours_per_customer=2.0,
        discount_budget=50_000.0,
        avg_discount_per_customer=1_000.0,
        target_risk_levels=["High", "Medium"],
    )
    for k, v in overrides.items():
        setattr(inputs, k, v)
    return inputs


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_eligible_customers_filtered_by_risk_level():
    """Only customers in target_risk_levels should be eligible.
    Low-risk customers must be excluded even if they exist in the data."""
    scores = make_risk_scores(n_high=10, n_medium=20, n_low=30)
    result = run_scenario(default_inputs(target_risk_levels=["High"]), risk_scores=scores)
    assert result.customers_eligible == 10
    assert set(result.targeted_customers["risk_level"].tolist()) == {"High"}


def test_cs_capacity_is_binding():
    """When CS hours are the tightest constraint, binding_constraint = cs_capacity."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    # 20 hours / 2 hours per customer = 10 limit; 30 eligible, budget not binding
    inputs = default_inputs(
        cs_capacity_hours=20.0,
        avg_hours_per_customer=2.0,
        discount_budget=1_000_000.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.cs_capacity_limit == 10
    assert result.customers_targeted == 10
    assert result.binding_constraint == "cs_capacity"


def test_budget_is_binding():
    """When the discount budget is the tightest constraint, binding_constraint = budget."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    # $5,000 / $1,000 per customer = 5 limit; CS capacity is not binding
    inputs = default_inputs(
        cs_capacity_hours=1_000.0,
        avg_hours_per_customer=2.0,
        discount_budget=5_000.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.budget_capacity_limit == 5
    assert result.customers_targeted == 5
    assert result.binding_constraint == "budget"


def test_no_binding_constraint_when_resources_exceed_eligible():
    """When both resources exceed the eligible count, no constraint is binding
    and every eligible customer is targeted."""
    scores = make_risk_scores(n_high=5, n_medium=5, n_low=0)
    inputs = default_inputs(
        cs_capacity_hours=10_000.0,
        discount_budget=1_000_000.0,
        target_risk_levels=["High", "Medium"],
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.customers_targeted == result.customers_eligible
    assert result.customers_excluded == 0
    assert result.binding_constraint == "none"


def test_customers_saved_uses_retention_success_rate():
    """n_saved = floor(n_targeted * retention_success_rate)."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        retention_success_rate=0.30,
        cs_capacity_hours=10_000.0,
        discount_budget=1_000_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    expected_saved = math.floor(result.customers_targeted * 0.30)
    assert result.customers_saved == expected_saved


def test_arr_remaining_equals_original_minus_saved():
    """Business accounting check: remaining = original - arr_saved (binary model)."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    assert abs(
        result.remaining_arr_at_risk
        - (result.original_arr_at_risk - result.expected_arr_saved)
    ) < 0.01


def test_arr_remaining_by_reduction_equals_original_minus_reduction():
    """Business accounting check: remaining = original - arr_saved (continuous model)."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    assert abs(
        result.remaining_arr_at_risk_by_reduction
        - (result.original_arr_at_risk - result.expected_arr_saved_by_reduction)
    ) < 0.01


def test_roi_is_arr_saved_divided_by_cost():
    """ROI multiple = expected_arr_saved / estimated_cost (binary model)."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    if result.estimated_cost > 0:
        expected_roi = result.expected_arr_saved / result.estimated_cost
        assert abs(result.roi_multiple - expected_roi) < 0.001


def test_customers_excluded_equals_eligible_minus_targeted():
    """Excluded count = eligible - targeted. Every eligible customer is accounted for."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        cs_capacity_hours=20.0,
        avg_hours_per_customer=2.0,
        discount_budget=1_000_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.customers_excluded == result.customers_eligible - result.customers_targeted


def test_impact_by_risk_level_customer_counts_sum_to_targeted():
    """Sum of customers_targeted across risk tiers must equal overall customers_targeted."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    total = sum(i.customers_targeted for i in result.impact_by_risk_level)
    assert total == result.customers_targeted


def test_impact_by_risk_level_arr_sums_to_total():
    """Sum of original_arr_at_risk across tiers must equal the overall eligible total."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    total_arr = sum(i.original_arr_at_risk for i in result.impact_by_risk_level)
    assert abs(total_arr - result.original_arr_at_risk) < 1.0


def test_high_only_targeting_excludes_medium_and_low():
    """Targeting only High excludes Medium and Low customers from all calculations."""
    scores = make_risk_scores(n_high=8, n_medium=15, n_low=20)
    inputs = default_inputs(
        target_risk_levels=["High"],
        cs_capacity_hours=10_000.0,
        discount_budget=1_000_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.customers_eligible == 8
    assert set(result.targeted_customers["risk_level"].tolist()) == {"High"}


def test_zero_discount_budget_targets_no_customers():
    """With $0 budget and non-zero cost per customer, no one can be targeted."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    inputs = default_inputs(
        discount_budget=0.0,
        avg_discount_per_customer=1_000.0,
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.budget_capacity_limit == 0
    assert result.customers_targeted == 0
    assert result.customers_saved == 0
    assert result.expected_arr_saved == 0.0
    assert result.estimated_cost == 0.0


def test_high_value_customers_prioritized_over_low_value():
    """When capacity is limited, High-risk customers (higher ARR at risk) are
    targeted before Medium-risk customers."""
    # High arr_at_risk = 100,000 * 0.70 = 70,000
    # Medium arr_at_risk = 50,000 * 0.35 = 17,500
    scores = make_risk_scores(n_high=5, n_medium=20, n_low=0)
    inputs = default_inputs(
        cs_capacity_hours=10.0,  # 10h / 2h per customer = 5 slots
        avg_hours_per_customer=2.0,
        discount_budget=1_000_000.0,
        target_risk_levels=["High", "Medium"],
    )
    result = run_scenario(inputs, risk_scores=scores)
    assert result.customers_targeted == 5
    # All 5 slots should go to High customers because they have higher ARR at risk
    assert all(r == "High" for r in result.targeted_customers["risk_level"].tolist())


def test_format_summary_returns_non_empty_string():
    """format_scenario_summary must return a non-empty string containing key labels."""
    scores = make_risk_scores(n_high=10, n_medium=20)
    result = run_scenario(default_inputs(), risk_scores=scores)
    summary = format_scenario_summary(result)
    assert isinstance(summary, str)
    assert len(summary) > 100
    assert "ARR" in summary
    assert "ROI" in summary
    assert "targeted" in summary
