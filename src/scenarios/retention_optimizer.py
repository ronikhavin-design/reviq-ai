"""
Retention Budget Optimizer: Phase 3b Business Decision Layer.

Answers the question: "Given a fixed CS budget and team capacity,
which specific customers should we target to maximize ARR saved?"

Where the Scenario Simulator gives a portfolio-level projection
("what would a campaign achieve?"), the optimizer produces a
per-customer ranked decision list ("who should we call first, and why?").

Algorithm: greedy selection by expected ROI (expected ARR saved divided
by intervention cost), subject to budget, CS hours, and optional
ROI floor constraints. When intervention costs are uniform per customer,
the greedy solution is identical to the exact linear program optimum,
making it both simple to implement and correct to use here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import REPORTS_DIR
from src.scenarios.scenario_simulator import load_risk_scores


DEFAULT_RISK_SCORES_PATH = REPORTS_DIR / "customer_risk_scores.csv"

# Default save probabilities by risk tier.
# Higher risk tier = harder to retain (customer is already far down the churn path).
# These can be overridden with historical win/loss data once available.
DEFAULT_SAVE_PROBABILITY_BY_LEVEL: dict[str, float] = {
    "High":   0.20,
    "Medium": 0.35,
    "Low":    0.50,
}


@dataclass
class OptimizerInput:
    """
    Constraints and assumptions for the retention budget optimizer.

    The key difference from ScenarioInput is that save probabilities
    can vary by risk tier, enabling a per-customer expected value
    calculation rather than a single bulk success rate for all customers.

    Resource constraints:
    - total_discount_budget: hard dollar cap on the campaign
    - total_cs_hours: hard cap on CS team time
    - max_customers: optional absolute customer count cap

    Quality gate:
    - minimum_roi: skip any customer where the expected ARR saved per
      dollar spent falls below this threshold, even if budget remains.
      Prevents spending money on interventions that are unlikely to pay off.
    """

    # Hard resource constraints
    total_discount_budget: float = 50_000.0
    total_cs_hours: float = 150.0

    # Which risk tiers are eligible for the campaign
    target_risk_levels: list[str] = field(default_factory=lambda: ["High", "Medium"])

    # Optional: cap total customers regardless of remaining resources
    max_customers: Optional[int] = None

    # Optional: skip customers where expected_roi falls below this value.
    # E.g., 2.0 means "only intervene when we expect to save at least $2
    # of ARR for every $1 we spend on the discount and CS time."
    minimum_roi: Optional[float] = None

    # Intervention cost per customer (uniform across the campaign)
    avg_discount_per_customer: float = 1_000.0
    avg_cs_hours_per_customer: float = 2.0

    # Probability that a CS intervention fully retains a customer, per tier.
    # Higher risk customers have lower save probability because they are
    # already further down the churn path when we reach them.
    save_probability_by_level: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SAVE_PROBABILITY_BY_LEVEL)
    )


@dataclass
class OptimizerResult:
    """
    Output of the greedy retention budget optimizer.

    selected_customers and excluded_customers both contain a
    'selection_reason' column explaining the decision for each customer.
    This makes the optimizer output self-documenting for CS teams.
    """

    selected_customers: pd.DataFrame    # Chosen customers with enriched metrics
    excluded_customers: pd.DataFrame    # Skipped customers with reason column

    # Counts
    n_eligible: int
    n_selected: int
    n_excluded: int

    # Aggregate financial outcomes
    total_expected_saved_arr: float
    total_cost: float
    total_cs_hours_used: float
    remaining_budget: float
    remaining_cs_hours: float
    expected_roi: float                 # total_expected_saved_arr / total_cost

    # Summary of why customers were excluded, keyed by reason category
    exclusion_counts: dict[str, int]


def _compute_customer_metrics(
    df: pd.DataFrame,
    inputs: OptimizerInput,
) -> pd.DataFrame:
    """
    Enrich the risk score DataFrame with per-customer optimizer metrics.

    For each customer:
    - estimated_save_probability: likelihood that a CS intervention retains them,
      varying by risk tier (higher risk = harder to retain)
    - estimated_intervention_cost: discount offered ($), uniform from inputs
    - estimated_cs_hours: time required ($), uniform from inputs
    - expected_saved_arr: arr_at_risk * save_probability
      (the expected ARR value of one successful intervention on this customer)
    - expected_roi: expected_saved_arr / intervention_cost
      (the ranking key for the greedy selection)
    """
    df = df.copy()

    df["estimated_save_probability"] = (
        df["risk_level"]
        .map(inputs.save_probability_by_level)
        .fillna(0.20)
    )

    df["estimated_intervention_cost"] = inputs.avg_discount_per_customer
    df["estimated_cs_hours"] = inputs.avg_cs_hours_per_customer

    df["expected_saved_arr"] = df["arr_at_risk"] * df["estimated_save_probability"]

    if inputs.avg_discount_per_customer > 0:
        df["expected_roi"] = df["expected_saved_arr"] / inputs.avg_discount_per_customer
    else:
        df["expected_roi"] = 0.0

    return df


def run_optimizer(
    inputs: OptimizerInput,
    risk_scores: Optional[pd.DataFrame] = None,
    risk_scores_path: Path = DEFAULT_RISK_SCORES_PATH,
) -> OptimizerResult:
    """
    Run the greedy retention budget optimizer and return a per-customer decision list.

    Algorithm:
    1. Filter to target_risk_levels and compute per-customer expected value.
    2. Apply minimum_roi floor: exclude customers whose expected ROI is
       too low to justify spending on them, regardless of remaining budget.
    3. Sort remaining customers by expected_roi descending.
       Secondary sort by expected_saved_arr to break ties toward higher
       absolute ARR value.
    4. Greedy selection: iterate in sorted order and select each customer
       if all constraints (budget, hours, max_customers) are still satisfied.
       If one customer cannot fit, skip them and check the next.
    5. Attach a selection_reason to every customer for explainability.

    Greedy correctness: when intervention costs are uniform, sorting by ROI
    and selecting greedily produces the same result as the optimal LP solution.
    The greedy approach is also easier to explain to CS managers and executives.

    Args:
        inputs: Constraints and cost assumptions.
        risk_scores: Pre-loaded DataFrame (pass this in tests to avoid
                     file system dependency).
        risk_scores_path: Path override for customer_risk_scores.csv.

    Returns:
        OptimizerResult with selected and excluded customer lists.
    """
    if risk_scores is None:
        risk_scores = load_risk_scores(risk_scores_path)

    # Step 1: Filter to eligible tiers and compute per-customer metrics.
    eligible = risk_scores[
        risk_scores["risk_level"].isin(inputs.target_risk_levels)
    ].copy()
    eligible = _compute_customer_metrics(eligible, inputs)
    n_eligible = len(eligible)

    # Step 2: Apply ROI floor filter.
    # Customers below minimum_roi are excluded before the greedy loop because
    # spending on them is not justified even if resources are still available.
    roi_excluded_dicts: list[dict] = []
    if inputs.minimum_roi is not None:
        below_floor = eligible["expected_roi"] < inputs.minimum_roi
        for _, row in eligible[below_floor].iterrows():
            d = row.to_dict()
            d["selection_reason"] = (
                f"excluded: ROI {row['expected_roi']:.1f}x below "
                f"minimum {inputs.minimum_roi:.1f}x"
            )
            roi_excluded_dicts.append(d)
        eligible = eligible[~below_floor].copy()

    # Step 3: Sort by expected_roi descending to prioritize highest-return customers.
    eligible = eligible.sort_values(
        ["expected_roi", "expected_saved_arr"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Step 4: Greedy selection under constraints.
    selected_dicts: list[dict] = []
    greedy_excluded_dicts: list[dict] = []

    remaining_budget = inputs.total_discount_budget
    remaining_hours = inputs.total_cs_hours
    n_selected = 0

    for _, row in eligible.iterrows():
        d = row.to_dict()

        if inputs.max_customers is not None and n_selected >= inputs.max_customers:
            d["selection_reason"] = "excluded: max customers limit reached"
            greedy_excluded_dicts.append(d)
            continue

        if row["estimated_intervention_cost"] > remaining_budget:
            d["selection_reason"] = (
                f"excluded: budget exhausted "
                f"(${remaining_budget:,.0f} remaining, "
                f"${row['estimated_intervention_cost']:,.0f} needed)"
            )
            greedy_excluded_dicts.append(d)
            continue

        if row["estimated_cs_hours"] > remaining_hours:
            d["selection_reason"] = (
                f"excluded: CS capacity exhausted "
                f"({remaining_hours:.1f}h remaining, "
                f"{row['estimated_cs_hours']:.1f}h needed)"
            )
            greedy_excluded_dicts.append(d)
            continue

        # All constraints satisfied: select this customer.
        d["selection_reason"] = (
            f"selected: ROI {row['expected_roi']:.1f}x, "
            f"${row['expected_saved_arr']:,.0f} expected ARR saved"
        )
        selected_dicts.append(d)
        remaining_budget -= row["estimated_intervention_cost"]
        remaining_hours -= row["estimated_cs_hours"]
        n_selected += 1

    # Step 5: Build output DataFrames.
    selected_df = (
        pd.DataFrame(selected_dicts).reset_index(drop=True)
        if selected_dicts
        else pd.DataFrame()
    )

    all_excluded = roi_excluded_dicts + greedy_excluded_dicts
    excluded_df = (
        pd.DataFrame(all_excluded).reset_index(drop=True)
        if all_excluded
        else pd.DataFrame()
    )

    # Step 6: Compute aggregate outcomes.
    total_expected_saved_arr = (
        float(selected_df["expected_saved_arr"].sum())
        if not selected_df.empty else 0.0
    )
    total_cost = (
        float(selected_df["estimated_intervention_cost"].sum())
        if not selected_df.empty else 0.0
    )
    total_cs_hours_used = (
        float(selected_df["estimated_cs_hours"].sum())
        if not selected_df.empty else 0.0
    )
    expected_roi = (
        total_expected_saved_arr / total_cost if total_cost > 0 else 0.0
    )

    # Step 7: Categorize exclusion reasons for the summary.
    exclusion_counts: dict[str, int] = {}
    if not excluded_df.empty and "selection_reason" in excluded_df.columns:
        for reason in excluded_df["selection_reason"]:
            if "budget exhausted" in reason:
                key = "budget_exhausted"
            elif "CS capacity exhausted" in reason:
                key = "cs_capacity_exhausted"
            elif "max customers" in reason:
                key = "max_customers_reached"
            elif "below minimum" in reason:
                key = "roi_below_minimum"
            else:
                key = "other"
            exclusion_counts[key] = exclusion_counts.get(key, 0) + 1

    return OptimizerResult(
        selected_customers=selected_df,
        excluded_customers=excluded_df,
        n_eligible=n_eligible,
        n_selected=n_selected,
        n_excluded=len(all_excluded),
        total_expected_saved_arr=total_expected_saved_arr,
        total_cost=total_cost,
        total_cs_hours_used=total_cs_hours_used,
        remaining_budget=remaining_budget,
        remaining_cs_hours=remaining_hours,
        expected_roi=expected_roi,
        exclusion_counts=exclusion_counts,
    )


def format_optimizer_summary(result: OptimizerResult) -> str:
    """Return a human-readable plain-text summary of optimizer results."""
    lines = [
        "Retention Budget Optimizer Results",
        "=" * 44,
        f"  Eligible customers:       {result.n_eligible}",
        f"  Selected for campaign:    {result.n_selected}",
        f"  Excluded:                 {result.n_excluded}",
        "",
        f"  Total expected ARR saved: ${result.total_expected_saved_arr:>11,.0f}",
        f"  Total cost:               ${result.total_cost:>11,.0f}",
        f"  Expected ROI:             {result.expected_roi:.1f}x",
        "",
        f"  CS hours used:            {result.total_cs_hours_used:.1f}h",
        f"  Remaining budget:         ${result.remaining_budget:>11,.0f}",
        f"  Remaining CS hours:       {result.remaining_cs_hours:.1f}h",
    ]
    if result.exclusion_counts:
        lines.append("")
        lines.append("  Exclusion reasons:")
        for reason, count in result.exclusion_counts.items():
            lines.append(f"    {reason}: {count} customer(s)")
    if not result.selected_customers.empty:
        lines.append("")
        lines.append("  Top 5 selected customers:")
        for _, row in result.selected_customers.head(5).iterrows():
            lines.append(
                f"    {row['customer_id']}: "
                f"ARR at risk ${row['arr_at_risk']:,.0f}, "
                f"save prob {row['estimated_save_probability']:.0%}, "
                f"ROI {row['expected_roi']:.1f}x"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    from loguru import logger

    inputs = OptimizerInput(
        total_discount_budget=50_000.0,
        total_cs_hours=150.0,
        target_risk_levels=["High", "Medium"],
        minimum_roi=2.0,
        avg_discount_per_customer=1_000.0,
        avg_cs_hours_per_customer=2.0,
    )

    try:
        result = run_optimizer(inputs)
        print(format_optimizer_summary(result))
    except FileNotFoundError as e:
        logger.error(str(e))
