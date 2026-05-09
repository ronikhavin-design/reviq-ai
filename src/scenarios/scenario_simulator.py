"""
Scenario Simulator: Phase 3 Business Decision Layer.

Answers the question: "What happens to our ARR at Risk if we run
a targeted retention campaign?"

The simulator translates churn model predictions into dollar-denominated
business outcomes. Given a set of business constraints (CS team capacity,
discount budget) and assumptions (expected success rate), it calculates
how many customers can be reached, how much ARR can be saved, and what
the campaign ROI looks like.

Inputs come from the churn model output (customer_risk_scores.csv).
The simulator does not retrain or modify the model. It takes model
output as given and projects forward under business assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import REPORTS_DIR


DEFAULT_RISK_SCORES_PATH = REPORTS_DIR / "customer_risk_scores.csv"


@dataclass
class ScenarioInput:
    """
    Business assumptions that define a retention scenario.

    These are set by the CS manager, VP of Customer Success, or Finance
    team based on available resources for the planning period.

    Two framing approaches for estimating impact are included:

    1. retention_success_rate (binary): "What fraction of customers we touch
       will be fully retained?" Controls the customer count that is saved.

    2. churn_reduction_rate (continuous): "By how much do we reduce each
       targeted customer's churn probability?" Produces a partial ARR
       recovery across all targeted customers.

    Both are computed and returned in ScenarioResult so stakeholders can
    compare a conservative (binary) vs. an optimistic (continuous) view.
    """

    # Fraction of targeted customers who are fully retained after intervention.
    # Realistic SaaS CS intervention range: 0.20 to 0.50.
    # Default: we save 3 out of every 10 customers we actively touch.
    retention_success_rate: float = 0.30

    # Fraction by which each targeted customer's churn probability is reduced.
    # E.g., 0.40 means a customer with 60% churn risk drops to 36%.
    # Used as a continuous alternative view of expected ARR recovery.
    churn_reduction_rate: float = 0.40

    # Total CS team hours available for proactive outreach this period.
    cs_capacity_hours: float = 150.0

    # Hours required per customer: prep, call, follow-up, internal notes.
    avg_hours_per_customer: float = 2.0

    # Total budget available for retention discounts or credits ($).
    discount_budget: float = 50_000.0

    # Average discount or credit offered per at-risk customer ($).
    # This is the cost of one retention attempt, not the ARR value saved.
    avg_discount_per_customer: float = 1_000.0

    # Which risk tiers are eligible for this campaign.
    # Options are any combination of: "High", "Medium", "Low".
    target_risk_levels: list[str] = field(default_factory=lambda: ["High", "Medium"])


@dataclass
class RiskLevelImpact:
    """ARR impact breakdown for a single risk tier."""

    risk_level: str
    customers_eligible: int
    customers_targeted: int
    customers_saved: int
    original_arr_at_risk: float
    expected_arr_saved: float
    expected_arr_saved_by_reduction: float


@dataclass
class ScenarioResult:
    """
    Projected business outcomes of a retention scenario.

    Two ARR impact estimates are computed side by side:
    - Binary (retention_success_rate): n_saved customers are fully retained.
    - Continuous (churn_reduction_rate): all targeted customers' churn risk
      is partially reduced.

    The binary model is the conservative floor; the continuous model is the
    optimistic ceiling. Real outcomes typically fall between the two.

    All dollar figures are in USD, matching the input data.
    """

    # Customer counts
    customers_eligible: int      # eligible based on target_risk_levels
    customers_targeted: int      # reachable given capacity and budget
    customers_saved: int         # expected fully retained (binary model)
    customers_excluded: int      # eligible but unreachable due to constraints

    # ARR impact: binary model (retention_success_rate)
    original_arr_at_risk: float
    expected_arr_saved: float
    remaining_arr_at_risk: float

    # ARR impact: continuous model (churn_reduction_rate)
    expected_arr_saved_by_reduction: float
    remaining_arr_at_risk_by_reduction: float

    # Constraint analysis
    cs_capacity_limit: int
    budget_capacity_limit: int
    # Which limit is binding: "cs_capacity", "budget", or "none"
    binding_constraint: str

    # Cost and return on investment
    estimated_cost: float
    roi_multiple: float               # binary: arr_saved / cost
    roi_multiple_by_reduction: float  # continuous: arr_saved_by_reduction / cost

    # Per-tier breakdown
    impact_by_risk_level: list[RiskLevelImpact]

    # Customers included in the campaign, sorted by arr_at_risk descending
    targeted_customers: pd.DataFrame


def load_risk_scores(path: Path = DEFAULT_RISK_SCORES_PATH) -> pd.DataFrame:
    """
    Load the churn model output from disk.

    Raises FileNotFoundError if run_pipeline.py has not been executed yet.
    Raises ValueError if required columns are missing from the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Risk scores not found at {path}. "
            "Run python run_pipeline.py to generate them."
        )
    df = pd.read_csv(path)
    required = {"customer_id", "arr_at_risk", "arr", "churn_probability", "risk_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Risk scores file is missing columns: {missing}")
    return df


def run_scenario(
    inputs: ScenarioInput,
    risk_scores: Optional[pd.DataFrame] = None,
    risk_scores_path: Path = DEFAULT_RISK_SCORES_PATH,
) -> ScenarioResult:
    """
    Run a retention scenario simulation and return projected business outcomes.

    Prioritization strategy: eligible customers are sorted by ARR at risk
    (descending) before applying capacity and budget limits. This maximizes
    expected ARR saved for a given spend by targeting the highest-value
    accounts first.

    Two ARR impact models are computed in every run:
    - Binary: floor(n_targeted * retention_success_rate) customers are
      fully saved. Their full arr_at_risk is counted as recovered.
    - Continuous: every targeted customer's arr_at_risk is reduced by
      churn_reduction_rate. Each customer contributes a partial recovery.

    Args:
        inputs: Business assumptions for the scenario.
        risk_scores: Pre-loaded DataFrame. If None, loaded from disk.
                     Pass this in tests to avoid file system dependency.
        risk_scores_path: Override path to customer_risk_scores.csv.

    Returns:
        ScenarioResult with all projected outcomes.
    """
    if risk_scores is None:
        risk_scores = load_risk_scores(risk_scores_path)

    # Step 1: Filter to eligible customers based on selected risk tiers.
    eligible = risk_scores[
        risk_scores["risk_level"].isin(inputs.target_risk_levels)
    ].copy()

    # Sort descending by ARR at risk so the highest-value accounts are
    # prioritized when capacity or budget limits how many we can reach.
    eligible = eligible.sort_values("arr_at_risk", ascending=False).reset_index(drop=True)
    n_eligible = len(eligible)

    # Step 2: Calculate how many customers each resource constraint allows.

    # CS capacity: how many customers can the team touch this period?
    if inputs.avg_hours_per_customer > 0:
        cs_capacity_limit = int(inputs.cs_capacity_hours / inputs.avg_hours_per_customer)
    else:
        # No time cost per customer means the CS capacity is not a constraint.
        cs_capacity_limit = n_eligible

    # Budget capacity: how many customers can we offer a discount to?
    if inputs.avg_discount_per_customer > 0:
        budget_capacity_limit = int(inputs.discount_budget / inputs.avg_discount_per_customer)
    else:
        # No cost per customer means the budget is not a constraint.
        budget_capacity_limit = n_eligible

    # Step 3: Apply the binding constraint.
    # n_targeted is capped by whichever limit hits first.
    n_targeted = min(n_eligible, cs_capacity_limit, budget_capacity_limit)

    if cs_capacity_limit >= n_eligible and budget_capacity_limit >= n_eligible:
        # Both resources are sufficient to reach all eligible customers.
        binding_constraint = "none"
    elif cs_capacity_limit <= budget_capacity_limit:
        binding_constraint = "cs_capacity"
    else:
        binding_constraint = "budget"

    n_excluded = n_eligible - n_targeted

    # Step 4: Select the top n_targeted customers (highest ARR at risk first).
    targeted = eligible.head(n_targeted).copy()

    # Step 5: Binary model.
    # A fraction of targeted customers are fully retained. We assume the saved
    # customers come from the top of the priority list (highest ARR at risk),
    # which is consistent with the same prioritization used to select targeted.
    n_saved = math.floor(n_targeted * inputs.retention_success_rate)
    saved = targeted.head(n_saved)

    original_arr_at_risk = float(eligible["arr_at_risk"].sum())
    expected_arr_saved = float(saved["arr_at_risk"].sum())
    remaining_arr_at_risk = original_arr_at_risk - expected_arr_saved

    # Step 6: Continuous model.
    # Each targeted customer's churn probability is partially reduced by
    # churn_reduction_rate. Their ARR at risk shrinks proportionally.
    # arr_saved per customer = arr_at_risk * churn_reduction_rate
    expected_arr_saved_by_reduction = float(
        (targeted["arr_at_risk"] * inputs.churn_reduction_rate).sum()
    )
    remaining_arr_at_risk_by_reduction = original_arr_at_risk - expected_arr_saved_by_reduction

    # Step 7: Campaign cost and ROI.
    # Cost = discounts committed to all targeted customers. We offer the
    # discount before knowing who will churn; it is the cost of the attempt.
    estimated_cost = n_targeted * inputs.avg_discount_per_customer

    roi_multiple = expected_arr_saved / estimated_cost if estimated_cost > 0 else 0.0
    roi_multiple_by_reduction = (
        expected_arr_saved_by_reduction / estimated_cost if estimated_cost > 0 else 0.0
    )

    # Step 8: Break down impact by risk tier for reporting.
    impact_by_risk_level = []
    for level in inputs.target_risk_levels:
        level_eligible = eligible[eligible["risk_level"] == level]
        level_targeted = targeted[targeted["risk_level"] == level]
        level_n_saved = math.floor(len(level_targeted) * inputs.retention_success_rate)
        level_saved = level_targeted.head(level_n_saved)
        level_arr_saved_by_reduction = float(
            (level_targeted["arr_at_risk"] * inputs.churn_reduction_rate).sum()
        )
        impact_by_risk_level.append(RiskLevelImpact(
            risk_level=level,
            customers_eligible=len(level_eligible),
            customers_targeted=len(level_targeted),
            customers_saved=level_n_saved,
            original_arr_at_risk=float(level_eligible["arr_at_risk"].sum()),
            expected_arr_saved=float(level_saved["arr_at_risk"].sum()),
            expected_arr_saved_by_reduction=level_arr_saved_by_reduction,
        ))

    return ScenarioResult(
        customers_eligible=n_eligible,
        customers_targeted=n_targeted,
        customers_saved=n_saved,
        customers_excluded=n_excluded,
        original_arr_at_risk=original_arr_at_risk,
        expected_arr_saved=expected_arr_saved,
        remaining_arr_at_risk=remaining_arr_at_risk,
        expected_arr_saved_by_reduction=expected_arr_saved_by_reduction,
        remaining_arr_at_risk_by_reduction=remaining_arr_at_risk_by_reduction,
        cs_capacity_limit=cs_capacity_limit,
        budget_capacity_limit=budget_capacity_limit,
        binding_constraint=binding_constraint,
        estimated_cost=estimated_cost,
        roi_multiple=roi_multiple,
        roi_multiple_by_reduction=roi_multiple_by_reduction,
        impact_by_risk_level=impact_by_risk_level,
        targeted_customers=targeted,
    )


def format_scenario_summary(result: ScenarioResult) -> str:
    """Return a human-readable plain-text summary of a scenario result."""
    lines = [
        "Retention Scenario Simulation",
        "=" * 44,
        f"  Eligible customers:     {result.customers_eligible}",
        f"  Targeted customers:     {result.customers_targeted}",
        f"  Excluded (over limit):  {result.customers_excluded}",
        f"  Binding constraint:     {result.binding_constraint}",
        f"  CS capacity limit:      {result.cs_capacity_limit} customers",
        f"  Budget capacity limit:  {result.budget_capacity_limit} customers",
        "",
        "Binary model (retention_success_rate):",
        f"  Customers saved:        {result.customers_saved}",
        f"  ARR at risk (before):   ${result.original_arr_at_risk:>12,.0f}",
        f"  Expected ARR saved:     ${result.expected_arr_saved:>12,.0f}",
        f"  ARR at risk (after):    ${result.remaining_arr_at_risk:>12,.0f}",
        f"  ROI:                    {result.roi_multiple:.1f}x",
        "",
        "Continuous model (churn_reduction_rate):",
        f"  Expected ARR saved:     ${result.expected_arr_saved_by_reduction:>12,.0f}",
        f"  ARR at risk (after):    ${result.remaining_arr_at_risk_by_reduction:>12,.0f}",
        f"  ROI:                    {result.roi_multiple_by_reduction:.1f}x",
        "",
        f"  Estimated campaign cost: ${result.estimated_cost:>11,.0f}",
        "",
        "Impact by risk level:",
    ]
    for impact in result.impact_by_risk_level:
        lines.append(
            f"  {impact.risk_level:<12}: "
            f"{impact.customers_targeted} targeted, "
            f"{impact.customers_saved} saved (binary), "
            f"${impact.expected_arr_saved:,.0f} ARR saved"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from loguru import logger

    inputs = ScenarioInput(
        retention_success_rate=0.30,
        churn_reduction_rate=0.40,
        cs_capacity_hours=150.0,
        avg_hours_per_customer=2.0,
        discount_budget=50_000.0,
        avg_discount_per_customer=1_000.0,
        target_risk_levels=["High", "Medium"],
    )

    try:
        result = run_scenario(inputs)
        print(format_scenario_summary(result))
    except FileNotFoundError as e:
        logger.error(str(e))
