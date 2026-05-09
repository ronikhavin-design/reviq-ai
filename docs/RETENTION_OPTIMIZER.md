# Retention Budget Optimizer

## What it does

The Retention Budget Optimizer answers one operational question:

**"Given a fixed CS budget and team capacity, which specific customers
should we target first to maximize ARR saved?"**

It produces a ranked, per-customer action list with a reason attached
to every decision. The CS team receives a prioritized call list, not
just aggregate projections.

This is Phase 3b of the RevIQ AI platform.

---

## How it differs from the Scenario Simulator

Both tools read the same input file (`customer_risk_scores.csv`) and
use similar constraint parameters. The difference is in what they output:

| | Scenario Simulator (3a) | Retention Optimizer (3b) |
|---|---|---|
| **Output level** | Portfolio aggregate | Per-customer decision |
| **Primary question** | "What would this campaign achieve?" | "Who should we call first?" |
| **Save probability** | One rate for all customers | Varies by risk tier |
| **ROI filter** | Not available | minimum_roi gate |
| **Output** | Summary numbers + bulk customer list | Ranked list with selection_reason per row |
| **Use case** | Executive planning, scenario comparison | CS team call list, daily operations |

The Scenario Simulator is a planning tool. The Optimizer is an execution tool.

---

## Business problem

After the churn model identifies 117 at-risk customers, the CS team
cannot call all of them. They have 150 hours this month and a $50,000
retention discount budget. They need to know:

- Which 50 customers are worth intervening on?
- In what order should they prioritize?
- Which customers are not worth the cost of intervention?
- How much ARR do we expect to recover?

The optimizer answers all of these.

---

## How it connects to the churn model

The optimizer reads `data/reports/customer_risk_scores.csv`, the same
file written by the churn model pipeline. Each row provides:

- `arr_at_risk`: the expected revenue loss if this customer churns
- `risk_level`: High, Medium, or Low (from the churn model)
- `churn_probability`: the raw model prediction

The optimizer adds one new concept that the churn model does not produce:
`estimated_save_probability`: the likelihood that a CS intervention
retains this specific customer. This varies by risk tier because
customers who are already at High risk are harder to turn around than
those caught earlier at Medium risk.

---

## Input parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `total_discount_budget` | float | 50,000 | Total dollar budget for retention discounts |
| `total_cs_hours` | float | 150.0 | Total CS team hours available |
| `target_risk_levels` | list | ["High", "Medium"] | Which tiers are eligible |
| `max_customers` | int or None | None | Hard cap on customer count |
| `minimum_roi` | float or None | None | Skip customers below this ROI threshold |
| `avg_discount_per_customer` | float | 1,000 | Discount offered per customer ($) |
| `avg_cs_hours_per_customer` | float | 2.0 | Hours per customer (prep, call, follow-up) |
| `save_probability_by_level` | dict | see below | Per-tier save probability |

Default save probabilities (can be overridden with historical data):

| Risk Level | Default Save Probability | Interpretation |
|---|---|---|
| High | 0.20 | 1 in 5 targeted High-risk customers is retained |
| Medium | 0.35 | About 1 in 3 targeted Medium-risk customers is retained |
| Low | 0.50 | 1 in 2 targeted Low-risk customers is retained |

---

## Algorithm: greedy selection by expected ROI

For each eligible customer, the optimizer computes:

```
expected_saved_arr = arr_at_risk * save_probability_for_their_tier
expected_roi = expected_saved_arr / avg_discount_per_customer
```

Customers are then sorted by `expected_roi` descending and selected
greedily until a constraint is hit. If one customer cannot fit (e.g.,
the remaining budget is too small), the optimizer skips them and
continues to the next, so smaller-cost candidates can still be selected.

**Why greedy is appropriate here:**

When intervention costs are uniform per customer (which is the typical
case in a SaaS CS campaign), sorting by ROI and selecting greedily
produces the same result as solving the exact linear program. The greedy
approach has two additional advantages:

1. It is explainable: "We selected every customer with positive expected
   ROI, in order of highest return per dollar, until we ran out of budget."
2. It is fast: O(n log n) due to sorting, not a solver runtime.

If intervention costs vary significantly between customers (e.g., some
require a high-touch executive call while others need only an email), a
linear program using `scipy.optimize` or `cvxpy` would produce a better
result. That is documented as a future enhancement.

---

## Selection reason codes

Every customer in the output (selected and excluded) has a `selection_reason`
column. This is intentional: the CS team should never receive a list without
knowing why each decision was made.

| Reason | Meaning |
|---|---|
| `selected: ROI Xx, $Y expected ARR saved` | Customer was selected; shows the expected return |
| `excluded: budget exhausted` | Budget was used up before reaching this customer |
| `excluded: CS capacity exhausted` | CS hours were used up before reaching this customer |
| `excluded: max customers limit reached` | Hard customer count cap was hit |
| `excluded: ROI Xx below minimum Yx` | Expected return was below the minimum_roi threshold |

---

## Output fields

| Field | Description |
|---|---|
| `selected_customers` | DataFrame of chosen customers with enriched metrics |
| `excluded_customers` | DataFrame of skipped customers with selection_reason |
| `n_eligible` | Total customers in target risk tiers |
| `n_selected` | Customers chosen for the campaign |
| `n_excluded` | Customers skipped (all reasons combined) |
| `total_expected_saved_arr` | Projected ARR recovery from selected customers |
| `total_cost` | Total discount budget committed |
| `total_cs_hours_used` | CS hours allocated |
| `remaining_budget` | Budget not committed |
| `remaining_cs_hours` | CS hours not allocated |
| `expected_roi` | total_expected_saved_arr / total_cost |
| `exclusion_counts` | Dict of reason category to count |

---

## Example usage

```python
from src.scenarios.retention_optimizer import OptimizerInput, run_optimizer, format_optimizer_summary

inputs = OptimizerInput(
    total_discount_budget=50_000.0,
    total_cs_hours=150.0,
    target_risk_levels=["High", "Medium"],
    minimum_roi=2.0,
    avg_discount_per_customer=1_000.0,
    avg_cs_hours_per_customer=2.0,
)

result = run_optimizer(inputs)
print(format_optimizer_summary(result))

# Export the call list for the CS team
result.selected_customers.to_csv("cs_call_list.csv", index=False)
```

## Running from the command line

```bash
python -m src.scenarios.retention_optimizer
```

Requires `data/reports/customer_risk_scores.csv`. Run `python run_pipeline.py` first.

---

## Planned enhancements (Phase 3c and beyond)

**Non-uniform costs:** If some customers require a $5,000 executive retention
package while others only need a $500 discount, a linear program (`scipy.optimize`
or `cvxpy`) would produce a better allocation. This is a natural next step once
the product team defines tiered intervention packages.

**Historical win/loss calibration:** The `save_probability_by_level` defaults are
assumptions. Once the CS team tracks intervention outcomes, these probabilities can
be calibrated to actual retention rates, making the optimizer progressively more
accurate.

**Dashboard integration (Phase 3b):** A Streamlit page will let the CS manager
set constraints with sliders, view the ranked call list as a sortable table,
and download it as a CSV. The optimizer is already stateless and fast enough
for real-time slider interactions.
