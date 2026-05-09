# Scenario Simulator

## What it does

The Scenario Simulator answers one executive-level question:

**"What happens to our ARR at Risk if we run a targeted retention campaign?"**

It takes the churn model's risk scores as input and projects the dollar-denominated
business impact of a retention intervention, given realistic constraints on the CS
team's time and budget.

This is Phase 3 of the RevIQ AI platform: the transition from *showing data* to
*recommending actions*.

---

## Business problem

The churn model tells you *who* is at risk and *how likely* they are to churn.
But it does not tell you what to do. A VP of Customer Success still needs to answer:

- How many customers can we realistically reach this month?
- Which ones should we prioritize?
- How much ARR can we save given our team's capacity and budget?
- What is the ROI of running this campaign versus not running it?

The Scenario Simulator answers all of these using the model's predictions as inputs.

---

## How it connects to the churn model

The simulator reads `data/reports/customer_risk_scores.csv`, which is generated
by the churn model pipeline. Each row represents one customer with:

| Column | Description |
|---|---|
| `churn_probability` | Model prediction (0 to 1) |
| `arr_at_risk` | churn_probability * ARR (expected revenue loss if customer churns) |
| `risk_level` | Categorical label: High, Medium, or Low |
| `arr` | Annual Recurring Revenue for this customer |

The simulator takes these as given. It does not retrain or modify the model.
It projects forward: "Given what the model predicts, what business outcome can we
achieve under specific resource constraints?"

---

## Input parameters

All parameters are set by the user (CS manager, VP of CS, or Finance team):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `retention_success_rate` | float | 0.30 | Fraction of targeted customers who are fully retained |
| `churn_reduction_rate` | float | 0.40 | Fraction by which each targeted customer's churn risk is reduced |
| `cs_capacity_hours` | float | 150.0 | Total CS team hours available for outreach this period |
| `avg_hours_per_customer` | float | 2.0 | Hours per customer (prep, call, follow-up, notes) |
| `discount_budget` | float | 50,000 | Total budget for retention discounts or credits ($) |
| `avg_discount_per_customer` | float | 1,000 | Average discount offered per at-risk customer ($) |
| `target_risk_levels` | list | ["High", "Medium"] | Which risk tiers are eligible for the campaign |

---

## Two impact models

Every run of the simulator computes two views of expected ARR savings:

### Binary model (retention_success_rate)

A fraction of targeted customers are fully retained. Their complete ARR at risk
is counted as recovered.

```
n_saved = floor(n_targeted * retention_success_rate)
arr_saved = sum(arr_at_risk for the top n_saved customers)
```

This is the **conservative view**: only fully retained customers count as wins.
Useful when reporting to Finance, where a partial outcome is hard to account for.

### Continuous model (churn_reduction_rate)

Every targeted customer's churn probability is partially reduced. Each customer
contributes a proportional ARR recovery, even if they ultimately churn.

```
arr_saved_per_customer = arr_at_risk * churn_reduction_rate
arr_saved = sum across all targeted customers
```

This is the **optimistic view**: every intervention reduces risk at some level.
Useful for showing the upper bound of what the campaign could achieve.

Real-world outcomes typically fall between the two estimates.

---

## Prioritization strategy

When capacity or budget prevents reaching all eligible customers, the simulator
prioritizes by ARR at risk (descending). The highest-value accounts are targeted
first, maximizing expected ARR saved per dollar spent.

This means a customer with $70,000 ARR at risk is always targeted before one with
$17,500 ARR at risk, regardless of risk tier label.

---

## Output fields

The `ScenarioResult` dataclass contains:

| Field | Description |
|---|---|
| `customers_eligible` | Count matching `target_risk_levels` |
| `customers_targeted` | Count reachable given constraints |
| `customers_saved` | Expected fully retained (binary model) |
| `customers_excluded` | Eligible but unreachable due to constraints |
| `original_arr_at_risk` | ARR at risk before any intervention |
| `expected_arr_saved` | Projected ARR recovery (binary model) |
| `remaining_arr_at_risk` | ARR still at risk after campaign (binary) |
| `expected_arr_saved_by_reduction` | Projected ARR recovery (continuous model) |
| `remaining_arr_at_risk_by_reduction` | ARR still at risk after campaign (continuous) |
| `cs_capacity_limit` | Max customers CS team can reach |
| `budget_capacity_limit` | Max customers budget can cover |
| `binding_constraint` | "cs_capacity", "budget", or "none" |
| `estimated_cost` | Total discount spend committed ($) |
| `roi_multiple` | ARR saved per $1 spent (binary model) |
| `roi_multiple_by_reduction` | ARR saved per $1 spent (continuous model) |
| `impact_by_risk_level` | Breakdown by risk tier (list of RiskLevelImpact) |
| `targeted_customers` | DataFrame of customers included in the campaign |

---

## Example usage

```python
from src.scenarios.scenario_simulator import ScenarioInput, run_scenario, format_scenario_summary

inputs = ScenarioInput(
    retention_success_rate=0.30,
    churn_reduction_rate=0.40,
    cs_capacity_hours=150.0,
    avg_hours_per_customer=2.0,
    discount_budget=50_000.0,
    avg_discount_per_customer=1_000.0,
    target_risk_levels=["High", "Medium"],
)

result = run_scenario(inputs)
print(format_scenario_summary(result))
```

Example output:

```
Retention Scenario Simulation
============================================
  Eligible customers:     117
  Targeted customers:     50
  Excluded (over limit):  67
  Binding constraint:     cs_capacity
  CS capacity limit:      75 customers
  Budget capacity limit:  50 customers

Binary model (retention_success_rate):
  Customers saved:        15
  ARR at risk (before):   $  4,200,000
  Expected ARR saved:     $    840,000
  ARR at risk (after):    $  3,360,000
  ROI:                    16.8x

Continuous model (churn_reduction_rate):
  Expected ARR saved:     $    560,000
  ARR at risk (after):    $  3,640,000
  ROI:                    11.2x

  Estimated campaign cost: $     50,000

Impact by risk level:
  High        : 35 targeted, 10 saved (binary), $700,000 ARR saved
  Medium      : 15 targeted, 4 saved (binary), $140,000 ARR saved
```

## Running from the command line

```bash
python -m src.scenarios.scenario_simulator
```

Requires `data/reports/customer_risk_scores.csv`. Run `python run_pipeline.py` first.

---

## How it will connect to the dashboard (Phase 3b)

The next step (Phase 3b) will add a Streamlit page where the CS manager can:

1. Select target risk tiers with a multi-select widget
2. Adjust each `ScenarioInput` field with a slider or number input
3. See the results update in real time (no page reload needed)
4. View a before/after ARR bar chart comparing binary vs. continuous projections
5. Download `targeted_customers` as a CSV for the CS team

The simulator backend is stateless and fast (under one second on 1,000 customers),
which makes it suitable for real-time slider interactions.
