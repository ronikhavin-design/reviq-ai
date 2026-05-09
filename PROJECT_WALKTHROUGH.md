# RevIQ AI: Project Walkthrough

This document explains the current MVP from top to bottom: what the project does, how each piece works, and why each decision was made. Written for someone who wants to understand the full picture before touching the code.

---

## Table of Contents

1. [What the project does](#1-what-the-project-does)
2. [Folder and file structure](#2-folder-and-file-structure)
3. [How the data is generated](#3-how-the-data-is-generated)
4. [How features are built](#4-how-features-are-built)
5. [How the churn model is trained](#5-how-the-churn-model-is-trained)
6. [How ARR at risk is calculated](#6-how-arr-at-risk-is-calculated)
7. [How SHAP explanations work](#7-how-shap-explanations-work)
8. [How the Streamlit dashboard works](#8-how-the-streamlit-dashboard-works)
9. [Full pipeline flow](#9-full-pipeline-flow)
10. [How to run everything](#10-how-to-run-everything)

---

## 1. What the project does

RevIQ AI is a machine learning platform for a fictional B2B SaaS company. It answers one central business question:

> **"Which of our customers are most likely to stop paying, how much revenue are we about to lose, and why?"**

It does this in four steps:

1. Simulates realistic customer data (subscriptions, product usage, support tickets, customer health)
2. Trains a machine learning model that predicts the probability each customer will churn
3. Combines that churn probability with each customer's revenue size and contract renewal timing to produce a **Revenue Risk Score**
4. Explains each prediction using SHAP so a business user can understand *why* a customer is flagged as risky

The output is a Streamlit dashboard that shows executives:
- How total ARR compares to target
- Which customers are at high risk
- How much revenue is at risk in dollars
- The specific factors driving each customer's risk

---

## 2. Folder and file structure

```
reviq-ai/
│
├── configs/
│   └── model_config.yaml          ← All tunable parameters in one place
│
├── src/
│   ├── config.py                  ← Paths and config loader used across the project
│   │
│   ├── data/
│   │   └── generate_saas_data.py  ← Creates all synthetic SaaS tables
│   │
│   ├── features/
│   │   └── build_features.py      ← Joins tables, creates ML-ready features
│   │
│   ├── models/
│   │   ├── train_churn_model.py   ← Trains and evaluates churn models, logs to MLflow
│   │   └── train_arr_forecast.py  ← Trains ARR forecast model with time-series CV
│   │
│   ├── risk/
│   │   └── revenue_risk_score.py  ← Turns churn probability into a business risk score
│   │
│   └── explainability/
│       └── shap_explainer.py      ← SHAP driver computation and formatting
│
├── app/
│   └── streamlit_app.py           ← The interactive dashboard
│
├── api/
│   └── main.py                    ← FastAPI endpoints for serving predictions
│
├── tests/
│   └── test_features.py           ← Automated tests for data and feature logic
│
├── data/
│   ├── synthetic/                 ← Raw generated CSV files
│   ├── processed/                 ← Feature matrices ready for model training
│   └── reports/                   ← Model outputs: risk scores, SHAP importance
│
├── models/                        ← Saved model files (.pkl)
├── mlruns/                        ← MLflow experiment logs
│
├── run_pipeline.py                ← Runs the full pipeline in one command
├── Dockerfile                     ← Container definition
├── docker-compose.yml             ← Runs API + dashboard together
└── requirements.txt               ← All Python dependencies
```

**The key design principle:** each folder has one responsibility. Data generation never touches model training. The risk score logic is separate from both. This makes it easy to change one piece without breaking another.

---

## 3. How the data is generated

**File:** `src/data/generate_saas_data.py`

Since we don't have access to a real company's confidential data, we generate synthetic data that behaves like the real thing. The data is structured around **six tables**, each representing a different source of signal about a customer.

### The six tables

| Table | What it contains | Rows |
|---|---|---|
| `customers.csv` | One row per customer: segment, plan, industry, country | 1,000 |
| `subscriptions.csv` | Monthly revenue (MRR, ARR), renewal date, churn flag | ~11,000 |
| `product_usage.csv` | Monthly logins, active users, feature adoption, API calls | ~11,000 |
| `support.csv` | Monthly ticket count, sentiment score, resolution time | ~11,000 |
| `customer_success.csv` | Monthly NPS, health score, days since last CS touchpoint | ~11,000 |
| `targets.csv` | Company-level ARR target vs actual per month | 24 |

### Customer segments and plans

Customers are divided into three segments with different behaviors:

| Segment | Share | Annual Churn Rate | Typical Plan |
|---|---|---|---|
| SMB | 50% | 22% | Starter, Growth |
| Mid-Market | 35% | 12% | Growth, Professional |
| Enterprise | 15% | 6% | Professional, Enterprise |

Plan pricing ranges:
```
Starter:      $500 – $1,500 / month
Growth:       $1,500 – $5,000 / month
Professional: $5,000 – $15,000 / month
Enterprise:   $15,000 – $80,000 / month
```

### How churn is simulated realistically

The critical design choice: **customers who will eventually churn show deteriorating signals before they leave.** This is what makes the data useful for a predictive model.

For a customer who will churn:
- **Usage decays gradually**: logins and feature adoption trend downward over time
- **Support tickets increase**: they have more problems and open more tickets
- **Sentiment drops**: support interactions become more negative
- **NPS falls**: from ~6 toward ~4
- **CS touchpoints become infrequent**: no one is reaching out to save them

For a healthy customer:
- Usage stays stable or grows
- Tickets are rare
- Sentiment is positive
- NPS hovers around 6–8
- CS manager checks in regularly

This is how you create a dataset where a model can actually learn meaningful patterns.

```python
# Example from generate_product_usage():
# Churning customers show declining usage trajectory
if will_churn:
    decay = max(0.0, 1.0 - (m / max_month) * rng.uniform(0.3, 0.8))
else:
    decay = rng.uniform(0.7, 1.1)

logins = max(0, int(rng.normal(base_logins * decay, base_logins * 0.15)))
```

The `decay` factor shrinks toward zero as a churning customer approaches their final month.

### Reproducibility

All random number generation uses a fixed seed (`random_seed: 42` in the config). Run `generate_all()` twice and you get the exact same dataset. This is essential for reproducible experiments.

---

## 4. How features are built

**File:** `src/features/build_features.py`

Raw data comes in six separate tables. The ML model needs one flat row per customer-month, with every relevant signal as a column. The feature pipeline joins the tables and engineers derived columns that are more predictive than the raw signals alone.

### Step 1: Join all tables

```python
merged = subs.merge(usage,   on=["customer_id", "month"], how="left")
merged = merged.merge(support, on=["customer_id", "month"], how="left")
merged = merged.merge(cs,      on=["customer_id", "month"], how="left")
merged = merged.merge(customers, on="customer_id",          how="left")
```

After this, every row represents one customer in one month, with all signals in a single row.

### Step 2: Rolling averages and trends

A single month's data is noisy. A 3-month rolling average is more stable. More importantly, the *trend* (whether a signal is going up or down) is often more predictive than its current value.

```python
# Average logins over last 3 months
logins_roll3m = rolling mean of logins over window of 3

# Trend: is the customer logging in more or less than 3 months ago?
logins_trend3m = (logins_this_month - logins_3_months_ago) / logins_3_months_ago
```

A customer whose logins dropped 40% over three months is a much stronger churn signal than a customer with low-but-stable logins.

### Step 3: Lag features

What happened *last month* is often a strong predictor of what happens *this month*. We shift each key signal by one month:

```python
for col in ["mrr", "logins", "tickets", "nps", "health_score"]:
    merged[f"{col}_lag1"] = merged.groupby("customer_id")[col].shift(1)
```

We also compute the MRR change percentage month-over-month:
```python
mrr_change_pct = (mrr - mrr_lag1) / mrr_lag1
```

A customer whose MRR shrank (because they downgraded) is more likely to churn next.

### Step 4: Encode categorical columns

The model can't read strings. Segments, plans, industries, and countries are converted to 0/1 columns:

```python
merged = pd.get_dummies(merged, columns=["segment", "plan", "industry", "country"], drop_first=True)
```

For example, `segment` becomes `segment_Mid-Market` and `segment_SMB` (Enterprise is the baseline, so it gets dropped to avoid redundancy).

### The full feature list

After all steps, each row has 41 features. The most important ones:

| Feature | What it measures |
|---|---|
| `mrr` | Current monthly revenue |
| `months_to_renewal` | How soon the contract expires |
| `logins_trend3m` | Is usage increasing or decreasing? |
| `feature_adoption_roll3m` | Average product depth over last 3 months |
| `tickets_roll3m` | Total support tickets over last 3 months |
| `sentiment_roll3m` | Average support sentiment over last 3 months |
| `nps` | Customer satisfaction score |
| `health_score` | Composite health from CS team |
| `last_touch_days` | Days since last customer success interaction |
| `mrr_change_pct` | Did MRR grow or shrink this month? |

The first month per customer is dropped (because lag features are undefined for month 1).

**Result:** 10,388 rows × 41 features, with a churn rate of ~1.2% per month.

---

## 5. How the churn model is trained

**File:** `src/models/train_churn_model.py`

### The problem framing

This is a **binary classification** problem. For each customer-month row, the model predicts:

- **0** = customer did not churn this month
- **1** = customer churned this month

The model learns which combination of features (low usage + high tickets + renewal approaching) predicts a `1`.

### Why 1.2% churn rate matters

Only 1.2% of rows are churned customers. This means the dataset is imbalanced: there are about 83 non-churning rows for every 1 churning row. A naive model that always predicts "not churning" would be 98.8% accurate, but completely useless for the business.

To fix this, XGBoost is configured with `scale_pos_weight: 5`, which tells the model to treat each positive (churn) example as if it were worth 5 negative examples. This forces the model to focus on getting churns right.

### Two models are trained and compared

**Model 1: Logistic Regression (baseline)**
- Simple, linear, fast to train
- Requires feature scaling (StandardScaler)
- Sets a baseline to beat

**Model 2: XGBoost**
- Gradient-boosted decision trees
- Handles non-linear relationships between features
- Handles imbalanced data via `scale_pos_weight`
- Generally outperforms linear models on tabular data

Both are trained on 80% of the data and evaluated on the held-out 20%.

### Why we use ROC-AUC and PR-AUC, not accuracy

The right metric for an imbalanced classification problem:

**ROC-AUC** (Area Under the ROC Curve): measures how well the model ranks customers by churn risk. A score of 1.0 means perfect ranking; 0.5 means random.

**PR-AUC** (Precision-Recall AUC): measures the trade-off between catching real churners (recall) and not wasting CS resources on false alarms (precision). More informative than ROC-AUC when positive examples are rare.

Results from our run:
```
Logistic Regression: ROC-AUC = 0.97,  PR-AUC = 0.24
XGBoost:             ROC-AUC = 0.97,  PR-AUC = 0.32  ← winner
```

XGBoost wins on PR-AUC, meaning it is better at identifying actual churners without flooding the CS team with false alarms.

### MLflow tracks every experiment

Every training run is logged automatically:

```python
with mlflow.start_run(run_name="churn_experiment"):
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("churn_rate", round(float(y.mean()), 4))
    mlflow.log_metrics({"roc_auc": ..., "pr_auc": ..., "f1": ...})
    mlflow.xgboost.log_model(model, "model")
```

This means you can always compare runs, reproduce results, and see exactly which parameters produced which metrics. MLflow stores everything in the `mlruns/` folder.

### The saved outputs

After training, two files are saved to `models/`:
- `churn_model.pkl`: the trained XGBoost model
- `churn_features.pkl`: the list of feature names the model expects (order matters)

---

## 6. How ARR at risk is calculated

**File:** `src/risk/revenue_risk_score.py`

After the model produces a churn probability for each customer, we need to answer a more actionable question: **"Which customers should we act on first?"**

A 90% churn probability on a $5,000/year customer is less urgent than a 40% churn probability on a $500,000/year customer. The Revenue Risk Score combines three dimensions:

### The formula

```
Revenue Risk Score = 0.50 × churn_probability
                   + 0.30 × arr_exposure (normalized)
                   + 0.20 × renewal_urgency
```

The weights are set in `configs/model_config.yaml` and can be adjusted without changing any code.

**Churn probability (50% weight):** The raw model output. A customer at 90% churn risk scores 0.45 from this component alone.

**ARR exposure (30% weight):** Each customer's ARR is normalized to 0–1 across all customers. The highest-revenue customer scores 1.0; the lowest scores 0. This ensures that large accounts are prioritized.

**Renewal urgency (20% weight):**
- Contract renews in ≤ 3 months → urgency = 1.0
- Contract renews in 4–6 months → urgency = 0.5
- Contract renews in > 6 months → urgency = 0.0

The final score is clipped to [0, 1] and labeled:
- **High**: score ≥ 0.65
- **Medium**: score ≥ 0.35
- **Low**: score < 0.35

### ARR at risk (dollars)

A separate column makes the financial impact concrete:

```python
arr_at_risk = churn_probability × arr
```

If a customer has $300,000 ARR and 80% churn probability, the ARR at risk is $240,000. This is what the CS team or executive should see: a dollar figure, not a raw probability.

### The output

`data/reports/customer_risk_scores.csv`: one row per customer, sorted by risk score descending. This file feeds directly into the dashboard.

From our run:
```
High risk:   35 customers
Medium risk: 82 customers
Low risk:    838 customers

Highest-risk customer: C0852, $938,400 ARR, 96.8% churn probability, $908,725 at risk
```

---

## 7. How SHAP explanations work

**File:** `src/explainability/shap_explainer.py`

The churn model is a black box: it takes 41 numbers as input and returns a probability. SHAP (SHapley Additive exPlanations) opens the box and answers: **"For this specific customer, which features pushed the probability up or down, and by how much?"**

### The idea behind SHAP

SHAP values come from cooperative game theory. The idea: treat each feature as a "player" in a game where the payoff is the model's prediction. A feature's SHAP value measures how much it contributed to moving the prediction away from the average.

A positive SHAP value means: "this feature pushed the churn probability higher than average."
A negative SHAP value means: "this feature pulled the churn probability lower than average."

### TreeExplainer: fast SHAP for tree models

XGBoost is a tree-based model, so we use `shap.TreeExplainer`, which computes exact SHAP values efficiently (no sampling needed):

```python
def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
```

`shap_values` returns an array of shape `(n_rows, n_features)`. Each cell is the SHAP contribution of that feature for that row.

### Reading the output for one customer

For customer C0852 (the highest risk customer):

```
Customer C0852: 97% churn risk

Main risk drivers:
  1. Login Trend (3M): ↑ increases risk       ← usage dropping fast
  2. Feature Adoption Rate: ↑ increases risk   ← stopped using the product
  3. Total Tickets (3M): ↑ increases risk      ← many support issues
  4. Days Since Last CS Touch: ↑ increases risk ← nobody reached out
  5. Net Promoter Score: ↑ increases risk      ← unhappy customer
```

Each arrow and label comes directly from the SHAP value: positive SHAP → increases risk → "↑ increases risk".

### Feature labels for business users

Raw feature names like `logins_roll3m` are not readable for a VP of Sales. We maintain a dictionary that translates them:

```python
FEATURE_LABELS = {
    "logins_roll3m":              "Avg Logins (3M)",
    "feature_adoption_roll3m":    "Avg Feature Adoption (3M)",
    "tickets_roll3m":             "Total Tickets (3M)",
    "last_touch_days":            "Days Since Last CS Touch",
    "nps":                        "Net Promoter Score",
    ...
}
```

The dashboard shows the translated names.

### Global vs. per-customer SHAP

**Per-customer:** "Why is *this* customer at risk?" Used in the Customer Deep Dive page of the dashboard.

**Global (mean absolute SHAP):** "Which features matter most across *all* customers?" Computed by averaging the absolute SHAP values across all rows. This tells you what drives churn company-wide, useful for product or CS strategy decisions.

---

## 8. How the Streamlit dashboard works

**File:** `app/streamlit_app.py`

Streamlit converts Python code into an interactive web app. Each `st.metric()` call renders a KPI box. Each `st.plotly_chart()` call renders an interactive chart. The sidebar controls which page is shown.

The dashboard has four pages:

### Page 1: Executive Summary

**Purpose:** Give a VP or CFO the one-screen view they need.

**Data sources:**
- `data/synthetic/targets.csv` (for the ARR trend chart)
- `data/reports/customer_risk_scores.csv` (for the risk metrics)

**What it shows:**
- Current ARR vs. last month
- Forecast gap vs. target (are we going to hit the number?)
- Total ARR at risk from high-risk customers
- Count of high-risk customers
- ARR trend chart (Actual vs. Target vs. Forecast)
- Risk distribution pie chart (High / Medium / Low)

### Page 2: Churn Risk

**Purpose:** Give the CS or Sales team a prioritized customer list to act on.

**Data source:** `data/reports/customer_risk_scores.csv`

**What it shows:**
- Summary counts by risk level
- Top 20 customers sorted by Revenue Risk Score, with ARR, churn probability, and ARR at risk
- Distribution histogram of churn probabilities across all customers

### Page 3: ARR Forecast

**Purpose:** Show how model predictions compare to actual ARR over time.

**Data sources:**
- `data/processed/arr_features.csv` (for the model inputs)
- `models/arr_forecast_model.pkl` (the trained model)
- `data/synthetic/targets.csv` (for the target line)

**What it shows:**
- A chart with three lines: Actual ARR, Model-Predicted ARR, and Target ARR

### Page 4: Customer Deep Dive

**Purpose:** Let a CS manager investigate one specific customer in detail.

**What it does:**
1. User selects a customer from a dropdown
2. Dashboard shows their KPIs: churn probability, risk level, ARR, ARR at risk, NPS, health score, days since CS touch
3. Computes SHAP values on-the-fly for that customer's most recent month
4. Displays the top 7 risk drivers with direction arrows

**Caching:** Functions decorated with `@st.cache_data` and `@st.cache_resource` run only once. After the first load, results are stored in memory. This makes subsequent page interactions fast.

---

## 9. Full pipeline flow

Here is the complete journey from nothing to a working dashboard:

```
configs/model_config.yaml
        │
        │  (parameters: n_customers=1000, n_months=24, seed=42)
        ▼
[1] src/data/generate_saas_data.py
        │
        │  generates 6 CSV tables in data/synthetic/
        │  customers, subscriptions, product_usage, support, customer_success, targets
        ▼
[2] src/features/build_features.py
        │
        │  joins all 6 tables into one flat DataFrame
        │  engineers: rolling means, trends, lag features, one-hot encoding
        │  saves:
        │    data/processed/churn_features.csv   (10,388 rows × 47 columns)
        │    data/processed/arr_features.csv     (22 rows × 12 columns)
        ▼
[3a] src/models/train_churn_model.py
        │
        │  splits data: 80% train / 20% test (stratified by churn label)
        │  trains: Logistic Regression baseline → XGBoost (winner)
        │  evaluates: ROC-AUC, PR-AUC, F1, Precision, Recall
        │  logs everything to MLflow (mlruns/)
        │  saves: models/churn_model.pkl, models/churn_features.pkl
        │
[3b] src/models/train_arr_forecast.py
        │
        │  uses TimeSeriesSplit (no shuffling, time order preserved)
        │  trains XGBoost regressor on lag + rolling ARR features
        │  saves: models/arr_forecast_model.pkl, models/arr_features.pkl
        ▼
[4] src/risk/revenue_risk_score.py
        │
        │  loads churn_model.pkl
        │  predicts churn probability for every row in churn_features.csv
        │  takes the latest month per customer
        │  computes Revenue Risk Score:
        │    score = 0.50 × churn_prob + 0.30 × arr_norm + 0.20 × renewal_urgency
        │  labels: High / Medium / Low
        │  computes: arr_at_risk = churn_prob × arr
        │  saves: data/reports/customer_risk_scores.csv
        ▼
[5] app/streamlit_app.py
        │
        │  reads: targets.csv, customer_risk_scores.csv, churn_features.csv
        │  loads: churn_model.pkl, arr_forecast_model.pkl
        │  renders: 4-page interactive dashboard
        │  on Customer Deep Dive: runs SHAP on-the-fly for selected customer
        ▼
     Browser: live dashboard at http://localhost:8501
```

**Everything is controlled by `run_pipeline.py`**, which runs steps 1–4 in sequence. Step 5 is the dashboard, launched separately.

---

## 10. How to run everything

### Prerequisites

```bash
cd C:\Users\ronik\reviq-ai
pip install -r requirements.txt
```

### Run the full pipeline (generates data + trains models + computes risk scores)

```bash
python run_pipeline.py
```

This takes about 30–60 seconds and produces all the files in `data/` and `models/`.

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

### Launch the API (optional)

```bash
uvicorn api.main:app --reload
```

API docs at `http://localhost:8000/docs`.

### Run tests

```bash
pytest tests/ -v
```

### Explore MLflow experiment results

```bash
mlflow ui
```

Opens at `http://localhost:5000`. Shows all training runs, metrics, and model artifacts.

### Change parameters without editing code

All key parameters live in `configs/model_config.yaml`:

```yaml
data:
  n_customers: 1000    # change to 5000 for a larger dataset
  n_months: 24         # change to 36 for 3 years of history

churn_model:
  xgboost:
    n_estimators: 300  # more trees = slower but potentially better
    scale_pos_weight: 5 # higher = model focuses more on catching churners

risk_score:
  weights:
    churn_probability: 0.50  # adjust to reprioritize the score formula
    arr_exposure: 0.30
    renewal_urgency: 0.20
  thresholds:
    high: 0.65   # lower this to flag more customers as high-risk
    medium: 0.35
```

After changing the config, re-run `python run_pipeline.py` to retrain.
