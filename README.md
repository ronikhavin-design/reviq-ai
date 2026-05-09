# RevIQ AI — SaaS Revenue Intelligence Platform

RevIQ AI is an end-to-end ML platform that predicts customer churn, forecasts ARR, and explains revenue risk for SaaS companies. It combines machine learning, SHAP explainability, and a FastAPI + Streamlit stack to give CS and Finance teams a single view of who is at risk, why, and what it costs.

Built as a portfolio project to demonstrate ML engineering, financial reasoning, and production system design — inspired by real FP&A work at an enterprise software company.

---

## Business Problem

SaaS companies lose revenue when customers churn silently. By the time a CS manager notices the signal, the renewal conversation is already too late. The problem has three layers:

1. **Detection:** Who is likely to churn — and how confident are we?
2. **Prioritization:** Which customers represent the most ARR at risk right now?
3. **Explanation:** Why is this customer flagged? What specific signals drove the model's prediction?

RevIQ AI addresses all three.

---

## Results

| Metric | Value |
|---|---|
| Churn model ROC-AUC | **0.97** |
| Churn model PR-AUC | **0.32** (vs 0.012 no-skill baseline) |
| High-risk customers identified | **35** |
| Medium-risk customers identified | **82** |
| Top customer ARR at risk | **$908,725** |
| Test suite | **7 tests passing** |

---

## Dashboard

> **Screenshot placeholder — run `streamlit run app/streamlit_app.py` to see it live**

The dashboard has 4 pages:
- **Executive Summary** — ARR vs target, risk distribution, top 10 at-risk accounts
- **Churn Risk** — full customer risk table, sortable by revenue at risk
- **ARR Forecast** — 3-month forward ARR projection vs actuals
- **Customer Deep Dive** — per-customer SHAP explanation of churn drivers

---

## Architecture

```
Synthetic Data (6 tables)
        │
        ▼
Feature Engineering (41 features: rolling means, lags, one-hot)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  XGBoost Churn Model          XGBoost ARR Forecast Model    │
│  ROC-AUC: 0.97                TimeSeriesSplit CV            │
│  PR-AUC:  0.32                MAPE tracked in MLflow        │
│  Tracked in MLflow                                          │
└──────────────────┬──────────────────────┬────────────────────┘
                   │                      │
                   ▼                      ▼
          SHAP Explainability      Revenue Risk Score
          (per-customer +          0.50 × churn_prob
           global importance)    + 0.30 × ARR exposure
                   │              + 0.20 × renewal urgency
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   FastAPI Service   │
                   │  /predict-churn     │
                   │  /revenue-risk-summary │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Streamlit Dashboard│
                   │  4 pages           │
                   └─────────────────────┘
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Data generation | Python, NumPy | Realistic decay-based churn simulation |
| Feature engineering | pandas | Rolling means, trends, lag features, one-hot encoding |
| Churn model | XGBoost | Handles class imbalance via `scale_pos_weight`, outperforms LR on PR-AUC |
| ARR forecast | XGBoost + TimeSeriesSplit | No data leakage; respects temporal ordering |
| Explainability | SHAP TreeExplainer | Per-customer and global feature attribution |
| Risk scoring | Custom formula | Composite of churn prob + ARR exposure + renewal urgency |
| Experiment tracking | MLflow | Nested runs: parent experiment → LR baseline + XGBoost |
| API | FastAPI + Pydantic | Typed request/response schemas, production-ready |
| Dashboard | Streamlit | 4-page app with caching; per-customer SHAP visualization |
| Tests | pytest | 7 tests covering data generation, features, risk score bounds |
| Containerization | Docker + docker-compose | One-command local deployment |
| Config | YAML | All tunable parameters in one file, no hardcoded values |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd reviq-ai
pip install -r requirements.txt

# 2. Run the full pipeline (generates data, trains models, scores customers)
python run_pipeline.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# 4. Or start the API
uvicorn api.main:app --reload
# POST /predict-churn, GET /revenue-risk-summary

# 5. Or run with Docker
docker-compose up
```

---

## Project Structure

```
reviq-ai/
├── configs/
│   └── model_config.yaml         # All tunable parameters
├── src/
│   ├── data/
│   │   └── generate_saas_data.py # 6-table synthetic SaaS generator
│   ├── features/
│   │   └── build_features.py     # Feature engineering pipeline (41 features)
│   ├── models/
│   │   ├── train_churn_model.py  # XGBoost + LR baseline, MLflow tracking
│   │   └── train_arr_forecast.py # XGBoost regressor, TimeSeriesSplit CV
│   ├── risk/
│   │   └── revenue_risk_score.py # Composite risk score + ARR-at-risk table
│   └── explainability/
│       └── shap_explainer.py     # SHAP values, global + per-customer charts
├── api/
│   └── main.py                   # FastAPI service
├── app/
│   └── streamlit_app.py          # 4-page Streamlit dashboard
├── notebooks/
│   ├── 01_eda_and_business_insights.ipynb  # Phase 2: EDA + 8 business findings
│   └── 02_model_evaluation.ipynb           # Phase 2: ROC, PR, confusion, SHAP, calibration
├── tests/
│   └── test_features.py          # 7 pytest tests
├── run_pipeline.py               # End-to-end orchestrator
├── Dockerfile
├── docker-compose.yml
└── ROADMAP.md                    # Phases 2–6 implementation plan
```

---

## Phase 2: Data Science Analysis Layer (added)

Phase 1 built the working MVP. Phase 2 adds the analytical depth expected from a production data science system.

### `notebooks/01_eda_and_business_insights.ipynb`

Exploratory analysis of 1,000 synthetic SaaS customers across 24 months:

- Customer portfolio breakdown by segment, plan, industry, country
- ARR distribution and churn rate by segment and plan (Enterprise churns the least; SMB churns the most)
- Usage signal comparison: healthy vs churned customers across 6 behavioral dimensions
- Renewal timing distribution and associated risk
- 24-month ARR trend across active, churned, and all customers
- 8 key business findings summarized with dollar implications

### `notebooks/02_model_evaluation.ipynb`

Full model evaluation with business context:

- **Class imbalance visualization** — shows why accuracy is a broken metric (predicting no churn for every customer achieves 98%+ accuracy while catching zero churners)
- **ROC vs PR curve comparison** — demonstrates that ROC-AUC is misleading for imbalanced data; PR-AUC is the right lens
- **Confusion matrix** — translates model errors into business costs: false negative = lost ARR, false positive = wasted CS call
- **Calibration plot** — checks whether predicted probabilities can be trusted as inputs to ARR-at-risk calculations
- **SHAP global importance** — which features actually drive churn predictions across the full customer base

### SHAP visualization improvements (`src/explainability/shap_explainer.py`)

Two new functions added:

- `plot_global_importance()` — publication-quality horizontal bar chart of mean |SHAP| per feature, saved as PNG to `data/reports/plots/`
- `plot_customer_bar()` — per-customer diverging bar chart: red bars increase churn risk, blue bars reduce it; saved per customer ID

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Data, features, models, risk score, dashboard, API, tests, Docker | ✅ Done |
| 2 | EDA notebook, model evaluation notebook, SHAP chart improvements | ✅ Done |
| 3 | Scenario simulator, retention budget optimizer (scipy) | Planned |
| 4 | RAG Revenue Analyst Copilot (LangChain + OpenAI + Chroma) | Planned |
| 5 | Monitoring (Evidently AI), CI (GitHub Actions), cloud deployment | Planned |
| 6 | README polish, CV bullets, LinkedIn post | Planned |

See [`ROADMAP.md`](ROADMAP.md) for full scope and rationale for each phase.

---

## Key Design Decisions

**Why PR-AUC over ROC-AUC?** With a 1.2% monthly churn rate, accuracy is misleading: a model that never flags any churn still achieves 98%+ accuracy. ROC-AUC is also unreliable here — it is dominated by the large number of true negatives, which are abundant when the positive class is rare. Both models score 0.97 on ROC-AUC despite XGBoost being 33% better at actually finding churners. PR-AUC ignores true negatives entirely and directly measures performance on the rare class we care about.

**Why XGBoost over Logistic Regression?** Both models reach ROC-AUC 0.97. But XGBoost achieves PR-AUC 0.32 vs LR's 0.24 — a 33% improvement in practical detection ability. XGBoost captures non-linear interactions between usage trends, NPS, and renewal timing that LR cannot model.

**Why `scale_pos_weight=5`?** The business cost of a false negative (missed churner = lost ARR) is higher than a false positive (unnecessary CS call). This parameter encodes that asymmetry directly into the loss function.

**Why SHAP over built-in feature importance?** XGBoost's native importance counts split frequency, which is biased toward high-cardinality features. SHAP measures the actual magnitude of each feature's contribution to each individual prediction — essential for explaining why a specific customer was flagged.

---

## Author

Built by Roni Khavin — FP&A Analyst transitioning into AI/ML engineering.
