# RevIQ AI: SaaS Revenue Intelligence Platform

RevIQ AI is an end-to-end ML and LLM platform that predicts customer churn, forecasts ARR, explains revenue risk, and answers plain-English business questions using a local RAG copilot. It gives CS and Finance teams a single view of who is at risk, why, and what to do about it.

Built as a portfolio project to demonstrate ML engineering, LLM integration, financial reasoning, and production system design, inspired by real FP&A work at an enterprise software company.

---

## Business Problem

SaaS companies lose revenue when customers churn silently. By the time a CS manager notices the signal, the renewal conversation is already too late. The problem has four layers:

1. **Detection:** Who is likely to churn, and how confident are we?
2. **Prioritization:** Which customers represent the most ARR at risk right now?
3. **Explanation:** Why is this customer flagged? What signals drove the prediction?
4. **Action:** How should we allocate our retention budget? Which customers should CS call first?

RevIQ AI addresses all four.

---

## Results

| Metric | Value |
|---|---|
| Churn model ROC-AUC | **0.97** |
| Churn model PR-AUC | **0.32** (vs 0.012 no-skill baseline) |
| High-risk customers identified | **35** |
| Medium-risk customers identified | **82** |
| ARR at risk (high-risk segment) | **$8.02M** |
| Top customer ARR at risk | **$908,725** |
| Test suite | **170 tests passing** |
| RAG knowledge base | **28 chunks across 5 reports** |

---

## Architecture

```
Synthetic Data (6 tables: customers, subscriptions, usage, support, CS, targets)
        |
        v
Feature Engineering  -- 47 features: rolling means, lags, trends, one-hot encoding
        |
        v
+---------------------------------------------------------------+
|  XGBoost Churn Model          XGBoost ARR Forecast Model     |
|  ROC-AUC: 0.97                TimeSeriesSplit CV             |
|  PR-AUC:  0.32                tracked in MLflow              |
|  tracked in MLflow                                           |
+-------------------+-------------------+-----------------------+
                    |                   |
                    v                   v
           SHAP Explainability    Revenue Risk Score
           per-customer +         0.50 x churn_prob
           global importance    + 0.30 x ARR exposure
                    |           + 0.20 x renewal urgency
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Decision Layer    |
                    |  Scenario Simulator |
                    | Retention Optimizer |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Report Generation   |   <-- Phase 4a
                    |  5 Markdown reports |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   RAG Pipeline      |   <-- Phase 4b/4c
                    |  TF-IDF chunking    |
                    |  vector index       |
                    |  LLM answer gen     |
                    |  (OpenAI optional)  |
                    +----------+----------+
                               |
               +---------------+---------------+
               |                               |
    +----------v----------+        +----------v----------+
    |   FastAPI Service   |        | Streamlit Dashboard |
    |  /predict-churn     |        |  6 pages            |
    |  /revenue-risk      |        |  Revenue Copilot    |
    +---------------------+        +---------------------+
```

---

## Dashboard

Run `streamlit run app/streamlit_app.py` to launch. The dashboard has 6 pages:

| Page | What it shows |
|---|---|
| **Executive Summary** | ARR vs target, risk distribution pie chart, revenue trend |
| **Churn Risk** | Full customer risk table sortable by ARR at risk, histogram |
| **ARR Forecast** | Actual vs predicted vs target ARR over time |
| **Customer Deep Dive** | Per-customer SHAP churn drivers, metrics, risk level |
| **Retention Planning** | Scenario Simulator (campaign ARR impact) + Retention Optimizer (ranked call list) |
| **Revenue Copilot** | Plain-English Q&A over the knowledge base, with source citations and evidence |

---

## Revenue Copilot

The Revenue Copilot answers plain-English business questions grounded in the knowledge base generated from the pipeline outputs. No hallucination: every answer cites the specific report section it came from.

**How it works:**

1. **Report generation** (`src/reports/generate_reports.py`): After each pipeline run, 5 structured Markdown reports are written to `data/reports/markdown/`. They contain pre-interpreted business language ("35 High-risk customers hold $8.02M in ARR at risk") rather than raw numbers.

2. **Chunking and retrieval** (`src/rag/`): Each report is split at `##` section boundaries into chunks. Chunks are embedded using TF-IDF (scikit-learn) and stored in a compressed local vector index. No external vector database required. Install `sentence-transformers` to upgrade to semantic embeddings automatically.

3. **Answer generation** (`src/rag/copilot.py`): The top-k most relevant chunks are sent to OpenAI `gpt-4o-mini` as grounded context. The LLM produces a cited, business-language answer.

4. **Fallback mode**: If `OPENAI_API_KEY` is not set, the Copilot displays the retrieved chunks directly with source attribution. No exception is raised and the dashboard remains fully functional.

Example questions the Copilot answers:
- "Which customers are most at risk of churning?"
- "What is our current ARR risk summary?"
- "What should the CS team focus on this week?"
- "How is the churn model performing?"

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
| Decision layer | Pure Python | Scenario Simulator + constrained Retention Optimizer |
| Experiment tracking | MLflow | Nested runs: parent + LR baseline + XGBoost |
| RAG reports | Python + Markdown | Business-language knowledge base generated from pipeline outputs |
| RAG retrieval | scikit-learn TF-IDF + numpy | Local vector index, no external database |
| RAG answer gen | OpenAI gpt-4o-mini | Optional; falls back gracefully without an API key |
| API | FastAPI + Pydantic | Typed request/response schemas |
| Dashboard | Streamlit | 6-page app with caching and session state |
| Visualization | Plotly | Interactive charts throughout the dashboard |
| Tests | pytest | 170 tests, all run without external API keys |
| CI | GitHub Actions | Runs on every push; installs deps and runs full test suite |
| Containerization | Docker + docker-compose | One-command local deployment |
| Config | YAML | All tunable parameters in one file |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd reviq-ai
pip install -r requirements.txt

# 2. Run the full pipeline
#    Generates data, trains models, scores customers,
#    builds RAG reports, and indexes the knowledge base.
python run_pipeline.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# 4. Optional: start the FastAPI service
uvicorn api.main:app --reload
# POST /predict-churn
# GET  /revenue-risk-summary
# GET  /health

# 5. Optional: run with Docker
docker-compose up --build
```

---

## Developer Commands

A `Makefile` is included for common tasks:

```bash
make install     # pip install -r requirements.txt
make test        # python -m pytest tests/ -v
make pipeline    # python run_pipeline.py  (all 6 steps)
make app         # streamlit run app/streamlit_app.py
make api         # uvicorn api.main:app --reload
make reports     # regenerate Markdown reports only (Phase 4a)
make index       # rebuild RAG vector index only (Phase 4b)
make docker-up   # docker-compose up --build
```

---

## Environment Variables

Copy `.env.example` to `.env` to configure optional settings:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables LLM-generated answers in the Revenue Copilot. Without it, the Copilot runs in fallback mode and shows retrieved text summaries. |

The project is fully functional without any API keys.

---

## Project Structure

```
reviq-ai/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions: install + pytest on every push
├── configs/
│   └── model_config.yaml             # All tunable parameters
├── src/
│   ├── config.py                     # Paths and config loader
│   ├── data/
│   │   └── generate_saas_data.py     # 6-table synthetic SaaS generator
│   ├── features/
│   │   └── build_features.py         # Feature engineering pipeline (47 features)
│   ├── models/
│   │   ├── train_churn_model.py      # XGBoost + LR baseline, MLflow tracking
│   │   └── train_arr_forecast.py     # XGBoost regressor, TimeSeriesSplit CV
│   ├── risk/
│   │   └── revenue_risk_score.py     # Composite risk score + ARR-at-risk table
│   ├── explainability/
│   │   └── shap_explainer.py         # SHAP values, global + per-customer charts
│   ├── scenarios/
│   │   ├── scenario_simulator.py     # Campaign-level ARR impact simulator
│   │   └── retention_optimizer.py    # Constrained per-customer budget optimizer
│   ├── reports/
│   │   └── generate_reports.py       # Phase 4a: 5 Markdown reports from pipeline outputs
│   └── rag/
│       ├── document_loader.py        # Phase 4b: Markdown chunking at ## boundaries
│       ├── vector_index.py           # Phase 4b: TF-IDF embedding + joblib index
│       ├── retriever.py              # Phase 4b: cosine similarity retrieval
│       └── copilot.py                # Phase 4c: LLM answer gen with fallback
├── api/
│   └── main.py                       # FastAPI service: /predict-churn, /revenue-risk
├── app/
│   └── streamlit_app.py              # 6-page Streamlit dashboard
├── notebooks/
│   ├── 01_eda_and_business_insights.ipynb
│   └── 02_model_evaluation.ipynb
├── tests/
│   ├── test_features.py              # Data generation + feature engineering (7 tests)
│   ├── test_report_generation.py     # Phase 4a: report generation (37 tests)
│   ├── test_rag_retrieval.py         # Phase 4b: chunking + index + retrieval (45 tests)
│   ├── test_rag_copilot.py           # Phase 4c: copilot + fallback (49 tests)
│   ├── test_scenario_simulator.py    # Scenario simulator (16 tests)
│   └── test_retention_optimizer.py   # Retention optimizer (16 tests)
├── run_pipeline.py                   # End-to-end orchestrator (6 steps)
├── Makefile                          # Developer workflow shortcuts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── ROADMAP.md
```

---

## Key Design Decisions

**Why PR-AUC over ROC-AUC?** With a 1.2% monthly churn rate, accuracy is misleading: a model that never flags any churn still achieves 98%+ accuracy. ROC-AUC is also unreliable here because it is dominated by the large number of true negatives. Both models score 0.97 on ROC-AUC despite XGBoost being 33% better at actually finding churners. PR-AUC ignores true negatives entirely and directly measures performance on the rare class we care about.

**Why XGBoost over Logistic Regression?** Both models reach ROC-AUC 0.97. But XGBoost achieves PR-AUC 0.32 vs LR's 0.24, a 33% improvement in practical detection ability. XGBoost captures non-linear interactions between usage trends, NPS, and renewal timing that LR cannot model.

**Why `scale_pos_weight=5`?** The business cost of a false negative (missed churner = lost ARR) is higher than a false positive (unnecessary CS call). This parameter encodes that asymmetry directly into the loss function.

**Why SHAP over built-in feature importance?** XGBoost's native importance counts split frequency, which is biased toward high-cardinality features. SHAP measures the actual magnitude of each feature's contribution to each individual prediction, essential for explaining why a specific customer was flagged.

**Why a local RAG pipeline instead of LangChain + Chroma?** The original roadmap planned to use LangChain and ChromaDB. The custom implementation is simpler (no framework abstractions), has zero cloud dependencies, is fully testable without API keys, and produces the same output: grounded, cited business answers. The only trade-off is that upgrading to more advanced retrieval strategies requires more manual work.

**Why TF-IDF as the default embedding backend?** TF-IDF is available in scikit-learn with no model downloads and no hardware requirements. It works well for the current knowledge base because the reports use consistent business vocabulary. Install `sentence-transformers` and the system upgrades to semantic embeddings automatically with no code changes.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Synthetic data, feature engineering, XGBoost models, SHAP, risk score, FastAPI, Streamlit (4 pages), Docker, MLflow | Done |
| 2 | EDA notebook, model evaluation notebook, SHAP chart improvements | Done |
| 3 | Scenario Simulator, Retention Budget Optimizer, Retention Planning dashboard page | Done |
| 4a | Report generation: 5 Markdown reports from pipeline outputs | Done |
| 4b | Local RAG retrieval: TF-IDF chunking, vector index, cosine similarity | Done |
| 4c | Copilot answer generation: OpenAI LLM with structured fallback | Done |
| 4d/4e | Revenue Copilot dashboard page with badge, top-k control, evidence expanders | Done |
| 5a | Fix requirements.txt, add .env.example, extend run_pipeline.py | Done |
| 5b | Makefile, GitHub Actions CI | Done |
| 5c | README rewrite | Done |
| Future | Semantic embeddings (sentence-transformers), Docker polish, cloud deployment | Planned |

---

## Author

Built by Roni Khavin, FP&A Analyst transitioning into AI/ML engineering.
