# RevIQ AI: Product Roadmap

This document describes how we will evolve the current MVP into a complete SaaS Revenue Intelligence & Retention Copilot. Each phase has a clear scope, a rationale, and the CV/interview value it unlocks.

The MVP (Phase 1) is already built and working. This roadmap covers Phases 2–6.

---

## Current state: Phase 1 (done)

What exists today:

- Synthetic SaaS data generator (1,000 customers × 24 months, 6 tables)
- Feature engineering pipeline (rolling means, lag features, one-hot encoding → 41 features)
- Churn prediction model: XGBoost, ROC-AUC 0.97, PR-AUC 0.32, tracked in MLflow
- ARR forecasting model: XGBoost regressor with time-series cross-validation
- Revenue Risk Score: composite of churn probability + ARR exposure + renewal urgency
- SHAP explainability: per-customer and global feature importance
- Streamlit dashboard: 4 pages (Executive Summary, Churn Risk, ARR Forecast, Customer Deep Dive)
- FastAPI prediction service: `/predict-churn`, `/revenue-risk-summary`
- pytest test suite: 7 tests, all passing
- Docker + docker-compose
- MLflow experiment tracking

**Gap:** The product works end-to-end, but it doesn't yet show enough *business insight*. The Executive Summary is informative but passive: it shows numbers, not recommendations. There is no way to simulate scenarios, optimize a budget, or ask questions in natural language.

---

## Phase 2: Richer model evaluation and business insight

**Goal:** Make the existing models more informative and credible. Add the outputs that matter for a real business decision.

### 2a. Proper model evaluation notebook

Right now, model performance is logged to MLflow but never visualized. We need a notebook that produces publication-quality charts:

- **ROC curve** and **Precision-Recall curve** for both models on the same axes, so the trade-off is visible
- **Confusion matrix** with business interpretation: "Of the 125 customers who churned in the test set, we correctly flagged 89 of them (recall = 71%). Of the customers we flagged, 43% actually churned (precision = 43%)."
- **Feature importance bar chart** from SHAP global values: which signals matter most company-wide?
- **Calibration plot**: when the model says 70% churn probability, is the actual rate actually 70%? If not, the numbers shown to executives are misleading.

**Why this matters for interviews:** Anyone can train a model. Explaining *why* your evaluation choices reflect business reality is what distinguishes a data scientist from someone who just runs notebooks.

### 2b. Customer segmentation

Add a module that clusters customers by behavior, independent of the churn model. K-Means or hierarchical clustering on usage + support + revenue features.

Purpose: discover customer archetypes. For example:
- **Power users**: high logins, high feature adoption, low tickets, expanding ARR
- **At-risk engagers**: medium usage but deteriorating, not yet churning
- **Disengaged**: low usage, low NPS, waiting to leave
- **High-touch dependents**: high support tickets but also high NPS, sticky but costly

Each cluster gets a label and a recommended action, displayed on the dashboard.

**CV value:** "Applied unsupervised learning to segment customers by behavioral patterns, enabling targeted retention strategies per cluster."

### 2c. SHAP visualization improvements

Right now SHAP outputs are text-only in the dashboard. Add:

- A waterfall chart for one customer, showing the model's base prediction, then each feature pushing it up or down, ending at the final prediction
- A beeswarm plot for global importance, showing how each feature affects predictions across the entire customer base
- Color coding: features pushing toward churn in red, features reducing risk in blue

These are standard SHAP visualizations that interviewers will recognize immediately.

---

## Phase 3: Executive intelligence layer

**Goal:** Move from "showing data" to "recommending actions." This is the difference between a reporting tool and an intelligence platform.

### 3a. Scenario simulator

**The question it answers:** "What happens to our ARR if we reduce churn in the Enterprise segment by 5%?"

The simulator lets a user move a slider (or type in a number) and immediately sees the projected impact on ARR.

How it works:
- User selects a segment (SMB / Mid-Market / Enterprise) and a churn reduction target (e.g., -3%)
- The system identifies all customers in that segment currently flagged as Medium or High risk
- It recalculates ARR at risk with the lower churn probability
- It shows the delta: "Reducing Enterprise churn by 5% would save $2.3M in ARR over the next 12 months"

This is a simple calculation but it has high perceived value because executives understand dollars saved, not model metrics.

**CV value:** "Built a revenue scenario simulator that quantifies the ARR impact of targeted churn reduction by segment."

### 3b. Retention budget optimizer

**The question it answers:** "We have $200,000 to spend on customer success and retention discounts. Which customers should we focus on, and how should we allocate the budget?"

This is a **constrained optimization problem**:

```
Maximize:  total ARR saved
Subject to:
  - Total retention budget ≤ $200,000
  - CS capacity ≤ 150 hours per month
  - Discount per customer ≤ 20% of ARR
  - Enterprise customers must receive a touchpoint
```

The optimizer uses `scipy.optimize` or `cvxpy` to solve this and returns:
- A ranked list of which customers to prioritize
- How much budget to allocate to each
- Expected ARR saved vs. cost spent (ROI estimate)

**Why this is powerful:** It directly connects the ML model to a business decision with a dollar figure attached. A VP who doesn't understand SHAP values will immediately understand "we should spend $45,000 to save $380,000."

**CV value:** "Built a constrained budget optimization engine using scipy to recommend customer retention investment under capacity and financial constraints."

### 3c. Win/Loss analysis module

Track which high-risk customers were saved vs. which churned, and compute:
- Retention rate by segment
- Average time from flag to intervention
- Which SHAP drivers were most predictive of actual churn vs. false alarms

This requires a feedback loop in the data, simulating that some customers flagged as high-risk receive an intervention and have their churn probability reduced.

---

## Phase 4: RAG Revenue Analyst Copilot

**Goal:** Allow any user (including non-technical ones) to ask questions about the data and get clear, sourced answers.

### What it does

A chat interface where a user can type:

> "Why is ARR below target this quarter?"
> "Which customers should CS focus on this week?"
> "Summarize the main churn drivers in the Mid-Market segment."
> "What would happen to revenue if we lost our top 5 Enterprise customers?"
> "Give me a retention brief for customer C0852."

The system reads from the model outputs, risk reports, and monthly summaries (not from the raw CSV files) and returns a clear business answer with references to the specific data it used.

### How it works (RAG architecture)

**Step 1: Report generation:** After each pipeline run, automatically generate structured Markdown reports:
- Monthly revenue summary (ARR, churn, expansion, target gap)
- Segment-level churn summary
- Top 20 at-risk customers with SHAP drivers
- Model performance summary

**Step 2: Ingestion and chunking:** Each report is split into chunks (paragraphs or sections), each chunk gets metadata: report date, segment, topic.

**Step 3: Embeddings:** Each chunk is converted to a vector embedding using the OpenAI embeddings API or a local model (sentence-transformers).

**Step 4: Vector store:** Embeddings are stored in Chroma (a local vector database). This allows fast semantic search.

**Step 5: Retrieval:** When a user asks a question, the question is also embedded. The vector store finds the most relevant chunks. The top 5-10 chunks are passed to the LLM as context.

**Step 6: Generation:** The LLM (via OpenAI API + LangChain) generates an answer grounded in the retrieved context, with citations like: *"According to the May 2026 segment report, Mid-Market churn increased by 2.1 percentage points, primarily driven by declining feature adoption."*

### Tech stack for this phase

| Component | Tool |
|---|---|
| LLM | OpenAI API (gpt-4o or gpt-4o-mini) |
| RAG framework | LangChain |
| Embeddings | OpenAI embeddings or sentence-transformers |
| Vector database | Chroma (local, no server needed) |
| Report generation | Python, Markdown templates |
| Chat UI | Streamlit `st.chat_message` |

**CV value:** "Implemented a RAG-based Revenue Analyst Copilot using LangChain, OpenAI APIs, embeddings and vector search to answer business questions over forecast reports and model outputs."

---

## Phase 5: Production engineering layer

**Goal:** Make the project look and behave like a real production system, not just a research project.

### 5a. Monitoring

When a model is deployed, it can degrade silently, either because the input data changes (data drift) or because the model's predictions become less accurate (model performance drift).

We will add:
- **Data drift report** using Evidently AI: each time new data arrives, compare the distribution of each feature against the training data distribution. Flag features that have shifted.
- **Model performance monitoring:** track ROC-AUC and PR-AUC on a rolling basis (simulated with the synthetic data by treating later months as "new" data)
- A monitoring page in the dashboard: a table of features and their drift status, a chart of model performance over time

### 5b. API test suite

Right now we have tests for data generation and feature engineering. We need to add:
- Tests for the FastAPI endpoints using `httpx` (the test client)
- Tests that verify the risk score formula is correct with known inputs
- Tests that catch model drift: assert that PR-AUC on a holdout set never drops below a threshold

### 5c. GitHub Actions CI

A workflow that runs automatically on every push to GitHub:

```yaml
# .github/workflows/ci.yml
on: [push]
jobs:
  test:
    steps:
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      - run: python run_pipeline.py
```

This proves to any reviewer that the project actually runs from scratch, not just on your machine.

### 5d. Cloud deployment

Deploy the dashboard and API to a cloud environment:

- **Streamlit Community Cloud** for the dashboard (free, public URL)
- **Render or Railway** for the FastAPI service (free tier available)
- Or **AWS**: S3 for data storage, EC2 or Elastic Beanstalk for the API, ECR for the Docker image

The goal is a live URL you can put in your CV and LinkedIn post.

---

## Phase 6: GitHub README and CV positioning

**Goal:** Package everything as a professional portfolio piece.

### 6a. README.md

A strong GitHub README is the first thing a recruiter or hiring manager sees. It should:

- Open with one paragraph explaining the business problem and solution
- Show a screenshot or GIF of the dashboard
- Have a clear "Architecture" section with a diagram
- List every technology used and *why* it was chosen
- Include quick-start instructions that actually work
- Link to the live demo

Structure:
```
# RevIQ AI

[Screenshot of dashboard]

RevIQ AI is an end-to-end ML platform that predicts customer churn,
forecasts ARR, and explains revenue risk for SaaS companies.

## Live Demo
[link to Streamlit app]

## Architecture
[diagram: data → features → models → API → dashboard → RAG copilot]

## Tech Stack
[table with components and tools]

## Quick Start
[3 commands to run the project]

## Results
- XGBoost churn model: ROC-AUC 0.97, PR-AUC 0.32
- 35 high-risk customers identified, $X.XM ARR at risk
- RAG copilot answers executive questions over revenue reports

## Project Structure
[tree]
```

### 6b. CV bullets

After all phases are complete, the CV section could read:

```
RevIQ AI: End-to-End ML and LLM Platform for SaaS Revenue Intelligence
github.com/[username]/reviq-ai | [live demo link]

• Built an end-to-end ML platform for SaaS churn prediction, ARR forecasting,
  and revenue risk scoring using Python, pandas, scikit-learn, and XGBoost.

• Developed a Revenue Risk Score combining churn probability, ARR exposure, and
  renewal urgency to prioritize customer retention interventions.

• Implemented SHAP explainability to translate model outputs into business-readable
  churn drivers for customer success and finance teams.

• Built a constrained retention budget optimizer using scipy to recommend customer
  intervention allocation under financial and CS capacity constraints.

• Implemented a RAG-based Revenue Analyst Copilot using LangChain, OpenAI APIs,
  embeddings, and vector search to answer executive questions over revenue reports.

• Deployed model predictions through FastAPI, packaged with Docker, tracked
  experiments with MLflow, and monitored for data drift using Evidently AI.

• Achieved ROC-AUC 0.97 and PR-AUC 0.32 on held-out test set; all components
  tested with pytest and validated with GitHub Actions CI.
```

### 6c. LinkedIn post

A short post announcing the project. Should be written from a personal angle ("I wanted to combine my FP&A background with ML"), not as a technical announcement. Post a GIF of the dashboard in action.

---

## Summary of phases

| Phase | What we add | Primary CV/interview value |
|---|---|---|
| 1 (done) | Data, features, models, risk score, dashboard, API, tests, Docker | ML pipeline, production structure, MLflow |
| 2 | Model evaluation, customer segmentation, SHAP charts | Model rigor, unsupervised learning, explainability |
| 3 | Scenario simulator, retention optimizer | Business impact, optimization, decision-making |
| 4 | RAG copilot with LangChain + OpenAI | LLM, RAG, AI Engineering |
| 5 | Monitoring, CI, cloud deployment | MLOps, production, cloud |
| 6 | README, CV positioning, LinkedIn | Communication, portfolio presentation |

---

## Guiding principles for the build

**Do not add a phase until the previous one is solid.** A project with 3 well-understood phases is stronger than one with 6 half-finished features.

**Every feature should be explainable in a 2-minute interview answer.** Before implementing anything, be able to say: "I built X because it solves Y business problem. It works by doing Z. The trade-off I made was W."

**Prioritize business clarity over technical complexity.** The goal is not to use the most impressive technology. The goal is to build something where every piece serves a clear purpose.

**The project tells one coherent story:** "I used my FP&A background to build an ML system that helps SaaS companies predict revenue risk, understand why customers churn, and make smarter retention decisions."
