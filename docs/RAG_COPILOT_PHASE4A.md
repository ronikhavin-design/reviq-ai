# RAG Revenue Analyst Copilot: Phase 4a

## What Phase 4a does

Phase 4a creates the knowledge base that the RAG Copilot will retrieve from.

It reads existing pipeline outputs (risk scores, ARR targets, trained models) and
generates five clean, business-language Markdown reports into `data/reports/markdown/`.
No LLM, no vector database, and no network calls are made at this stage.

The reports are the "documents" in the retrieval-augmented generation architecture.
Every later phase depends on these documents being accurate, well-structured, and
consistently formatted.

---

## Reports generated

| File | Audience | Content |
|---|---|---|
| `executive_summary.md` | CEO, CFO, Board | ARR trend, forecast vs. target, high-level risk |
| `churn_risk_summary.md` | VP Customer Success | Risk tier distribution, ARR at risk, model drivers |
| `top_customers_at_risk.md` | CS team, AEs | Ranked table of top 20 accounts with health indicators |
| `retention_planning_summary.md` | VP CS, CFO | Three budget scenarios with expected ARR saved and ROI |
| `model_performance_summary.md` | Data team, Finance | Model types, feature importances, ARR forecast MAE |

All reports are written to: `data/reports/markdown/`

---

## Structure of each report

Every report follows the same Markdown structure to enable consistent chunking
in Phase 4b:

```
# Report Title

*Generated: YYYY-MM-DD*
*Source files: file1.csv, file2.pkl*

---

## Key Metrics      (or Overview)
## Interpretation
## Top Risks
## Recommended Actions
```

The `##` heading boundary is the natural chunk boundary for the RAG retrieval layer.
Each section can be embedded independently, so a question like
"what are the top churn risk drivers?" retrieves the Churn Risk Drivers section
rather than the entire document.

---

## Running the report generator

```bash
python -m src.reports.generate_reports
```

Requires the churn model pipeline to have run first:

```bash
python run_pipeline.py
```

The generator reads:
- `data/reports/customer_risk_scores.csv` (churn model output)
- `data/synthetic/targets.csv` (ARR targets)
- `models/churn_model.pkl` and `models/churn_features.pkl` (for feature importances)
- `models/arr_forecast_model.pkl` and `models/arr_features.pkl` (for MAE)

If the model files do not exist, `model_performance_summary.md` is skipped and
the other four reports are still generated.

---

## How these reports become the RAG knowledge base

### Phase 4a (this phase): document generation

The report generator (`src/reports/generate_reports.py`) converts pipeline outputs
into structured Markdown files. These files are human-readable and can be reviewed
by the CS or Finance team independently of any AI system.

### Phase 4b: chunking and embedding

The Markdown files will be split into chunks at `##` section boundaries. Each chunk
will be embedded using a local embedding model (e.g., sentence-transformers) and
stored in a local vector database (e.g., FAISS or Chroma running on disk).

A chunk looks like:

```
Source: churn_risk_summary.md > Top Churn Risk Drivers
Content: "The following features are the strongest predictors of churn..."
```

No API calls to OpenAI or any cloud service are required for this step.

### Phase 4c: retrieval and answer generation

When a user asks a question (e.g., "Which customers should we call this week?"),
the query is embedded and matched against stored chunks using cosine similarity.
The top-k matching chunks are retrieved and injected into a prompt sent to a local
or cloud LLM. The LLM generates a grounded answer citing the retrieved evidence.

The full flow:

```
User question
    -> embed query
    -> retrieve top-k chunks from vector DB
    -> construct prompt: [system] + [retrieved context] + [question]
    -> LLM generates answer citing chunk sources
    -> answer displayed in Streamlit "Copilot" tab
```

---

## Why documents before retrieval

A common mistake in RAG systems is embedding raw data (CSVs, JSON blobs) directly.
This produces retrieval results that are syntactically correct but semantically
disconnected from the question. An executive asking "what is our revenue at risk?"
does not benefit from retrieving a raw row from `customer_risk_scores.csv`.

By generating business-language documents first, we ensure:

1. **Retrieval relevance**: Chunks contain the same vocabulary as user questions.
2. **Answer quality**: The LLM receives pre-interpreted context, not raw numbers.
3. **Auditability**: A human can read the source document and verify the answer.
4. **Regeneration**: When the pipeline reruns, `generate_all_reports()` is called
   again and the vector DB is rebuilt from fresh documents.

---

## Module design

```
src/reports/generate_reports.py

generate_executive_summary(targets, risk)          -> str
generate_churn_risk_summary(risk, fi)              -> str
generate_top_customers_at_risk(risk, top_n)        -> str
generate_retention_planning_summary(risk)          -> str
generate_model_performance_summary(churn, arr)     -> str
write_report(content, filename, output_dir)        -> Path
load_churn_model_info()                            -> dict | None
load_arr_model_info()                              -> dict | None
generate_all_reports(risk, targets, output_dir,
                     churn_model_info, arr_model_info) -> dict[str, Path]
```

All generator functions accept DataFrames directly so they can be tested
without file system access. `generate_all_reports` uses the sentinel value
`"load_from_disk"` for model info parameters to distinguish "not provided"
from "explicitly None" (which means skip model performance report).

---

## Planned enhancements (Phase 4b and beyond)

**Phase 4b: chunking and vector indexing**

- Split each report at `##` heading boundaries
- Embed chunks with `sentence-transformers` (local, no API key required)
- Store in FAISS or Chroma on disk at `data/reports/vector_store/`
- Add a `src/rag/indexer.py` module that reads `data/reports/markdown/` and builds the index

**Phase 4c: retrieval and LLM answer generation**

- Add `src/rag/retriever.py` for similarity search
- Add `src/rag/copilot.py` for prompt construction and LLM call
- Add a "Revenue Copilot" tab to the Streamlit dashboard
- Support local LLMs (Ollama) and cloud LLMs (Anthropic Claude via API)

**Regeneration on pipeline rerun**

- Add a `generate_reports` step to `run_pipeline.py` so fresh reports are
  automatically written after each model retrain
- Rebuild the vector index automatically when reports change
