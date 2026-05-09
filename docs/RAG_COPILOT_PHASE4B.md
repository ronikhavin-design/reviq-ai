# RAG Revenue Analyst Copilot: Phase 4b

## What Phase 4b does

Phase 4b builds the local retrieval layer over the Markdown reports generated
in Phase 4a. It splits each report into chunks, embeds them as vectors, and
stores a searchable index on disk. Given a plain-English question, the retriever
returns the most relevant chunks with scores and source attribution.

No LLM, no API calls, and no answer generation happen at this stage.
The output of Phase 4b is a `Retriever` object that returns structured
`RetrievalResult` objects ready to be injected into an LLM prompt in Phase 4c.

---

## Architecture

```
Phase 4a output          Phase 4b modules              Phase 4b output
data/reports/markdown/
  executive_summary.md
  churn_risk_summary.md     document_loader.py
  top_customers_at_risk.md  -> chunk_markdown()
  retention_planning_        -> load_all_reports()      list[DocumentChunk]
    summary.md                                               |
  model_performance_                                         v
    summary.md            vector_index.py               VectorIndex
                          -> VectorIndex.build()        (chunks + vectors + backend)
                          -> VectorIndex.save()              |
                          -> VectorIndex.load()              v
                                                        retriever.py
                          retriever.py                  -> Retriever.query()
                          -> build_retriever()          -> format_context()
```

---

## Modules

### `src/rag/document_loader.py`

Reads Markdown files and splits them into `DocumentChunk` objects.

**Chunking strategy**: Split at `##` heading boundaries. The text before the
first `##` (title, date, source files) becomes a `document_header` chunk.
Each `##` section becomes its own chunk with the heading line included in the
content so that the section title is searchable.

```python
@dataclass
class DocumentChunk:
    chunk_id: str       # "executive_summary__01"
    source_file: str    # "executive_summary.md"
    report_type: str    # "executive_summary"
    section_title: str  # "Key Metrics"
    content: str        # Full text including the ## heading line
```

Key functions:
- `chunk_markdown(content, source_file) -> list[DocumentChunk]`
- `load_markdown_file(path) -> list[DocumentChunk]` (raises FileNotFoundError if missing)
- `load_all_reports(markdown_dir) -> list[DocumentChunk]` (returns [] if dir missing)


### `src/rag/vector_index.py`

Embeds chunks and stores a dense vector index.

**Embedding backends**:

| Backend | When used | Notes |
|---|---|---|
| `TFIDFBackend` | Always available | scikit-learn TF-IDF, no model download |
| `SentenceTransformerBackend` | If installed | Semantic embeddings, better recall |
| `"auto"` selection | Default | Uses sentence-transformers if installed, else TF-IDF |

**Storage format**: A single `joblib`-compressed `index.pkl` under
`data/reports/vector_index/`, containing the chunk list, the dense embedding
matrix, and the fitted backend object so queries can be embedded with the
same model used at build time.

Key methods:
- `VectorIndex.build(chunks, backend_name="auto") -> VectorIndex`
- `VectorIndex.save(index_dir)` and `VectorIndex.load(index_dir)`
- `VectorIndex.embed_query(text) -> np.ndarray`


### `src/rag/retriever.py`

Accepts a question and returns ranked results.

**Similarity metric**: Cosine similarity computed with numpy. Both stored
vectors and query vectors are L2-normalized before the dot product, so scores
are bounded in [0.0, 1.0]. No FAISS or external vector database is required.

```python
@dataclass
class RetrievalResult:
    rank: int           # 1 = most relevant
    score: float        # Cosine similarity in [0.0, 1.0]
    chunk_id: str
    source_file: str
    report_type: str
    section_title: str
    content: str
```

Key functions:
- `Retriever.query(question, top_k=5) -> list[RetrievalResult]`
- `Retriever.format_context(results, max_chars_per_chunk=800) -> str`
- `build_retriever(markdown_dir, index_dir, backend, force_rebuild) -> Retriever`

---

## Running Phase 4b

**Step 1**: Generate the Markdown reports (Phase 4a):

```bash
python -m src.reports.generate_reports
```

**Step 2**: Build the retrieval index and test queries:

```bash
python -m src.rag.retriever
```

This builds the index at `data/reports/vector_index/index.pkl` and runs five
demo questions, printing the top 3 retrieved chunks per question.

---

## Example retrieval

```python
from src.rag.retriever import build_retriever

retriever = build_retriever()  # loads cached index or builds from markdown dir

results = retriever.query("Which customers are at highest risk of churning?", top_k=3)

for r in results:
    print(f"[{r.rank}] score={r.score:.3f} | {r.source_file} > {r.section_title}")
    print(r.content[:200])
    print()

# Prepare context for LLM injection (Phase 4c)
context = retriever.format_context(results)
```

Example output:
```
[1] score=0.821 | churn_risk_summary.md > Risk Distribution
## Risk Distribution

| Risk Level | Customers | ARR at Risk | ...

[2] score=0.743 | top_customers_at_risk.md > Customer Risk Table
## Customer Risk Table
...

[3] score=0.612 | churn_risk_summary.md > Recommended Actions
...
```

---

## How Phase 4b connects to Phase 4a

Phase 4a produces business-language Markdown documents. Phase 4b does not
re-read the raw pipeline data (CSVs, model files). It reads only the generated
Markdown files. This separation has two benefits:

1. **Regeneration**: When the pipeline reruns and new reports are generated,
   calling `build_retriever(force_rebuild=True)` rebuilds the index from the
   fresh documents automatically.

2. **Answer quality**: The chunks contain pre-interpreted business context
   ("35 High-risk customers hold $8.02M in ARR at risk") rather than raw rows.
   An LLM receiving these chunks can answer "what is our ARR at risk?" with
   a complete, cited, human-readable answer.

---

## What Phase 4c should be

Phase 4c adds LLM-powered answer generation on top of the retriever built here.

**Inputs**: A user question and the output of `Retriever.format_context(results)`.

**Outputs**: A grounded, cited answer in plain English.

**Implementation plan**:

1. Add `src/rag/copilot.py`:
   - `CopilotInput(question, top_k, max_context_chars)` dataclass
   - `CopilotResult(answer, sources, context_used)` dataclass
   - `answer_question(question, retriever, llm_client) -> CopilotResult`

2. Prompt structure:
   ```
   System: You are a SaaS revenue analyst. Answer using only the provided context.
           Cite your sources by mentioning the report section name.

   Context:
   [Source: churn_risk_summary.md > Risk Distribution]
   ...retrieved text...

   Question: Which customers should we prioritize for CS outreach this week?
   ```

3. LLM options (in order of preference for a local-first project):
   - **Anthropic Claude API** (primary): fast, no hardware requirement
   - **Ollama** (local alternative): runs a quantized model on-device, no API key
   - Both can be wrapped behind a common interface so the choice is a config flag.

4. Add a "Revenue Copilot" tab to the Streamlit dashboard:
   - `st.text_input` for the question
   - Show retrieved chunks as expandable "Evidence" sections
   - Show the LLM answer as the main output
   - Show source file and section for each cited chunk

5. Tests: `tests/test_rag_copilot.py`
   - Mock the LLM client so tests run without an API key
   - Test that the prompt is well-formed
   - Test that sources are cited correctly
   - Test that blank questions return a safe response

---

## Upgrading to semantic embeddings

Once `sentence-transformers` is installed, the index will automatically use it:

```bash
pip install sentence-transformers
python -m src.rag.retriever  # force_rebuild=True rebuilds with new backend
```

The default model (`all-MiniLM-L6-v2`) is 80MB and runs on CPU. It produces
384-dimensional dense embeddings that capture semantic similarity, so a question
like "which accounts are likely to cancel?" will retrieve churn-related chunks
even if the word "churn" does not appear in the question.

No code changes are needed. The `"auto"` backend selection in `_select_backend()`
handles the upgrade automatically.
