# RAG Revenue Analyst Copilot: Phase 4c

## What Phase 4c does

Phase 4c adds LLM-powered answer generation on top of the Phase 4b retriever.
Given a plain-English question, the Copilot retrieves the most relevant knowledge
base chunks and sends them to an LLM with a structured prompt. The result is a
grounded, cited business answer ready to display in the dashboard.

If no OpenAI API key is configured (or the call fails for any reason), the Copilot
returns a structured summary of the retrieved chunks without raising an exception.
No LangChain, no cloud dependencies beyond the optional OpenAI call.

---

## Architecture

```
User question
     |
     v
answer_question()
     |
     +-- Retriever.query()           Phase 4b: top-k chunks with scores
     |
     +-- build_prompt()              Labeled context blocks + question
     |
     +-- _call_openai()              OpenAI gpt-4o-mini (if OPENAI_API_KEY set)
     |     |
     |     +-- None returned?
     |           |
     |           v
     +-- _build_fallback_answer()    Structured chunk summary, no LLM
     |
     v
CopilotResult
  .answer           Grounded business answer (LLM) or chunk summary (fallback)
  .sources_text     Numbered citation list
  .context_used     Exact prompt sent to LLM
  .backend          "openai" | "fallback" | "override"
  .n_chunks_retrieved
  .retrieval_scores
```

---

## Module: `src/rag/copilot.py`

### Constants

```python
OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "..."   # SaaS revenue analyst persona with citation guidelines
```

### `CopilotResult` dataclass

| Field | Type | Description |
|---|---|---|
| `question` | `str` | The stripped user question |
| `answer` | `str` | LLM answer or fallback chunk summary |
| `sources_text` | `str` | Formatted numbered citation list |
| `context_used` | `str` | Exact prompt content sent to the LLM |
| `backend` | `str` | `"openai"`, `"fallback"`, or `"override"` |
| `n_chunks_retrieved` | `int` | Number of chunks returned by the retriever |
| `retrieval_scores` | `list[float]` | Cosine similarity scores for each chunk |

### `build_prompt(question, results) -> str`

Combines retrieved chunks into labeled context sections followed by the user
question. Each chunk is prefixed with its source file, section title, and
relevance score. This string is sent as the user message; `SYSTEM_PROMPT` is
sent separately as the system message.

```
Context:

[Source: churn_risk_summary.md > Risk Distribution | relevance: 85%]
## Risk Distribution
...

---

[Source: top_customers_at_risk.md > Customer Risk Table | relevance: 74%]
...

Question: Which customers are at highest risk of churning?
```

Returns a safe prompt even when `results` is empty.

### `format_sources(results) -> str`

Returns a numbered Markdown citation list:

```
**Sources:**
  1. churn_risk_summary.md > Risk Distribution (relevance: 85%)
  2. top_customers_at_risk.md > Customer Risk Table (relevance: 74%)
```

Returns an empty string for an empty results list.

### `_build_fallback_answer(question, results) -> str`

Used when no LLM is available. Shows at most 3 retrieved chunks with heading
lines stripped and content truncated to 500 characters each. Includes a hint
to configure `OPENAI_API_KEY` to receive an interpreted answer.

### `_call_openai(prompt, model) -> Optional[str]`

Returns the LLM response text on success. Returns `None` in any of these cases:

- `OPENAI_API_KEY` environment variable is not set or is empty
- The `openai` package is not installed
- Any network or API error occurs

The `openai` package is imported inside a `try/except` block so the module
remains importable even if `openai` is not installed.

### `answer_question(question, top_k, retriever, _llm_override) -> CopilotResult`

Main entry point. Arguments:

| Parameter | Default | Description |
|---|---|---|
| `question` | required | The user's plain-English question |
| `top_k` | `5` | Number of chunks to retrieve |
| `retriever` | `None` | Pre-built Retriever; loads default index if None |
| `_llm_override` | `None` | Test injection: callable that takes the prompt and returns a string or None |

A blank question returns a safe `CopilotResult` immediately without any
retrieval or LLM call.

---

## Running Phase 4c

**Prerequisite**: Phase 4a reports must exist:

```bash
python -m src.reports.generate_reports
```

**Run the demo** (fallback mode without API key, or LLM mode if key is set):

```bash
python -m src.rag.copilot
```

**With OpenAI** (set the key in your environment before running):

```
OPENAI_API_KEY=sk-...  python -m src.rag.copilot
```

---

## Example usage

```python
from src.rag.copilot import answer_question

result = answer_question(
    "Which customers are at highest risk of churning?",
    top_k=5,
)

print(result.answer)
print(result.sources_text)
print(f"Backend: {result.backend}, Chunks: {result.n_chunks_retrieved}")
```

---

## Backend behavior

| Condition | `backend` field | Answer source |
|---|---|---|
| `OPENAI_API_KEY` set and call succeeds | `"openai"` | LLM-generated |
| `OPENAI_API_KEY` not set | `"fallback"` | Retrieved chunk summary |
| `openai` package not installed | `"fallback"` | Retrieved chunk summary |
| Any API error | `"fallback"` | Retrieved chunk summary |
| `_llm_override` returns a string | `"override"` | Override function result |
| `_llm_override` returns `None` | `"fallback"` | Retrieved chunk summary |

---

## Testing

Tests run with no API key using `_llm_override`:

```bash
python -m pytest tests/test_rag_copilot.py -v
```

Coverage:
- `build_prompt`: prompt structure, source headers, relevance percentages, empty results
- `format_sources`: numbered list, empty string for no results
- `_build_fallback_answer`: heading stripping, truncation, 3-chunk cap, empty results
- `_call_openai`: returns None for missing key, not installed, and API errors
- `answer_question`: override backend, fallback backend, empty questions, top_k, result fields

---

## What Phase 4d should be

Phase 4d adds a "Revenue Copilot" tab to the Streamlit dashboard:

1. `st.text_input` for the question
2. Show the LLM (or fallback) answer as the primary output
3. Collapsible "Evidence" sections showing each retrieved chunk
4. Source file and section attribution for each cited chunk
5. A "Backend" badge showing `openai` vs `fallback`
