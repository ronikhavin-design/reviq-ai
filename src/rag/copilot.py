"""
Copilot: Phase 4c of the RevIQ AI RAG Copilot.

Connects the Phase 4b retriever to an LLM to generate grounded, cited
business answers. The full pipeline is:

  1. Retrieve top-k relevant chunks with the Phase 4b Retriever.
  2. Build a prompt (context sections + question).
  3. Call the LLM (OpenAI if OPENAI_API_KEY is set, else use fallback).
  4. Return a CopilotResult with the answer, sources, and metadata.

Fallback mode: if OPENAI_API_KEY is not configured, or the openai package is
not installed, or the API call fails, the module returns a structured answer
built directly from the retrieved chunks. No exception is raised.

No LangChain, no tool calling, and no cloud dependencies beyond the optional
OpenAI API call.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import RetrievalResult, Retriever, build_retriever

OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a SaaS revenue analyst for RevIQ AI. Answer the user's business question
using only the provided context extracted from the RevIQ AI knowledge base.

Guidelines:
- Lead with a direct, specific answer to the question.
- Cite the source document and section for each key claim.
- Include a recommended action if one is clearly supported by the context.
- Use business language appropriate for a VP or C-suite audience.
- If the context does not contain enough information to fully answer the question,
  state what is known and acknowledge the gap.\
"""


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class CopilotResult:
    """Output of the Copilot answer generation step."""

    question: str
    answer: str
    sources_text: str               # Formatted citation list for display
    context_used: str               # Exact prompt content sent to the LLM
    backend: str                    # "openai", "fallback", or "override" (test injection)
    n_chunks_retrieved: int
    retrieval_scores: list[float] = field(default_factory=list)


# ── Prompt and source formatting ──────────────────────────────────────────────

def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    """
    Build the user-message portion of the LLM prompt.

    Combines the retrieved chunks into labeled context sections followed
    by the user question. This string is sent to the LLM as the user
    message, with SYSTEM_PROMPT sent separately as the system message.

    Args:
        question: The user's plain-English question.
        results:  Retrieved chunks from the Phase 4b Retriever.

    Returns:
        A formatted prompt string. Safe to call with an empty results list.
    """
    if not results:
        return (
            f"Question: {question}\n\n"
            "(No context was retrieved from the knowledge base for this question.)"
        )

    blocks = []
    for r in results:
        header = (
            f"[Source: {r.source_file} > {r.section_title}"
            f" | relevance: {r.score:.0%}]"
        )
        blocks.append(f"{header}\n{r.content}")

    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def format_sources(results: list[RetrievalResult]) -> str:
    """
    Format retrieved results as a numbered citation list.

    Returns an empty string when no results are provided so the caller can
    safely include it in any output without a guard.
    """
    if not results:
        return ""
    lines = ["**Sources:**"]
    for r in results:
        lines.append(
            f"  {r.rank}. {r.source_file} > {r.section_title}"
            f" (relevance: {r.score:.0%})"
        )
    return "\n".join(lines)


# ── Fallback answer ───────────────────────────────────────────────────────────

def _build_fallback_answer(question: str, results: list[RetrievalResult]) -> str:
    """
    Build a structured answer from retrieved chunks without calling an LLM.

    Used when OPENAI_API_KEY is not configured, the openai package is missing,
    or the API call fails. The output is clearly labelled so the user knows no
    LLM interpretation has been applied.
    """
    if not results:
        return (
            f"No relevant information was found in the knowledge base "
            f"for the question: '{question}'. "
            "Try regenerating reports with "
            "`python -m src.reports.generate_reports`."
        )

    intro = (
        "The following was retrieved directly from the RevIQ AI knowledge base. "
        "Configure OPENAI_API_KEY to receive an interpreted answer.\n"
    )

    sections = []
    for r in results[:3]:
        # Remove the ## heading line since section_title already captures it.
        body_lines = [
            line for line in r.content.split("\n") if not line.startswith("## ")
        ]
        body = "\n".join(body_lines).strip()
        if len(body) > 500:
            body = body[:500] + "..."
        sections.append(
            f"**{r.section_title}** "
            f"(from {r.source_file}, relevance: {r.score:.0%})\n{body}"
        )

    return intro + "\n\n".join(sections)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str = OPENAI_MODEL) -> Optional[str]:
    """
    Attempt to call the OpenAI chat completions API.

    Returns the response text on success. Returns None if:
    - OPENAI_API_KEY is not set in the environment
    - The openai package is not installed
    - Any network or API error occurs

    Callers should treat None as a signal to use fallback mode.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import openai  # noqa: PLC0415

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def answer_question(
    question: str,
    top_k: int = 5,
    retriever: Optional[Retriever] = None,
    _llm_override: Optional[Callable[[str], Optional[str]]] = None,
) -> CopilotResult:
    """
    Answer a business question using the RAG pipeline.

    Retrieves the top-k most relevant chunks, builds a prompt, and calls
    the LLM (or falls back to a structured retrieval summary if no API key
    is configured).

    Args:
        question:      The user's plain-English question.
        top_k:         Number of chunks to retrieve from the index.
        retriever:     Pre-built Retriever. If None, the default index is loaded.
        _llm_override: Callable that takes the prompt string and returns an answer
                       string (or None to trigger fallback). Intended for tests
                       only, so no external API call is needed during testing.

    Returns:
        CopilotResult containing the answer, sources, context, and metadata.
    """
    question = question.strip()

    if not question:
        return CopilotResult(
            question=question,
            answer="Please enter a question to get an answer.",
            sources_text="",
            context_used="",
            backend="fallback",
            n_chunks_retrieved=0,
        )

    if retriever is None:
        retriever = build_retriever()

    results = retriever.query(question, top_k=top_k)
    prompt = build_prompt(question, results)
    sources_text = format_sources(results)
    scores = [r.score for r in results]

    if _llm_override is not None:
        llm_answer = _llm_override(prompt)
        backend = "override" if llm_answer is not None else "fallback"
    else:
        llm_answer = _call_openai(prompt)
        backend = "openai" if llm_answer is not None else "fallback"

    if llm_answer is None:
        answer = _build_fallback_answer(question, results)
        backend = "fallback"
    else:
        answer = llm_answer

    return CopilotResult(
        question=question,
        answer=answer,
        sources_text=sources_text,
        context_used=prompt,
        backend=backend,
        n_chunks_retrieved=len(results),
        retrieval_scores=scores,
    )


if __name__ == "__main__":
    from loguru import logger

    demo_questions = [
        "Which customers are at highest risk of churning?",
        "What is our current ARR and how does it compare to target?",
        "How should we allocate our retention budget this month?",
    ]

    logger.info("Loading retrieval index...")
    ret = build_retriever()
    logger.info(f"Index ready: {ret.n_chunks} chunks, backend: {ret.backend_name}")

    for q in demo_questions:
        logger.info(f"\nQuestion: {q}")
        result = answer_question(q, top_k=4, retriever=ret)
        logger.info(f"Backend: {result.backend} | Chunks: {result.n_chunks_retrieved}")
        logger.info(f"\n{result.answer}")
        if result.sources_text:
            logger.info(f"\n{result.sources_text}")
        logger.info("-" * 60)
