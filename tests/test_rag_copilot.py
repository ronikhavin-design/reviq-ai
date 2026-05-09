"""
Tests for Phase 4c: src/rag/copilot.py

All tests run without an OpenAI API key by using the _llm_override parameter
or by patching the environment. No external API calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.copilot import (
    CopilotResult,
    _build_fallback_answer,
    _call_openai,
    answer_question,
    build_prompt,
    format_sources,
)
from src.rag.retriever import RetrievalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_result(
    rank: int = 1,
    score: float = 0.85,
    source_file: str = "churn_risk_summary.md",
    section_title: str = "Risk Distribution",
    content: str = "## Risk Distribution\n\n35 high-risk customers hold $8.02M ARR.",
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk_id=f"churn_risk_summary__{rank:02d}",
        source_file=source_file,
        report_type="churn_risk_summary",
        section_title=section_title,
        content=content,
    )


def make_results(n: int = 3) -> list[RetrievalResult]:
    titles = ["Risk Distribution", "Recommended Actions", "Model Inputs", "Key Metrics", "Action Plan"]
    files = [
        "churn_risk_summary.md",
        "churn_risk_summary.md",
        "model_performance_summary.md",
        "executive_summary.md",
        "retention_planning_summary.md",
    ]
    return [
        make_result(
            rank=i + 1,
            score=round(max(0.1, 0.9 - i * 0.1), 2),
            source_file=files[i % len(files)],
            section_title=titles[i % len(titles)],
            content=f"## {titles[i % len(titles)]}\n\nContent for section {i + 1}.",
        )
        for i in range(n)
    ]


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_question(self):
        results = make_results(2)
        prompt = build_prompt("Who is at risk?", results)
        assert "Who is at risk?" in prompt

    def test_contains_source_header(self):
        results = [make_result()]
        prompt = build_prompt("test", results)
        assert "churn_risk_summary.md" in prompt
        assert "Risk Distribution" in prompt

    def test_contains_relevance_percentage(self):
        results = [make_result(score=0.85)]
        prompt = build_prompt("test", results)
        assert "85%" in prompt

    def test_contains_chunk_content(self):
        results = [make_result(content="## Section\n\nSpecific content here.")]
        prompt = build_prompt("test", results)
        assert "Specific content here." in prompt

    def test_multiple_chunks_separated(self):
        results = make_results(3)
        prompt = build_prompt("test", results)
        assert "---" in prompt

    def test_empty_results_returns_safe_prompt(self):
        prompt = build_prompt("What is ARR?", [])
        assert "What is ARR?" in prompt
        assert "No context was retrieved" in prompt

    def test_prompt_starts_with_context(self):
        results = make_results(1)
        prompt = build_prompt("test", results)
        assert prompt.startswith("Context:")

    def test_question_appears_after_context(self):
        results = make_results(1)
        prompt = build_prompt("My question?", results)
        context_pos = prompt.index("Context:")
        question_pos = prompt.index("My question?")
        assert question_pos > context_pos

    def test_no_em_dash(self):
        results = make_results(3)
        prompt = build_prompt("test question", results)
        assert "—" not in prompt


# ── format_sources ────────────────────────────────────────────────────────────

class TestFormatSources:
    def test_returns_empty_string_for_no_results(self):
        assert format_sources([]) == ""

    def test_starts_with_sources_header(self):
        results = make_results(1)
        text = format_sources(results)
        assert text.startswith("**Sources:**")

    def test_contains_source_file(self):
        results = [make_result(source_file="churn_risk_summary.md")]
        text = format_sources(results)
        assert "churn_risk_summary.md" in text

    def test_contains_section_title(self):
        results = [make_result(section_title="Risk Distribution")]
        text = format_sources(results)
        assert "Risk Distribution" in text

    def test_contains_relevance_percentage(self):
        results = [make_result(score=0.72)]
        text = format_sources(results)
        assert "72%" in text

    def test_ranks_are_numbered(self):
        results = make_results(3)
        text = format_sources(results)
        assert "1." in text
        assert "2." in text
        assert "3." in text

    def test_multiple_sources_all_present(self):
        results = make_results(3)
        text = format_sources(results)
        for r in results:
            assert r.section_title in text

    def test_no_em_dash(self):
        results = make_results(3)
        text = format_sources(results)
        assert "—" not in text


# ── _build_fallback_answer ────────────────────────────────────────────────────

class TestBuildFallbackAnswer:
    def test_empty_results_returns_not_found_message(self):
        answer = _build_fallback_answer("What is ARR?", [])
        assert "No relevant information" in answer

    def test_empty_results_includes_regenerate_hint(self):
        answer = _build_fallback_answer("Any question?", [])
        assert "generate_reports" in answer

    def test_contains_retrieved_content(self):
        results = [make_result(content="## Risk Distribution\n\nKey risk data here.")]
        answer = _build_fallback_answer("test", results)
        assert "Key risk data here." in answer

    def test_strips_heading_lines(self):
        results = [make_result(content="## Risk Distribution\n\nBody text.")]
        answer = _build_fallback_answer("test", results)
        assert "## Risk Distribution" not in answer

    def test_includes_source_file(self):
        results = [make_result(source_file="churn_risk_summary.md")]
        answer = _build_fallback_answer("test", results)
        assert "churn_risk_summary.md" in answer

    def test_includes_relevance_score(self):
        results = [make_result(score=0.75)]
        answer = _build_fallback_answer("test", results)
        assert "75%" in answer

    def test_truncates_long_content(self):
        long_content = "## Section\n\n" + "x" * 1000
        results = [make_result(content=long_content)]
        answer = _build_fallback_answer("test", results)
        assert "..." in answer

    def test_uses_at_most_three_chunks(self):
        results = make_results(5)
        answer = _build_fallback_answer("test", results)
        assert "Key Metrics" not in answer  # 4th section title, should be excluded

    def test_includes_configure_openai_hint(self):
        results = make_results(2)
        answer = _build_fallback_answer("test", results)
        assert "OPENAI_API_KEY" in answer

    def test_no_em_dash(self):
        results = make_results(3)
        answer = _build_fallback_answer("test", results)
        assert "—" not in answer


# ── _call_openai ──────────────────────────────────────────────────────────────

class TestCallOpenAI:
    def test_returns_none_when_no_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            result = _call_openai("test prompt")
        assert result is None

    def test_returns_none_when_openai_not_installed(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}, clear=False):
            with patch.dict("sys.modules", {"openai": None}):
                result = _call_openai("test prompt")
        assert result is None

    def test_returns_none_on_api_error(self):
        import types

        fake_openai = types.ModuleType("openai")

        class FakeClient:
            def __init__(self, api_key):
                pass

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("Simulated API failure")

        fake_openai.OpenAI = FakeClient

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}, clear=False):
            with patch.dict("sys.modules", {"openai": fake_openai}):
                result = _call_openai("test prompt")
        assert result is None


# ── answer_question ───────────────────────────────────────────────────────────

class TestAnswerQuestion:
    """
    All tests use _llm_override to avoid any real API calls.
    A minimal fake Retriever is injected so no vector index is needed.
    """

    class _FakeRetriever:
        def __init__(self, results):
            self._results = results

        def query(self, question, top_k=5):
            return self._results[:top_k]

    def _retriever(self, n=2):
        return self._FakeRetriever(make_results(n))

    # ── Empty question ─────────────────────────────────────────────────────────

    def test_empty_question_returns_safe_result(self):
        result = answer_question("", retriever=self._retriever())
        assert isinstance(result, CopilotResult)
        assert result.answer != ""
        assert result.n_chunks_retrieved == 0

    def test_whitespace_only_question_returns_safe_result(self):
        result = answer_question("   ", retriever=self._retriever())
        assert result.n_chunks_retrieved == 0

    def test_empty_question_backend_is_fallback(self):
        result = answer_question("", retriever=self._retriever())
        assert result.backend == "fallback"

    # ── Override returning a string ────────────────────────────────────────────

    def test_override_answer_is_used(self):
        override = lambda prompt: "Injected answer from test."
        result = answer_question(
            "What is churn risk?",
            retriever=self._retriever(),
            _llm_override=override,
        )
        assert result.answer == "Injected answer from test."

    def test_override_backend_is_override(self):
        override = lambda prompt: "Any answer."
        result = answer_question(
            "What is churn risk?",
            retriever=self._retriever(),
            _llm_override=override,
        )
        assert result.backend == "override"

    def test_override_receives_prompt_with_question(self):
        captured = []
        def override(prompt):
            captured.append(prompt)
            return "ok"

        answer_question(
            "Specific question text?",
            retriever=self._retriever(),
            _llm_override=override,
        )
        assert "Specific question text?" in captured[0]

    # ── Override returning None (triggers fallback) ────────────────────────────

    def test_override_returning_none_triggers_fallback(self):
        result = answer_question(
            "What is ARR?",
            retriever=self._retriever(),
            _llm_override=lambda p: None,
        )
        assert result.backend == "fallback"

    def test_override_returning_none_fallback_answer_non_empty(self):
        result = answer_question(
            "What is ARR?",
            retriever=self._retriever(),
            _llm_override=lambda p: None,
        )
        assert len(result.answer) > 0

    # ── Sources and metadata ───────────────────────────────────────────────────

    def test_sources_text_populated(self):
        result = answer_question(
            "Which customers are at risk?",
            retriever=self._retriever(2),
            _llm_override=lambda p: "answer",
        )
        assert "Sources:" in result.sources_text

    def test_n_chunks_retrieved_matches_retriever(self):
        result = answer_question(
            "test question",
            retriever=self._retriever(3),
            _llm_override=lambda p: "answer",
        )
        assert result.n_chunks_retrieved == 3

    def test_retrieval_scores_populated(self):
        result = answer_question(
            "test question",
            retriever=self._retriever(2),
            _llm_override=lambda p: "answer",
        )
        assert len(result.retrieval_scores) == 2
        assert all(isinstance(s, float) for s in result.retrieval_scores)

    def test_context_used_contains_question(self):
        result = answer_question(
            "Context test question?",
            retriever=self._retriever(1),
            _llm_override=lambda p: "answer",
        )
        assert "Context test question?" in result.context_used

    def test_question_is_stripped(self):
        result = answer_question(
            "  trimmed question  ",
            retriever=self._retriever(1),
            _llm_override=lambda p: "answer",
        )
        assert result.question == "trimmed question"

    # ── Empty retrieval (zero results) ─────────────────────────────────────────

    def test_zero_retrieval_results_falls_back(self):
        result = answer_question(
            "Unanswerable?",
            retriever=self._FakeRetriever([]),
            _llm_override=lambda p: None,
        )
        assert result.backend == "fallback"
        assert result.n_chunks_retrieved == 0

    def test_zero_retrieval_sources_text_empty(self):
        result = answer_question(
            "Unanswerable?",
            retriever=self._FakeRetriever([]),
            _llm_override=lambda p: "override answer",
        )
        assert result.sources_text == ""

    # ── top_k parameter ────────────────────────────────────────────────────────

    def test_top_k_limits_retrieved_chunks(self):
        result = answer_question(
            "test",
            top_k=1,
            retriever=self._retriever(5),
            _llm_override=lambda p: "answer",
        )
        assert result.n_chunks_retrieved == 1

    # ── CopilotResult fields ──────────────────────────────────────────────────

    def test_result_is_copilot_result_instance(self):
        result = answer_question(
            "test",
            retriever=self._retriever(),
            _llm_override=lambda p: "answer",
        )
        assert isinstance(result, CopilotResult)

    def test_result_fields_all_present(self):
        result = answer_question(
            "test",
            retriever=self._retriever(),
            _llm_override=lambda p: "answer",
        )
        assert hasattr(result, "question")
        assert hasattr(result, "answer")
        assert hasattr(result, "sources_text")
        assert hasattr(result, "context_used")
        assert hasattr(result, "backend")
        assert hasattr(result, "n_chunks_retrieved")
        assert hasattr(result, "retrieval_scores")

    # ── No em dashes in output ────────────────────────────────────────────────

    def test_no_em_dash_in_fallback_answer(self):
        result = answer_question(
            "test",
            retriever=self._retriever(),
            _llm_override=lambda p: None,
        )
        assert "—" not in result.answer
        assert "—" not in result.sources_text
