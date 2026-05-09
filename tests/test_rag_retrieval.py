"""
Tests for the RAG retrieval layer (Phase 4b).

All tests use synthetic data to avoid file system dependencies.
File persistence tests use pytest's tmp_path fixture.
Index building always forces the TF-IDF backend for determinism.
"""

import numpy as np
import pytest
from pathlib import Path

from src.rag.document_loader import (
    DocumentChunk,
    chunk_markdown,
    load_all_reports,
    load_markdown_file,
)
from src.rag.vector_index import VectorIndex, TFIDFBackend
from src.rag.retriever import Retriever, RetrievalResult, _cosine_similarity


# ── Fixtures and helpers ──────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """\
# Executive Revenue Summary

*Generated: 2026-05-09*
*Source files: targets.csv, customer_risk_scores.csv*

---

## Key Metrics

| Metric | Value |
|---|---|
| Current ARR | $39.52M |
| ARR at Risk | $8.02M |

## Interpretation

Revenue is declining month-over-month. The forecast sits below target.

## Top Risks

- High-risk customers hold 53% of portfolio ARR at risk.
- Forecast is below target by $4.88M.

## Recommended Actions

1. Authorize CS outreach to all High-risk accounts.
2. Review pipeline coverage with Finance.
"""


def make_generic_chunks(n: int = 5, topic: str = "general") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"{topic}__{i:02d}",
            source_file=f"{topic}.md",
            report_type=topic,
            section_title=f"Section {i}",
            content=(
                f"This section covers {topic} area {i}. "
                f"It contains relevant information for testing the retrieval pipeline."
            ),
        )
        for i in range(n)
    ]


def make_topic_chunks() -> list[DocumentChunk]:
    """
    Three chunks with clearly distinct vocabulary so TF-IDF can reliably
    separate them when a topic-specific query is issued.
    """
    return [
        DocumentChunk(
            chunk_id="churn__01",
            source_file="churn_risk_summary.md",
            report_type="churn_risk_summary",
            section_title="Risk Distribution",
            content=(
                "The churn model scored 955 customers. 35 are classified as High-risk. "
                "Churn probability and customer health score are the strongest predictors "
                "of attrition. Customers with declining login frequency and low NPS scores "
                "are most likely to churn. The churn rate across the portfolio is 3.7 percent. "
                "Monitor health score weekly to detect early warning signals of churn."
            ),
        ),
        DocumentChunk(
            chunk_id="arr__01",
            source_file="executive_summary.md",
            report_type="executive_summary",
            section_title="Key Metrics",
            content=(
                "Current Annual Recurring Revenue is $39.52M against a board target of $44.40M. "
                "The forecast is $4.88M below target. Revenue is declining month over month. "
                "New bookings and expansion revenue are needed to close the gap to target. "
                "The ARR trend shows three consecutive months of decline. "
                "Finance and Sales teams should review pipeline coverage urgently."
            ),
        ),
        DocumentChunk(
            chunk_id="retention__01",
            source_file="retention_planning_summary.md",
            report_type="retention_planning_summary",
            section_title="Scenario Comparison",
            content=(
                "The retention budget optimizer selects customers using a greedy algorithm "
                "sorted by return on investment. With a $50,000 discount budget and 150 CS hours, "
                "the standard scenario selects 50 customers for outreach. "
                "Expected ARR saved is $4.2M. The optimizer prioritizes accounts where "
                "expected savings per dollar spent is highest, favoring High-risk accounts "
                "with large contract values."
            ),
        ),
    ]


def build_tfidf_index(chunks: list[DocumentChunk]) -> VectorIndex:
    """Build a TF-IDF VectorIndex (backend forced for test determinism)."""
    return VectorIndex.build(chunks, backend_name="tfidf")


# ── document_loader: chunk_markdown ──────────────────────────────────────────

def test_chunk_markdown_returns_nonempty_list():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_markdown_creates_one_chunk_per_section():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    section_titles = [c.section_title for c in chunks]
    for expected in ["Key Metrics", "Interpretation", "Top Risks", "Recommended Actions"]:
        assert expected in section_titles


def test_chunk_markdown_header_chunk_captured():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    header = next((c for c in chunks if c.section_title == "document_header"), None)
    assert header is not None
    assert "Executive Revenue Summary" in header.content


def test_chunk_markdown_all_metadata_fields_present():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.source_file == "executive_summary.md"
        assert chunk.report_type == "executive_summary"
        assert chunk.section_title
        assert chunk.content


def test_chunk_markdown_chunk_ids_are_unique():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_markdown_report_type_derived_from_stem():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "churn_risk_summary.md")
    assert all(c.report_type == "churn_risk_summary" for c in chunks)


def test_chunk_markdown_no_headings_produces_single_header_chunk():
    content = "# Simple Report\n\nOnly a header section, no ## headings."
    chunks = chunk_markdown(content, "simple.md")
    assert len(chunks) == 1
    assert chunks[0].section_title == "document_header"


def test_chunk_markdown_section_content_includes_heading_line():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, "executive_summary.md")
    key_metrics = next(c for c in chunks if c.section_title == "Key Metrics")
    assert "## Key Metrics" in key_metrics.content


# ── document_loader: file I/O ─────────────────────────────────────────────────

def test_load_markdown_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_markdown_file(tmp_path / "does_not_exist.md")


def test_load_markdown_file_returns_chunks(tmp_path):
    md_file = tmp_path / "executive_summary.md"
    md_file.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    chunks = load_markdown_file(md_file)
    assert len(chunks) >= 4
    assert all(c.source_file == "executive_summary.md" for c in chunks)


def test_load_all_reports_returns_empty_for_missing_dir(tmp_path):
    result = load_all_reports(tmp_path / "nonexistent")
    assert result == []


def test_load_all_reports_loads_all_md_files(tmp_path):
    for name in ["report_a.md", "report_b.md"]:
        (tmp_path / name).write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    chunks = load_all_reports(tmp_path)
    source_files = {c.source_file for c in chunks}
    assert "report_a.md" in source_files
    assert "report_b.md" in source_files


def test_load_all_reports_ignores_non_md_files(tmp_path):
    (tmp_path / "report.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a report", encoding="utf-8")
    chunks = load_all_reports(tmp_path)
    assert all(c.source_file.endswith(".md") for c in chunks)


def test_load_all_reports_empty_dir_returns_empty(tmp_path):
    result = load_all_reports(tmp_path)
    assert result == []


# ── VectorIndex: build ────────────────────────────────────────────────────────

def test_vector_index_builds_from_chunks():
    index = build_tfidf_index(make_generic_chunks(5))
    assert index is not None
    assert len(index) == 5


def test_vector_index_vectors_shape_matches_chunk_count():
    chunks = make_generic_chunks(7)
    index = build_tfidf_index(chunks)
    assert index.vectors.ndim == 2
    assert index.vectors.shape[0] == 7


def test_vector_index_backend_name_is_tfidf():
    index = build_tfidf_index(make_generic_chunks(3))
    assert index.backend_name == "tfidf"


def test_vector_index_build_raises_for_empty_chunk_list():
    with pytest.raises(ValueError):
        VectorIndex.build([], backend_name="tfidf")


def test_vector_index_embed_query_returns_1d_array():
    index = build_tfidf_index(make_generic_chunks(5))
    vec = index.embed_query("What is the churn rate?")
    assert vec.ndim == 1
    assert vec.shape[0] == index.vectors.shape[1]


# ── VectorIndex: persistence ──────────────────────────────────────────────────

def test_vector_index_save_creates_index_file(tmp_path):
    index = build_tfidf_index(make_generic_chunks(4))
    index.save(tmp_path)
    assert (tmp_path / "index.pkl").exists()


def test_vector_index_load_restores_chunk_count(tmp_path):
    index = build_tfidf_index(make_generic_chunks(4))
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert len(loaded) == 4


def test_vector_index_load_restores_backend_name(tmp_path):
    index = build_tfidf_index(make_generic_chunks(4))
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.backend_name == "tfidf"


def test_vector_index_load_restores_chunk_ids(tmp_path):
    chunks = make_generic_chunks(3, topic="test")
    index = build_tfidf_index(chunks)
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]


def test_vector_index_load_raises_for_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path)


def test_vector_index_save_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "dir"
    index = build_tfidf_index(make_generic_chunks(2))
    index.save(nested)
    assert (nested / "index.pkl").exists()


# ── _cosine_similarity: unit tests ───────────────────────────────────────────

def test_cosine_similarity_identical_vectors_score_is_one():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores = _cosine_similarity(v, matrix)
    assert abs(scores[0] - 1.0) < 1e-5


def test_cosine_similarity_orthogonal_vectors_score_is_zero():
    v = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 1.0]], dtype=np.float32)
    scores = _cosine_similarity(v, matrix)
    assert abs(scores[0]) < 1e-5


def test_cosine_similarity_zero_query_returns_all_zeros():
    v = np.array([0.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = _cosine_similarity(v, matrix)
    assert all(s == 0.0 for s in scores)


def test_cosine_similarity_scores_bounded_between_0_and_1():
    rng = np.random.default_rng(42)
    v = rng.random(16).astype(np.float32)
    matrix = rng.random((20, 16)).astype(np.float32)
    scores = _cosine_similarity(v, matrix)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


# ── Retriever: query behavior ─────────────────────────────────────────────────

def test_retriever_returns_correct_number_of_results():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("churn risk customers", top_k=2)
    assert len(results) == 2


def test_retriever_results_sorted_by_score_descending():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("ARR revenue forecast", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_ranks_start_at_one_and_increment():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("risk", top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_retriever_result_has_all_required_fields():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("churn probability", top_k=1)
    r = results[0]
    assert r.rank == 1
    assert 0.0 <= r.score <= 1.0
    assert r.chunk_id
    assert r.source_file
    assert r.report_type
    assert r.section_title
    assert r.content


def test_retriever_top_k_capped_at_total_chunk_count():
    index = build_tfidf_index(make_generic_chunks(3))
    retriever = Retriever(index)
    results = retriever.query("something", top_k=100)
    assert len(results) == 3


def test_retriever_returns_empty_list_for_blank_question():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    assert retriever.query("") == []
    assert retriever.query("   ") == []


def test_retriever_finds_churn_chunk_for_churn_question():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query(
        "customer churn probability high risk attrition health score", top_k=1
    )
    assert results[0].report_type == "churn_risk_summary"


def test_retriever_finds_arr_chunk_for_arr_question():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query(
        "Annual Recurring Revenue ARR board target forecast below decline", top_k=1
    )
    assert results[0].report_type == "executive_summary"


def test_retriever_scores_are_floats_between_0_and_1():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("budget optimizer retention", top_k=3)
    for r in results:
        assert isinstance(r.score, float)
        assert 0.0 <= r.score <= 1.0


# ── Retriever: format_context ─────────────────────────────────────────────────

def test_format_context_returns_nonempty_string():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("churn risk", top_k=2)
    context = retriever.format_context(results)
    assert isinstance(context, str)
    assert len(context) > 50


def test_format_context_contains_source_attribution():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    results = retriever.query("churn risk", top_k=1)
    context = retriever.format_context(results)
    assert "[Source:" in context
    assert "churn_risk_summary.md" in context


def test_format_context_truncates_long_chunks():
    long_chunk = DocumentChunk(
        chunk_id="long__00",
        source_file="long.md",
        report_type="long",
        section_title="Long Section",
        content="word " * 500,
    )
    index = build_tfidf_index([long_chunk])
    retriever = Retriever(index)
    results = retriever.query("word", top_k=1)
    context = retriever.format_context(results, max_chars_per_chunk=100)
    assert "..." in context


def test_format_context_empty_results_returns_empty_string():
    index = build_tfidf_index(make_topic_chunks())
    retriever = Retriever(index)
    context = retriever.format_context([])
    assert context == ""


# ── build_retriever: integration ──────────────────────────────────────────────

def test_build_retriever_from_markdown_dir(tmp_path):
    md_dir = tmp_path / "markdown"
    md_dir.mkdir()
    (md_dir / "churn_risk_summary.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    idx_dir = tmp_path / "vector_index"
    from src.rag.retriever import build_retriever
    retriever = build_retriever(
        markdown_dir=md_dir, index_dir=idx_dir, backend="tfidf"
    )
    assert retriever.n_chunks > 0
    assert (idx_dir / "index.pkl").exists()


def test_build_retriever_loads_cached_index(tmp_path):
    md_dir = tmp_path / "markdown"
    md_dir.mkdir()
    (md_dir / "churn_risk_summary.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    idx_dir = tmp_path / "vector_index"
    from src.rag.retriever import build_retriever
    r1 = build_retriever(markdown_dir=md_dir, index_dir=idx_dir, backend="tfidf")
    r2 = build_retriever(markdown_dir=md_dir, index_dir=idx_dir)
    assert r1.n_chunks == r2.n_chunks


def test_build_retriever_raises_for_empty_markdown_dir(tmp_path):
    empty_md_dir = tmp_path / "empty"
    empty_md_dir.mkdir()
    idx_dir = tmp_path / "vector_index"
    from src.rag.retriever import build_retriever
    with pytest.raises(FileNotFoundError):
        build_retriever(markdown_dir=empty_md_dir, index_dir=idx_dir, backend="tfidf")
