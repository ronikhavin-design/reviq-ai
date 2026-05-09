"""
Retriever: Phase 4b of the RevIQ AI RAG Copilot.

Accepts a natural-language question, embeds it with the same backend used
at index build time, ranks all indexed chunks by cosine similarity, and
returns the top-k most relevant DocumentChunks with scores and metadata.

No LLM, no API calls, and no answer generation at this stage.
The retriever output (retrieved chunks + format_context) is designed to be
passed directly as context to an LLM in Phase 4c.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import REPORTS_DIR
from src.rag.document_loader import load_all_reports
from src.rag.vector_index import INDEX_DIR, MARKDOWN_DIR, VectorIndex

DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """A single chunk returned by the retriever with attribution and score."""

    rank: int           # 1-indexed rank (1 = most relevant)
    score: float        # Cosine similarity in [0.0, 1.0]
    chunk_id: str       # Unique identifier of the chunk
    source_file: str    # Markdown file the chunk came from
    report_type: str    # Logical report type (e.g. "churn_risk_summary")
    section_title: str  # Section heading (e.g. "Top Risks")
    content: str        # Full text of the chunk


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row of matrix.

    Args:
        query_vec: 1D array of shape (d,)
        matrix:    2D array of shape (n, d)

    Returns:
        1D array of shape (n,) with similarity scores clipped to [0.0, 1.0].
    """
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm < 1e-10:
        return np.zeros(len(matrix), dtype=np.float32)
    q = query_vec / q_norm
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-10, 1.0, row_norms)
    normed = matrix / row_norms
    scores = normed @ q
    return np.clip(scores, 0.0, 1.0)


class Retriever:
    """
    Retrieves the most relevant document chunks for a natural-language question.

    Embeds the query using the same backend used at index build time, ranks
    all indexed chunks by cosine similarity, and returns the top-k results
    with scores and full metadata for attribution.
    """

    def __init__(self, index: VectorIndex) -> None:
        self._index = index

    @property
    def n_chunks(self) -> int:
        return len(self._index)

    @property
    def backend_name(self) -> str:
        return self._index.backend_name

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievalResult]:
        """
        Return the top-k most relevant chunks for a plain-English question.

        Args:
            question: The user's question.
            top_k:    Maximum number of results to return.

        Returns:
            List of RetrievalResult sorted by score descending (rank 1 = best).
            Returns an empty list for a blank question.
        """
        if not question.strip():
            return []

        query_vec = self._index.embed_query(question)
        scores = _cosine_similarity(query_vec, self._index.vectors)

        actual_k = min(top_k, len(self._index))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._index.chunks[idx]
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(scores[idx]),
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    report_type=chunk.report_type,
                    section_title=chunk.section_title,
                    content=chunk.content,
                )
            )
        return results

    def format_context(
        self,
        results: list[RetrievalResult],
        max_chars_per_chunk: int = 800,
    ) -> str:
        """
        Format retrieved results into a context block ready for LLM injection.

        Each chunk is prefixed with its source file and section heading so that
        an LLM can cite the evidence. This is the interface to Phase 4c.

        Args:
            results:             List of RetrievalResult from query().
            max_chars_per_chunk: Truncate each chunk body to this length.

        Returns:
            A single string with all retrieved chunks separated by dividers.
        """
        blocks: list[str] = []
        for r in results:
            header = f"[Source: {r.source_file} > {r.section_title}]"
            body = r.content
            if len(body) > max_chars_per_chunk:
                body = body[:max_chars_per_chunk] + "..."
            blocks.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(blocks)


def build_retriever(
    markdown_dir: Path = MARKDOWN_DIR,
    index_dir: Path = INDEX_DIR,
    backend: str = "auto",
    force_rebuild: bool = False,
) -> Retriever:
    """
    Load or build the vector index and return a Retriever.

    If a saved index exists at index_dir and force_rebuild is False, the
    existing index is loaded from disk. Otherwise, Markdown reports are
    loaded from markdown_dir, embedded, and the index is saved to index_dir.

    Args:
        markdown_dir:  Directory containing .md report files.
        index_dir:     Directory where index.pkl is persisted.
        backend:       "auto", "tfidf", or "sentence-transformers".
        force_rebuild: Rebuild the index even if a cached version exists.

    Returns:
        A Retriever ready to answer questions.

    Raises:
        FileNotFoundError: If markdown_dir has no .md files and no index exists.
    """
    index_file = index_dir / "index.pkl"
    if not force_rebuild and index_file.exists():
        index = VectorIndex.load(index_dir)
        return Retriever(index)

    chunks = load_all_reports(markdown_dir)
    if not chunks:
        raise FileNotFoundError(
            f"No Markdown reports found in {markdown_dir}. "
            "Run `python -m src.reports.generate_reports` first."
        )
    index = VectorIndex.build(chunks, backend_name=backend)
    index.save(index_dir)
    return Retriever(index)


if __name__ == "__main__":
    from loguru import logger

    logger.info("Building retrieval index...")
    retriever = build_retriever(force_rebuild=True)
    logger.info(
        f"Index built: {retriever.n_chunks} chunks, backend: {retriever.backend_name}"
    )

    demo_questions = [
        "Which customers are at highest risk of churning?",
        "What is our current ARR and forecast versus target?",
        "How should we allocate our retention budget?",
        "What are the top churn risk drivers from the model?",
        "What is the ARR forecast model accuracy?",
    ]

    for q in demo_questions:
        logger.info(f"\nQuestion: {q}")
        results = retriever.query(q, top_k=3)
        for r in results:
            logger.info(
                f"  [{r.rank}] score={r.score:.3f} | "
                f"{r.source_file} > {r.section_title}"
            )
