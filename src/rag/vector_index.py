"""
Vector Index: Phase 4b of the RevIQ AI RAG Copilot.

Builds and persists a local dense vector index over DocumentChunk objects.

Backend selection:
  "auto"                  Use sentence-transformers if installed, else TF-IDF.
  "tfidf"                 scikit-learn TF-IDF (always available, no downloads).
  "sentence-transformers" Semantic embeddings via the sentence-transformers library.

The index is stored as a single joblib-compressed file under
data/reports/vector_index/index.pkl, containing the chunk list, the
embedded matrix, and the fitted backend object so queries can be embedded
with the same model used at build time.

No external API calls are made.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import REPORTS_DIR
from src.rag.document_loader import DocumentChunk, load_all_reports

MARKDOWN_DIR = REPORTS_DIR / "markdown"
INDEX_DIR = REPORTS_DIR / "vector_index"


# ── Embedding backends ────────────────────────────────────────────────────────

class TFIDFBackend:
    """
    TF-IDF embedding backend using scikit-learn.

    Produces L2-normalized sparse vectors converted to dense float32 arrays
    for uniform handling alongside other backends. No model download required.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=8_000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        self._fitted = False

    @property
    def name(self) -> str:
        return "tfidf"

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        return self._vectorizer.transform(texts).toarray().astype(np.float32)


class SentenceTransformerBackend:
    """
    Sentence-transformers embedding backend.

    Produces dense semantic embeddings that capture meaning beyond keyword
    overlap. Requires `pip install sentence-transformers`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def fit(self, texts: list[str]) -> None:
        pass  # Sentence-transformers require no fitting step

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


def _select_backend(
    backend_name: str,
) -> TFIDFBackend | SentenceTransformerBackend:
    """Resolve a backend name to a concrete backend instance."""
    if backend_name == "auto":
        try:
            return SentenceTransformerBackend()
        except ImportError:
            return TFIDFBackend()
    if backend_name == "tfidf":
        return TFIDFBackend()
    if backend_name == "sentence-transformers":
        return SentenceTransformerBackend()
    raise ValueError(
        f"Unknown backend '{backend_name}'. "
        "Choose from: auto, tfidf, sentence-transformers."
    )


# ── Vector index ──────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Dense vector index over DocumentChunk objects.

    Stores the embedded chunk matrix and the fitted backend object so that
    queries can be embedded using the exact same model used at build time.
    Persisted to disk as a single joblib-compressed file.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
        backend_name: str,
        backend: TFIDFBackend | SentenceTransformerBackend,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors      # shape: (n_chunks, n_features)
        self._backend_name = backend_name
        self._backend = backend

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def chunks(self) -> list[DocumentChunk]:
        return self._chunks

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def __len__(self) -> int:
        return len(self._chunks)

    # ── Query embedding ───────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string using the backend used at build time."""
        vec = self._backend.embed([text])
        return vec[0]               # shape: (n_features,)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """Persist the index to disk under index_dir as index.pkl."""
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "vectors": self._vectors,
            "backend_name": self._backend_name,
            "backend": self._backend,
            "created_at": datetime.now().isoformat(),
            "n_chunks": len(self._chunks),
        }
        joblib.dump(payload, index_dir / "index.pkl", compress=3)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        """Load a previously saved index from disk."""
        index_file = index_dir / "index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(
                f"No index found at {index_file}. "
                "Call VectorIndex.build() and .save() first."
            )
        payload = joblib.load(index_file)
        return cls(
            chunks=payload["chunks"],
            vectors=payload["vectors"],
            backend_name=payload["backend_name"],
            backend=payload["backend"],
        )

    # ── Builder ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks: list[DocumentChunk],
        backend_name: str = "auto",
    ) -> "VectorIndex":
        """
        Build a vector index from a list of DocumentChunks.

        Selects the embedding backend based on backend_name. With "auto",
        sentence-transformers is used if installed, otherwise TF-IDF.
        """
        if not chunks:
            raise ValueError("Cannot build a vector index from an empty chunk list.")
        backend = _select_backend(backend_name)
        texts = [chunk.content for chunk in chunks]
        backend.fit(texts)
        vectors = backend.embed(texts)
        return cls(chunks, vectors, backend.name, backend)
