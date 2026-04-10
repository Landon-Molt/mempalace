"""
chroma_adapter.py — wrap any provider as a ChromaDB ``EmbeddingFunction``.

ChromaDB's API expects an ``EmbeddingFunction`` object with a ``__call__``
method that takes a list of documents and returns a list of vectors. This
adapter lets us plug any object that implements ``embed_batch(list[str])``
into the ChromaDB collection API.

The adapter also exposes ``.name()`` for ChromaDB's provenance checks and
caches the embedder's ``embed_dim`` so we can emit it into the palace meta
sidecar without re-calling the backend.
"""

from __future__ import annotations

from typing import Any, List, Protocol

try:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError:  # pragma: no cover — chromadb is always installed
    Documents = list  # type: ignore
    Embeddings = list  # type: ignore

    class EmbeddingFunction:  # type: ignore
        pass


class _EmbedderLike(Protocol):
    """Minimum interface the adapter needs from a provider."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

    @property
    def embed_dim(self) -> int | None: ...

    @property
    def model_name(self) -> str: ...


class ChromaProviderAdapter(EmbeddingFunction):  # type: ignore[misc]
    """Wrap a provider (OpenAICompatibleClient, SentenceTransformersEmbedder)
    as a ChromaDB EmbeddingFunction.

    ChromaDB 0.5.x and 0.6.x call ``__call__(input)`` where ``input`` is a
    list of strings. We forward to the provider's ``embed_batch`` method.
    """

    def __init__(self, embedder: _EmbedderLike):
        self._embedder = embedder

    # ChromaDB's newer EmbeddingFunction API uses ``__call__(self, input)``
    # with a single ``input`` argument typed as Documents. We accept both
    # ``input`` and ``texts`` as aliases so the adapter is resilient to minor
    # ChromaDB version drift.
    def __call__(self, input: Any = None, texts: Any = None) -> List[List[float]]:  # noqa: A002
        docs = input if input is not None else texts
        if docs is None:
            return []
        if isinstance(docs, str):
            docs = [docs]
        return self._embedder.embed_batch(list(docs))

    def name(self) -> str:
        """Used by ChromaDB's provenance metadata."""
        return f"mempalace_provider:{self._embedder.model_name}"

    @property
    def embedder(self) -> _EmbedderLike:
        """Expose the underlying embedder for dim/meta access."""
        return self._embedder
