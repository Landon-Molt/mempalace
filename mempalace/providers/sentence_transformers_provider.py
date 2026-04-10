"""
sentence_transformers_provider.py — local HuggingFace embedder fallback.

Used when:
  - The user explicitly sets ``embedding.provider = "sentence_transformers"``
  - Auto-mode probes oMLX, finds it unreachable, and falls through

The import of ``sentence_transformers`` is lazy (inside ``__init__``) so that
users who don't pick this provider don't pay the ~400MB torch dependency.

Default model is ``paraphrase-multilingual-MiniLM-L12-v2`` — 384-dim,
multilingual (50+ languages including EN/ZH/DE), ~400MB. Matches what
MemChinesePalace chose as its default.
"""

from __future__ import annotations

from typing import List, Optional


class SentenceTransformersEmbedder:
    """Local HuggingFace embedder via the ``sentence-transformers`` package.

    Compatible with the informal EmbedderProtocol:
      - ``embed(text) -> list[float]``
      - ``embed_batch(texts) -> list[list[float]]``
      - ``embed_dim`` property (known eagerly from model config)
      - ``model_name`` property
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._normalize = normalize

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or choose a different "
                "embedding provider."
            ) from e

        self._model = SentenceTransformer(model_name, device=device)
        # Dim is known eagerly from the model config.
        self._dim: Optional[int] = self._model.get_sentence_embedding_dimension()

    # ── embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(
            [text],
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    # ── metadata ──────────────────────────────────────────────────────────

    @property
    def embed_dim(self) -> Optional[int]:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_kind(self) -> str:
        return "sentence_transformers"
