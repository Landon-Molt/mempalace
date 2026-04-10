"""
rerankers.py — optional search-result reranking via cross-encoder models.

Wraps the ``OpenAICompatibleClient.rerank()`` endpoint to reorder search
candidates by semantic relevance. The reranker sees the full text of both
the query and each candidate, giving it a much richer signal than the
embedding-similarity score alone.

The search pipeline becomes:

1. ChromaDB vector search → N * multiplier candidates (fast, approximate)
2. Reranker → top N candidates (slower, precise)

This two-stage approach is standard in production retrieval systems and
typically improves recall@5 by 10-20%.

Usage::

    from mempalace.rerankers import Reranker

    reranker = Reranker(client)  # OpenAICompatibleClient with rerank_model
    reranked = reranker.rerank(
        query="why did we switch databases",
        candidates=[
            {"text": "Decided PostgreSQL...", "metadata": {...}, "similarity": 0.82},
            {"text": "Auth0 migration...", "metadata": {...}, "similarity": 0.79},
            ...
        ],
        top_n=5,
    )

When the reranker is disabled or unavailable, ``rerank()`` returns the
input candidates unchanged — the search pipeline degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mempalace.rerankers")


class Reranker:
    """Two-stage reranker for search results.

    Parameters
    ----------
    client : OpenAICompatibleClient
        Must have ``rerank_model`` configured.
    candidate_multiplier : int
        How many extra candidates to fetch from ChromaDB so the reranker
        has a good selection. Default 3 (fetch 3× what the user asked for,
        rerank down to the requested count).
    """

    def __init__(self, client, candidate_multiplier: int = 3):
        self._client = client
        self.candidate_multiplier = max(1, int(candidate_multiplier))

    @property
    def model_name(self) -> str:
        return getattr(self._client, "_rerank_model", "unknown")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates by cross-encoder relevance.

        Parameters
        ----------
        query : str
            The user's search query.
        candidates : list of dict
            Each dict has at least ``"text"`` (the document content).
            May also have ``"metadata"``, ``"similarity"``, ``"id"``, etc.
        top_n : int, optional
            Return at most this many results. Defaults to len(candidates).

        Returns
        -------
        list of dict
            Same shape as input, reordered by rerank score descending.
            Each dict gets an added ``"rerank_score"`` key.
        """
        if not candidates:
            return []

        n = int(top_n) if top_n else len(candidates)
        texts = [c.get("text", "") for c in candidates]

        try:
            # client.rerank returns [(index, score), ...] sorted desc.
            ranked_pairs = self._client.rerank(query, texts, top_n=n)
        except Exception as e:
            logger.warning("Reranker failed (%s); returning original order", e)
            return candidates[:n]

        # Map scores back onto the candidate dicts.
        result = []
        for idx, score in ranked_pairs:
            if 0 <= idx < len(candidates):
                entry = dict(candidates[idx])
                entry["rerank_score"] = score
                result.append(entry)

        return result[:n]


def create_reranker(config) -> Optional[Reranker]:
    """Build a Reranker from config, or return None if disabled/unavailable.

    Reads the ``rerank`` config section. If ``enabled`` is False or the
    reranker client can't be resolved, returns None so the search pipeline
    skips reranking silently.
    """
    from .providers import resolve_reranker

    client = resolve_reranker(config)
    if client is None:
        return None

    # Read candidate_multiplier from config, default 3.
    rerank_cfg = {}
    if hasattr(config, "rerank"):
        rerank_cfg = config.rerank if isinstance(config.rerank, dict) else {}
    elif isinstance(config, dict):
        rerank_cfg = config.get("rerank", {})

    multiplier = int(rerank_cfg.get("candidate_multiplier", 3))
    return Reranker(client, candidate_multiplier=multiplier)
