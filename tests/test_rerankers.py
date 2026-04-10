"""Tests for mempalace.rerankers — the two-stage reranking module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mempalace.rerankers import Reranker, create_reranker


# ── Reranker class ───────────────────────────────────────────────────────


class TestReranker:
    def test_rerank_reorders_by_score(self):
        mock_client = MagicMock()
        mock_client.rerank.return_value = [(2, 0.95), (0, 0.80), (1, 0.60)]
        r = Reranker(mock_client)

        candidates = [
            {"text": "doc A", "similarity": 0.9},
            {"text": "doc B", "similarity": 0.85},
            {"text": "doc C", "similarity": 0.7},
        ]
        result = r.rerank("query", candidates)

        assert len(result) == 3
        assert result[0]["text"] == "doc C"  # highest rerank score
        assert result[0]["rerank_score"] == 0.95
        assert result[1]["text"] == "doc A"
        assert result[2]["text"] == "doc B"

    def test_rerank_top_n_limits(self):
        mock_client = MagicMock()
        mock_client.rerank.return_value = [(0, 0.9), (1, 0.8), (2, 0.7)]
        r = Reranker(mock_client)

        candidates = [{"text": f"doc{i}"} for i in range(3)]
        result = r.rerank("q", candidates, top_n=2)
        assert len(result) == 2

    def test_rerank_empty_candidates(self):
        mock_client = MagicMock()
        r = Reranker(mock_client)
        assert r.rerank("q", []) == []
        mock_client.rerank.assert_not_called()

    def test_rerank_preserves_candidate_fields(self):
        mock_client = MagicMock()
        mock_client.rerank.return_value = [(0, 0.9)]
        r = Reranker(mock_client)

        candidates = [{"text": "doc", "wing": "proj", "room": "auth", "similarity": 0.5}]
        result = r.rerank("q", candidates)

        assert result[0]["wing"] == "proj"
        assert result[0]["room"] == "auth"
        assert result[0]["similarity"] == 0.5  # original score preserved
        assert result[0]["rerank_score"] == 0.9  # new score added

    def test_rerank_fallback_on_error(self):
        mock_client = MagicMock()
        mock_client.rerank.side_effect = RuntimeError("connection failed")
        r = Reranker(mock_client)

        candidates = [{"text": "a"}, {"text": "b"}]
        result = r.rerank("q", candidates, top_n=2)
        # On error, returns original candidates unchanged
        assert len(result) == 2
        assert result[0]["text"] == "a"

    def test_candidate_multiplier(self):
        mock_client = MagicMock()
        r = Reranker(mock_client, candidate_multiplier=5)
        assert r.candidate_multiplier == 5

    def test_candidate_multiplier_min_1(self):
        mock_client = MagicMock()
        r = Reranker(mock_client, candidate_multiplier=0)
        assert r.candidate_multiplier == 1


# ── create_reranker ──────────────────────────────────────────────────────


class TestCreateReranker:
    def test_returns_none_when_disabled(self):
        config = {"rerank": {"enabled": False}}
        assert create_reranker(config) is None

    def test_returns_none_when_no_rerank_config(self):
        config = {}
        assert create_reranker(config) is None

    @patch("mempalace.providers.resolve_reranker")
    def test_returns_reranker_when_configured(self, mock_resolve):
        mock_client = MagicMock()
        mock_resolve.return_value = mock_client
        config = MagicMock()
        config.rerank = {"enabled": True, "candidate_multiplier": 4}

        r = create_reranker(config)
        assert r is not None
        assert r.candidate_multiplier == 4

    @patch("mempalace.providers.resolve_reranker")
    def test_default_multiplier_is_3(self, mock_resolve):
        mock_client = MagicMock()
        mock_resolve.return_value = mock_client
        config = MagicMock()
        config.rerank = {"enabled": True}

        r = create_reranker(config)
        assert r.candidate_multiplier == 3


# ── Integration with search_memories ─────────────────────────────────────


class TestSearchRerank:
    @patch("mempalace.searcher.get_collection")
    @patch("mempalace.rerankers.create_reranker")
    def test_search_with_rerank_fetches_more_candidates(
        self, mock_create_reranker, mock_get_col
    ):
        from mempalace.searcher import search_memories

        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [[{"wing": "w", "room": "r", "source_file": "f"}] * 3],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_get_col.return_value = mock_col

        mock_reranker = MagicMock()
        mock_reranker.candidate_multiplier = 3
        mock_reranker.rerank.return_value = [
            {"text": "doc3", "wing": "w", "room": "r", "source_file": "f",
             "similarity": 0.7, "rerank_score": 0.95},
        ]
        mock_create_reranker.return_value = mock_reranker

        result = search_memories("test", "/fake", n_results=1, rerank=True)

        # Should have fetched 1 * 3 = 3 candidates for reranking
        query_call = mock_col.query.call_args
        assert query_call.kwargs.get("n_results", query_call[1].get("n_results")) == 3
        assert result["reranked"] is True
        assert len(result["results"]) == 1

    @patch("mempalace.searcher.get_collection")
    def test_search_without_rerank_uses_original_count(self, mock_get_col):
        from mempalace.searcher import search_memories

        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"wing": "w", "room": "r", "source_file": "f"}] * 2],
            "distances": [[0.1, 0.2]],
        }
        mock_get_col.return_value = mock_col

        result = search_memories("test", "/fake", n_results=2, rerank=False)
        assert result["reranked"] is False
        assert len(result["results"]) == 2
