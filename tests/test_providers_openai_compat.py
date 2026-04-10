"""Unit tests for mempalace.providers.openai_compatible.

Uses mocked ``requests`` responses so the tests don't depend on a live
oMLX server. Covers the tricky parts of the port:

- batch embedding cap at 48
- lone UTF-16 surrogate stripping
- null-embedding retry at oMLX's batch bug
- timeout / unavailable handling
- rerank score sorting + parsing
- generate payload shape + Qwen /no_think auto-append
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mempalace.providers.errors import ProviderError, ProviderUnavailableError
from mempalace.providers.openai_compatible import (
    OpenAICompatibleClient,
    _normalize_base_url,
    _strip_lone_surrogates,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _mock_response(status_code: int, payload):
    """Build a mock requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
    resp.json.return_value = payload
    return resp


# ── _strip_lone_surrogates ────────────────────────────────────────────────


def test_strip_lone_surrogates_removes_high():
    assert _strip_lone_surrogates("hello\ud800world") == "helloworld"


def test_strip_lone_surrogates_removes_low():
    assert _strip_lone_surrogates("hello\udc00world") == "helloworld"


def test_strip_lone_surrogates_preserves_valid_text():
    # Plain ASCII, CJK, emoji, and German umlauts all pass through unchanged.
    text = "Hello 世界 🎉 München straße"
    assert _strip_lone_surrogates(text) == text


def test_strip_lone_surrogates_empty():
    assert _strip_lone_surrogates("") == ""


# ── _normalize_base_url ───────────────────────────────────────────────────


def test_normalize_base_url_adds_v1():
    assert _normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"


def test_normalize_base_url_strips_trailing_slash():
    assert _normalize_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/v1"


def test_normalize_base_url_keeps_existing_v1():
    assert _normalize_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_normalize_base_url_v1_with_trailing_slash():
    assert _normalize_base_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1"


# ── single embed ──────────────────────────────────────────────────────────


def test_embed_returns_vector_and_caches_dim():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    mock_vec = [0.1, 0.2, 0.3, 0.4]
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"data": [{"embedding": mock_vec}]})
        v = client.embed("hello")
    assert v == mock_vec
    assert client.embed_dim == 4
    # Second call shouldn't change cached dim.
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"data": [{"embedding": mock_vec}]})
        client.embed("world")
    assert client.embed_dim == 4


def test_embed_strips_lone_surrogates_before_send():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(200, {"data": [{"embedding": [0.1]}]})
        client.embed("hello\ud800world")
        # Inspect the payload sent to requests.post.
        call_args = mock_post.call_args
        body = json.loads(call_args.kwargs["data"])
        assert body["input"] == "helloworld"


def test_embed_raises_when_model_not_configured():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")  # no embed_model
    with pytest.raises(ProviderError, match="embed_model"):
        client.embed("hello")


def test_embed_raises_on_5xx_as_unavailable():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(503, "service unavailable")
        with pytest.raises(ProviderUnavailableError):
            client.embed("hello")


def test_embed_raises_on_4xx_as_provider_error():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(404, "not found")
        with pytest.raises(ProviderError):
            client.embed("hello")


def test_embed_raises_on_network_error_as_unavailable():
    import requests as requests_mod

    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests_mod.exceptions.ConnectionError("refused")
        with pytest.raises(ProviderUnavailableError):
            client.embed("hello")


# ── batch embed ───────────────────────────────────────────────────────────


def test_embed_batch_splits_at_batch_size_cap():
    """Verify the oMLX batch-48 cap is enforced — a 100-item input should
    produce 3 sub-batches: 48 + 48 + 4."""
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
        embed_batch_size=48,
    )
    texts = [f"text{i}" for i in range(100)]

    def fake_post(url, **kwargs):
        body = json.loads(kwargs["data"])
        batch = body["input"]
        return _mock_response(
            200,
            {"data": [{"index": i, "embedding": [float(i)]} for i in range(len(batch))]},
        )

    with patch("requests.post", side_effect=fake_post) as mock_post:
        vecs = client.embed_batch(texts)

    assert len(vecs) == 100
    assert all(len(v) == 1 for v in vecs)
    # Three calls: 48 + 48 + 4.
    assert mock_post.call_count == 3


def test_embed_batch_retries_null_embedding_individually():
    """If the server returns a null embedding at some position (oMLX
    bug), the client should fall back to an individual embed call for
    that item."""
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
        embed_batch_size=48,
    )
    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        body = json.loads(kwargs["data"])
        if isinstance(body["input"], list):
            # Batch: first item returns null, rest return real vectors.
            data = []
            for i, _ in enumerate(body["input"]):
                if i == 0:
                    data.append({"index": 0, "embedding": None})
                else:
                    data.append({"index": i, "embedding": [float(i)]})
            return _mock_response(200, {"data": data})
        else:
            # Single-item retry.
            return _mock_response(200, {"data": [{"embedding": [99.0]}]})

    with patch("requests.post", side_effect=fake_post):
        vecs = client.embed_batch(["a", "b", "c"])

    assert vecs[0] == [99.0]  # retry result
    assert vecs[1] == [1.0]
    assert vecs[2] == [2.0]


def test_embed_batch_empty_input_returns_empty():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
    )
    assert client.embed_batch([]) == []


def test_embed_batch_respects_custom_batch_size():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="test-model",
        embed_batch_size=2,
    )
    texts = ["a", "b", "c", "d", "e"]

    def fake_post(url, **kwargs):
        body = json.loads(kwargs["data"])
        batch = body["input"]
        return _mock_response(
            200,
            {"data": [{"index": i, "embedding": [float(i)]} for i in range(len(batch))]},
        )

    with patch("requests.post", side_effect=fake_post) as mock_post:
        client.embed_batch(texts)

    # 3 calls for 5 items with batch_size=2 → 2+2+1.
    assert mock_post.call_count == 3


# ── generate ──────────────────────────────────────────────────────────────


def test_generate_returns_completion_content():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="gpt-4o",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "the answer is 4"}}]}
        )
        result = client.generate("what is 2+2?")
    assert result == "the answer is 4"


def test_generate_auto_appends_no_think_for_qwen():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="Qwen3.5-35B-A3B-8bit",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "ok"}}]}
        )
        client.generate("test prompt")
        body = json.loads(mock_post.call_args.kwargs["data"])
    user_msg = body["messages"][-1]["content"]
    assert user_msg.endswith("/no_think")


def test_generate_does_not_append_no_think_for_non_qwen():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="gpt-4o",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "ok"}}]}
        )
        client.generate("test prompt")
        body = json.loads(mock_post.call_args.kwargs["data"])
    assert "/no_think" not in body["messages"][-1]["content"]


def test_generate_thinking_true_skips_no_think_marker():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="Qwen3.5-35B-A3B-8bit",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "ok"}}]}
        )
        client.generate("test prompt", thinking=True)
        body = json.loads(mock_post.call_args.kwargs["data"])
    assert "/no_think" not in body["messages"][-1]["content"]


def test_generate_adds_system_message_when_given():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="gpt-4o",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "ok"}}]}
        )
        client.generate("hi", system="be helpful")
        body = json.loads(mock_post.call_args.kwargs["data"])
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "be helpful"
    assert body["messages"][1]["role"] == "user"


def test_generate_raises_when_model_not_configured():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")  # no generate_model
    with pytest.raises(ProviderError, match="generate_model"):
        client.generate("hello")


# ── rerank ────────────────────────────────────────────────────────────────


def test_rerank_returns_sorted_pairs():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        rerank_model="Qwen3-Reranker-4B-4bit-MLX",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200,
            {
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            },
        )
        pairs = client.rerank("query", ["a", "b", "c"])
    assert pairs[0] == (1, 0.9)
    assert pairs[1] == (2, 0.5)
    assert pairs[2] == (0, 0.2)


def test_rerank_accepts_score_key_fallback():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        rerank_model="some-reranker",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200,
            {"results": [{"index": 0, "score": 0.7}, {"index": 1, "score": 0.3}]},
        )
        pairs = client.rerank("q", ["x", "y"])
    assert pairs[0] == (0, 0.7)
    assert pairs[1] == (1, 0.3)


def test_rerank_top_n_limits_results():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        rerank_model="some-reranker",
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = _mock_response(
            200,
            {
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.5},
                    {"index": 2, "relevance_score": 0.9},
                ]
            },
        )
        pairs = client.rerank("q", ["a", "b", "c"], top_n=2)
    assert len(pairs) == 2
    assert pairs[0] == (2, 0.9)
    assert pairs[1] == (1, 0.5)


def test_rerank_empty_documents_returns_empty():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        rerank_model="some-reranker",
    )
    assert client.rerank("query", []) == []


def test_rerank_raises_when_model_not_configured():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")
    with pytest.raises(ProviderError, match="rerank_model"):
        client.rerank("query", ["a"])


# ── list_models ───────────────────────────────────────────────────────────


def test_list_models_returns_ids():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            200,
            {"data": [{"id": "Qwen3-Embedding-0.6B-8bit"}, {"id": "gpt-4o"}]},
        )
        models = client.list_models()
    assert "Qwen3-Embedding-0.6B-8bit" in models
    assert "gpt-4o" in models


def test_list_models_returns_empty_on_failure():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")
    import requests as requests_mod

    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests_mod.exceptions.ConnectionError("refused")
        assert client.list_models() == []


def test_model_exists_true_and_false():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": [{"id": "foo"}]})
        assert client.model_exists("foo") is True
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": [{"id": "foo"}]})
        assert client.model_exists("bar") is False


# ── identity properties ──────────────────────────────────────────────────


def test_model_name_prefers_embed_model():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        embed_model="e",
        generate_model="g",
        rerank_model="r",
    )
    assert client.model_name == "e"


def test_model_name_falls_back_to_generate():
    client = OpenAICompatibleClient(
        base_url="http://host:8000/v1",
        generate_model="g",
    )
    assert client.model_name == "g"


def test_provider_kind_is_openai_compatible():
    client = OpenAICompatibleClient(base_url="http://host:8000/v1")
    assert client.provider_kind == "openai_compatible"
