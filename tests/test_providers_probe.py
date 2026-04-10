"""Unit tests for mempalace.providers.probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from mempalace.providers.probe import probe_omlx


def _mock_response(status_code: int, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error"
        )
    resp.json.return_value = payload
    return resp


def test_probe_returns_model_list_on_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(
            200,
            {
                "data": [
                    {"id": "Qwen3-Embedding-0.6B-8bit"},
                    {"id": "Qwen3-Reranker-4B-4bit-MLX"},
                    {"id": "gpt-4o"},
                ]
            },
        )
        models = probe_omlx("http://127.0.0.1:8000/v1")
    assert models is not None
    assert "Qwen3-Embedding-0.6B-8bit" in models
    assert len(models) == 3


def test_probe_normalizes_trailing_v1():
    """Accepts both forms: with and without /v1 suffix."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": [{"id": "foo"}]})
        probe_omlx("http://127.0.0.1:8000")
        url_called = mock_get.call_args.args[0]
        assert url_called.endswith("/v1/models")


def test_probe_normalizes_no_v1_suffix():
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": [{"id": "foo"}]})
        probe_omlx("http://127.0.0.1:8000/v1")
        url_called = mock_get.call_args.args[0]
        assert url_called.endswith("/v1/models")
        # Should NOT double up to /v1/v1/models.
        assert "/v1/v1/" not in url_called


def test_probe_returns_none_on_connection_error():
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        assert probe_omlx("http://127.0.0.1:9999") is None


def test_probe_returns_none_on_timeout():
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        assert probe_omlx("http://slow.host/v1") is None


def test_probe_returns_none_on_http_error():
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(500, {})
        assert probe_omlx("http://host/v1") is None


def test_probe_returns_none_on_malformed_json():
    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp
        assert probe_omlx("http://host/v1") is None


def test_probe_returns_none_when_data_missing():
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"not_data": []})
        assert probe_omlx("http://host/v1") is None


def test_probe_returns_none_for_empty_model_list():
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": []})
        assert probe_omlx("http://host/v1") is None


def test_probe_timeout_value_is_short():
    """Probe should use a short timeout so init flow doesn't hang."""
    with patch("requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"data": [{"id": "x"}]})
        probe_omlx("http://127.0.0.1:8000")
        # timeout keyword should be present and <= 5 seconds
        timeout = mock_get.call_args.kwargs.get("timeout")
        assert timeout is not None
        assert timeout <= 5.0
