"""Tests for the provider configuration step added to onboarding (Phase 4).

Covers: oMLX probe integration, _ask_stack_mode with various choices,
configure_providers config.json writeback, auto-accept path, and the
non-interactive (--yes) flow.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mempalace.onboarding import (
    _probe_providers,
    _ask_stack_mode,
    configure_providers,
)


# ── _probe_providers ─────────────────────────────────────────────────────


class TestProbeProviders:
    @patch("mempalace.providers.probe.probe_omlx")
    def test_returns_available_with_suggestions(self, mock_probe):
        mock_probe.return_value = [
            "Qwen3-Embedding-0.6B-8bit",
            "Qwen3-Reranker-4B-4bit-MLX",
            "Qwen3.5-35B-A3B-8bit",
        ]
        info = _probe_providers()
        assert info["available"] is True
        assert len(info["models"]) == 3
        assert info["suggestions"]["embedding"] == "Qwen3-Embedding-0.6B-8bit"
        assert info["suggestions"]["rerank"] == "Qwen3-Reranker-4B-4bit-MLX"
        assert info["suggestions"]["generate"] == "Qwen3.5-35B-A3B-8bit"

    @patch("mempalace.providers.probe.probe_omlx")
    def test_returns_unavailable_when_probe_fails(self, mock_probe):
        mock_probe.return_value = None
        info = _probe_providers()
        assert info["available"] is False
        assert info["models"] == []
        assert info["suggestions"] == {}

    @patch("mempalace.providers.probe.probe_omlx")
    def test_handles_models_without_embedding(self, mock_probe):
        mock_probe.return_value = ["gpt-4o", "llama-3-70B"]
        info = _probe_providers()
        assert info["available"] is True
        assert info["suggestions"]["embedding"] is None
        assert info["suggestions"]["generate"] == "gpt-4o"


# ── _ask_stack_mode ──────────────────────────────────────────────────────


class TestAskStackMode:
    def test_auto_accept_with_omlx(self):
        omlx_info = {
            "available": True,
            "models": ["embed", "rerank"],
            "suggestions": {
                "embedding": "Qwen3-Embedding-0.6B-8bit",
                "rerank": "Qwen3-Reranker-4B-4bit-MLX",
                "generate": "Qwen3.5-35B",
            },
        }
        config = _ask_stack_mode(omlx_info, auto_accept=True)
        assert config["embedding"]["provider"] == "openai_compatible"
        assert config["embedding"]["model"] == "Qwen3-Embedding-0.6B-8bit"
        assert config["rerank"]["enabled"] is True
        assert config["rerank"]["model"] == "Qwen3-Reranker-4B-4bit-MLX"
        # Auto-accept defaults to raw (no compression)
        assert "compression" not in config

    def test_auto_accept_without_omlx(self):
        omlx_info = {"available": False, "models": [], "suggestions": {}}
        config = _ask_stack_mode(omlx_info, auto_accept=True)
        assert "embedding" not in config
        assert "rerank" not in config

    @patch("builtins.input", side_effect=["1"])  # raw only
    def test_interactive_raw_only(self, mock_input):
        omlx_info = {"available": False, "models": [], "suggestions": {}}
        config = _ask_stack_mode(omlx_info, auto_accept=False)
        assert "compression" not in config

    @patch("builtins.input", side_effect=["2"])  # raw + aaak
    def test_interactive_aaak(self, mock_input):
        omlx_info = {"available": False, "models": [], "suggestions": {}}
        config = _ask_stack_mode(omlx_info, auto_accept=False)
        assert config["compression"]["format"] == "aaak"
        assert config["compression"]["rule_only"] is True

    @patch("builtins.input", side_effect=["3", "n"])  # wenjian, no LLM
    def test_interactive_wenjian_rule_only(self, mock_input):
        omlx_info = {
            "available": True,
            "models": ["gen"],
            "suggestions": {"embedding": None, "rerank": None, "generate": "gen-model"},
        }
        config = _ask_stack_mode(omlx_info, auto_accept=False)
        assert config["compression"]["format"] == "wenjian"
        assert config["compression"]["rule_only"] is True

    @patch("builtins.input", side_effect=["3", "y"])  # wenjian + LLM
    def test_interactive_wenjian_with_llm(self, mock_input):
        omlx_info = {
            "available": True,
            "models": ["gen"],
            "suggestions": {"embedding": None, "rerank": None, "generate": "gen-model"},
        }
        config = _ask_stack_mode(omlx_info, auto_accept=False)
        assert config["compression"]["format"] == "wenjian"
        assert config["compression"]["llm_model"] == "gen-model"
        assert "rule_only" not in config["compression"]


# ── configure_providers ──────────────────────────────────────────────────


class TestConfigureProviders:
    @patch("mempalace.onboarding._probe_providers")
    def test_auto_accept_writes_config_file(self, mock_probe):
        mock_probe.return_value = {
            "available": True,
            "models": ["Qwen3-Embedding-0.6B-8bit", "Qwen3-Reranker-4B-4bit-MLX"],
            "suggestions": {
                "embedding": "Qwen3-Embedding-0.6B-8bit",
                "rerank": "Qwen3-Reranker-4B-4bit-MLX",
                "generate": None,
            },
        }
        with tempfile.TemporaryDirectory() as d:
            # Seed a minimal config.json
            config_file = os.path.join(d, "config.json")
            with open(config_file, "w") as f:
                json.dump({"palace_path": "/tmp/palace"}, f)

            result = configure_providers(config_dir=d, auto_accept=True)

            # Verify the config was written back with new sections
            with open(config_file) as f:
                written = json.load(f)
            assert written["palace_path"] == "/tmp/palace"  # preserved
            assert written["embedding"]["model"] == "Qwen3-Embedding-0.6B-8bit"
            assert written["rerank"]["enabled"] is True

    @patch("mempalace.onboarding._probe_providers")
    def test_no_omlx_writes_nothing_extra(self, mock_probe):
        mock_probe.return_value = {
            "available": False,
            "models": [],
            "suggestions": {},
        }
        with tempfile.TemporaryDirectory() as d:
            config_file = os.path.join(d, "config.json")
            with open(config_file, "w") as f:
                json.dump({"palace_path": "/tmp"}, f)

            result = configure_providers(config_dir=d, auto_accept=True)
            with open(config_file) as f:
                written = json.load(f)
            # No embedding/rerank sections added
            assert "embedding" not in written
            assert "rerank" not in written

    @patch("mempalace.onboarding._probe_providers")
    def test_merges_with_existing_config(self, mock_probe):
        """Existing config fields should be preserved, not overwritten."""
        mock_probe.return_value = {
            "available": True,
            "models": ["embed"],
            "suggestions": {"embedding": "my-embed", "rerank": None, "generate": None},
        }
        with tempfile.TemporaryDirectory() as d:
            config_file = os.path.join(d, "config.json")
            with open(config_file, "w") as f:
                json.dump({"palace_path": "/my/palace", "custom_field": 42}, f)

            configure_providers(config_dir=d, auto_accept=True)
            with open(config_file) as f:
                written = json.load(f)
            assert written["palace_path"] == "/my/palace"
            assert written["custom_field"] == 42
            assert written["embedding"]["model"] == "my-embed"

    @patch("mempalace.onboarding._probe_providers")
    def test_creates_config_dir_when_writing_updates(self, mock_probe):
        """When oMLX is detected and config needs writing, the dir is created."""
        mock_probe.return_value = {
            "available": True,
            "models": ["embed"],
            "suggestions": {"embedding": "em", "rerank": None, "generate": None},
        }
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "sub", "dir")
            configure_providers(config_dir=nested, auto_accept=True)
            assert os.path.isdir(nested)
            assert os.path.exists(os.path.join(nested, "config.json"))
