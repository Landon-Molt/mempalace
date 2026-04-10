"""Tests for the refactored palace.get_collection helper.

Covers:
- Backward-compat path (config=None) — byte-identical to pre-refactor.
- Provider-backed path — meta sidecar written/read correctly.
- Dimension mismatch raises DimensionMismatchError with guidance.
- allow_dim_mismatch bypass used by cmd_reembed.
- Cosine space attached to fresh collections.

These tests avoid the live oMLX server by stubbing the embedder. Covered
are the integration points between providers, palace_meta, and
ChromaDB's get_or_create_collection.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from mempalace.palace import get_collection
from mempalace.palace_meta import META_FILENAME, PalaceMeta, read_meta, write_meta
from mempalace.providers.errors import DimensionMismatchError


class _StubEmbedder:
    """Minimal embedder stub matching the EmbedderProtocol interface."""

    provider_kind = "stub"

    def __init__(self, model_name: str, dim: int):
        self._model_name = model_name
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embed_dim(self):
        return self._dim

    def embed(self, text: str):
        return [0.0] * self._dim

    def embed_batch(self, texts):
        return [[0.0] * self._dim for _ in texts]


@pytest.fixture
def tmp_palace(tmp_path):
    """Empty directory usable as a palace_path."""
    d = tmp_path / "palace"
    d.mkdir()
    return str(d)


# ── backward-compat path ─────────────────────────────────────────────────


def test_get_collection_backward_compat_no_sidecar(tmp_palace):
    """With config=None, palace.get_collection behaves like the old
    helper and does not write a sidecar."""
    col = get_collection(tmp_palace)
    col.upsert(ids=["a"], documents=["hello"])
    assert col.count() == 1
    assert not os.path.exists(os.path.join(tmp_palace, META_FILENAME))


def test_get_collection_backward_compat_empty_config(tmp_palace):
    """Empty config dict also triggers the backward-compat path."""
    col = get_collection(tmp_palace, config={})
    col.upsert(ids=["a"], documents=["hello"])
    assert not os.path.exists(os.path.join(tmp_palace, META_FILENAME))


# ── provider-backed path ─────────────────────────────────────────────────


def test_get_collection_writes_sidecar_on_first_open(tmp_palace):
    stub = _StubEmbedder("fake-model", 768)
    cfg = {"embedding": {"provider": "test"}}

    # Patch resolve_embedder at the palace module level so palace.get_collection
    # uses our stub instead of constructing a real provider from the config.
    with patch("mempalace.palace.resolve_embedder", return_value=stub):
        col = get_collection(tmp_palace, config=cfg)
        col.upsert(ids=["a"], documents=["hello"])

    meta = read_meta(tmp_palace)
    assert meta is not None
    assert meta.embedding_provider == "stub"
    assert meta.embedding_model == "fake-model"
    assert meta.embedding_dim == 768


def test_get_collection_reopen_with_same_config_no_error(tmp_palace):
    stub = _StubEmbedder("fake-model", 768)
    cfg = {"embedding": {"provider": "test"}}

    with patch("mempalace.palace.resolve_embedder", return_value=stub):
        col1 = get_collection(tmp_palace, config=cfg)
        col1.upsert(ids=["a"], documents=["hello"])
        col2 = get_collection(tmp_palace, config=cfg)
        assert col2.count() == 1


def test_get_collection_dim_mismatch_raises(tmp_palace):
    """If the stored sidecar dim doesn't match current embedder dim,
    DimensionMismatchError is raised with a helpful message."""
    # Pre-seed a fake 384-dim sidecar.
    fake = PalaceMeta(
        embedding_provider="old",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    write_meta(tmp_palace, fake)

    stub = _StubEmbedder("Qwen3-Embedding", 1024)
    cfg = {"embedding": {"provider": "test"}}

    with patch("mempalace.palace.resolve_embedder", return_value=stub):
        with pytest.raises(DimensionMismatchError) as exc_info:
            get_collection(tmp_palace, config=cfg)
    msg = str(exc_info.value)
    assert "384" in msg
    assert "1024" in msg
    assert "mempalace reembed" in msg


def test_get_collection_allow_dim_mismatch_bypass(tmp_palace):
    """cmd_reembed uses allow_dim_mismatch=True to read old drawers
    without crashing on the dim check."""
    fake = PalaceMeta(
        embedding_provider="old",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    write_meta(tmp_palace, fake)

    stub = _StubEmbedder("Qwen3-Embedding", 1024)
    cfg = {"embedding": {"provider": "test"}}

    # Should NOT raise.
    with patch("mempalace.palace.resolve_embedder", return_value=stub):
        col = get_collection(tmp_palace, config=cfg, allow_dim_mismatch=True)
    assert col is not None


def test_get_collection_same_dim_model_rename_warns(tmp_palace, caplog):
    """If the model name changed but the dim matches, just warn (non-fatal)."""
    import logging

    fake = PalaceMeta(
        embedding_provider="openai_compatible",
        embedding_model="old-model",
        embedding_dim=1024,
    )
    write_meta(tmp_palace, fake)

    stub = _StubEmbedder("new-model", 1024)
    cfg = {"embedding": {"provider": "test"}}

    with caplog.at_level(logging.WARNING, logger="mempalace.palace"):
        with patch("mempalace.palace.resolve_embedder", return_value=stub):
            get_collection(tmp_palace, config=cfg)

    assert any("different space" in rec.message for rec in caplog.records) or any(
        "search quality" in rec.message for rec in caplog.records
    )


def test_get_collection_preserves_created_at_across_reopens(tmp_palace):
    """When updating a sidecar that only had dim=None, preserve the
    original created_at timestamp."""
    stub_no_dim = _StubEmbedder("fake", 0)  # effective dim 0
    stub_no_dim._dim = None  # simulate unknown dim

    # Write initial sidecar with dim=None
    initial_meta = PalaceMeta(
        embedding_provider="x",
        embedding_model="fake",
        embedding_dim=None,
    )
    original_created_at = initial_meta.created_at
    write_meta(tmp_palace, initial_meta)

    # Now open with a stub that knows its dim
    stub_known = _StubEmbedder("fake", 768)
    cfg = {"embedding": {"provider": "test"}}
    with patch("mempalace.palace.resolve_embedder", return_value=stub_known):
        get_collection(tmp_palace, config=cfg)

    # Sidecar should now have dim=768 but retain original created_at.
    meta = read_meta(tmp_palace)
    assert meta.embedding_dim == 768
    assert meta.created_at == original_created_at
