"""Unit tests for mempalace.palace_meta (the sidecar metadata file)."""

from __future__ import annotations

import json
import os

import pytest

from mempalace.palace_meta import (
    META_FILENAME,
    META_SCHEMA_VERSION,
    PalaceMeta,
    meta_from_embedder,
    read_meta,
    write_meta,
)
from mempalace.providers.errors import DimensionMismatchError


# ── read_meta ─────────────────────────────────────────────────────────────


def test_read_meta_returns_none_when_file_missing(tmp_path):
    assert read_meta(str(tmp_path)) is None


def test_read_meta_returns_none_when_parent_missing(tmp_path):
    # Subdir doesn't exist at all.
    missing = tmp_path / "nope"
    assert read_meta(str(missing)) is None


def test_read_meta_returns_none_on_corrupt_json(tmp_path):
    path = tmp_path / META_FILENAME
    path.write_text("not valid json {{")
    assert read_meta(str(tmp_path)) is None


def test_read_meta_returns_none_when_root_is_not_dict(tmp_path):
    path = tmp_path / META_FILENAME
    path.write_text(json.dumps(["this", "is", "a", "list"]))
    assert read_meta(str(tmp_path)) is None


def test_read_meta_parses_valid_file(tmp_path):
    path = tmp_path / META_FILENAME
    payload = {
        "version": 1,
        "embedding_provider": "openai_compatible",
        "embedding_model": "Qwen3-Embedding-0.6B-8bit",
        "embedding_dim": 1024,
        "collection_name": "mempalace_drawers",
        "created_at": "2026-04-10T00:00:00Z",
        "updated_at": "2026-04-10T00:00:00Z",
    }
    path.write_text(json.dumps(payload))
    meta = read_meta(str(tmp_path))
    assert meta is not None
    assert meta.embedding_provider == "openai_compatible"
    assert meta.embedding_model == "Qwen3-Embedding-0.6B-8bit"
    assert meta.embedding_dim == 1024
    assert meta.version == 1


def test_read_meta_tolerates_missing_optional_fields(tmp_path):
    """Older sidecar files without some fields should still load with sane
    defaults rather than crashing."""
    path = tmp_path / META_FILENAME
    path.write_text(json.dumps({"embedding_provider": "foo", "embedding_model": "bar"}))
    meta = read_meta(str(tmp_path))
    assert meta is not None
    assert meta.embedding_provider == "foo"
    assert meta.embedding_dim is None
    assert meta.collection_name == "mempalace_drawers"


# ── write_meta ────────────────────────────────────────────────────────────


def test_write_meta_creates_directory_and_file(tmp_path):
    target = tmp_path / "nested" / "palace"
    meta = PalaceMeta(
        embedding_provider="openai_compatible",
        embedding_model="m",
        embedding_dim=768,
    )
    write_meta(str(target), meta)
    assert (target / META_FILENAME).exists()


def test_write_meta_round_trip(tmp_path):
    original = PalaceMeta(
        embedding_provider="openai_compatible",
        embedding_model="Qwen3-Embedding-0.6B-8bit",
        embedding_dim=1024,
    )
    write_meta(str(tmp_path), original)
    loaded = read_meta(str(tmp_path))
    assert loaded is not None
    assert loaded.embedding_provider == original.embedding_provider
    assert loaded.embedding_model == original.embedding_model
    assert loaded.embedding_dim == original.embedding_dim
    assert loaded.version == META_SCHEMA_VERSION


def test_write_meta_updates_updated_at_timestamp(tmp_path):
    meta = PalaceMeta(
        embedding_provider="x",
        embedding_model="y",
        embedding_dim=10,
        updated_at="2020-01-01T00:00:00+00:00",
    )
    write_meta(str(tmp_path), meta)
    loaded = read_meta(str(tmp_path))
    # write_meta always bumps updated_at to "now"
    assert loaded.updated_at != "2020-01-01T00:00:00+00:00"


def test_write_meta_is_atomic_via_tmp_file(tmp_path):
    """The write uses a .tmp file and os.replace so a crash mid-write
    shouldn't leave a partial file."""
    meta = PalaceMeta(embedding_provider="x", embedding_model="y", embedding_dim=10)
    write_meta(str(tmp_path), meta)
    # The .tmp file should not exist after successful write.
    tmp_artifact = tmp_path / (META_FILENAME + ".tmp")
    assert not tmp_artifact.exists()


def test_write_meta_sets_restrictive_permissions_on_unix(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX-only permission test")
    meta = PalaceMeta(embedding_provider="x", embedding_model="y", embedding_dim=10)
    write_meta(str(tmp_path), meta)
    mode = os.stat(tmp_path / META_FILENAME).st_mode & 0o777
    assert mode == 0o600


# ── meta_from_embedder ───────────────────────────────────────────────────


class _FakeEmbedder:
    def __init__(self, provider, model, dim):
        self.provider_kind = provider
        self.model_name = model
        self.embed_dim = dim


def test_meta_from_embedder_captures_identity():
    e = _FakeEmbedder("openai_compatible", "Qwen3-Embedding-0.6B-8bit", 1024)
    meta = meta_from_embedder(e)
    assert meta.embedding_provider == "openai_compatible"
    assert meta.embedding_model == "Qwen3-Embedding-0.6B-8bit"
    assert meta.embedding_dim == 1024
    assert meta.collection_name == "mempalace_drawers"


def test_meta_from_embedder_custom_collection_name():
    e = _FakeEmbedder("sentence_transformers", "bge-small", 384)
    meta = meta_from_embedder(e, collection_name="custom_coll")
    assert meta.collection_name == "custom_coll"


def test_meta_from_embedder_tolerates_missing_attrs():
    class _Bare:
        pass

    meta = meta_from_embedder(_Bare())
    assert meta.embedding_provider == "unknown"
    assert meta.embedding_model == "unknown"
    assert meta.embedding_dim is None


# ── DimensionMismatchError ────────────────────────────────────────────────


def test_dim_mismatch_message_includes_actionable_guidance():
    try:
        raise DimensionMismatchError(
            palace_path="/tmp/foo",
            stored_dim=384,
            stored_model="all-MiniLM-L6-v2",
            current_dim=1024,
            current_model="Qwen3-Embedding-0.6B-8bit",
        )
    except DimensionMismatchError as e:
        msg = str(e)
        assert "384" in msg
        assert "1024" in msg
        assert "all-MiniLM-L6-v2" in msg
        assert "Qwen3-Embedding-0.6B-8bit" in msg
        assert "mempalace reembed" in msg
        assert "/tmp/foo" in msg


def test_dim_mismatch_exposes_attributes():
    e = DimensionMismatchError(
        palace_path="/tmp/x",
        stored_dim=768,
        stored_model="a",
        current_dim=1024,
        current_model="b",
    )
    assert e.palace_path == "/tmp/x"
    assert e.stored_dim == 768
    assert e.current_dim == 1024
