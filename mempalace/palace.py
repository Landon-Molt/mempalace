"""
palace.py — Shared palace operations.

Consolidates ChromaDB access patterns used by miners, the MCP server, the
search layer, and the CLI. The canonical entry point is
:func:`get_collection`, which opens (or creates) the palace's ChromaDB
collection with a config-resolved embedding provider.

Every caller that touches ChromaDB should go through ``get_collection`` and
pass the same ``config`` object. Bypassing this helper is a recipe for
subtle vector-space mismatches: writes go through one embedder, reads go
through another, and retrieval quality silently collapses.

Backward compat: when ``config`` is ``None`` (the default), the helper
behaves exactly like the pre-refactor version — no provider resolution, no
sidecar meta, no dim check. Existing installs that never set an embedding
config see byte-identical behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import chromadb

from .palace_meta import (
    META_FILENAME,
    PalaceMeta,
    meta_from_embedder,
    read_meta,
    write_meta,
)
from .providers import (
    DimensionMismatchError,
    ProviderError,
    ProviderUnavailableError,
    resolve_embedder,
)
from .providers.chroma_adapter import ChromaProviderAdapter

logger = logging.getLogger("mempalace.palace")

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}


def get_collection(
    palace_path: str,
    collection_name: str = "mempalace_drawers",
    config: Optional[Any] = None,
    *,
    allow_dim_mismatch: bool = False,
):
    """Open or create a palace ChromaDB collection.

    Parameters
    ----------
    palace_path : str
        Filesystem path to the palace directory (the ChromaDB data dir).
        Created with 0700 permissions if missing.
    collection_name : str
        ChromaDB collection to open or create. Defaults to
        ``"mempalace_drawers"``.
    config : MempalaceConfig, dict, or None
        Source of embedding configuration. If ``None``, the helper falls
        back to ChromaDB's built-in default — equivalent to pre-refactor
        behavior. If provided, the helper resolves an embedder via
        ``providers.resolve_embedder`` and attaches it to the collection
        as its ``embedding_function``.
    allow_dim_mismatch : bool
        Set to ``True`` by ``mempalace reembed`` to bypass the sidecar
        dimension check (it's about to rewrite every vector anyway).
        Never set this from regular read/write paths.

    Raises
    ------
    DimensionMismatchError
        If the palace's sidecar meta records a dimension that differs from
        the dimension the current config resolves to. This is a hard stop
        — the user must run ``mempalace reembed`` or revert the config.
    ProviderUnavailableError
        If an OpenAI-compatible backend is configured but unreachable
        (network error, 5xx, timeout).
    """
    os.makedirs(palace_path, exist_ok=True)
    try:
        os.chmod(palace_path, 0o700)
    except (OSError, NotImplementedError):
        pass

    client = chromadb.PersistentClient(path=palace_path)
    embedder = resolve_embedder(config)

    if embedder is None:
        # Backward-compat path: ChromaDB default embedding function, no
        # sidecar interaction. Matches pre-refactor behavior byte-for-byte.
        try:
            return client.get_collection(collection_name)
        except Exception:
            return client.create_collection(collection_name)

    # Warmup: discover embedding dimension if the provider doesn't already
    # know it. This is a single tiny embed call (<100ms against oMLX).
    if embedder.embed_dim is None:
        try:
            embedder.embed("mempalace warmup")
        except ProviderUnavailableError:
            # Can't open the collection safely without knowing the embedder
            # is working. Surface the error to the caller.
            raise
        except ProviderError as e:
            logger.warning("embedder warmup failed: %s", e)
        except Exception as e:
            logger.warning("embedder warmup raised unexpectedly: %s", e)

    current_dim = embedder.embed_dim
    current_model = embedder.model_name

    # Sidecar meta: enforce dimension match; warn on model rename.
    stored_meta = read_meta(palace_path)
    if stored_meta is not None:
        stored_dim = stored_meta.embedding_dim
        dims_differ = (
            stored_dim is not None
            and current_dim is not None
            and current_dim != stored_dim
        )
        if dims_differ and not allow_dim_mismatch:
            raise DimensionMismatchError(
                palace_path=palace_path,
                stored_dim=stored_dim,
                stored_model=stored_meta.embedding_model,
                current_dim=current_dim,
                current_model=current_model,
            )
        # Same-dim model rename: non-breaking but worth a heads-up.
        if (
            not dims_differ
            and stored_meta.embedding_model != current_model
        ):
            logger.warning(
                "Palace at %s was created with model %r; current config "
                "resolves to %r (same dim=%s). Vectors may be in a different "
                "space. Consider `mempalace reembed` if search quality drops.",
                palace_path,
                stored_meta.embedding_model,
                current_model,
                current_dim,
            )

    adapter = ChromaProviderAdapter(embedder)
    # Request cosine distance for fresh collections. ChromaDB ignores the
    # metadata when the collection already exists, so this is safe to pass
    # on every open. Cosine is the correct space for most modern embedders
    # (sentence-transformers, BGE, Qwen3-Embedding, OpenAI Ada/Embed-3),
    # all of which are designed for cosine similarity.
    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=adapter,
        metadata={"hnsw:space": "cosine"},
    )

    # Write sidecar on first creation, or update if dim has newly become
    # known (i.e., warmup succeeded on this call for the first time).
    if stored_meta is None or (
        stored_meta.embedding_dim is None and current_dim is not None
    ):
        try:
            meta = meta_from_embedder(embedder, collection_name=collection_name)
            # Preserve created_at when updating an existing meta that
            # was missing its dim.
            if stored_meta is not None:
                meta.created_at = stored_meta.created_at
            write_meta(palace_path, meta)
        except Exception as e:
            logger.warning("Failed to write palace meta: %s", e)

    return col


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    When check_mtime=True (used by project miner), returns False if the file
    has been modified since it was last mined, so it gets re-mined.
    When check_mtime=False (used by convo miner), just checks existence.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        if check_mtime:
            stored_meta = results.get("metadatas", [{}])[0]
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return float(stored_mtime) == current_mtime
        return True
    except Exception:
        return False
