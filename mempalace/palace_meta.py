"""
palace_meta.py — sidecar metadata for a MemPalace data directory.

Each palace directory gets a small JSON file at ``.mempalace_meta.json``
recording which embedding model and provider was used to populate it.
This lets us detect dimension mismatches at open time: if the config now
resolves to a 1024-dim embedder but the palace was built with a 384-dim
one, queries would be computing vectors in the wrong space and retrieval
would be silently broken. We fail loudly instead and direct the user to
``mempalace reembed``.

The sidecar is optional. Missing/corrupt meta just means we skip the
dim check and proceed, which preserves backward compatibility for
existing palaces that predate this refactor.

Schema::

    {
      "version": 1,
      "embedding_provider": "openai_compatible",
      "embedding_model": "Qwen3-Embedding-0.6B-8bit",
      "embedding_dim": 1024,
      "collection_name": "mempalace_drawers",
      "created_at": "2026-04-10T04:00:00Z",
      "updated_at": "2026-04-10T04:00:00Z"
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mempalace.palace_meta")

META_FILENAME = ".mempalace_meta.json"
META_SCHEMA_VERSION = 1


@dataclass
class PalaceMeta:
    """Recorded state of a palace's embedding configuration."""

    embedding_provider: str
    embedding_model: str
    embedding_dim: Optional[int]
    collection_name: str = "mempalace_drawers"
    version: int = META_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PalaceMeta":
        return cls(
            embedding_provider=data.get("embedding_provider", "unknown"),
            embedding_model=data.get("embedding_model", "unknown"),
            embedding_dim=data.get("embedding_dim"),
            collection_name=data.get("collection_name", "mempalace_drawers"),
            version=int(data.get("version", 0)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def _meta_path(palace_path: str) -> Path:
    return Path(palace_path) / META_FILENAME


def read_meta(palace_path: str) -> Optional[PalaceMeta]:
    """Return the sidecar meta for a palace, or None if missing/unreadable.

    Never raises — a corrupt meta file is logged at WARNING level and
    treated as missing. Callers should interpret ``None`` as "unknown
    state, proceed without dim check."
    """
    path = _meta_path(palace_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Palace meta at %s is unreadable (%s); treating as missing", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Palace meta at %s is not a JSON object; treating as missing", path)
        return None
    return PalaceMeta.from_dict(data)


def write_meta(palace_path: str, meta: PalaceMeta) -> None:
    """Persist the sidecar meta to ``{palace_path}/.mempalace_meta.json``.

    Creates the parent directory if missing. Restricts the file to owner
    read/write on Unix (matching palace.py's chmod 0o700 convention).
    On Windows the permission bits are ignored silently.
    """
    path = _meta_path(palace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.updated_at = datetime.now(timezone.utc).isoformat()
    data = meta.to_dict()
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def meta_from_embedder(embedder, collection_name: str = "mempalace_drawers") -> PalaceMeta:
    """Build a fresh ``PalaceMeta`` from a provider instance.

    ``embedder`` must expose ``provider_kind``, ``model_name``, and
    ``embed_dim`` (possibly ``None`` if unknown). Both our providers
    (``OpenAICompatibleClient``, ``SentenceTransformersEmbedder``) satisfy
    this informal interface.
    """
    return PalaceMeta(
        embedding_provider=getattr(embedder, "provider_kind", "unknown"),
        embedding_model=getattr(embedder, "model_name", "unknown"),
        embedding_dim=getattr(embedder, "embed_dim", None),
        collection_name=collection_name,
    )
