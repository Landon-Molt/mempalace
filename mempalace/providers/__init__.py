"""
mempalace.providers — pluggable backends for embeddings, generation, reranking.

This package lets MemPalace talk to different inference providers through a
single config surface:

- **OpenAI-compatible HTTP** (cloud OpenAI, local oMLX, vLLM, LMStudio, TEI,
  llamafile, and anything else that speaks the OpenAI /v1 API). A single
  ``OpenAICompatibleClient`` covers embedding, chat generation, and reranking.
- **sentence-transformers** (local HuggingFace models). Fallback when no
  network provider is available or when the user wants strict-offline mode.
- **ChromaDB default** (``all-MiniLM-L6-v2`` via ONNX). Returned as ``None``,
  which signals ``palace.get_collection`` to let ChromaDB pick its own
  embedding function — this is the path that preserves byte-identical
  behavior for existing installs that don't set any new config.

Public factory functions (``resolve_*``) take a ``MempalaceConfig`` (or any
mapping-like config object exposing ``.embedding`` / ``.llm`` / ``.rerank``)
and return concrete provider instances. They are intentionally permissive:
pass ``None`` or an empty config and they return ``None``, meaning "use the
default / feature is disabled."
"""

from __future__ import annotations

from typing import Any, Optional

from .errors import (
    ProviderError,
    ProviderUnavailableError,
    DimensionMismatchError,
)
from .openai_compatible import OpenAICompatibleClient
from .probe import probe_omlx

__all__ = [
    "OpenAICompatibleClient",
    "probe_omlx",
    "resolve_embedder",
    "resolve_llm",
    "resolve_reranker",
    "ProviderError",
    "ProviderUnavailableError",
    "DimensionMismatchError",
]


# ── helpers ────────────────────────────────────────────────────────────────

def _get_cfg_section(config: Any, name: str) -> dict:
    """Extract a config section as a dict, tolerating several shapes.

    Accepts:
    - None → {}
    - dict-like with ``.get(name, {})``
    - an object with an attribute named ``name`` returning a dict
    - a ``MempalaceConfig`` instance (has a property accessor)
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        section = config.get(name, {})
    else:
        section = getattr(config, name, {}) or {}
    if not isinstance(section, dict):
        return {}
    return section


# ── embedder ──────────────────────────────────────────────────────────────

def resolve_embedder(config: Any) -> Optional[Any]:
    """Return an embedding provider instance based on config.

    Returns ``None`` if no embedding configuration is present — this signals
    the caller (``palace.get_collection``) to use ChromaDB's built-in default
    embedding function, preserving pre-refactor behavior for existing installs.

    Supported providers:

    - ``"auto"`` — probe oMLX at ``base_url`` (default ``http://127.0.0.1:8000/v1``).
      If reachable, pick an embedding model from the loaded set (preferring
      ones with "Embedding" in the name). If unreachable, fall through to
      sentence-transformers with a multilingual default.
    - ``"openai_compatible"`` — construct an ``OpenAICompatibleClient`` with the
      given ``base_url`` and ``model``.
    - ``"sentence_transformers"`` — load a local HuggingFace model via the
      ``sentence_transformers`` package (lazy import).
    - ``"default"`` — explicit opt-in to ChromaDB's default (returns None).
    """
    cfg = _get_cfg_section(config, "embedding")
    if not cfg:
        return None

    provider_name = (cfg.get("provider") or "default").lower()

    if provider_name == "default":
        return None

    if provider_name == "auto":
        base_url = cfg.get("base_url") or "http://127.0.0.1:8000/v1"
        models = probe_omlx(base_url)
        if models:
            chosen = cfg.get("model")
            if not chosen:
                # Prefer an embedding model if one is loaded.
                for m in models:
                    lower = m.lower()
                    if "embedding" in lower or "embed" in lower:
                        chosen = m
                        break
            if chosen:
                return OpenAICompatibleClient(
                    base_url=base_url,
                    api_key=cfg.get("api_key"),
                    embed_model=chosen,
                    embed_batch_size=int(cfg.get("batch_size") or 48),
                    timeout_seconds=float(cfg.get("timeout_seconds") or 30.0),
                )
        # fall through to sentence-transformers
        from .sentence_transformers_provider import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            model_name=cfg.get("model") or "paraphrase-multilingual-MiniLM-L12-v2",
        )

    if provider_name in ("openai_compatible", "openai"):
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url or not model:
            raise ProviderError(
                "embedding.provider=openai_compatible requires both 'base_url' and 'model'"
            )
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=cfg.get("api_key"),
            embed_model=model,
            embed_batch_size=int(cfg.get("batch_size") or 48),
            timeout_seconds=float(cfg.get("timeout_seconds") or 30.0),
        )

    if provider_name in ("sentence_transformers", "st", "hf"):
        from .sentence_transformers_provider import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            model_name=cfg.get("model") or "paraphrase-multilingual-MiniLM-L12-v2",
        )

    raise ProviderError(f"Unknown embedding provider: {provider_name!r}")


# ── llm (chat / generation) ────────────────────────────────────────────────

def resolve_llm(config: Any) -> Optional[OpenAICompatibleClient]:
    """Return an LLM client for text generation, or None if disabled.

    Only supports OpenAI-compatible backends. Used by the summarizer (Phase 2)
    to call LLMs for AAAK/Wenjian compression.
    """
    cfg = _get_cfg_section(config, "llm")
    if not cfg or not cfg.get("enabled", True) is not False and not cfg:
        return None
    # Treat empty dict or explicit disabled as off.
    if not cfg:
        return None
    if cfg.get("enabled") is False:
        return None

    base_url = cfg.get("base_url") or "http://127.0.0.1:8000/v1"
    model = cfg.get("model")
    if not model:
        # Try auto-detect from oMLX.
        models = probe_omlx(base_url)
        if not models:
            return None
        # Prefer a generation model — skip embedding/rerank names.
        for m in models:
            lower = m.lower()
            if "embed" not in lower and "rerank" not in lower:
                model = m
                break
        if not model:
            return None

    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=cfg.get("api_key"),
        generate_model=model,
        timeout_seconds=float(cfg.get("timeout_seconds") or 60.0),
    )


# ── reranker ───────────────────────────────────────────────────────────────

def resolve_reranker(config: Any) -> Optional[OpenAICompatibleClient]:
    """Return a reranker client, or None if disabled.

    Only supports OpenAI-compatible backends with a Cohere-style ``/rerank``
    endpoint. oMLX exposes this at ``/v1/rerank``.
    """
    cfg = _get_cfg_section(config, "rerank")
    if not cfg or cfg.get("enabled") is False:
        return None

    base_url = cfg.get("base_url") or "http://127.0.0.1:8000/v1"
    model = cfg.get("model")
    if not model:
        models = probe_omlx(base_url)
        if not models:
            return None
        for m in models:
            if "rerank" in m.lower():
                model = m
                break
        if not model:
            return None

    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=cfg.get("api_key"),
        rerank_model=model,
        timeout_seconds=float(cfg.get("timeout_seconds") or 30.0),
    )
