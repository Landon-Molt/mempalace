"""
openai_compatible.py — unified HTTP client for OpenAI-compatible servers.

This is a Python port of ``/Users/peter-bot/Projects/qmd-omlx/src/omlx-llm.ts``.
It covers three endpoints through a single class:

- ``POST /v1/embeddings``      — OpenAI embeddings API
- ``POST /v1/chat/completions`` — OpenAI chat API
- ``POST /v1/rerank``          — Cohere-style rerank API (oMLX exposes this)

Works against any server that implements this surface: cloud OpenAI,
local oMLX, vLLM, LMStudio, TEI, llamafile, and so on. Peter's primary
target is oMLX running at ``http://127.0.0.1:8000`` with Qwen3 models.

Known quirks ported from qmd-omlx:

1. **Embedding batch cap at 48.** oMLX has a bug where batches of ≥64
   return ``None`` for the first item. We cap batches at 48 and fall
   back to single-item retries if we get a null embedding back.
2. **Lone UTF-16 surrogate stripping.** oMLX returns HTTP 500 when text
   contains unpaired high/low surrogates. In Python 3 these are rare
   (strings are native Unicode) but can sneak in via ill-formed JSON
   or base64-decoded content. We strip them pre-send as a safety net.
3. **30s request timeout.** oMLX is fast; anything over 30s is a hang.
4. **Embedding dimension is discovered on first call** and cached. Used
   by the palace meta sidecar to detect dim mismatches.

The class is deliberately sync, not async. MemPalace's existing code is
all synchronous; adding an async client would force threading gymnastics
at every call site. Performance is fine: oMLX responds in <100ms for
typical embed calls, and batching keeps throughput high.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .errors import ProviderError, ProviderUnavailableError

logger = logging.getLogger("mempalace.providers.openai_compat")

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_EMBED_BATCH_SIZE = 48
_DEFAULT_TIMEOUT_SECONDS = 30.0


# ── helpers ────────────────────────────────────────────────────────────────

# Match a high surrogate (U+D800..U+DBFF) not followed by a low surrogate,
# and a low surrogate not preceded by a high surrogate. We scan character by
# character in Python because Python's ``re`` module doesn't support surrogate
# lookarounds the way JavaScript regex does.
_HIGH_SURR_ONLY = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])")
_LOW_SURR_ORPHAN = re.compile(r"(?<![\ud800-\udbff])[\udc00-\udfff]")


def _strip_lone_surrogates(text: str) -> str:
    """Remove unpaired high/low UTF-16 surrogate code units.

    Python 3 strings are arbitrary Unicode, so invalid surrogates are rare
    but can appear when decoding mangled input. This defensive strip prevents
    oMLX from 500ing on broken text.
    """
    if not text:
        return text
    result = _HIGH_SURR_ONLY.sub("", text)
    result = _LOW_SURR_ORPHAN.sub("", result)
    return result


def _normalize_base_url(url: str) -> str:
    """Ensure the URL ends in ``/v1`` (without trailing slash)."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


# ── client ────────────────────────────────────────────────────────────────

class OpenAICompatibleClient:
    """OpenAI-compatible HTTP client for embed + generate + rerank.

    One instance can be configured for one or more of the three roles
    (embed_model, generate_model, rerank_model). It's fine to leave any
    role unset — trying to call a role that isn't configured raises
    ``ProviderError``.

    Parameters
    ----------
    base_url : str
        Server root. ``/v1`` is appended automatically if missing.
    api_key : str, optional
        If set, sent as ``Authorization: Bearer {api_key}``. Required for
        cloud OpenAI; typically unused for local servers.
    embed_model : str, optional
        Model name used for ``embed`` / ``embed_batch``.
    generate_model : str, optional
        Model name used for ``generate``.
    rerank_model : str, optional
        Model name used for ``rerank``.
    embed_batch_size : int
        Max batch size for embedding requests. Defaults to 48 to dodge the
        oMLX batch-bug at 64.
    timeout_seconds : float
        HTTP request timeout. Defaults to 30s.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        embed_model: Optional[str] = None,
        generate_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        embed_batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ):
        self._base_url = _normalize_base_url(base_url)
        self._api_key = api_key
        self._embed_model = embed_model
        self._generate_model = generate_model
        self._rerank_model = rerank_model
        self._embed_batch_size = max(1, int(embed_batch_size))
        self._timeout_seconds = float(timeout_seconds)
        self._embed_dim: Optional[int] = None

        # Lazy import so requests is only touched when the class is used.
        import requests  # noqa: F401

    # ── identity (used by ChromaProviderAdapter, meta sidecar) ────────────

    @property
    def model_name(self) -> str:
        """Return the embed model name (primary identity for sidecar meta)."""
        return self._embed_model or self._generate_model or self._rerank_model or "openai_compatible:unknown"

    @property
    def embed_dim(self) -> Optional[int]:
        return self._embed_dim

    @property
    def provider_kind(self) -> str:
        return "openai_compatible"

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── core HTTP ────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        url = f"{self._base_url}{path}"
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                data=json.dumps(body).encode("utf-8"),
                timeout=self._timeout_seconds,
            )
        except requests.exceptions.RequestException as e:
            raise ProviderUnavailableError(f"POST {url} failed: {e}") from e

        if resp.status_code >= 500:
            raise ProviderUnavailableError(
                f"POST {url} returned {resp.status_code}: {resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"POST {url} returned {resp.status_code}: {resp.text[:500]}"
            )

        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise ProviderError(f"POST {url} returned non-JSON: {resp.text[:500]}") from e

    def _get(self, path: str) -> Optional[Dict[str, Any]]:
        import requests

        url = f"{self._base_url}{path}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self._timeout_seconds)
        except requests.exceptions.RequestException as e:
            logger.debug("GET %s failed: %s", url, e)
            return None
        if resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            return None

    # ── embeddings ───────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns the vector as a list of floats."""
        if not self._embed_model:
            raise ProviderError("embed_model is not configured on this client")

        cleaned = _strip_lone_surrogates(text)
        payload = self._post(
            "/embeddings",
            {"model": self._embed_model, "input": cleaned},
        )

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ProviderError(f"embedding response has no 'data' array: {payload}")

        vec = data[0].get("embedding")
        if not isinstance(vec, list):
            raise ProviderError(f"embedding response item has no 'embedding' list: {data[0]}")

        if self._embed_dim is None:
            self._embed_dim = len(vec)
        return [float(x) for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings.

        Splits into sub-batches of ``embed_batch_size`` to avoid oMLX's bug
        where batches ≥64 can return ``None`` for the first item. If any
        sub-batch still returns a null embedding, we retry that item
        individually via ``embed``.
        """
        if not self._embed_model:
            raise ProviderError("embed_model is not configured on this client")
        if not texts:
            return []

        results: List[List[float]] = [None] * len(texts)  # type: ignore
        batch_size = self._embed_batch_size

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            cleaned = [_strip_lone_surrogates(t) for t in chunk]

            try:
                payload = self._post(
                    "/embeddings",
                    {"model": self._embed_model, "input": cleaned},
                )
            except ProviderUnavailableError:
                raise
            except ProviderError as e:
                logger.error("embed_batch sub-batch at offset %d failed: %s", start, e)
                raise

            data = payload.get("data")
            if not isinstance(data, list):
                raise ProviderError(f"embedding response has no 'data' array at offset {start}")

            for item in data:
                idx_in_chunk = item.get("index", 0)
                abs_idx = start + idx_in_chunk
                vec = item.get("embedding")
                if isinstance(vec, list):
                    results[abs_idx] = [float(x) for x in vec]
                else:
                    # oMLX batch-bug: null embedding at the first item of a
                    # large batch. Retry this one individually.
                    logger.warning(
                        "embed_batch: null embedding at index %d, retrying individually",
                        abs_idx,
                    )
                    results[abs_idx] = self.embed(texts[abs_idx])

        # Fill any still-missing entries (shouldn't happen, but defensive)
        for i, r in enumerate(results):
            if r is None:
                results[i] = self.embed(texts[i])

        if self._embed_dim is None:
            for r in results:
                if r:
                    self._embed_dim = len(r)
                    break

        return results  # type: ignore

    # ── generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.8,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        thinking: bool = False,
    ) -> str:
        """Generate text via ``/v1/chat/completions``.

        Parameters
        ----------
        prompt : str
            The user message.
        max_tokens : int
            Response length cap.
        temperature : float
            Lower is more deterministic. 0.2 is a good default for
            structured outputs like Wenjian/AAAK compression.
        top_p : float
            Nucleus sampling.
        stop : list of str, optional
            Stop sequences.
        system : str, optional
            System message to prepend.
        thinking : bool
            If False (default), and the prompt targets a Qwen3 model that
            supports thinking mode, append ``/no_think`` to enable
            non-thinking (fast) mode. Setting this True leaves the prompt
            alone so the model decides.
        """
        if not self._generate_model:
            raise ProviderError("generate_model is not configured on this client")

        user_content = _strip_lone_surrogates(prompt)
        if not thinking and "qwen" in self._generate_model.lower():
            # Qwen3 non-thinking convention. Harmless for other models that
            # don't look for the marker.
            if not user_content.rstrip().endswith("/no_think"):
                user_content = user_content.rstrip() + " /no_think"

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": _strip_lone_surrogates(system)})
        messages.append({"role": "user", "content": user_content})

        body: Dict[str, Any] = {
            "model": self._generate_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            body["stop"] = list(stop)

        payload = self._post("/chat/completions", body)

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(f"chat response has no 'choices': {payload}")

        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise ProviderError(f"chat response choice has no 'content': {choices[0]}")

        return content

    # ── reranking ────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[str],
        *,
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank documents against a query.

        Returns a list of ``(original_index, relevance_score)`` tuples
        sorted descending by score. Length is ``min(top_n or len(documents),
        len(documents))``.
        """
        if not self._rerank_model:
            raise ProviderError("rerank_model is not configured on this client")
        if not documents:
            return []

        cleaned_docs = [_strip_lone_surrogates(d) for d in documents]
        cleaned_query = _strip_lone_surrogates(query)
        n = int(top_n) if top_n is not None else len(cleaned_docs)

        payload = self._post(
            "/rerank",
            {
                "model": self._rerank_model,
                "query": cleaned_query,
                "documents": cleaned_docs,
                "top_n": n,
            },
        )

        results = payload.get("results")
        if not isinstance(results, list):
            raise ProviderError(f"rerank response has no 'results' array: {payload}")

        pairs: List[Tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(idx, int) and isinstance(score, (int, float)):
                pairs.append((idx, float(score)))

        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:n]

    # ── discovery ────────────────────────────────────────────────────────

    def list_models(self) -> List[str]:
        """GET ``/v1/models`` and return loaded model IDs."""
        payload = self._get("/models")
        if not payload:
            return []
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        return [entry.get("id") for entry in data if isinstance(entry, dict) and "id" in entry]

    def model_exists(self, model: str) -> bool:
        return model in set(self.list_models())
