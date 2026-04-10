"""
probe.py — detect reachable OpenAI-compatible endpoints.

Used at init time and at ``resolve_*`` auto-mode to decide whether an oMLX
server (or any other OpenAI-compatible server) is available.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("mempalace.providers.probe")

_DEFAULT_OMLX_URL = "http://127.0.0.1:8000/v1"
_PROBE_TIMEOUT_SECONDS = 2.0


def probe_omlx(base_url: str = _DEFAULT_OMLX_URL) -> Optional[List[str]]:
    """GET ``{base_url}/models`` and return the list of model IDs.

    Returns ``None`` on any failure (network error, non-200, malformed JSON,
    timeout). Never raises — this is called from the init flow and must not
    crash mempalace when oMLX is simply not running.

    The 2-second timeout is deliberate: if oMLX is up, it answers in <100ms;
    if it's not up, any longer wait is just a slow failure.

    Accepts either ``http://host:port`` or ``http://host:port/v1`` — we
    normalize by stripping a trailing ``/v1``.
    """
    import requests  # lazy — avoids circular imports and optional-dep issues

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    url = f"{normalized}/v1/models"

    try:
        resp = requests.get(url, timeout=_PROBE_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.debug("probe_omlx failed at %s: %s", url, e)
        return None

    data = payload.get("data")
    if not isinstance(data, list):
        return None

    models: List[str] = []
    for entry in data:
        if isinstance(entry, dict):
            model_id = entry.get("id")
            if isinstance(model_id, str):
                models.append(model_id)
    return models if models else None
