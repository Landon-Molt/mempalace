"""
language_detect.py — minimal language detection for Wenjian compression.

Returns a set of detected language codes. Used to pick the compression
strategy: pure ZH → full Classical Chinese, mixed EN+ZH → hybrid, DE
or other → rule-based only (no Wenjian idioms forced on non-CJK text).

NOT a general-purpose language detector — just looks for Unicode range
presence to decide whether CJK compression is appropriate.
"""

from __future__ import annotations

import re
from typing import Set

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-zA-Z]{3,}")
_GERMAN = re.compile(r"[äöüßÄÖÜ]")


def detect_languages(text: str) -> Set[str]:
    """Detect which languages are present in the text.

    Returns a set of 2-letter codes: ``{"zh"}``, ``{"en"}``, ``{"de"}``,
    or combinations like ``{"zh", "en"}``.

    Empty input returns ``{"en"}`` as a safe default.
    """
    if not text or not text.strip():
        return {"en"}

    langs: Set[str] = set()

    if _CJK.search(text):
        langs.add("zh")
    if _LATIN.search(text):
        langs.add("en")
    if _GERMAN.search(text):
        langs.add("de")

    return langs if langs else {"en"}


def is_cjk_dominant(text: str, threshold: float = 0.3) -> bool:
    """Return True if CJK characters make up ≥ threshold of the text.

    Used to decide whether to apply full Wenjian compression (which is
    most effective on Chinese-heavy text) vs the rule-based fallback.
    """
    if not text:
        return False
    cjk_count = len(_CJK.findall(text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return False
    return cjk_count / total >= threshold
