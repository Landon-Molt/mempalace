"""
mempalace.formats — pluggable compression format system.

Provides two built-in formats:

- **AAAK** — MemPalace's original rule-based dialect. Entity codes,
  structured markers, sentence truncation. 100% deterministic, no LLM
  needed. Wrapped from ``mempalace.dialect.Dialect``.
- **Wenjian** (文简) — Classical Chinese shorthand for AI memory. Uses
  the natural information density of Chinese characters plus an expanded
  idiom dictionary. Has both a rule-based fallback and an LLM-assisted
  path for higher quality. Ported and improved from MemChinesePalace.

The ``resolve_format(name)`` factory returns a Format instance by name.
The ``Summarizer`` (in ``mempalace.summarizer``) uses it to compress
drawers into closet entries.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    CompressedEntry,
    Format,
    ValidationReport,
    extract_entities,
    validate_preservation,
)

__all__ = [
    "CompressedEntry",
    "Format",
    "ValidationReport",
    "extract_entities",
    "validate_preservation",
    "resolve_format",
]


def resolve_format(name: str, **kwargs) -> Format:
    """Instantiate a compression format by name.

    Parameters
    ----------
    name : str
        ``"aaak"`` or ``"wenjian"``.
    **kwargs
        Passed to the format constructor. Wenjian accepts ``llm``,
        ``tokenizer``, ``idiom_map``, ``language_detector``. AAAK
        accepts ``entities``, ``skip_names``.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    normalized = name.lower().strip()

    if normalized in ("aaak", "dialect"):
        from .aaak import AAKFormat
        return AAKFormat(**kwargs)

    if normalized in ("wenjian", "文简", "wj"):
        from .wenjian import WenjianFormat
        return WenjianFormat(**kwargs)

    raise ValueError(
        f"Unknown compression format: {name!r}. "
        f"Available: 'aaak', 'wenjian'"
    )
