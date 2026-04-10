"""
base.py — Format protocol and shared data types for the compression system.

Every compression format (AAAK, Wenjian, future additions) implements the
``Format`` protocol defined here. The protocol is deliberately minimal:

- ``compress()`` takes raw text and returns a ``CompressedEntry``
- ``validate()`` checks that a compressed form preserves key entities
- ``prompt_template()`` returns the LLM prompt for assisted compression
- ``decompress_hint()`` returns guidance text for expanding a compressed entry

``CompressedEntry`` carries the compressed text plus metadata about the
compression (token counts, preservation ratio, format name). It's what
gets stored in the ``mempalace_compressed`` ChromaDB collection.

``ValidationReport`` is the result of ``validate()`` — a simple struct
with a list of missing entities and a boolean verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class CompressedEntry:
    """A single compressed memory entry, ready to store in the closet collection."""

    text: str
    format_name: str
    memory_type: str = ""
    importance: int = 2  # 1-5 scale (★ to ★★★★★)
    status: str = ""  # [定]/[疑]/[废]/[进]/[毕]/[阻] or empty
    original_token_count: int = 0
    compressed_token_count: int = 0
    entities_preserved_ratio: float = 1.0
    missing_entities: List[str] = field(default_factory=list)
    llm_model: str = ""  # which model produced this (empty if rule-based)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        if self.compressed_token_count == 0:
            return 0.0
        return self.original_token_count / self.compressed_token_count


@dataclass
class ValidationReport:
    """Result of checking whether a compressed form preserves key entities."""

    original_entities: Set[str]
    preserved_entities: Set[str]
    missing_entities: Set[str]
    preservation_ratio: float
    is_acceptable: bool  # True if ratio >= threshold (default 0.7)

    @property
    def summary(self) -> str:
        if self.is_acceptable:
            return f"OK ({self.preservation_ratio:.0%} preserved)"
        missing = ", ".join(sorted(self.missing_entities)[:5])
        return f"LOW ({self.preservation_ratio:.0%}) — missing: {missing}"


# ── Entity extraction (shared across formats) ──────────────────────────


# Patterns that match things we want to see preserved in compressed text.
_COMMON_WORDS = frozenset({
    "the", "and", "but", "for", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "are", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "this", "that", "with", "have", "from", "each",
    "which", "their", "will", "other", "about", "many", "then", "them",
    "these", "some", "would", "make", "like", "been", "most", "only",
    "over", "such", "after", "also", "back", "could", "into", "just",
    "because", "based", "when", "where", "what", "here", "every",
    "first", "since", "between", "before", "under", "should", "while",
    "still", "being", "through", "during", "both", "those", "same",
    "another", "much", "than", "very", "well", "long", "last", "even",
    "down", "right", "real", "best", "true", "false", "none", "each",
    "they", "year", "years", "day", "days", "time", "work", "data",
    "case", "test", "file", "used", "need", "using", "does", "done",
    "given", "made", "take", "taken", "keep", "kept", "find", "found",
    "show", "shown", "turn", "runs", "start", "end",
    "decided", "completed", "deployed", "discovered", "recommend",
    "successfully", "previously", "currently", "recently", "finally",
    "target", "handled", "updated", "removed", "changed", "added",
    "die", "das", "der", "den", "dem", "des", "ein", "eine", "einer",
})

_ENTITY_PATTERNS = [
    # CamelCase / mixed-case tech terms (GraphQL, PostgreSQL, ChromaDB, OAuth)
    # Matches words that start with uppercase, have lowercase, then more uppercase.
    re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]*\b"),
    # Mixed-case words containing digits (Auth0, OAuth2, GPT4, S3)
    re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b"),
    # ALL_CAPS identifiers (API, JWT, SQL, CI)
    re.compile(r"\b[A-Z]{2,}[0-9]*\b"),
    # Proper nouns: initial cap, 3+ chars, not a common English word
    re.compile(r"\b[A-Z][a-z]{2,}\b"),
    # Dates: YYYY-MM-DD, DD.MM.YYYY, YY/MM/DD, Q1/Q2/Q3/Q4
    re.compile(r"\b\d{2,4}[-/.]\d{1,2}[-/.]\d{2,4}\b"),
    re.compile(r"\bQ[1-4]\b"),
    # Prices / amounts: $240, €50, 25/mo
    re.compile(r"\$\d+(?:\.\d+)?"),
    re.compile(r"€\d+(?:\.\d+)?"),
    re.compile(r"\b\d+/(?:mo|yr|year|month|day|hr)\b"),
    # Version numbers: v3.2, v2.0.1
    re.compile(r"\bv\d+(?:\.\d+)+\b"),
    # URLs (simplified — just grab the domain)
    re.compile(r"https?://[^\s,)\"']+"),
    # Chinese names (2-3 chars followed by role suffix or punctuation)
    re.compile(r"(?<=[\s,，。；])([\u4e00-\u9fff]{2,3})(?=[\s,，。；工统设运品])"),
    # Standalone CJK sequences (names, terms) 2-4 chars
    re.compile(r"[\u4e00-\u9fff]{2,4}"),
    # German umlauts in proper nouns (München, Straße)
    re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]+(?:ung|keit|heit|schaft)\b"),
]


def extract_entities(text: str) -> Set[str]:
    """Extract names, dates, prices, version numbers, URLs, and other
    entities from text that should be preserved across compression.

    Returns a set of strings found in the original text. The compressed
    form is validated by checking how many of these appear in it.

    This is deliberately over-inclusive: a false positive (extracting a
    non-entity) just means the preservation check is a bit stricter, which
    is safer than missing a real entity.
    """
    entities: Set[str] = set()
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(0).strip()
            if len(entity) < 2:
                continue
            # Filter out common English words captured by the proper-noun
            # pattern. A word like "The" or "Monday" is not an entity.
            if entity.lower() in _COMMON_WORDS:
                continue
            entities.add(entity)
    return entities


def validate_preservation(
    original: str,
    compressed_text: str,
    threshold: float = 0.7,
) -> ValidationReport:
    """Check whether key entities from the original appear in the
    compressed text.

    Returns a ``ValidationReport`` with the preservation ratio and the
    list of missing entities. The default threshold is 0.7 (70%) — below
    that, the compression is flagged as unacceptable and the caller
    (summarizer) may retry with explicit entity-preservation guidance.

    The check is case-insensitive for English entities (Auth0 matches
    auth0) but case-sensitive for CJK (美云 ≠ 美雲).
    """
    original_entities = extract_entities(original)
    if not original_entities:
        return ValidationReport(
            original_entities=set(),
            preserved_entities=set(),
            missing_entities=set(),
            preservation_ratio=1.0,
            is_acceptable=True,
        )

    compressed_lower = compressed_text.lower()
    preserved: Set[str] = set()
    missing: Set[str] = set()

    for entity in original_entities:
        # CJK characters: exact match (case doesn't apply).
        if any("\u4e00" <= c <= "\u9fff" for c in entity):
            if entity in compressed_text:
                preserved.add(entity)
            else:
                missing.add(entity)
        else:
            # Latin / numeric: case-insensitive match.
            if entity.lower() in compressed_lower:
                preserved.add(entity)
            else:
                missing.add(entity)

    ratio = len(preserved) / len(original_entities)
    return ValidationReport(
        original_entities=original_entities,
        preserved_entities=preserved,
        missing_entities=missing,
        preservation_ratio=ratio,
        is_acceptable=ratio >= threshold,
    )


# ── Format protocol ──────────────────────────────────────────────────────


@runtime_checkable
class Format(Protocol):
    """Interface that every compression format implements.

    The ``Summarizer`` orchestrator calls these methods to produce, validate,
    and describe compressed entries. A format can be purely rule-based
    (like AAAK) or LLM-assisted (like Wenjian).
    """

    name: str

    def compress(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompressedEntry:
        """Compress text using the format's rule-based path.

        This is the deterministic, offline, no-LLM-needed path. It must
        work even when no LLM is configured. The Summarizer may replace
        this with an LLM-assisted version if configured to do so.
        """
        ...

    def validate(
        self,
        original: str,
        compressed: CompressedEntry,
        threshold: float = 0.7,
    ) -> ValidationReport:
        """Check entity preservation between original and compressed form."""
        ...

    def prompt_template(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the LLM prompt for assisted compression.

        The Summarizer sends this to the configured LLM and parses the
        response as a CompressedEntry. The format controls the prompt
        wording, few-shot examples, and output rules.
        """
        ...

    def decompress_hint(self, entry: CompressedEntry) -> str:
        """Return guidance for expanding a compressed entry back to
        natural language.

        Not a true inverse — compression is lossy. This is a hint that
        an LLM can use to reconstruct the gist.
        """
        ...
