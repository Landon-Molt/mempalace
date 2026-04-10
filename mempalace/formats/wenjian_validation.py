"""
wenjian_validation.py — entity preservation checker for Wenjian compression.

Wraps the generic ``base.validate_preservation`` with Wenjian-specific
logic: detects when the compressed form uses Chinese name abbreviations
(e.g., 伟明 for Kai) and counts those as preserved rather than missing.

Also provides ``suggest_preservation_directive`` which generates a
one-line instruction for the LLM retry prompt telling it exactly which
entities to preserve.
"""

from __future__ import annotations

from typing import List, Set

from .base import ValidationReport, extract_entities, validate_preservation


def validate_wenjian(
    original: str,
    compressed_text: str,
    threshold: float = 0.7,
    person_map: dict = None,
) -> ValidationReport:
    """Validate entity preservation with Wenjian-specific awareness.

    Parameters
    ----------
    original : str
        The original text before compression.
    compressed_text : str
        The Wenjian-compressed output.
    threshold : float
        Minimum preservation ratio (0.7 = 70%).
    person_map : dict, optional
        Mapping of English/full names to Chinese abbreviations:
        ``{"Kai": "伟明", "Maya": "美云"}``. If an entity in the
        original maps to a Chinese abbreviation that IS present in the
        compressed text, it counts as preserved rather than missing.
    """
    base_report = validate_preservation(original, compressed_text, threshold=threshold)

    if not person_map or not base_report.missing_entities:
        return base_report

    # Re-check missing entities through the person map.
    still_missing: Set[str] = set()
    extra_preserved: Set[str] = set()

    for entity in base_report.missing_entities:
        # Check if the entity maps to a Chinese abbreviation that's present.
        mapped = person_map.get(entity)
        if mapped and mapped in compressed_text:
            extra_preserved.add(entity)
        else:
            still_missing.add(entity)

    if not extra_preserved:
        return base_report

    # Recalculate with the additional preservations.
    all_preserved = base_report.preserved_entities | extra_preserved
    total = len(base_report.original_entities)
    ratio = len(all_preserved) / total if total > 0 else 1.0

    return ValidationReport(
        original_entities=base_report.original_entities,
        preserved_entities=all_preserved,
        missing_entities=still_missing,
        preservation_ratio=ratio,
        is_acceptable=ratio >= threshold,
    )


def suggest_preservation_directive(missing: Set[str]) -> str:
    """Generate a one-line instruction for the LLM retry prompt.

    Used when the first compression attempt fails the preservation check.
    The directive lists the specific entities that must appear in the
    compressed output.
    """
    if not missing:
        return ""
    entities_str = ", ".join(sorted(missing)[:10])
    return f"⚠ 务必保留以下实体（原样或中文等价）：{entities_str}"
