"""
summarizer.py — orchestrator for LLM-assisted compression.

Sits between the CLI (``cmd_compress``) and the format plugins (AAAK,
Wenjian). Decides how to compress based on config:

- If an LLM is configured, generates a prompt via the format's
  ``prompt_template()``, sends it to the LLM, parses the response as a
  ``CompressedEntry``, validates entity preservation, and retries once
  if the preservation check fails.
- If no LLM is configured (or ``rule_only=True``), falls through to the
  format's deterministic ``compress()`` method.
- Always runs ``validate()`` and records the preservation ratio in the
  entry's metadata.

Usage::

    from mempalace.summarizer import Summarizer
    from mempalace.formats import resolve_format
    from mempalace.providers import resolve_llm

    fmt = resolve_format("wenjian")
    llm = resolve_llm(config)  # or None for rule-only
    s = Summarizer(fmt, llm=llm)

    entry = s.compress("The team decided to migrate Auth0 to Clerk...")
    print(entry.text)           # 议 26/Q1 金蝉迁身份…★★★★
    print(entry.compression_ratio)  # 4.2
    print(entry.entities_preserved_ratio)  # 0.85
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import re

from .formats.base import CompressedEntry, ValidationReport

logger = logging.getLogger("mempalace.summarizer")

# ── response parsing ──────────────────────────────────────────────────────

# Wenjian type markers that indicate the start of actual compressed output.
_WENJIAN_TYPE_MARKERS = {"议", "事", "得", "好", "策"}

# AAAK output typically starts with a digit (the ZID line) or with wing|room.
_AAAK_START = re.compile(r"^\d+:|^[a-z_]+\|", re.MULTILINE)


def _strip_thinking(response: str, format_name: str = "wenjian") -> str:
    """Extract the actual compressed output from an LLM response that may
    contain chain-of-thought reasoning.

    Many models (especially Qwen3 family) emit thinking traces before the
    answer, even when asked not to. This parser strips common CoT patterns:

    1. ``<think>...</think>`` XML tags (Qwen3 format)
    2. Everything before the first Wenjian type marker (议/事/得/好/策)
    3. Everything before the first AAAK-style line (0:... or wing|room|...)
    4. Lines starting with "Thinking" / "Step" / numbered reasoning

    If no pattern matches, returns the raw response trimmed.
    """
    text = response.strip()
    if not text:
        return text

    # 1. Strip <think>...</think> tags (Qwen3 explicit thinking blocks)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1b. For models that emit "Thinking Process:" or similar headers:
    # don't strip the whole block (the answer may be at the end). Instead,
    # fall through to step 2/3 which will find the actual output by its
    # format markers. We only strip if the thinking block is clearly
    # separated by a blank line and followed by the actual output.
    # (Step 2 handles finding the output.)

    # 2. For Wenjian: find the first line starting with a type marker.
    if format_name in ("wenjian", "文简", "wj"):
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and stripped[0] in _WENJIAN_TYPE_MARKERS:
                # Take from this line to the end (might be multi-line entry)
                idx = text.index(stripped)
                return text[idx:].strip()

    # 3. For AAAK: find the first AAAK-style line.
    if format_name in ("aaak", "dialect"):
        match = _AAAK_START.search(text)
        if match:
            return text[match.start():].strip()

    # 4. Generic: if the response has a "---" separator, take the last section.
    if "\n---\n" in text:
        parts = text.split("\n---\n")
        return parts[-1].strip()

    # 5. Last resort: take the last non-empty paragraph.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        # The last paragraph is most likely the actual output.
        candidate = paragraphs[-1]
        # But if it's still reasoning text, take the whole thing.
        if len(candidate) < len(text) * 0.5:
            return candidate

    return text


class Summarizer:
    """Orchestrates compression using a Format plugin and optional LLM.

    Parameters
    ----------
    fmt : Format
        A format plugin (AAAK or Wenjian) from ``mempalace.formats``.
    llm : OpenAICompatibleClient, optional
        An LLM client for assisted compression. If None, uses the
        format's rule-based ``compress()`` method.
    rule_only : bool
        If True, never use the LLM even if one is configured.
    max_retries : int
        Number of times to retry LLM compression if entity preservation
        fails. Default 1 (= try once, retry once with preservation
        directive).
    preservation_threshold : float
        Minimum entity preservation ratio to accept an LLM compression.
        Default 0.7 (70%).
    """

    def __init__(
        self,
        fmt,
        llm=None,
        rule_only: bool = False,
        max_retries: int = 1,
        preservation_threshold: float = 0.7,
    ):
        self._fmt = fmt
        self._llm = llm
        self._rule_only = rule_only
        self._max_retries = max(0, int(max_retries))
        self._threshold = float(preservation_threshold)

    @property
    def format_name(self) -> str:
        return self._fmt.name

    @property
    def has_llm(self) -> bool:
        return self._llm is not None and not self._rule_only

    def compress(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompressedEntry:
        """Compress text, choosing the best available path.

        1. If LLM is available and not rule_only → LLM-assisted compression
           with entity validation and optional retry.
        2. Otherwise → format's deterministic rule-based compression.

        Either way, validation runs and the preservation ratio is recorded
        on the returned ``CompressedEntry``.
        """
        if self.has_llm:
            entry = self._llm_compress(
                text,
                memory_type=memory_type,
                importance=importance,
                context=context,
            )
        else:
            entry = self._fmt.compress(
                text,
                memory_type=memory_type,
                importance=importance,
                context=context,
            )

        # Validate entity preservation regardless of path.
        report = self._fmt.validate(text, entry, threshold=self._threshold)
        entry.entities_preserved_ratio = report.preservation_ratio
        entry.missing_entities = sorted(report.missing_entities)

        return entry

    def _llm_compress(
        self,
        text: str,
        *,
        memory_type: str,
        importance: int,
        context: Optional[Dict[str, Any]],
    ) -> CompressedEntry:
        """LLM-assisted compression with retry on preservation failure."""
        from .formats.wenjian_tokenizer import count_tokens

        missing_entities: Optional[List[str]] = None

        for attempt in range(1 + self._max_retries):
            try:
                # Build prompt (with missing_entities on retry).
                prompt = self._fmt.prompt_template(
                    text,
                    memory_type=memory_type,
                    importance=importance,
                    context=context,
                    **({"missing_entities": missing_entities} if missing_entities else {}),
                )

                # Send to LLM. Use a generous token limit because some
                # models emit chain-of-thought before the answer.
                llm_response = self._llm.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.2,
                    system=(
                        "You are a compression engine. Output ONLY the "
                        "compressed text. No reasoning, no explanation, "
                        "no thinking process. Just the result."
                    ),
                )

                if not llm_response or not llm_response.strip():
                    logger.warning(
                        "LLM returned empty response on attempt %d; "
                        "falling through to rule-based",
                        attempt + 1,
                    )
                    break

                # Strip chain-of-thought reasoning that many models emit
                # despite being asked not to think. Extract just the
                # compressed output.
                compressed_text = _strip_thinking(
                    llm_response, format_name=self._fmt.name
                )

                # Build entry.
                original_tokens = count_tokens(text, self._llm.model_name if self._llm else "")
                compressed_tokens = count_tokens(
                    compressed_text, self._llm.model_name if self._llm else ""
                )

                entry = CompressedEntry(
                    text=compressed_text,
                    format_name=self._fmt.name,
                    memory_type=memory_type or "",
                    importance=importance,
                    original_token_count=original_tokens,
                    compressed_token_count=compressed_tokens,
                    llm_model=getattr(self._llm, "model_name", ""),
                )

                # Validate.
                report = self._fmt.validate(text, entry, threshold=self._threshold)
                if report.is_acceptable:
                    entry.entities_preserved_ratio = report.preservation_ratio
                    entry.missing_entities = sorted(report.missing_entities)
                    return entry

                # Preservation too low — retry with missing entities.
                missing_entities = sorted(report.missing_entities)
                logger.info(
                    "Attempt %d: preservation %.0f%% (below %.0f%%), "
                    "missing: %s — retrying with directive",
                    attempt + 1,
                    report.preservation_ratio * 100,
                    self._threshold * 100,
                    missing_entities[:5],
                )

            except Exception as e:
                logger.warning(
                    "LLM compression failed on attempt %d (%s); "
                    "falling through to rule-based",
                    attempt + 1,
                    e,
                )
                break

        # All LLM attempts failed or preservation was never acceptable.
        # Fall back to rule-based.
        logger.info("Falling back to rule-based %s compression", self._fmt.name)
        return self._fmt.compress(
            text,
            memory_type=memory_type,
            importance=importance,
            context=context,
        )
