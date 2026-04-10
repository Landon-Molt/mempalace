"""
aaak.py — Adapter wrapping MemPalace's existing Dialect class as a Format plugin.

No changes to ``mempalace/dialect.py`` itself — this module simply adapts
the existing ``Dialect`` API to the ``Format`` protocol so AAAK and Wenjian
can be used interchangeably by the Summarizer.

AAAK is 100% rule-based: regex, TF scoring, keyword matching. No LLM
needed. The ``prompt_template()`` method still returns a useful prompt
that an LLM could use to refine the output, but the default ``compress()``
path never calls an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import CompressedEntry, Format, ValidationReport, validate_preservation


class AAKFormat:
    """AAAK compression format — adapter for ``mempalace.dialect.Dialect``.

    Implements the ``Format`` protocol.
    """

    name: str = "aaak"

    def __init__(
        self,
        entities: Optional[Dict[str, str]] = None,
        skip_names: Optional[list] = None,
        config_path: Optional[str] = None,
        **_kwargs,
    ):
        from mempalace.dialect import Dialect

        if config_path:
            self._dialect = Dialect.from_config(config_path)
        else:
            self._dialect = Dialect(
                entities=entities or {},
                skip_names=skip_names or [],
            )

    def compress(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompressedEntry:
        """Compress text using the AAAK rule-based dialect.

        Delegates to ``Dialect.compress(text, metadata=context)``.
        """
        metadata = dict(context or {})
        compressed = self._dialect.compress(text, metadata=metadata)
        stats = self._dialect.compression_stats(text, compressed)

        return CompressedEntry(
            text=compressed,
            format_name=self.name,
            memory_type=memory_type,
            importance=importance,
            original_token_count=stats.get("original_tokens_est", 0),
            compressed_token_count=stats.get("summary_tokens_est", 0),
            entities_preserved_ratio=1.0,  # set by validate() below
            metadata=metadata,
        )

    def validate(
        self,
        original: str,
        compressed: CompressedEntry,
        threshold: float = 0.7,
    ) -> ValidationReport:
        """Check entity preservation between original and AAAK output."""
        return validate_preservation(original, compressed.text, threshold=threshold)

    def prompt_template(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 2,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a prompt an LLM could use to produce AAAK output.

        While AAAK is normally rule-based, an LLM can sometimes produce
        better entity detection and topic extraction. This prompt describes
        the AAAK format and asks the LLM to compress.
        """
        return (
            "You are an AAAK compression engine for the MemPalace memory system.\n"
            "AAAK format:\n"
            "- Header: wing|room|date|source\n"
            "- Body: ZID line with entities, topics, key quote, emotions, flags\n"
            "- Entities as 3-letter uppercase codes (e.g., ALC=Alice)\n"
            "- Emotions as *markers* (e.g., *warm*, *fierce*)\n"
            "- Pipe-separated fields\n\n"
            f"Compress the following text into AAAK format:\n\n{text}\n\n"
            "Output ONLY the AAAK-formatted result, no explanation."
        )

    def decompress_hint(self, entry: CompressedEntry) -> str:
        """Return guidance for expanding AAAK back to natural language."""
        return (
            "This is an AAAK-compressed memory. To read it:\n"
            "- 3-letter codes are entity names (expand them)\n"
            "- *markers* are emotional context\n"
            "- Pipe-separated fields: entities|topics|quote|emotions|flags\n"
            "- Expand naturally, inferring full sentences from the compressed form.\n\n"
            f"AAAK text:\n{entry.text}"
        )
