"""
wenjian.py — Classical Chinese compression format for MemPalace.

Ported from MemChinesePalace's ``compressor.py`` with these improvements:

1. **Idiom dictionary is 7× larger** (35 vs 5), domain-tagged
2. **Few-shot LLM prompt** with one worked example per MemoryType
3. **Entity preservation validation** with re-prompt on failure
4. **Model-aware tokenizer** (not hardcoded to cl100k_base)
5. **Language-aware strategy** — pure ZH, hybrid EN+ZH, DE rule-only
6. **Grammar-aware rule fallback** with phrase-level replacements
7. **Backward-compatible** with MCC's original WenjianEntry format

The class implements the ``Format`` protocol from ``formats.base``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set

from .base import CompressedEntry, ValidationReport, validate_preservation
from .language_detect import detect_languages, is_cjk_dominant
from .wenjian_idioms import ALL_IDIOMS, TRIGGER_TO_IDIOM, format_idiom_table_for_prompt
from .wenjian_prompts import WENJIAN_SPEC_SHORT, build_compress_prompt, build_expand_prompt
from .wenjian_tokenizer import count_tokens, get_tokenizer
from .wenjian_validation import validate_wenjian


# ── Memory type detection (heuristic) ─────────────────────────────────────

_TYPE_KEYWORDS = {
    "议": ["decided", "decision", "chose", "picked", "selected", "agreed", "resolved",
           "决定", "选择", "议定", "敲定", "定了"],
    "事": ["deployed", "shipped", "released", "milestone", "launched", "completed",
           "发布", "上线", "完成", "里程碑", "交付"],
    "得": ["discovered", "found", "realized", "learned", "insight", "noticed",
           "发现", "意识到", "洞见", "注意到"],
    "好": ["prefers", "always", "likes", "habit", "convention", "standard",
           "偏好", "习惯", "惯用", "总是", "喜欢"],
    "策": ["recommend", "suggest", "advise", "should", "consider", "propose",
           "建议", "推荐", "应该", "可以考虑"],
}


def _detect_memory_type(text: str) -> str:
    """Heuristic detection of memory type from text content."""
    text_lower = text.lower()
    scores = {}
    for mt, keywords in _TYPE_KEYWORDS.items():
        scores[mt] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "议"


# ── Importance detection (heuristic) ─────────────────────────────────────

_IMPORTANCE_SIGNALS = {
    5: ["critical", "urgent", "must", "breaking", "security", "data loss",
        "紧急", "关键", "必须", "安全", "数据丢失"],
    4: ["important", "key decision", "major", "significant", "architecture",
        "重要", "关键决策", "架构", "重大"],
    3: ["decided", "completed", "shipped", "resolved",
        "决定", "完成", "发布"],
    2: ["preference", "habit", "minor", "note",
        "偏好", "习惯", "小", "备注"],
}


def _detect_importance(text: str) -> int:
    text_lower = text.lower()
    for level in [5, 4, 3, 2]:
        for signal in _IMPORTANCE_SIGNALS[level]:
            if signal in text_lower:
                return level
    return 2


# ── Rule-based compression engine ────────────────────────────────────────

# Chinese particles to remove (ordered longer → shorter to avoid partial matches)
_REMOVABLE_PARTICLES = [
    "也就是说", "接下来", "所以说", "然后",
    "就是", "其实", "这个", "那个", "这些", "那些",
    "的", "了", "着", "过", "吗", "呢", "啊", "呀", "哦", "嗯",
]

# Phrase-level replacements (Chinese)
_PHRASE_REPLACEMENTS = [
    ("我们决定", "议定"),
    ("我们团队", "本队"),
    ("我们的项目", "本项"),
    ("已经完成", "已毕"),
    ("正在进行", "进行中"),
    ("需要注意", "注"),
    ("建议使用", "荐"),
    ("大家都同意", "众从"),
    ("所有人同意", "众从"),
]

# Role abbreviations (English → Chinese suffix)
_ROLE_MAP = {
    "lead": "统", "CTO": "统", "CEO": "统", "head": "统", "director": "统",
    "engineer": "工", "developer": "工", "dev": "工", "programmer": "工",
    "designer": "设", "design": "设", "UX": "设",
    "ops": "运", "devops": "运", "infra": "运", "SRE": "运",
    "product": "品", "PM": "品", "manager": "品",
    "backend": "后端", "frontend": "前端", "fullstack": "全栈",
}


class WenjianFormat:
    """Classical Chinese compression format implementing the Format protocol.

    Has both a deterministic rule-based path (works offline, ~3-5x) and an
    LLM-assisted path (higher quality, ~8-15x) selected by the Summarizer.
    """

    name: str = "wenjian"

    def __init__(
        self,
        model_name: str = "",
        person_map: Optional[Dict[str, str]] = None,
        tokenizer: Optional[Callable[[str], int]] = None,
        **_kwargs,
    ):
        """
        Parameters
        ----------
        model_name : str
            LLM model name for tokenizer calibration (e.g., "Qwen3.5-35B").
        person_map : dict, optional
            Maps full/English names to Chinese abbreviations:
            ``{"Kai": "伟明", "Maya": "美云"}``. Used for validation.
        tokenizer : callable, optional
            Custom token counter. Defaults to ``wenjian_tokenizer.count_tokens``.
        """
        self._model_name = model_name
        self._person_map = person_map or {}
        self._count_tokens = tokenizer or get_tokenizer(model_name)

    # ── Format protocol: compress ────────────────────────────────────────

    def compress(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> CompressedEntry:
        """Rule-based Wenjian compression.

        Applies particle removal, phrase replacement, role abbreviation,
        and idiom substitution. Produces a valid Wenjian entry string.

        For higher quality, the Summarizer sends ``prompt_template()``
        to an LLM instead.
        """
        if not memory_type:
            memory_type = _detect_memory_type(text)
        if importance <= 0:
            importance = _detect_importance(text)

        original_tokens = self._count_tokens(text)
        compressed = self._rule_compress(text)

        # Prepend type marker + importance stars
        stars = "★" * min(max(importance, 1), 5)
        wenjian_text = f"{memory_type} {compressed}{stars}"

        compressed_tokens = self._count_tokens(wenjian_text)

        return CompressedEntry(
            text=wenjian_text,
            format_name=self.name,
            memory_type=memory_type,
            importance=importance,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            entities_preserved_ratio=1.0,  # set by validate()
        )

    # ── Format protocol: validate ────────────────────────────────────────

    def validate(
        self,
        original: str,
        compressed: CompressedEntry,
        threshold: float = 0.7,
    ) -> ValidationReport:
        """Check entity preservation, accounting for person_map mappings."""
        return validate_wenjian(
            original,
            compressed.text,
            threshold=threshold,
            person_map=self._person_map,
        )

    # ── Format protocol: prompt_template ─────────────────────────────────

    def prompt_template(
        self,
        text: str,
        *,
        memory_type: str = "",
        importance: int = 0,
        context: Optional[Dict[str, Any]] = None,
        missing_entities: Optional[List[str]] = None,
    ) -> str:
        """Build the few-shot LLM prompt for Wenjian compression."""
        if not memory_type:
            memory_type = _detect_memory_type(text)
        if importance <= 0:
            importance = _detect_importance(text)

        # Detect domain from text for idiom table focus
        domain = self._detect_domain(text)

        return build_compress_prompt(
            text=text,
            memory_type=memory_type,
            importance=importance,
            domain=domain,
            missing_entities=list(missing_entities) if missing_entities else None,
        )

    # ── Format protocol: decompress_hint ─────────────────────────────────

    def decompress_hint(self, entry: CompressedEntry) -> str:
        """Return an LLM prompt for expanding Wenjian back to natural language."""
        return build_expand_prompt(entry.text)

    # ── Internal: rule-based compression ─────────────────────────────────

    def _rule_compress(self, text: str) -> str:
        """Deterministic compression: particles, phrases, roles, idioms.

        Improvement over MCC's original: phrase-level replacements before
        particle removal (avoids breaking sentence structure), and idiom
        substitution from the expanded dictionary.
        """
        result = text

        # 1. Phrase-level replacements (longer patterns first)
        for pattern, replacement in _PHRASE_REPLACEMENTS:
            result = result.replace(pattern, replacement)

        # 2. English phrase → idiom substitution (case-insensitive)
        for trigger, idiom in TRIGGER_TO_IDIOM.items():
            # Use word-boundary-ish matching for English triggers
            pattern = re.compile(re.escape(trigger), re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(idiom.chars, result, count=1)

        # 3. Role abbreviation (English → Chinese suffix)
        for role_en, role_zh in _ROLE_MAP.items():
            # Match "Kai (backend lead)" → "Kai后端统"
            result = re.sub(
                rf"\b{re.escape(role_en)}\b",
                role_zh,
                result,
                flags=re.IGNORECASE,
            )

        # 4. Particle removal (Chinese virtual words)
        for particle in _REMOVABLE_PARTICLES:
            result = result.replace(particle, "")

        # 5. Causal pattern compression
        result = re.sub(r"因为(.{1,30})", r"以\1故", result)
        result = re.sub(r"由于(.{1,30}?)原因", r"以\1故", result)
        result = re.sub(r"相比之?下?", "较", result)

        # 6. Clean up whitespace
        result = re.sub(r"\s+", " ", result).strip()

        # 7. Remove trailing period if present (Wenjian entries end with status/stars)
        if result.endswith("。"):
            result = result[:-1]

        return result

    # ── Internal: domain detection ───────────────────────────────────────

    @staticmethod
    def _detect_domain(text: str) -> Optional[str]:
        """Heuristic domain detection for idiom table focus."""
        text_lower = text.lower()
        tech_kw = ["api", "database", "deploy", "code", "git", "sql", "cache",
                    "server", "migrate", "architecture", "performance", "latency"]
        biz_kw = ["cost", "revenue", "market", "contract", "timeline", "budget",
                   "pricing", "customer", "growth", "launch"]
        tech_score = sum(1 for kw in tech_kw if kw in text_lower)
        biz_score = sum(1 for kw in biz_kw if kw in text_lower)
        if tech_score >= 2:
            return "arch"
        if biz_score >= 2:
            return "biz"
        return None
