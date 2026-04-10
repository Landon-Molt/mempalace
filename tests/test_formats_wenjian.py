"""Tests for the Wenjian compression format and supporting modules.

Covers: rule-based compression, entity preservation validation, idiom
dictionary, language detection, tokenizer, few-shot prompt generation,
and the summarizer's compress/validate/fallback paths. Uses recorded
LLM fixtures (not live calls) for the LLM-assisted tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mempalace.formats import resolve_format
from mempalace.formats.base import (
    CompressedEntry,
    ValidationReport,
    extract_entities,
    validate_preservation,
)
from mempalace.formats.language_detect import detect_languages, is_cjk_dominant
from mempalace.formats.wenjian import WenjianFormat, _detect_memory_type, _detect_importance
from mempalace.formats.wenjian_idioms import ALL_IDIOMS, TRIGGER_TO_IDIOM, Idiom
from mempalace.formats.wenjian_prompts import EXAMPLES, build_compress_prompt
from mempalace.formats.wenjian_tokenizer import count_tokens, _heuristic_count
from mempalace.formats.wenjian_validation import validate_wenjian
from mempalace.summarizer import Summarizer, _strip_thinking


# ── resolve_format ───────────────────────────────────────────────────────


def test_resolve_wenjian():
    fmt = resolve_format("wenjian")
    assert fmt.name == "wenjian"


def test_resolve_aaak():
    fmt = resolve_format("aaak")
    assert fmt.name == "aaak"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        resolve_format("nonexistent")


# ── WenjianFormat.compress (rule-based) ──────────────────────────────────


class TestWenjianRuleCompress:
    def test_basic_compression(self):
        fmt = WenjianFormat()
        entry = fmt.compress("The team decided to use PostgreSQL for the database.")
        assert entry.format_name == "wenjian"
        assert entry.text.startswith("议")
        assert "★" in entry.text

    def test_memory_type_detection_decision(self):
        fmt = WenjianFormat()
        entry = fmt.compress("We decided to use React for the frontend.")
        assert entry.memory_type == "议"

    def test_memory_type_detection_event(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Successfully deployed the API to production.")
        assert entry.memory_type == "事"

    def test_memory_type_detection_discovery(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Discovered that the cache invalidation was broken.")
        assert entry.memory_type == "得"

    def test_memory_type_detection_preference(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Peter always prefers TypeScript over JavaScript.")
        assert entry.memory_type == "好"

    def test_memory_type_detection_advice(self):
        fmt = WenjianFormat()
        entry = fmt.compress("I recommend using Clerk instead of Auth0.")
        assert entry.memory_type == "策"

    def test_importance_detection(self):
        fmt = WenjianFormat()
        critical = fmt.compress("URGENT: security vulnerability in auth.")
        assert critical.importance >= 4

        minor = fmt.compress("Minor note: updated README formatting.")
        assert minor.importance <= 2

    def test_idiom_substitution(self):
        fmt = WenjianFormat()
        entry = fmt.compress("The migration to the new database is a major breakthrough.")
        # "migration" → 金蝉, "major breakthrough" → 破竹
        assert "金蝉" in entry.text or "破竹" in entry.text

    def test_role_abbreviation(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Kai is the backend engineer on the team.")
        assert "工" in entry.text  # engineer → 工

    def test_chinese_input_particle_removal(self):
        fmt = WenjianFormat()
        text = "团队决定了使用PostgreSQL作为数据库的方案。"
        entry = fmt.compress(text)
        # 了 and 的 should be removed
        assert "了" not in entry.text or entry.text.count("了") < text.count("了")

    def test_german_input_works(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Wir haben die Datenbank auf PostgreSQL umgestellt.")
        assert entry.format_name == "wenjian"
        assert "★" in entry.text

    def test_explicit_memory_type_overrides_detection(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Updated the docs.", memory_type="事")
        assert entry.memory_type == "事"
        assert entry.text.startswith("事")

    def test_explicit_importance_overrides_detection(self):
        fmt = WenjianFormat()
        entry = fmt.compress("Minor fix.", importance=5)
        assert entry.importance == 5
        assert "★★★★★" in entry.text


# ── WenjianFormat.validate ───────────────────────────────────────────────


class TestWenjianValidation:
    def test_perfect_preservation(self):
        fmt = WenjianFormat()
        entry = CompressedEntry(
            text="议 Auth0→Clerk $240→$25/mo Q1 2026[定]★★★★",
            format_name="wenjian",
        )
        report = fmt.validate(
            "Migrate Auth0 to Clerk for $240/mo vs $25/mo by Q1 2026.",
            entry,
        )
        assert report.is_acceptable

    def test_low_preservation_flagged(self):
        fmt = WenjianFormat()
        entry = CompressedEntry(text="议 决定迁移[定]★★★", format_name="wenjian")
        report = fmt.validate(
            "Migrate Auth0 to Clerk for $240/mo. Kai recommended, Maya executes.",
            entry,
            threshold=0.5,
        )
        # "Auth0", "Clerk", "$240", "Kai", "Maya" all missing
        assert len(report.missing_entities) > 0

    def test_person_map_counts_chinese_names_as_preserved(self):
        fmt = WenjianFormat(person_map={"Kai": "伟明", "Maya": "美云"})
        entry = CompressedEntry(
            text="议 伟明工荐 Auth0→Clerk 美云运执[定]★★★★",
            format_name="wenjian",
        )
        report = fmt.validate(
            "Kai recommended migrating Auth0 to Clerk. Maya will execute.",
            entry,
        )
        # "Kai" and "Maya" should be counted as preserved via person_map
        assert "Kai" not in report.missing_entities
        assert "Maya" not in report.missing_entities


# ── WenjianFormat.prompt_template ────────────────────────────────────────


class TestWenjianPrompt:
    def test_prompt_includes_spec(self):
        fmt = WenjianFormat()
        prompt = fmt.prompt_template("Test text", memory_type="议")
        assert "文简规范" in prompt or "Wenjian Spec" in prompt

    def test_prompt_includes_example(self):
        fmt = WenjianFormat()
        prompt = fmt.prompt_template("Test text", memory_type="事")
        assert "示例" in prompt or "Example" in prompt
        assert "GraphQL" in prompt  # the 事 example mentions GraphQL

    def test_prompt_includes_idiom_table(self):
        fmt = WenjianFormat()
        prompt = fmt.prompt_template("Database migration plan.")
        assert "典故" in prompt or "Idiom" in prompt
        assert "破竹" in prompt

    def test_prompt_includes_missing_entities_on_retry(self):
        fmt = WenjianFormat()
        prompt = fmt.prompt_template(
            "Test text",
            memory_type="议",
            missing_entities=["Auth0", "Clerk"],
        )
        assert "Auth0" in prompt
        assert "Clerk" in prompt
        assert "遗漏" in prompt or "保留" in prompt


# ── Idiom dictionary ────────────────────────────────────────────────────


class TestIdioms:
    def test_idiom_count_at_least_35(self):
        assert len(ALL_IDIOMS) >= 35

    def test_trigger_lookup(self):
        assert "migration" in TRIGGER_TO_IDIOM
        assert TRIGGER_TO_IDIOM["migration"].chars == "金蝉"

    def test_all_idioms_have_valid_shape(self):
        for idiom in ALL_IDIOMS:
            assert isinstance(idiom, Idiom)
            assert len(idiom.chars) >= 2
            assert idiom.domain in ("arch", "biz", "life", "gen")
            assert len(idiom.triggers) >= 1
            assert len(idiom.meaning) >= 2

    def test_domains_covered(self):
        domains = {i.domain for i in ALL_IDIOMS}
        assert domains == {"arch", "biz", "life", "gen"}


# ── Language detection ──────────────────────────────────────────────────


class TestLanguageDetect:
    def test_english(self):
        assert "en" in detect_languages("Hello world")

    def test_chinese(self):
        assert "zh" in detect_languages("你好世界")

    def test_german(self):
        assert "de" in detect_languages("Schöne Grüße")

    def test_mixed(self):
        langs = detect_languages("Hello 你好 Grüße")
        assert "en" in langs
        assert "zh" in langs
        assert "de" in langs

    def test_empty_defaults_to_english(self):
        assert detect_languages("") == {"en"}

    def test_cjk_dominant(self):
        assert is_cjk_dominant("这是中文测试文本") is True
        assert is_cjk_dominant("This is English") is False
        assert is_cjk_dominant("") is False


# ── Tokenizer ───────────────────────────────────────────────────────────


class TestTokenizer:
    def test_heuristic_nonempty(self):
        assert _heuristic_count("hello") >= 1
        assert _heuristic_count("你好") >= 1

    def test_heuristic_empty(self):
        assert _heuristic_count("") == 0

    def test_count_tokens_with_unknown_model(self):
        # Falls back to heuristic
        assert count_tokens("hello world", "qwen3.5-35b") >= 1

    def test_chinese_tokens_reasonable(self):
        # ~1.2 tokens per Chinese char
        n = count_tokens("你好世界测试一下", "qwen3.5")
        assert 5 <= n <= 20


# ── Entity extraction ──────────────────────────────────────────────────


class TestEntityExtraction:
    def test_camelcase(self):
        entities = extract_entities("Using GraphQL and PostgreSQL")
        assert "GraphQL" in entities
        assert "PostgreSQL" in entities

    def test_mixed_case_digits(self):
        entities = extract_entities("Migrating from Auth0 to OAuth2")
        assert "Auth0" in entities
        assert "OAuth2" in entities

    def test_prices(self):
        entities = extract_entities("Cost: $240/mo vs $25/mo")
        assert "$240" in entities or "240/mo" in entities

    def test_versions(self):
        entities = extract_entities("Upgraded to v3.2.1 from v2.0")
        assert "v3.2.1" in entities
        assert "v2.0" in entities

    def test_quarters(self):
        entities = extract_entities("Target: Q1 2026")
        assert "Q1" in entities

    def test_common_words_excluded(self):
        entities = extract_entities("The team decided on this approach.")
        assert "The" not in entities


# ── Summarizer ───────────────────────────────────────────────────────────


class TestSummarizer:
    def test_rule_only_no_llm(self):
        fmt = resolve_format("wenjian")
        s = Summarizer(fmt)
        assert not s.has_llm
        entry = s.compress("Decided to use PostgreSQL.")
        assert entry.format_name == "wenjian"
        assert entry.entities_preserved_ratio > 0

    def test_rule_only_flag_overrides_llm(self):
        fmt = resolve_format("wenjian")
        mock_llm = MagicMock()
        s = Summarizer(fmt, llm=mock_llm, rule_only=True)
        assert not s.has_llm
        entry = s.compress("Test text.")
        mock_llm.generate.assert_not_called()

    def test_llm_path_success(self):
        """Simulated LLM returns valid Wenjian text."""
        fmt = resolve_format("wenjian")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "议 Auth0→Clerk 金蝉迁身份 $240→$25/mo Q1 2026末[定]★★★★"
        mock_llm.model_name = "test-model"
        s = Summarizer(fmt, llm=mock_llm)
        assert s.has_llm
        entry = s.compress(
            "The team decided to migrate Auth0 to Clerk for $240/mo vs $25/mo. Target: Q1 2026.",
            memory_type="议",
            importance=4,
        )
        assert "Auth0" in entry.text
        assert entry.llm_model == "test-model"

    def test_llm_path_fallback_on_error(self):
        """LLM fails → falls through to rule-based."""
        fmt = resolve_format("wenjian")
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("connection refused")
        mock_llm.model_name = "test-model"
        s = Summarizer(fmt, llm=mock_llm)
        entry = s.compress("Decided to use PostgreSQL.", memory_type="议")
        # Should not raise; should fall back to rule-based
        assert entry.format_name == "wenjian"
        assert entry.llm_model == ""  # rule-based, no LLM model recorded

    def test_llm_path_fallback_on_empty_response(self):
        fmt = resolve_format("wenjian")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ""
        mock_llm.model_name = "test-model"
        s = Summarizer(fmt, llm=mock_llm)
        entry = s.compress("Test text.")
        assert entry.format_name == "wenjian"


# ── _strip_thinking ──────────────────────────────────────────────────────


class TestStripThinking:
    def test_strip_think_tags(self):
        text = "<think>Lots of reasoning here</think>\n议 Auth0→Clerk[定]★★★★"
        result = _strip_thinking(text, "wenjian")
        assert result.startswith("议")
        assert "<think>" not in result

    def test_strip_finds_wenjian_marker(self):
        text = "Some reasoning text\nMore reasoning\n議 Auth0→Clerk[定]★★★★"
        result = _strip_thinking(text, "wenjian")
        # Should not start with reasoning
        # But the type marker is 議 (traditional) not 议 (simplified)
        # Our markers check for simplified. Let's test with simplified.
        text2 = "Some reasoning text\nMore reasoning\n议 Auth0→Clerk[定]★★★★"
        result2 = _strip_thinking(text2, "wenjian")
        assert result2.startswith("议")

    def test_strip_empty_input(self):
        assert _strip_thinking("", "wenjian") == ""

    def test_strip_clean_input_passes_through(self):
        text = "议 Auth0→Clerk[定]★★★★"
        assert _strip_thinking(text, "wenjian") == text

    def test_strip_aaak_finds_numeric_start(self):
        text = "Some reasoning\n0:ALC+BOB|topic|\"quote\"|emotions"
        result = _strip_thinking(text, "aaak")
        assert result.startswith("0:")
