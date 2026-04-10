"""
wenjian_prompts.py — few-shot prompt templates for LLM-assisted Wenjian compression.

MCC's original used a single-shot prompt with no examples. We improve this
with one worked example per MemoryType (议/事/得/好/策), showing the LLM
exactly what good Wenjian output looks like.

The prompt structure:
1. Role + spec (concise — ~100 tokens)
2. Idiom table (domain-appropriate subset)
3. Few-shot example for the detected memory type
4. The text to compress
5. Output rules (preserve entities, start with type marker, etc.)
"""

from __future__ import annotations

from .wenjian_idioms import format_idiom_table_for_prompt

# ── Wenjian spec (concise version for prompt injection) ─────────────────

WENJIAN_SPEC_SHORT = """【文简规范 · Wenjian Spec】
省虚词(的/了/着/过/吗/呢/啊)·英文术语保留原样·
类型：议(决策)事(事件)得(发现)好(偏好)策(建议)·
状态：[定][疑][废][进][毕][阻]·
重要：★~★★★★★·时间：26/03/15格式·
人名缩至双字+角色后缀(统/工/设/运/品)"""

# ── Few-shot examples (one per MemoryType) ──────────────────────────────

EXAMPLES = {
    "议": {
        "original": (
            "The team decided to migrate authentication from Auth0 to Clerk. "
            "Kai (backend lead, 3 years) recommended this based on pricing "
            "($240/mo → $25/mo) and developer experience. Maya (infra) will "
            "handle migration. Target: Q1 2026 end."
        ),
        "compressed": "议 26/Q1末 金蝉迁身份：Auth0→Clerk。伟明工荐（价240→25/mo，工便）[定]。美云运执。★★★★",
    },
    "事": {
        "original": (
            "Successfully deployed the new GraphQL API to production on "
            "March 15, 2026. Reduced latency by 40% and unblocked 3 "
            "pending frontend features."
        ),
        "compressed": "事 26/03/15 破竹：新GraphQL API上线·延迟↓40%·解锁3前端特性[毕]★★★",
    },
    "得": {
        "original": (
            "Discovered that PostgreSQL's JSON indexing performance degrades "
            "significantly with nested depth greater than 4 levels. This "
            "explains why our complex queries are slow."
        ),
        "compressed": "得 PostgreSQL JSON索引·嵌套>4层性能衰·释疑复杂查询慢[定]★★★",
    },
    "好": {
        "original": (
            "Peter always prefers async-first architecture, uses TypeScript "
            "exclusively for frontend, and reviews all code before merging."
        ),
        "compressed": "好 Peter：异步优先·前端必TypeScript·合并前必评审★★",
    },
    "策": {
        "original": (
            "For scaling databases, recommend sharding by tenant_id rather "
            "than time-based partitions. Time-based causes query complexity "
            "issues at scale."
        ),
        "compressed": "策 DB扩展：以tenant_id分片，勿分时间·时间分片致查询复杂[定]★★★",
    },
}

# ── Prompt builder ──────────────────────────────────────────────────────


def build_compress_prompt(
    text: str,
    memory_type: str = "议",
    importance: int = 3,
    domain: str = None,
    missing_entities: list = None,
) -> str:
    """Build the full LLM prompt for Wenjian compression.

    Parameters
    ----------
    text : str
        The original text to compress.
    memory_type : str
        One of 议/事/得/好/策. Used to pick the few-shot example.
    importance : int
        1-5 scale. Injected as ★-count suggestion.
    domain : str, optional
        If known, injects domain-specific idioms only (arch/biz/life/gen).
    missing_entities : list, optional
        If this is a retry after a failed entity-preservation check,
        include the list of missing entities so the LLM knows to preserve
        them explicitly.
    """
    stars = "★" * min(max(importance, 1), 5)
    example = EXAMPLES.get(memory_type, EXAMPLES["议"])
    idiom_table = format_idiom_table_for_prompt(domain)

    parts = [
        "你是文简压缩引擎。将内容压缩为文言文速记格式。",
        "",
        WENJIAN_SPEC_SHORT,
        "",
        idiom_table,
        "",
        "【示例 / Example】",
        f"原文：{example['original']}",
        f"文简：{example['compressed']}",
        "",
    ]

    if missing_entities:
        entities_str = ", ".join(missing_entities[:10])
        parts.extend([
            f"⚠ 上次压缩遗漏了关键实体：{entities_str}",
            "请务必在文简中保留以上实体。",
            "",
        ])

    parts.extend([
        f"待压缩内容（{memory_type}，建议重要度{stars}）：",
        "---",
        text,
        "---",
        "",
        "规则：",
        f"1. 以 {memory_type} 开头",
        "2. 保留所有英文技术术语·数字·日期·人名",
        "3. 删除所有多余虚词",
        "4. 适当使用上述典故词",
        "5. 末尾标注状态[定/疑/废/进/毕/阻]+重要度★",
        "6. 直接输出文简，不要解释",
        "",
        "文简：",
    ])

    return "\n".join(parts)


def build_expand_prompt(wenjian_text: str) -> str:
    """Build a prompt for expanding Wenjian back to natural language."""
    return (
        "你是文简解读引擎。将以下文简还原为完整现代汉语/英语。\n\n"
        f"{WENJIAN_SPEC_SHORT}\n\n"
        f"文简：{wenjian_text}\n\n"
        "规则：\n"
        "1. 保持所有技术术语不变\n"
        "2. 补全省略的主语和时态\n"
        "3. 不要添加额外说明\n\n"
        "展开内容："
    )
