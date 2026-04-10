"""
wenjian_idioms.py — expanded Classical Chinese idiom dictionary for Wenjian.

MCC's original had 5 entries. We expand to 35+ domain-tagged idioms with
genuine utility for tech decisions, events, problems, preferences, and
lifecycle tracking.

Each entry is a tuple:
    (idiom, domain, english_triggers, chinese_meaning)

- ``idiom``: 2-4 character Classical Chinese expression
- ``domain``: "arch" (tech/architecture), "biz" (business/people),
  "life" (lifecycle/status), "gen" (general)
- ``english_triggers``: list of English phrases that map to this idiom
- ``chinese_meaning``: brief gloss in modern Chinese

Used by both the rule-based compressor (substitutes English trigger
phrases with idioms) and the LLM prompt (listed as available vocabulary
so the model knows to use them).
"""

from __future__ import annotations

from typing import List, NamedTuple


class Idiom(NamedTuple):
    chars: str
    domain: str
    triggers: List[str]
    meaning: str


# ── Tech / Architecture (12) ─────────────────────────────────────────────

TECH_IDIOMS = [
    Idiom("破竹", "arch", ["major breakthrough", "rapid progress", "breakthrough"], "重大突破，进展顺利"),
    Idiom("金蝉", "arch", ["migration", "migrate", "refactor", "refactoring"], "需迁移/重构"),
    Idiom("定鼎", "arch", ["final decision", "architecture decision", "locked in", "finalized"], "最终敲定架构决策"),
    Idiom("亡羊", "arch", ["technical debt", "tech debt", "bug discovered", "found a bug"], "已发现需补救的缺陷/技术债"),
    Idiom("一石", "arch", ["one stone", "multi-purpose", "kills two birds", "dual benefit"], "一举多得的方案"),
    Idiom("纸上", "arch", ["theoretical", "not yet proven", "in theory", "on paper"], "理论上的/未经验证"),
    Idiom("沙盘", "arch", ["proof of concept", "POC", "prototype", "spike"], "概念验证/原型"),
    Idiom("殊途", "arch", ["fallback", "plan B", "contingency", "alternative"], "备选方案/应急"),
    Idiom("大厦", "arch", ["major system", "pillar", "core component", "foundation"], "主要系统/核心组件"),
    Idiom("齿轮", "arch", ["integration", "coordination", "coupling", "interface"], "协调/集成点"),
    Idiom("堡垒", "arch", ["bottleneck", "critical path", "blocking", "blocker"], "瓶颈/关键路径"),
    Idiom("灯塔", "arch", ["reference implementation", "standard", "benchmark", "gold standard"], "参考实现/标准"),
]

# ── Business / People (10) ────────────────────────────────────────────────

BIZ_IDIOMS = [
    Idiom("虎狼", "biz", ["aggressive growth", "rapid scaling", "blitz"], "激进增长"),
    Idiom("磨刀", "biz", ["preparation", "skill building", "learning phase", "ramp up"], "准备/积累阶段"),
    Idiom("跨越", "biz", ["new market", "boundary crossing", "expansion"], "跨界/拓展"),
    Idiom("众星", "biz", ["team consensus", "everyone agrees", "unanimous", "alignment"], "团队共识"),
    Idiom("舞台", "biz", ["public commitment", "announcement", "launch", "demo day"], "公开承诺/发布"),
    Idiom("暗礁", "biz", ["hidden risk", "unknown unknown", "trap", "pitfall"], "隐藏风险"),
    Idiom("秤砣", "biz", ["key person", "tiebreaker", "decision maker", "gatekeeper"], "关键决策人"),
    Idiom("枯荣", "biz", ["boom and bust", "cycle", "ups and downs"], "盛衰循环"),
    Idiom("双赢", "biz", ["win-win", "mutually beneficial", "both sides gain"], "双方获益"),
    Idiom("回音", "biz", ["feedback loop", "iteration", "feedback cycle", "retrospective"], "反馈循环"),
]

# ── Lifecycle / Status (9) ────────────────────────────────────────────────

LIFE_IDIOMS = [
    Idiom("议定", "life", ["decided", "resolved", "locked in"], "已决定"),
    Idiom("进行", "life", ["in progress", "ongoing", "underway", "WIP"], "进行中"),
    Idiom("毕成", "life", ["completed", "done", "shipped", "finished", "delivered"], "已完成"),
    Idiom("悬而", "life", ["pending", "awaiting", "undecided", "open question"], "悬而未决"),
    Idiom("翻覆", "life", ["reversal", "rollback", "reversed", "undone", "reverted"], "推翻/回滚"),
    Idiom("搁浅", "life", ["stalled", "on hold", "paused", "frozen"], "搁置/暂停"),
    Idiom("冷灶", "life", ["dormant", "low priority", "backburner", "de-prioritized"], "低优先级/搁置"),
    Idiom("重锤", "life", ["force prioritize", "escalate", "urgent", "critical"], "强制优先/紧急"),
    Idiom("妥协", "life", ["trade-off", "compromise", "accepted trade-off"], "权衡取舍"),
]

# ── General / Cross-domain (4) ────────────────────────────────────────────

GEN_IDIOMS = [
    Idiom("恍然", "gen", ["aha moment", "realization", "eureka", "suddenly understood"], "恍然大悟"),
    Idiom("铭记", "gen", ["important to remember", "key takeaway", "never forget"], "务必铭记"),
    Idiom("前车", "gen", ["lessons learned", "past mistake", "cautionary tale"], "前车之鉴"),
    Idiom("画龙", "gen", ["finishing touch", "key detail", "crucial last step"], "画龙点睛"),
]


# ── Combined registry ────────────────────────────────────────────────────

ALL_IDIOMS: List[Idiom] = TECH_IDIOMS + BIZ_IDIOMS + LIFE_IDIOMS + GEN_IDIOMS

# Lookup: english trigger (lowercased) → Idiom
TRIGGER_TO_IDIOM = {}
for _idiom in ALL_IDIOMS:
    for _trigger in _idiom.triggers:
        TRIGGER_TO_IDIOM[_trigger.lower()] = _idiom

# Lookup: idiom chars → Idiom
CHARS_TO_IDIOM = {i.chars: i for i in ALL_IDIOMS}

# Grouped by domain for prompt injection
IDIOMS_BY_DOMAIN = {
    "arch": TECH_IDIOMS,
    "biz": BIZ_IDIOMS,
    "life": LIFE_IDIOMS,
    "gen": GEN_IDIOMS,
}


def format_idiom_table_for_prompt(domain: str = None) -> str:
    """Format idioms as a compact table string for LLM prompt injection.

    If domain is None, include all domains.
    """
    idioms = IDIOMS_BY_DOMAIN.get(domain, ALL_IDIOMS) if domain else ALL_IDIOMS
    lines = ["典故词/Idioms:"]
    for i in idioms:
        triggers_str = ", ".join(i.triggers[:2])
        lines.append(f"  {i.chars} = {triggers_str} ({i.meaning})")
    return "\n".join(lines)
