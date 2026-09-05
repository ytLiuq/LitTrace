"""Semantic query planning shared by UI, MCP, and legacy search paths."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from littrace.cache import cache_key, read_text_cache, write_text_cache
from littrace.config import LitTraceConfig
from littrace.llm import chat_completion


class QueryPlan(BaseModel):
    canonical_topic: str
    query_variants: list[str] = Field(default_factory=list)


_TERM_MAP = {
    "电容式": "capacitive",
    "压阻式": "piezoresistive",
    "压阻": "piezoresistive",
    "压力": "pressure",
    "传感器": "sensor",
    "传感": "sensing",
    "柔性": "flexible",
    "可穿戴": "wearable",
    "阵列": "array",
    "制造": "fabrication",
    "工艺": "process",
    "薄膜": "thin film",
    "材料": "materials",
    "石墨烯": "graphene",
    "MXene": "MXene",
}


async def plan_query_variants(
    topic: str,
    config: LitTraceConfig,
) -> list[str]:
    normalized = re.sub(r"\s+", " ", topic or "").strip()
    if not normalized:
        return []
    fallback = deterministic_query_variants(normalized)
    if not getattr(config.api, "enable_semantic_query_planner", True):
        return fallback
    if not config.llm.enabled or not config.llm.api_key:
        return fallback
    cache_id = cache_key(f"query-plan-v1\n{config.llm.model}\n{normalized}")
    cached = read_text_cache(config, "query-plans", cache_id)
    if cached:
        try:
            return _normalize_plan(json.loads(cached), normalized, fallback)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    reply = await chat_completion(
        config,
        "你是学术检索查询规划器。只返回 JSON，不要解释。",
        (
            f"原始研究主题：{normalized}\n"
            "请生成适合学术数据库检索的中英文 query_variants（最多 6 条）。"
            "保留原始主题；英文变体只做忠实翻译、同义词和常见学术表达扩展，"
            "不得凭空添加材料、机制、应用或年份。JSON 字段："
            "canonical_topic（简短英文主题），query_variants（字符串数组）。"
        ),
        workspace=None,
        json_mode=True,
    )
    if not reply.used_llm or not reply.text.strip():
        return fallback
    try:
        payload = json.loads(_extract_json(reply.text))
        plan = _normalize_plan(payload, normalized, fallback)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    write_text_cache(
        config,
        "query-plans",
        cache_id,
        json.dumps(
            {"canonical_topic": plan[0] if plan else normalized, "query_variants": plan},
            ensure_ascii=False,
        ),
    )
    return plan


def deterministic_query_variants(topic: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", topic or "").strip()
    if not normalized:
        return []
    translated = normalized
    for source, target in sorted(_TERM_MAP.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(source, f" {target} ")
    translated = re.sub(r"\s+", " ", translated).strip()
    variants = [normalized]
    if translated and translated != normalized and re.search(r"[A-Za-z]", translated):
        variants.append(translated)
    return list(dict.fromkeys(variants))[:6]


def _normalize_plan(
    payload: object,
    original: str,
    fallback: list[str],
) -> list[str]:
    if not isinstance(payload, dict):
        return fallback
    raw = payload.get("query_variants")
    values = raw if isinstance(raw, list) else []
    variants = [
        re.sub(r"\s+", " ", value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]
    return list(dict.fromkeys([original, *variants, *fallback]))[:6]


def _extract_json(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    return stripped[start : end + 1] if start >= 0 and end >= start else stripped
