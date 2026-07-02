from __future__ import annotations

import json

from littrace.config import LitTraceConfig
from littrace.intent import ChatIntent, parse_chat_intent
from littrace.llm import chat_completion


class IntentParseError(RuntimeError):
    pass


async def parse_chat_intent_semantic(
    message: str,
    config: LitTraceConfig,
) -> ChatIntent:
    rule_intent = parse_chat_intent(message)
    if not config.llm.intent_parser_enabled:
        return rule_intent
    if not config.llm.enabled:
        raise IntentParseError("LLM 意图解析已启用，但 LLM 当前被禁用。")
    if not config.llm.api_key:
        raise IntentParseError("LLM 意图解析已启用，但没有配置 LLM API key。")

    reply = await chat_completion(
        config,
        _INTENT_SYSTEM_PROMPT,
        _intent_user_message(message, rule_intent),
        workspace=None,
    )
    if not reply.used_llm:
        raise IntentParseError(_format_llm_failure(config, reply.error))
    try:
        payload = json.loads(_extract_json(reply.text))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise IntentParseError("LLM 意图解析返回了无效 JSON。")
    return _merge_intents(rule_intent, payload)


_INTENT_SYSTEM_PROMPT = """You parse user intent for LitTrace, a literature research assistant.
Return only strict JSON. Do not answer the user.
Schema:
{
  "actions": ["search"|"download"|"select_downloads"|"deselect_downloads"|"parse"|"table"|"storyline"|"document"|"autonomous_review"|"list_context"|"agent_status"|"show_context"|"hide_context"],
  "topic": "string or null",
  "year_min": 2024 or null,
  "journals": ["canonical journal names"],
  "skip_download": true|false,
  "select_all_downloads": true|false,
  "clear_download_selection": true|false,
  "auto_replan": true|false,
  "parse_strategy": "text_only"|"ocr"|null,
  "select_indices": [1,2],
  "deselect_indices": [3]
}
Prefer search when the user asks to investigate, survey, find papers, understand a topic, or research a field.
Prefer storyline when the user asks for routes, main lines, evolution, narrative, development logic, or how papers relate.
Prefer table when the user asks for metrics, comparison, performance, benchmark, or matrix.
Do not include actions that the user explicitly negates."""


def _intent_user_message(message: str, rule_intent: ChatIntent) -> str:
    return (
        f"User message:\n{message}\n\n"
        "Rule parser draft:\n"
        f"{json.dumps(rule_intent.__dict__, ensure_ascii=False)}"
    )


def _format_llm_failure(config: LitTraceConfig, error: str | None) -> str:
    detail = error or "unknown_error"
    hint = ""
    if "503" in detail or "Service Temporarily Unavailable" in detail:
        hint = " DeepSeek 服务暂时不可用，稍后重试通常可以恢复。"
    elif "Timeout" in detail or "timed out" in detail.lower():
        hint = " LLM 请求超时，请稍后重试，或检查网络、base_url、模型名与服务状态。"
    elif "404" in detail or "model" in detail.lower():
        hint = f" 请检查模型名是否可用；当前模型为 {config.llm.model}。"
    return f"LLM 意图解析失败：{detail}.{hint}"


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped


def _merge_intents(rule_intent: ChatIntent, payload: dict[str, object]) -> ChatIntent:
    merged = ChatIntent(
        actions=_clean_actions(payload.get("actions"), rule_intent.actions),
        topic=_clean_topic(payload.get("topic")) or rule_intent.topic,
        year_min=_clean_year(payload.get("year_min")) or rule_intent.year_min,
        journals=_clean_string_list(payload.get("journals")) or rule_intent.journals,
        skip_download=bool(payload.get("skip_download")) or rule_intent.skip_download,
        show_context=rule_intent.show_context,
        select_all_downloads=bool(payload.get("select_all_downloads")) or rule_intent.select_all_downloads,
        clear_download_selection=bool(payload.get("clear_download_selection")) or rule_intent.clear_download_selection,
        auto_replan=bool(payload.get("auto_replan")) or rule_intent.auto_replan,
        parse_strategy=_clean_parse_strategy(payload.get("parse_strategy")) or rule_intent.parse_strategy,
        select_indices=_clean_int_list(payload.get("select_indices")) or rule_intent.select_indices,
        deselect_indices=_clean_int_list(payload.get("deselect_indices")) or rule_intent.deselect_indices,
    )
    if rule_intent.show_context is not None:
        merged.show_context = rule_intent.show_context
    merged.actions = _dedupe([*rule_intent.actions, *merged.actions])
    if merged.skip_download:
        merged.actions = [action for action in merged.actions if action != "download"]
    return merged


def _clean_actions(value: object, fallback: list[str]) -> list[str]:
    allowed = {
        "search",
        "download",
        "select_downloads",
        "deselect_downloads",
        "parse",
        "table",
        "storyline",
        "document",
        "autonomous_review",
        "list_context",
        "agent_status",
        "show_context",
        "hide_context",
    }
    actions = [item for item in _clean_string_list(value) if item in allowed]
    return _dedupe(actions or fallback)


def _clean_topic(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip(" ：:，,。.")
    if not cleaned or cleaned.lower() in {"null", "none", "n/a"}:
        return None
    return cleaned


def _clean_year(value: object) -> int | None:
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    return None


def _clean_parse_strategy(value: object) -> str | None:
    return value if value in {"text_only", "ocr"} else None


def _clean_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _clean_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and item > 0]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
