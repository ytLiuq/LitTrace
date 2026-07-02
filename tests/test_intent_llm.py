import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.intent import parse_chat_intent
from littrace.intent_llm import (
    IntentParseError,
    _format_llm_failure,
    _merge_intents,
    parse_chat_intent_semantic,
)


def test_semantic_intent_merge_adds_actions_from_llm_payload():
    rule = parse_chat_intent("这些文章放在一起到底走了哪几条路？")
    merged = _merge_intents(
        rule,
        {
            "actions": ["storyline"],
            "topic": None,
            "year_min": None,
            "journals": [],
            "skip_download": False,
        },
    )

    assert "storyline" in merged.actions


def test_semantic_intent_merge_preserves_skip_download_negation():
    rule = parse_chat_intent("帮我找几篇柔性压力传感器论文，先别下载")
    merged = _merge_intents(rule, {"actions": ["search", "download"], "skip_download": True})

    assert "search" in merged.actions
    assert "download" not in merged.actions
    assert merged.skip_download


def test_semantic_intent_failure_message_explains_503():
    message = _format_llm_failure(
        LitTraceConfig(),
        "HTTPStatusError: Server error '503 Service Temporarily Unavailable'",
    )

    assert "DeepSeek 服务暂时不可用" in message


def test_semantic_intent_failure_message_explains_timeout():
    message = _format_llm_failure(LitTraceConfig(), "ReadTimeout")

    assert "LLM 请求超时" in message


@pytest.mark.anyio
async def test_semantic_intent_parser_errors_without_key_when_enabled():
    with pytest.raises(IntentParseError, match="没有配置 LLM API key"):
        await parse_chat_intent_semantic(
            "帮我找几篇柔性压力传感器论文",
            LitTraceConfig(llm=LLMConfig(api_key=None, enabled=True, intent_parser_enabled=True)),
        )


@pytest.mark.anyio
async def test_semantic_intent_parser_allows_explicit_offline_mode():
    intent = await parse_chat_intent_semantic(
        "帮我找几篇柔性压力传感器论文",
        LitTraceConfig(llm=LLMConfig(api_key=None, enabled=True, intent_parser_enabled=False)),
    )

    assert "search" in intent.actions
