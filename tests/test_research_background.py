import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.research_background import assess_research_background


@pytest.mark.anyio
async def test_research_background_llm_review_can_reject(monkeypatch):
    async def fake_chat_completion(config, system_prompt, user_message, workspace=None, json_mode=False):
        return type(
            "Reply",
            (),
            {
                "used_llm": True,
                "text": (
                    '{"accepted": false, "reason": "topic_too_vague", '
                    '"suggestions": ["请明确材料体系和应用场景"], "confidence": 0.91}'
                ),
            },
        )()

    monkeypatch.setattr("littrace.research_background.chat_completion", fake_chat_completion)

    config = LitTraceConfig(llm=LLMConfig(enabled=True, api_key="fake-key"))
    assessment = await assess_research_background(
        "我研究柔性压力传感器的器件机理和长期稳定性",
        config,
    )

    assert assessment.accepted is False
    assert assessment.reason == "topic_too_vague"
    assert assessment.review_source == "llm"
    assert assessment.reviewer_model == config.llm.model


@pytest.mark.anyio
async def test_research_background_llm_review_normalizes_string_suggestions(monkeypatch):
    async def fake_chat_completion(config, system_prompt, user_message, workspace=None, json_mode=False):
        return type(
            "Reply",
            (),
            {
                "used_llm": True,
                "text": (
                    '{"accepted": true, "suggestions": "可补充应用场景", '
                    '"topic": "柔性压阻压力传感器", '
                    '"retrieval_policy": {"canonical_topic": "flexible piezoresistive pressure sensor", '
                    '"query_variants": ["flexible piezoresistive pressure sensor"], '
                    '"required_concept_groups": [["flexible"], ["pressure"], ["piezoresistive"]]}}'
                ),
            },
        )()

    monkeypatch.setattr("littrace.research_background.chat_completion", fake_chat_completion)
    assessment = await assess_research_background(
        "我研究柔性压阻压力传感器的长期稳定性与微结构设计",
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="fake-key")),
    )

    assert assessment.accepted is True
    assert assessment.suggestions == ["可补充应用场景"]
    assert assessment.retrieval_policy is not None


@pytest.mark.anyio
async def test_research_background_llm_review_can_accept(monkeypatch):
    async def fake_chat_completion(config, system_prompt, user_message, workspace=None, json_mode=False):
        return type(
            "Reply",
            (),
            {
                "used_llm": True,
                "text": (
                    '{"accepted": true, "topic": "柔性压阻压力传感器", '
                    '"retrieval_policy": {"canonical_topic": "flexible piezoresistive pressure sensor", '
                    '"query_variants": ["flexible piezoresistive pressure sensor"], '
                    '"required_concept_groups": [["flexible"], ["pressure"], ["piezoresistive", "resistive"]], '
                    '"excluded_concepts": ["capacitive", "optical"], "boost_concepts": ["stability"]}, '
                    '"confidence": 0.94}'
                ),
            },
        )()

    monkeypatch.setattr("littrace.research_background.chat_completion", fake_chat_completion)

    config = LitTraceConfig(llm=LLMConfig(enabled=True, api_key="fake-key"))
    assessment = await assess_research_background(
        "我研究柔性压力传感器的器件机理和长期稳定性",
        config,
    )

    assert assessment.accepted is True
    assert assessment.topic == "柔性压阻压力传感器"
    assert assessment.retrieval_policy is not None
    assert assessment.retrieval_policy.canonical_topic == "flexible piezoresistive pressure sensor"
    assert assessment.review_source == "llm"
    assert assessment.reviewer_model == config.llm.model
