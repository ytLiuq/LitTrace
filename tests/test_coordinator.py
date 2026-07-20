import pytest

from littrace.config import LitTraceConfig, LLMConfig
from littrace.coordinator import LitTraceCoordinator
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.runtime.memory import (
    EpisodicMemory,
    MemoryRecord,
    PreferenceMemory,
    SessionMemory,
    WorkingMemory,
)


@pytest.mark.anyio
async def test_coordinator_marks_ambiguous_intent_and_persists_pending(monkeypatch):
    async def fake_parse(message, config):
        from littrace.intent import ChatIntent

        return ChatIntent(
            actions=[],
            topic=None,
            confidence=0.21,
            ambiguous=True,
            ambiguity_reasons=["主题不明确"],
            clarification_questions=["你要检索什么主题？"],
        )

    monkeypatch.setattr("littrace.coordinator.parse_chat_intent_semantic", fake_parse)

    turn = await LitTraceCoordinator().prepare_turn(
        ChatRequest(message="检索"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False)),
    )

    assert turn.intent is not None
    assert turn.early_response is not None
    assert turn.early_response.action == "clarify_intent"
    assert turn.workspace.context.filters.pending_intent is not None
    assert "pending_intent_active" in turn.memory_view.warnings


@pytest.mark.anyio
async def test_coordinator_can_cancel_pending_intent(monkeypatch):
    async def fake_parse(message, config):
        from littrace.intent import ChatIntent

        return ChatIntent(actions=[], topic="MXene", confidence=0.88)

    monkeypatch.setattr("littrace.coordinator.parse_chat_intent_semantic", fake_parse)

    workspace = LiteratureWorkspace()
    workspace.context.filters.pending_intent = {
        "actions": ["search"],
        "topic": "MXene 柔性压力传感器",
        "confidence": 0.2,
        "ambiguous": True,
    }

    turn = await LitTraceCoordinator().prepare_turn(
        ChatRequest(message="取消"),
        workspace,
        LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False)),
    )

    assert turn.early_response is not None
    assert turn.early_response.action == "cancel_pending_intent"
    assert turn.workspace.context.filters.pending_intent is None
    assert turn.early_response.intent_confidence == 0.88


@pytest.mark.anyio
async def test_coordinator_uses_session_memory_when_provided(monkeypatch):
    async def fake_parse(message, config):
        from littrace.intent import ChatIntent

        return ChatIntent(actions=["document"], topic="MXene", confidence=0.91)

    monkeypatch.setattr("littrace.coordinator.parse_chat_intent_semantic", fake_parse)

    session_memory = SessionMemory(
        working=WorkingMemory(active_paper_ids=["p1"]),
        episodic=EpisodicMemory(
            records=[
                MemoryRecord(
                    kind="episodic",
                    scope="session",
                    source="artifact_index",
                    content={"artifact_id": "snap1", "kind": "workspace_snapshot"},
                )
            ]
        ),
        preference=PreferenceMemory(values={"preferred_parser": "docling"}),
    )

    turn = await LitTraceCoordinator().prepare_turn(
        ChatRequest(message="生成报告"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False)),
        session_memory=session_memory,
    )

    assert turn.memory_view.preferences["preferred_parser"] == "docling"
    assert turn.memory_view.recent_episodes
    assert turn.task is not None
    assert turn.task.topic == "MXene"
    assert turn.task.evidence_policy == "verified_or_corroborated"
