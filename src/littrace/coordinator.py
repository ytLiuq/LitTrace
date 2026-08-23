from __future__ import annotations

from pydantic import BaseModel

from littrace.config import LitTraceConfig
from littrace.intent import (
    ChatIntent,
    intent_from_payload,
    intent_to_payload,
    merge_pending_intent,
)
from littrace.intent_llm import IntentParseError, parse_chat_intent_semantic
from littrace.models import (
    ChatRequest,
    ChatResponse,
    LiteratureWorkspace,
    ResearchTask,
    WorkspaceSummary,
)
from littrace.runtime.memory import MemoryView, SessionMemory, build_memory_view


class CoordinatorTurn(BaseModel):
    """Prepared single-agent coordinator turn.

    The coordinator owns user-facing intent/memory preparation. Execution is
    still delegated to chat/workflow/skills while the architecture is being
    migrated toward Single Coordinator + Skills.
    """

    intent: ChatIntent | None = None
    task: ResearchTask | None = None
    workspace: LiteratureWorkspace
    memory_view: MemoryView
    early_response: ChatResponse | None = None


class LitTraceCoordinator:
    """Single coordinator facade for chat turns."""

    async def prepare_turn(
        self,
        request: ChatRequest,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
        session_memory: SessionMemory | None = None,
    ) -> CoordinatorTurn:
        message = request.message.strip()
        try:
            intent = await parse_chat_intent_semantic(message, config)
        except IntentParseError as exc:
            return CoordinatorTurn(
                workspace=workspace,
                memory_view=build_memory_view(
                    workspace,
                    purpose="planning",
                    session_memory=session_memory,
                ),
                early_response=ChatResponse(
                    reply=(
                        f"{exc}\n\n"
                        "我没有继续执行，也没有回退到关键词规则。请检查 .env.local/config.yaml "
                        "里的 LLM 配置，或显式关闭 llm.intent_parser_enabled。"
                    ),
                    action="intent_parse_error",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    warnings=[str(exc)],
                ),
            )

        pending_payload = workspace.context.filters.pending_intent
        if pending_payload and isinstance(pending_payload, dict):
            pending_intent = intent_from_payload(pending_payload)
            if _is_cancel_message(message):
                workspace.context.filters.pending_intent = None
                return CoordinatorTurn(
                    intent=intent,
                    workspace=workspace,
                    memory_view=build_memory_view(
                        workspace,
                        purpose="planning",
                        session_memory=session_memory,
                    ),
                    early_response=_with_intent(
                        ChatResponse(
                            reply="已取消上一条待澄清指令。",
                            action="cancel_pending_intent",
                            workspace=WorkspaceSummary.from_workspace(workspace),
                        ),
                        intent,
                    ),
                )
            if _can_merge_pending_intent(pending_intent, intent, message):
                intent = merge_pending_intent(pending_intent, intent, message)
                workspace.context.filters.pending_intent = None

        if intent.ambiguous:
            workspace.context.filters.pending_intent = intent_to_payload(intent)
            questions = intent.clarification_questions or ["你希望我下一步具体执行什么？"]
            return CoordinatorTurn(
                intent=intent,
                workspace=workspace,
                memory_view=build_memory_view(
                    workspace,
                    purpose="planning",
                    session_memory=session_memory,
                ),
                early_response=ChatResponse(
                    reply="我先确认一下，避免跑偏：\n"
                    + "\n".join(f"- {question}" for question in questions),
                    action="clarify_intent",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    intent_confidence=intent.confidence,
                    ambiguous_intent=True,
                    ambiguity_reasons=intent.ambiguity_reasons,
                    clarification_questions=questions,
                    warnings=intent.ambiguity_reasons,
                ),
            )

        return CoordinatorTurn(
            intent=intent,
            task=_research_task(intent),
            workspace=workspace,
            memory_view=build_memory_view(
                workspace,
                purpose=_memory_purpose_for_intent(intent),
                session_memory=session_memory,
            ),
        )


def _memory_purpose_for_intent(intent: ChatIntent) -> str:
    if any(action in intent.actions for action in ["document", "storyline", "table"]):
        return "synthesis"
    if "autonomous_review" in intent.actions:
        return "review"
    return "planning"


def _research_task(intent: ChatIntent) -> ResearchTask:
    actions = list(dict.fromkeys(intent.actions))
    topic = intent.topic or "当前文献上下文"
    requires_freshness = any(
        token in topic.lower() for token in ("latest", "recent", "最新", "截至")
    )
    return ResearchTask(
        topic=topic,
        requested_actions=actions,
        year_min=intent.year_min,
        journals=intent.journals,
        requires_freshness=requires_freshness,
    )


def _is_cancel_message(message: str) -> bool:
    return message.lower().strip() in {"取消", "算了", "cancel", "never mind"}


def _can_merge_pending_intent(
    pending: ChatIntent,
    current: ChatIntent,
    message: str,
) -> bool:
    if not pending.ambiguous:
        return False
    if _is_cancel_message(message):
        return False
    if current.actions and not current.ambiguous:
        return False
    if current.topic and current.topic != message.strip():
        return True
    if current.topic and pending.actions:
        return True
    return bool(pending.actions and not current.actions and len(message.strip()) >= 3)


def _with_intent(response: ChatResponse, intent: ChatIntent) -> ChatResponse:
    response.intent_confidence = intent.confidence
    response.ambiguous_intent = intent.ambiguous
    response.ambiguity_reasons = list(intent.ambiguity_reasons)
    response.clarification_questions = list(intent.clarification_questions)
    return response
