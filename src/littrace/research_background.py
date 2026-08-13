from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.llm import chat_completion
from littrace.models import LiteratureWorkspace, TopicRetrievalPolicy


class ResearchBackgroundAssessment(BaseModel):
    accepted: bool
    background: str | None = None
    topic: str | None = None
    retrieval_policy: TopicRetrievalPolicy | None = None
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    review_source: str = "classifier"
    reviewer_model: str | None = None
    confidence: float | None = None


class ResearchBackgroundReviewSchema(BaseModel):
    accepted: bool
    reason: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    topic: str | None = None
    retrieval_policy: TopicRetrievalPolicy | None = None
    confidence: float | None = None


async def assess_research_background(
    text: str | None,
    config: LitTraceConfig | None = None,
    *,
    _repair_attempted: bool = False,
) -> ResearchBackgroundAssessment:
    assessment = _classify_research_background(text)
    if not assessment.accepted:
        return assessment
    llm_config = getattr(config, "llm", None)
    if llm_config is None or not getattr(llm_config, "enabled", False) or not getattr(
        llm_config, "api_key", None
    ):
        return assessment
    llm_review = await _review_research_background_with_llm(config, assessment.background or "")
    if llm_review is None:
        return assessment
    if not llm_review.accepted:
        if (
            llm_review.reason == "weak_research_signal"
            and not _repair_attempted
        ):
            repaired = await _repair_weak_research_background(config, assessment.background or "")
            if repaired:
                return await assess_research_background(
                    repaired, config, _repair_attempted=True
                )
        return _reject(
            llm_review.reason or "llm_review_rejected",
            llm_review.suggestions
            or [
                "请把研究背景写成一个明确、可持续追踪的科研主题，包含对象、问题、指标和应用场景。",
            ],
            review_source="llm",
            reviewer_model=config.llm.model,
            confidence=llm_review.confidence,
        )
    return ResearchBackgroundAssessment(
        accepted=True,
        background=assessment.background,
        topic=llm_review.topic or assessment.topic,
        retrieval_policy=llm_review.retrieval_policy,
        suggestions=llm_review.suggestions,
        review_source="llm",
        reviewer_model=config.llm.model,
        confidence=llm_review.confidence,
    )


async def _repair_weak_research_background(
    config: LitTraceConfig,
    background: str,
) -> str | None:
    reply = await chat_completion(
        config,
        "你负责把科研主题补全为可检索的研究背景。不得改变原研究方向或虚构材料、机制、应用；"
        "仅补足研究对象、问题、指标和时间范围。只返回补全后的中文研究背景，不要解释。",
        background,
        workspace=None,
    )
    if not reply.used_llm:
        return None
    repaired = " ".join(reply.text.split())
    return repaired if len(repaired) >= 18 else None


def _classify_research_background(text: str | None) -> ResearchBackgroundAssessment:
    background = " ".join(str(text or "").split())
    if not background:
        return _reject(
            "missing_background",
            [
                "请先描述你的研究背景，例如材料体系、器件/任务、关键指标、应用场景和时间范围。",
            ],
        )
    if len(background) < 18:
        return _reject(
            "background_too_short",
            ["研究背景太短，无法稳定生成检索主题。请补充研究对象、问题和目标指标。"],
        )
    lowered = background.lower()
    vague_markers = {
        "你好",
        "hello",
        "hi",
        "帮我下载",
        "下载论文",
        "找论文",
        "随便",
        "不知道",
        "测试",
    }
    if any(marker in lowered for marker in vague_markers) and len(background) < 40:
        return _reject(
            "not_research_topic",
            ["这更像普通指令或问候，不足以作为长期科研主题。"],
        )
    research_markers = [
        "research",
        "study",
        "paper",
        "literature",
        "sensor",
        "sensing",
        "material",
        "polymer",
        "composite",
        "device",
        "mechanism",
        "performance",
        "pressure",
        "flexible",
        "catalyst",
        "battery",
        "model",
        "algorithm",
        "biomedical",
        "研究",
        "文献",
        "论文",
        "材料",
        "器件",
        "传感",
        "性能",
        "机理",
        "柔性",
        "压力",
        "目标",
        "应用",
    ]
    if not any(marker in lowered for marker in research_markers):
        return _reject(
            "weak_research_signal",
            ["没有识别到明确科研对象或研究问题。"],
        )
    return ResearchBackgroundAssessment(
        accepted=True,
        background=background,
        topic=_topic_from_background(background),
    )


def workspace_has_research_background(workspace: LiteratureWorkspace) -> bool:
    filters = workspace.context.filters
    return bool((filters.research_background or "").strip())


def set_workspace_research_background(
    workspace: LiteratureWorkspace,
    background: str,
    *,
    topic: str | None = None,
    retrieval_policy: TopicRetrievalPolicy | None = None,
) -> LiteratureWorkspace:
    filters = workspace.context.filters
    filters.research_background = background
    filters.research_background_status = "accepted"
    filters.research_background_rejection_reason = None
    filters.research_background_set_at = datetime.now(UTC).isoformat()
    filters.topic = topic or _topic_from_background(background)
    filters.research_retrieval_policy = retrieval_policy
    if not filters.discipline:
        filters.discipline = "materials chemistry"
    return workspace


def mark_workspace_research_background_rejected(
    workspace: LiteratureWorkspace,
    reason: str,
) -> LiteratureWorkspace:
    filters = workspace.context.filters
    filters.research_background_status = "rejected"
    filters.research_background_rejection_reason = reason
    return workspace


def _reject(
    reason: str,
    suggestions: list[str],
    *,
    review_source: str = "classifier",
    reviewer_model: str | None = None,
    confidence: float | None = None,
) -> ResearchBackgroundAssessment:
    return ResearchBackgroundAssessment(
        accepted=False,
        reason=reason,
        suggestions=[
            *suggestions,
            "推荐格式：我研究的是<材料/系统>在<应用场景>中的<核心问题>，关注<指标/机理>，希望跟踪<近几年/特定来源>的论文。",
        ],
        review_source=review_source,
        reviewer_model=reviewer_model,
        confidence=confidence,
    )


async def _review_research_background_with_llm(
    config: LitTraceConfig,
    background: str,
) -> ResearchBackgroundReviewSchema | None:
    prompt = (
        "你是研究主题质量门禁审核器。请判断用户给出的内容是否适合作为一个可持续追踪的科研主题。"
        "标准：必须有明确研究对象、研究问题或目标、最好有应用场景或指标；纯问候、下载命令、闲聊、过泛主题都应拒绝。"
        "请只返回 JSON，字段为 accepted, reason, suggestions, topic, retrieval_policy, confidence。"
    )
    user_message = (
        "研究背景：\n"
        f"{background}\n\n"
        "如果适合长期科研跟踪，请 accepted=true；"
        "如果不适合，请 accepted=false 并给出简洁修改建议。\n\n"
        "accepted=true 时必须同时给出 retrieval_policy："
        "canonical_topic 是简洁、可检索的英文主题；query_variants 是 3-6 条英文检索式；"
        "required_concept_groups 只包含定义研究对象或核心机制的 2-4 个必须满足的同义词组，"
        "例如 [[\"flexible\"], [\"pressure\"], [\"piezoresistive\", \"resistive\"]]；"
        "excluded_concepts 仅列出该主题明确排除的机制或对象；boost_concepts 列出用于排序的可选术语。"
        "材料路线、微结构、性能指标、可靠性和应用场景通常属于 boost_concepts，除非用户明确要求它们不可缺少。"
        "用户已明确限定机制时，应在 excluded_concepts 中列出会导致方向偏移的竞争机制。"
        "不得凭空添加用户没有限定的材料、机制或应用；若机制未限定，不得将任何机制放入 required 或 excluded。"
    )
    reply = await chat_completion(
        config,
        prompt,
        user_message,
        workspace=None,
        json_mode=True,
    )
    if not reply.used_llm or not reply.text.strip():
        return None
    try:
        payload = json.loads(_extract_json(reply.text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(payload.get("suggestions"), str):
        payload["suggestions"] = [payload["suggestions"]]
    try:
        return ResearchBackgroundReviewSchema.model_validate(payload)
    except Exception:
        return None


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


def _topic_from_background(background: str, max_length: int = 120) -> str:
    topic = background.strip()
    if len(topic) <= max_length:
        return topic
    return topic[: max_length - 1].rstrip() + "…"
