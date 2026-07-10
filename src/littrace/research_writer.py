from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.llm import LLMReply, chat_completion
from littrace.log import get_logger
from littrace.models import LiteratureWorkspace, coerce_parsed
from littrace.storyline import build_storyline_from_workspace

logger = get_logger("research_writer")


# ── Schema for LLM output validation (Dimension 3) ─────────────


class ResearchAnswerSchema(BaseModel):
    """Validate the structure of an evidence-grounded research answer.

    The LLM is asked to return JSON with 'answer', 'citations', and
    'unsupported_warnings' fields. This schema validates that structure
    before the text is passed downstream to the citation guard.
    """

    answer: str = Field(..., min_length=10, description="The research answer text")
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings used in the answer",
    )
    unsupported_warnings: list[str] = Field(
        default_factory=list,
        description="Points where evidence was insufficient",
    )


class StorylineNarrativeSchema(BaseModel):
    """Validate the structure of a storyline narrative LLM output."""

    narrative: str = Field(..., min_length=10, description="The storyline narrative text")
    claims_used: list[str] = Field(
        default_factory=list,
        description="Claim IDs referenced in the narrative",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings used in the narrative",
    )


_WRITER_SYSTEM_PROMPT = (
    "You are LitTrace Research Writer. Answer in Chinese. "
    "Use only the provided literature context, parsed evidence snippets, performance cells, "
    "and citation records. Do not invent papers, metrics, claims, or access links. "
    "When evidence is insufficient, say what is missing. "
    "Organize research storylines as: previous solution -> remaining limitation -> later response. "
    "End with a '引用与访问链接' section using the provided citation records.\n\n"
    "IMPORTANT: Return your response as a JSON object with this exact structure:\n"
    '{"answer": "your full research answer in Chinese", '
    '"citations": ["citation 1", "citation 2"], '
    '"unsupported_warnings": ["any points where evidence was insufficient"]}'
)

_STORYLINE_SYSTEM_PROMPT = (
    "You are LitTrace Storyline Writer. Write in Chinese. "
    "Use only the supplied claims and evidence. "
    "Never make broad field-history claims unless the evidence states them. "
    "Focus on: previous work solved what, what limitation remained, and how later work responded. "
    "End with citations and access links.\n\n"
    "IMPORTANT: Return your response as a JSON object with this exact structure:\n"
    '{"narrative": "your full storyline narrative in Chinese", '
    '"claims_used": ["claim text 1", "claim text 2"], '
    '"citations": ["citation 1", "citation 2"]}'
)


async def write_evidence_grounded_answer(
    config: LitTraceConfig,
    question: str,
    workspace: LiteratureWorkspace,
) -> LLMReply:
    if not workspace.context.active_papers:
        return LLMReply(
            text="当前还没有文献上下文。请先检索论文，再让我总结、比较或讲发展脉络。",
            used_llm=False,
            error="empty_workspace",
        )
    if _workspace_is_mock(workspace):
        return LLMReply(
            text=(
                "当前文献上下文来自 mock/开发样例，不是真实联网文献。"
                "我不能基于这些内容总结研究路线或生成结论。请先进行真实联网检索。"
            ),
            used_llm=False,
            error="mock_workspace",
        )
    if not _has_parsed_full_text(workspace):
        return LLMReply(
            text="当前没有可用的 PDF 全文解析证据。metadata/abstract fallback 已禁用，请先获取并解析全文。",
            used_llm=False,
            error="missing_full_text",
        )

    user_message = _writer_payload(question, workspace)
    llm_reply = await chat_completion(
        config,
        _WRITER_SYSTEM_PROMPT,
        user_message,
        workspace=None,
        json_mode=True,
    )
    return _validate_research_answer(llm_reply)


def fallback_evidence_answer(question: str, workspace: LiteratureWorkspace) -> str:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    if not papers:
        return "当前还没有文献上下文。请先检索论文。"
    if _workspace_is_mock(workspace):
        return (
            "当前文献上下文来自 mock/开发样例，不是真实联网文献。"
            "我不能基于这些内容总结研究路线或生成结论。请先进行真实联网检索。"
        )
    if not _has_parsed_full_text(workspace):
        return "当前没有可用的 PDF 全文解析证据。metadata/abstract fallback 已禁用，请先获取并解析全文。"
    lines = [f"我会基于当前 {len(papers)} 篇文献回答：{question}"]
    lines.append("当前回答只基于 PDF 全文解析片段和已抽取指标；证据不足处会保持保守。")
    lines.append("引用与访问链接：")
    for record in citation_records_for_papers(papers):
        lines.append(f"- {record.citation_text} {record.access_url}")
    return "\n".join(lines)


async def write_storyline_narrative(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> LLMReply:
    claims = build_storyline_from_workspace(workspace)
    if _workspace_is_mock(workspace):
        return LLMReply(
            text="当前文献上下文来自 mock/开发样例，不是真实联网文献。请先进行真实联网检索。",
            used_llm=False,
            error="mock_workspace",
        )
    if not claims:
        return LLMReply(
            text="当前证据不足以生成真实的发展脉络。建议先解析 PDF 全文。",
            used_llm=False,
            error="no_storyline_claims",
        )
    payload = _storyline_payload(workspace)
    llm_reply = await chat_completion(
        config,
        _STORYLINE_SYSTEM_PROMPT,
        payload,
        workspace=None,
        json_mode=True,
    )
    return _validate_storyline_narrative(llm_reply)


def _writer_payload(question: str, workspace: LiteratureWorkspace) -> str:
    lines = [f"User question: {question}", "", "Papers:"]
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    citations = citation_records_for_papers(papers)
    for paper in papers:
        lines.append(
            f"- id={paper.paper_id}; title={paper.title}; year={paper.year}; "
            f"journal={paper.journal}; publisher={paper.publisher}; doi={paper.doi}; "
            f"abstract={paper.abstract or ''}"
        )

    if workspace.parsed_papers:
        lines.append("")
        lines.append("Parsed evidence:")
        for paper_id, parsed in workspace.parsed_papers.items():
            parsed = coerce_parsed(parsed)
            for section in (parsed.sections or [])[:6]:
                if not isinstance(section, dict):
                    continue
                text = str(section.get("text") or "")[:700]
                name = str(section.get("name") or "section")
                lines.append(f"- paper={paper_id}; section={name}; text={text}")

    if workspace.performance_cells:
        lines.append("")
        lines.append("Performance cells:")
        for cell in workspace.performance_cells[:30]:
            lines.append(
                f"- paper={cell.paper_id}; metric={cell.metric}; value={cell.value}; "
                f"unit={cell.unit}; evidence={cell.evidence.snippet}"
            )

    lines.append("")
    lines.append("Citation records:")
    for record in citations:
        lines.append(
            f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}"
        )
    return "\n".join(lines)


def _workspace_is_mock(workspace: LiteratureWorkspace) -> bool:
    if getattr(workspace.context.filters, "search_mode", None) == "mock":
        return True
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    return any(".mock" in (paper.doi or "").lower() for paper in papers)


def _has_parsed_full_text(workspace: LiteratureWorkspace) -> bool:
    return any(bool(coerce_parsed(parsed).parsed) for parsed in workspace.parsed_papers.values())


def _storyline_payload(workspace: LiteratureWorkspace) -> str:
    lines = ["Storyline claims:"]
    for claim in build_storyline_from_workspace(workspace):
        lines.append(
            f"- type={claim.claim_type}; confidence={claim.confidence}; claim={claim.claim}"
        )
        for evidence in claim.evidence:
            lines.append(
                f"  evidence paper={evidence.paper_id}; section={evidence.section}; "
                f"page={evidence.page}; snippet={evidence.snippet}"
            )
    lines.append("")
    lines.append("Citation records:")
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    for record in citation_records_for_papers(papers):
        lines.append(
            f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}"
        )
    return "\n".join(lines)


# ── Schema validation helpers (Dimension 3) ───────────────────


def _validate_research_answer(reply: LLMReply) -> LLMReply:
    """Validate LLM output against ResearchAnswerSchema.

    If the LLM returned valid JSON, extract the 'answer' field as the reply text.
    If validation fails, fall back to using the raw content as-is (the citation
    guard downstream will catch unsupported claims).
    """
    if not reply.used_llm or not reply.text:
        return reply
    try:
        data = json.loads(reply.text)
        validated = ResearchAnswerSchema.model_validate(data)
        answer_text = validated.answer
        if validated.citations:
            answer_text += "\n\n引用与访问链接：\n" + "\n".join(
                f"- {c}" for c in validated.citations
            )
        return LLMReply(text=answer_text, used_llm=True)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "research_answer_schema_failed",
            extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]},
        )
        # Fallback: use raw text as-is; citation guard will catch issues
        return reply


def _validate_storyline_narrative(reply: LLMReply) -> LLMReply:
    """Validate LLM output against StorylineNarrativeSchema.

    If the LLM returned valid JSON, extract the 'narrative' field as the reply text.
    If validation fails, fall back to using the raw content as-is.
    """
    if not reply.used_llm or not reply.text:
        return reply
    try:
        data = json.loads(reply.text)
        validated = StorylineNarrativeSchema.model_validate(data)
        narrative_text = validated.narrative
        if validated.citations:
            narrative_text += "\n\n引用与访问链接：\n" + "\n".join(
                f"- {c}" for c in validated.citations
            )
        return LLMReply(text=narrative_text, used_llm=True)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "storyline_schema_failed",
            extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]},
        )
        return reply
