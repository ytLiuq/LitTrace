from __future__ import annotations

import json
from hashlib import sha256

from pydantic import BaseModel, Field, ValidationError, field_validator

from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.llm import LLMReply, chat_completion
from littrace.log import get_logger
from littrace.evidence.claims import (
    record_claim_verification,
    register_evidence,
    verify_structured_claim,
)
from littrace.evidence.tables import extract_performance_cells
from littrace.models import Claim, ClaimKind, EvidenceSpan, LiteratureWorkspace, coerce_parsed
from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.publication import evaluate_publication

logger = get_logger("research_writer")


# ── Schema for LLM output validation (Dimension 3) ─────────────


class ResearchAnswerSchema(BaseModel):
    """Validate the structure of an evidence-grounded research answer.

    The LLM is asked to return JSON with 'answer', 'citations', and
    'unsupported_warnings' fields. This schema validates that structure
    before the text is passed downstream to the citation guard.
    """

    answer: str = Field(default="", description="Optional non-conclusion preface")
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings used in the answer",
    )
    claims: list["AnswerClaimSchema"] = Field(
        default_factory=list,
        description="Atomic answer claims tied to supplied evidence IDs.",
    )
    unsupported_warnings: list[str] = Field(
        default_factory=list,
        description="Points where evidence was insufficient",
    )


class StorylineNarrativeSchema(BaseModel):
    """Validate the structure of a storyline narrative LLM output."""

    narrative: str = Field(default="", description="Optional non-conclusion preface")
    claims_used: list[str] = Field(
        default_factory=list,
        description="Claim IDs referenced in the narrative",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings used in the narrative",
    )
    claims: list["AnswerClaimSchema"] = Field(
        default_factory=list,
        description="Atomic storyline claims tied to supplied evidence IDs.",
    )


class AnswerClaimSchema(BaseModel):
    text: str = Field(..., min_length=5)
    evidence_ids: list[str] = Field(min_length=1)
    support_quotes: dict[str, str] = Field(default_factory=dict)
    claim_kind: ClaimKind = ClaimKind.QUALITATIVE
    metric: str | None = None
    expected_value: float | None = None
    expected_unit: str | None = None
    expected_value_min: float | None = None
    expected_value_max: float | None = None
    expected_uncertainty: float | None = None
    requires_corroboration: bool = False
    requires_freshness: bool = False

    @field_validator("claim_kind", mode="before")
    @classmethod
    def _normalize_claim_kind(cls, value: object) -> object:
        # OpenAI-compatible models commonly emit this natural-language alias.
        if isinstance(value, str) and value.strip().lower() == "quantitative":
            return ClaimKind.NUMERIC
        return value


_WRITER_SYSTEM_PROMPT = (
    "You are LitTrace Research Writer. Answer in Chinese. "
    "Use only the provided evidence IDs, literature context, parsed evidence snippets, performance cells, "
    "and citation records. Do not invent papers, metrics, claims, evidence IDs, or access links. "
    "When evidence is insufficient, say what is missing. "
    "Organize research storylines as: previous solution -> remaining limitation -> later response. "
    "End with a '引用与访问链接' section using the provided citation records.\n\n"
    "IMPORTANT: Return your response as a JSON object with this exact structure:\n"
    '{"answer": "your full research answer in Chinese", '
    '"citations": ["citation 1", "citation 2"], '
    '"claims": [{"text": "atomic claim", "evidence_ids": ["ev-1"], '
    '"support_quotes": {"ev-1": "exact text from supplied evidence"}, '
    '"claim_kind": "qualitative", "metric": null}], '
    '"unsupported_warnings": ["any points where evidence was insufficient"]}'
)

_STORYLINE_SYSTEM_PROMPT = (
    "You are LitTrace Storyline Writer. Write in Chinese. "
    "Use only the supplied evidence IDs, claims, and evidence. "
    "Never make broad field-history claims unless the evidence states them. "
    "Focus on: previous work solved what, what limitation remained, and how later work responded. "
    "End with citations and access links.\n\n"
    "IMPORTANT: Return your response as a JSON object with this exact structure:\n"
    '{"narrative": "your full storyline narrative in Chinese", '
    '"claims_used": ["claim text 1", "claim text 2"], '
    '"citations": ["citation 1", "citation 2"], '
    '"claims": [{"text": "atomic claim", "evidence_ids": ["ev-1"], '
    '"support_quotes": {"ev-1": "exact text from supplied evidence"}, '
    '"claim_kind": "qualitative", "metric": null}]}'
)


async def write_evidence_grounded_answer(
    config: LitTraceConfig,
    question: str,
    workspace: LiteratureWorkspace,
    rag_evidence: list[EvidenceSpan] | None = None,
) -> LLMReply:
    selected_papers = _selected_papers_for_answer(workspace, rag_evidence)
    if not selected_papers:
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
    if not _has_parsed_full_text(workspace) and not rag_evidence:
        return LLMReply(
            text="当前没有可用的 PDF 全文解析证据。metadata/abstract fallback 已禁用，请先获取并解析全文。",
            used_llm=False,
            error="missing_full_text",
        )
    await _ensure_numeric_evidence(config, question, workspace)
    user_message, evidence_registry = _writer_payload(
        question,
        workspace,
        rag_evidence=rag_evidence,
    )
    return await _validated_research_completion(
        config, _WRITER_SYSTEM_PROMPT, user_message, evidence_registry, workspace
    )


def fallback_evidence_answer(
    question: str,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig | None = None,
) -> str:
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
    release_block = _release_blocked_reply(config or LitTraceConfig(), workspace)
    if release_block is not None:
        return release_block.text
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
    release_block = _release_blocked_reply(config, workspace)
    if release_block is not None:
        return release_block
    payload, evidence_registry = _storyline_payload(workspace)
    return await _validated_storyline_completion(
        config, _STORYLINE_SYSTEM_PROMPT, payload, evidence_registry, workspace
    )


def _writer_payload(
    question: str,
    workspace: LiteratureWorkspace,
    rag_evidence: list[EvidenceSpan] | None = None,
) -> tuple[str, dict[str, EvidenceSpan]]:
    lines = [f"User question: {question}", "", "Papers:"]
    evidence_registry: dict[str, EvidenceSpan] = {}
    selected_papers = _selected_papers_for_answer(workspace, rag_evidence)
    papers = [workspace.papers[paper_id] for paper_id in selected_papers if paper_id in workspace.papers]
    citations = citation_records_for_papers(papers)
    for paper in papers:
        lines.append(
            f"- id={paper.paper_id}; title={paper.title}; year={paper.year}; "
            f"journal={paper.journal}; publisher={paper.publisher}; doi={paper.doi}"
        )

    # When retrieval found question-specific source spans, do not dilute them
    # with a second copy of the leading sections from every paper. It both
    # harms evidence selection and can push the structured answer past the LLM
    # request timeout.
    if workspace.parsed_papers and not rag_evidence:
        lines.append("")
        lines.append("Parsed evidence:")
        for paper_id, parsed in workspace.parsed_papers.items():
            parsed = coerce_parsed(parsed)
            for section in (parsed.sections or [])[:6]:
                if not isinstance(section, dict):
                    continue
                text = str(section.get("text") or "")[:700]
                name = str(section.get("name") or "section")
                evidence_id = _section_evidence_id(paper_id, name, text)
                provenance = section.get("evidence")
                provenance = provenance if isinstance(provenance, dict) else {}
                evidence_registry[evidence_id] = EvidenceSpan(
                    paper_id=paper_id,
                    evidence_id=evidence_id,
                    section=name,
                    page=provenance.get("page"),
                    snippet=text or name,
                    parser=provenance.get("parser"),
                    parser_version=provenance.get("parser_version"),
                    source_record_id=provenance.get("source_record_id"),
                    content_hash=provenance.get("content_hash"),
                    captured_at=provenance.get("captured_at"),
                )
                lines.append(
                    f"- evidence_id={evidence_id}; paper={paper_id}; section={name}; text={text}"
                )

    if workspace.performance_cells:
        lines.append("")
        lines.append("Performance cells:")
        for cell in workspace.performance_cells[:30]:
            evidence_id = _performance_evidence_id(
                cell.paper_id, cell.metric, cell.value, cell.unit
            )
            evidence_registry[evidence_id] = cell.evidence.model_copy(
                update={
                    "evidence_id": evidence_id,
                    "column_label": cell.metric,
                    "observed_value": cell.value,
                    "observed_unit": cell.unit,
                    "observed_value_min": cell.value_min,
                    "observed_value_max": cell.value_max,
                    "observed_uncertainty": cell.uncertainty,
                }
            )
            lines.append(
                f"- evidence_id={evidence_id}; paper={cell.paper_id}; metric={cell.metric}; value={cell.value}; "
                f"unit={cell.unit}; evidence={cell.evidence.snippet}"
            )

    if rag_evidence:
        lines.append("")
        lines.append("RAG evidence:")
        for index, span in enumerate(rag_evidence, start=1):
            if not span.evidence_id:
                continue
            # RAG IDs include both a profile and chunk digest. They are durable
            # internally but unnecessarily error-prone for an LLM to copy.
            # Use a compact, content-addressed alias in the writer contract and
            # retain the original identifier in source_record_id for audit.
            alias = f"rag-{(span.content_hash or str(index))[:16]}"
            registered_span = span.model_copy(update={"evidence_id": alias})
            evidence_registry[alias] = registered_span
            location = []
            if registered_span.section:
                location.append(f"section={registered_span.section}")
            if registered_span.page is not None:
                location.append(f"page={registered_span.page}")
            if registered_span.table_id:
                location.append(f"table_id={registered_span.table_id}")
            loc_text = "; ".join(location)
            location_part = f"; {loc_text}" if loc_text else ""
            lines.append(
                f"- evidence_id={alias}; paper={registered_span.paper_id}{location_part}; "
                f"text={registered_span.snippet or ''}"
            )

    lines.append("")
    lines.append(
    "Claim requirements: each claim must be atomic. Every evidence_id must have a "
        "support_quotes entry copied verbatim from that evidence text. A Chinese translation "
        "is allowed only when the quote is verbatim. Do not use causal or comparative wording "
        "unless the cited quote explicitly states it."
    )
    lines.append(
        "Coverage requirements: when the question asks for multiple samples or conditions, "
        "cover every requested sample/condition that has supplied evidence. Do not infer a "
        "ranking for a sample whose corresponding metric is absent or marked not given."
    )
    if workspace.performance_cells:
        lines.append(
            "For every numeric claim, cite a Performance cells evidence_id (not a generic RAG "
            "evidence_id), set claim_kind to numeric, and provide metric, expected_value, and "
            "expected_unit exactly as registered."
        )
    lines.append("")
    lines.append("Citation records:")
    for record in citations:
        lines.append(
            f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}"
        )
    register_evidence(workspace, list(evidence_registry.values()))
    return "\n".join(lines), evidence_registry


def _selected_papers_for_answer(
    workspace: LiteratureWorkspace,
    rag_evidence: list[EvidenceSpan] | None = None,
) -> list[str]:
    selected = list(workspace.context.active_papers)
    for span in rag_evidence or []:
        if span.paper_id not in selected:
            selected.append(span.paper_id)
    return selected


def _workspace_is_mock(workspace: LiteratureWorkspace) -> bool:
    if getattr(workspace.context.filters, "search_mode", None) == "mock":
        return True
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    return any(".mock" in (paper.doi or "").lower() for paper in papers)


def _has_parsed_full_text(workspace: LiteratureWorkspace) -> bool:
    return any(bool(coerce_parsed(parsed).parsed) for parsed in workspace.parsed_papers.values())


async def _ensure_numeric_evidence(
    config: LitTraceConfig,
    question: str,
    workspace: LiteratureWorkspace,
) -> None:
    """Populate traceable metric cells before an answer that can require numbers."""

    if workspace.performance_cells or not config.llm.enabled:
        return
    normalized = question.lower()
    numeric_terms = (
        "性能", "参数", "数值", "指标", "模量", "刚性", "弹性", "强度", "灵敏度",
        "performance", "metric", "modulus", "stiff", "elastic", "sensitivity", "strength",
    )
    if not any(term in normalized for term in numeric_terms):
        return
    try:
        _, report = await extract_performance_cells(workspace, config)
        logger.info(
            "writer_numeric_evidence_ready",
            extra={"cell_count": len(workspace.performance_cells), "score": report.score},
        )
    except Exception as exc:
        # The writer can still return quote-bound qualitative claims. Numeric
        # claims remain draft-only unless a later extraction supplies values.
        logger.warning(
            "writer_numeric_evidence_unavailable",
            extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]},
        )


def _release_blocked_reply(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> LLMReply | None:
    report = evaluate_publication(workspace, config)
    if report.release_ready:
        return None
    blockers = "; ".join(report.release_blockers[:2])
    return LLMReply(
        text=(
            f"当前证据尚未通过 Claim 发布门禁，不能生成最终研究结论。请先补充或核对证据：{blockers}"
        ),
        used_llm=False,
        error="claim_release_blocked",
    )


def _storyline_payload(workspace: LiteratureWorkspace) -> tuple[str, dict[str, EvidenceSpan]]:
    lines = ["Storyline claims:"]
    evidence_registry: dict[str, EvidenceSpan] = {}
    for claim in build_storyline_from_workspace(workspace):
        lines.append(
            f"- type={claim.claim_type}; confidence={claim.confidence}; claim={claim.claim}"
        )
        for evidence in claim.evidence:
            evidence_id = _span_evidence_id(evidence)
            evidence_registry[evidence_id] = evidence
            lines.append(
                f"  evidence_id={evidence_id}; paper={evidence.paper_id}; section={evidence.section}; "
                f"page={evidence.page}; snippet={evidence.snippet}"
            )
    lines.append("")
    lines.append("Citation records:")
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    for record in citation_records_for_papers(papers):
        lines.append(
            f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}"
        )
    register_evidence(workspace, list(evidence_registry.values()))
    return "\n".join(lines), evidence_registry


# ── Schema validation helpers (Dimension 3) ───────────────────


async def _validated_research_completion(
    config: LitTraceConfig,
    system_prompt: str,
    user_message: str,
    evidence_registry: dict[str, EvidenceSpan],
    workspace: LiteratureWorkspace,
) -> LLMReply:
    return await _validated_json_completion(
        config,
        system_prompt,
        user_message,
        lambda reply: _parse_research_answer(reply, evidence_registry, workspace),
        "research answer",
    )


async def _validated_storyline_completion(
    config: LitTraceConfig,
    system_prompt: str,
    user_message: str,
    evidence_registry: dict[str, EvidenceSpan],
    workspace: LiteratureWorkspace,
) -> LLMReply:
    return await _validated_json_completion(
        config,
        system_prompt,
        user_message,
        lambda reply: _parse_storyline_narrative(reply, evidence_registry, workspace),
        "storyline narrative",
    )


async def _validated_json_completion(
    config: LitTraceConfig,
    system_prompt: str,
    user_message: str,
    parser,
    response_kind: str,
) -> LLMReply:
    """Retry malformed structured output once, then refuse rather than leak raw text."""

    reply = await chat_completion(
        config,
        system_prompt,
        user_message,
        workspace=None,
        json_mode=True,
    )
    parsed = parser(reply)
    if parsed is not None:
        return parsed
    if not reply.used_llm:
        return reply

    logger.warning("llm_schema_retry", extra={"response_kind": response_kind})
    retry = await chat_completion(
        config,
        (
            f"{system_prompt}\n\nYour previous response did not satisfy the required JSON schema. "
            "Return only one valid JSON object matching the requested schema."
        ),
        user_message,
        workspace=None,
        json_mode=True,
    )
    parsed = parser(retry)
    if parsed is not None:
        return parsed
    if not retry.used_llm:
        return retry
    logger.warning("llm_schema_refused", extra={"response_kind": response_kind})
    return LLMReply(
        text="模型未能生成可验证的结构化回答；为避免输出未经校验的结论，本次回答已拒绝。",
        used_llm=False,
        error=f"invalid_{response_kind.replace(' ', '_')}_schema",
    )


def _parse_research_answer(
    reply: LLMReply,
    evidence_registry: dict[str, EvidenceSpan],
    workspace: LiteratureWorkspace,
) -> LLMReply | None:
    if not reply.used_llm or not reply.text:
        return reply
    try:
        data = json.loads(reply.text)
        validated = ResearchAnswerSchema.model_validate(data)
        if not validated.claims:
            return _draft_preface_reply(validated.answer, workspace)
        return LLMReply(
            text=_render_verified_claims(validated.claims, evidence_registry, workspace),
            used_llm=True,
        )
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.warning(
            "research_answer_schema_failed",
            extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]},
        )
        return None


def _draft_preface_reply(answer: str, workspace: LiteratureWorkspace) -> LLMReply:
    """Return a clearly labelled draft when no atomic claim was supplied.

    A free-form model response must never become a released conclusion, but
    rejecting an otherwise useful request outright makes the chat surface
    brittle. Keep it as draft context and attach deterministic citations.
    """

    papers = [
        workspace.papers[paper_id]
        for paper_id in workspace.context.active_papers
        if paper_id in workspace.papers
    ]
    records = citation_records_for_papers(papers)
    lines = [
        "以下内容为草稿性说明，尚未生成可发布的原子 Claim：",
        answer.strip() or "当前模型未提供结构化 Claim；请基于下列文献继续核对。",
        "",
        "引用与访问链接：",
    ]
    lines.extend(f"- {record.citation_text} {record.access_url}" for record in records)
    return LLMReply(text="\n".join(lines), used_llm=True, error="draft_without_claims")


def _parse_storyline_narrative(
    reply: LLMReply,
    evidence_registry: dict[str, EvidenceSpan],
    workspace: LiteratureWorkspace,
) -> LLMReply | None:
    if not reply.used_llm or not reply.text:
        return reply
    try:
        data = json.loads(reply.text)
        validated = StorylineNarrativeSchema.model_validate(data)
        return LLMReply(
            text=_render_verified_claims(validated.claims, evidence_registry, workspace),
            used_llm=True,
        )
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.warning(
            "storyline_schema_failed",
            extra={"error": f"{exc.__class__.__name__}: {exc}"[:200]},
        )
        return None


def _validate_claim_evidence_ids(
    claims: list[AnswerClaimSchema],
    evidence_registry: dict[str, EvidenceSpan],
) -> None:
    if not claims:
        raise ValueError("Structured output must include at least one evidence-linked claim.")
    referenced = {evidence_id for claim in claims for evidence_id in claim.evidence_ids}
    unknown = referenced - evidence_registry.keys()
    if unknown:
        raise ValueError(f"Unknown evidence IDs: {', '.join(sorted(unknown))}")


def _render_verified_claims(
    claims: list[AnswerClaimSchema],
    evidence_registry: dict[str, EvidenceSpan],
    workspace: LiteratureWorkspace,
) -> str:
    """Render only verified model claims and local citation records.

    The model cannot control the final narrative or bibliography: it selects
    pre-registered evidence IDs, and this function validates and renders them.
    """

    _validate_claim_evidence_ids(claims, evidence_registry)
    selected_paper_ids: set[str] = set()
    lines = ["以下结论已逐条绑定至当前会话中的可定位证据："]
    withheld_claims = 0
    for claim in claims:
        evidence = [evidence_registry[evidence_id] for evidence_id in claim.evidence_ids]
        structured_claim = Claim(
            text=claim.text,
            claim_kind=claim.claim_kind,
            evidence_ids=claim.evidence_ids,
            support_quotes=_canonical_support_quotes(claim, evidence_registry),
            metric=claim.metric,
            expected_value=claim.expected_value,
            expected_unit=claim.expected_unit,
            expected_value_min=claim.expected_value_min,
            expected_value_max=claim.expected_value_max,
            expected_uncertainty=claim.expected_uncertainty,
            requires_corroboration=claim.requires_corroboration,
            requires_freshness=claim.requires_freshness,
            retrieval_cutoff_at=getattr(workspace.context.filters, "search_completed_at", None),
            claim_origin="llm_quote_bound",
        )
        verification = verify_structured_claim(structured_claim, evidence_registry)
        if not verification.publishable:
            requirements = "; ".join(verification.missing_requirements) or verification.status
            record_claim_verification(workspace, structured_claim, verification)
            selected_paper_ids.update(span.paper_id for span in evidence)
            withheld_claims += 1
            lines.append(f"- [草稿，未发布] {claim.text}（待核对：{requirements}）")
            continue
        record_claim_verification(workspace, structured_claim, verification)
        anchors = ", ".join(
            f"{evidence_id} / {evidence_registry[evidence_id].paper_id}"
            for evidence_id in claim.evidence_ids
        )
        lines.append(f"- {claim.text} [证据: {anchors}]")
        selected_paper_ids.update(span.paper_id for span in evidence)

    records = citation_records_for_papers(
        [
            workspace.papers[paper_id]
            for paper_id in sorted(selected_paper_ids)
            if paper_id in workspace.papers
        ]
    )
    if not records:
        raise ValueError("Evidence-linked claims did not resolve to local citation records.")
    if withheld_claims:
        lines[0] = "以下内容含草稿性断言，未通过发布门禁的结论已明确标注，不应作为最终研究结论："
    lines.extend(["", "引用与访问链接："])
    lines.extend(f"- {record.citation_text} {record.access_url}" for record in records)
    return "\n".join(lines)


def _canonical_support_quotes(
    claim: AnswerClaimSchema,
    evidence_registry: dict[str, EvidenceSpan],
) -> dict[str, str]:
    """Anchor numeric claims to the immutable metric evidence text.

    Numeric evidence is already value-, unit-, and metric-validated below.
    Models frequently return a shortened table fragment as their quote even
    when they selected the correct metric-cell ID; retaining that fragment
    would fail a purely textual substring check. Persist the registered source
    snippet instead, while leaving every non-numeric quote model-supplied.
    """

    quotes = dict(claim.support_quotes)
    if claim.claim_kind != ClaimKind.NUMERIC:
        return quotes
    for evidence_id in claim.evidence_ids:
        span = evidence_registry[evidence_id]
        if isinstance(span.observed_value, int | float) and span.snippet:
            quotes[evidence_id] = span.snippet
    return quotes


def _section_evidence_id(paper_id: str, section: str, text: str) -> str:
    digest = sha256(f"{paper_id}\0{section}\0{text}".encode()).hexdigest()[:12]
    return f"section:{paper_id}:{section}:{digest}"


def _performance_evidence_id(
    paper_id: str,
    metric: str,
    value: float | str,
    unit: str | None,
) -> str:
    return f"metric:{paper_id}:{metric}:{value}:{unit or ''}"


def _span_evidence_id(evidence) -> str:
    return _section_evidence_id(
        evidence.paper_id,
        evidence.section or evidence.table_id or "evidence",
        evidence.snippet or "",
    )
