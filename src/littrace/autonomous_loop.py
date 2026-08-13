from __future__ import annotations

import inspect
import json

from pydantic import BaseModel, Field, ValidationError

from littrace.citation_guard import guard_citations, remove_unsupported_sentences
from littrace.config import LitTraceConfig
from littrace.evaluation.harnesses import check_performance_cells, check_storyline_claims
from littrace.llm import chat_completion
from littrace.log import get_logger
from littrace.models import (
    ReviewFinding,
    ReviewLoopReport,
    ReviewRound,
    EvidenceSpan,
    LiteratureWorkspace,
    coerce_parsed,
)
from littrace.publication import evaluate_publication
from littrace.retrieval.rag_search import rag_hits_to_evidence_spans, search_workspace_rag
from littrace.research_writer import write_evidence_grounded_answer
from littrace.skill_runner import (
    build_quality_report_skill,
    build_storyline_skill,
    extract_tables_skill,
    parse_workspace_skill,
)

logger = get_logger("autonomous_loop")


# ---------------------------------------------------------------------------
# Schema for optional Reviewer output (Dimension 3: schema validation)
# ---------------------------------------------------------------------------


class ReviewerFindingItem(BaseModel):
    """A single finding from the bounded Reviewer."""

    reviewer: str = Field(description="Persona name, e.g. Method Reviewer")
    severity: str = Field(default="warning", description="error | warning | info")
    finding: str = Field(description="The critique text in Chinese")
    suggested_fix: str | None = None


class ReviewerFindingSchema(BaseModel):
    """Validated schema for the optional reviewer's structured output."""

    critiques: list[ReviewerFindingItem] = Field(default_factory=list)


# Keywords that promote a reviewer finding to error severity.
_ERROR_KEYWORDS = ("unsupported", "缺少证据", "无证据", "错误", "严重", "阻断", "cannot", "invalid")


async def run_review_loop(
    config: LitTraceConfig,
    objective: str,
    workspace: LiteratureWorkspace,
    max_rounds: int = 2,
    enable_optional_reviewer: bool = True,
    auto_replan: bool = False,
    rag_evidence: list[EvidenceSpan] | None = None,
) -> ReviewLoopReport:
    """Run mandatory quality gates plus an optional bounded, read-only reviewer."""

    if not workspace.context.active_papers:
        return ReviewLoopReport(
            objective=objective,
            final_answer="当前还没有文献上下文，无法启动审查流程。请先检索论文。",
            passed=False,
            score=0.0,
            replan_actions=["search_papers"],
            warnings=["empty_workspace"],
        )

    if rag_evidence is None:
        rag_evidence = await _rag_evidence_for_workspace(config, objective, workspace)
    draft = await _initial_draft(config, objective, workspace, rag_evidence=rag_evidence)
    rounds: list[ReviewRound] = []
    final_answer = draft
    final_score = 0.0
    passed = False
    aggregate_actions: list[str] = []
    aggregate_executed_actions: list[str] = []
    aggregate_warnings: list[str] = []

    for round_index in range(1, max_rounds + 1):
        critiques = _run_quality_gates(final_answer, workspace, config)
        if enable_optional_reviewer:
            critiques.extend(
                await _run_optional_reviewer(config, objective, final_answer, workspace)
            )
        revised = _revise_draft(final_answer, critiques, workspace)
        score = _round_score(critiques, workspace, config)
        replan_actions = _replan_actions(critiques, workspace)
        executed_actions: list[str] = []
        if auto_replan and replan_actions:
            workspace, executed_actions = await _execute_safe_replan_actions(
                config,
                workspace,
                replan_actions,
            )
            if executed_actions:
                followup_critiques = _run_quality_gates(revised, workspace, config)
                critiques.extend(
                    ReviewFinding(
                        reviewer="Readiness Gate",
                        severity="info",
                        finding=f"已执行自动重规划动作：{action}",
                        suggested_fix="基于更新后的 workspace 继续下一轮复核。",
                    )
                    for action in executed_actions
                )
                critiques.extend(followup_critiques)
                revised = _revise_draft(revised, followup_critiques, workspace)
                score = _round_score(critiques, workspace, config)
        passed = not any(item.severity == "error" for item in critiques)
        rounds.append(
            ReviewRound(
                round_index=round_index,
                writer_draft=final_answer,
                critiques=critiques,
                revised_draft=revised,
                passed=passed,
                score=score,
                replan_actions=replan_actions,
                executed_replan_actions=executed_actions,
            )
        )
        final_answer = revised
        final_score = score
        aggregate_actions.extend(
            action for action in replan_actions if action not in aggregate_actions
        )
        aggregate_executed_actions.extend(
            action for action in executed_actions if action not in aggregate_executed_actions
        )
        aggregate_warnings.extend(item.finding for item in critiques if item.severity != "info")
        if passed:
            break

    publication = evaluate_publication(workspace, config)
    final_citation_report = guard_citations(final_answer, workspace)
    release_blockers = list(publication.release_blockers)
    if not final_citation_report.passed:
        release_blockers.append(
            "Final autonomous answer has unsupported citation-bearing sentences."
        )
    release_ready = not release_blockers
    if not release_ready:
        final_answer = (
            "当前审查结果未通过最终发布检查，不能作为研究结论输出。"
            "请根据以下阻断项补充或核对证据：\n"
            + "\n".join(f"- {blocker}" for blocker in release_blockers[:5])
        )

    return ReviewLoopReport(
        objective=objective,
        final_answer=final_answer,
        rounds=rounds,
        passed=passed and release_ready,
        score=round(final_score, 3),
        replan_actions=aggregate_actions,
        executed_replan_actions=aggregate_executed_actions,
        warnings=[*aggregate_warnings, *final_citation_report.warnings],
        release_ready=release_ready,
        release_blockers=release_blockers,
    )


async def _initial_draft(
    config: LitTraceConfig,
    objective: str,
    workspace: LiteratureWorkspace,
    rag_evidence: list[EvidenceSpan] | None = None,
) -> str:
    write_fn = write_evidence_grounded_answer
    if "rag_evidence" in inspect.signature(write_fn).parameters:
        reply = await write_fn(
            config,
            objective,
            workspace,
            rag_evidence=rag_evidence,
        )
    else:
        reply = await write_fn(config, objective, workspace)
    if reply.used_llm and reply.text.strip():
        return reply.text
    raise RuntimeError(f"LLM unavailable for initial draft: {reply.error}")


async def _rag_evidence_for_workspace(
    config: LitTraceConfig,
    objective: str,
    workspace: LiteratureWorkspace,
) -> list[EvidenceSpan]:
    if not config.rag.enabled or config.rag.backend != "pgvector":
        return []
    try:
        result = await search_workspace_rag(
            config,
            workspace,
            objective,
            top_k=config.rag.top_k,
        )
    except Exception:
        return []
    if result is None:
        return []
    evidence = rag_hits_to_evidence_spans(result.profile, result.hits, query=objective)
    workspace.context.filters.rag_profile = result.profile.model_dump(mode="json")
    workspace.context.filters.rag_enabled = True
    workspace.context.filters.rag_backend = result.profile.backend
    workspace.context.filters.rag_last_query = objective
    workspace.context.filters.rag_last_hit_count = len(evidence)
    workspace.context.filters.rag_source_routes = list(result.profile.source_routes)
    return evidence


def _run_quality_gates(
    draft: str,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> list[ReviewFinding]:
    critiques: list[ReviewFinding] = []
    citation_report = guard_citations(draft, workspace)
    for sentence in citation_report.unsupported_sentences:
        critiques.append(
            ReviewFinding(
                reviewer="Citation Gate",
                severity="error",
                finding=f"句子缺少论文级锚点或访问链接：{sentence}",
                suggested_fix="删除该句，或补充 paper id、DOI、标题锚点、访问链接之一。",
            )
        )

    storyline_claims = build_storyline_skill(workspace)
    storyline_harness = check_storyline_claims(storyline_claims)
    for finding in storyline_harness.errors:
        critiques.append(
            ReviewFinding(
                reviewer="Storyline Gate",
                severity="error",
                finding=f"发展脉络证据不足：{finding}",
                suggested_fix="先解析全文，或把因果叙事降级为元数据趋势。",
            )
        )
    for warning in storyline_harness.warnings:
        critiques.append(
            ReviewFinding(
                reviewer="Storyline Gate",
                severity="warning",
                finding=warning,
                suggested_fix="增加跨论文证据，避免宽泛历史叙述。",
            )
        )

    table_harness = check_performance_cells(workspace.performance_cells)
    if "性能" in draft or "对比" in draft or "performance" in draft.lower():
        if not workspace.performance_cells:
            critiques.append(
                ReviewFinding(
                    reviewer="Table Gate",
                    severity="warning",
                    finding="草稿涉及性能/对比，但当前没有 performance cells。",
                    suggested_fix="运行 PDF/OCR 解析和表格抽取，或明确说明缺少可比数据。",
                )
            )
        for finding in table_harness.errors:
            critiques.append(
                ReviewFinding(
                    reviewer="Table Gate",
                    severity="error",
                    finding=f"性能指标缺少可追溯证据：{finding}",
                    suggested_fix="补充页码、表格编号或原文片段。",
                )
            )
        for warning in table_harness.warnings[:5]:
            critiques.append(
                ReviewFinding(
                    reviewer="Table Gate",
                    severity="warning",
                    finding=warning,
                    suggested_fix="补齐单位、方向或可比性说明。",
                )
            )

    quality = build_quality_report_skill(config, workspace)
    if quality.metrics.get("parsed_rate", 0.0) == 0 and workspace.context.active_papers:
        critiques.append(
            ReviewFinding(
                reviewer="Readiness Gate",
                severity="warning",
                finding="当前 active papers 尚未形成 parsed full text。",
                suggested_fix="优先执行 full-text resolve/download/parse，再生成最终学术叙述。",
            )
        )
    if not any(item.severity == "error" for item in critiques):
        critiques.append(
            ReviewFinding(
                reviewer="Quality Gates",
                severity="info",
                finding="未发现阻断性证据问题；可作为当前上下文下的审慎草稿。",
                suggested_fix="继续补充全文和结构化表格可提高结论密度。",
            )
        )
    return critiques


async def _run_optional_reviewer(
    config: LitTraceConfig,
    objective: str,
    draft: str,
    workspace: LiteratureWorkspace,
) -> list[ReviewFinding]:
    if not config.llm.enabled or not config.llm.api_key:
        return []
    payload = _reviewer_payload(objective, draft, workspace)
    system_prompt = (
        "You are one temporary, read-only academic reviewer. "
        "Return a JSON object with a 'critiques' array. "
        "Each critique must have: reviewer (string), severity (error|warning|info), "
        "finding (Chinese text), suggested_fix (Chinese text or null). "
        "Review method, evidence, and synthesis quality. Do not introduce new papers or facts. "
        "Do not request tools or mutate state. If evidence is insufficient, suggest a bounded replan action."
    )
    reply = await chat_completion(
        config,
        system_prompt,
        payload,
        workspace=None,
        json_mode=True,
    )
    if not reply.used_llm or not reply.text.strip():
        return []

    # --- Schema validation (Dimension 3) ---
    try:
        raw = json.loads(reply.text)
    except json.JSONDecodeError:
        logger.warning("reviewer_json_parse_failed", extra={"text_len": len(reply.text)})
        return []

    try:
        validated = ReviewerFindingSchema.model_validate(raw)
    except ValidationError as exc:
        logger.warning(
            "reviewer_schema_validation_failed",
            extra={"errors": exc.errors()[:3]},
        )
        return []

    critiques: list[ReviewFinding] = []
    for item in validated.critiques:
        severity = item.severity if item.severity in ("error", "warning", "info") else "warning"
        lowered = item.finding.lower()
        if any(kw in lowered for kw in _ERROR_KEYWORDS):
            severity = "error"
        critiques.append(
            ReviewFinding(
                reviewer=item.reviewer or "Optional Reviewer",
                severity=severity,
                finding=item.finding,
                suggested_fix=item.suggested_fix
                or "用当前 workspace 证据补足，或将相关结论降级为待验证假设。",
            )
        )
    return critiques[:8]


def _reviewer_payload(
    objective: str,
    draft: str,
    workspace: LiteratureWorkspace,
) -> str:
    lines = [f"Objective: {objective}", "", "Draft:", draft[:4000], "", "Workspace evidence:"]
    for paper_id in workspace.context.active_papers[:12]:
        paper = workspace.papers[paper_id]
        lines.append(
            f"- paper={paper.paper_id}; title={paper.title}; year={paper.year}; doi={paper.doi}"
        )
    if workspace.performance_cells:
        lines.append("Performance cells:")
        for cell in workspace.performance_cells[:12]:
            lines.append(
                f"- {cell.paper_id}: {cell.metric}={cell.value} {cell.unit or ''}; evidence={cell.evidence.snippet}"
            )
    if workspace.parsed_papers:
        lines.append("Parsed snippets:")
        for paper_id, parsed in list(workspace.parsed_papers.items())[:4]:
            parsed = coerce_parsed(parsed)
            for section in (parsed.sections or [])[:2]:
                if isinstance(section, dict):
                    lines.append(f"- {paper_id}: {str(section.get('text') or '')[:500]}")
    return "\n".join(lines)


def _revise_draft(
    draft: str,
    critiques: list[ReviewFinding],
    workspace: LiteratureWorkspace,
) -> str:
    citation_errors = [
        item
        for item in critiques
        if item.reviewer == "Citation Gate" and item.severity == "error"
    ]
    revised = draft
    if citation_errors:
        revised = remove_unsupported_sentences(revised, guard_citations(revised, workspace))

    warnings = [item for item in critiques if item.severity in {"warning", "error"}]
    if warnings:
        revised = revised.rstrip()
        revised += "\n\n质量门与可选审稿后的限制说明："
        for item in warnings[:6]:
            revised += f"\n- {item.reviewer}: {item.finding}"
    return revised


def _round_score(
    critiques: list[ReviewFinding],
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> float:
    quality = build_quality_report_skill(config, workspace)
    score = 0.62
    score += 0.12 * quality.metrics.get("citation_guard_pass", 0.0)
    score += 0.08 * quality.metrics.get("parsed_rate", 0.0)
    score += 0.06 if quality.metrics.get("comparison_matrix_count", 0.0) else 0.0
    score += 0.06 if quality.metrics.get("storyline_claim_count", 0.0) else 0.0
    score -= 0.12 * sum(1 for item in critiques if item.severity == "error")
    score -= 0.04 * sum(1 for item in critiques if item.severity == "warning")
    return max(0.0, min(0.98, score))


def _replan_actions(
    critiques: list[ReviewFinding],
    workspace: LiteratureWorkspace,
) -> list[str]:
    actions: list[str] = []
    if not workspace.full_text_reports and workspace.context.active_papers:
        actions.append("resolve_full_text")
    if not workspace.parsed_papers and workspace.context.active_papers:
        actions.append("parse_full_text_with_paddleocr")
    if any(item.reviewer == "Table Gate" for item in critiques):
        actions.append("extract_tables_and_structured_artifacts")
    if any(item.reviewer == "Citation Gate" for item in critiques):
        actions.append("rerun_citation_guard_after_revision")
    if any(item.reviewer == "Storyline Gate" for item in critiques):
        actions.append("rebuild_storyline_from_parsed_evidence")
    return actions


async def _execute_safe_replan_actions(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    actions: list[str],
) -> tuple[LiteratureWorkspace, list[str]]:
    executed: list[str] = []
    if "parse_full_text_with_paddleocr" in actions:
        workspace, report = await parse_workspace_skill(workspace, config)
        if report.get("parsed_count", 0):
            executed.append("parse_full_text_with_paddleocr")
    if "extract_tables_and_structured_artifacts" in actions:
        workspace, _ = await extract_tables_skill(workspace, config)
        executed.append("extract_tables_and_structured_artifacts")
    if "rebuild_storyline_from_parsed_evidence" in actions:
        claims = build_storyline_skill(workspace)
        workspace.context.filters.storyline_claim_count = len(claims)
        executed.append("rebuild_storyline_from_parsed_evidence")
    if "rerun_citation_guard_after_revision" in actions:
        executed.append("rerun_citation_guard_after_revision")
    return workspace, executed
