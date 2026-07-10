from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from littrace.citation_guard import guard_citations, remove_unsupported_sentences
from littrace.config import LitTraceConfig
from littrace.harnesses import check_performance_cells, check_storyline_claims
from littrace.llm import chat_completion
from littrace.log import get_logger
from littrace.models import (
    AgentCritique,
    AgentDebateRound,
    AutonomousResearchLoopReport,
    LiteratureWorkspace,
    coerce_parsed,
)
from littrace.quality_report import build_quality_report
from littrace.research_writer import write_evidence_grounded_answer
from littrace.storyline import build_storyline_from_workspace
from littrace.parsing import parse_workspace_papers
from littrace.tables import extract_performance_cells

logger = get_logger("autonomous_loop")


# ---------------------------------------------------------------------------
# Schema for LLM debate critique output (Dimension 3: Schema validation)
# ---------------------------------------------------------------------------


class DebateCritiqueItem(BaseModel):
    """A single critique from one LLM debate persona."""

    reviewer: str = Field(description="Persona name, e.g. Method Reviewer")
    severity: str = Field(default="warning", description="error | warning | info")
    finding: str = Field(description="The critique text in Chinese")
    suggested_fix: str | None = None


class DebateCritiqueSchema(BaseModel):
    """Validated schema for _smart_debate_critiques LLM output."""

    critiques: list[DebateCritiqueItem] = Field(default_factory=list)


# Keywords that indicate error-level severity (kept for backward compat)
_ERROR_KEYWORDS = ("unsupported", "缺少证据", "无证据", "错误", "严重", "阻断", "cannot", "invalid")


async def run_autonomous_research_loop(
    config: LitTraceConfig,
    objective: str,
    workspace: LiteratureWorkspace,
    max_rounds: int = 2,
    enable_smart_debate: bool = True,
    auto_replan: bool = False,
) -> AutonomousResearchLoopReport:
    """Run a bounded writer/reviewer/reviser/replanner loop over current evidence."""

    if not workspace.context.active_papers:
        return AutonomousResearchLoopReport(
            objective=objective,
            final_answer="当前还没有文献上下文，无法启动多 agent 修订循环。请先检索论文。",
            passed=False,
            score=0.0,
            replan_actions=["search_papers"],
            warnings=["empty_workspace"],
        )

    draft = await _initial_draft(config, objective, workspace)
    rounds: list[AgentDebateRound] = []
    final_answer = draft
    final_score = 0.0
    passed = False
    aggregate_actions: list[str] = []
    aggregate_executed_actions: list[str] = []
    aggregate_warnings: list[str] = []

    for round_index in range(1, max_rounds + 1):
        critiques = _review_draft(final_answer, workspace, config)
        if enable_smart_debate:
            critiques.extend(
                await _smart_debate_critiques(config, objective, final_answer, workspace)
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
                followup_critiques = _review_draft(revised, workspace, config)
                critiques.extend(
                    AgentCritique(
                        reviewer="Replanning Agent",
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
            AgentDebateRound(
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

    return AutonomousResearchLoopReport(
        objective=objective,
        final_answer=final_answer,
        rounds=rounds,
        passed=passed,
        score=round(final_score, 3),
        replan_actions=aggregate_actions,
        executed_replan_actions=aggregate_executed_actions,
        warnings=aggregate_warnings,
    )


async def _initial_draft(
    config: LitTraceConfig,
    objective: str,
    workspace: LiteratureWorkspace,
) -> str:
    reply = await write_evidence_grounded_answer(config, objective, workspace)
    if reply.used_llm and reply.text.strip():
        return reply.text
    raise RuntimeError(f"LLM unavailable for initial draft: {reply.error}")


def _review_draft(
    draft: str,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> list[AgentCritique]:
    critiques: list[AgentCritique] = []
    citation_report = guard_citations(draft, workspace)
    for sentence in citation_report.unsupported_sentences:
        critiques.append(
            AgentCritique(
                reviewer="Citation Auditor",
                severity="error",
                finding=f"句子缺少论文级锚点或访问链接：{sentence}",
                suggested_fix="删除该句，或补充 paper id、DOI、标题锚点、访问链接之一。",
            )
        )

    storyline_claims = build_storyline_from_workspace(workspace)
    storyline_harness = check_storyline_claims(storyline_claims)
    for finding in storyline_harness.errors:
        critiques.append(
            AgentCritique(
                reviewer="Storyline Skeptic",
                severity="error",
                finding=f"发展脉络证据不足：{finding}",
                suggested_fix="先解析全文，或把因果叙事降级为元数据趋势。",
            )
        )
    for warning in storyline_harness.warnings:
        critiques.append(
            AgentCritique(
                reviewer="Storyline Skeptic",
                severity="warning",
                finding=warning,
                suggested_fix="增加跨论文证据，避免宽泛历史叙述。",
            )
        )

    table_harness = check_performance_cells(workspace.performance_cells)
    if "性能" in draft or "对比" in draft or "performance" in draft.lower():
        if not workspace.performance_cells:
            critiques.append(
                AgentCritique(
                    reviewer="Table Auditor",
                    severity="warning",
                    finding="草稿涉及性能/对比，但当前没有 performance cells。",
                    suggested_fix="运行 PDF/OCR 解析和表格抽取，或明确说明缺少可比数据。",
                )
            )
        for finding in table_harness.errors:
            critiques.append(
                AgentCritique(
                    reviewer="Table Auditor",
                    severity="error",
                    finding=f"性能指标缺少可追溯证据：{finding}",
                    suggested_fix="补充页码、表格编号或原文片段。",
                )
            )
        for warning in table_harness.warnings[:5]:
            critiques.append(
                AgentCritique(
                    reviewer="Table Auditor",
                    severity="warning",
                    finding=warning,
                    suggested_fix="补齐单位、方向或可比性说明。",
                )
            )

    quality = build_quality_report(config, workspace)
    if quality.metrics.get("parsed_rate", 0.0) == 0 and workspace.context.active_papers:
        critiques.append(
            AgentCritique(
                reviewer="Replanning Agent",
                severity="warning",
                finding="当前 active papers 尚未形成 parsed full text。",
                suggested_fix="优先执行 full-text resolve/download/parse，再生成最终学术叙述。",
            )
        )
    if not any(item.severity == "error" for item in critiques):
        critiques.append(
            AgentCritique(
                reviewer="Lead Reviewer",
                severity="info",
                finding="未发现阻断性证据问题；可作为当前上下文下的审慎草稿。",
                suggested_fix="继续补充全文和结构化表格可提高结论密度。",
            )
        )
    return critiques


async def _smart_debate_critiques(
    config: LitTraceConfig,
    objective: str,
    draft: str,
    workspace: LiteratureWorkspace,
) -> list[AgentCritique]:
    if not config.llm.enabled or not config.llm.api_key:
        return []
    payload = _smart_debate_payload(objective, draft, workspace)
    system_prompt = (
        "You are a skeptical academic multi-agent review council. "
        "Return a JSON object with a 'critiques' array. "
        "Each critique must have: reviewer (string), severity (error|warning|info), "
        "finding (Chinese text), suggested_fix (Chinese text or null). "
        "Use the personas: Method Reviewer, Evidence Reviewer, Synthesis Reviewer. "
        "Do not introduce new papers or facts. If evidence is insufficient, ask for replan actions."
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
        logger.warning("debate_json_parse_failed", extra={"text_len": len(reply.text)})
        return _parse_debate_as_text(reply.text)

    try:
        validated = DebateCritiqueSchema.model_validate(raw)
    except ValidationError as exc:
        logger.warning(
            "debate_schema_validation_failed",
            extra={"errors": exc.errors()[:3]},
        )
        return _parse_debate_as_text(reply.text)

    critiques: list[AgentCritique] = []
    for item in validated.critiques:
        severity = item.severity if item.severity in ("error", "warning", "info") else "warning"
        lowered = item.finding.lower()
        if any(kw in lowered for kw in _ERROR_KEYWORDS):
            severity = "error"
        critiques.append(
            AgentCritique(
                reviewer=item.reviewer or "LLM Debate Reviewer",
                severity=severity,
                finding=item.finding,
                suggested_fix=item.suggested_fix
                or "用当前 workspace 证据补足，或将相关结论降级为待验证假设。",
            )
        )
    return critiques[:8]


def _parse_debate_as_text(text: str) -> list[AgentCritique]:
    """Fallback parser for non-JSON LLM output — preserves backward compat."""
    critiques: list[AgentCritique] = []
    for line in text.splitlines():
        text_line = line.strip(" -\t")
        if not text_line:
            continue
        severity = "warning"
        lowered = text_line.lower()
        if any(kw in lowered for kw in _ERROR_KEYWORDS):
            severity = "error"
        critiques.append(
            AgentCritique(
                reviewer="LLM Debate Reviewer",
                severity=severity,
                finding=text_line,
                suggested_fix="用当前 workspace 证据补足，或将相关结论降级为待验证假设。",
            )
        )
    return critiques[:8]


def _smart_debate_payload(
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
    critiques: list[AgentCritique],
    workspace: LiteratureWorkspace,
) -> str:
    citation_errors = [
        item
        for item in critiques
        if item.reviewer == "Citation Auditor" and item.severity == "error"
    ]
    revised = draft
    if citation_errors:
        revised = remove_unsupported_sentences(revised, guard_citations(revised, workspace))

    warnings = [item for item in critiques if item.severity in {"warning", "error"}]
    if warnings:
        revised = revised.rstrip()
        revised += "\n\n多 agent 复核后的限制说明："
        for item in warnings[:6]:
            revised += f"\n- {item.reviewer}: {item.finding}"
    return revised


def _round_score(
    critiques: list[AgentCritique],
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> float:
    quality = build_quality_report(config, workspace)
    score = 0.62
    score += 0.12 * quality.metrics.get("citation_guard_pass", 0.0)
    score += 0.08 * quality.metrics.get("parsed_rate", 0.0)
    score += 0.06 if quality.metrics.get("comparison_matrix_count", 0.0) else 0.0
    score += 0.06 if quality.metrics.get("storyline_claim_count", 0.0) else 0.0
    score -= 0.12 * sum(1 for item in critiques if item.severity == "error")
    score -= 0.04 * sum(1 for item in critiques if item.severity == "warning")
    return max(0.0, min(0.98, score))


def _replan_actions(
    critiques: list[AgentCritique],
    workspace: LiteratureWorkspace,
) -> list[str]:
    actions: list[str] = []
    if not workspace.full_text_reports and workspace.context.active_papers:
        actions.append("resolve_full_text")
    if not workspace.parsed_papers and workspace.context.active_papers:
        actions.append("parse_full_text_with_paddleocr")
    if any(item.reviewer == "Table Auditor" for item in critiques):
        actions.append("extract_tables_and_structured_artifacts")
    if any(item.reviewer == "Citation Auditor" for item in critiques):
        actions.append("rerun_citation_guard_after_revision")
    if any(item.reviewer == "Storyline Skeptic" for item in critiques):
        actions.append("rebuild_storyline_from_parsed_evidence")
    return actions


async def _execute_safe_replan_actions(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    actions: list[str],
) -> tuple[LiteratureWorkspace, list[str]]:
    executed: list[str] = []
    if "parse_full_text_with_paddleocr" in actions:
        workspace, report = parse_workspace_papers(workspace, config)
        if report.get("parsed_count", 0):
            executed.append("parse_full_text_with_paddleocr")
    if "extract_tables_and_structured_artifacts" in actions:
        workspace, _ = await extract_performance_cells(workspace, config)
        executed.append("extract_tables_and_structured_artifacts")
    if "rebuild_storyline_from_parsed_evidence" in actions:
        claims = build_storyline_from_workspace(workspace)
        workspace.context.filters.storyline_claim_count = len(claims)
        executed.append("rebuild_storyline_from_parsed_evidence")
    if "rerun_citation_guard_after_revision" in actions:
        executed.append("rerun_citation_guard_after_revision")
    return workspace, executed
