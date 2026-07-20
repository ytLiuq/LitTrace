from __future__ import annotations

from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.evaluation.harnesses import HarnessResult, check_citations, check_storyline_claims
from littrace.evidence.claims import (
    register_evidence,
    verify_structured_claim,
    workspace_evidence_registry,
)
from littrace.models import (
    Claim,
    ClaimKind,
    ClaimStatus,
    EvidenceSpan,
    LiteratureWorkspace,
    ResearchDocumentReport,
    ResearchDocumentSection,
    StructuredArtifact,
    VerificationReport,
)
from littrace.evaluation.quality_report import build_quality_report
from littrace.evidence.storyline import build_storyline_from_workspace
from littrace.evidence.tables import build_comparison_matrices


def build_research_document_report(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    title: str | None = None,
) -> ResearchDocumentReport:
    """Build an evidence-first Markdown research report from the active chat context."""

    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    citations = citation_records_for_papers(papers)
    citation_harness = check_citations(citations)
    storyline = build_storyline_from_workspace(workspace)
    storyline_harness = check_storyline_claims(storyline)
    matrix = build_comparison_matrices(workspace)
    artifacts = _structured_artifacts(workspace)
    autonomous_review = _autonomous_review_summary(workspace)
    quality = build_quality_report(config, workspace)
    doc_title = title or _infer_title(workspace)
    verification_reports = _verification_reports(workspace, storyline, matrix)
    release_blockers = _release_blockers(
        verification_reports,
        has_analytic_claims=bool(verification_reports),
        strict_all_claims=config.publication_policy.strict_all_claims,
        require_publishable_claim=config.publication_policy.require_publishable_claim,
    )
    quality_metrics = _with_verification_metrics(
        quality.metrics, verification_reports, release_blockers
    )
    warnings = [
        *_harness_warnings("citations", citation_harness),
        *_harness_warnings("storyline", storyline_harness),
        *quality.warnings,
        *release_blockers,
    ]
    published_only = not config.publication_policy.strict_all_claims
    visible_storyline = (
        _published_storyline(storyline, verification_reports) if published_only else storyline
    )

    sections = [
        _abstract_section(workspace, quality.metrics),
        _methods_section(workspace, quality.metrics),
        _literature_section(workspace),
        _synthesis_section(workspace),
        _storyline_section(visible_storyline),
        _matrix_section(matrix, verification_reports if published_only else None),
        _artifact_section(artifacts),
        autonomous_review,
        _verification_section(verification_reports, release_blockers),
        _limitations_section(quality_metrics, warnings),
        _quality_section(quality_metrics),
        _references_section(citations),
    ]
    evidence_count = sum(len(section.evidence) for section in sections)
    markdown = _render_markdown(doc_title, sections, warnings, release_blockers)
    from littrace.publication import create_release_snapshot

    release_snapshot = create_release_snapshot(
        workspace,
        verification_reports,
        release_ready=not release_blockers,
        release_blockers=release_blockers,
        config=config,
        report_markdown=markdown,
    )
    return ResearchDocumentReport(
        title=doc_title,
        markdown=markdown,
        sections=sections,
        citation_records=citations,
        verification_reports=verification_reports,
        release_ready=not release_blockers,
        release_blockers=release_blockers,
        evidence_count=evidence_count,
        quality_metrics=quality_metrics,
        warnings=warnings,
        release_snapshot=release_snapshot,
    )


def _infer_title(workspace: LiteratureWorkspace) -> str:
    topic = getattr(workspace.context.filters, "topic", None) or getattr(
        workspace.context.filters, "discipline", None
    )
    if isinstance(topic, str) and topic.strip():
        return f"LitTrace Research Report: {topic.strip()}"
    return "LitTrace Research Report"


def _abstract_section(
    workspace: LiteratureWorkspace,
    metrics: dict[str, float],
) -> ResearchDocumentSection:
    active_count = len(workspace.context.active_papers)
    parsed_rate = metrics.get("parsed_rate", 0.0)
    matrix_count = metrics.get("comparison_matrix_count", 0.0)
    storyline_count = metrics.get("storyline_claim_count", 0.0)
    body = (
        f"本报告基于当前会话中的 {active_count} 篇 active papers 形成，目标是给出一个"
        "可追溯、可复核的学术综述草案，而不是替代人工阅读全文后的最终结论。"
        f"当前全文解析率为 {parsed_rate:.2f}，已形成 {matrix_count:.0f} 个性能对比矩阵，"
        f"并识别 {storyline_count:.0f} 条发展脉络证据。"
        "报告中的论文级判断均需能够回到 DOI、访问链接、页码、表格、图注或原文片段。"
    )
    return ResearchDocumentSection(title="摘要", body=body)


def _methods_section(
    workspace: LiteratureWorkspace,
    metrics: dict[str, float],
) -> ResearchDocumentSection:
    filters = workspace.context.filters
    routes = getattr(filters, "source_routes", None) or []
    year_min = getattr(filters, "year_min", None) or "未限定"
    search_mode = getattr(filters, "search_mode", None) or "unknown"
    lines = [
        "本报告采用证据优先的会话内综述流程：先检索和筛选文献，再解析可获得全文，"
        "随后抽取性能指标、结构化图表/公式证据，并用 citation/storyline/table harness 进行复核。",
        f"- Search mode: {search_mode}",
        f"- Source routes: {', '.join(routes) if isinstance(routes, list) and routes else 'not recorded'}",
        f"- Year lower bound: {year_min}",
        f"- Parsed full-text rate: {metrics.get('parsed_rate', 0.0):.3f}",
        f"- Citation guard pass: {metrics.get('citation_guard_pass', 0.0):.3f}",
    ]
    return ResearchDocumentSection(title="方法与证据来源", body="\n".join(lines))


def _literature_section(workspace: LiteratureWorkspace) -> ResearchDocumentSection:
    lines: list[str] = []
    evidence: list[EvidenceSpan] = []
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    if not papers:
        lines.append("当前 chat 上下文还没有 active papers。")
    for index, paper in enumerate(papers, start=1):
        source = paper.journal or paper.publisher or "unknown source"
        access = paper.pdf_url or (paper.source_urls[0] if paper.source_urls else None)
        lines.append(f"{index}. {paper.title} ({paper.year or 'n.d.'}, {source})")
        if paper.doi:
            lines.append(f"   DOI: https://doi.org/{paper.doi}")
        if access:
            lines.append(f"   Access: {access}")
        evidence.append(
            EvidenceSpan(
                paper_id=paper.paper_id,
                section="metadata",
                snippet=f"{paper.title}; {paper.year or 'n.d.'}; {source}",
                confidence=0.75,
            )
        )
    return ResearchDocumentSection(
        title="文献上下文",
        body="\n".join(lines),
        evidence=evidence,
    )


def _synthesis_section(workspace: LiteratureWorkspace) -> ResearchDocumentSection:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    if not papers:
        return ResearchDocumentSection(title="主题综合", body="当前没有足够文献形成主题综合。")

    years = [paper.year for paper in papers if paper.year is not None]
    sources = sorted({paper.journal or paper.publisher or "unknown source" for paper in papers})
    recent = [paper for paper in papers if paper.year is not None and paper.year >= 2024]
    lines = [
        "当前证据更适合支持一份范围受限的 narrative review：它可以描述检索到的研究对象、"
        "材料体系、性能指标和证据缺口，但不应扩大为领域共识或因果定论。",
        f"- 文献年份范围：{min(years)}-{max(years)}" if years else "- 文献年份范围：未知",
        f"- 2024 年及以后文献：{len(recent)} / {len(papers)}",
        f"- 主要来源：{', '.join(sources[:8])}",
    ]
    return ResearchDocumentSection(title="主题综合", body="\n".join(lines))


def _storyline_section(storyline) -> ResearchDocumentSection:
    lines: list[str] = []
    evidence: list[EvidenceSpan] = []
    if not storyline:
        lines.append("当前证据不足以生成 solution-limit-response 发展链。")
    for claim in storyline:
        lines.append(f"- **{claim.claim_type}**: {claim.claim}")
        for span in claim.evidence[:4]:
            evidence.append(span)
            location = f"p.{span.page}" if span.page is not None else span.section or "evidence"
            snippet = (span.snippet or "").replace("\n", " ")[:180]
            lines.append(f"  - [{span.paper_id}] {location}: {snippet}")
    return ResearchDocumentSection(
        title="发展脉络",
        body="\n".join(lines),
        evidence=evidence,
    )


def _matrix_section(
    matrix,
    verification_reports: list[VerificationReport] | None = None,
) -> ResearchDocumentSection:
    lines: list[str] = []
    evidence: list[EvidenceSpan] = []
    if not matrix.matrices:
        lines.append("当前没有可审计的性能矩阵。建议先解析 PDF/OCR 并运行表格抽取。")
    for item in matrix.matrices:
        lines.append(f"### {item.metric}")
        if item.warnings:
            lines.append("Warnings: " + "; ".join(item.warnings))
        lines.append("| Paper | Year | Value | Unit | Comparable | Evidence |")
        lines.append("|---|---:|---:|---|---|---|")
        for row in item.rows:
            if verification_reports is not None and not _row_is_publishable(
                row, verification_reports
            ):
                continue
            evidence.append(row.evidence)
            snippet = (
                row.evidence.snippet or row.evidence.table_id or row.evidence.section or ""
            ).replace("\n", " ")[:120]
            lines.append(
                f"| {row.title or row.paper_id} | {row.year or ''} | {row.value} | "
                f"{row.unit or ''} | {row.comparable} | {snippet} |"
            )
    return ResearchDocumentSection(
        title="性能对比",
        body="\n".join(lines),
        evidence=evidence,
    )


def _published_storyline(storyline, reports: list[VerificationReport]):
    published = {report.claim for report in reports if report.publishable}
    return [claim for claim in storyline if claim.claim in published]


def _row_is_publishable(row, reports: list[VerificationReport]) -> bool:
    unit = f" {row.unit}" if row.unit else ""
    claim = f"{row.title or row.paper_id} reports {row.metric} = {row.value}{unit}."
    return any(report.claim == claim and report.publishable for report in reports)


def _artifact_section(artifacts: list[StructuredArtifact]) -> ResearchDocumentSection:
    lines: list[str] = []
    evidence: list[EvidenceSpan] = []
    if not artifacts:
        lines.append("当前没有抽取到结构化图表、公式或图注证据。")
    for artifact in artifacts[:20]:
        evidence.append(artifact.evidence)
        label = f" {artifact.label}" if artifact.label else ""
        location = (
            f"p.{artifact.evidence.page}" if artifact.evidence.page is not None else "evidence"
        )
        text = artifact.text.replace("\n", " ")[:220]
        lines.append(
            f"- **{artifact.artifact_type}{label}** [{artifact.paper_id}, {location}]: {text}"
        )
    return ResearchDocumentSection(
        title="图表与公式证据",
        body="\n".join(lines),
        evidence=evidence,
    )


def _verification_reports(
    workspace: LiteratureWorkspace, storyline, matrix
) -> list[VerificationReport]:
    """Verify only analytic claims that the generated report presents as findings."""

    registry = workspace_evidence_registry(workspace)
    claims: list[Claim] = list(workspace.claims)
    cells_by_evidence_id = {
        cell.evidence.evidence_id: cell
        for cell in workspace.performance_cells
        if cell.evidence.evidence_id is not None
    }
    for story_claim in storyline:
        register_evidence(workspace, story_claim.evidence)
        registry.update(
            {
                span.evidence_id: span
                for span in story_claim.evidence
                if span.evidence_id is not None
            }
        )
        evidence_ids = [span.evidence_id for span in story_claim.evidence if span.evidence_id]
        quotes = {
            span.evidence_id: span.snippet or ""
            for span in story_claim.evidence
            if span.evidence_id
        }
        claims.append(
            Claim(
                text=story_claim.claim,
                claim_kind=ClaimKind.CAUSAL
                if story_claim.claim_type.endswith("chain")
                else ClaimKind.QUALITATIVE,
                evidence_ids=evidence_ids,
                support_quotes=quotes,
                requires_corroboration=True,
                critical=False,
            )
        )
    for item in matrix.matrices:
        for row in item.rows:
            unit = f" {row.unit}" if row.unit else ""
            cell = cells_by_evidence_id.get(row.evidence.evidence_id)
            evidence = row.evidence.model_copy(
                update={
                    "observed_value": row.value,
                    "observed_unit": row.unit,
                    "observed_value_min": cell.value_min if cell else None,
                    "observed_value_max": cell.value_max if cell else None,
                    "observed_uncertainty": cell.uncertainty if cell else None,
                }
            )
            register_evidence(workspace, [evidence])
            if evidence.evidence_id:
                registry[evidence.evidence_id] = evidence
                claims.append(
                    Claim(
                        text=f"{row.title or row.paper_id} reports {row.metric} = {row.value}{unit}.",
                        claim_kind=ClaimKind.NUMERIC,
                        evidence_ids=[evidence.evidence_id],
                        support_quotes={evidence.evidence_id: evidence.snippet or ""},
                        metric=row.metric,
                        expected_value=float(row.value)
                        if isinstance(row.value, int | float)
                        else None,
                        expected_unit=row.unit,
                        critical=False,
                    )
                )
    reports = [verify_structured_claim(claim, registry) for claim in claims]
    workspace.claims = claims
    workspace.claim_verification_reports = reports
    return reports


def _release_blockers(
    reports: list[VerificationReport],
    *,
    has_analytic_claims: bool,
    strict_all_claims: bool,
    require_publishable_claim: bool,
) -> list[str]:
    blockers: list[str] = []
    if not has_analytic_claims:
        return ["Release requires at least one verified or corroborated analytic claim."]
    if require_publishable_claim and not any(report.publishable for report in reports):
        blockers.append("Release requires at least one verified or corroborated analytic claim.")
    for report in reports:
        if report.publishable or (not strict_all_claims and not report.critical):
            continue
        requirements = "; ".join(report.missing_requirements) or "Claim lacks publishable evidence."
        blockers.append(
            f"Claim verification blocks release ({report.status}): {report.claim} — {requirements}"
        )
    return blockers


def _with_verification_metrics(
    metrics: dict[str, float],
    reports: list[VerificationReport],
    release_blockers: list[str],
) -> dict[str, float]:
    result = dict(metrics)
    total = len(reports)
    for status in ClaimStatus:
        result[f"{status.value}_claim_count"] = float(
            sum(report.status == status for report in reports)
        )
    result["non_publishable_claim_count"] = float(sum(not report.publishable for report in reports))
    result["claim_traceability_rate"] = (
        sum(any(span.has_location for span in report.evidence) for report in reports) / total
        if total
        else 1.0
    )
    result["draftable_claim_count"] = float(sum(report.draftable for report in reports))
    result["claim_release_gate_pass"] = 0.0 if release_blockers else 1.0
    return result


def _verification_section(
    reports: list[VerificationReport],
    release_blockers: list[str],
) -> ResearchDocumentSection:
    gate = "blocked" if release_blockers else "passed"
    lines = [f"Release gate: **{gate}**."]
    if not reports:
        lines.append("当前报告未生成可供发布的分析性断言；仅保留文献元数据、引用和证据缺口。")
    for report in reports:
        lines.append(f"- **{report.status}**: {report.claim}")
        if not report.publishable:
            lines.append("  - Publication: withheld pending verification.")
        lines.append(f"  - Evidence spans: {len(report.evidence)}")
        if report.missing_requirements:
            lines.append(f"  - Remaining requirements: {'; '.join(report.missing_requirements)}")
    return ResearchDocumentSection(title="Claim Verification", body="\n".join(lines))


def _autonomous_review_summary(workspace: LiteratureWorkspace) -> ResearchDocumentSection:
    raw = getattr(workspace.context.filters, "autonomous_loop_report", None)
    if not isinstance(raw, dict):
        return ResearchDocumentSection(
            title="多 Agent 复核与修订",
            body="当前会话尚未运行 Autonomous Review Council；建议在最终报告前运行自动审稿/反驳/修订循环。",
        )
    rounds = raw.get("rounds") or []
    lines = [
        f"- Passed: {raw.get('passed')}",
        f"- Score: {raw.get('score')}",
        f"- Release ready: {raw.get('release_ready', False)}",
    ]
    actions = raw.get("replan_actions") or []
    executed = raw.get("executed_replan_actions") or []
    if actions:
        lines.append(f"- Replan actions: {', '.join(str(item) for item in actions)}")
    if executed:
        lines.append(f"- Executed actions: {', '.join(str(item) for item in executed)}")
    for item in rounds[:3]:
        if not isinstance(item, dict):
            continue
        critiques = item.get("critiques") or []
        lines.append(f"### Round {item.get('round_index')}")
        lines.append(f"- Round score: {item.get('score')}; passed: {item.get('passed')}")
        for critique in critiques[:5]:
            if not isinstance(critique, dict):
                continue
            reviewer = critique.get("reviewer") or "Reviewer"
            severity = critique.get("severity") or "info"
            finding = str(critique.get("finding") or "")[:220]
            lines.append(f"- **{reviewer} [{severity}]**: {finding}")
    release_blockers = raw.get("release_blockers") or []
    if release_blockers:
        lines.append("### Release blockers")
        lines.extend(f"- {blocker}" for blocker in release_blockers[:5])
    final_answer = str(raw.get("final_answer") or "").strip()
    if raw.get("release_ready") is True and final_answer:
        lines.append("### Revised answer excerpt")
        lines.append(final_answer[:1000])
    return ResearchDocumentSection(
        title="多 Agent 复核与修订",
        body="\n".join(lines),
    )


def _limitations_section(
    metrics: dict[str, float],
    warnings: list[str],
) -> ResearchDocumentSection:
    lines = [
        "以下限制会影响报告可下结论的强度：",
        f"- Local PDF coverage: {metrics.get('local_pdf_rate', 0.0):.3f}",
        f"- Parsed full-text coverage: {metrics.get('parsed_rate', 0.0):.3f}",
        f"- Verified full-text candidate rate: {metrics.get('verified_full_text_candidate_rate', 0.0):.3f}",
        f"- Performance cell count: {metrics.get('performance_cell_count', 0.0):.0f}",
    ]
    if warnings:
        lines.append("- Harness warnings indicate that some claims should remain provisional:")
        for warning in warnings[:6]:
            lines.append(f"  - {warning}")
    else:
        lines.append("- 当前没有阻断性 harness warnings。")
    lines.append(
        "建议优先补全文解析、PaddleOCR 图表/公式证据和单位可比性校验，再扩展为更完整的学术综述。"
    )
    return ResearchDocumentSection(title="局限性与下一步", body="\n".join(lines))


def _quality_section(metrics: dict[str, float]) -> ResearchDocumentSection:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key in sorted(metrics):
        value = metrics[key]
        lines.append(f"| {key} | {value:.3f} |")
    return ResearchDocumentSection(title="质量指标", body="\n".join(lines))


def _references_section(citations) -> ResearchDocumentSection:
    lines: list[str] = []
    if not citations:
        lines.append("当前没有可引用文献。")
    for record in citations:
        lines.append(f"- {record.citation_text}")
        lines.append(f"  Access: {record.access_url}")
    return ResearchDocumentSection(title="引用与访问链接", body="\n".join(lines))


def _structured_artifacts(workspace: LiteratureWorkspace) -> list[StructuredArtifact]:
    raw = getattr(workspace.context.filters, "structured_artifacts", [])
    artifacts: list[StructuredArtifact] = []
    if not isinstance(raw, list):
        return artifacts
    for item in raw:
        if isinstance(item, StructuredArtifact):
            artifacts.append(item)
        elif isinstance(item, dict):
            artifacts.append(StructuredArtifact.model_validate(item))
    return artifacts


def _harness_warnings(name: str, harness: HarnessResult) -> list[str]:
    return [f"{name}: {item}" for item in [*harness.errors, *harness.warnings]]


def _render_markdown(
    title: str,
    sections: list[ResearchDocumentSection],
    warnings: list[str],
    release_blockers: list[str],
) -> str:
    lines = [f"# {title}", ""]
    if release_blockers:
        lines.extend(
            [
                "> **DRAFT - NOT FOR PUBLICATION**",
                "> Claim verification has not passed. This document is an evidence review draft, not a final conclusion.",
                "",
            ]
        )
    lines.append(
        "本文档是 LitTrace 基于当前会话证据生成的学术化研究报告草案。"
        "其写作原则是：所有论文级判断必须有可追踪证据，缺证处明确标注为限制或待补证。"
    )
    lines.append("")
    for section in sections:
        lines.append(f"## {section.title}")
        lines.append(section.body or "无。")
        lines.append("")
    lines.append("## Harness Warnings")
    if not warnings:
        lines.append("No blocking warnings.")
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)
