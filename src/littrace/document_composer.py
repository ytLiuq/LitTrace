from __future__ import annotations

from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.harnesses import HarnessResult, check_citations, check_storyline_claims
from littrace.models import (
    EvidenceSpan,
    LiteratureWorkspace,
    ResearchDocumentReport,
    ResearchDocumentSection,
    StructuredArtifact,
)
from littrace.quality_report import build_quality_report
from littrace.storyline import build_storyline_from_workspace
from littrace.tables import build_comparison_matrices


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
    quality = build_quality_report(config, workspace)
    doc_title = title or _infer_title(workspace)
    warnings = [
        *_harness_warnings("citations", citation_harness),
        *_harness_warnings("storyline", storyline_harness),
        *quality.warnings,
    ]

    sections = [
        _abstract_section(workspace, quality.metrics),
        _methods_section(workspace, quality.metrics),
        _literature_section(workspace),
        _synthesis_section(workspace),
        _storyline_section(storyline),
        _matrix_section(matrix),
        _artifact_section(artifacts),
        _limitations_section(quality.metrics, warnings),
        _quality_section(quality.metrics),
        _references_section(citations),
    ]
    evidence_count = sum(len(section.evidence) for section in sections)
    markdown = _render_markdown(doc_title, sections, warnings)
    return ResearchDocumentReport(
        title=doc_title,
        markdown=markdown,
        sections=sections,
        citation_records=citations,
        evidence_count=evidence_count,
        quality_metrics=quality.metrics,
        warnings=warnings,
    )


def _infer_title(workspace: LiteratureWorkspace) -> str:
    topic = workspace.context.filters.get("topic") or workspace.context.filters.get("discipline")
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
    routes = filters.get("source_routes") or []
    year_min = filters.get("year_min") or "未限定"
    search_mode = filters.get("search_mode") or "unknown"
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


def _matrix_section(matrix) -> ResearchDocumentSection:
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
            evidence.append(row.evidence)
            snippet = (row.evidence.snippet or row.evidence.table_id or row.evidence.section or "").replace(
                "\n", " "
            )[:120]
            lines.append(
                f"| {row.title or row.paper_id} | {row.year or ''} | {row.value} | "
                f"{row.unit or ''} | {row.comparable} | {snippet} |"
            )
    return ResearchDocumentSection(
        title="性能对比",
        body="\n".join(lines),
        evidence=evidence,
    )


def _artifact_section(artifacts: list[StructuredArtifact]) -> ResearchDocumentSection:
    lines: list[str] = []
    evidence: list[EvidenceSpan] = []
    if not artifacts:
        lines.append("当前没有抽取到结构化图表、公式或图注证据。")
    for artifact in artifacts[:20]:
        evidence.append(artifact.evidence)
        label = f" {artifact.label}" if artifact.label else ""
        location = f"p.{artifact.evidence.page}" if artifact.evidence.page is not None else "evidence"
        text = artifact.text.replace("\n", " ")[:220]
        lines.append(f"- **{artifact.artifact_type}{label}** [{artifact.paper_id}, {location}]: {text}")
    return ResearchDocumentSection(
        title="图表与公式证据",
        body="\n".join(lines),
        evidence=evidence,
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
    raw = workspace.context.filters.get("structured_artifacts", [])
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
) -> str:
    lines = [f"# {title}", ""]
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
