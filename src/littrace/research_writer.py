from __future__ import annotations

from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.llm import LLMReply, chat_completion
from littrace.models import LiteratureWorkspace
from littrace.storyline import build_storyline_from_workspace


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

    system_prompt = (
        "You are LitTrace Research Writer. Answer in Chinese. "
        "Use only the provided literature context, parsed evidence snippets, performance cells, "
        "and citation records. Do not invent papers, metrics, claims, or access links. "
        "When evidence is insufficient, say what is missing. "
        "Organize research storylines as: previous solution -> remaining limitation -> later response. "
        "End with a '引用与访问链接' section using the provided citation records."
    )
    user_message = _writer_payload(question, workspace)
    return await chat_completion(config, system_prompt, user_message, workspace=None)


def fallback_evidence_answer(question: str, workspace: LiteratureWorkspace) -> str:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    if not papers:
        return "当前还没有文献上下文。请先检索论文。"
    lines = [f"我会基于当前 {len(papers)} 篇文献回答：{question}"]
    lines.append("当前证据主要来自标题、摘要、解析片段和已抽取指标；证据不足处会保持保守。")
    lines.append("引用与访问链接：")
    for record in citation_records_for_papers(papers):
        lines.append(f"- {record.citation_text} {record.access_url}")
    return "\n".join(lines)


async def write_storyline_narrative(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
) -> LLMReply:
    claims = build_storyline_from_workspace(workspace)
    if not claims:
        return LLMReply(
            text="当前证据不足以生成真实的发展脉络。建议先解析 PDF 全文。",
            used_llm=False,
            error="no_storyline_claims",
        )
    system_prompt = (
        "You are LitTrace Storyline Writer. Write in Chinese. "
        "Use only the supplied claims and evidence. "
        "Never make broad field-history claims unless the evidence states them. "
        "Focus on: previous work solved what, what limitation remained, and how later work responded. "
        "End with citations and access links."
    )
    payload = _storyline_payload(workspace)
    return await chat_completion(config, system_prompt, payload, workspace=None)


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
            for section in (parsed.get("sections") or [])[:6]:
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
        lines.append(f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}")
    return "\n".join(lines)


def _storyline_payload(workspace: LiteratureWorkspace) -> str:
    lines = ["Storyline claims:"]
    for claim in build_storyline_from_workspace(workspace):
        lines.append(f"- type={claim.claim_type}; confidence={claim.confidence}; claim={claim.claim}")
        for evidence in claim.evidence:
            lines.append(
                f"  evidence paper={evidence.paper_id}; section={evidence.section}; "
                f"page={evidence.page}; snippet={evidence.snippet}"
            )
    lines.append("")
    lines.append("Citation records:")
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    for record in citation_records_for_papers(papers):
        lines.append(f"- paper={record.paper_id}; citation={record.citation_text}; url={record.access_url}")
    return "\n".join(lines)
