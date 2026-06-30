from __future__ import annotations

import json
import re

from littrace.citations import citation_records_for_papers
from littrace.models import LiteratureWorkspace
from littrace.session import ChatSession
from littrace.storyline import build_storyline_from_workspace
from littrace.tables import build_comparison_matrices


def export_session_bundle(session: ChatSession, workspace: LiteratureWorkspace) -> dict[str, str]:
    session.artifacts_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = session.artifacts_dir / "research_brief.md"
    bibtex_path = session.artifacts_dir / "references.bib"
    json_path = session.artifacts_dir / "workspace_export.json"

    markdown_path.write_text(render_markdown_brief(workspace), encoding="utf-8")
    bibtex_path.write_text(render_bibtex(workspace), encoding="utf-8")
    json_path.write_text(
        json.dumps(workspace.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "markdown": str(markdown_path),
        "bibtex": str(bibtex_path),
        "json": str(json_path),
    }


def render_markdown_brief(workspace: LiteratureWorkspace) -> str:
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    citations = citation_records_for_papers(papers)
    matrix = build_comparison_matrices(workspace)
    storyline = build_storyline_from_workspace(workspace)

    lines = ["# LitTrace Research Brief", ""]
    lines.append("## Literature Context")
    if not papers:
        lines.append("No active papers.")
    for index, paper in enumerate(papers, start=1):
        source = paper.journal or paper.publisher or "unknown source"
        lines.append(f"{index}. {paper.title} ({paper.year or 'n.d.'}, {source})")
        if paper.doi:
            lines.append(f"   DOI: https://doi.org/{paper.doi}")
    lines.append("")

    lines.append("## Performance Matrix")
    if not matrix.matrices:
        lines.append("No performance metrics extracted.")
    for item in matrix.matrices:
        lines.append(f"### {item.metric}")
        if item.warnings:
            lines.append("Warnings: " + "; ".join(item.warnings))
        lines.append("| Paper | Value | Unit | Comparable | Evidence |")
        lines.append("|---|---:|---|---|---|")
        for row in item.rows:
            evidence = row.evidence.snippet or row.evidence.table_id or row.evidence.section or ""
            lines.append(
                f"| {row.title or row.paper_id} | {row.value} | {row.unit or ''} | {row.comparable} | {evidence[:80]} |"
            )
    lines.append("")

    lines.append("## Storyline")
    if not storyline:
        lines.append("No grounded storyline claims available.")
    for claim in storyline:
        lines.append(f"- **{claim.claim_type}**: {claim.claim}")
        evidence_ids = ", ".join(sorted({e.paper_id for e in claim.evidence}))
        lines.append(f"  Evidence: {evidence_ids}")
    lines.append("")

    lines.append("## References")
    for citation in citations:
        lines.append(f"- {citation.citation_text}")
        lines.append(f"  Access: {citation.access_url}")
    lines.append("")
    return "\n".join(lines)


def render_bibtex(workspace: LiteratureWorkspace) -> str:
    entries = []
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        key = _bibtex_key(paper.authors[0] if paper.authors else "unknown", paper.year, paper.title)
        fields = {
            "title": paper.title,
            "author": " and ".join(paper.authors) if paper.authors else None,
            "year": str(paper.year) if paper.year else None,
            "journal": paper.journal,
            "publisher": paper.publisher,
            "doi": paper.doi,
            "url": str(paper.source_urls[0]) if paper.source_urls else None,
        }
        body = "\n".join(
            f"  {name} = {{{value}}}," for name, value in fields.items() if value
        )
        entries.append(f"@article{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")


def _bibtex_key(author: str, year: int | None, title: str) -> str:
    last = re.sub(r"[^A-Za-z0-9]", "", author.split()[-1] if author.split() else author)
    first_word = re.sub(r"[^A-Za-z0-9]", "", title.split()[0] if title.split() else "paper")
    return f"{last}{year or 'nd'}{first_word}"
