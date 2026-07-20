from __future__ import annotations

from datetime import datetime
from pathlib import Path

from littrace.sentinel.resource_pack import ResourcePack
from littrace.sentinel.state import DigestRecord, SentinelState


def build_digest_markdown(resource_pack: ResourcePack, state: SentinelState) -> str:
    lines = [
        f"# {state.watchlist.topic} briefing",
        "",
        f"- watchlist: `{state.watchlist.watchlist_id}`",
        f"- objective: {resource_pack.objective}",
        f"- papers: {len(resource_pack.papers)}",
        f"- structured documents: {len(resource_pack.structured_document_refs)}",
        f"- performance cells: {len(resource_pack.performance_cell_refs)}",
        "",
    ]
    if resource_pack.missing_evidence:
        lines.append("## Missing Evidence")
        for item in resource_pack.missing_evidence:
            lines.append(f"- {item}")
        lines.append("")
    if resource_pack.quality_warnings:
        lines.append("## Warnings")
        for warning in resource_pack.quality_warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## Papers")
    for paper in resource_pack.papers[:12]:
        suffix = f" ({paper.year})" if paper.year else ""
        lines.append(f"- {paper.title}{suffix}")
    lines.append("")
    lines.append("## Generated At")
    lines.append(datetime.now().isoformat(timespec="seconds"))
    return "\n".join(lines)


def save_digest(root: Path, run_id: str, markdown: str, resource_pack: ResourcePack, state: SentinelState) -> tuple[Path, DigestRecord]:
    digest_dir = root / "digests"
    digest_dir.mkdir(parents=True, exist_ok=True)
    target = digest_dir / f"{run_id}.md"
    target.write_text(markdown, encoding="utf-8")
    record = DigestRecord(
        run_id=run_id,
        digest_path=str(target),
        paper_count=len(resource_pack.papers),
        alert_count=1 if resource_pack.quality_warnings else 0,
        claim_count=len(resource_pack.storyline_claim_refs),
    )
    state.digest_history.append(record)
    return target, record
