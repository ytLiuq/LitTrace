from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.models import CitationRecord, LiteratureWorkspace, PaperMetadata
from littrace.sentinel.state import SentinelState
from littrace.tool_contracts import ToolArtifactRef


class ResourcePack(BaseModel):
    schema_version: str = "littrace.resource_pack.v1"
    watchlist_id: str
    objective: str
    papers: list[PaperMetadata] = Field(default_factory=list)
    citation_records: list[CitationRecord] = Field(default_factory=list)
    full_text_report_refs: list[str] = Field(default_factory=list)
    structured_document_refs: list[str] = Field(default_factory=list)
    performance_cell_refs: list[str] = Field(default_factory=list)
    comparison_matrix_refs: list[str] = Field(default_factory=list)
    storyline_claim_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    artifact_refs: list[ToolArtifactRef] = Field(default_factory=list)
    created_at: str | None = None


def build_resource_pack(
    workspace: LiteratureWorkspace,
    state: SentinelState,
    *,
    objective: str | None = None,
    artifact_refs: list[ToolArtifactRef] | None = None,
    quality_warnings: list[str] | None = None,
) -> ResourcePack:
    objective_text = objective or state.watchlist.objective or state.watchlist.topic
    papers = [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]
    citation_records = [record for paper in papers for record in _citation_records_for_paper(paper)]
    full_text_report_refs = list(workspace.full_text_reports.keys())
    structured_document_refs = list(workspace.context.filters.structured_document_paths.values())
    if not structured_document_refs:
        structured_document_refs = [
            f"{paper_id}:parsed" for paper_id, parsed in workspace.parsed_papers.items() if parsed.parsed
        ]
    performance_cell_refs = [
        f"{cell.paper_id}:{cell.metric}"
        for cell in workspace.performance_cells
        if getattr(cell, "paper_id", None) and getattr(cell, "metric", None)
    ]
    storyline_claim_refs = [
        getattr(claim, "evidence_id", "") or getattr(claim, "claim_id", "")
        for claim in []
    ]
    missing_evidence: list[str] = []
    if not papers:
        missing_evidence.append("no_active_papers")
    if not structured_document_refs:
        missing_evidence.append("no_structured_documents")
    if not performance_cell_refs:
        missing_evidence.append("no_performance_cells")
    return ResourcePack(
        watchlist_id=state.watchlist.watchlist_id,
        objective=objective_text,
        papers=papers,
        citation_records=citation_records,
        full_text_report_refs=full_text_report_refs,
        structured_document_refs=structured_document_refs,
        performance_cell_refs=performance_cell_refs,
        comparison_matrix_refs=[],
        storyline_claim_refs=storyline_claim_refs,
        missing_evidence=missing_evidence,
        quality_warnings=list(quality_warnings or []),
        artifact_refs=list(artifact_refs or []),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def _citation_records_for_paper(paper: PaperMetadata) -> list[CitationRecord]:
    if not paper.doi:
        return []
    return [
        CitationRecord(
            paper_id=paper.paper_id,
            citation_text=paper.title,
            access_url=f"https://doi.org/{paper.doi}",
            doi=paper.doi,
        )
    ]


def save_resource_pack(root: Path, resource_pack: ResourcePack, *, run_id: str | None = None) -> Path:
    target_dir = root / "evidence_base" / "resource_packs"
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_id or (resource_pack.created_at or "pack").replace(":", "")
    watchlist_component = _bounded_filename_component(resource_pack.watchlist_id)
    target = target_dir / f"{watchlist_component}-{suffix}.json"
    target.write_text(resource_pack.model_dump_json(indent=2), encoding="utf-8")
    return target


def _bounded_filename_component(value: str, *, max_length: int = 32) -> str:
    """Keep Sentinel artifact paths below common Windows path limits."""

    if len(value) <= max_length:
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return f"{value[:prefix_length]}-{digest}"
