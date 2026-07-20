"""Canonical work identity and transparent merge decisions."""

from __future__ import annotations

import re
from hashlib import sha256

from littrace.models import (
    CanonicalWork,
    LiteratureWorkspace,
    PaperMetadata,
    ResolutionDecision,
    SourceRecord,
)


def canonical_work_id(paper: PaperMetadata) -> str:
    if paper.doi:
        return f"doi:{paper.doi.strip().lower()}"
    title = re.sub(r"\W+", " ", paper.title.lower()).strip()
    return f"title:{sha256(title.encode()).hexdigest()[:16]}"


def register_paper_identity(workspace: LiteratureWorkspace, paper: PaperMetadata) -> CanonicalWork:
    """Persist the canonical-work and source-record decision for a paper."""

    work_id = canonical_work_id(paper)
    existing = workspace.canonical_works.get(work_id)
    if existing is None:
        existing = CanonicalWork(
            work_id=work_id,
            canonical_paper_id=paper.paper_id,
            doi=paper.doi,
            version_paper_ids=[paper.paper_id],
        )
        workspace.canonical_works[work_id] = existing
        strategy = "doi" if paper.doi else "normalized_title"
        reason = f"Created canonical work using {strategy}."
    else:
        if paper.paper_id not in existing.version_paper_ids:
            existing.version_paper_ids.append(paper.paper_id)
        strategy = "doi" if paper.doi else "normalized_title"
        reason = f"Merged paper into existing canonical work using {strategy}."
    source_id = f"metadata:{paper.paper_id}"
    workspace.source_records.setdefault(
        source_id,
        SourceRecord(
            source_record_id=source_id,
            paper_id=paper.paper_id,
            source_name=paper.publisher or paper.journal or "metadata",
            source_url=str(paper.source_urls[0]) if paper.source_urls else None,
            content_hash=sha256(paper.model_dump_json().encode()).hexdigest(),
        ),
    )
    decision_id = f"resolve:{sha256(f'{work_id}:{paper.paper_id}'.encode()).hexdigest()[:16]}"
    if not any(item.decision_id == decision_id for item in workspace.resolution_decisions):
        workspace.resolution_decisions.append(
            ResolutionDecision(
                decision_id=decision_id,
                canonical_work_id=work_id,
                candidate_paper_ids=list(existing.version_paper_ids),
                strategy=strategy,
                confidence=1.0 if paper.doi else 0.82,
                reason=reason,
            )
        )
    return existing
