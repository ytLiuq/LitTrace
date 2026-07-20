"""Single publication gate for user-visible research conclusions."""

import json
from datetime import UTC, datetime
from hashlib import sha256

from littrace.config import LitTraceConfig
from littrace.evidence.document_composer import build_research_document_report
from littrace.evidence.storyline import _render_structured_storyline_report
from littrace.models import (
    LiteratureWorkspace,
    ReleaseSnapshot,
    ResearchDocumentReport,
    VerificationReport,
)


def evaluate_publication(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    *,
    title: str | None = None,
) -> ResearchDocumentReport:
    """Return the canonical release decision for a workspace."""

    return build_research_document_report(workspace, config, title=title)


def create_release_snapshot(
    workspace: LiteratureWorkspace,
    reports: list[VerificationReport],
    *,
    release_ready: bool,
    release_blockers: list[str],
    config: LitTraceConfig | None = None,
    report_markdown: str | None = None,
) -> ReleaseSnapshot:
    """Create and retain a deterministic, replayable release decision."""

    workspace_payload = workspace.model_dump(mode="json", exclude={"release_snapshots"})
    workspace_hash = sha256(
        json.dumps(workspace_payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    claim_ids = sorted({report.claim_id for report in reports if report.claim_id})
    evidence_ids = sorted(
        {span.evidence_id for report in reports for span in report.evidence if span.evidence_id}
    )
    source_event_ids = sorted(event.event_id for event in workspace.source_events)
    report_hash = sha256((report_markdown or "").encode()).hexdigest() if report_markdown else None
    config_hash = sha256(config.model_dump_json().encode()).hexdigest() if config else None
    fingerprint = "\0".join(
        [
            workspace_hash,
            *claim_ids,
            *source_event_ids,
            report_hash or "",
            config_hash or "",
            str(release_ready),
            *sorted(release_blockers),
        ]
    )
    snapshot_id = f"release:{sha256(fingerprint.encode()).hexdigest()[:16]}"
    existing = next(
        (item for item in workspace.release_snapshots if item.snapshot_id == snapshot_id), None
    )
    if existing is not None:
        return existing
    snapshot = ReleaseSnapshot(
        snapshot_id=snapshot_id,
        created_at=datetime.now(UTC).isoformat(),
        workspace_hash=workspace_hash,
        claim_ids=claim_ids,
        evidence_ids=evidence_ids,
        source_event_ids=source_event_ids,
        report_hash=report_hash,
        config_hash=config_hash,
        release_ready=release_ready,
        release_blockers=release_blockers,
        tool_versions={"claim_verifier": "v2", "publication_gate": "v2"},
    )
    workspace.release_snapshots.append(snapshot)
    return snapshot


def draft_notice(report: ResearchDocumentReport) -> str:
    blockers = "\n".join(f"- {blocker}" for blocker in report.release_blockers)
    return "\n".join(
        [
            "> **DRAFT - NOT FOR PUBLICATION**",
            "> Claim verification has not passed. This is evidence review material, not a final conclusion.",
            blockers,
        ]
    )


def render_publication_storyline(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> tuple[str, ResearchDocumentReport]:
    """Render storyline text with the same release decision as reports and answers."""

    report = evaluate_publication(workspace, config)
    body = _render_structured_storyline_report(workspace)
    if report.release_ready:
        return body, report
    return f"{draft_notice(report)}\n\n{body}", report
