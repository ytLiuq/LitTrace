from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header

from littrace.api.backend import api_app
from littrace.api.auth import resolve_request_session
from littrace.artifact_registry import artifact_registry_from_config
from littrace.artifact_store import BlobRef, artifact_store_from_config
from littrace.models import (
    ComparisonMatrixReport,
    LiteratureWorkspace,
    ResearchDocumentReport,
    ResearchRunResult,
)
from littrace.skill_runner import (
    build_comparison_matrix_skill,
    build_research_report_skill,
    extract_tables_skill,
    parse_workspace_skill,
)
from littrace.publication import render_publication_storyline
from littrace.evidence.storyline_review import StorylineReviewReport, review_storyline
from littrace.evidence.tables import (
    ArtifactNeedReport,
    decide_artifact_extraction_need,
)


router = APIRouter()


@router.post("/parse/context", response_model=LiteratureWorkspace)
async def parse_context() -> LiteratureWorkspace:
    workspace, _ = await parse_workspace_skill(api_app.WORKSPACE, api_app.load_config())
    api_app._set_workspace(workspace)
    return api_app.WORKSPACE


@router.post("/tables/extract", response_model=ResearchRunResult)
async def tables_extract() -> ResearchRunResult:
    workspace, harness = await extract_tables_skill(api_app.WORKSPACE, api_app.load_config())
    api_app._set_workspace(workspace)
    return ResearchRunResult(
        workspace=api_app.WORKSPACE,
        table_harness=harness.model_dump(),
        comparison_matrix=build_comparison_matrix_skill(api_app.WORKSPACE),
    )


@router.get("/tables/matrix", response_model=ComparisonMatrixReport)
def tables_matrix() -> ComparisonMatrixReport:
    return build_comparison_matrix_skill(api_app.WORKSPACE)


@router.get("/artifacts/need", response_model=ArtifactNeedReport)
def artifacts_need() -> ArtifactNeedReport:
    return decide_artifact_extraction_need(api_app.WORKSPACE)


@router.get("/artifacts/{artifact_id}/download-link", response_model=object)
def artifact_download_link(
    artifact_id: str,
    session_id: str,
    expires_seconds: int = 3600,
    x_littrace_session_id: Annotated[str | None, Header(alias="X-LitTrace-Session-Id")] = None,
) -> dict[str, object]:
    config = api_app.load_config()
    auth = resolve_request_session(
        config,
        route_session_id=session_id,
        header_session_id=x_littrace_session_id,
    )
    registry = artifact_registry_from_config(config)
    record = registry.find_in_session(artifact_id, session_id=auth.session_id)
    if record is None:
        return {"found": False, "authorized": False, "artifact_id": artifact_id}
    ref = BlobRef(
        backend=record.backend,
        bucket=record.bucket,
        object_key=record.object_key,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        content_type=record.content_type,
        metadata={
            "session_id": record.session_id,
            "kind": record.kind,
        },
    )
    return {
        "found": True,
        "authorized": True,
        "artifact_id": artifact_id,
        "kind": record.kind,
        "download_url": artifact_store_from_config(config).signed_url(
            ref,
            expires_seconds=expires_seconds,
        ),
        "content_type": record.content_type,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    }


@router.get("/storyline/report")
def storyline_report() -> dict[str, object]:
    markdown, report = render_publication_storyline(api_app.WORKSPACE, api_app.load_config())
    return {
        "markdown": markdown,
        "release_ready": report.release_ready,
        "release_blockers": report.release_blockers,
    }


@router.get("/reports/research", response_model=ResearchDocumentReport)
async def research_report(title: str | None = None) -> ResearchDocumentReport:
    report = await build_research_report_skill(
        api_app.WORKSPACE,
        api_app.load_config(),
        title=title,
    )
    api_app.WORKSPACE.context.filters.document_report = report.model_dump(mode="json")
    return report


@router.get("/storyline/review", response_model=StorylineReviewReport)
def storyline_review() -> StorylineReviewReport:
    return review_storyline(api_app.WORKSPACE)
