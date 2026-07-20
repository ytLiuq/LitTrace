from __future__ import annotations

from fastapi import APIRouter

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


class _AppProxy:
    def __getattr__(self, name: str):
        from littrace.api import app as api_app

        return getattr(api_app, name)


api_app = _AppProxy()
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
