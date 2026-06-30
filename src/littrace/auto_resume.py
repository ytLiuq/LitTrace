from __future__ import annotations

from pydantic import BaseModel, Field

from littrace.attachments import check_download_presence
from littrace.config import LitTraceConfig
from littrace.export import export_session_bundle
from littrace.models import LiteratureWorkspace
from littrace.parsing import parse_workspace_papers
from littrace.session import ChatSession
from littrace.tables import build_comparison_matrices, extract_performance_cells


class AutoResumeResult(BaseModel):
    ready_to_parse_count: int
    parsed_count: int
    performance_cell_count: int
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def auto_resume_downloaded_pdfs(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    session: ChatSession | None = None,
) -> tuple[LiteratureWorkspace, AutoResumeResult]:
    presence = check_download_presence(config, workspace)
    warnings = list(presence.warnings)
    if presence.ready_to_parse_count:
        workspace, parse_report = parse_workspace_papers(workspace, config)
        workspace, table_harness = extract_performance_cells(workspace)
        matrix = build_comparison_matrices(workspace)
        warnings.extend(parse_report.get("warnings", []))
        warnings.extend(table_harness.warnings)
        warnings.extend(matrix.warnings)
    else:
        parse_report = {"parsed_count": 0}

    artifacts = export_session_bundle(session, workspace) if session else {}
    result = AutoResumeResult(
        ready_to_parse_count=presence.ready_to_parse_count,
        parsed_count=int(parse_report.get("parsed_count") or 0),
        performance_cell_count=len(workspace.performance_cells),
        artifact_paths=artifacts,
        warnings=warnings,
    )
    return workspace, result
