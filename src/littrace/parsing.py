from __future__ import annotations

from pathlib import Path

from littrace.access import paper_storage_dir
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.ocr.registry import build_ocr_tool
from littrace.ocr.tool import OCRMode, OCRTool


def parse_workspace_papers(
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    tool: OCRTool | None = None,
    mode: OCRMode = OCRMode.ACCURATE,
) -> tuple[LiteratureWorkspace, dict[str, object]]:
    paper_lookup = workspace.papers
    parser = tool or build_ocr_tool(config, paper_lookup)
    parsed_count = 0
    metadata_only_count = 0
    missing_pdf_count = 0

    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        pdf_path = local_pdf_path(config, paper)
        if not pdf_path.exists():
            missing_pdf_count += 1
            pdf_path = Path(f"{paper.paper_id}.pdf")
        parsed = parser.parse_pdf(pdf_path, mode=mode)
        workspace.parsed_papers[paper_id] = parsed.model_dump(mode="json")
        if parsed.parsed:
            parsed_count += 1
        else:
            metadata_only_count += 1

    return workspace, {
        "parser": getattr(parser, "name", parser.__class__.__name__),
        "active_papers": len(workspace.context.active_papers),
        "parsed_count": parsed_count,
        "metadata_only_count": metadata_only_count,
        "missing_pdf_count": missing_pdf_count,
    }


def local_pdf_path(config: LitTraceConfig, paper: PaperMetadata) -> Path:
    return paper_storage_dir(config, paper) / "paper.pdf"
