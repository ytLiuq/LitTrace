from __future__ import annotations

from pathlib import Path

from littrace.models import EvidenceSpan, PaperMetadata
from littrace.ocr.tool import OCRMode, OCRTool, ParsedPaper


class MetadataOnlyOCRTool:
    name = "metadata_only"

    def __init__(self, paper_lookup: dict[str, PaperMetadata] | None = None):
        self.paper_lookup = paper_lookup or {}

    def parse_pdf(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        paper = self.paper_lookup.get(pdf_path.stem)
        return ParsedPaper(
            pdf_path=pdf_path,
            title=paper.title if paper else None,
            abstract=paper.abstract if paper else None,
            sections=[
                {
                    "name": "metadata",
                    "text": paper.abstract or "",
                    "evidence": EvidenceSpan(
                        paper_id=paper.paper_id if paper else pdf_path.stem,
                        section="metadata",
                        snippet=paper.abstract if paper else None,
                        parser=self.name,
                        confidence=0.5 if paper and paper.abstract else 0.0,
                    ).model_dump(),
                }
            ]
            if paper and paper.abstract
            else [],
            parser_reports=[
                {
                    "parser": self.name,
                    "mode": mode,
                    "preferred_engines": preferred_engines or [],
                    "note": "No local PDF parser configured; metadata only.",
                }
            ],
            parsed=False,
            error="No local PDF parser configured; metadata only.",
        )


def default_ocr_tool(paper_lookup: dict[str, PaperMetadata] | None = None) -> OCRTool:
    return MetadataOnlyOCRTool(paper_lookup)
