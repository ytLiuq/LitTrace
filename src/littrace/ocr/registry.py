from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.models import PaperMetadata
from littrace.ocr.docling_adapter import DoclingOCRTool
from littrace.ocr.metadata_only import MetadataOnlyOCRTool
from littrace.ocr.paddleocr_adapter import PaddleOCRTool
from littrace.ocr.tool import OCRTool


def build_ocr_tool(
    config: LitTraceConfig,
    paper_lookup: dict[str, PaperMetadata] | None = None,
) -> OCRTool:
    backend = config.parsing.default_parser.lower()
    if backend == "docling":
        return DoclingOCRTool()
    if backend in {"paddleocr", "paddlerocr"}:
        return PaddleOCRTool(config.parsing.paddleocr)
    return MetadataOnlyOCRTool(paper_lookup)
