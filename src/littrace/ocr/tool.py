from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from littrace.models import EvidenceSpan, ParsedPaper, ParsedTable  # noqa: F401


class OCRMode(StrEnum):
    FAST = "fast"
    ACCURATE = "accurate"
    TABLES = "tables"
    EQUATIONS = "equations"
    FIGURES = "figures"


# Re-export for backward compatibility
__all__ = ["OCRMode", "ParsedTable", "ParsedPaper", "OCRTool"]


class OCRTool(Protocol):
    name: str

    def parse_pdf(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        """Parse a PDF and return provenance-rich paper structure."""
