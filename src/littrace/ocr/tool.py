from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from littrace.models import EvidenceSpan


class OCRMode(StrEnum):
    FAST = "fast"
    ACCURATE = "accurate"
    TABLES = "tables"
    EQUATIONS = "equations"
    FIGURES = "figures"


class ParsedTable(BaseModel):
    table_id: str
    caption: str | None = None
    cells: list[dict[str, object]] = Field(default_factory=list)
    evidence: EvidenceSpan


class ParsedPaper(BaseModel):
    pdf_path: Path
    title: str | None = None
    abstract: str | None = None
    sections: list[dict[str, object]] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    figures: list[dict[str, object]] = Field(default_factory=list)
    equations: list[dict[str, object]] = Field(default_factory=list)
    parser_reports: list[dict[str, object]] = Field(default_factory=list)
    parsed: bool = False
    error: str | None = None


class OCRTool(Protocol):
    name: str

    def parse_pdf(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        """Parse a PDF and return provenance-rich paper structure."""
