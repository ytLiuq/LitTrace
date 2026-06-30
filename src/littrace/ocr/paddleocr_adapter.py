from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from littrace.config import PaddleOCRParserConfig
from littrace.models import EvidenceSpan
from littrace.ocr.tool import OCRMode, ParsedPaper


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class PaddleOCRTool:
    name = "paddleocr"

    def __init__(self, config: PaddleOCRParserConfig | None = None):
        self.config = config or PaddleOCRParserConfig()

    def parse_pdf(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        if pdf_path.suffix.lower() == ".pdf":
            return self.parse_pdf_pages(pdf_path, mode=mode, preferred_engines=preferred_engines)
        return self.parse_image(pdf_path, mode=mode, preferred_engines=preferred_engines)

    def parse_pdf_pages(
        self,
        pdf_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        if not pdf_path.exists():
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "PDF file does not exist.",
                    }
                ],
                parsed=False,
                error="PDF file does not exist.",
            )
        try:
            with TemporaryDirectory(prefix="littrace-paddleocr-") as tmp_dir:
                page_images = render_pdf_pages_to_images(
                    pdf_path,
                    Path(tmp_dir),
                    scale=self.config.pdf_render_scale,
                    max_pages=self.config.max_pages,
                )
                sections: list[dict[str, object]] = []
                reports: list[dict[str, object]] = []
                for page_number, image_path in page_images:
                    parsed_page = self.parse_image(
                        image_path,
                        mode=mode,
                        preferred_engines=preferred_engines,
                    )
                    reports.extend(parsed_page.parser_reports)
                    for section in parsed_page.sections:
                        section = dict(section)
                        section["name"] = f"page_{page_number}_ocr_text"
                        evidence = dict(section.get("evidence") or {})
                        evidence["paper_id"] = pdf_path.stem
                        evidence["page"] = page_number
                        evidence["section"] = section["name"]
                        section["evidence"] = evidence
                        sections.append(section)
                return ParsedPaper(
                    pdf_path=pdf_path,
                    sections=sections,
                    parser_reports=[
                        {
                            "parser": self.name,
                            "mode": mode,
                            "preferred_engines": preferred_engines or [],
                            "pdf_pages_rendered": len(page_images),
                            "pdf_render_scale": self.config.pdf_render_scale,
                        },
                        *reports,
                    ],
                    parsed=bool(sections),
                    error=None if sections else "No OCR text was extracted.",
                )
        except ImportError:
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "pypdfium2 is not installed. Install with: pip install -e '.[parsers]'",
                    }
                ],
                parsed=False,
                error="pypdfium2 is not installed.",
            )
        except Exception as exc:
            return ParsedPaper(
                pdf_path=pdf_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                ],
                parsed=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def parse_image(
        self,
        image_path: Path,
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> ParsedPaper:
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            return ParsedPaper(
                pdf_path=image_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": f"Unsupported raster image suffix: {image_path.suffix}",
                    }
                ],
                parsed=False,
                error="Unsupported image format for PaddleOCR.",
            )
        if not image_path.exists():
            return ParsedPaper(
                pdf_path=image_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "Image file does not exist.",
                    }
                ],
                parsed=False,
                error="Image file does not exist.",
            )

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return ParsedPaper(
                pdf_path=image_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": "PaddleOCR is not installed. Install with: pip install -e '.[parsers]'",
                    }
                ],
                parsed=False,
                error="PaddleOCR is not installed.",
            )

        try:
            ocr = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.lang,
            )
            raw_result = ocr.ocr(str(image_path), cls=self.config.use_angle_cls)
            lines = normalize_paddleocr_result(raw_result)
            text = "\n".join(line["text"] for line in lines)
            return ParsedPaper(
                pdf_path=image_path,
                sections=[
                    {
                        "name": "ocr_text",
                        "text": text,
                        "evidence": EvidenceSpan(
                            paper_id=image_path.stem,
                            section="ocr_text",
                            snippet=text[:500],
                            parser=self.name,
                            confidence=_average_confidence(lines),
                        ).model_dump(),
                    }
                ],
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "line_count": len(lines),
                    }
                ],
                parsed=True,
            )
        except Exception as exc:
            return ParsedPaper(
                pdf_path=image_path,
                parser_reports=[
                    {
                        "parser": self.name,
                        "mode": mode,
                        "preferred_engines": preferred_engines or [],
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                ],
                parsed=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )


def normalize_paddleocr_result(raw_result: Any) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    pages = _as_pages(raw_result)
    for page in pages:
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            bbox = item[0]
            payload = item[1]
            if not isinstance(payload, (list, tuple)) or not payload:
                continue
            text = str(payload[0])
            confidence = float(payload[1]) if len(payload) > 1 else 0.0
            lines.append({"text": text, "confidence": confidence, "bbox": bbox})
    return lines


def render_pdf_pages_to_images(
    pdf_path: Path,
    output_dir: Path,
    scale: float = 2.0,
    max_pages: int | None = None,
) -> list[tuple[int, Path]]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError("pypdfium2 is not installed.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    page_count = len(document)
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    rendered: list[tuple[int, Path]] = []
    for index in range(page_count):
        page = document[index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image_path = output_dir / f"page_{index + 1}.png"
        image.save(image_path)
        rendered.append((index + 1, image_path))
    return rendered


def _as_pages(raw_result: Any) -> list[Any]:
    if not isinstance(raw_result, list):
        return [raw_result]
    if _looks_like_ocr_item(raw_result):
        return [[raw_result]]
    if raw_result and all(_looks_like_ocr_item(item) for item in raw_result):
        return [raw_result]
    return raw_result


def _looks_like_ocr_item(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and bool(value[1])
        and isinstance(value[1][0], str)
    )


def _average_confidence(lines: list[dict[str, object]]) -> float:
    if not lines:
        return 0.0
    return sum(float(line.get("confidence") or 0.0) for line in lines) / len(lines)
