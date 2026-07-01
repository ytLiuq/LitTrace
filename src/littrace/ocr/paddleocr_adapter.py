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
        self._ocr_engine: Any | None = None

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
                sections, reports = self._parse_page_images(
                    pdf_path,
                    page_images,
                    mode=mode,
                    preferred_engines=preferred_engines,
                )
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
                            "ocr_batch_size": self.config.ocr_batch_size,
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

    def _parse_page_images(
        self,
        pdf_path: Path,
        page_images: list[tuple[int, Path]],
        mode: OCRMode,
        preferred_engines: list[str] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        sections: list[dict[str, object]] = []
        reports: list[dict[str, object]] = []
        try:
            batch_results = self.parse_images_batch(
                [image_path for _, image_path in page_images],
                mode=mode,
                preferred_engines=preferred_engines,
            )
        except Exception as exc:
            batch_results = []
            reports.append(
                {
                    "parser": self.name,
                    "mode": mode,
                    "preferred_engines": preferred_engines or [],
                    "batch_error": f"{exc.__class__.__name__}: {exc}",
                    "fallback": "sequential_parse_image",
                }
            )
            for _, image_path in page_images:
                batch_results.append(self.parse_image(image_path, mode, preferred_engines))

        for (page_number, _), parsed_page in zip(page_images, batch_results, strict=False):
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
        return sections, reports

    def parse_images_batch(
        self,
        image_paths: list[Path],
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> list[ParsedPaper]:
        valid_paths = [path for path in image_paths if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(valid_paths) != len(image_paths):
            return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        ocr = self._get_ocr_engine(PaddleOCR)
        if not hasattr(ocr, "predict"):
            return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        parsed_pages: list[ParsedPaper] = []
        for batch in _chunks(image_paths, max(self.config.ocr_batch_size, 1)):
            raw_results = ocr.predict(
                [str(path) for path in batch],
                use_textline_orientation=self.config.use_angle_cls,
            )
            raw_pages = _align_batch_results(raw_results, len(batch))
            for image_path, raw_page in zip(batch, raw_pages, strict=False):
                parsed_pages.append(
                    _parsed_paper_from_lines(
                        image_path,
                        normalize_paddleocr_result(raw_page),
                        mode,
                        preferred_engines,
                        self.name,
                        batched=True,
                    )
                )
        return parsed_pages

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
            ocr = self._get_ocr_engine(PaddleOCR)
            if hasattr(ocr, "predict"):
                raw_result = ocr.predict(
                    str(image_path),
                    use_textline_orientation=self.config.use_angle_cls,
                )
            else:
                raw_result = ocr.ocr(str(image_path), cls=self.config.use_angle_cls)
            lines = normalize_paddleocr_result(raw_result)
            return _parsed_paper_from_lines(
                image_path,
                lines,
                mode,
                preferred_engines,
                self.name,
                batched=False,
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

    def _get_ocr_engine(self, paddleocr_cls: Any) -> Any:
        if self._ocr_engine is None:
            self._ocr_engine = paddleocr_cls(
                use_textline_orientation=self.config.use_angle_cls,
                lang=self.config.lang,
            )
        return self._ocr_engine


def normalize_paddleocr_result(raw_result: Any) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    pages = _as_pages(raw_result)
    for page in pages:
        if isinstance(page, dict):
            lines.extend(_normalize_paddleocr_v3_page(page))
            continue
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


def _parsed_paper_from_lines(
    image_path: Path,
    lines: list[dict[str, object]],
    mode: OCRMode,
    preferred_engines: list[str] | None,
    parser_name: str,
    batched: bool,
) -> ParsedPaper:
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
                    parser=parser_name,
                    confidence=_average_confidence(lines),
                ).model_dump(),
            }
        ]
        if text
        else [],
        parser_reports=[
            {
                "parser": parser_name,
                "mode": mode,
                "preferred_engines": preferred_engines or [],
                "line_count": len(lines),
                "batched": batched,
            }
        ],
        parsed=bool(text),
        error=None if text else "No OCR text was extracted.",
    )


def _chunks(values: list[Path], size: int) -> list[list[Path]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _align_batch_results(raw_results: Any, expected: int) -> list[Any]:
    if isinstance(raw_results, list) and len(raw_results) == expected:
        return raw_results
    if expected == 1:
        return [raw_results]
    return list(raw_results) if isinstance(raw_results, list) else [raw_results]


def _normalize_paddleocr_v3_page(page: dict[str, Any]) -> list[dict[str, object]]:
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    boxes = page.get("rec_polys") or page.get("rec_boxes") or []
    lines: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        if text is None:
            continue
        confidence = scores[index] if index < len(scores) else 0.0
        bbox = boxes[index] if index < len(boxes) else None
        lines.append(
            {
                "text": str(text),
                "confidence": float(confidence or 0.0),
                "bbox": _jsonable_bbox(bbox),
            }
        )
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


def _jsonable_bbox(value: Any) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _average_confidence(lines: list[dict[str, object]]) -> float:
    if not lines:
        return 0.0
    return sum(float(line.get("confidence") or 0.0) for line in lines) / len(lines)
