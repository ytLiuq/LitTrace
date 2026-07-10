from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
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
        self.progress_callback: Any | None = None

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
            self._emit_progress(page_number, len(page_images), parsed_page)
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

    def _emit_progress(
        self,
        page_number: int,
        total_pages: int,
        parsed_page: ParsedPaper,
    ) -> None:
        if not self.progress_callback:
            return
        event = {
            "parser": self.name,
            "page": page_number,
            "total_pages": total_pages,
            "parsed": parsed_page.parsed,
            "section_count": len(parsed_page.sections),
            "error": parsed_page.error,
        }
        self.progress_callback(event)

    def parse_images_batch(
        self,
        image_paths: list[Path],
        mode: OCRMode = OCRMode.ACCURATE,
        preferred_engines: list[str] | None = None,
    ) -> list[ParsedPaper]:
        if self.config.ocr_page_workers > 1:
            return self._parse_images_parallel(image_paths, mode, preferred_engines)

        valid_paths = [path for path in image_paths if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(valid_paths) != len(image_paths):
            return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        # Obtain an OCR engine.  If paddleocr is not importable we fall back
        # to per-image parsing, unless an engine has been injected (e.g. via
        # monkeypatch or DI), in which case we use it directly.
        ocr: Any = None
        try:
            from paddleocr import PaddleOCR

            ocr = self._get_ocr_engine(PaddleOCR)
        except ImportError:
            # If _get_ocr_engine was overridden (e.g. monkeypatched) it may
            # still return a usable engine without the import.
            try:
                ocr = self._get_ocr_engine(None)
            except Exception:
                pass
            if ocr is None:
                return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        if not hasattr(ocr, "predict"):
            return [self.parse_image(path, mode, preferred_engines) for path in image_paths]

        parsed_pages: list[ParsedPaper] = []
        for batch in _chunks(image_paths, max(self.config.ocr_batch_size, 1)):
            cached_pages: list[ParsedPaper | None] = [
                self._read_cached_page(path, mode, preferred_engines, batched=True)
                for path in batch
            ]
            missing = [
                (index, path)
                for index, (path, cached) in enumerate(zip(batch, cached_pages, strict=False))
                if cached is None
            ]
            raw_pages_by_index: dict[int, Any] = {}
            if missing:
                raw_results = ocr.predict(
                    [str(path) for _, path in missing],
                    use_textline_orientation=self.config.use_angle_cls,
                )
                raw_pages = _align_batch_results(raw_results, len(missing))
                raw_pages_by_index = {
                    index: raw_page
                    for (index, _), raw_page in zip(missing, raw_pages, strict=False)
                }
            for index, image_path in enumerate(batch):
                cached = cached_pages[index]
                if cached is not None:
                    parsed_pages.append(cached)
                    continue
                raw_page = raw_pages_by_index.get(index)
                lines = normalize_paddleocr_result(raw_page)
                parsed_page = _parsed_paper_from_lines(
                    image_path,
                    lines,
                    mode,
                    preferred_engines,
                    self.name,
                    batched=True,
                )
                self._write_cached_page(image_path, mode, preferred_engines, parsed_page)
                parsed_pages.append(parsed_page)
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
            cached = self._read_cached_page(image_path, mode, preferred_engines, batched=False)
            if cached is not None:
                return cached
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
            parsed = _parsed_paper_from_lines(
                image_path,
                lines,
                mode,
                preferred_engines,
                self.name,
                batched=False,
            )
            self._write_cached_page(image_path, mode, preferred_engines, parsed)
            return parsed
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

    def _parse_images_parallel(
        self,
        image_paths: list[Path],
        mode: OCRMode,
        preferred_engines: list[str] | None,
    ) -> list[ParsedPaper]:
        workers = max(1, self.config.ocr_page_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda path: PaddleOCRTool(self.config).parse_image(
                        path,
                        mode,
                        preferred_engines,
                    ),
                    image_paths,
                )
            )

    def _cache_path(
        self,
        image_path: Path,
        mode: OCRMode,
        preferred_engines: list[str] | None,
    ) -> Path:
        root = self.config.cache_dir or Path(".littrace-cache") / "paddleocr"
        digest = hashlib.sha256()
        digest.update(image_path.read_bytes())
        digest.update(str(mode).encode())
        digest.update(self.config.lang.encode())
        digest.update(str(self.config.use_angle_cls).encode())
        digest.update(",".join(preferred_engines or []).encode())
        return root / f"{digest.hexdigest()}.json"

    def _read_cached_page(
        self,
        image_path: Path,
        mode: OCRMode,
        preferred_engines: list[str] | None,
        batched: bool,
    ) -> ParsedPaper | None:
        if not self.config.cache_enabled or not image_path.exists():
            return None
        path = self._cache_path(image_path, mode, preferred_engines)
        if not path.exists():
            return None
        try:
            parsed = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        parsed.parser_reports.append(
            {
                "parser": self.name,
                "mode": mode,
                "preferred_engines": preferred_engines or [],
                "cache_hit": True,
                "batched": batched,
            }
        )
        return parsed

    def _write_cached_page(
        self,
        image_path: Path,
        mode: OCRMode,
        preferred_engines: list[str] | None,
        parsed: ParsedPaper,
    ) -> None:
        if not self.config.cache_enabled or not image_path.exists() or not parsed.parsed:
            return
        path = self._cache_path(image_path, mode, preferred_engines)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(parsed.model_dump_json(), encoding="utf-8")


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
