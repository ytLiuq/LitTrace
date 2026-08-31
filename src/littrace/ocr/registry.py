from __future__ import annotations

import logging
from typing import Iterable

from littrace.config import LitTraceConfig
from littrace.models import PaperMetadata
from littrace.ocr.docling_adapter import DoclingOCRTool
from littrace.ocr.paddleocr_adapter import PaddleOCRTool
from littrace.ocr.tool import OCRTool

logger = logging.getLogger(__name__)


def _importable(module: str) -> bool:
    """Return True if ``module`` can be imported in the current environment.

    Used by the parser registry to gracefully fall back when the configured
    backend is missing its heavy dependency (e.g. ``paddleocr`` not installed
    after ``pip install -e .`` without the ``[parsers]`` extra). Before this
    fallback, littrace silently produced zero parsed papers — the
    PaddleOCRTool constructed successfully but ``parse_pdf`` returned
    ``parsed=False, error="PaddleOCR is not installed"`` for every paper.
    """
    import importlib

    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def build_ocr_tool(
    config: LitTraceConfig,
    paper_lookup: dict[str, PaperMetadata] | None = None,
) -> OCRTool:
    strategy = config.parsing.parse_strategy.lower()
    backend = config.parsing.default_parser.lower()
    if strategy in {"text_only", "text-only", "text"}:
        backend = "docling"
    elif strategy in {"ocr", "paddleocr", "paddlerocr"}:
        backend = "paddleocr"

    if backend in {"docling", "text_only", "text-only", "text"}:
        return DoclingOCRTool(config)
    if backend in {"paddleocr", "paddlerocr"}:
        if _importable("paddleocr") and _importable("pypdfium2"):
            return PaddleOCRTool(config.parsing.paddleocr)
        # Fallback: paddleocr (or its pdf renderer) is not installed. Walk
        # ``preferred_engines`` and pick the first one whose heavy deps are
        # importable. ``docling`` ships with ``docling`` + ``pypdfium2``
        # only, so it works without the ``[parsers]`` extra.
        fallback_chain: Iterable[str] = list(config.parsing.preferred_engines or [])
        for candidate in fallback_chain:
            if candidate in {"docling", "text_only", "text-only", "text"} and _importable(
                "docling"
            ):
                logger.warning(
                    "parsing.default_parser='paddleocr' but the paddleocr package "
                    "is not importable. Switching to '%s' for this session. Install "
                    "paddleocr via `pip install -e '.[parsers]'` to use the "
                    "configured backend.",
                    candidate,
                )
                return DoclingOCRTool(config)
        raise RuntimeError(
            "paddleocr is configured as the default parser but neither paddleocr "
            "nor any preferred fallback (e.g. docling) is importable. Run "
            "`pip install -e '.[parsers]'` or set parsing.default_parser to "
            "an available backend."
        )
    raise ValueError(
        f"Unsupported parser backend '{config.parsing.default_parser}'. "
        "Configure parsing.default_parser as docling or paddleocr."
    )
