"""Process-isolated single-PDF Docling worker.

The parent parser invokes this module for text-only batches so a native
Docling/PyTorch failure is contained to one PDF instead of terminating the
whole retrieval/RAG process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from littrace.config import LitTraceConfig
from littrace.models import ParsedPaper
from littrace.ocr.docling_adapter import DoclingOCRTool
from littrace.ocr.tool import OCRMode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--mode", choices=[mode.value for mode in OCRMode], default=OCRMode.FAST.value)
    args = parser.parse_args()
    config = LitTraceConfig()
    config.parsing.parse_strategy = "text_only" if args.mode == OCRMode.FAST.value else "auto"
    result = DoclingOCRTool(config).parse_pdf(args.pdf, mode=OCRMode(args.mode))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
    return 0 if result.parsed else 1


if __name__ == "__main__":
    raise SystemExit(main())
