#!/usr/bin/env python3
"""Benchmark real Docling parsing with a bounded worker pool."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from littrace.ocr.docling_adapter import DoclingOCRTool
from littrace.ocr.tool import OCRMode


def _parse(path: Path) -> dict[str, object]:
    started = time.perf_counter()
    parsed = DoclingOCRTool().parse_pdf(path, mode=OCRMode.ACCURATE)
    markdown = str(parsed.structured_document.get("markdown") or "")
    return {
        "path": str(path),
        "seconds": round(time.perf_counter() - started, 3),
        "parsed": parsed.parsed,
        "markdown_chars": len(markdown),
        "error": parsed.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    workers = max(1, args.workers)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_parse, path) for path in args.paths]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    successful = [result for result in results if result["parsed"]]
    print(
        json.dumps(
            {
                "workers": workers,
                "documents": len(results),
                "successful": len(successful),
                "elapsed_seconds": round(elapsed, 3),
                "throughput_documents_per_minute": round(len(successful) * 60 / elapsed, 3),
                "p95_seconds": sorted(result["seconds"] for result in results)[
                    max(0, int(len(results) * 0.95) - 1)
                ],
                "results": sorted(results, key=lambda result: str(result["path"])),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
