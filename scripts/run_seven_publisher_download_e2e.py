#!/usr/bin/env python3
"""Run real seven-publisher PDF download E2E with isolated output folders.

This script intentionally talks to the real network and a local Chrome CDP
browser. It is not a unit test and should be run only on a user's machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CASES: list[tuple[str, str]] = [
    ("wiley", "10.1002/mame.202400237"),
    ("springer_nature", "10.1038/srep14751"),
    ("mdpi", "10.3390/s23052443"),
    ("ieee", "10.1109/SENSORS43011.2019.8956652"),
    ("acs", "10.1021/acsomega.3c04786"),
    ("elsevier", "10.1016/j.matdes.2025.114201"),
    ("rsc", "10.1039/d2ma00987k"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/littrace-seven-publisher-e2e")
    parser.add_argument("--cdp", default="http://127.0.0.1:19222")
    parser.add_argument("--email", default="research@sjtu.edu.cn")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--user-wait-seconds", type=float, default=60.0)
    args = parser.parse_args()

    root = Path(args.out_dir).expanduser().resolve()
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "paper-pdf-downloader"
        / "scripts"
        / "universal_paper_downloader.py"
    )
    results = []
    for publisher, doi in CASES:
        case_dir = root / publisher
        case_dir.mkdir(parents=True, exist_ok=True)
        target = case_dir / f"{publisher}.pdf"
        log = case_dir / "download.log"
        if target.exists():
            target.unlink()
        command = [
            sys.executable,
            "-u",
            str(script),
            doi,
            "-o",
            str(target),
            "--cdp",
            args.cdp,
            "--email",
            args.email,
            "--user-wait-seconds",
            str(args.user_wait_seconds),
        ]
        started = time.monotonic()
        status = "failed"
        error = None
        try:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            log.write_text(
                (completed.stdout or "") + "\nSTDERR:\n" + (completed.stderr or ""),
                encoding="utf-8",
            )
            if completed.returncode == 0:
                status = "passed"
            else:
                error = f"returncode={completed.returncode}"
        except subprocess.TimeoutExpired as exc:
            log.write_text(
                (exc.stdout or "") + "\nSTDERR:\n" + (exc.stderr or "") + "\nTIMEOUT\n",
                encoding="utf-8",
            )
            status = "timeout"
            error = f"timeout after {args.timeout}s"

        verified_pdf = _is_pdf(target)
        size = target.stat().st_size if target.exists() else 0
        if status == "passed" and not verified_pdf:
            status = "failed"
            error = "script succeeded but output is not a verified PDF"
        item = {
            "publisher": publisher,
            "doi": doi,
            "status": status,
            "verified_pdf": verified_pdf,
            "size": size,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "target": str(target),
            "log": str(log),
            "error": error,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    summary = {
        "passed": sum(item["status"] == "passed" for item in results),
        "total": len(results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
