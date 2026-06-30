from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig


class GoldenEvalReport(BaseModel):
    golden_set_dir: str
    case_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def run_golden_eval(config: LitTraceConfig) -> GoldenEvalReport:
    root = config.eval.golden_set_dir
    cases = _load_cases(root)
    warnings: list[str] = []
    if not root.exists():
        warnings.append(f"Golden set directory does not exist: {root}")
    if not cases:
        warnings.append("No golden cases found. Add JSONL files under eval/golden.")
    metrics = {
        "case_count": float(len(cases)),
        "has_expected_doi_rate": _rate(cases, "expected_dois"),
        "has_expected_metrics_rate": _rate(cases, "expected_metrics"),
        "has_expected_storyline_rate": _rate(cases, "expected_storyline_claims"),
    }
    return GoldenEvalReport(
        golden_set_dir=str(root),
        case_count=len(cases),
        metrics=metrics,
        warnings=warnings,
    )


def _load_cases(root: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    if not root.exists():
        return cases
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def _rate(cases: list[dict[str, object]], key: str) -> float:
    if not cases:
        return 0.0
    return sum(bool(case.get(key)) for case in cases) / len(cases)
