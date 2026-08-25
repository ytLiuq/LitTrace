"""Round 10 tests for the rollout JSONL → harness converter.

Tests cover the converter directly (without booting a real
App Server) and exercise the end-to-end path through the
existing ``HarnessEngine.run`` API. A small fixture builder
writes a known-good rollout file, then the assertions check
that ``RolloutConverter`` produces the expected
``RolloutEvaluationBundle`` shape.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from littrace.evaluation.harnesses import HarnessConfig, HarnessEngine
from littrace.evaluation.rollout_eval import (
    RolloutConverter,
    convert_directory,
    convert_file,
    merge_bundles,
)
from littrace.evaluation.harnesses import RetryHealthItem
from littrace.models import LinkStatus

pytestmark = pytest.mark.eval


def _write_rollout(path: Path, lines: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _fixture_lines() -> list[dict[str, object]]:
    """A representative rollout with one full turn, two tool
    calls (one repeated → counted as a retry by the
    converter), one ``item/completed`` carrying two cited
    paper_ids, and a separate ``system_error`` for a second
    turn that never finished.
    """
    return [
        {
            "type": "session_meta", "ts": "2026-01-01T00:00:00Z",
            "session_id": "sess-1", "codex_thread_id": "thr-1",
        },
        {
            "type": "turn_start", "ts": "2026-01-01T00:00:01Z",
            "turn_id": "t-1", "thread_id": "thr-1",
            "user_text": "hello",
        },
        {
            "type": "event", "ts": "2026-01-01T00:00:02Z",
            "turn_id": "t-1", "method": "search_papers",
            "params": {"q": "MXene"},
        },
        {
            "type": "event", "ts": "2026-01-01T00:00:02Z",
            "turn_id": "t-1", "method": "search_papers",
            "params": {"q": "MXene"},
        },
        {
            "type": "event", "ts": "2026-01-01T00:00:03Z",
            "turn_id": "t-1", "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "cited_paper_ids": ["p1", "p2"],
                    "text": "answer text",
                }
            },
        },
        {
            "type": "turn_complete", "ts": "2026-01-01T00:00:04Z",
            "turn_id": "t-1", "status": "completed", "reply": "answer text",
        },
        {
            "type": "session_meta", "ts": "2026-01-02T00:00:00Z",
            "session_id": "sess-2", "codex_thread_id": "thr-2",
        },
        {
            "type": "turn_start", "ts": "2026-01-02T00:00:01Z",
            "turn_id": "t-2", "thread_id": "thr-2",
            "user_text": "q2",
        },
        {
            "type": "system_error", "ts": "2026-01-02T00:00:02Z",
            "turn_id": "t-2", "error_code": "internal_server_error",
            "message": "boom",
        },
    ]


def test_convert_file_produces_typed_bundle(tmp_path: Path) -> None:
    rollout = tmp_path / "sess-1.jsonl"
    _write_rollout(rollout, _fixture_lines())

    bundle = convert_file(rollout)
    # The fixture contains two session_meta events; the
    # converter keeps the LAST one, matching how the rollout
    # file records successive LitTrace sessions in a single
    # JSONL. Operators who need per-session bundling should
    # split the file before calling ``convert_file``.
    assert bundle.session_id == "sess-2"
    assert bundle.thread_id == "thr-2"
    assert len(bundle.turns) == 2
    turn_by_id = {turn.turn_id: turn for turn in bundle.turns}
    assert turn_by_id["t-1"].status == "completed"
    assert turn_by_id["t-1"].reply == "answer text"
    assert turn_by_id["t-1"].tool_call_count == 1  # deduped by method
    assert turn_by_id["t-2"].error_code == "internal_server_error"
    # Citations extracted from the item/completed agentMessage.
    assert len(bundle.citations) == 2
    assert {c.paper_id for c in bundle.citations} == {"p1", "p2"}
    assert all(c.link_status == LinkStatus.UNCHECKED for c in bundle.citations)
    # Retries: search_papers fired twice in t-1, so one retry
    # of one (delta = 1, total = 2, retry_rate = 0.5).
    assert len(bundle.retries) == 1
    retry = bundle.retries[0]
    assert retry.operation == "search_papers"
    assert retry.total_calls == 2
    assert retry.total_retries == 1
    assert retry.retry_rate == 0.5
    assert retry.failure_rate == 0.0
    # Tool-call records carry every event the rollout pushed.
    assert len(bundle.tool_calls) == 3
    assert bundle.tool_calls[0].method == "search_papers"
    # Errors surface as a flat list so a downstream check can
    # assert on the surface without re-parsing the trace.
    assert len(bundle.errors) == 1
    assert bundle.errors[0]["error_code"] == "internal_server_error"


def test_convert_directory_recursively_merges_bundles(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "sub"
    _write_rollout(tmp_path / "a.jsonl", _fixture_lines())
    _write_rollout(nested / "b.jsonl", _fixture_lines())
    # A non-rollout file under the root must be ignored.
    (tmp_path / "README.md").write_text("ignore me", encoding="utf-8")

    bundles = convert_directory(tmp_path)
    assert len(bundles) == 2
    merged = merge_bundles(bundles)
    assert len(merged["check_citations"]) == 4
    assert len(merged["__turns__"]) == 4
    # search_papers appears in both files but the merge
    # collapses them into one bucket per operation.
    retry_operations = {r.operation for r in merged["check_retry_health"]}
    assert retry_operations == {"search_papers"}
    # Errors dedup by (turn_id, error_code) so the same
    # system_error appearing in two files does not double-count.
    assert len(merged["__errors__"]) == 1


def test_to_check_items_routes_into_harness_engine(tmp_path: Path) -> None:
    """The end-to-end path: converter -> bundle -> items map ->
    HarnessEngine.run with the standard check names.
    """
    rollout = tmp_path / "sess-1.jsonl"
    _write_rollout(rollout, _fixture_lines())

    bundle = convert_file(rollout)
    items_map = bundle.to_check_items()
    assert "check_citations" in items_map
    assert "check_retry_health" in items_map
    # The two extra keys are reserved for operator-defined
    # custom checks and never consumed by the canonical engine.
    assert "__turns__" in items_map
    assert "__tool_calls__" in items_map

    engine = HarnessEngine()
    citations_report = engine.run("check_citations", items_map["check_citations"])
    # Both citations carry ``UNCHECKED`` so the standard
    # ``check_citations`` check flags them and the report
    # fails.
    assert citations_report.item_count == 2
    assert not citations_report.passed
    assert any(
        "access link is not verified" in f.message
        for f in citations_report.findings
    )

    retries_report = engine.run(
        "check_retry_health", items_map["check_retry_health"],
    )
    assert retries_report.item_count == 1
    # search_papers fired twice in t-1 → retry_rate = 0.5
    # which is right at the default threshold so no error
    # findings fire.
    assert retries_report.passed


def test_permissive_reader_skips_malformed_lines(tmp_path: Path) -> None:
    rollout = tmp_path / "sess-1.jsonl"
    _write_rollout(rollout, _fixture_lines()[:1])  # only session_meta
    # Append a malformed line and a partial JSON.
    with rollout.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
        fh.write("{this is : not, valid json}\n")
        fh.write(json.dumps(_fixture_lines()[1]) + "\n")
    bundle = convert_file(rollout)
    # The valid events are still extracted; the malformed
    # lines are dropped without aborting the conversion.
    assert bundle.session_id == "sess-1"
    assert bundle.turns[0].turn_id == "t-1"


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    assert convert_directory(tmp_path) == []


def test_converter_picks_up_optional_access_url(tmp_path: Path) -> None:
    """When the rollout captures an access_url, the converter
    forwards it; otherwise the placeholder URL is used.
    """
    lines = _fixture_lines()
    # Replace the second search_papers event with an
    # item/completed that carries an explicit access_url.
    lines[4] = {
        "type": "event", "ts": "2026-01-01T00:00:03Z",
        "turn_id": "t-1", "method": "item/completed",
        "params": {
            "item": {
                "type": "agentMessage",
                "cited_paper_ids": ["p1"],
                "text": "answer",
                "access_url": "https://example.com/p1.pdf",
            }
        },
    }
    rollout = tmp_path / "sess-1.jsonl"
    _write_rollout(rollout, lines)
    bundle = convert_file(rollout)
    assert len(bundle.citations) == 1
    assert str(bundle.citations[0].access_url) == "https://example.com/p1.pdf"


def test_retry_aggregation_distinct_turns(tmp_path: Path) -> None:
    """A tool call repeated across different turns is NOT
    treated as a retry (each turn is a fresh invocation).
    """
    lines = _fixture_lines()
    # Remove the duplicate search_papers so the only repeat
    # is the second-turn search_papers added below.
    lines.pop(3)
    lines.insert(5, {
        "type": "event", "ts": "2026-01-02T00:00:01Z",
        "turn_id": "t-2", "method": "search_papers",
        "params": {"q": "MXene"},
    })
    rollout = tmp_path / "sess-1.jsonl"
    _write_rollout(rollout, lines)
    bundle = convert_file(rollout)
    search = next(r for r in bundle.retries if r.operation == "search_papers")
    assert search.total_calls == 2
    assert search.total_retries == 0
    assert search.retry_rate == 0.0
