"""Unit tests for ``littrace.autonomous_loop.run_review_loop`` and helpers.

Strategy: ports-and-adapters. Build a real ``LitTraceConfig`` +
``LiteratureWorkspace``, monkeypatch the LLM, RAG, and skill collaborators
via the ``littrace.autonomous_loop`` namespace, then exercise each branch of
``run_review_loop`` and its 9 private helpers. Synchronous ``def test_*``
functions wrap ``asyncio.run(coro())`` — pytest-asyncio is intentionally NOT
a dependency for this project.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from littrace.autonomous_loop import (
    _execute_safe_replan_actions,
    _initial_draft,
    _rag_evidence_for_workspace,
    _revise_draft,
    _reviewer_payload,
    _round_score,
    _replan_actions,
    _run_optional_reviewer,
    _run_quality_gates,
    run_review_loop,
)
from littrace.citation_guard import CitationGuardReport, guard_citations
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.evaluation.harnesses import HarnessReport
from littrace.evaluation.quality_report import QualityReport
from littrace.llm import LLMReply
from littrace.models import (
    EvidenceSpan,
    FullTextResolutionReport,
    LiteratureWorkspace,
    PaperMetadata,
    ParsedPaper,
    PerformanceCell,
    ReviewFinding,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Builders — keep tests readable; matches the project fixture style.
# ---------------------------------------------------------------------------


def _paper(
    pid: str = "p1",
    title: str = "Traceable Sensor Paper",
    year: int = 2026,
    doi: str = "10.1000/example",
) -> PaperMetadata:
    return PaperMetadata(paper_id=pid, title=title, year=year, doi=doi)


def _parsed_paper() -> ParsedPaper:
    return ParsedPaper(
        parsed=True,
        title="Traceable Sensor Paper",
        sections=[{"name": "Results", "text": "Method improves sensitivity."}],
    )


def _cell(pid: str = "p1") -> PerformanceCell:
    return PerformanceCell(
        paper_id=pid,
        metric="gauge factor",
        value=12.0,
        evidence=EvidenceSpan(
            paper_id=pid,
            snippet="GF=12 at strain=1%",
            confidence=0.9,
            page=1,
            section="Results",
        ),
        higher_is_better=True,
    )


def _workspace(with_full_text: bool = True) -> LiteratureWorkspace:
    ws = add_papers(LiteratureWorkspace(), [_paper()])
    ws.parsed_papers["p1"] = _parsed_paper()
    ws.performance_cells = [_cell()]
    if with_full_text:
        ws.full_text_reports["p1"] = FullTextResolutionReport(paper_id="p1")
    return ws


def _config(
    llm_enabled: bool = True,
    api_key: str | None = "test-key",
    rag_enabled: bool = False,
    rag_backend: str = "pgvector",
) -> LitTraceConfig:
    cfg = LitTraceConfig(llm=LLMConfig(enabled=llm_enabled, api_key=api_key))
    cfg.rag.enabled = rag_enabled
    cfg.rag.backend = rag_backend
    return cfg


def _run(coro: Any) -> Any:
    """Wrap asyncio.run for synchronous tests."""
    return asyncio.run(coro)


def _harness_report(
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    passed: bool = True,
    score: float = 1.0,
    check_name: str = "test",
) -> HarnessReport:
    from littrace.evaluation.harnesses import HarnessFinding, Severity

    findings: list[HarnessFinding] = []
    for msg in errors or []:
        findings.append(HarnessFinding(severity=Severity.ERROR, message=msg))
    for msg in warnings or []:
        findings.append(HarnessFinding(severity=Severity.WARNING, message=msg))
    return HarnessReport(
        check_name=check_name,
        passed=passed and not findings,
        score=score,
        findings=findings,
        item_count=len(findings),
    )


# ---------------------------------------------------------------------------
# 1. run_review_loop: empty workspace
# ---------------------------------------------------------------------------


def test_review_loop_empty_workspace_returns_search_papers_replan():
    report = _run(
        run_review_loop(_config(), "总结当前文献", LiteratureWorkspace())
    )
    assert report.passed is False
    assert report.replan_actions == ["search_papers"]
    assert "empty_workspace" in report.warnings
    assert "无法启动审查流程" in report.final_answer
    assert report.rounds == []


# ---------------------------------------------------------------------------
# 2. _initial_draft: raises when LLM is unavailable
# ---------------------------------------------------------------------------


def test_initial_draft_raises_when_llm_disabled(monkeypatch):
    async def fake_writer(config, question, workspace, rag_evidence=None):
        return LLMReply(text="", used_llm=False, error="no api key")

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer
    )

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        _run(_initial_draft(_config(llm_enabled=True), "q", _workspace()))


def test_initial_draft_passes_through_text(monkeypatch):
    async def fake_writer(config, question, workspace, rag_evidence=None):
        return LLMReply(text="草稿内容。", used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer
    )

    text = _run(_initial_draft(_config(), "q", _workspace()))
    assert text == "草稿内容。"


# ---------------------------------------------------------------------------
# 3. _initial_draft: rag_evidence kwarg branch
# ---------------------------------------------------------------------------


def test_initial_draft_uses_rag_evidence_kwarg_when_writer_supports_it(monkeypatch):
    captured: dict[str, Any] = {}

    async def writer_with_rag(config, question, workspace, rag_evidence=None):
        captured["rag_evidence"] = rag_evidence
        captured["called"] = True
        return LLMReply(text="OK。", used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", writer_with_rag
    )

    evidence = [
        EvidenceSpan(paper_id="p1", snippet="RAG hit", parser="rag"),
    ]
    _run(_initial_draft(_config(), "q", _workspace(), rag_evidence=evidence))
    assert captured["called"]
    assert captured["rag_evidence"] is evidence


def test_initial_draft_falls_back_when_writer_signature_omits_rag_evidence(monkeypatch):
    captured: dict[str, Any] = {}

    async def writer_legacy(config, question, workspace):
        captured["called"] = True
        return LLMReply(text="OK。", used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", writer_legacy
    )

    _run(_initial_draft(_config(), "q", _workspace(), rag_evidence=[]))
    assert captured["called"]


# ---------------------------------------------------------------------------
# 4. _rag_evidence_for_workspace: disabled / wrong backend / error
# ---------------------------------------------------------------------------


def test_rag_evidence_disabled_when_backend_not_pgvector():
    cfg = _config(rag_enabled=True, rag_backend="weaviate")
    call_count = {"n": 0}

    async def should_not_run(*args, **kwargs):
        call_count["n"] += 1
        return None

    import littrace.autonomous_loop as loop_mod

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(loop_mod, "search_workspace_rag", should_not_run)
        result = _run(_rag_evidence_for_workspace(cfg, "q", _workspace()))
    finally:
        monkeypatch.undo()

    assert result == []
    assert call_count["n"] == 0


def test_rag_evidence_search_raises_returns_empty(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("pg down")

    monkeypatch.setattr("littrace.autonomous_loop.search_workspace_rag", boom)

    cfg = _config(rag_enabled=True, rag_backend="pgvector")
    assert _run(_rag_evidence_for_workspace(cfg, "q", _workspace())) == []


# ---------------------------------------------------------------------------
# 5. _run_quality_gates: Citation Gate
# ---------------------------------------------------------------------------


def test_quality_gates_citation_error_for_unsupported_sentence(monkeypatch):
    def fake_guard(text, workspace, *, claim_hints=None):
        return CitationGuardReport(
            passed=False,
            checked_sentence_count=1,
            unsupported_sentences=["该研究表明性能提升30%，缺少引用。"],
            warnings=[],
        )

    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations", fake_guard
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(
            metrics={"parsed_rate": 1.0}
        ),
    )

    findings = _run_quality_gates(
        "该研究表明性能提升30%，缺少引用。", _workspace(), _config()
    )
    citation_errors = [
        f
        for f in findings
        if f.reviewer == "Citation Gate" and f.severity == "error"
    ]
    assert citation_errors


# ---------------------------------------------------------------------------
# 6. _run_quality_gates: Storyline Gate
# ---------------------------------------------------------------------------


def test_quality_gates_storyline_translates_harness_to_errors_and_warnings(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.check_storyline_claims",
        lambda claims, config=None: _harness_report(
            errors=["发展脉络证据不足"],
            warnings=["缺少跨论文证据"],
            check_name="storyline",
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(
            metrics={"parsed_rate": 1.0}
        ),
    )

    findings = _run_quality_gates(
        "纯叙述段落不含主张。", _workspace(), _config()
    )
    story_errors = [
        f for f in findings
        if f.reviewer == "Storyline Gate" and f.severity == "error"
    ]
    story_warnings = [
        f for f in findings
        if f.reviewer == "Storyline Gate" and f.severity == "warning"
    ]
    assert len(story_errors) == 1 and "证据不足" in story_errors[0].finding
    assert len(story_warnings) == 1 and "缺少" in story_warnings[0].finding


# ---------------------------------------------------------------------------
# 7. _run_quality_gates: Table Gate (draft contains 性能 / 对比 / performance)
# ---------------------------------------------------------------------------


def test_quality_gates_table_gate_triggers_on_performance_keyword(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(
            metrics={"parsed_rate": 1.0}
        ),
    )

    ws = _workspace()
    ws.performance_cells = []  # no cells → Table Gate warning expected

    findings = _run_quality_gates(
        "本文比较了不同传感器的性能。", ws, _config()
    )
    table_warnings = [
        f for f in findings
        if f.reviewer == "Table Gate" and f.severity == "warning"
    ]
    assert table_warnings
    assert "performance cells" in table_warnings[0].finding or "可比" in table_warnings[0].finding


def test_quality_gates_table_gate_skipped_when_no_keyword(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(
            metrics={"parsed_rate": 1.0}
        ),
    )

    findings = _run_quality_gates(
        "纯叙述段落，不含性能对比。", _workspace(), _config()
    )
    assert not any(f.reviewer == "Table Gate" for f in findings)


# ---------------------------------------------------------------------------
# 8. _run_quality_gates: Readiness Gate + Quality Gates info
# ---------------------------------------------------------------------------


def test_quality_gates_readiness_warns_when_parsed_rate_zero(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(metrics={"parsed_rate": 0.0}),
    )

    ws = _workspace()
    findings = _run_quality_gates("段落。", ws, _config())
    readiness = [
        f for f in findings
        if f.reviewer == "Readiness Gate" and f.severity == "warning"
    ]
    assert readiness


def test_quality_gates_appends_info_when_no_errors(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda config, workspace: QualityReport(metrics={"parsed_rate": 1.0}),
    )

    findings = _run_quality_gates("纯叙述。", _workspace(), _config())
    info = [
        f for f in findings
        if f.reviewer == "Quality Gates" and f.severity == "info"
    ]
    assert info


# ---------------------------------------------------------------------------
# 9. _revise_draft: citation errors strip sentences; warnings append block
# ---------------------------------------------------------------------------


def test_revise_draft_strips_unsupported_citations(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=False,
            checked_sentence_count=1,
            unsupported_sentences=["该研究表明性能提升30%，缺少引用。"],
        ),
    )

    draft = (
        "前置内容。该研究表明性能提升30%，缺少引用。后置内容。"
    )
    critiques = [
        ReviewFinding(
            reviewer="Citation Gate",
            severity="error",
            finding="句子缺少论文级锚点或访问链接：…",
            suggested_fix="…",
        )
    ]
    revised = _revise_draft(draft, critiques, _workspace())
    # remove_unsupported_sentences only strips exact unsupported substrings
    assert "缺少引用" not in revised
    assert "前置内容" in revised


def test_revise_draft_appends_warning_block():
    draft = "草稿。"
    critiques = [
        ReviewFinding(
            reviewer="Storyline Gate",
            severity="warning",
            finding="缺少跨论文证据",
        )
    ]
    revised = _revise_draft(draft, critiques, _workspace())
    assert "质量门与可选审稿后的限制说明" in revised
    assert "Storyline Gate" in revised


# ---------------------------------------------------------------------------
# 10. _round_score: clamping + penalties
# ---------------------------------------------------------------------------


def test_round_score_full_credit_when_no_critiques(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda cfg, ws: QualityReport(
            metrics={
                "citation_guard_pass": 1.0,
                "parsed_rate": 1.0,
                "comparison_matrix_count": 1.0,
                "storyline_claim_count": 1.0,
            }
        ),
    )
    score = _round_score([], _workspace(), _config())
    # 0.62 + 0.12 + 0.08 + 0.06 + 0.06 = 0.94, clamped to 0.98
    assert score == pytest.approx(0.94, abs=0.01)


def test_round_score_penalized_by_errors_and_warnings(monkeypatch):
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda cfg, ws: QualityReport(
            metrics={"citation_guard_pass": 0.0, "parsed_rate": 0.0}
        ),
    )
    critiques = [
        ReviewFinding(reviewer="x", severity="error", finding="e1"),
        ReviewFinding(reviewer="y", severity="error", finding="e2"),
        ReviewFinding(reviewer="z", severity="warning", finding="w1"),
    ]
    score = _round_score(critiques, _workspace(), _config())
    # 0.62 - 0.24 - 0.04 = 0.34
    assert score == pytest.approx(0.34, abs=0.01)


# ---------------------------------------------------------------------------
# 11. _replan_actions: each branch
# ---------------------------------------------------------------------------


def test_replan_actions_no_full_text_and_no_parsed():
    ws = _workspace(with_full_text=False)
    ws.parsed_papers = {}
    actions = _replan_actions([], ws)
    assert "resolve_full_text" in actions
    assert "parse_full_text_with_paddleocr" in actions


def test_replan_actions_table_gate_finding_adds_extract():
    ws = _workspace()
    critiques = [
        ReviewFinding(
            reviewer="Table Gate", severity="warning", finding="x"
        )
    ]
    actions = _replan_actions(critiques, ws)
    assert "extract_tables_and_structured_artifacts" in actions


def test_replan_actions_citation_finding_adds_rerun():
    ws = _workspace()
    critiques = [
        ReviewFinding(
            reviewer="Citation Gate", severity="error", finding="x"
        )
    ]
    actions = _replan_actions(critiques, ws)
    assert "rerun_citation_guard_after_revision" in actions


def test_replan_actions_storyline_finding_adds_rebuild():
    ws = _workspace()
    critiques = [
        ReviewFinding(
            reviewer="Storyline Gate", severity="warning", finding="x"
        )
    ]
    actions = _replan_actions(critiques, ws)
    assert "rebuild_storyline_from_parsed_evidence" in actions


# ---------------------------------------------------------------------------
# 12. _execute_safe_replan_actions: per-branch behavior
# ---------------------------------------------------------------------------


def test_execute_safe_replan_actions_calls_parse_skill(monkeypatch):
    async def fake_parse(ws, cfg, **kwargs):
        ws.parsed_papers["p1"] = _parsed_paper()
        return ws, {"parsed_count": 1, "warnings": []}

    monkeypatch.setattr(
        "littrace.autonomous_loop.parse_workspace_skill", fake_parse
    )

    new_ws, executed = _run(
        _execute_safe_replan_actions(
            _config(), _workspace(), ["parse_full_text_with_paddleocr"]
        )
    )
    assert "parse_full_text_with_paddleocr" in executed
    assert new_ws.parsed_papers["p1"].parsed is True


def test_execute_safe_replan_actions_parse_skipped_when_zero(monkeypatch):
    async def fake_parse(ws, cfg, **kwargs):
        return ws, {"parsed_count": 0, "warnings": ["no PDFs"]}

    monkeypatch.setattr(
        "littrace.autonomous_loop.parse_workspace_skill", fake_parse
    )

    _, executed = _run(
        _execute_safe_replan_actions(
            _config(), _workspace(), ["parse_full_text_with_paddleocr"]
        )
    )
    assert executed == []


def test_execute_safe_replan_actions_extract_always_appended(monkeypatch):
    async def fake_extract(ws, cfg, **kwargs):
        ws.performance_cells = [_cell()]
        return ws, HarnessReport(
            check_name="extract",
            passed=True,
            score=1.0,
            findings=[],
            item_count=1,
        )

    monkeypatch.setattr(
        "littrace.autonomous_loop.extract_tables_skill", fake_extract
    )

    _, executed = _run(
        _execute_safe_replan_actions(
            _config(), _workspace(), ["extract_tables_and_structured_artifacts"]
        )
    )
    assert "extract_tables_and_structured_artifacts" in executed


# ---------------------------------------------------------------------------
# 13. _run_optional_reviewer: LLM gate + JSON parsing + keyword promotion
# ---------------------------------------------------------------------------


def test_optional_reviewer_skipped_when_llm_disabled():
    result = _run(
        _run_optional_reviewer(
            _config(llm_enabled=False, api_key=None),
            "q",
            "d",
            _workspace(),
        )
    )
    assert result == []


def test_optional_reviewer_skipped_when_no_api_key():
    result = _run(
        _run_optional_reviewer(
            _config(llm_enabled=True, api_key=None),
            "q",
            "d",
            _workspace(),
        )
    )
    assert result == []


def test_optional_reviewer_empty_reply_returns_empty(monkeypatch):
    async def fake_chat(cfg, sys_, payload, workspace=None, *, json_mode=False):
        return LLMReply(text="", used_llm=False, error="empty")

    monkeypatch.setattr(
        "littrace.autonomous_loop.chat_completion", fake_chat
    )

    result = _run(
        _run_optional_reviewer(_config(), "q", "d", _workspace())
    )
    assert result == []


def test_optional_reviewer_parses_json_and_promotes_error_keywords(monkeypatch):
    payload_dict = {
        "critiques": [
            {
                "reviewer": "Method",
                "severity": "warning",
                # contains "unsupported" → promoted to error
                "finding": "This claim is unsupported, cannot verify.",
                "suggested_fix": None,
            }
        ]
    }

    async def fake_chat(cfg, sys_, payload, workspace=None, *, json_mode=False):
        return LLMReply(text=json.dumps(payload_dict), used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.chat_completion", fake_chat
    )

    out = _run(
        _run_optional_reviewer(_config(), "q", "d", _workspace())
    )
    assert len(out) == 1
    assert out[0].severity == "error"
    assert out[0].reviewer == "Method"


def test_optional_reviewer_truncates_to_eight(monkeypatch):
    payload_dict = {
        "critiques": [
            {
                "reviewer": f"R{i}",
                "severity": "warning",
                "finding": f"finding {i}",
                "suggested_fix": None,
            }
            for i in range(12)
        ]
    }

    async def fake_chat(cfg, sys_, payload, workspace=None, *, json_mode=False):
        return LLMReply(text=json.dumps(payload_dict), used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.chat_completion", fake_chat
    )

    out = _run(
        _run_optional_reviewer(_config(), "q", "d", _workspace())
    )
    assert len(out) == 8


def test_optional_reviewer_drops_invalid_json(monkeypatch):
    async def fake_chat(cfg, sys_, payload, workspace=None, *, json_mode=False):
        return LLMReply(text="not json {{{", used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.chat_completion", fake_chat
    )

    result = _run(
        _run_optional_reviewer(_config(), "q", "d", _workspace())
    )
    assert result == []


def test_optional_reviewer_drops_schema_violation(monkeypatch):
    # missing required "critiques" field
    async def fake_chat(cfg, sys_, payload, workspace=None, *, json_mode=False):
        return LLMReply(text=json.dumps({"oops": []}), used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.chat_completion", fake_chat
    )

    result = _run(
        _run_optional_reviewer(_config(), "q", "d", _workspace())
    )
    assert result == []


# ---------------------------------------------------------------------------
# 14. run_review_loop: release_blocker rewrite + auto_replan
# ---------------------------------------------------------------------------


def test_release_blocker_rewrites_final_answer_when_citation_fails(monkeypatch):
    async def fake_writer(config, question, workspace, rag_evidence=None):
        return LLMReply(text="修订前的研究结论。", used_llm=True)

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=False,
            checked_sentence_count=1,
            unsupported_sentences=["修订前的研究结论。"],
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda cfg, ws: QualityReport(metrics={"parsed_rate": 1.0}),
    )

    report = _run(
        run_review_loop(_config(), "总结", _workspace())
    )
    assert report.release_ready is False
    assert report.release_blockers, "expected release_blockers to be non-empty"
    assert "未通过最终发布检查" in report.final_answer
    assert "修订前的研究结论。" not in report.final_answer
    assert report.passed is False


def test_auto_replan_executes_parse_and_appends_readiness_info(monkeypatch):
    async def fake_writer(config, question, workspace, rag_evidence=None):
        return LLMReply(text="草稿。", used_llm=True)

    async def fake_parse(ws, cfg, **kwargs):
        ws.parsed_papers["p1"] = _parsed_paper()
        return ws, {"parsed_count": 1, "warnings": []}

    monkeypatch.setattr(
        "littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.guard_citations",
        lambda text, workspace, **_: CitationGuardReport(
            passed=True, checked_sentence_count=0
        ),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_storyline_skill",
        lambda workspace, **_: [],
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.build_quality_report_skill",
        lambda cfg, ws: QualityReport(metrics={"parsed_rate": 0.0}),
    )
    monkeypatch.setattr(
        "littrace.autonomous_loop.parse_workspace_skill", fake_parse
    )

    # Trigger `parse_full_text_with_paddleocr` replan: workspace with active
    # papers but no parsed papers makes _replan_actions add the action.
    ws = _workspace()
    ws.parsed_papers = {}

    report = _run(
        run_review_loop(
            _config(),
            "总结",
            ws,
            auto_replan=True,
            max_rounds=1,
        )
    )

    assert "parse_full_text_with_paddleocr" in report.executed_replan_actions
    # The auto_replan path appends an info finding for the executed action.
    info_findings = [
        f
        for f in report.rounds[-1].critiques
        if f.severity == "info"
        and "已执行自动重规划动作" in f.finding
    ]
    assert info_findings, "auto_replan should append an info finding"


# ---------------------------------------------------------------------------
# 15. _reviewer_payload: truncation + listing
# ---------------------------------------------------------------------------


def test_reviewer_payload_truncates_long_draft():
    import re

    ws = _workspace()
    long = "x" * 10_000
    payload = _reviewer_payload("obj", long, ws)
    assert "Draft:\n" in payload
    # The first x-block is the draft slice — its length is exactly 4000.
    # Other occurrences in the payload (e.g. doi "10.1000/example") are
    # not part of the truncation behaviour under test.
    match = re.search(r"x+", payload)
    assert match is not None
    assert len(match.group(0)) == 4000


def test_reviewer_payload_includes_active_papers_and_cells():
    ws = _workspace()
    payload = _reviewer_payload("obj", "draft", ws)
    assert "p1" in payload
    assert "gauge factor" in payload