import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.llm import LLMReply
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell
from littrace.research_writer import (
    _performance_evidence_id,
    _writer_payload,
    fallback_evidence_answer,
    write_evidence_grounded_answer,
)


def test_writer_uses_compact_rag_evidence_aliases():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", doi="10.1000/example")],
    )
    rag = EvidenceSpan(
        paper_id="p1",
        evidence_id="rag:rag:long-profile-id:rag:long-chunk-id",
        source_record_id="rag:rag:long-profile-id:rag:long-chunk-id",
        section="Results",
        snippet="The measured Young's modulus was 4.66 kPa.",
        content_hash="a" * 64,
        parser="rag",
    )

    payload, registry = _writer_payload("报告模量", workspace, rag_evidence=[rag])

    assert "evidence_id=rag-aaaaaaaaaaaaaaaa" in payload
    assert "rag-aaaaaaaaaaaaaaaa" in registry
    assert registry["rag-aaaaaaaaaaaaaaaa"].source_record_id == rag.source_record_id


def test_fallback_evidence_answer_refuses_without_full_text():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Paper",
                year=2026,
                doi="10.1000/example",
            )
        ],
    )

    answer = fallback_evidence_answer("总结一下", workspace)

    assert "metadata/abstract fallback 已禁用" in answer
    assert "https://doi.org/10.1000/example" not in answer


def test_fallback_evidence_answer_includes_references_with_full_text():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [
                        {
                            "name": "Results",
                            "text": "The full text reports stable pressure sensing.",
                        }
                    ],
                }
            }
        ),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Paper",
                year=2026,
                doi="10.1000/example",
            )
        ],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.5,
            unit="kPa-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                page=4,
                snippet="Sensitivity reached 12.5 kPa-1.",
            ),
        )
    )

    answer = fallback_evidence_answer("总结一下", workspace)

    assert "引用与访问链接" in answer
    assert "https://doi.org/10.1000/example" in answer


def test_fallback_evidence_answer_blocks_unreleased_claims():
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [{"name": "Results", "text": "Full text evidence."}],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", doi="10.1000/example")],
    )

    answer = fallback_evidence_answer("总结一下", workspace)

    assert "Claim 发布门禁" in answer


def test_fallback_evidence_answer_refuses_mock_workspace():
    workspace = add_papers(
        LiteratureWorkspace(context={"filters": {"search_mode": "mock"}}),
        [
            PaperMetadata(
                paper_id="p1",
                title="Mock Paper",
                year=2026,
                doi="10.1002/adfm.mock2026001",
            )
        ],
    )

    answer = fallback_evidence_answer("总结主要路线", workspace)

    assert "mock/开发样例" in answer
    assert "10.1002/adfm.mock2026001" not in answer


@pytest.mark.anyio
async def test_writer_reports_empty_workspace_without_llm():
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结一下",
        LiteratureWorkspace(),
    )

    assert not reply.used_llm
    assert reply.error == "empty_workspace"


@pytest.mark.anyio
async def test_writer_refuses_mock_workspace_without_llm():
    workspace = add_papers(
        LiteratureWorkspace(context={"filters": {"search_mode": "mock"}}),
        [PaperMetadata(paper_id="p1", title="Mock Paper", doi="10.1002/adfm.mock2026001")],
    )

    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结主要路线",
        workspace,
    )

    assert not reply.used_llm
    assert reply.error == "mock_workspace"


@pytest.mark.anyio
async def test_writer_refuses_missing_full_text_without_llm():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Real Paper", doi="10.1000/real")],
    )

    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结主要路线",
        workspace,
    )

    assert not reply.used_llm
    assert reply.error == "missing_full_text"


def _parsed_workspace() -> LiteratureWorkspace:
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [
                        {
                            "name": "Results",
                            "text": "The full text reports stable pressure sensing.",
                        }
                    ],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Real Paper", doi="10.1000/real")],
    )
    workspace.performance_cells.append(
        PerformanceCell(
            paper_id="p1",
            metric="sensitivity",
            value=12.5,
            unit="kPa-1",
            evidence=EvidenceSpan(
                paper_id="p1",
                page=4,
                snippet="Sensitivity reached 12.5 kPa-1.",
            ),
        )
    )
    return workspace


def _metric_evidence_id(workspace: LiteratureWorkspace) -> str:
    cell = workspace.performance_cells[0]
    return _performance_evidence_id(cell.paper_id, cell.metric, cell.value, cell.unit)


@pytest.mark.anyio
async def test_writer_does_not_preblock_new_answer_for_old_unreleased_claims():
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结性能",
        _parsed_workspace().model_copy(update={"performance_cells": []}),
    )

    assert not reply.used_llm
    assert reply.error == "llm_disabled"


@pytest.mark.anyio
async def test_writer_publishes_quote_bound_translation(monkeypatch):
    from littrace import research_writer

    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [
                        {
                            "name": "Results",
                            "text": "The gelatin hydrogel was lyophilized to form a rigid aerogel.",
                            "evidence": {
                                "page": 2,
                                "parser": "docling",
                                "parser_version": "docling:v1",
                                "source_record_id": "paper:p1",
                                "content_hash": "a" * 64,
                                "captured_at": "2026-08-10T00:00:00+00:00",
                            },
                        }
                    ],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Real Paper", doi="10.1000/real")],
    )

    async def fake_completion(*args, **kwargs):
        payload = args[2]
        evidence_id = payload.split("evidence_id=", 1)[1].split(";", 1)[0]
        return LLMReply(
            text=(
                '{"claims": [{"text": "纯明胶水凝胶冻干后形成刚性气凝胶。", '
                f'"evidence_ids": ["{evidence_id}"], '
                f'"support_quotes": {{"{evidence_id}": '
                '"The gelatin hydrogel was lyophilized to form a rigid aerogel."}}]}'
            ),
            used_llm=True,
        )

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "说明明胶气凝胶的刚性。",
        workspace,
    )

    assert reply.used_llm
    assert "纯明胶水凝胶冻干后形成刚性气凝胶" in reply.text
    assert "草稿，未发布" not in reply.text
    assert workspace.claim_verification_reports[-1].publishable


@pytest.mark.anyio
async def test_writer_publishes_numeric_claim_against_performance_cell(monkeypatch):
    from littrace import research_writer

    workspace = _parsed_workspace()
    workspace.performance_cells[0] = workspace.performance_cells[0].model_copy(
        update={
            "evidence": EvidenceSpan(
                paper_id="p1",
                page=4,
                snippet="12.5 kPa-1.",
            )
        }
    )
    evidence_id = _metric_evidence_id(workspace)

    async def fake_completion(*args, **kwargs):
        return LLMReply(
            text=(
                '{"claims": [{"text": "该器件的灵敏度为12.5 kPa-1。", '
                f'"evidence_ids": ["{evidence_id}"], '
                f'"support_quotes": {{"{evidence_id}": "model-truncated table quote"}}, '
                '"claim_kind": "numeric", "metric": "sensitivity", '
                '"expected_value": 12.5, "expected_unit": "kPa-1"}]}'
            ),
            used_llm=True,
        )

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "给出器件的灵敏度性能参数。",
        workspace,
    )

    assert reply.used_llm
    assert "草稿，未发布" not in reply.text
    assert workspace.claim_verification_reports[-1].publishable


@pytest.mark.anyio
async def test_writer_retries_malformed_json_once(monkeypatch):
    from littrace import research_writer

    workspace = _parsed_workspace()
    evidence_id = _metric_evidence_id(workspace)
    replies = iter(
        [
            LLMReply(text="not json", used_llm=True),
            LLMReply(
                text=(
                    '{"claims": [{"text": "该材料在全文中表现出稳定的压力传感。", '
                    f'"evidence_ids": ["{evidence_id}"], '
                    f'"support_quotes": {{"{evidence_id}": "Sensitivity reached 12.5 kPa-1."}}'
                    + "}]}"
                ),
                used_llm=True,
            ),
        ]
    )
    calls = 0

    async def fake_completion(*args, **kwargs):
        nonlocal calls
        calls += 1
        return next(replies)

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结性能",
        workspace,
    )

    assert calls == 2
    assert reply.used_llm
    assert "以下结论已逐条绑定" in reply.text
    assert "草稿，未发布" not in reply.text
    assert "引用与访问链接" in reply.text
    assert any(claim.text == "该材料在全文中表现出稳定的压力传感。" for claim in workspace.claims)
    assert workspace.claim_verification_reports[-1].publishable


@pytest.mark.anyio
async def test_writer_labels_a_json_preface_without_claims_as_draft(monkeypatch):
    from littrace import research_writer

    workspace = _parsed_workspace()

    async def fake_completion(*args, **kwargs):
        return LLMReply(text='{"answer": "仅供复核的研究说明。"}', used_llm=True)

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结性能",
        workspace,
    )

    assert reply.used_llm
    assert reply.error == "draft_without_claims"
    assert "草稿性说明" in reply.text
    assert "引用与访问链接" in reply.text


@pytest.mark.anyio
async def test_writer_retries_unknown_evidence_id_then_refuses(monkeypatch):
    from littrace import research_writer

    async def fake_completion(*args, **kwargs):
        return LLMReply(
            text=(
                '{"claims": [{"text": "未经登记的结论。", "evidence_ids": ["invented-evidence"]}]}'
            ),
            used_llm=True,
        )

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结性能",
        _parsed_workspace(),
    )

    assert not reply.used_llm
    assert reply.error == "invalid_research_answer_schema"
    assert "未经登记的结论" not in reply.text


@pytest.mark.anyio
async def test_writer_refuses_after_second_malformed_json(monkeypatch):
    from littrace import research_writer

    async def fake_completion(*args, **kwargs):
        return LLMReply(text="unsafe free-form conclusion", used_llm=True)

    monkeypatch.setattr(research_writer, "chat_completion", fake_completion)
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结性能",
        _parsed_workspace(),
    )

    assert not reply.used_llm
    assert reply.error == "invalid_research_answer_schema"
    assert "unsafe free-form conclusion" not in reply.text
