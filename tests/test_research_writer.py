import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.llm import LLMReply
from littrace.models import EvidenceSpan, LiteratureWorkspace, PaperMetadata, PerformanceCell
from littrace.research_writer import (
    _performance_evidence_id,
    fallback_evidence_answer,
    write_evidence_grounded_answer,
)


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
async def test_writer_blocks_final_answer_without_publishable_claim():
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key")),
        "总结性能",
        _parsed_workspace().model_copy(update={"performance_cells": []}),
    )

    assert not reply.used_llm
    assert reply.error == "claim_release_blocked"


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
    assert "草稿性断言" in reply.text
    assert "草稿，未发布" in reply.text
    assert "引用与访问链接" in reply.text
    assert any(claim.text == "该材料在全文中表现出稳定的压力传感。" for claim in workspace.claims)
    assert not workspace.claim_verification_reports[-1].publishable


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
