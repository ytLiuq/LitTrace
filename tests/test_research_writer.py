import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.research_writer import fallback_evidence_answer, write_evidence_grounded_answer


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

    answer = fallback_evidence_answer("总结一下", workspace)

    assert "引用与访问链接" in answer
    assert "https://doi.org/10.1000/example" in answer


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
