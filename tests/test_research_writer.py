import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.research_writer import fallback_evidence_answer, write_evidence_grounded_answer


def test_fallback_evidence_answer_includes_references():
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

    assert "引用与访问链接" in answer
    assert "https://doi.org/10.1000/example" in answer


@pytest.mark.anyio
async def test_writer_reports_empty_workspace_without_llm():
    reply = await write_evidence_grounded_answer(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结一下",
        LiteratureWorkspace(),
    )

    assert not reply.used_llm
    assert reply.error == "empty_workspace"
