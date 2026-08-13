import pytest

from littrace.autonomous_loop import run_review_loop
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.llm import LLMReply
from littrace.models import LiteratureWorkspace, PaperMetadata
from littrace.retrieval.pgvector_store import RagSearchHit
from littrace.retrieval.rag_profile import RagProfile
from littrace.retrieval.rag_search import RagSearchResult


@pytest.mark.anyio
async def test_autonomous_loop_reports_empty_workspace():
    report = await run_review_loop(
        LitTraceConfig(llm=LLMConfig(enabled=False)),
        "总结当前文献",
        LiteratureWorkspace(),
    )

    assert not report.passed
    assert "search_papers" in report.replan_actions


@pytest.mark.anyio
async def test_autonomous_loop_raises_when_llm_disabled_with_papers():
    """LLM disabled → RuntimeError (no degradation)."""
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(
                paper_id="p1",
                title="Traceable Sensor Paper",
                year=2026,
                doi="10.1000/example",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await run_review_loop(
            LitTraceConfig(llm=LLMConfig(enabled=False)),
            "请比较性能并讲发展脉络",
            workspace,
        )


@pytest.mark.anyio
async def test_autonomous_loop_raises_when_llm_disabled_even_with_parsed(monkeypatch):
    """LLM disabled → RuntimeError even when papers are parsed (no degradation)."""
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Sensor Paper", year=2026)],
    )

    async def fake_parse(workspace, config):
        workspace.parsed_papers["p1"] = {
            "sections": [
                {
                    "name": "Results",
                    "text": "Method improves sensitivity and discusses limitation.",
                    "evidence": {"page": 1, "parser": "fake"},
                }
            ],
            "parsed": True,
        }
        return workspace, {"parsed_count": 1, "failed_count": 0}

    monkeypatch.setattr("littrace.autonomous_loop.parse_workspace_skill", fake_parse)

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await run_review_loop(
            LitTraceConfig(llm=LLMConfig(enabled=False)),
            "请自动重规划并比较性能",
            workspace,
            auto_replan=True,
        )


@pytest.mark.anyio
async def test_autonomous_loop_rechecks_publication_gate_before_final_answer(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [{"name": "Results", "text": "Full text evidence."}],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026)],
    )

    async def fake_writer(*args, **kwargs):
        return LLMReply(text="修订前的研究结论。", used_llm=True)

    monkeypatch.setattr("littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer)
    config = LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key"))
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"

    report = await run_review_loop(
        config,
        "总结当前文献",
        workspace,
        enable_optional_reviewer=False,
    )

    assert not report.release_ready
    assert not report.passed
    assert report.release_blockers
    assert "修订前的研究结论。" not in report.final_answer
    assert "未通过最终发布检查" in report.final_answer


@pytest.mark.anyio
async def test_autonomous_loop_passes_rag_evidence_into_writer(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(
            parsed_papers={
                "p1": {
                    "parsed": True,
                    "sections": [{"name": "Results", "text": "Full text evidence."}],
                }
            }
        ),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026)],
    )
    workspace.context.filters.search_mode = "live"
    workspace.context.filters.parsed_full_text_count = 1
    workspace.context.filters.downloaded_full_text_count = 1
    profile = RagProfile(
        profile_id="rag:123",
        session_id="s1",
        namespace="s1",
        topic="Traceable Paper",
        query_variants=["Traceable Paper"],
        source_routes=["crossref"],
        backend="pgvector",
        postgres_schema="littrace_rag",
        collection_name="littrace_s1",
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-v3",
        embedding_dimension=1024,
        chunk_target_tokens=700,
        chunk_overlap_tokens=120,
        top_k=12,
        refresh_frequency="daily",
        auto_refresh_enabled=True,
        auto_download_open_access=True,
        login_required_policy="queue_only",
    )
    workspace.context.filters.rag_profile = profile.model_dump(mode="json")
    captured: dict[str, object] = {}
    config = LitTraceConfig(llm=LLMConfig(enabled=True, api_key="test-key"))
    config.rag.enabled = True
    config.rag.backend = "pgvector"
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"

    async def fake_search_workspace_rag(config_arg, workspace_arg, objective, top_k=None):
        captured["objective"] = objective
        return RagSearchResult(
            profile=profile,
            hits=[
                RagSearchHit(
                    chunk_id="chunk:1",
                    paper_id="p1",
                    text="RAG says the sensor reached high sensitivity.",
                    score=0.93,
                    chunk_hash="hash:1",
                    section="Results",
                )
            ],
        )

    async def fake_writer(config_arg, objective, workspace_arg, rag_evidence=None):
        captured["rag_evidence"] = rag_evidence or []
        return LLMReply(text='{"claims":[],"answer":"ok"}', used_llm=True)

    monkeypatch.setattr("littrace.autonomous_loop.search_workspace_rag", fake_search_workspace_rag)
    monkeypatch.setattr("littrace.autonomous_loop.write_evidence_grounded_answer", fake_writer)

    await run_review_loop(
        config,
        "总结当前文献",
        workspace,
        enable_optional_reviewer=False,
    )

    assert captured["objective"] == "总结当前文献"
    assert len(captured["rag_evidence"]) == 1
    assert captured["rag_evidence"][0].parser == "rag"
