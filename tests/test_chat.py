import json
from types import SimpleNamespace

import pytest

from littrace.chat import _format_search_result_reply, handle_chat
from littrace.config import LLMConfig, LitTraceConfig
from littrace.context import add_papers
from littrace.llm import LLMReply
from littrace.models import (
    ChatRequest,
    ReviewLoopReport,
    EvidenceSpan,
    LiteratureWorkspace,
    PaperMetadata,
    PerformanceCell,
    ResearchRunResult,
    WorkflowTrace,
)
from littrace.retrieval.pgvector_store import RagSearchHit
from littrace.retrieval.rag_profile import RagProfile
from littrace.retrieval.rag_search import RagSearchResult
from littrace.session import create_chat_session


def _mock_llm_reply(monkeypatch, text="这是一个包含引用与访问链接的研究回答。", used_llm=True):
    async def _fake_chat(config, system_prompt, user_message, workspace=None, **kwargs):
        if "LitTrace Research Writer" in system_prompt:
            evidence_id = next(
                line.split("evidence_id=", 1)[1].split(";", 1)[0]
                for line in user_message.splitlines()
                if "evidence_id=metric:" in line
            )
            return LLMReply(
                text=json.dumps(
                    {
                        "claims": [
                            {
                                "text": text,
                                "evidence_ids": [evidence_id],
                                "support_quotes": {evidence_id: "Sensitivity reached 12.5 kPa-1."},
                            }
                        ]
                    }
                ),
                used_llm=used_llm,
            )
        return LLMReply(text=text, used_llm=used_llm)

    monkeypatch.setattr("littrace.research_writer.chat_completion", _fake_chat)


def _offline_config() -> LitTraceConfig:
    return LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False))


@pytest.mark.anyio
async def test_chat_reports_intent_parser_missing_key_without_fallback():
    response, workspace = await handle_chat(
        ChatRequest(message="帮我找几篇柔性压力传感器论文"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(api_key=None, enabled=True, intent_parser_enabled=True)),
    )

    assert response.action == "intent_parse_error"
    assert "没有配置 LLM API key" in response.reply
    assert not workspace.context.active_papers


@pytest.mark.anyio
async def test_chat_search_updates_workspace():
    response, workspace = await handle_chat(
        ChatRequest(message="检索 MXene flexible sensor 的最新论文", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert workspace.context.active_papers
    assert response.citations
    assert response.publisher_routes is not None
    assert response.research_result is not None
    assert response.research_result.workflow_trace is not None
    assert any(
        step.node == "minimum_evidence_gate"
        for step in response.research_result.workflow_trace.steps
    )


@pytest.mark.anyio
async def test_chat_research_request_runs_search_instead_of_help():
    response, workspace = await handle_chat(
        ChatRequest(message="我想了解一下薄膜压敏传感阵列的相关文献，请帮我调研一下", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert workspace.context.active_papers
    assert "mock/开发样例检索" in response.reply


@pytest.mark.anyio
async def test_chat_merges_pending_ambiguous_intent():
    response, workspace = await handle_chat(
        ChatRequest(message="检索", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "clarify_intent"
    assert workspace.context.filters.pending_intent is not None

    response, workspace = await handle_chat(
        ChatRequest(message="MXene 柔性压力传感器", live=False),
        workspace,
        _offline_config(),
    )

    assert response.action == "search"
    assert response.intent_confidence is not None
    assert response.intent_confidence >= 0.76
    assert workspace.context.filters.pending_intent is None


@pytest.mark.anyio
async def test_chat_can_cancel_pending_intent():
    response, workspace = await handle_chat(
        ChatRequest(message="检索", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "clarify_intent"

    response, workspace = await handle_chat(
        ChatRequest(message="取消", live=False),
        workspace,
        _offline_config(),
    )

    assert response.action == "cancel_pending_intent"
    assert workspace.context.filters.pending_intent is None


def test_search_reply_mentions_expanded_year_range():
    workspace = LiteratureWorkspace()
    workspace.context.filters.search_mode = "live"
    reply = _format_search_result_reply(
        "rare topic",
        workspace,
        expanded_year_range=True,
        original_year_min=2024,
    )

    assert "已自动扩大检索年限" in reply


def test_search_reply_refuses_analysis_below_five_papers():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(4)],
    )
    workspace.context.filters.search_mode = "live"

    reply = _format_search_result_reply("rare topic", workspace)

    assert "全文证据还不足 5 篇" in reply
    assert "我先不做分析型结论" in reply
    assert "宽召回" not in reply
    assert "成功解析" not in reply


def test_search_reply_lists_all_results_above_five():
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(7)],
    )
    workspace.context.filters.search_mode = "live"

    reply = _format_search_result_reply("topic", workspace)

    assert "全文证据已达到最低门槛" in reply
    assert "7. Paper 6" in reply
    assert "宽召回" not in reply


@pytest.mark.anyio
async def test_search_only_with_full_text_evidence_returns_answer_not_metrics(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}", doi=f"10.1000/p{i}")
            for i in range(5)
        ],
    )
    workspace.context.filters.search_mode = "live"
    workspace.context.filters.candidate_pool_count = 12
    workspace.context.filters.valid_candidate_count = 10
    workspace.context.filters.downloaded_full_text_count = 5
    workspace.context.filters.parsed_full_text_count = 5
    for paper_id in workspace.context.active_papers:
        workspace.parsed_papers[paper_id] = {
            "parsed": True,
            "sections": [{"name": "Results", "text": "Full text evidence."}],
        }
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

    async def fake_run_research_graph(*_args, **_kwargs):
        return ResearchRunResult(workspace=workspace, workflow_trace=WorkflowTrace())

    monkeypatch.setattr("littrace.chat.run_research_graph", fake_run_research_graph)
    _mock_llm_reply(monkeypatch, text="这是一个包含引用与访问链接的研究回答。")

    response, _ = await handle_chat(
        ChatRequest(message="我想了解一下 carbon PDMS pressure sensor", live=True),
        LiteratureWorkspace(),
        LitTraceConfig(
            llm=LLMConfig(enabled=True, api_key="fake-key", intent_parser_enabled=False)
        ),
    )

    assert response.action == "search"
    assert "引用与访问链接" in response.reply
    assert "宽召回" not in response.reply
    assert "成功解析" not in response.reply


@pytest.mark.anyio
async def test_chat_show_and_hide_context():
    workspace = LiteratureWorkspace()
    response, workspace = await handle_chat(
        ChatRequest(message="隐藏上下文"),
        workspace,
        _offline_config(),
    )
    assert response.action == "hide_context"
    assert not workspace.context.visible_to_user

    response, workspace = await handle_chat(
        ChatRequest(message="显示上下文"),
        workspace,
        _offline_config(),
    )
    assert response.action == "show_context"
    assert workspace.context.visible_to_user


@pytest.mark.anyio
async def test_chat_help_for_unknown_intent():
    response, _ = await handle_chat(
        ChatRequest(message="你好"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "help"


@pytest.mark.anyio
async def test_chat_composite_search_and_table():
    response, workspace = await handle_chat(
        ChatRequest(
            message="检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表", live=False
        ),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert response.action == "search"
    assert "我已围绕" in response.reply
    assert response.comparison_matrix is None
    assert any("不足 5 篇" in warning for warning in response.warnings)
    assert "download" not in response.action
    assert workspace.context.filters.year_min is None
    assert workspace.context.filters.expanded_year_range_from == 2024


@pytest.mark.anyio
async def test_chat_mock_search_does_not_pass_minimum_evidence_gate():
    response, _ = await handle_chat(
        ChatRequest(message="检索 碳基PDMS柔性薄膜传感器长时间受压漂移", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert "mock/开发样例检索" in response.reply
    gate = [
        step
        for step in response.research_result.workflow_trace.steps
        if step.node == "minimum_evidence_gate"
    ][-1]
    assert gate.status == "blocked"
    assert gate.outputs["real_search"] is False


@pytest.mark.anyio
async def test_chat_refuses_to_summarize_mock_context_routes():
    _, workspace = await handle_chat(
        ChatRequest(message="检索 碳基PDMS柔性薄膜传感器长时间受压漂移", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    response, _ = await handle_chat(
        ChatRequest(message="请总结这些论文的主要路线"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert "mock/开发样例" in response.reply
    assert "不能基于这些内容总结研究路线" in response.reply
    assert "mock202" not in response.reply


@pytest.mark.anyio
async def test_chat_trace_starts_with_intent_parsing():
    response, _ = await handle_chat(
        ChatRequest(message="检索 carbon PDMS pressure sensor drift", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    first = response.research_result.workflow_trace.steps[0]
    assert first.node == "parse_user_intent"
    assert first.outputs["minimum_required"] == 5
    assert first.outputs["query_variant_count"] >= 1


@pytest.mark.anyio
async def test_chat_search_trace_includes_evidence_quality_gate():
    response, _ = await handle_chat(
        ChatRequest(message="检索 carbon PDMS pressure sensor", live=False),
        LiteratureWorkspace(),
        _offline_config(),
    )

    assert any(
        step.node == "evidence_quality_gate"
        for step in response.research_result.workflow_trace.steps
    )


@pytest.mark.anyio
async def test_chat_passes_rag_hits_into_writer(monkeypatch, tmp_path):
    config = LitTraceConfig(
        storage={"sessions_dir": tmp_path},
        llm=LLMConfig(enabled=True, api_key="fake-key", intent_parser_enabled=False),
    )
    config.rag.enabled = True
    config.rag.postgres_dsn = "postgresql://littrace:littrace@localhost:5433/littrace"
    session = create_chat_session(config)
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id=f"p{i}", title=f"Paper {i}") for i in range(5)],
    )
    workspace.context.filters.search_mode = "live"
    workspace.context.filters.research_background = "MXene pressure sensor literature analysis"
    workspace.context.filters.research_background_status = "accepted"
    workspace.context.filters.parsed_full_text_count = 5
    workspace.context.filters.downloaded_full_text_count = 5
    for paper_id in workspace.context.active_papers:
        workspace.parsed_papers[paper_id] = {
            "parsed": True,
            "sections": [{"name": "Results", "text": "Full text evidence."}],
        }
    captured: dict[str, object] = {}

    profile = RagProfile(
        profile_id="rag:123",
        session_id=session.session_id,
        namespace=session.session_id,
        topic="MXene pressure sensor",
        query_variants=["MXene pressure sensor"],
        source_routes=["crossref", "openalex"],
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

    async def fake_search_workspace_rag(config_arg, workspace_arg, question, top_k=None):
        captured["search_question"] = question
        return RagSearchResult(
            profile=profile,
            hits=[
                RagSearchHit(
                    chunk_id="chunk:1",
                    paper_id="p0",
                    text="RAG says the sensor reached high sensitivity.",
                    score=0.91,
                    chunk_hash="hash:1",
                    section="Results",
                )
            ],
        )

    async def fake_writer(config_arg, question, workspace_arg, rag_evidence=None):
        captured["rag_evidence"] = rag_evidence or []
        return LLMReply(text="writer-called", used_llm=False, error="writer-test")

    monkeypatch.setattr("littrace.chat.search_workspace_rag", fake_search_workspace_rag)
    monkeypatch.setattr("littrace.chat.write_evidence_grounded_answer", fake_writer)

    async def fake_prepare_turn(request_arg, workspace_arg, config_arg, session_memory=None):
        return SimpleNamespace(
            intent=SimpleNamespace(
                actions=[],
                topic=None,
                year_min=None,
                confidence=1.0,
                ambiguous=False,
                ambiguity_reasons=[],
                clarification_questions=[],
            ),
            workspace=workspace_arg,
            memory_view=SimpleNamespace(purpose="synthesis", warnings=[]),
            early_response=None,
        )

    monkeypatch.setattr("littrace.chat.coordinator.prepare_turn", fake_prepare_turn)

    response, _ = await handle_chat(
        ChatRequest(message="请分析一下这批论文", session_id=session.session_id),
        workspace,
        config,
    )

    assert response.action == "llm_error"
    assert captured["search_question"] == "请分析一下这批论文"
    assert len(captured["rag_evidence"]) == 1
    assert captured["rag_evidence"][0].parser == "rag"
    assert workspace.context.filters.rag_last_hit_count == 1
    assert workspace.context.filters.rag_source_routes == ["crossref", "openalex"]


@pytest.mark.anyio
async def test_chat_can_select_downloads_by_index():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="p1", title="First"),
            PaperMetadata(paper_id="p2", title="Second"),
        ],
    )

    response, workspace = await handle_chat(
        ChatRequest(message="选择第 1、2 篇下载"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "select_downloads"
    assert workspace.context.selected_for_download == ["p1", "p2"]

    response, workspace = await handle_chat(
        ChatRequest(message="取消选择第 2 篇"),
        workspace,
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert workspace.context.selected_for_download == ["p1"]


@pytest.mark.anyio
async def test_chat_reports_component_status():
    response, _ = await handle_chat(
        ChatRequest(message="agent状态"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(enabled=False, intent_parser_enabled=False)),
    )

    assert response.action == "component_status"
    assert "LitTrace Coordinator" in response.reply
    assert "Citation and Evidence Gates" in response.reply
    assert "Optional Reviewer" in response.reply


@pytest.mark.anyio
async def test_chat_runs_autonomous_review_loop(monkeypatch):
    workspace = add_papers(
        LiteratureWorkspace(),
        [PaperMetadata(paper_id="p1", title="Traceable Paper", year=2026, doi="10.1000/example")],
    )
    workspace.parsed_papers["p1"] = {
        "parsed": True,
        "sections": [{"name": "Results", "text": "Key evidence for review."}],
    }
    workspace.context.filters.search_mode = "live"
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

    _mock_llm_reply(monkeypatch, text="可选 Reviewer 审稿后的研究结论。")

    response, workspace = await handle_chat(
        ChatRequest(message="请多轮反驳并修订当前结论"),
        workspace,
        LitTraceConfig(
            llm=LLMConfig(enabled=True, api_key="fake-key", intent_parser_enabled=False)
        ),
    )

    assert response.action == "autonomous_review"
    assert "质量门与可选 Reviewer 审查" in response.reply
    assert workspace.context.filters.autonomous_loop_report is not None
    assert "可选 Reviewer 审稿后的研究结论" in response.reply
    assert workspace.context.filters.autonomous_loop_report["release_ready"] is True


@pytest.mark.anyio
async def test_chat_hides_unreleased_autonomous_answer(monkeypatch):
    async def fake_autonomous_loop(*args, **kwargs):
        return ReviewLoopReport(
            objective="复核",
            final_answer="不应直接展示的未发布结论。",
            passed=False,
            score=0.1,
            release_ready=False,
            release_blockers=["Claim verification blocks release."],
        )

    monkeypatch.setattr("littrace.chat.run_review_loop", fake_autonomous_loop)
    response, workspace = await handle_chat(
        ChatRequest(message="请多轮反驳并修订当前结论"),
        LiteratureWorkspace(),
        LitTraceConfig(llm=LLMConfig(intent_parser_enabled=False)),
    )

    assert response.action == "autonomous_review"
    assert "自主审查结果未通过最终发布门禁" in response.reply
    assert "不应直接展示的未发布结论。" not in response.reply
    assert workspace.context.filters.autonomous_loop_report["release_ready"] is False
