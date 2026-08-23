from __future__ import annotations

from littrace.autonomous_loop import run_review_loop
from littrace.citation_guard import guard_citations, remove_unsupported_sentences
from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.context import apply_context_update, _merge_filters
from littrace.coordinator import LitTraceCoordinator
from littrace.evaluation.harnesses import (
    check_hallucination_grounding,
    HallucinationCheckItem,
)
from littrace.intent import ChatIntent
from littrace.log import get_logger, timed
from littrace.chat_parts.formatting import (
    format_component_status,
    format_current_papers,
    format_search_result_reply,
)
from littrace.chat_parts.policies import (
    insufficient_real_evidence_reply,
    is_analysis_request,
    mock_context_refusal,
    workspace_evidence_quality,
    workspace_has_real_minimum_evidence,
    workspace_is_mock,
)
from littrace.models import (
    ChatRequest,
    ChatResponse,
    ContextUpdate,
    EvidenceSpan,
    WorkspaceSummary,
    LiteratureWorkspace,
    PaperSearchRequest,
    ResearchRunResult,
    WorkflowTraceStep,
)
from littrace.research_writer import (
    write_evidence_grounded_answer,
    write_storyline_narrative,
)
from littrace.research_background import (
    assess_research_background,
    mark_workspace_research_background_rejected,
    set_workspace_research_background,
    workspace_has_research_background,
)
from littrace.retrieval.rag_search import (
    rag_hits_to_evidence_spans,
    search_workspace_rag,
)
from littrace.runtime.memory import SessionMemory
from littrace.skill_runner import (
    build_comparison_matrix_skill,
    build_download_plan_skill,
    build_research_report_skill,
    build_storyline_skill,
    extract_tables_skill,
    parse_workspace_skill,
)
from littrace.tool_contracts import ToolCallContext
from littrace.workflow import run_research_graph
from littrace.retrieval.search import build_query_variants

logger = get_logger("chat")
coordinator = LitTraceCoordinator()


MIN_ANALYSIS_PAPERS = 5


async def handle_chat(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    session_memory: SessionMemory | None = None,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    message = request.message.strip()
    with timed("intent_parse"):
        turn = await coordinator.prepare_turn(
            request,
            workspace,
            config,
            session_memory=session_memory,
        )
    if turn.early_response is not None:
        logger.info(
            "coordinator_early_response",
            extra={
                "action": turn.early_response.action,
                "topic": turn.intent.topic if turn.intent else None,
            },
        )
        return turn.early_response, turn.workspace
    intent = turn.intent or ChatIntent()
    workspace = turn.workspace
    logger.info(
        "intent_parsed",
        extra={
            "actions": intent.actions,
            "topic": intent.topic,
            "year_min": intent.year_min,
            "confidence": intent.confidence,
            "ambiguous": intent.ambiguous,
            "memory_purpose": turn.memory_view.purpose,
            "memory_warnings": turn.memory_view.warnings,
        },
    )

    background_response = await _research_background_gate(request, workspace, config, intent)
    if background_response is not None:
        return _with_intent(background_response, intent), workspace

    quick_response = _route_quick_action(intent, workspace)
    if quick_response is not None:
        return quick_response, workspace

    if _should_run_composite(intent):
        logger.info("composite_intent", extra={"actions": intent.actions})
        return await _run_composite_intent(intent, request, workspace, config)

    if _is_analysis_request(message) and _workspace_is_mock(workspace):
        logger.warning("blocked_mock_context", extra={"msg_preview": message[:200]})
        return (
            _with_intent(
                ChatResponse(
                    reply=_mock_context_refusal(),
                    action="blocked_mock_context",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    citations=[],
                    warnings=["当前上下文来自 mock/开发样例，已禁止生成研究结论。"],
                ),
                intent,
            ),
            workspace,
        )

    if _is_analysis_request(message) and not _workspace_has_real_minimum_evidence(workspace):
        active = len(workspace.context.active_papers)
        parsed = len(workspace.parsed_papers)
        logger.warning(
            "insufficient_evidence",
            extra={
                "active_papers": active,
                "parsed_papers": parsed,
                "min_required": MIN_ANALYSIS_PAPERS,
            },
        )
        return (
            _with_intent(
                ChatResponse(
                    reply=_insufficient_real_evidence_reply(workspace),
                    action="insufficient_real_evidence",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    citations=_active_citations(workspace),
                    warnings=[f"真实相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已拒绝生成分析型结论。"],
                ),
                intent,
            ),
            workspace,
        )

    rag_evidence, rag_warnings = await _rag_evidence_for_workspace(config, request, workspace)
    llm_reply = await write_evidence_grounded_answer(
        config,
        message,
        workspace,
        rag_evidence=rag_evidence,
    )
    if llm_reply.used_llm:
        claim_hints = config.citation_guard.claim_hints or None
        guard = guard_citations(llm_reply.text, workspace, claim_hints=claim_hints)
        reply_text = remove_unsupported_sentences(llm_reply.text, guard)
        workspace.guard_reports.append(guard.model_dump())

        # Dimension 2: Run hallucination grounding harness
        hallucination_report = check_hallucination_grounding(
            [
                HallucinationCheckItem(
                    text=reply_text,
                    checked_sentence_count=guard.checked_sentence_count,
                    unsupported_sentence_count=len(guard.unsupported_sentences),
                    unsupported_sentences=guard.unsupported_sentences,
                    source="chat",
                )
            ]
        )
        extra_warnings = list(hallucination_report.errors[:3]) + list(
            hallucination_report.warnings[:3]
        )

        return (
            _with_intent(
                ChatResponse(
                    reply=reply_text,
                    action="llm_chat",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    citations=_active_citations(workspace),
                    warnings=(
                        rag_warnings
                        + guard.warnings
                        + guard.unsupported_sentences[:3]
                        + extra_warnings
                    ),
                ),
                intent,
            ),
            workspace,
        )
    # LLM unavailable — no degradation, return error
    if workspace.context.active_papers:
        logger.error("llm_unavailable", extra={"error": llm_reply.error})
        return (
            _with_intent(
                ChatResponse(
                    reply=(
                        f"LLM 调用失败，无法生成回答。错误：{llm_reply.error}\n\n"
                        "请检查 LLM 配置（DEEPSEEK_API_KEY / base_url / model）后重试。"
                    ),
                    action="llm_error",
                    workspace=WorkspaceSummary.from_workspace(workspace),
                    citations=_active_citations(workspace),
                    warnings=[llm_reply.error or "llm_unavailable"],
                ),
                intent,
            ),
            workspace,
        )

    return (
        _with_intent(
            ChatResponse(
                reply=(
                    "我可以用对话方式帮你检索论文、显示/隐藏文献上下文、规划下载、解析全文、"
                    "抽取性能表格，或生成有证据约束的发展脉络。你可以说：检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表。"
                ),
                action="help",
                workspace=WorkspaceSummary.from_workspace(workspace),
            ),
            intent,
        ),
        workspace,
    )


def _route_quick_action(
    intent: ChatIntent,
    workspace: LiteratureWorkspace,
) -> ChatResponse | None:
    """Handle show/hide-context, list-context, component-status, and
    select/deselect-downloads actions in one place. Returns the response
    to short-circuit the rest of handle_chat, or None when the intent
    should fall through to the composite / analysis branches."""
    if "show_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=True))
        return _with_intent(
            _response("已显示当前文献上下文。", "show_context", workspace), intent
        )
    if "hide_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=False))
        return _with_intent(
            _response("已隐藏当前文献上下文，后续对话会保持简洁。", "hide_context", workspace),
            intent,
        )
    if intent.actions == ["list_context"]:
        return _with_intent(
            ChatResponse(
                reply=_format_current_papers(workspace),
                action="list_context",
                workspace=WorkspaceSummary.from_workspace(workspace),
                citations=_active_citations(workspace),
            ),
            intent,
        )
    if intent.actions == ["component_status"]:
        return _with_intent(
            ChatResponse(
                reply=_format_component_status(),
                action="component_status",
                workspace=WorkspaceSummary.from_workspace(workspace),
            ),
            intent,
        )
    if any(action in intent.actions for action in ["select_downloads", "deselect_downloads"]):
        workspace, reply = _apply_download_selection(workspace, intent)
        return _with_intent(
            ChatResponse(
                reply=reply,
                action="select_downloads",
                workspace=WorkspaceSummary.from_workspace(workspace),
                citations=_active_citations(workspace),
            ),
            intent,
        )
    return None


async def _compose_storyline_narrative(
    config: LitTraceConfig,
    workspace: LiteratureWorkspace,
    storyline,
    replies: list[str],
    warnings: list[str],
) -> tuple[list[str], list[str], None]:
    """Generate a storyline narrative with citation guard + hallucination
    harness. Returns the (possibly appended) ``replies`` and
    ``warnings`` lists.
    """
    narrative = await write_storyline_narrative(config, workspace)
    if not narrative.used_llm:
        replies.append("已基于当前证据生成发展脉络草案；低证据部分会保持保守。")
        if narrative.error:
            warnings.append(narrative.error)
        return replies, warnings, None
    claim_hints = config.citation_guard.claim_hints or None
    guard = guard_citations(narrative.text, workspace, claim_hints=claim_hints)
    cleaned_narrative = remove_unsupported_sentences(narrative.text, guard)
    workspace.guard_reports.append(guard.model_dump())
    replies.append(cleaned_narrative)
    warnings.extend(guard.warnings)
    warnings.extend(guard.unsupported_sentences[:3])
    halu_report = check_hallucination_grounding(
        [
            HallucinationCheckItem(
                text=cleaned_narrative,
                checked_sentence_count=guard.checked_sentence_count,
                unsupported_sentence_count=len(guard.unsupported_sentences),
                unsupported_sentences=guard.unsupported_sentences,
                source="chat:storyline",
            )
        ]
    )
    warnings.extend(halu_report.errors[:2])
    warnings.extend(halu_report.warnings[:2])
    return replies, warnings, None


async def _execute_parse_action(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
    request: ChatRequest,
    config: LitTraceConfig,
    action: str,
) -> tuple[LiteratureWorkspace, str | None, str | None, str]:
    """Run the ``parse`` action: parse downloaded PDFs in the workspace.

    Returns ``(workspace, reply, warning, action)``. ``reply`` and
    ``warning`` are None when nothing happened. ``action`` is updated
    to ``"parse_full_text"`` when parse runs as a standalone action.
    """
    parse_config = config
    if intent.parse_strategy:
        parse_config = config.model_copy(deep=True)
        parse_config.parsing.parse_strategy = intent.parse_strategy
    try:
        workspace, report = await parse_workspace_skill(
            workspace,
            parse_config,
            context=_tool_context("chat.composite", intent, request),
        )
        reply = (
            f"已尝试解析全文：成功解析 {report['parsed_count']} 篇，"
            f"失败 {report['failed_count']} 篇。"
        )
        warning = str(report)
        new_action = "parse_full_text" if action == "composite" else action
        return workspace, reply, warning, new_action
    except RuntimeError as exc:
        return workspace, None, str(exc), action


async def _execute_document_action(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
    request: ChatRequest,
    config: LitTraceConfig,
    replies: list[str],
    warnings: list[str],
    action: str,
    evidence_quality: dict[str, int],
) -> tuple[list[str], list[str], str]:
    """Run the ``document`` action: build a research report from the
    workspace's parsed evidence. Returns updated ``(replies, warnings,
    action)``. ``action`` is updated to ``"compose_document"`` when
    document runs as a standalone action.
    """
    if evidence_quality["parsed_count"] == 0:
        warnings.append("当前缺少全文解析证据，已禁止基于题录/摘要生成报告。")
        return replies, warnings, action
    try:
        report = await build_research_report_skill(
            workspace,
            config,
            context=_tool_context("chat.composite", intent, request),
        )
        workspace.context.filters.document_report = report.model_dump(mode="json")
        if report.release_ready:
            replies.append(
                f"已生成可发布的研究报告：{len(report.sections)} 个章节，"
                f"{report.evidence_count} 条证据锚点，{len(report.citation_records)} 条引用。"
            )
        else:
            replies.append("已生成研究报告草稿，但发布门禁未通过；仅可作为待补证草稿。")
        if report.warnings:
            warnings.extend(report.warnings[:5])
        new_action = "compose_document" if action == "composite" else action
        return replies, warnings, new_action
    except RuntimeError as exc:
        warnings.append(str(exc))
        return replies, warnings, action


async def _execute_autonomous_review_action(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
    request: ChatRequest,
    config: LitTraceConfig,
    replies: list[str],
    warnings: list[str],
    action: str,
) -> tuple[list[str], list[str], str]:
    """Run the ``autonomous_review`` action: quality gates + reviewer
    loop. Returns updated ``(replies, warnings, action)``.
    """
    rag_evidence, rag_warnings = await _rag_evidence_for_workspace(
        config, request, workspace
    )
    loop_report = await run_review_loop(
        config,
        intent.topic or request.message,
        workspace,
        auto_replan=intent.auto_replan,
        rag_evidence=rag_evidence,
    )
    workspace.context.filters.autonomous_loop_report = loop_report.model_dump(mode="json")
    warnings.extend(rag_warnings)
    replies.append(
        f"已完成质量门与可选 Reviewer 审查：{len(loop_report.rounds)} 轮，"
        f"score={loop_report.score:.3f}，passed={loop_report.passed}。"
    )
    if loop_report.release_ready:
        replies.append(loop_report.final_answer)
    else:
        replies.append("自主审查结果未通过最终发布门禁，未输出修订后的研究结论。")
        warnings.extend(loop_report.release_blockers[:3])
    warnings.extend(loop_report.warnings[:5])
    new_action = "autonomous_review" if action == "composite" else action
    return replies, warnings, new_action


async def _execute_table_action(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
    request: ChatRequest,
    config: LitTraceConfig,
    action: str,
) -> tuple[LiteratureWorkspace, str | None, str, list[str]]:
    """Run the ``table`` action: extract performance metrics into a matrix."""
    try:
        workspace, harness = await extract_tables_skill(
            workspace,
            config,
            context=_tool_context("chat.composite", intent, request),
        )
        matrix = build_comparison_matrix_skill(workspace)
        reply = (
            f"已生成性能对比表：抽取 {len(workspace.performance_cells)} 个指标单元，"
            f"形成 {len(matrix.matrices)} 个指标矩阵。"
        )
        warnings = [*harness.errors, *harness.warnings, *matrix.warnings]
        new_action = "build_table" if action == "composite" else action
        return workspace, reply, new_action, warnings
    except RuntimeError as exc:
        return workspace, None, action, [str(exc)]


async def _execute_search_action(
    intent: ChatIntent,
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> tuple[ResearchRunResult, LiteratureWorkspace, str, list[str], int, bool]:
    """Run the ``search`` action: build a research graph, optionally retry
    with an expanded year range when too few papers came back, and
    return the result + a few side-effect fields the caller still needs
    for the composite reply.
    """
    search_year_min = intent.year_min or config.literature_context.default_year_min
    topic = intent.topic or request.message
    query_variants = build_query_variants(topic)
    research_result = await run_research_graph(
        PaperSearchRequest(
            topic=topic,
            year_min=search_year_min,
            limit=40,
            live=request.live,
            min_relevant_results=MIN_ANALYSIS_PAPERS,
            query_variants=query_variants,
        ),
        config,
        audit_citations_enabled=False,
        plan_downloads_enabled=False,
        route_publishers_enabled=True,
        parse_full_text_enabled=False,
        extract_tables_enabled=False,
        build_storyline_enabled=False,
        compose_document_enabled=False,
        autonomous_review_enabled=False,
    )
    workspace = research_result.workspace
    workspace = _apply_literature_filters(workspace, intent)
    expanded_year_range = False
    if (
        len(workspace.context.active_papers) < MIN_ANALYSIS_PAPERS
        and search_year_min is not None
    ):
        before_expand_count = len(workspace.context.active_papers)
        expanded_year_range = True
        _append_chat_trace_step(
            research_result,
            node="evidence_gate",
            status="expanded",
            reason=(
                f"最近年份范围内只找到 {before_expand_count} 篇，少于最低分析门槛 "
                f"{MIN_ANALYSIS_PAPERS} 篇，因此扩大检索年限。"
            ),
            inputs={"year_min": search_year_min, "paper_count": before_expand_count},
            outputs={"retry_year_min": None},
        )
        pre_retry_steps = (
            list(research_result.workflow_trace.steps)
            if research_result.workflow_trace is not None
            else []
        )
        research_result = await run_research_graph(
            PaperSearchRequest(
                topic=topic,
                year_min=None,
                limit=40,
                live=request.live,
                min_relevant_results=MIN_ANALYSIS_PAPERS,
                query_variants=query_variants,
            ),
            config,
            audit_citations_enabled=False,
            plan_downloads_enabled=False,
            route_publishers_enabled=True,
            parse_full_text_enabled=False,
            extract_tables_enabled=False,
            build_storyline_enabled=False,
            compose_document_enabled=False,
            autonomous_review_enabled=False,
        )
        if research_result.workflow_trace is not None and pre_retry_steps:
            research_result.workflow_trace.steps = [
                *pre_retry_steps,
                *research_result.workflow_trace.steps,
            ]
        workspace = research_result.workspace
        if intent.journals:
            expanded_intent = ChatIntent(journals=intent.journals)
            workspace = _apply_literature_filters(workspace, expanded_intent)
        workspace.context.filters.expanded_year_range_from = search_year_min
    _prepend_intent_trace_step(
        research_result,
        topic=topic,
        intent=intent,
        live=request.live if request.live is not None else config.api.enable_live_search,
        query_variant_count=len(query_variants),
    )
    return (
        research_result,
        workspace,
        topic,
        query_variants,
        search_year_min,
        expanded_year_range,
    )


async def _run_composite_intent(
    intent: ChatIntent,
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    replies: list[str] = []
    warnings: list[str] = []
    action = "composite"
    download_plan = None
    matrix = None
    research_result = None
    publisher_routes = None
    has_minimum_evidence = True

    if "search" in intent.actions:
        (research_result, workspace, topic, query_variants, search_year_min, expanded_year_range) = (
            await _execute_search_action(intent, request, workspace, config)
        )
        search_mode = getattr(workspace.context.filters, "search_mode", None)
        is_real_search = search_mode == "live"
        parsed_full_text_count = int(
            getattr(workspace.context.filters, "parsed_full_text_count", None) or 0
        )
        downloaded_full_text_count = int(
            getattr(workspace.context.filters, "downloaded_full_text_count", None) or 0
        )
        has_minimum_evidence = (
            is_real_search
            and len(workspace.context.active_papers) >= MIN_ANALYSIS_PAPERS
            and parsed_full_text_count >= MIN_ANALYSIS_PAPERS
        )
        _append_chat_trace_step(
            research_result,
            node="minimum_evidence_gate",
            status="passed" if has_minimum_evidence else "blocked",
            reason=(
                f"当前上下文有 {len(workspace.context.active_papers)} 篇已下载全文文献；"
                f"检索模式为 {search_mode or 'unknown'}；最低分析门槛要求 "
                f"{MIN_ANALYSIS_PAPERS} 篇真实全文解析文献。"
            ),
            inputs={"minimum_required": MIN_ANALYSIS_PAPERS},
            outputs={
                "paper_count": len(workspace.context.active_papers),
                "downloaded_full_text_count": downloaded_full_text_count,
                "parsed_full_text_count": parsed_full_text_count,
                "search_mode": search_mode,
                "real_search": is_real_search,
            },
        )
        _append_query_expansion_trace(research_result, intent.topic or request.message)
        _append_search_diagnostics_trace(research_result, workspace)
        _append_evidence_quality_trace(research_result, workspace)
        if _is_search_only_intent(intent):
            if has_minimum_evidence:
                rag_evidence, rag_warnings = await _rag_evidence_for_workspace(
                    config,
                    request,
                    workspace,
                )
                llm_reply = await write_evidence_grounded_answer(
                    config,
                    request.message,
                    workspace,
                    rag_evidence=rag_evidence,
                )
                if llm_reply.used_llm:
                    claim_hints = config.citation_guard.claim_hints or None
                    guard = guard_citations(llm_reply.text, workspace, claim_hints=claim_hints)
                    cleaned = remove_unsupported_sentences(llm_reply.text, guard)
                    workspace.guard_reports.append(guard.model_dump())
                    replies.append(cleaned)
                    warnings.extend(rag_warnings)
                    warnings.extend(guard.warnings)
                    warnings.extend(guard.unsupported_sentences[:3])

                    # Dimension 2: Hallucination harness
                    halu_report = check_hallucination_grounding(
                        [
                            HallucinationCheckItem(
                                text=cleaned,
                                checked_sentence_count=guard.checked_sentence_count,
                                unsupported_sentence_count=len(guard.unsupported_sentences),
                                unsupported_sentences=guard.unsupported_sentences,
                                source="chat:search_only",
                            )
                        ]
                    )
                    warnings.extend(halu_report.errors[:2])
                    warnings.extend(halu_report.warnings[:2])
                else:
                    logger.error("llm_unavailable_composite", extra={"error": llm_reply.error})
                    replies.append(
                        f"LLM 调用失败，无法生成研究回答。错误：{llm_reply.error}\n"
                        "请检查 LLM 配置后重试。"
                    )
                    if llm_reply.error:
                        warnings.append(llm_reply.error)
            else:
                replies.append(
                    _format_search_result_reply(
                        intent.topic or request.message,
                        workspace,
                        expanded_year_range=expanded_year_range,
                        original_year_min=search_year_min,
                    )
                )
        publisher_routes = research_result.publisher_routes
        action = "search"

    if "parse" in intent.actions:
        workspace, parse_reply, parse_warning, action = await _execute_parse_action(
            workspace, intent, request, config, action
        )
        if parse_reply:
            replies.append(parse_reply)
        if parse_warning:
            warnings.append(parse_warning)

    evidence_quality = _workspace_evidence_quality(workspace)
    if "table" in intent.actions and has_minimum_evidence:
        workspace, table_reply, action, table_warnings = _execute_table_action(
            workspace, intent, request, config, action
        )
        if table_reply:
            replies.append(table_reply)
        warnings.extend(table_warnings)
    elif "table" in intent.actions and not has_minimum_evidence:
        warnings.append(f"相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已跳过性能对比表。")

    if "storyline" in intent.actions and _workspace_is_mock(workspace):
        replies.append(_mock_context_refusal())
        warnings.append("当前上下文来自 mock/开发样例，已跳过发展脉络分析。")
    elif "storyline" in intent.actions and has_minimum_evidence:
        if evidence_quality["parsed_count"] == 0:
            warnings.append(
                "当前没有可用全文解析证据，已禁止基于题录/摘要生成强结论；请先获取并解析 PDF 全文。"
            )
        storyline = build_storyline_skill(workspace)
        if not storyline:
            replies.append("当前证据不足以生成真实的发展脉络。建议先检索并解析全文。")
        else:
            replies, warnings, _ = await _compose_storyline_narrative(
                config, workspace, storyline, replies, warnings
            )
        result = await run_research_graph(
            PaperSearchRequest(topic=intent.topic or request.message, live=False),
            config,
            audit_citations_enabled=False,
            plan_downloads_enabled=False,
            build_storyline_enabled=False,
        )
        result.workspace = workspace
        result.storyline = storyline
        research_result = result
        action = "build_storyline" if action == "composite" else action
    elif "storyline" in intent.actions and not has_minimum_evidence:
        warnings.append(f"相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已跳过发展脉络分析。")

    if "document" in intent.actions and has_minimum_evidence:
        replies, warnings, action = await _execute_document_action(
            workspace, intent, request, config, replies, warnings, action, evidence_quality
        )
    elif "document" in intent.actions and not has_minimum_evidence:
        warnings.append(f"相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已跳过报告生成。")

    if "autonomous_review" in intent.actions:
        replies, warnings, action = await _execute_autonomous_review_action(
            workspace, intent, request, config, replies, warnings, action
        )

    if "download" in intent.actions and not intent.skip_download:
        try:
            download_plan = await build_download_plan_skill(
                config,
                workspace,
                context=_tool_context("chat.composite", intent, request),
            )
            replies.append(
                f"已生成下载计划：{download_plan.downloadable_count} 篇可处理，其中 {download_plan.requires_login_count} 篇需要登录。"
            )
            action = "plan_downloads" if action == "composite" else action
        except RuntimeError as exc:
            warnings.append(str(exc))

    if not replies and "search" in intent.actions:
        replies.append(_format_search_result_reply(intent.topic or request.message, workspace))
    if not replies:
        replies.append("已理解你的指令，但当前没有可执行动作。")

    return (
        _with_intent(
            ChatResponse(
                reply="\n".join(replies),
                action=action,
                workspace=WorkspaceSummary.from_workspace(workspace),
                research_result=research_result,
                citations=_active_citations(workspace),
                download_plan=download_plan,
                publisher_routes=publisher_routes,
                comparison_matrix=matrix,
                warnings=warnings,
            ),
            intent,
        ),
        workspace,
    )


def _response(reply: str, action: str, workspace: LiteratureWorkspace) -> ChatResponse:
    return ChatResponse(reply=reply, action=action, workspace=WorkspaceSummary.from_workspace(workspace))


def _with_intent(response: ChatResponse, intent: ChatIntent) -> ChatResponse:
    response.intent_confidence = intent.confidence
    response.ambiguous_intent = intent.ambiguous
    response.ambiguity_reasons = list(intent.ambiguity_reasons)
    response.clarification_questions = list(intent.clarification_questions)
    return response


def _tool_context(caller: str, intent: ChatIntent, request: ChatRequest) -> ToolCallContext:
    return ToolCallContext(
        caller=caller,
        session_id=request.session_id,
        intent=",".join(intent.actions),
        metadata={
            "topic": intent.topic,
            "confidence": intent.confidence,
            "ambiguous": intent.ambiguous,
        },
    )


def _active_papers(workspace: LiteratureWorkspace):
    return [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]


def _active_citations(workspace: LiteratureWorkspace):
    return citation_records_for_papers(_active_papers(workspace))


def _workspace_is_mock(workspace: LiteratureWorkspace) -> bool:
    return workspace_is_mock(workspace, _active_papers(workspace))


def _workspace_has_real_minimum_evidence(workspace: LiteratureWorkspace) -> bool:
    return workspace_has_real_minimum_evidence(workspace, _active_papers(workspace))


def _workspace_evidence_quality(workspace: LiteratureWorkspace) -> dict[str, int]:
    return workspace_evidence_quality(workspace)


def _is_search_only_intent(intent: ChatIntent) -> bool:
    non_search_actions = {
        action
        for action in intent.actions
        if action
        not in {
            "search",
            "download",
        }
    }
    return "search" in intent.actions and not non_search_actions


def _is_analysis_request(message: str) -> bool:
    return is_analysis_request(message)


def _mock_context_refusal() -> str:
    return mock_context_refusal()


def _insufficient_real_evidence_reply(workspace: LiteratureWorkspace) -> str:
    return insufficient_real_evidence_reply(workspace)


def _format_component_status() -> str:
    return format_component_status()


def _apply_download_selection(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
) -> tuple[LiteratureWorkspace, str]:
    active_ids = list(workspace.context.active_papers)
    if intent.select_all_downloads:
        selected = active_ids
    else:
        selected = list(workspace.context.selected_for_download)

    if intent.clear_download_selection:
        selected = []

    for index in intent.select_indices:
        paper_id = _paper_id_for_index(active_ids, index)
        if paper_id and paper_id not in selected:
            selected.append(paper_id)

    for index in intent.deselect_indices:
        paper_id = _paper_id_for_index(active_ids, index)
        if paper_id in selected:
            selected.remove(paper_id)

    workspace.context.selected_for_download = selected
    if not active_ids:
        return workspace, "当前上下文还没有文献，暂时无法选择下载。"
    if not selected:
        return workspace, "已清空下载选择。"

    names = []
    for paper_id in selected:
        paper = workspace.papers[paper_id]
        names.append(f"{active_ids.index(paper_id) + 1}. {paper.title}")
    return workspace, "已选择下载：\n" + "\n".join(names)


def _apply_literature_filters(
    workspace: LiteratureWorkspace,
    intent: ChatIntent,
) -> LiteratureWorkspace:
    if intent.year_min is None and not intent.journals:
        return workspace
    active = []
    excluded = list(workspace.context.excluded_papers)
    for paper_id in workspace.context.active_papers:
        paper = workspace.papers[paper_id]
        keep = True
        if intent.year_min is not None and paper.year is not None and paper.year < intent.year_min:
            keep = False
        if intent.journals:
            source = f"{paper.journal or ''} {paper.publisher or ''}".lower()
            keep = keep and any(journal.lower() in source for journal in intent.journals)
        if keep:
            active.append(paper_id)
        elif paper_id not in excluded:
            excluded.append(paper_id)
    workspace.context.active_papers = active
    workspace.context.excluded_papers = excluded
    _merge_filters(
        workspace.context.filters, {"year_min": intent.year_min, "journals": intent.journals}
    )
    return workspace


def _format_current_papers(workspace: LiteratureWorkspace) -> str:
    return format_current_papers(workspace)


def _format_search_result_reply(
    topic: str,
    workspace: LiteratureWorkspace,
    expanded_year_range: bool = False,
    original_year_min: int | None = None,
) -> str:
    return format_search_result_reply(topic, workspace, expanded_year_range, original_year_min)


def _append_chat_trace_step(
    research_result,
    node: str,
    status: str,
    reason: str,
    inputs: dict[str, object] | None = None,
    outputs: dict[str, object] | None = None,
) -> None:
    if research_result is None or research_result.workflow_trace is None:
        return
    research_result.workflow_trace.steps.append(
        WorkflowTraceStep(
            node=node,
            status=status,
            reason=reason,
            inputs=inputs or {},
            outputs=outputs or {},
        )
    )


def _prepend_intent_trace_step(
    research_result,
    topic: str,
    intent: ChatIntent,
    live: bool,
    query_variant_count: int,
) -> None:
    if research_result is None or research_result.workflow_trace is None:
        return
    step = WorkflowTraceStep(
        node="parse_user_intent",
        status="completed",
        reason="拆解用户输入，提取研究主题、动作、年份限制和检索模式。",
        inputs={"message_topic": topic},
        outputs={
            "topic": topic,
            "actions": intent.actions,
            "year_min": intent.year_min,
            "journals": intent.journals,
            "live_search": live,
            "minimum_required": MIN_ANALYSIS_PAPERS,
            "query_variant_count": query_variant_count,
        },
        next_node="plan_sources",
        next_reason="意图拆解完成后进入检索源规划。",
    )
    research_result.workflow_trace.steps = [step, *research_result.workflow_trace.steps]


def _append_search_diagnostics_trace(research_result, workspace: LiteratureWorkspace) -> None:
    diagnostics = getattr(workspace.context.filters, "search_diagnostics", None)
    if (
        research_result is None
        or research_result.workflow_trace is None
        or not isinstance(diagnostics, dict)
    ):
        return
    variants = diagnostics.get("query_variants") or []
    source_counts = diagnostics.get("source_counts") or {}
    filtered_counts = diagnostics.get("filtered_counts") or {}
    errors = diagnostics.get("errors") or []
    research_result.workflow_trace.steps.append(
        WorkflowTraceStep(
            node="query_expansion_diagnostics",
            status="completed",
            reason=(
                "为避免中文窄查询漏召回，检索层已使用英文同义词和材料/器件概念扩展；"
                "若仍不足五篇，通常是源返回少、HTTP 错误、或相关性过滤剔除了弱相关记录。"
            ),
            inputs={"query_variants": variants[:6]},
            outputs={
                "source_counts": source_counts,
                "filtered_counts": filtered_counts,
                "errors": errors[:5],
            },
        )
    )


def _append_evidence_quality_trace(research_result, workspace: LiteratureWorkspace) -> None:
    if research_result is None or research_result.workflow_trace is None:
        return
    quality = _workspace_evidence_quality(workspace)
    status = "strong" if quality["parsed_count"] else "missing_full_text"
    reason = (
        "已有全文解析证据，可支持更强的分析。"
        if quality["parsed_count"]
        else "当前没有全文解析证据；metadata/abstract fallback 已禁用。"
    )
    research_result.workflow_trace.steps.append(
        WorkflowTraceStep(
            node="evidence_quality_gate",
            status=status,
            reason=reason,
            inputs={"active_papers": quality["active_count"]},
            outputs=quality,
        )
    )


def _append_query_expansion_trace(research_result, topic: str) -> None:
    variants = build_query_variants(topic)
    if research_result is None or research_result.workflow_trace is None or len(variants) <= 1:
        return
    research_result.workflow_trace.steps.append(
        WorkflowTraceStep(
            node="query_expansion",
            status="completed",
            reason=(
                "原始中文材料主题较窄，直接投给 OpenAlex/Crossref 容易漏召回；"
                "因此扩展为英文材料、器件、漂移和稳定性同义词后再检索。"
            ),
            inputs={"original_topic": topic},
            outputs={"query_variants": variants[:6]},
        )
    )


def _should_run_composite(intent: ChatIntent) -> bool:
    composite_actions = {
        "search",
        "download",
        "parse",
        "table",
        "storyline",
        "document",
        "autonomous_review",
    }
    return any(action in composite_actions for action in intent.actions)


def _paper_id_for_index(active_ids: list[str], index: int) -> str | None:
    position = index - 1
    if position < 0 or position >= len(active_ids):
        return None
    return active_ids[position]


async def _research_background_gate(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
    intent: ChatIntent,
) -> ChatResponse | None:
    if not request.session_id and not request.research_background:
        return None
    if workspace_has_research_background(workspace):
        return None

    allowed_actions = {"list_context", "show_context", "hide_context", "component_status"}
    if any(action in allowed_actions for action in intent.actions):
        return None

    explicit_background = request.research_background
    candidate = explicit_background or request.message
    assessment = await assess_research_background(candidate, config)
    if not assessment.accepted:
        mark_workspace_research_background_rejected(
            workspace,
            assessment.reason or "invalid",
        )
        return ChatResponse(
            reply=(
                "这个 session 需要先设置一个明确的研究背景，我暂时不会把当前内容作为长期科研主题。\n\n"
                + "\n".join(f"- {item}" for item in assessment.suggestions)
            ),
            action="research_background_required",
            workspace=WorkspaceSummary.from_workspace(workspace),
            warnings=[assessment.reason or "invalid_research_background"],
        )

    set_workspace_research_background(
        workspace,
        assessment.background or str(candidate or ""),
        topic=assessment.topic,
        retrieval_policy=assessment.retrieval_policy,
    )
    return ChatResponse(
        reply=(
            "已记录这个 session 的研究背景，并会把它作为长期记忆用于每日文献检索、PDF 下载和 RAG 更新。\n\n"
            f"研究主题：{workspace.context.filters.topic}\n\n"
            "接下来你可以告诉我具体要检索、下载、比较或分析什么。"
        ),
        action="research_background_set",
        workspace=WorkspaceSummary.from_workspace(workspace),
    )


async def _rag_evidence_for_workspace(
    config: LitTraceConfig,
    request: ChatRequest,
    workspace: LiteratureWorkspace,
) -> tuple[list[EvidenceSpan], list[str]]:
    warnings: list[str] = []
    if not config.rag.enabled or config.rag.backend != "pgvector":
        return [], warnings
    try:
        result = await search_workspace_rag(
            config,
            workspace,
            request.message,
            top_k=config.rag.top_k,
        )
    except Exception as exc:
        warnings.append(f"rag_search_failed:{exc.__class__.__name__}")
        return [], warnings
    if result is None:
        return [], warnings
    rag_evidence = rag_hits_to_evidence_spans(result.profile, result.hits, query=request.message)
    workspace.context.filters.rag_profile = result.profile.model_dump(mode="json")
    workspace.context.filters.rag_enabled = True
    workspace.context.filters.rag_backend = result.profile.backend
    workspace.context.filters.rag_last_query = request.message
    workspace.context.filters.rag_last_hit_count = len(rag_evidence)
    workspace.context.filters.rag_source_routes = list(result.profile.source_routes)
    return rag_evidence, warnings
