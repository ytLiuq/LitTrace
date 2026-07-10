from __future__ import annotations

from littrace.access import build_download_plan
from littrace.agents import agent_runtime_statuses
from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.citation_guard import guard_citations, remove_unsupported_sentences
from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.context import apply_context_update, _merge_filters
from littrace.document_composer import build_research_document_report
from littrace.harnesses import (
    check_hallucination_grounding,
    HallucinationCheckItem,
)
from littrace.intent import ChatIntent, parse_chat_intent
from littrace.intent_llm import IntentParseError, parse_chat_intent_semantic
from littrace.log import get_logger, timed, cost_tracker
from littrace.models import (
    ChatRequest,
    ChatResponse,
    ContextUpdate,
    LiteratureWorkspace,
    PaperSearchRequest,
    WorkflowTraceStep,
    coerce_parsed,
)
from littrace.parsing import parse_workspace_papers
from littrace.research_writer import (
    write_evidence_grounded_answer,
    write_storyline_narrative,
)
from littrace.storyline import build_storyline_from_workspace
from littrace.tables import build_comparison_matrices, extract_performance_cells
from littrace.workflow import run_research_graph
from littrace.search import build_query_variants

logger = get_logger("chat")


MIN_ANALYSIS_PAPERS = 5


async def handle_chat(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    message = request.message.strip()
    with timed("intent_parse"):
        try:
            intent = await parse_chat_intent_semantic(message, config)
        except IntentParseError as exc:
            logger.warning(
                "intent_parse_failed", extra={"msg_preview": message[:200], "error": str(exc)}
            )
            return (
                ChatResponse(
                    reply=(
                        f"{exc}\n\n"
                        "我没有继续执行，也没有回退到关键词规则。请检查 .env.local/config.yaml "
                        "里的 LLM 配置，或显式关闭 llm.intent_parser_enabled。"
                    ),
                    action="intent_parse_error",
                    workspace=workspace,
                    warnings=[str(exc)],
                ),
                workspace,
            )
    logger.info(
        "intent_parsed",
        extra={"actions": intent.actions, "topic": intent.topic, "year_min": intent.year_min},
    )

    if "show_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=True))
        return _response("已显示当前文献上下文。", "show_context", workspace), workspace

    if "hide_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=False))
        return _response(
            "已隐藏当前文献上下文，后续对话会保持简洁。", "hide_context", workspace
        ), workspace

    if intent.actions == ["list_context"]:
        return (
            ChatResponse(
                reply=_format_current_papers(workspace),
                action="list_context",
                workspace=workspace,
                citations=_active_citations(workspace),
            ),
            workspace,
        )

    if intent.actions == ["agent_status"]:
        return (
            ChatResponse(
                reply=_format_agent_status(),
                action="agent_status",
                workspace=workspace,
            ),
            workspace,
        )

    if any(action in intent.actions for action in ["select_downloads", "deselect_downloads"]):
        workspace, reply = _apply_download_selection(workspace, intent)
        return (
            ChatResponse(
                reply=reply,
                action="select_downloads",
                workspace=workspace,
                citations=_active_citations(workspace),
            ),
            workspace,
        )

    if _should_run_composite(intent):
        logger.info("composite_intent", extra={"actions": intent.actions})
        return await _run_composite_intent(intent, request, workspace, config)

    if _is_analysis_request(message) and _workspace_is_mock(workspace):
        logger.warning("blocked_mock_context", extra={"msg_preview": message[:200]})
        return (
            ChatResponse(
                reply=_mock_context_refusal(),
                action="blocked_mock_context",
                workspace=workspace,
                citations=[],
                warnings=["当前上下文来自 mock/开发样例，已禁止生成研究结论。"],
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
            ChatResponse(
                reply=_insufficient_real_evidence_reply(workspace),
                action="insufficient_real_evidence",
                workspace=workspace,
                citations=_active_citations(workspace),
                warnings=[f"真实相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已拒绝生成分析型结论。"],
            ),
            workspace,
        )

    llm_reply = await write_evidence_grounded_answer(config, message, workspace)
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
            ChatResponse(
                reply=reply_text,
                action="llm_chat",
                workspace=workspace,
                citations=_active_citations(workspace),
                warnings=guard.warnings + guard.unsupported_sentences[:3] + extra_warnings,
            ),
            workspace,
        )
    # LLM unavailable — no degradation, return error
    if workspace.context.active_papers:
        logger.error("llm_unavailable", extra={"error": llm_reply.error})
        return (
            ChatResponse(
                reply=(
                    f"LLM 调用失败，无法生成回答。错误：{llm_reply.error}\n\n"
                    "请检查 LLM 配置（DEEPSEEK_API_KEY / base_url / model）后重试。"
                ),
                action="llm_error",
                workspace=workspace,
                citations=_active_citations(workspace),
                warnings=[llm_reply.error or "llm_unavailable"],
            ),
            workspace,
        )

    return (
        ChatResponse(
            reply=(
                "我可以用对话方式帮你检索论文、显示/隐藏文献上下文、规划下载、解析全文、"
                "抽取性能表格，或生成有证据约束的发展脉络。你可以说：检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表。"
            ),
            action="help",
            workspace=workspace,
        ),
        workspace,
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
                llm_reply = await write_evidence_grounded_answer(config, request.message, workspace)
                if llm_reply.used_llm:
                    claim_hints = config.citation_guard.claim_hints or None
                    guard = guard_citations(llm_reply.text, workspace, claim_hints=claim_hints)
                    cleaned = remove_unsupported_sentences(llm_reply.text, guard)
                    workspace.guard_reports.append(guard.model_dump())
                    replies.append(cleaned)
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
        parse_config = config
        if intent.parse_strategy:
            parse_config = config.model_copy(deep=True)
            parse_config.parsing.parse_strategy = intent.parse_strategy
        workspace, report = parse_workspace_papers(workspace, parse_config)
        replies.append(
            f"已尝试解析全文：成功解析 {report['parsed_count']} 篇，失败 {report['failed_count']} 篇。"
        )
        warnings.append(str(report))
        action = "parse_full_text" if action == "composite" else action

    evidence_quality = _workspace_evidence_quality(workspace)
    if "table" in intent.actions and has_minimum_evidence:
        workspace, harness = await extract_performance_cells(workspace, config)
        matrix = build_comparison_matrices(workspace)
        replies.append(
            f"已生成性能对比表：抽取 {len(workspace.performance_cells)} 个指标单元，形成 {len(matrix.matrices)} 个指标矩阵。"
        )
        warnings.extend([*harness.errors, *harness.warnings, *matrix.warnings])
        action = "build_table" if action == "composite" else action
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
        storyline = build_storyline_from_workspace(workspace)
        if not storyline:
            replies.append("当前证据不足以生成真实的发展脉络。建议先检索并解析全文。")
        else:
            narrative = await write_storyline_narrative(config, workspace)
            if narrative.used_llm:
                claim_hints = config.citation_guard.claim_hints or None
                guard = guard_citations(narrative.text, workspace, claim_hints=claim_hints)
                cleaned_narrative = remove_unsupported_sentences(narrative.text, guard)
                workspace.guard_reports.append(guard.model_dump())
                replies.append(cleaned_narrative)
                warnings.extend(guard.warnings)
                warnings.extend(guard.unsupported_sentences[:3])

                # Dimension 2: Hallucination harness for storyline
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
            else:
                replies.append("已基于当前证据生成发展脉络草案；低证据部分会保持保守。")
                if narrative.error:
                    warnings.append(narrative.error)
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
        if evidence_quality["parsed_count"] == 0:
            warnings.append("当前缺少全文解析证据，已禁止基于题录/摘要生成报告。")
        report = build_research_document_report(workspace, config)
        workspace.context.filters.document_report = report.model_dump(mode="json")
        replies.append(
            f"已生成可审计研究报告：{len(report.sections)} 个章节，"
            f"{report.evidence_count} 条证据锚点，{len(report.citation_records)} 条引用。"
        )
        if report.warnings:
            warnings.extend(report.warnings[:5])
        action = "compose_document" if action == "composite" else action
    elif "document" in intent.actions and not has_minimum_evidence:
        warnings.append(f"相关文献不足 {MIN_ANALYSIS_PAPERS} 篇，已跳过报告生成。")

    if "autonomous_review" in intent.actions:
        loop_report = await run_autonomous_research_loop(
            config,
            intent.topic or request.message,
            workspace,
            auto_replan=intent.auto_replan,
        )
        workspace.context.filters.autonomous_loop_report = loop_report.model_dump(mode="json")
        replies.append(
            f"已完成多 agent 审稿/反驳/修订循环：{len(loop_report.rounds)} 轮，"
            f"score={loop_report.score:.3f}，passed={loop_report.passed}。"
        )
        replies.append(loop_report.final_answer)
        warnings.extend(loop_report.warnings[:5])
        action = "autonomous_review" if action == "composite" else action

    if "download" in intent.actions and not intent.skip_download:
        papers = _active_papers(workspace)
        download_plan = build_download_plan(
            config, papers, set(workspace.context.selected_for_download)
        )
        replies.append(
            f"已生成下载计划：{download_plan.downloadable_count} 篇可处理，其中 {download_plan.requires_login_count} 篇需要登录。"
        )
        action = "plan_downloads" if action == "composite" else action

    if not replies and "search" in intent.actions:
        replies.append(_format_search_result_reply(intent.topic or request.message, workspace))
    if not replies:
        replies.append("已理解你的指令，但当前没有可执行动作。")

    return (
        ChatResponse(
            reply="\n".join(replies),
            action=action,
            workspace=workspace,
            research_result=research_result,
            citations=_active_citations(workspace),
            download_plan=download_plan,
            publisher_routes=publisher_routes,
            comparison_matrix=matrix,
            warnings=warnings,
        ),
        workspace,
    )


def _response(reply: str, action: str, workspace: LiteratureWorkspace) -> ChatResponse:
    return ChatResponse(reply=reply, action=action, workspace=workspace)


def _active_papers(workspace: LiteratureWorkspace):
    return [workspace.papers[paper_id] for paper_id in workspace.context.active_papers]


def _active_citations(workspace: LiteratureWorkspace):
    return citation_records_for_papers(_active_papers(workspace))


def _workspace_is_mock(workspace: LiteratureWorkspace) -> bool:
    if getattr(workspace.context.filters, "search_mode", None) == "mock":
        return True
    return any((paper.doi or "").lower().find(".mock") >= 0 for paper in _active_papers(workspace))


def _workspace_has_real_minimum_evidence(workspace: LiteratureWorkspace) -> bool:
    return (
        getattr(workspace.context.filters, "search_mode", None) == "live"
        and not _workspace_is_mock(workspace)
        and len(workspace.context.active_papers) >= MIN_ANALYSIS_PAPERS
    )


def _workspace_evidence_quality(workspace: LiteratureWorkspace) -> dict[str, int]:
    active = set(workspace.context.active_papers)
    parsed_count = sum(
        bool(coerce_parsed(parsed).parsed)
        for paper_id, parsed in workspace.parsed_papers.items()
        if paper_id in active
    )
    full_text_count = sum(1 for paper_id in active if paper_id in workspace.full_text_reports)
    performance_count = len(workspace.performance_cells)
    return {
        "active_count": len(active),
        "full_text_report_count": full_text_count,
        "parsed_count": parsed_count,
        "performance_cell_count": performance_count,
        "candidate_pool_count": getattr(
            workspace.context.filters, "candidate_pool_count", len(active)
        ),
    }


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
    lowered = message.lower()
    return any(
        token in lowered
        for token in [
            "总结",
            "主要路线",
            "研究路线",
            "技术路线",
            "脉络",
            "发展",
            "故事",
            "综述",
            "比较",
            "分析",
            "summarize",
            "route",
            "storyline",
            "review",
        ]
    )


def _mock_context_refusal() -> str:
    return (
        "当前文献上下文来自 mock/开发样例，不是真实联网文献。我不能基于这些内容总结研究路线或生成结论。\n\n"
        "请先重新进行真实联网检索；当 live search 返回至少 5 篇真实相关文献后，我再按证据总结主要路线。"
    )


def _insufficient_real_evidence_reply(workspace: LiteratureWorkspace) -> str:
    count = len(workspace.context.active_papers)
    mode = getattr(workspace.context.filters, "search_mode", None) or "unknown"
    return (
        f"当前只有 {count} 篇候选文献，检索模式为 {mode}；还没有达到 "
        f"{MIN_ANALYSIS_PAPERS} 篇真实相关文献的分析门槛。\n\n"
        "我先不做总结型结论。请先继续真实联网检索、扩大关键词，或解析 PDF 全文后再总结主要路线。"
    )


def _format_agent_status() -> str:
    lines = ["当前 Agent 开发状态："]
    for status in agent_runtime_statuses():
        flag = "可执行" if status.implemented else "待开发"
        node = f"，节点：{status.workflow_node}" if status.workflow_node else ""
        remaining = "；剩余：" + " / ".join(status.remaining_work) if status.remaining_work else ""
        lines.append(f"- {status.name}: {flag}，runtime: {status.runtime}{node}{remaining}")
    return "\n".join(lines)


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
    if not workspace.context.active_papers:
        return "当前上下文还没有文献。你可以先让我检索一个主题。"
    lines = ["当前上下文文献："]
    for index, paper_id in enumerate(workspace.context.active_papers, start=1):
        paper = workspace.papers[paper_id]
        year = paper.year or "n.d."
        journal = paper.journal or paper.publisher or "unknown source"
        selected = "，已选下载" if paper_id in workspace.context.selected_for_download else ""
        lines.append(f"{index}. {paper.title} ({year}, {journal}{selected})")
    return "\n".join(lines)


def _format_search_result_reply(
    topic: str,
    workspace: LiteratureWorkspace,
    expanded_year_range: bool = False,
    original_year_min: int | None = None,
) -> str:
    search_mode = getattr(workspace.context.filters, "search_mode", None)
    lines = [
        f"我已围绕“{topic}”检索并尝试获取全文。",
    ]
    if search_mode == "mock":
        lines.append(
            "- 当前是 mock/开发样例检索，不是真实联网文献；这些结果只能用于验证流程和界面。"
        )
        lines.append(
            "- 真实调研需要启用 live search 后重新检索，达到 5 篇真实文献后我再做分析型结论。"
        )
        return "\n".join(lines)
    if expanded_year_range:
        lines.append(f"最近年份范围（{original_year_min} 年以来）证据不足，已自动扩大检索年限。")
    if not workspace.context.active_papers:
        login_ids = getattr(workspace.context.filters, "requires_login_candidate_ids", None) or []
        lines.append(
            "目前还没有可进入上下文的已下载全文文献，因此我不会基于题录或摘要给出分析结论。"
        )
        if login_ids:
            lines.append("部分相关文献可能需要 publisher 登录授权后才能下载全文。")
        return "\n".join(lines)
    if len(workspace.context.active_papers) < MIN_ANALYSIS_PAPERS:
        lines.append(f"当前全文证据还不足 {MIN_ANALYSIS_PAPERS} 篇，我先不做分析型结论。")
        lines.append("可以继续扩大关键词/年份，或完成需要授权的 publisher 登录后再继续。")
        return "\n".join(lines)

    lines.append("全文证据已达到最低门槛，可以继续提出分析问题。")
    for index, paper_id in enumerate(workspace.context.active_papers, start=1):
        paper = workspace.papers[paper_id]
        year = paper.year or "n.d."
        source = paper.journal or paper.publisher or "unknown source"
        lines.append(f"{index}. {paper.title} ({year}, {source})")
    return "\n".join(lines)


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
