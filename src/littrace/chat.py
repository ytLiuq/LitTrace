from __future__ import annotations

from littrace.access import build_download_plan
from littrace.citations import citation_records_for_papers
from littrace.config import LitTraceConfig
from littrace.context import apply_context_update
from littrace.intent import ChatIntent, parse_chat_intent
from littrace.models import (
    ChatRequest,
    ChatResponse,
    ContextUpdate,
    LiteratureWorkspace,
    PaperSearchRequest,
)
from littrace.parsing import parse_workspace_papers
from littrace.storyline import build_storyline_from_workspace
from littrace.tables import build_comparison_matrices, extract_performance_cells
from littrace.workflow import run_research_graph, run_search_preview


async def handle_chat(
    request: ChatRequest,
    workspace: LiteratureWorkspace,
    config: LitTraceConfig,
) -> tuple[ChatResponse, LiteratureWorkspace]:
    message = request.message.strip()
    intent = parse_chat_intent(message)

    if "show_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=True))
        return _response("已显示当前文献上下文。", "show_context", workspace), workspace

    if "hide_context" in intent.actions:
        workspace = apply_context_update(workspace, ContextUpdate(visible_to_user=False))
        return _response("已隐藏当前文献上下文，后续对话会保持简洁。", "hide_context", workspace), workspace

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

    if _should_run_composite(intent):
        return await _run_composite_intent(intent, request, workspace, config)

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

    if "search" in intent.actions:
        workspace = await run_search_preview(
            PaperSearchRequest(
                topic=intent.topic or request.message,
                year_min=intent.year_min or config.literature_context.default_year_min,
                live=request.live,
            ),
            config,
        )
        workspace = _apply_literature_filters(workspace, intent)
        replies.append(
            f"已围绕“{intent.topic}”检索并更新上下文，当前保留 {len(workspace.context.active_papers)} 篇文献。"
        )
        action = "search"

    if "parse" in intent.actions:
        workspace, report = parse_workspace_papers(workspace, config)
        replies.append(
            f"已尝试解析全文：解析 {report['parsed_count']} 篇，metadata-only {report['metadata_only_count']} 篇。"
        )
        warnings.append(str(report))
        action = "parse_full_text" if action == "composite" else action

    if "table" in intent.actions:
        workspace, harness = extract_performance_cells(workspace)
        matrix = build_comparison_matrices(workspace)
        replies.append(
            f"已生成性能对比表：抽取 {len(workspace.performance_cells)} 个指标单元，形成 {len(matrix.matrices)} 个指标矩阵。"
        )
        warnings.extend([*harness.errors, *harness.warnings, *matrix.warnings])
        action = "build_table" if action == "composite" else action

    if "storyline" in intent.actions:
        storyline = build_storyline_from_workspace(workspace)
        if not storyline:
            replies.append("当前证据不足以生成真实的发展脉络。建议先检索并解析全文。")
        else:
            replies.append("已基于当前证据生成发展脉络草案；低证据部分会保持保守。")
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

    if "download" in intent.actions and not intent.skip_download:
        papers = _active_papers(workspace)
        download_plan = build_download_plan(config, papers, set(workspace.context.selected_for_download))
        replies.append(
            f"已生成下载计划：{download_plan.downloadable_count} 篇可处理，其中 {download_plan.requires_login_count} 篇需要登录。"
        )
        action = "plan_downloads" if action == "composite" else action

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
    workspace.context.filters.update({"year_min": intent.year_min, "journals": intent.journals})
    return workspace


def _format_current_papers(workspace: LiteratureWorkspace) -> str:
    if not workspace.context.active_papers:
        return "当前上下文还没有文献。你可以先让我检索一个主题。"
    lines = ["当前上下文文献："]
    for index, paper_id in enumerate(workspace.context.active_papers, start=1):
        paper = workspace.papers[paper_id]
        year = paper.year or "n.d."
        journal = paper.journal or paper.publisher or "unknown source"
        lines.append(f"{index}. {paper.title} ({year}, {journal})")
    return "\n".join(lines)


def _should_run_composite(intent: ChatIntent) -> bool:
    return any(
        action in intent.actions
        for action in ["search", "download", "parse", "table", "storyline"]
    )
