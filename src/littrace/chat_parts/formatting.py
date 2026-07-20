from __future__ import annotations

from littrace.agents import agent_runtime_statuses
from littrace.models import LiteratureWorkspace

MIN_ANALYSIS_PAPERS = 5


def format_agent_status() -> str:
    lines = ["当前 Agent 开发状态："]
    for status in agent_runtime_statuses():
        flag = "可执行" if status.implemented else "待开发"
        node = f"，节点：{status.workflow_node}" if status.workflow_node else ""
        remaining = "；剩余：" + " / ".join(status.remaining_work) if status.remaining_work else ""
        lines.append(f"- {status.name}: {flag}，runtime: {status.runtime}{node}{remaining}")
    return "\n".join(lines)


def format_current_papers(workspace: LiteratureWorkspace) -> str:
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


def format_search_result_reply(
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
