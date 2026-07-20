from __future__ import annotations

from littrace.models import LiteratureWorkspace, coerce_parsed

MIN_ANALYSIS_PAPERS = 5


def workspace_is_mock(workspace: LiteratureWorkspace, active_papers) -> bool:
    if getattr(workspace.context.filters, "search_mode", None) == "mock":
        return True
    return any((paper.doi or "").lower().find(".mock") >= 0 for paper in active_papers)


def workspace_has_real_minimum_evidence(workspace: LiteratureWorkspace, active_papers) -> bool:
    return (
        getattr(workspace.context.filters, "search_mode", None) == "live"
        and not workspace_is_mock(workspace, active_papers)
        and len(workspace.context.active_papers) >= MIN_ANALYSIS_PAPERS
    )


def workspace_evidence_quality(workspace: LiteratureWorkspace) -> dict[str, int]:
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


def is_analysis_request(message: str) -> bool:
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


def mock_context_refusal() -> str:
    return (
        "当前文献上下文来自 mock/开发样例，不是真实联网文献。我不能基于这些内容总结研究路线或生成结论。\n\n"
        "请先重新进行真实联网检索；当 live search 返回至少 5 篇真实相关文献后，我再按证据总结主要路线。"
    )


def insufficient_real_evidence_reply(workspace: LiteratureWorkspace) -> str:
    count = len(workspace.context.active_papers)
    mode = getattr(workspace.context.filters, "search_mode", None) or "unknown"
    return (
        f"当前只有 {count} 篇候选文献，检索模式为 {mode}；还没有达到 "
        f"{MIN_ANALYSIS_PAPERS} 篇真实相关文献的分析门槛。\n\n"
        "我先不做总结型结论。请先继续真实联网检索、扩大关键词，或解析 PDF 全文后再总结主要路线。"
    )
