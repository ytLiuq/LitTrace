from __future__ import annotations

import asyncio
from dataclasses import dataclass

from littrace.attachments import attach_pdf_to_paper, check_download_presence
from littrace.auto_resume import auto_resume_downloaded_pdfs
from littrace.chat import handle_chat
from littrace.config import load_config
from littrace.export import export_session_bundle
from littrace.golden_eval import run_golden_eval
from littrace.login_flow import launch_login_for_paper
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.pdf_benchmark import benchmark_pdf_parsing
from littrace.session import append_message, create_chat_session, save_workspace
from littrace.storyline import render_structured_storyline_report


@dataclass
class ShellState:
    workspace: LiteratureWorkspace
    session_id: str
    session_root: str
    context_visible: bool = True


def main() -> None:
    asyncio.run(run_shell())


async def run_shell() -> None:
    config = load_config()
    session = create_chat_session(config)
    state = ShellState(
        workspace=LiteratureWorkspace(),
        session_id=session.session_id,
        session_root=str(session.root),
    )
    print("LitTrace agent shell")
    print(
        "输入研究任务开始。命令：/context /hide-context /show-context /papers "
        "/login N /attach N path.pdf /check-downloads /resume-downloads /parse /table /storyline "
        "/dashboard /storyline-report /benchmark /golden-eval /export /quit"
    )
    print("对话例子：选择第 1、3 篇下载；全部下载；取消选择第 2 篇；生成发展脉络。")
    print(f"session: {state.session_id}")
    print(f"folder:  {state.session_root}")
    print()

    while True:
        try:
            message = input("littrace > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not message:
            continue
        if message in {"/quit", "/exit"}:
            print("bye")
            return
        if message == "/hide-context":
            state.context_visible = False
            state.workspace.context.visible_to_user = False
            print("已隐藏上下文窗。")
            continue
        if message == "/show-context":
            state.context_visible = True
            state.workspace.context.visible_to_user = True
            print("已显示上下文窗。")
            print(format_context_panel(state.workspace))
            continue
        if message in {"/context", "/papers"}:
            print(format_context_panel(state.workspace))
            continue
        if message in {"/dashboard", "/tui"}:
            print(format_dashboard(state))
            continue
        if message == "/parse":
            message = "解析当前文献全文"
        if message == "/table":
            message = "生成当前文献性能对比表"
        if message == "/storyline":
            message = "生成当前文献发展脉络"
        if message == "/storyline-report":
            print(render_structured_storyline_report(state.workspace))
            continue
        if message.startswith("/login "):
            index = _parse_index_arg(message)
            paper_id = _paper_id_for_index(state.workspace, index) if index else None
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            paper = state.workspace.papers[paper_id]
            result = launch_login_for_paper(config, paper)
            print(f"登录页: {result.login_url or '无'}")
            print(f"目标路径: {result.target_path or '无'}")
            for instruction in result.instructions:
                print(f"- {instruction}")
            if result.error:
                print(f"错误: {result.error}")
            continue
        if message.startswith("/attach "):
            parsed = _parse_attach_args(message)
            if not parsed:
                print("用法：/attach N /path/to/paper.pdf")
                continue
            index, source_path = parsed
            paper_id = _paper_id_for_index(state.workspace, index)
            if not paper_id:
                print("没有找到这个编号的文献。")
                continue
            result = attach_pdf_to_paper(config, state.workspace, paper_id, source_path)
            print(f"PDF 绑定: {'成功' if result.attached else '失败'}")
            print(f"目标路径: {result.target_path}")
            if result.error:
                print(f"错误: {result.error}")
            save_workspace(session, state.workspace)
            continue
        if message == "/check-downloads":
            report = check_download_presence(config, state.workspace)
            print(
                f"PDF 检测：{report.ready_to_parse_count} 篇已就绪，"
                f"{report.missing_count} 篇缺失。"
            )
            for item in report.items[:12]:
                marker = "ok" if item.exists else "missing"
                print(f"- [{marker}] {item.title}: {item.expected_path}")
            if report.ready_to_parse_count:
                print("可运行 /resume-downloads 自动解析已就绪 PDF 并写入 artifacts。")
            continue
        if message == "/resume-downloads":
            state.workspace, result = auto_resume_downloaded_pdfs(config, state.workspace, session)
            save_workspace(session, state.workspace)
            print(
                f"自动恢复：ready={result.ready_to_parse_count}, parsed={result.parsed_count}, "
                f"performance_cells={result.performance_cell_count}"
            )
            if result.artifact_paths:
                print("Artifacts:")
                for name, path in result.artifact_paths.items():
                    print(f"- {name}: {path}")
            continue
        if message == "/benchmark":
            report = benchmark_pdf_parsing(state.workspace, config)
            print(
                "PDF/OCR benchmark: "
                f"active={report.active_papers}, local_pdf={report.local_pdf_count}, "
                f"parsed={report.parsed_count}, metadata_only={report.metadata_only_count}, "
                f"page_evidence={report.parsed_with_page_evidence}, "
                f"avg_conf={report.average_evidence_confidence}"
            )
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message == "/golden-eval":
            report = run_golden_eval(config)
            print(f"Golden eval: cases={report.case_count}, dir={report.golden_set_dir}")
            for name, value in report.metrics.items():
                print(f"- {name}: {value}")
            if report.warnings:
                print("注意：" + "；".join(report.warnings))
            continue
        if message == "/export":
            paths = export_session_bundle(session, state.workspace)
            print("已导出研究包：")
            for name, path in paths.items():
                print(f"- {name}: {path}")
            continue

        response, workspace = await handle_chat(
            ChatRequest(message=message, session_id=state.session_id),
            state.workspace,
            config,
        )
        response.session_id = state.session_id
        response.session_root = state.session_root
        state.workspace = workspace
        state.context_visible = state.workspace.context.visible_to_user
        save_workspace(session, state.workspace)
        append_message(session, "user", message)
        append_message(session, "assistant", response)
        print()
        print(f"LitTrace: {response.reply}")
        if response.download_plan:
            print(
                f"下载计划：{response.download_plan.downloadable_count} 篇可处理，"
                f"{response.download_plan.requires_login_count} 篇需要登录。"
            )
        if response.publisher_routes:
            routes = response.publisher_routes.get("routes", [])
            login_count = sum(1 for route in routes if route.get("requires_login"))
            print(f"出版商路线：{len(routes)} 条，{login_count} 条可能需要登录。")
        if response.comparison_matrix:
            print(f"性能矩阵：{len(response.comparison_matrix.matrices)} 个指标组。")
        if response.warnings:
            print("注意：" + "；".join(response.warnings[:3]))
        if state.context_visible:
            print()
            print(format_context_panel(state.workspace))
        print()


def format_context_panel(workspace: LiteratureWorkspace) -> str:
    ids = workspace.context.active_papers
    if not ids:
        return "[上下文窗] 当前没有文献。"
    selected = set(workspace.context.selected_for_download)
    visibility = "显示" if workspace.context.visible_to_user else "隐藏"
    lines = [
        f"[上下文窗:{visibility}] 当前文献 {len(ids)} 篇，已选下载 {len(selected)} 篇",
        "提示：可输入“选择第 1、3 篇下载”“全部下载”“取消选择第 2 篇”。",
    ]
    for index, paper_id in enumerate(ids[:12], start=1):
        paper = workspace.papers[paper_id]
        year = paper.year or "n.d."
        source = paper.journal or paper.publisher or "unknown source"
        marker = "*" if paper_id in selected else " "
        lines.append(
            f"{marker} {index}. {paper.title} "
            f"({year}, {source}, {paper.access_type}, id={paper.paper_id})"
        )
    if len(ids) > 12:
        lines.append(f"... 还有 {len(ids) - 12} 篇")
    return "\n".join(lines)


def format_dashboard(state: ShellState) -> str:
    workspace = state.workspace
    active = len(workspace.context.active_papers)
    selected = len(workspace.context.selected_for_download)
    parsed = len(workspace.parsed_papers)
    cells = len(workspace.performance_cells)
    visible = "显示" if workspace.context.visible_to_user else "隐藏"
    lines = [
        "[LitTrace Dashboard]",
        f"session: {state.session_id}",
        f"folder:  {state.session_root}",
        f"context: {active} papers, panel={visible}, selected_downloads={selected}",
        f"parsing: {parsed} parsed records, performance_cells={cells}",
        "commands: /context /login N /attach N path.pdf /check-downloads /parse /table /storyline-report /benchmark /export",
    ]
    return "\n".join(lines)


def _parse_index_arg(message: str) -> int | None:
    parts = message.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _parse_attach_args(message: str) -> tuple[int, str] | None:
    parts = message.split(maxsplit=2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


def _paper_id_for_index(workspace: LiteratureWorkspace, index: int | None) -> str | None:
    if index is None:
        return None
    position = index - 1
    if position < 0 or position >= len(workspace.context.active_papers):
        return None
    return workspace.context.active_papers[position]


if __name__ == "__main__":
    main()
