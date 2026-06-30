from __future__ import annotations

import asyncio
from dataclasses import dataclass

from littrace.chat import handle_chat
from littrace.config import load_config
from littrace.export import export_session_bundle
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.session import append_message, create_chat_session, save_workspace


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
    print("输入研究任务开始。命令：/context /hide-context /show-context /papers /export /quit")
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


if __name__ == "__main__":
    main()
