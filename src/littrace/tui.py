from __future__ import annotations

import asyncio
import curses
import textwrap
from dataclasses import dataclass, field

from littrace.chat import handle_chat
from littrace.config import load_config
from littrace.models import ChatRequest, LiteratureWorkspace
from littrace.session import append_message, create_chat_session, save_workspace
from littrace.evidence.tables import decide_artifact_extraction_need


@dataclass
class TUIState:
    workspace: LiteratureWorkspace
    session_id: str
    session_root: str
    messages: list[tuple[str, str]] = field(default_factory=list)
    input_text: str = ""
    context_visible: bool = True
    status: str = "Ready"


def main() -> None:
    curses.wrapper(lambda screen: asyncio.run(run_tui(screen)))


async def run_tui(screen) -> None:
    curses.curs_set(1)
    screen.keypad(True)
    config = load_config()
    session = create_chat_session(config)
    state = TUIState(
        workspace=LiteratureWorkspace(),
        session_id=session.session_id,
        session_root=str(session.root),
        messages=[
            (
                "LitTrace",
                "本地可视化 Agent 已启动。输入研究任务，或输入 /ocr-choice、/hide-context、/show-context、/quit。",
            )
        ],
    )

    while True:
        _draw(screen, state)
        key = screen.getch()
        if key in {10, 13, curses.KEY_ENTER}:
            text = state.input_text.strip()
            state.input_text = ""
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                return
            await _handle_input(text, state, config, session)
            continue
        if key in {curses.KEY_BACKSPACE, 127, 8}:
            state.input_text = state.input_text[:-1]
            continue
        if key == curses.KEY_RESIZE:
            continue
        if 0 <= key <= 255:
            char = chr(key)
            if char.isprintable():
                state.input_text += char


async def _handle_input(text: str, state: TUIState, config, session) -> None:
    state.messages.append(("You", text))
    if text == "/hide-context":
        state.context_visible = False
        state.workspace.context.visible_to_user = False
        state.messages.append(("LitTrace", "已隐藏右侧文献上下文窗。"))
        return
    if text == "/show-context":
        state.context_visible = True
        state.workspace.context.visible_to_user = True
        state.messages.append(("LitTrace", "已显示右侧文献上下文窗。"))
        return
    if text == "/ocr-choice":
        state.messages.append(("LitTrace", "\n".join(render_ocr_choice_lines(state.workspace))))
        return

    state.status = "Thinking..."
    response, workspace = await handle_chat(
        ChatRequest(message=text, session_id=state.session_id),
        state.workspace,
        config,
    )
    state.workspace = workspace
    state.context_visible = workspace.context.visible_to_user
    save_workspace(session, state.workspace)
    append_message(session, "user", text)
    append_message(session, "assistant", response)
    state.messages.append(("LitTrace", response.reply))
    if response.warnings:
        state.messages.append(("Warnings", "；".join(response.warnings[:4])))
    state.status = f"Action: {response.action}"


def _draw(screen, state: TUIState) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    if height < 10 or width < 50:
        screen.addstr(0, 0, "Window too small for LitTrace TUI.")
        screen.refresh()
        return

    context_width = min(46, max(30, width // 3)) if state.context_visible else 0
    chat_width = width - context_width - (1 if context_width else 0)
    input_height = 3
    header = f" LitTrace TUI | session {state.session_id} | /ocr-choice /hide-context /show-context /quit "
    _add_clipped(screen, 0, 0, header, width, curses.A_REVERSE)

    chat_height = height - input_height - 2
    _draw_chat(screen, state, 1, 0, chat_height, chat_width)
    if state.context_visible and context_width:
        divider_x = chat_width
        for y in range(1, height - input_height):
            _add_clipped(screen, y, divider_x, "|", 1)
        _draw_context(screen, state, 1, chat_width + 1, chat_height, context_width - 1)

    status = f" {state.status} | folder: {state.session_root} "
    _add_clipped(screen, height - input_height - 1, 0, status, width, curses.A_REVERSE)
    prompt = "littrace > "
    _add_clipped(screen, height - input_height, 0, prompt + state.input_text, width)
    cursor_x = min(len(prompt) + len(state.input_text), width - 1)
    screen.move(height - input_height, cursor_x)
    screen.refresh()


def _draw_chat(screen, state: TUIState, top: int, left: int, height: int, width: int) -> None:
    lines: list[str] = []
    for role, text in state.messages:
        prefix = f"{role}: "
        wrapped = wrap_text(text, max(width - len(prefix), 10))
        if not wrapped:
            lines.append(prefix)
            continue
        lines.append(prefix + wrapped[0])
        for line in wrapped[1:]:
            lines.append(" " * len(prefix) + line)
        lines.append("")
    visible = lines[-height:]
    for offset, line in enumerate(visible):
        _add_clipped(screen, top + offset, left, line, width)


def _draw_context(screen, state: TUIState, top: int, left: int, height: int, width: int) -> None:
    lines = render_context_lines(state.workspace)
    lines.extend(["", "OCR 选择:"])
    lines.extend(render_ocr_choice_lines(state.workspace))
    for offset, line in enumerate(lines[:height]):
        attr = curses.A_BOLD if offset == 0 else curses.A_NORMAL
        _add_clipped(screen, top + offset, left, line, width, attr)


def render_context_lines(workspace: LiteratureWorkspace) -> list[str]:
    ids = workspace.context.active_papers
    if not ids:
        return ["文献上下文", "当前没有文献。", "输入：检索 MXene flexible sensor"]
    selected = set(workspace.context.selected_for_download)
    pool_count = getattr(workspace.context.filters, "candidate_pool_count", len(ids))
    lines = [f"文献上下文 ({len(ids)} / 候选池 {pool_count} 篇)"]
    for index, paper_id in enumerate(ids[:12], start=1):
        paper = workspace.papers[paper_id]
        marker = "*" if paper_id in selected else " "
        source = paper.journal or paper.publisher or "unknown"
        year = paper.year or "n.d."
        lines.append(f"{marker}{index}. {paper.title}")
        lines.append(f"   {year} | {source}")
    if len(ids) > 12:
        lines.append(f"... 还有 {len(ids) - 12} 篇")
    return lines


def render_ocr_choice_lines(workspace: LiteratureWorkspace) -> list[str]:
    report = decide_artifact_extraction_need(workspace)
    lines = [f"建议: {report.recommended_parse_strategy}", report.reason]
    for button in report.buttons:
        marker = "[推荐]" if button.get("recommended") == "true" else "[可选]"
        lines.append(f"{marker} {button['label']}")
        lines.append(f"  输入: 解析PDF {button['parse_strategy']}")
    return lines


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return lines


def _add_clipped(
    screen, y: int, x: int, text: str, width: int, attr: int = curses.A_NORMAL
) -> None:
    if width <= 0:
        return
    try:
        screen.addstr(y, x, text[: max(width - 1, 0)], attr)
    except curses.error:
        pass


if __name__ == "__main__":
    main()
