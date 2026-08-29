from __future__ import annotations

import asyncio
import curses
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

from littrace.agent_runtime import handle_agent_chat
from littrace.config import load_config
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import (
    append_message,
    create_chat_session,
    save_workspace,
)
from littrace.evidence.tables import decide_artifact_extraction_need


# Round 17 — `mcpServer/elicitation/request` vocabulary. Codex sends
# three action verbs; the TUI maps each to a single keypress so the
# operator never has to type a free-form response.
_DECLINE_RESPONSE: dict[str, Any] = {
    "action": "decline",
    "content": None,
    "_meta": None,
}
_ACCEPT_RESPONSE: dict[str, Any] = {
    "action": "accept",
    "content": None,
    "_meta": None,
}
_CANCEL_RESPONSE: dict[str, Any] = {
    "action": "cancel",
    "content": None,
    "_meta": None,
}


@dataclass
class ApprovalRequest:
    """A pending ``mcpServer/elicitation/request`` waiting for the operator.

    The handler installed on the AppServerClient creates one of these
    per codex elicitation, parks a Future on it, and awaits the
    future's resolution. The TUI's main loop sets ``state.pending_approval``
    so the modal can render; the operator's Y/N/Esc keypress resolves
    the future, which unblocks the reader loop and lets codex proceed.
    """

    request_id: int | str
    server: str
    tool: str
    arguments: dict[str, Any]
    future: asyncio.Future


@dataclass
class TUIState:
    workspace: LiteratureWorkspace
    session_id: str
    session_root: str
    messages: list[tuple[str, str]] = field(default_factory=list)
    input_text: str = ""
    context_visible: bool = True
    status: str = "Ready"
    # Round 17: background turn bookkeeping. ``inflight_task`` is the
    # Task running ``handle_agent_chat``; while it is set the input
    # row is locked and the status row shows "Thinking…". ``pending_approval``
    # shadows the input row with the Y/N modal when the App Server
    # asks the operator to approve an MCP tool call.
    inflight_task: asyncio.Task | None = None
    pending_approval: ApprovalRequest | None = None


def main() -> None:
    curses.wrapper(lambda screen: asyncio.run(run_tui(screen)))


async def run_tui(screen) -> None:
    curses.curs_set(1)
    screen.keypad(True)
    # ``nodelay(True)`` makes ``getch()`` return -1 instead of blocking
    # so the loop can interleave drawing with the inflight chat task
    # and surface an approval modal while a turn is running.
    screen.nodelay(True)
    config = load_config()
    session = create_chat_session(config)
    state = TUIState(
        workspace=LiteratureWorkspace(),
        session_id=session.session_id,
        session_root=str(session.root),
        messages=[
            (
                "LitTrace",
                "本地可视化 Agent 已启动 (codex 模式)。MCP 工具调用时审批弹窗会出现在此处。",
            )
        ],
    )
    try:
        await _event_loop(screen, state, config, session)
    finally:
        # Cancel any in-flight turn so the App Server is not left
        # waiting for a reply to its own elicitations.
        if state.inflight_task is not None and not state.inflight_task.done():
            state.inflight_task.cancel()
            try:
                await state.inflight_task
            except (asyncio.CancelledError, Exception):
                pass
        # Resolve any pending approval so the reader loop does not
        # block on a Future whose owner has already exited.
        if state.pending_approval is not None and not state.pending_approval.future.done():
            state.pending_approval.future.set_result(_CANCEL_RESPONSE)


async def _event_loop(screen, state: TUIState, config, session) -> None:
    """Main loop. ``nodelay(True)`` lets us poll the inflight task
    between keystrokes and surface an approval modal mid-stream."""
    while True:
        _draw(screen, state)
        # If the previous turn finished, clear the bookkeeping so
        # the input row unlocks. The chat task already wrote its
        # outcome into ``state.messages`` / ``state.workspace``.
        if state.inflight_task is not None and state.inflight_task.done():
            state.inflight_task = None
        key = screen.getch()
        if key == -1:
            # No input: yield to the event loop so the inflight task
            # can progress and any pending reader-loop callbacks
            # (including elicitations) get a chance to run.
            await asyncio.sleep(0.05)
            continue
        if state.pending_approval is not None:
            await _handle_approval_key(key, state)
            continue
        if state.inflight_task is not None:
            # Input is locked while a turn is in flight; the only
            # escape is Ctrl-C, which ``curses.wrapper`` translates
            # into a KeyboardInterrupt that unwinds the loop.
            continue
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


async def _handle_approval_key(key: int, state: TUIState) -> None:
    approval = state.pending_approval
    assert approval is not None
    if key in (ord("y"), ord("Y")):
        approval.future.set_result(_ACCEPT_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: accepted — running tool"
        return
    if key in (ord("n"), ord("N")):
        approval.future.set_result(_DECLINE_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: declined"
        return
    if key == 27:  # Esc
        approval.future.set_result(_CANCEL_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: cancelled"
        return


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

    state.status = "Thinking…"
    handler = _make_elicitation_handler(state)
    state.inflight_task = asyncio.create_task(
        _run_chat(state, config, session, text, handler),
        name="littrace-tui-chat",
    )


async def _run_chat(
    state: TUIState,
    config,
    session,
    text: str,
    handler,
) -> None:
    """Body of the background chat task. Writes its outcome onto
    ``state`` so the next draw picks it up without us needing to
    await it from the main loop."""
    try:
        response, workspace = await handle_agent_chat(
            ChatRequest(message=text, session_id=state.session_id),
            state.workspace,
            config,
            session=session,
            elicitation_handler=handler,
        )
    except asyncio.CancelledError:
        state.status = "Cancelled"
        raise
    except Exception as exc:  # pragma: no cover - defensive
        state.messages.append(("Error", f"{exc.__class__.__name__}: {exc}"))
        state.status = f"Error: {exc.__class__.__name__}"
        return

    state.workspace = workspace
    state.context_visible = workspace.context.visible_to_user
    save_workspace(session, state.workspace, config=config)
    append_message(session, "user", text)
    append_message(session, "assistant", response.reply)
    state.messages.append(("LitTrace", response.reply))
    if response.warnings:
        state.messages.append(("Warnings", "；".join(response.warnings[:4])))
    state.status = f"Action: {response.action}"


def _make_elicitation_handler(state: TUIState):
    """Return an async callable for the AppServerClient to invoke
    on a ``mcpServer/elicitation/request`` notification.

    The handler stashes a Future on ``state.pending_approval`` and
    awaits it; the main loop resolves the Future with the operator's
    Y/N/Esc decision so codex can resume.
    """

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        approval = ApprovalRequest(
            request_id=params.get("serverRequestId")
            or params.get("requestId")
            or "",
            server=str(params.get("server") or params.get("serverName") or "littrace"),
            tool=str(params.get("tool") or params.get("name") or "?"),
            arguments=dict(params.get("arguments") or params.get("args") or {}),
            future=future,
        )
        state.pending_approval = approval
        try:
            return await future
        finally:
            if state.pending_approval is approval:
                state.pending_approval = None

    return handler


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

    if state.pending_approval is not None:
        _draw_approval_modal(screen, state, height, width)
    else:
        prompt = "littrace > "
        _add_clipped(screen, height - input_height, 0, prompt + state.input_text, width)
        cursor_x = min(len(prompt) + len(state.input_text), width - 1)
        screen.move(height - input_height, cursor_x)
    screen.refresh()


def _draw_approval_modal(screen, state: TUIState, height: int, width: int) -> None:
    approval = state.pending_approval
    assert approval is not None
    try:
        args_text = json.dumps(approval.arguments, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        args_text = repr(approval.arguments)
    lines = [
        "MCP Tool Approval Required",
        "",
        f"Server: {approval.server}",
        f"Tool:   {approval.tool}",
        "",
        "Arguments:",
    ]
    for raw in args_text.splitlines():
        wrapped = textwrap.wrap(raw, width=max(width // 2 - 6, 20)) or [""]
        lines.extend(wrapped)
    lines.extend(["", "[Y] Accept   [N] Decline   [Esc] Cancel"])

    box_width = min(max((len(line) for line in lines), default=20) + 4, width - 4)
    box_height = min(len(lines) + 2, height - 4)
    box_y = max(1, (height - box_height) // 2)
    box_x = max(0, (width - box_width) // 2)

    # Border (drawn char-by-char so nodelay redraws do not flicker).
    for dy in range(box_height):
        for dx in range(box_width):
            ch = " "
            if dy == 0 and dx == 0:
                ch = curses.ACS_ULCORNER
            elif dy == 0 and dx == box_width - 1:
                ch = curses.ACS_URCORNER
            elif dy == box_height - 1 and dx == 0:
                ch = curses.ACS_LLCORNER
            elif dy == box_height - 1 and dx == box_width - 1:
                ch = curses.ACS_LRCORNER
            elif dy == 0 or dy == box_height - 1:
                ch = curses.ACS_HLINE
            elif dx == 0 or dx == box_width - 1:
                ch = curses.ACS_VLINE
            try:
                screen.addch(box_y + dy, box_x + dx, ch)
            except curses.error:
                pass

    for offset, line in enumerate(lines[: box_height - 2]):
        attr = curses.A_BOLD if offset == 0 else curses.A_NORMAL
        _add_clipped(screen, box_y + 1 + offset, box_x + 2, line, box_width - 4, attr)


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