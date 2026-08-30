from __future__ import annotations

import asyncio
import curses
import json
import os
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Any

from littrace.agent_runtime import handle_agent_chat
from littrace.codex_runtime.errors import AppServerError
from littrace.config import AgentRuntimeMode, load_config
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import (
    append_message,
    create_chat_session,
    save_workspace,
)
from littrace.evidence.tables import decide_artifact_extraction_need


# Round 20: TUI must be a strict Codex surface. A failure to start Codex
# is rendered as a full-screen modal with a remediation list — never as
# a silent fallback to the legacy coordinator.
class CodexStartupError(RuntimeError):
    """Raised when the Codex App Server cannot be reached at TUI startup.

    Carries a ``remediation`` list of concrete steps the operator should
    try before retrying. ``message`` is a short human-readable summary.
    """

    def __init__(self, message: str, remediation: list[str]) -> None:
        super().__init__(message)
        self.remediation = list(remediation)


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

    Round 19: ``content_lines`` is the pre-rendered argument block
    (one item per terminal row, already word-wrapped). The modal
    pages through it with Up/Down/PgUp/PgDn so an
    ``enqueue_download`` carrying 50 paper IDs is not silently
    truncated to the box width.
    """

    request_id: int | str
    server: str
    tool: str
    arguments: dict[str, Any]
    future: asyncio.Future
    content_lines: list[str] = field(default_factory=list)
    scroll_offset: int = 0


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
    # Round 19: queue of further elicitations that arrived while
    # another was still pending. ``pending_approval`` is always the
    # one the operator currently sees in the modal; once resolved,
    # the head of ``approval_queue`` is promoted so the next
    # operator decision lands on the right request.
    approval_queue: list[ApprovalRequest] = field(default_factory=list)
    # Round 19: ``?`` keystroke toggles the in-TUI help overlay.
    show_help: bool = False
    # Round 19: configurable idle tick (``LITTRACE_TUI_TICK_MS`` env
    # var, default 50 ms). Lets operators on slow terminals dial up
    # the redraw interval to reduce flicker / CPU.
    tick_seconds: float = 0.05
    # Round 19: cached modal body size from the last draw. The
    # scroll keys clamp ``scroll_offset`` against this so the modal
    # never scrolls past its visible window.
    last_modal_body_rows: int = 16


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
    # Round 20: TUI must be a strict Codex surface. We refuse legacy
    # mode here so a silent fallback cannot happen later. A ``legacy``
    # value is overwritten with a one-shot warning; an explicit
    # ``LITTRACE_AGENT_RUNTIME=legacy`` env var is a hard error.
    config = _resolve_tui_config(config)
    session = create_chat_session(config)
    state = TUIState(
        workspace=LiteratureWorkspace(),
        session_id=session.session_id,
        session_root=str(session.root),
        messages=[
            (
                "LitTrace",
                "本地可视化 Agent 已启动 (Codex App Server mode)。MCP 工具调用时审批弹窗会出现在此处。",
            )
        ],
    )
    # Round 19: configurable redraw cadence. ``LITTRACE_TUI_TICK_MS``
    # lets slow-terminal operators dial down the redraw rate without
    # touching the code. Clamped to [10, 1000] ms so a typo doesn't
    # freeze the loop or burn CPU.
    tick_ms_raw = os.environ.get("LITTRACE_TUI_TICK_MS", "50")
    try:
        tick_ms = max(10, min(int(tick_ms_raw), 1000))
    except (TypeError, ValueError):
        tick_ms = 50
    state.tick_seconds = tick_ms / 1000.0
    # Round 20: run the Codex handshake before opening the chat loop so
    # "codex not installed / not logged in" surfaces to the operator
    # before they have typed a single character. ``_codex_startup_preflight``
    # is best-effort: if it raises ``CodexStartupError`` or
    # ``AppServerError``, we render a full-screen error modal and exit.
    try:
        await _codex_startup_preflight(config, session)
    except (CodexStartupError, AppServerError) as exc:
        _render_startup_error(screen, exc)
        return
    # The first user turn spins up its own Codex thread, so the
    # preflight only confirms reachability + auth rather than
    # holding onto a thread id. Surface that fact in the welcome
    # message so the operator knows the probe succeeded.
    state.messages.append(
        ("LitTrace", "Codex App Server 已就绪 (认证通过)。请输入你的研究问题。")
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
            _resolve_approval_future(state.pending_approval, _CANCEL_RESPONSE)


def _resolve_tui_config(config):
    """Round 20: force TUI into ``codex_app_server`` mode.

    Behaviour:
      - ``LITTRACE_AGENT_RUNTIME=legacy`` (explicit operator override)
        is a hard error — the TUI is now the canonical Codex surface
        and we never silently downgrade it.
      - ``config.agent_runtime.mode == LEGACY`` (e.g. inherited from an
        older config) is overwritten with a one-shot warning so an
        upgrade does not lock operators out.
      - ``fallback_to_legacy`` is forced to False for the TUI lifetime
        so the route layer can never silently re-enter ``handle_legacy_chat``.
    """
    explicit_env = os.environ.get("LITTRACE_AGENT_RUNTIME", "").strip().lower()
    if explicit_env == "legacy":
        raise SystemExit(
            "littrace-tui requires the Codex App Server. Unset "
            "LITTRACE_AGENT_RUNTIME or set it to 'codex_app_server'."
        )
    if config.agent_runtime.mode == AgentRuntimeMode.LEGACY:
        config.agent_runtime.mode = AgentRuntimeMode.CODEX_APP_SERVER
    config.agent_runtime.fallback_to_legacy = False
    return config


async def _codex_startup_preflight(config, session):
    """Round 20: prove Codex is reachable before the operator types anything.

    Returns the Codex thread id (str) on success. Raises
    :class:`CodexStartupError` with a remediation list when the binary
    is missing, authentication fails, or the ``initialize`` handshake
    times out. The caller renders the error as a full-screen modal.
    """
    command = list(config.agent_runtime.codex_command or ["codex", "app-server"])
    binary = command[0]
    binary_path = shutil.which(binary)
    if binary_path is None:
        raise CodexStartupError(
            f"Codex CLI 未找到: '{binary}' 不在 PATH。",
            remediation=[
                "安装 Codex CLI: `npm install -g @openai/codex`",
                "或者把 codex 可执行文件加到 PATH",
                "或者在 config.yaml 里把 agent_runtime.codex_command 改成绝对路径",
                "然后重新运行 littrace-tui",
            ],
        )
    timeout_env = os.environ.get("LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS")
    try:
        timeout_seconds = float(timeout_env) if timeout_env else None
    except ValueError:
        timeout_seconds = None
    if timeout_seconds is None:
        timeout_seconds = max(
            float(getattr(config.agent_runtime, "startup_timeout_seconds", 20.0)) * 1.5,
            20.0,
        )
    from littrace.codex_runtime.client import AppServerClient
    from littrace.codex_runtime.runtime import (
        CodexAppServerRuntimeManager,
        shared_runtime_manager,
    )
    factory = getattr(
        config.agent_runtime,
        "client_factory",
        None,
    )
    # Build the runtime-manager key from the same fingerprint the
    # chat service uses (``CodexAppServerChatService._shared_runtime_manager``)
    # so the preflight reuses the cached codex process the first
    # turn would otherwise start cold. The previous implementation
    # called ``shared_runtime_manager(config=config)`` and
    # ``CodexAppServerRuntimeManager(config=config)`` — both
    # signatures are wrong; the manager takes ``(command,
    # client_options=...)`` and the registry takes ``(key, command,
    # client_options=...)``.
    startup_timeout = float(
        getattr(config.agent_runtime, "startup_timeout_seconds", 20.0)
    )
    request_timeout = float(
        getattr(config.agent_runtime, "request_timeout_seconds", 600.0)
    )
    key = (
        tuple(command),
        startup_timeout,
        request_timeout,
        tuple(),
    )
    try:
        manager: CodexAppServerRuntimeManager
        if factory and factory is not AppServerClient:
            manager = CodexAppServerRuntimeManager(
                command,
                client_factory=factory,
            )
        else:
            manager = shared_runtime_manager(
                key, tuple(command), client_options=None,
            )
        thread_id = await manager.use(
            lambda client: _app_server_initialize(client, timeout_seconds)
        )
    except AppServerError as exc:
        raise CodexStartupError(
            f"Codex App Server 握手失败: {exc}",
            remediation=_remediation_for_app_server_error(str(exc)),
        ) from exc
    except Exception as exc:
        raise CodexStartupError(
            f"Codex App Server 启动失败: {exc.__class__.__name__}: {exc}",
            remediation=_remediation_for_app_server_error(str(exc)),
        ) from exc
    return thread_id


def _remediation_for_app_server_error(detail: str) -> list[str]:
    detail_lower = detail.lower()
    if "not found" in detail_lower or "no such file" in detail_lower:
        return [
            "确认 codex CLI 已安装: `which codex`",
            "安装: `npm install -g @openai/codex`",
            "在 config.yaml 中把 agent_runtime.codex_command 指向绝对路径",
        ]
    if "timed out" in detail_lower or "timeout" in detail_lower:
        return [
            "Windows 冷启动可能较慢，调高超时: `set LITTRACE_CODEX_STARTUP_TIMEOUT_SECONDS=60`",
            "先单独运行 `codex app-server` 确认可以启动",
            "在 config.yaml 里把 agent_runtime.startup_timeout_seconds 调大",
        ]
    if "auth" in detail_lower or "login" in detail_lower or "unauthorized" in detail_lower:
        return [
            "运行 `littrace setup-browser --launch` 完成 Codex 登录",
            "或者手动登录 codex 后重试",
        ]
    return [
        "运行 `littrace doctor` 检查 Codex 健康",
        "运行 `littrace setup-browser` 完成认证",
        "查看 Codex 日志: `littrace-tui` 启动时 stderr 会有 Codex 输出",
    ]


async def _app_server_initialize(client, timeout_seconds: float) -> str:
    """Confirm the shared client is reachable and authenticated.

    The client was already initialized by
    :meth:`CodexAppServerRuntimeManager._ensure_client` (which sends
    ``initialize`` + ``clientInfo`` + ``initialized`` and starts the
    codex subprocess), so this helper just needs a benign RPC that
    proves the operator's auth is good -- ``read_account`` is exactly
    that: it returns ``{"account": null, "requiresOpenaiAuth": bool}``
    without committing any state. We do NOT start a thread here;
    the first user turn does that, and starting one in preflight
    would leak an unused Codex thread on every TUI launch.

    The previous implementation called
    ``client.request("initialize", {})`` on the already-initialized
    client, which codex 0.150+ rejects with
    ``Invalid request: missing field clientInfo`` because the second
    initialize must carry the same clientInfo payload as the first.
    """
    del timeout_seconds  # read_account uses its own client-side timeout
    account = await client.read_account(refresh_token=False)
    # Match the canonical check in
    # CodexAppServerChatService._require_authentication: only raise
    # if BOTH requiresOpenaiAuth is True AND the loaded account is
    # null. codex 0.150 reports requiresOpenaiAuth=True even when
    # account.email is populated (the flag is a hint that the user
    # has not finished the in-app "trust this device" step), so the
    # older "raise on any requiresOpenaiAuth=True" check would lock
    # every ChatGPT-login operator out of the TUI.
    if (
        bool(account.get("requiresOpenaiAuth"))
        and account.get("account") is None
    ):
        from littrace.codex_runtime.errors import AppServerError
        # Surface as AppServerError so the outer except arm wraps it
        # into CodexStartupError with the auth remediation steps.
        raise AppServerError(
            "Codex 未登录: 在 Codex 终端里运行 `codex login` 完成认证"
        )
    # Return a stable probe marker so callers can assert success
    # without depending on a thread id that no longer makes sense
    # to start in preflight.
    return "ready"


def _render_startup_error(screen, exc) -> None:
    """Round 20: full-screen modal showing the startup error and remediation.

    Mimics the approval-modal layout so the operator gets the same
    visual language for both codex startup failures and mid-turn
    approval requests. ``Any key returns`` is the only escape — there
    is no recovery path inside the TUI.
    """
    message = str(exc)
    remediation = getattr(exc, "remediation", None) or []
    height, width = screen.getmaxyx()
    lines: list[str] = []
    lines.append("LitTrace TUI 无法连接到 Codex App Server")
    lines.append("")
    lines.append(f"原因: {message}")
    if remediation:
        lines.append("")
        lines.append("修复步骤:")
        for step in remediation:
            for wrapped in textwrap.wrap(step, width=width - 6) or [step]:
                lines.append(f"  - {wrapped}")
    lines.append("")
    lines.append("按任意键返回。")
    start_row = max(0, (height - len(lines)) // 2)
    box_left = 1
    box_right = width - 2
    try:
        for offset, line in enumerate(lines):
            row = start_row + offset
            if row >= height - 1:
                break
            truncated = line[: max(0, box_right - box_left - 1)]
            screen.addstr(row, box_left, truncated)
    except curses.error:
        # Terminal too small to render; the operator already sees the
        # message on stderr from the AppServerError log.
        pass
    screen.refresh()
    screen.nodelay(False)
    screen.getch()


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
            await asyncio.sleep(state.tick_seconds)
            continue
        # Round 19: ``?`` toggles the help overlay regardless of
        # input lock / approval state — operators need it the most
        # when the modal is up and they cannot remember the keys.
        if key == ord("?") and state.pending_approval is None:
            state.show_help = not state.show_help
            continue
        if state.show_help:
            # Any non-``?`` key while the help is up dismisses it.
            if key != ord("?"):
                state.show_help = False
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
    # Round 19: scroll keys act on the modal's content_lines BEFORE
    # the decision keys. They never resolve the future; the modal
    # stays open until Y/N/Esc is pressed.
    if key in (curses.KEY_UP, ord("k")):
        approval.scroll_offset = max(0, approval.scroll_offset - 1)
        return
    if key in (curses.KEY_DOWN, ord("j")):
        max_offset = max(
            len(approval.content_lines) - _approval_modal_body_rows(state), 0
        )
        approval.scroll_offset = min(max_offset, approval.scroll_offset + 1)
        return
    if key == curses.KEY_PPAGE:
        approval.scroll_offset = max(0, approval.scroll_offset - 10)
        return
    if key == curses.KEY_NPAGE:
        max_offset = max(
            len(approval.content_lines) - _approval_modal_body_rows(state), 0
        )
        approval.scroll_offset = min(max_offset, approval.scroll_offset + 10)
        return
    if key in (curses.KEY_HOME, ord("g")):
        approval.scroll_offset = 0
        return
    if key in (curses.KEY_END, ord("G")):
        max_offset = max(
            len(approval.content_lines) - _approval_modal_body_rows(state), 0
        )
        approval.scroll_offset = max_offset
        return
    if key in (ord("y"), ord("Y")):
        _resolve_approval_future(approval, _ACCEPT_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: accepted — running tool"
        _advance_approval_queue(state)
        return
    if key in (ord("n"), ord("N")):
        _resolve_approval_future(approval, _DECLINE_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: declined"
        _advance_approval_queue(state)
        return
    if key == 27:  # Esc
        _resolve_approval_future(approval, _CANCEL_RESPONSE)
        state.pending_approval = None
        state.status = "Approval: cancelled"
        _advance_approval_queue(state)
        return


def _advance_approval_queue(state: TUIState) -> None:
    """Pop the head of ``approval_queue`` onto ``pending_approval``.

    Called immediately after the operator resolves the current
    approval. If the queue is empty (the common case), this is a
    no-op and ``pending_approval`` stays ``None`` so the next
    E2E handler call parks directly. Round 19.
    """
    if state.approval_queue:
        state.pending_approval = state.approval_queue.pop(0)


def _approval_modal_body_rows(state: TUIState) -> int:
    """How many argument rows fit inside the current modal box.

    Reads ``state.last_modal_body_rows`` which is refreshed on every
    draw. Defaults to 16 if the modal has not yet been drawn.
    """
    return max(1, getattr(state, "last_modal_body_rows", 16))


def _resolve_approval_future(approval: ApprovalRequest, payload: dict[str, Any]) -> None:
    """Set the approval future's result, with cross-loop safety.

    The handler installed on the AppServerClient creates its Future
    on the AppServerClient thread's event loop. The TUI's main loop
    runs on a different loop; calling ``future.set_result`` from
    here schedules a wakeup via ``loop.call_soon`` on the future's
    owning loop, but ``call_soon`` is NOT thread-safe and silently
    drops the callback. ``call_soon_threadsafe`` is the documented
    cross-loop equivalent.
    """
    future = approval.future
    fut_loop = getattr(future, "_loop", None)
    if fut_loop is not None and fut_loop is not asyncio.get_running_loop():
        fut_loop.call_soon_threadsafe(future.set_result, payload)
    else:
        future.set_result(payload)


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
    except AppServerError as exc:
        # Round 20: surface Codex App Server failures with a structured
        # remediation bubble so the operator sees concrete steps instead
        # of a raw RuntimeError. We never silently fall back to legacy
        # because ``fallback_to_legacy`` is forced off by
        # ``_resolve_tui_config``.
        state.messages.append(
            (
                "Error",
                "Codex App Server 不可用: "
                f"{exc}\n\n修复步骤:\n"
                + "\n".join(f"- {step}" for step in _remediation_for_app_server_error(str(exc))),
            )
        )
        state.status = "Error: Codex App Server unavailable"
        return
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


_TOOL_NAME_RE = __import__("re").compile(r'tool "([^"]+)"')


def _tool_from_message(message: object) -> str | None:
    """Round 20: module-level helper so the Window's elicitation
    modal can use the same parser as the TUI."""
    if not isinstance(message, str):
        return None
    match = _TOOL_NAME_RE.search(message)
    return match.group(1) if match else None


def _parse_elicitation_payload(
    params: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Round 20: parse ``(server, tool, arguments)`` from the
    ``mcpServer/elicitation/request`` wire payload.

    Accepts both the codex 0.150+ shape (``serverName`` + tool name
    encoded inside ``message`` + ``_meta.tool_params``) and the
    older flat shape (``server`` / ``tool`` / ``arguments``). The
    handler nests ``_extract_params`` delegate so the TUI keeps its
    closure shape, and the Window imports the helper from here.
    """
    server = str(
        params.get("server")
        or params.get("serverName")
        or "littrace"
    )
    tool = params.get("tool") or params.get("name")
    if not tool:
        tool = _tool_from_message(params.get("message"))
    if not tool:
        tool = "?"
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    arguments = (
        meta.get("tool_params")
        if isinstance(meta.get("tool_params"), dict)
        else None
    )
    if arguments is None:
        arguments = params.get("arguments") or params.get("args") or {}
    return server, str(tool), dict(arguments)


def _make_elicitation_handler(state: TUIState):
    """Return an async callable for the AppServerClient to invoke
    on a ``mcpServer/elicitation/request`` notification.

    Round 18 wire-format note: codex-cli 0.150.1 does NOT send the
    intuitive ``{server, tool, arguments}`` triple. The actual
    payload is::

        {
          "threadId": "...",
          "turnId": "...",
          "serverName": "littrace",
          "mode": "form",
          "_meta": {
            "codex_approval_kind": "mcp_tool_call",
            "tool_description": "...",
            "tool_params": {"topic": "...", ...},
            "tool_params_display": [...],
          },
          "message": 'Allow the littrace MCP server to run tool "search_papers"?',
          "requestedSchema": {"type": "object", "properties": {}}
        }

    The server name lives at ``serverName``; the tool name is
    encoded inside the ``message`` string (``tool "X"``); the
    arguments live under ``_meta.tool_params``. Older codex
    snapshots that predate 0.150 may send the flat shape; we
    fall back to that for back-compat.

    Round 18 cross-loop note: the AppServerClient's reader loop
    runs on a dedicated thread + asyncio loop. The Future this
    handler returns is bound to that loop; the TUI main loop
    (which resolves the Future on Y/N/Esc) runs on a different
    loop. ``Future.set_result`` from a foreign thread schedules
    wakeup via ``loop.call_soon`` (not threadsafe) and silently
    drops the callback. The TUI main loop therefore must use
    ``fut_loop.call_soon_threadsafe(future.set_result, payload)``
    when it sets the result.
    """

    import re as _re

    _TOOL_NAME_RE = _re.compile(r'tool "([^"]+)"')

    def _tool_from_message(message: object) -> str | None:
        if not isinstance(message, str):
            return None
        match = _TOOL_NAME_RE.search(message)
        return match.group(1) if match else None

    # Delegate to the module-level helper so the same parser is used
    # by both the TUI handler and the Window handler (Round 20).
    def _extract_params(params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        return _parse_elicitation_payload(params)

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        server, tool, arguments = _extract_params(params)
        approval = ApprovalRequest(
            request_id=params.get("serverRequestId")
            or params.get("requestId")
            or "",
            server=server,
            tool=tool,
            arguments=arguments,
            future=future,
        )
        # Pre-render the argument block once so the modal can page
        # through it. Doing this here keeps the draw loop's cost flat
        # regardless of how many fields the tool exposes.
        try:
            args_text = json.dumps(arguments, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = repr(arguments)
        approval.content_lines = [
            line for line in args_text.splitlines()
        ] or [""]
        # Round 19 — queue behaviour. If the operator has not yet
        # resolved a previous approval, this one waits in
        # ``approval_queue`` instead of overwriting the current
        # modal. The TUI main loop pops the head into
        # ``pending_approval`` after each Y/N/Esc.
        if state.pending_approval is not None:
            state.approval_queue.append(approval)
        else:
            state.pending_approval = approval
        try:
            return await future
        finally:
            # Only clear if we were the active approval. If we were
            # queued behind another, the queue already owns our slot
            # and clearing here would let a sibling handler promote
            # itself prematurely.
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
    header = f" LitTrace TUI | session {state.session_id} | /ocr-choice /hide-context /show-context /quit | ?=help "
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
    elif state.show_help:
        _draw_help_overlay(screen, state, height, width)
    else:
        prompt = "littrace > "
        _add_clipped(screen, height - input_height, 0, prompt + state.input_text, width)
        cursor_x = min(len(prompt) + len(state.input_text), width - 1)
        screen.move(height - input_height, cursor_x)
    screen.refresh()


def _draw_approval_modal(screen, state: TUIState, height: int, width: int) -> None:
    approval = state.pending_approval
    assert approval is not None
    # Pre-header lines (always visible at top of the box).
    header_lines = [
        "MCP Tool Approval Required",
        "",
        f"Server: {approval.server}",
        f"Tool:   {approval.tool}",
        "",
        "Arguments:",
    ]
    # Footer with the decision keys + scroll hint.
    footer_lines = [
        "",
        "[Y] Accept   [N] Decline   [Esc] Cancel",
        "[↑/↓ PgUp/PgDn g/G Home/End] scroll",
    ]
    # Round 19: scrollable body. Re-wrap each pre-rendered line to
    # the modal width so a single huge line (a long URL, an array
    # literal) does not blow past ``box_width - 4``.
    body_wrap_width = max(width // 2 - 6, 20)
    body_lines: list[str] = []
    for raw in approval.content_lines:
        wrapped = textwrap.wrap(raw, width=body_wrap_width) or [""]
        body_lines.extend(wrapped)
    # Reserve rows for header + footer inside the modal.
    # box_height is fixed by terminal size; body rows = total - hdr - ftr.
    inner_height = height - 4  # same cap as before
    body_rows = max(1, inner_height - len(header_lines) - len(footer_lines))
    state.last_modal_body_rows = body_rows
    # Clamp scroll_offset against the body now that we know its size.
    max_offset = max(len(body_lines) - body_rows, 0)
    if approval.scroll_offset > max_offset:
        approval.scroll_offset = max_offset
    visible_body = body_lines[
        approval.scroll_offset : approval.scroll_offset + body_rows
    ]
    # Pad the visible body so the footer stays anchored at the bottom.
    while len(visible_body) < body_rows:
        visible_body.append("")

    lines = header_lines + visible_body + footer_lines

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


def _draw_help_overlay(screen, state: TUIState, height: int, width: int) -> None:
    """Render the ``?`` help overlay — one page of keybindings.

    Round 19. Rendered as a centered curses box on top of the input
    row. Any non-``?`` keypress dismisses the overlay (handled in
    the event loop). Keeps the chat / context panes visible
    underneath so the operator can still see what they were doing
    before they hit ``?``.
    """
    lines = [
        "LitTrace TUI Help",
        "",
        "Chat input:",
        "  Enter ............ submit message",
        "  Backspace ........ delete last char",
        "  /quit /exit ...... leave TUI",
        "  /ocr-choice ...... show OCR parsing suggestions",
        "  /hide-context .... hide right-hand context pane",
        "  /show-context .... show right-hand context pane",
        "",
        "Approval modal (MCP tool call):",
        "  Y ................ accept and run tool",
        "  N ................ decline (the tool is not called)",
        "  Esc .............. cancel (turn aborts)",
        "  ↑ / k ............ scroll up",
        "  ↓ / j ............ scroll down",
        "  PgUp / PgDn ...... scroll page",
        "  g / Home ......... jump to top",
        "  G / End .......... jump to bottom",
        "",
        "Other:",
        "  ? ................ toggle this help",
        "  Ctrl-C ........... abort the current turn",
        "",
        "Env knobs:",
        "  LITTRACE_TUI_TICK_MS .. redraw interval (default 50)",
    ]
    box_width = min(max((len(line) for line in lines), default=20) + 4, width - 4)
    box_height = min(len(lines) + 2, height - 4)
    box_y = max(1, (height - box_height) // 2)
    box_x = max(0, (width - box_width) // 2)
    # Border.
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