"""GUI-agnostic controller that owns LitTrace session state and the chat /
workspace / RAG side effects. Both the Tk shell (`littrace.window`) and
the Qt WebEngine shell (`littrace.window_qt`) bind to the same
``ShellController`` so business logic lives in exactly one place.

The controller never imports Tk or Qt. It only emits Python events on a
plain ``ShellEventBus`` and runs asyncio work in a worker thread, the
same pattern the Tk shell has used since the codex App Server integration
landed. Concrete shells translate these events into widget updates.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from littrace.agent_runtime import handle_agent_chat
from littrace.config import LitTraceConfig
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import ChatSession, create_chat_session, load_workspace

try:
    # Hold a long-lived CodexAppServerChatService so the Codex CLI subprocess
    # and JSON-RPC handshake are reused across turns. Without this every chat
    # turn pays the ~7s spawn+initialize tax; with it a follow-up turn is
    # typically <2s of pure LLM round-trip latency.
    from littrace.codex_runtime.service import CodexAppServerChatService

    _HAS_CODEX_SERVICE = True
except Exception:  # pragma: no cover - defensive
    _HAS_CODEX_SERVICE = False


@dataclass(frozen=True)
class ShellEvent:
    """Immutable event payload broadcast on the ``ShellEventBus``."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[ShellEvent], None]


class ShellEventBus:
    """Minimal synchronous pub/sub used by both Tk and Qt shells."""

    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    def emit(self, event: ShellEvent) -> None:
        # Iterate over a copy so handlers may safely unsubscribe themselves
        # without invalidating the iterator (Qt shells sometimes rebuild
        # widgets in response to events).
        for handler in list(self._subscribers):
            try:
                handler(event)
            except Exception:  # pragma: no cover - defensive
                # Shells are responsible for their own error logging. We
                # intentionally do not raise from event delivery — a buggy
                # view must not stop the controller from running the next
                # user turn.
                pass


class ShellController:
    """Owns session, workspace, and the chat pipeline.

    Lifecycle is identical between Tk and Qt shells:

    - ``ShellController(config)`` builds the controller and loads the
      default chat session.
    - ``bind(event_bus)`` wires the bus; shells subscribe to events they
      care about.
    - ``submit_user_message(text)`` schedules an async chat turn on the
      controller's private worker thread and emits
      ``message_appended`` / ``status_changed`` events.
    - ``refresh_context()`` etc. emit ``workspace_refreshed`` /
      ``rag_panel_refreshed`` / ``ocr_buttons_refreshed``.
    """

    EVENT_MESSAGE_APPENDED = "message_appended"
    EVENT_STATUS_CHANGED = "status_changed"
    EVENT_THINKING = "thinking"
    EVENT_WORKSPACE_REFRESHED = "workspace_refreshed"
    EVENT_RAG_PANEL_REFRESHED = "rag_panel_refreshed"
    EVENT_OCR_BUTTONS_REFRESHED = "ocr_buttons_refreshed"
    EVENT_SESSION_HISTORY_REFRESHED = "session_history_refreshed"
    EVENT_ERROR = "error"

    def __init__(self, config: LitTraceConfig) -> None:
        self._config = config
        self._session: ChatSession = create_chat_session(config)
        self._workspace: LiteratureWorkspace = LiteratureWorkspace()
        self.bus = ShellEventBus()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # Reuse a single CodexAppServerChatService across turns so we
        # don't pay the codex-app-server spawn + JSON-RPC initialize cost
        # on every chat turn. Built lazily on the worker thread on first
        # use so the controller's main-thread constructor stays light.
        self._service: Any = None
        self._service_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the controller's worker thread. Idempotent.

        Also kicks off a background ``_prime_service()`` coroutine that
        builds the CodexAppServerChatService and runs an empty thread so
        the first ``codex app-server -c initialize`` handshake is paid
        during window startup, not during the user's first message.
        """
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, name="littrace-shell-loop", daemon=True
        )
        self._loop_thread.start()
        # Wait until the loop is actually accepting submissions so the
        # very first ``submit_user_message`` does not race with startup.
        while self._loop is None:
            threading.Event().wait(0.01)
        # Warm the CodexAppServerChatService off the event loop so the
        # spawn+initialize handshake happens while the window is still
        # being painted, not after the user has already typed their
        # first message. Failure here is non-fatal: the lazy
        # ``_run_chat_turn`` path still works if priming fails (e.g. the
        # codex binary is missing on $PATH).
        try:
            asyncio.run_coroutine_threadsafe(self._prime_service(), self._loop)
        except Exception:  # pragma: no cover - defensive
            pass

    async def _prime_service(self) -> None:
        """Pre-build ``CodexAppServerChatService`` and force the runtime
        manager to spin up its CodexAppServer subprocess + JSON-RPC
        handshake. Called from ``start()`` so first-turn latency is just
        the LLM round-trip instead of spawn + initialize + turn.
        """
        if not (
            _HAS_CODEX_SERVICE
            and self._config.agent_runtime.mode == "codex_app_server"
        ):
            return
        with self._service_lock:
            if self._service is None:
                self._service = CodexAppServerChatService(self._config)
        # Delegate to ``service.warmup()`` which builds the runtime
        # manager and forces it to open the client. Best-effort: failure
        # here leaves ``self._service`` set so the lazy chat path still
        # works, just without the spawn cost amortised into startup.
        try:
            await self._service.warmup()
        except Exception:
            pass

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    # ------------------------------------------------------------------
    # Accessors used by shells
    # ------------------------------------------------------------------

    @property
    def session(self) -> ChatSession:
        return self._session

    @property
    def workspace(self) -> LiteratureWorkspace:
        return self._workspace

    @property
    def config(self) -> LitTraceConfig:
        return self._config

    def bind(self, bus: ShellEventBus) -> None:
        """Replace the event bus (used in tests)."""
        self.bus = bus

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> None:
        self.bus.emit(ShellEvent(kind=kind, payload=payload))

    # ------------------------------------------------------------------
    # Chat pipeline
    # ------------------------------------------------------------------

    def submit_user_message(self, text: str) -> None:
        if not text.strip():
            return
        self._emit(self.EVENT_STATUS_CHANGED, text="处理中…")
        self._emit(self.EVENT_THINKING, active=True, label="正在理解任务…")
        self._emit(self.EVENT_MESSAGE_APPENDED, role="user", text=text)
        if self._loop is None:
            self._emit(self.EVENT_ERROR, message="controller event loop not ready")
            return
        asyncio.run_coroutine_threadsafe(
            self._run_chat_turn(text), self._loop
        )

    async def _run_chat_turn(self, text: str) -> None:
        request = ChatRequest(
            session_id=self._session.session_id,
            message=text,
            current_workspace=self._workspace,
            trace_id=f"{self._session.session_id}",
        )
        # Use the long-lived CodexAppServerChatService when in codex_app_server
        # mode so the App Server subprocess + JSON-RPC handshake are reused
        # across turns. Falls back to the legacy top-level dispatcher for
        # other modes or if the service import failed at module load.
        service = None
        if (
            _HAS_CODEX_SERVICE
            and self._config.agent_runtime.mode == "codex_app_server"
        ):
            with self._service_lock:
                if self._service is None:
                    self._service = CodexAppServerChatService(self._config)
                service = self._service

        async def _thinking(label: str) -> None:
            self._emit(self.EVENT_THINKING, active=True, label=label)

        # Surface the early pipeline steps to the UI so the user sees the
        # model is moving instead of staring at a frozen chat input.
        await _thinking("正在解析任务意图…")

        try:
            self._emit(self.EVENT_THINKING, active=True, label="调用 Codex / 模型…")
            if service is not None:
                response, workspace = await service.chat(
                    request, self._workspace, self._session
                )
            else:
                response, workspace = await handle_agent_chat(
                    request,
                    self._workspace,
                    self._config,
                    session=self._session,
                )
        except Exception as exc:
            self._emit(self.EVENT_ERROR, message=f"{type(exc).__name__}: {exc}")
            self._emit(self.EVENT_THINKING, active=False)
            self._emit(self.EVENT_STATUS_CHANGED, text="错误")
            return
        with self._lock:
            self._workspace = workspace
        self._emit(
            self.EVENT_MESSAGE_APPENDED,
            role="assistant",
            text=response.reply,
            action=response.action,
            warnings=response.warnings,
        )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        self._emit(self.EVENT_THINKING, active=False)
        self._emit(self.EVENT_STATUS_CHANGED, text="就绪")

    # ------------------------------------------------------------------
    # Refresh hooks (mirrors Tk shell's refresh_* methods)
    # ------------------------------------------------------------------

    def refresh_context(self) -> None:
        self._emit(self.EVENT_WORKSPACE_REFRESHED)

    def refresh_ocr_buttons(self) -> None:
        self._emit(self.EVENT_OCR_BUTTONS_REFRESHED)

    def refresh_session_history(self) -> None:
        self._emit(self.EVENT_SESSION_HISTORY_REFRESHED)

    def refresh_rag_panel(self) -> None:
        self._emit(self.EVENT_RAG_PANEL_REFRESHED)

    def list_active_papers(self) -> Iterable[Any]:
        """Return the active paper metadata list, or [] when missing."""
        ids = list(self._workspace.context.active_papers)
        papers: list[Any] = []
        for pid in ids:
            paper = self._workspace.papers.get(pid)
            if paper is not None:
                papers.append(paper)
        return papers