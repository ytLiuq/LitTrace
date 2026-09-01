"""GUI-agnostic controller that owns LitTrace session state and the chat /
workspace / RAG side effects. The Qt WebEngine shell
(``littrace.window_qt``) binds to this ``ShellController`` so business
logic lives in exactly one place. The legacy Tk shell
(``littrace.window``) was removed in Round 17 — the project now ships
a single ``littrace-qt`` entry point.

The controller never imports Tk or Qt. It only emits Python events on a
plain ``ShellEventBus`` and runs asyncio work in a worker thread, the
same pattern the codex App Server integration landed. Concrete shells
translate these events into widget updates.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
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
    # Round 17: warmup lifecycle. ``EVENT_WARMUP_STARTED`` fires when
    # ``_prime_service`` kicks off the CodexAppServerChatService spawn
    # so the shell can show "正在准备 Codex…" instead of a frozen
    # "就绪". ``EVENT_WARMUP_DONE`` fires once the spawn + JSON-RPC
    # ``initialize`` handshake completes (or the warmup is skipped
    # because the runtime mode / service import vetoed it). Both
    # events carry ``phase`` ("spawning" / "initializing" / "ready"
    # / "failed") and an optional ``detail`` for the status strip.
    EVENT_WARMUP_STARTED = "warmup_started"
    EVENT_WARMUP_DONE = "warmup_done"
    # Round 17: streaming lifecycle. ``EVENT_ASSISTANT_STREAM_OPEN``
    # fires once at the start of a turn so the shell can open a
    # streaming bubble (cursor pinned to its end). ``EVENT_ASSISTANT_DELTA``
    # fires for every codex ``item/agentMessage/delta`` frame, with
    # ``delta`` carrying the raw text to append to the bubble. Shells
    # must be tolerant of ``delta`` arriving between turns (the
    # server occasionally flushes a tail frame after ``completed``)
    # by either dropping it or appending to a now-frozen bubble —
    # either way the final ``EVENT_MESSAGE_APPENDED`` carries the
    # authoritative full text, so a stray delta never breaks the
    # chat scrollback.
    EVENT_ASSISTANT_STREAM_OPEN = "assistant_stream_open"
    EVENT_ASSISTANT_DELTA = "assistant_delta"
    # Round 17: OAuth lifecycle. ``EVENT_AUTH_REQUIRED`` fires when
    # the controller detects that the Codex App Server's auth.json
    # is missing / expired / revoked. The shell reacts by surfacing
    # a one-shot dialog with the ``codex login --device-auth``
    # instructions and a "Re-check" button the user clicks after
    # running the command in their terminal. ``EVENT_AUTH_OK``
    # fires when a subsequent re-check sees the auth restored, so
    # the shell can drop the warning banner and resume normal chat.
    EVENT_AUTH_REQUIRED = "auth_required"
    EVENT_AUTH_OK = "auth_ok"
    # Round 17: progress heartbeat. Fires every
    # ``THINKING_PROGRESS_INTERVAL`` seconds while a turn is in
    # flight, carrying the elapsed time so the shell can render
    # "已思考 3.2s …" instead of a frozen "思考中…". Stops when
    # ``EVENT_THINKING`` fires with ``active=False`` or when the
    # controller's chat turn coroutine exits.
    EVENT_THINKING_PROGRESS = "thinking_progress"
    THINKING_PROGRESS_INTERVAL = 1.5
    # Round 19: emit when a local slash command (e.g. /papers, /workflow)
    # produces a result. The GUI chat panel renders these as a
    # system-style bubble so the user can see command output without
    # leaving the chat surface.
    EVENT_SLASH_RESULT = "slash_result"

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
        # Set when ``_prime_service`` finishes (or skips because the
        # mode / ``_HAS_CODEX_SERVICE`` flag vetoed the warm-up).
        # ``_run_chat_turn`` waits up to 30 s for this before falling
        # through, so a user turn that races the background warm-up
        # never lands on a not-yet-ready service and silently
        # downgrades to the legacy ``handle_agent_chat`` path.
        self._service_ready = threading.Event()
        # Round 17: throttle auth re-checks so the shell's "Re-check"
        # button doesn't hammer the auth.json file on every click.
        # The last-check timestamp is recorded here; the controller
        # only re-emits ``EVENT_AUTH_REQUIRED`` if the previous check
        # is older than ``AUTH_RECHECK_MIN_INTERVAL`` seconds AND the
        # state is still bad. ``EVENT_AUTH_OK`` always fires when
        # the check passes, so the shell can drop the warning
        # banner as soon as the user fixes the auth.
        self._last_auth_check_ts: float = 0.0
        self._auth_required_emitted: bool = False

    AUTH_RECHECK_MIN_INTERVAL = 5.0

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
        # Round 17: fire-and-forget auth probe so the shell can
        # prompt the user to ``codex login --device-auth`` before
        # the first chat turn lands a 401. Cheap (parses a single
        # auth.json file) so running it on the main thread is fine.
        try:
            self.check_codex_auth()
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
            self._service_ready.set()
            return
        with self._service_lock:
            if self._service is None:
                self._service = CodexAppServerChatService(self._config)
        # Announce the warmup so the shell can flip the status bar
        # from "就绪" to "正在准备 Codex…". Without this the user
        # sees "就绪" → 5-10 s of frozen silence → first token,
        # which looks like the app is hung. With the event the user
        # sees "正在准备 Codex…(spawning)" → "...(initializing)" →
        # "就绪" → first user turn feels instant.
        self._emit(self.EVENT_WARMUP_STARTED, phase="spawning")
        # ``service.warmup()`` builds the runtime manager and forces it
        # to open the codex subprocess + run the JSON-RPC ``initialize``
        # handshake. A previous version awaited the warmup here, but
        # the runtime_manager's lock serialised the warmup's no-op
        # turn against the user's first real turn, so the chat
        # thread waited the full ``service_ready.wait`` timeout (30 s)
        # for nothing. The warmup now runs in the background; the
        # first user turn pays the spawn cost (5-10 s) once and then
        # every subsequent turn reuses the warm client.
        asyncio.create_task(self._run_warmup())
        self._service_ready.set()

    async def _run_warmup(self) -> None:
        # Step through the warmup with progress events so the shell
        # can show "spawning → initializing → ready" instead of one
        # long indeterminate wait. The two ``WARMUP_STARTED`` /
        # ``WARMUP_DONE`` events frame the whole thing; the second
        # event also fires on failure so the shell always ends up
        # back in "就绪" (or a non-blocking warning) and never
        # freezes on "正在准备 Codex…".
        try:
            self._emit(self.EVENT_WARMUP_STARTED, phase="initializing")
            ok = await self._service.warmup()
        except Exception as exc:
            self._emit(
                self.EVENT_WARMUP_DONE,
                phase="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return
        self._emit(
            self.EVENT_WARMUP_DONE,
            phase="ready" if ok else "failed",
            detail="" if ok else "codex app-server 启动失败；首轮对话会自动重试",
        )

    async def _emit_progress_loop(self, started_at: float) -> None:
        """Emit ``EVENT_THINKING_PROGRESS`` heartbeats while a turn
        is in flight.

        Round 17: a chat turn can stall in the codex App Server for
        30+ seconds (long context, slow model, network jitter), and
        the static "思考中…" label looks frozen. The heartbeat
        keeps the shell's thinking strip alive with elapsed-time
        updates without spamming — ``THINKING_PROGRESS_INTERVAL``
        caps the rate at ~0.7 Hz.

        The loop is cancelled when ``_run_chat_turn``'s ``finally``
        block runs, which always happens whether the turn
        completes, errors, or is interrupted, so the heartbeat
        never outlives the turn.
        """
        try:
            while True:
                await asyncio.sleep(self.THINKING_PROGRESS_INTERVAL)
                elapsed = time.monotonic() - started_at
                self._emit(
                    self.EVENT_THINKING_PROGRESS,
                    elapsed_seconds=round(elapsed, 1),
                )
        except asyncio.CancelledError:
            return

    async def _prime_real_turn(self) -> None:
        """Run a one-token ``.`` turn so the OpenAI API path is warm
        before the user sends their first real message. ``silent=True``
        suppresses the ``message_appended`` event so the dummy "**.**"
        answer does not pollute the user's chat scrollback.
        """
        # Call ``service.chat`` directly — going through
        # ``_run_chat_turn`` would deadlock because that path waits
        # on ``_service_ready`` which is the event we are *about* to
        # set at the bottom of ``_prime_service``. The dummy reply is
        # discarded.
        try:
            request = ChatRequest(
                session_id=self._session.session_id,
                message=".",
                current_workspace=self._workspace,
                trace_id=f"{self._session.session_id}-warmup",
            )
            await self._service.chat(
                request, self._workspace, self._session,
            )
        except Exception:
            pass

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Auth (Round 17)
    # ------------------------------------------------------------------

    def check_codex_auth(self, *, force: bool = False) -> None:
        """Inspect the CodexAppServer auth.json and emit
        ``EVENT_AUTH_REQUIRED`` / ``EVENT_AUTH_OK`` accordingly.

        Cheap (parses a single small JSON file) and idempotent, so
        it's safe to call from any number of triggers (window
        startup, "Re-check" button, post-warmup fail).

        ``force=True`` skips the recheck throttle so a click on
        the dialog's "立即重试" button always re-reads the file.
        The user might have just run ``codex login --device-auth``
        in another terminal — the throttle exists to make sure we
        don't re-poll every 200 ms during a tight UI loop.
        """
        now = time.monotonic()
        if (
            not force
            and (now - self._last_auth_check_ts) < self.AUTH_RECHECK_MIN_INTERVAL
        ):
            return
        self._last_auth_check_ts = now
        status = self._read_codex_auth_status()
        if status.ok:
            self._auth_required_emitted = False
            self._emit(
                self.EVENT_AUTH_OK,
                detail=status.detail,
            )
            return
        # Only emit ``EVENT_AUTH_REQUIRED`` once per "still bad"
        # streak so the shell doesn't keep re-opening the dialog
        # on every check. The user has to either click "Re-check"
        # (force=True) or actually fix the auth before we emit
        # again.
        if self._auth_required_emitted and not force:
            return
        self._auth_required_emitted = True
        self._emit(
            self.EVENT_AUTH_REQUIRED,
            reason=status.reason,
            detail=status.detail,
            codex_home=str(self._codex_auth_path()),
        )

    def _codex_auth_path(self) -> "Path":
        """Return the FIRST existing ``auth.json`` path that the
        App Server might consult, falling back to the configured
        mode's primary path so the caller can detect "missing".

        Round 17: codex 0.149.0-alpha.4.3's app-server reads from
        the host's ``~/.codex/auth.json`` regardless of the
        ``codex_home_mode`` setting, even when ``codex login`` ran
        against an isolated LitTrace home. We therefore probe both
        locations and return whichever file actually exists, so a
        user with ``codex_home_mode: shared`` who logged in inside
        ``data/codex-home`` (the documented workaround) still gets
        detected as authenticated.

        Falls back to the SHARED primary path when neither file
        exists — the dialog will surface a "missing" status, which
        is the right hint when the user genuinely hasn't logged
        in.
        """
        from pathlib import Path

        runtime = self._config.agent_runtime
        shared_path = Path.home() / ".codex" / "auth.json"
        isolated_path = runtime.codex_home.expanduser().resolve() / "auth.json"
        for candidate in (shared_path, isolated_path):
            if candidate.exists():
                return candidate
        # No auth.json on disk — return whichever path the
        # controller's configured mode expects so the caller can
        # show the right hint ("run codex login" without spelling
        # out a path).
        return (
            shared_path
            if runtime.codex_home_mode.value == "shared"
            else isolated_path
        )

    @dataclass(frozen=True)
    class _AuthStatus:
        ok: bool
        reason: str = ""
        detail: str = ""

    def _read_codex_auth_status(self) -> "_AuthStatus":
        """Inspect the CodexAppServer auth.json file and decide
        whether the user is currently authenticated.

        Round 17: the file's ``tokens.id_token`` is a JWT whose
        ``exp`` claim carries the expiry timestamp. We decode the
        payload (no signature verification — this is a UI signal,
        not a security boundary; the codex app-server validates
        on the actual API call) and compare to ``time.time()``.
        The "soon to expire" window is 1 hour so the user has a
        heads-up before the first 401 lands.

        Three failure modes are reported distinctly so the
        dialog can suggest the right remediation:
          * missing file → ``codex login --device-auth``
          * file present but no ``tokens`` block → ``codex login``
          * id_token expired → ``codex login --device-auth`` (refresh)
        """
        from pathlib import Path

        path = self._codex_auth_path()
        if not path.exists():
            return self._AuthStatus(
                ok=False,
                reason="missing_auth_file",
                detail=f"找不到 {path}；请在终端跑 codex login --device-auth",
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._AuthStatus(
                ok=False,
                reason="auth_file_unreadable",
                detail=f"{path} 解析失败：{exc.__class__.__name__}: {exc}",
            )
        tokens = raw.get("tokens")
        if not isinstance(tokens, dict):
            return self._AuthStatus(
                ok=False,
                reason="no_tokens",
                detail=f"{path} 没有 tokens 字段；请重新 codex login",
            )
        id_token = tokens.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            return self._AuthStatus(
                ok=False,
                reason="no_id_token",
                detail=f"{path} 缺少 id_token；请重新 codex login",
            )
        exp = _decode_jwt_exp(id_token)
        if exp is None:
            return self._AuthStatus(
                ok=False,
                reason="unparseable_jwt",
                detail=f"{path} 的 id_token 不是合法 JWT",
            )
        now = time.time()
        if exp <= now:
            return self._AuthStatus(
                ok=False,
                reason="token_expired",
                detail=f"id_token 已于 {_fmt_unix(exp)} 过期；请在终端跑 codex login --device-auth",
            )
        if exp - now < 3600:
            # Soon-to-expire: still "ok" for the next chat turn,
            # but warn so the user can refresh proactively.
            return self._AuthStatus(
                ok=True,
                reason="token_expiring_soon",
                detail=f"id_token 将于 {_fmt_unix(exp)} 过期（不到 1 小时）",
            )
        return self._AuthStatus(
            ok=True,
            reason="ok",
            detail=f"id_token 有效（至 {_fmt_unix(exp)}）",
        )

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
            self._run_chat_turn(text, silent=False), self._loop
        )

    async def _run_chat_turn(self, text: str, *, silent: bool = False) -> None:
        # Round 17: spawn the progress heartbeat task alongside the
        # turn coroutine. The heartbeat emits ``EVENT_THINKING_PROGRESS``
        # every 1.5 s while the turn is in flight and is cancelled
        # before the turn's terminal status fires (so the user
        # never sees a stale "已思考 12s" line after the answer
        # lands).
        progress_task: asyncio.Task | None = None
        turn_started_at = time.monotonic()
        try:
            progress_task = asyncio.create_task(
                self._emit_progress_loop(turn_started_at)
            )
            await self._drive_chat_turn(text, silent=silent)
        finally:
            if progress_task is not None:
                progress_task.cancel()
                try:
                    await progress_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _drive_chat_turn(self, text: str, *, silent: bool) -> None:
        # Wait for ``_prime_service`` to finish (or skip) before
        # touching the service — otherwise a user turn that races
        # the background warm-up can land on a not-yet-ready client
        # and silently downgrades to the legacy ``handle_agent_chat``
        # path (which doesn't take ``on_delta`` and therefore never
        # streams). The wait is bounded so a hung warm-up never
        # freezes the chat thread.
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._service_ready.wait, 30.0,
            )
        except Exception:
            pass
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
                # Round 17: stream assistant deltas so the chat bubble
                # fills token-by-token instead of waiting for the full
                # turn to land. ``on_delta`` is a SYNC callback (the
                # codex client awaits the returned coroutine if any,
                # so a plain function is fine) — we just ``_emit`` the
                # delta event and the Qt shell queues the bubble
                # update via ``QMetaObject.invokeMethod``. ``_emit``
                # only does a Python-side pub/sub dispatch, no Qt
                # involvement, so calling it from the controller's
                # asyncio worker thread is safe.
                #
                # ``EVENT_ASSISTANT_STREAM_OPEN`` fires before the
                # first delta so the shell can open a bubble with the
                # cursor pinned to its end; ``append_delta`` is a
                # no-op until that fires.
                stream_open_emitted = False

                def _on_delta(delta: str) -> None:
                    nonlocal stream_open_emitted
                    if not stream_open_emitted:
                        stream_open_emitted = True
                        self._emit(self.EVENT_ASSISTANT_STREAM_OPEN)
                    self._emit(self.EVENT_ASSISTANT_DELTA, delta=delta)

                response, workspace = await service.chat(
                    request,
                    self._workspace,
                    self._session,
                    on_delta=_on_delta,
                )
            else:
                response, workspace = await handle_agent_chat(
                    request,
                    self._workspace,
                    self._config,
                    session=self._session,
                )
        except Exception as exc:
            # Round 17: classify the failure before forwarding it
            # to the shell. ``codex_runtime.errors`` already maps
            # transport failures to ``CodexErrorCode`` enums; we
            # translate the code into a user-facing one-liner plus
            # a "what to do next" hint. A bare ``Exception`` falls
            # through to the previous generic message.
            error_payload = _classify_chat_error(exc)
            self._emit(self.EVENT_ERROR, **error_payload)
            # 401 specifically should re-probe the auth file — the
            # token may have just expired since the last check.
            if error_payload.get("error_code") == "unauthorized":
                self._auth_required_emitted = False
                self.check_codex_auth(force=True)
            self._emit(self.EVENT_THINKING, active=False)
            self._emit(self.EVENT_STATUS_CHANGED, text="错误")
            return
        with self._lock:
            self._workspace = workspace
        if not silent:
            self._emit(
                self.EVENT_MESSAGE_APPENDED,
                role="assistant",
                text=response.reply,
                action=response.action,
                warnings=response.warnings,
            )
            self._emit(self.EVENT_WORKSPACE_REFRESHED)
        self._emit(self.EVENT_THINKING, active=False)
        self._emit(self.EVENT_STATUS_CHANGED, text="就绪" if not silent else "已就绪")

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

    # ------------------------------------------------------------------
    # Session switching (Round 17)
    # ------------------------------------------------------------------

    def switch_session(self, session_id: str) -> bool:
        """Switch the controller's active session to ``session_id``.

        Round 17: ``TracePanel.set_sessions`` renders a list of
        historical sessions but the click was previously a no-op —
        the user could see other sessions but never load them. Now
        ``switch_session`` re-binds the controller to the chosen
        session's workspace + chat history and emits a
        ``session_history_refreshed`` event so the shell can re-render
        the chat scrollback.

        Returns True when the switch succeeded; False when the
        requested session doesn't exist (or is archived). The shell
        surfaces the failure as a chat bubble so the user knows
        nothing happened.
        """
        from littrace.session import load_existing_session

        target = load_existing_session(self._config, session_id)
        if target is None:
            self._emit(self.EVENT_ERROR, **{
                "error_code": "other",
                "message": f"找不到 session {session_id}",
                "suggestion": "该 session 可能已被删除或归档。",
                "raw": f"load_existing_session returned None for {session_id}",
            })
            return False
        with self._lock:
            self._session = target
            self._workspace = load_workspace(target)
        self._emit(self.EVENT_SESSION_HISTORY_REFRESHED)
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        # Drop any in-flight streaming anchor so the new session's
        # first chat turn starts with a clean slate.
        self._emit(self.EVENT_STATUS_CHANGED, text="已切换 session")
        return True

    def list_sessions(self) -> list[Any]:
        """Return the available session summaries for the shell's
        history list.

        Round 17: exposed on the controller so the Qt shell can
        re-render the session list after a switch (the new
        active session should appear bolded in the list).
        """
        from littrace.session import list_chat_sessions

        return list_chat_sessions(self._config)

    # ------------------------------------------------------------------
    # Slash command dispatch (Round 19)
    # ------------------------------------------------------------------

    def submit_slash_command(self, name: str, args: str = "") -> None:
        """Execute a LitTrace slash command locally and emit the result.

        Round 19: the GUI's ``_on_send`` previously forwarded every
        ``/foo`` straight into Codex as a plain chat message, which
        Codex has no way to interpret — so the 33 entries in
        ``COMMAND_CATALOG`` were visually present but functionally
        dead. The CLI's ``cli.py`` loop, by contrast, parses each
        command and runs a real skill. This method closes that gap by
        running the same skill the CLI would, but emitting the
        formatted result through ``EVENT_SLASH_RESULT`` instead of
        printing to stdout.

        For commands whose natural home is the chat pipeline
        (``/parse``, ``/table``, ``/storyline``, ``/full-text``) we
        re-route into ``submit_user_message`` with the same Chinese
        intent string the CLI uses, so the LLM gets exactly the same
        prompt regardless of which shell invoked the slash.
        """
        handler = _SLASH_HANDLERS.get(name)
        if handler is None:
            self._emit(
                self.EVENT_SLASH_RESULT,
                name=name,
                text=f"未知命令 /{name} — 输入 / 看可用命令。",
            )
            return
        try:
            handler(self, args)
        except Exception as exc:  # pragma: no cover - defensive
            self._emit(
                self.EVENT_SLASH_RESULT,
                name=name,
                text=f"/{name} 失败：{exc.__class__.__name__}: {exc}",
            )

    # Slash command handlers -------------------------------------------

    def _slash_emit(self, name: str, text: str) -> None:
        """Helper used by every local slash handler to publish a
        system-style result. Centralised so future formatting (e.g.
        folding into a richer renderer) needs to change in one spot.
        """
        self._emit(self.EVENT_SLASH_RESULT, name=name, text=text)

    def _slash_show_context(self, args: str) -> None:
        # Display handler — used by /papers and /context. Emits a
        # ``context`` slash_result so the chat panel can render the
        # formatted panel.
        from littrace.cli import format_context_panel

        text = format_context_panel(self._workspace)
        self._slash_emit("context", text)

    def _slash_dashboard(self, args: str) -> None:
        # ``format_dashboard`` expects a ``ShellState`` — build a
        # throwaway with the controller's session + workspace so the
        # CLI formatter remains the single source of truth for the
        # dashboard text.
        from littrace.cli import ShellState, format_dashboard

        state = ShellState(
            workspace=self._workspace,
            session_id=self._session.session_id,
            session_root=str(self._session.root),
            context_visible=self._workspace.context.visible_to_user,
        )
        self._slash_emit("dashboard", format_dashboard(state))

    def _slash_workflow(self, args: str) -> None:
        from littrace.skill_runner import build_workflow_status

        report = build_workflow_status(self._workspace)
        lines = [
            f"Workflow: ready={report.ready_count}, "
            f"blocked={report.blocked_count}, "
            f"complete={report.complete_count}",
        ]
        for transition in report.transitions:
            lines.append(
                f"- {transition.source} -> {transition.target}: "
                f"{transition.status} | {transition.artifact}"
            )
        if report.recommended_next_steps:
            lines.append("下一步：" + "，".join(report.recommended_next_steps))
        self._slash_emit("workflow", "\n".join(lines))

    def _slash_quality(self, args: str) -> None:
        from littrace.skill_runner import build_quality_report_skill

        report = build_quality_report_skill(self._config, self._workspace)
        lines = ["Quality metrics:"]
        for name, value in report.metrics.items():
            lines.append(f"- {name}: {value}")
        if report.warnings:
            lines.append("注意：" + "；".join(report.warnings[:8]))
        self._slash_emit("quality", "\n".join(lines))

    def _slash_quality_audits(self, args: str) -> None:
        from littrace.quality_audits import (
            audit_parser,
            audit_tables,
            audit_storyline,
        )

        lines: list[str] = []
        for report in [
            audit_parser(self._config, self._workspace),
            audit_tables(self._workspace),
            audit_storyline(self._workspace),
        ]:
            status = "passed" if report.passed else "needs work"
            lines.append(f"- {report.component}: {status} ({report.score})")
            for finding in report.findings[:3]:
                lines.append(f"  - {finding}")
        self._slash_emit("quality-audits", "\n".join(lines))

    def _slash_ocr_choice(self, args: str) -> None:
        from littrace.parse_jobs import decide_artifact_extraction_need

        report = decide_artifact_extraction_need(self._workspace)
        lines = [
            f"OCR 建议: {report.recommended_parse_strategy}",
            f"理由: {report.reason}",
            "按钮:",
        ]
        for button in report.buttons:
            marker = "推荐" if button.get("recommended") == "true" else "可选"
            lines.append(
                f"- [{marker}] {button['label']} -> "
                f"parse_strategy={button['parse_strategy']}"
            )
            lines.append(f"  {button['description']}")
        self._slash_emit("ocr-choice", "\n".join(lines))

    def _slash_storyline_report(self, args: str) -> None:
        from littrace.publication import render_publication_storyline

        markdown, _ = render_publication_storyline(self._workspace, self._config)
        self._slash_emit("storyline-report", markdown)

    def _slash_storyline_review(self, args: str) -> None:
        from littrace.publication import review_storyline

        report = review_storyline(self._workspace)
        lines = [
            f"Storyline review: "
            f"{'passed' if report.passed else 'needs work'} "
            f"({report.claim_count} claims)",
        ]
        for warning in report.warnings:
            lines.append(f"- {warning}")
        self._slash_emit("storyline-review", "\n".join(lines))

    def _slash_doctor(self, args: str) -> None:
        # The CLI helper prints; capture via redirect to give the GUI
        # a one-shot doctor summary in chat.
        import io
        from contextlib import redirect_stdout
        from littrace.cli import _print_doctor

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_doctor(self._config)
        self._slash_emit("doctor", buf.getvalue() or "（无输出）")

    def _slash_setup_browser(self, args: str) -> None:
        import io
        from contextlib import redirect_stdout
        from littrace.cli import _print_browser_setup

        launch = self._config.cdp_downloader.auto_launch_chrome
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_browser_setup(self._config, profile_name=None, launch=False)
        self._slash_emit("setup-browser", buf.getvalue() or "（无输出）")

    def _slash_hide_context(self, args: str) -> None:
        with self._lock:
            self._workspace = self._workspace.model_copy(
                update={
                    "context": self._workspace.context.model_copy(
                        update={"visible_to_user": False},
                    ),
                },
            )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        self._slash_emit("hide-context", "已隐藏上下文窗。")

    def _slash_reveal_context(self, args: str) -> None:
        """Toggle the ``/show-context`` action: flip
        ``visible_to_user`` back to True and re-render the panel.
        Renamed from ``_slash_show_context`` to avoid colliding with
        the ``/papers`` display handler — both used the same name in
        Round 19's initial draft, and the latter silently shadowed
        the former so /papers rendered an empty bubble.
        """
        with self._lock:
            self._workspace = self._workspace.model_copy(
                update={
                    "context": self._workspace.context.model_copy(
                        update={"visible_to_user": True},
                    ),
                },
            )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        from littrace.cli import format_context_panel

        self._slash_emit("show-context", format_context_panel(self._workspace))

    def _slash_init_config(self, args: str) -> None:
        from littrace.config_wizard import write_config_template

        result = write_config_template()
        text = f"Config: {'created' if result.created else 'not changed'} at {result.path}"
        if result.warnings:
            text += "\n注意：" + "；".join(result.warnings)
        self._slash_emit("init-config", text)

    def _slash_set_bg(self, args: str) -> None:
        from littrace.research_background import set_workspace_research_background
        from littrace.session import save_workspace

        if not args:
            self._slash_emit(
                "set-bg", "用法：/set-bg 我研究的是<材料>在<场景>中的<问题>"
            )
            return
        with self._lock:
            self._workspace = set_workspace_research_background(self._workspace, args)
        save_workspace(self._session, self._workspace, config=self._config)
        filters = self._workspace.context.filters
        text = (
            f"研究背景已设置 (status={filters.research_background_status})\n"
            f"  主题: {filters.topic}\n"
            f"  时间: {filters.research_background_set_at}"
        )
        self._slash_emit("set-bg", text)

    def _slash_route_to_chat(self, args: str, *, intent: str, name: str) -> None:
        """Re-route a GUI slash command into the chat pipeline with the
        same Chinese prompt the CLI uses for the same command. We
        don't try to re-implement the skill — the LLM is the executor.
        """
        self._slash_emit(name, f"已交给 Codex：{intent}")
        self.submit_user_message(intent)

    def _slash_export(self, args: str) -> None:
        from littrace.export import export_session_bundle

        bundle = export_session_bundle(
            self._session, self._workspace, self._config
        )
        # Round 19: ``bundle`` is the {filename: contents} dict the
        # API route would return. Render a short summary so the GUI
        # chat bubble stays compact; the user can open the resulting
        # files under ``<session>/artifacts/``.
        files = ", ".join(sorted(bundle.keys())) or "(empty)"
        self._slash_emit(
            "export",
            f"导出完成 → {files}",
        )

    # ------------------------------------------------------------------
    # Active paper management (Round 17) — kept here so the class
    # boundary stays clean after the Round 19 slash-command insert.
    # ------------------------------------------------------------------

    def deactivate_paper(self, paper_id: str) -> bool:
        """Remove ``paper_id`` from the active-papers list.

        Round 17: ``ContextPanel`` exposes a right-click
        "取消激活" action on each active paper so the user can
        prune the working set without restarting the session.
        Previously the only way to drop a paper was to edit the
        session's workspace JSON by hand.

        Returns True when the paper was actually removed; False
        when the id wasn't in the active list (so the shell can
        show a status message instead of a confusing empty refresh).
        """
        with self._lock:
            active = list(self._workspace.context.active_papers)
            if paper_id not in active:
                return False
            active = [pid for pid in active if pid != paper_id]
            self._workspace = self._workspace.model_copy(
                update={
                    "context": self._workspace.context.model_copy(
                        update={"active_papers": active},
                    ),
                },
            )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        return True

    # ------------------------------------------------------------------
    # Round 19: paper importance + pin/importance helpers. The
    # ``LiteratureContext`` model already carries ``pinned_papers``;
    # the new ``importance_levels`` field (added in this round) lets
    # the user mark a paper as 1=normal, 2=important, 3=critical.
    # ------------------------------------------------------------------

    def toggle_paper_pin(self, paper_id: str) -> bool:
        """Toggle whether ``paper_id`` is pinned. Returns the new
        pinned state (True = now pinned).
        """
        with self._lock:
            pinned = list(self._workspace.context.pinned_papers)
            if paper_id in pinned:
                pinned = [pid for pid in pinned if pid != paper_id]
                new_state = False
            else:
                pinned.append(paper_id)
                new_state = True
            self._workspace = self._workspace.model_copy(
                update={
                    "context": self._workspace.context.model_copy(
                        update={"pinned_papers": pinned},
                    ),
                },
            )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        return new_state

    def mark_rag_refresh(self, indexed_chunks: int = 0) -> None:
        """Record that a RAG refresh just completed.

        Round 19: the GUI's ``RAGPanel`` shows the timestamp /
        chunk count of the most recent refresh so the user can
        tell at a glance whether the index is fresh or stale
        (≥ 24 h ago). The controller owns the timestamp so the
        state survives ``_refresh_rag_panel`` re-emits and is
        queryable from non-GUI tests.
        """
        with self._lock:
            self._last_rag_refresh_at = time.time()
            self._last_rag_indexed_chunks = int(indexed_chunks)
        self._emit(self.EVENT_RAG_PANEL_REFRESHED)

    def get_rag_refresh_status(self) -> dict[str, Any]:
        """Return the most recent RAG refresh status. Both fields
        are optional — a fresh install returns ``None`` for both
        and the GUI renders a friendly placeholder.
        """
        return {
            "timestamp": getattr(self, "_last_rag_refresh_at", None),
            "indexed_chunks": getattr(self, "_last_rag_indexed_chunks", 0),
        }

    def set_paper_importance(self, paper_id: str, level: int) -> bool:
        """Set the importance of ``paper_id`` to ``level``
        (1=normal, 2=important, 3=critical, 0=clear). Returns True
        on success, False when the paper is not in the active set.
        """
        with self._lock:
            if paper_id not in self._workspace.context.active_papers:
                return False
            levels = dict(self._workspace.context.importance_levels)
            if level <= 0:
                levels.pop(paper_id, None)
            else:
                levels[paper_id] = level
            self._workspace = self._workspace.model_copy(
                update={
                    "context": self._workspace.context.model_copy(
                        update={"importance_levels": levels},
                    ),
                },
            )
        self._emit(self.EVENT_WORKSPACE_REFRESHED)
        return True


# Slash command dispatch table (Round 19)
#
# The map is defined at module load time so a future ``/help`` command
# can introspect it without re-iterating the controller class. Handlers
# that just want to push a pre-canned intent into the chat pipeline
# share ``_ControllerSlaskRouter._slash_route_to_chat`` so the routing
# table stays small.


def _slash_chat_intent_handler(intent: str):
    def _handler(controller: "ShellController", args: str) -> None:
        controller._slash_route_to_chat(args, intent=intent, name=intent.lstrip("/"))
    return _handler


_SLASH_HANDLERS: dict[str, Callable[["ShellController", str], None]] = {
    # Tier A: display-only (formatted text, no skill side-effects)
    "context": lambda c, a: c._slash_show_context(a),
    "papers": lambda c, a: c._slash_show_context(a),
    "dashboard": lambda c, a: c._slash_dashboard(a),
    "workflow": lambda c, a: c._slash_workflow(a),
    "quality": lambda c, a: c._slash_quality(a),
    "quality-audits": lambda c, a: c._slash_quality_audits(a),
    "ocr-choice": lambda c, a: c._slash_ocr_choice(a),
    "storyline-report": lambda c, a: c._slash_storyline_report(a),
    "storyline-review": lambda c, a: c._slash_storyline_review(a),
    "doctor": lambda c, a: c._slash_doctor(a),
    "setup-browser": lambda c, a: c._slash_setup_browser(a),
    "hide-context": lambda c, a: c._slash_hide_context(a),
    "show-context": lambda c, a: c._slash_reveal_context(a),
    "export": lambda c, a: c._slash_export(a),
    "init-config": lambda c, a: c._slash_init_config(a),
    # Tier B: state mutation
    "set-bg": lambda c, a: c._slash_set_bg(a),
    # Tier A: route through chat pipeline (LLM executes the skill)
    "parse": _slash_chat_intent_handler("解析当前文献全文"),
    "table": _slash_chat_intent_handler("生成当前文献性能对比表"),
    "storyline": _slash_chat_intent_handler("生成当前文献发展脉络"),
    "full-text": _slash_chat_intent_handler("为当前文献构建 full-text context"),
}


def get_slash_command_names() -> list[str]:
    """Return the sorted list of slash command names this controller
    knows about. The Qt shell uses it to populate ``COMMAND_CATALOG``
    so the popup stays in sync with the dispatch table — adding a new
    slash in ``shell_controller`` only requires an entry in the GUI's
    ``_populate_command_catalog`` list.
    """
    return sorted(_SLASH_HANDLERS)


def _decode_jwt_exp(token: str) -> float | None:
    """Return the ``exp`` claim of a JWT, or ``None`` if the token
    can't be decoded.

    Round 17 helper for ``_read_codex_auth_status``. We only need
    the payload (base64url-decoded JSON) — no signature
    verification, since this is a UI-side expiry probe, not a
    security check. The codex App Server validates the signature
    on every API call and surfaces 401s as ``UnauthorizedError``,
    which the service layer already turns into
    ``EVENT_AUTH_REQUIRED`` at chat-turn time.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    # Pad to a multiple of 4 so base64 doesn't complain about
    # missing ``=`` padding (JWTs strip the trailing ``=``).
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
    except Exception:
        return None
    try:
        payload_obj = json.loads(payload)
    except (TypeError, ValueError):
        return None
    exp = payload_obj.get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def _fmt_unix(ts: float) -> str:
    """Round 17 helper for auth dialog text. Returns a human-
    readable ``YYYY-MM-DD HH:MM`` timestamp in local time so the
    user can sanity-check the JWT ``exp`` claim without
    remembering that codex uses UTC.
    """
    import time as _time
    return _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ts))


# Round 17: error taxonomy for ``EVENT_ERROR``. The Qt shell looks
# up ``error_code`` and ``suggestion`` to render a friendly
# message and a "what to do next" hint, instead of dumping the
# raw ``TypeError: ...`` traceback into the chat scrollback.
# Add new entries here when ``CodexErrorCode`` gains new values;
# the controller test suite asserts every enum value has a
# mapping.
_CHAT_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
    "context_window_exceeded": (
        "对话历史过长，模型已无法继续阅读。",
        "请开一个新对话（点击左侧 Session 列表中的「新建」），或者精简之前的问题。",
    ),
    "session_budget_exceeded": (
        "本次会话的 token 预算已用完。",
        "请开新对话或联系管理员调整上限。",
    ),
    "usage_limit_exceeded": (
        "已达到 ChatGPT 套餐的用量上限。",
        "请稍后再试，或在 codex 客户端升级套餐。",
    ),
    "active_turn_not_steerable": (
        "当前轮次无法被插入新指令。",
        "请等待上一轮回答完成后再发送。",
    ),
    "bad_request": (
        "codex 拒绝了这个请求（参数或状态不合法）。",
        "请重试一次；如果持续失败，重启 littrace-qt。",
    ),
    "unauthorized": (
        "codex 登录已过期或被撤销。",
        "请在终端跑 `codex login --device-auth`，完成后回到本窗口点「重新检查」。",
    ),
    "sandbox_error": (
        "codex 的沙箱拒绝了这次操作。",
        "请检查 config.yaml 的 agent_runtime.sandbox_policy；"
        "需要写文件时改为 workspace-write 或 danger-full-access。",
    ),
    "internal_server_error": (
        "codex 内部错误。",
        "请稍后重试；如果持续失败，重启 littrace-qt。",
    ),
    "other": (
        "对话出错。",
        "请重试一次，必要时重启 littrace-qt。",
    ),
}


def _classify_chat_error(exc: BaseException) -> dict[str, str]:
    """Map a chat-turn exception to the ``EVENT_ERROR`` payload.

    Round 17: instead of forwarding ``f"{type(exc).__name__}: {exc}"``
    as a single string, return a dict with the structured
    ``error_code`` (string form of ``CodexErrorCode`` when
    available), a one-line ``message`` for the chat bubble, and a
    multi-line ``suggestion`` for the shell's details popup.
    """
    # Try the structured ``CodexErrorCode`` first — codex
    # transport failures already carry it.
    code_str: str | None = None
    error_code = getattr(exc, "error_code", None)
    if error_code is not None:
        code_str = getattr(error_code, "value", str(error_code))
    # Fall back to substring heuristics for legacy exceptions
    # that pre-date the structured-error vocabulary.
    if code_str is None:
        message = str(exc).lower()
        if "unauthorized" in message or "401" in message:
            code_str = "unauthorized"
        elif "context window" in message or "context length" in message:
            code_str = "context_window_exceeded"
        elif "usage limit" in message or "rate limit" in message:
            code_str = "usage_limit_exceeded"
        elif "sandbox" in message:
            code_str = "sandbox_error"
        else:
            code_str = "other"
    friendly, suggestion = _CHAT_ERROR_MESSAGES.get(
        code_str, _CHAT_ERROR_MESSAGES["other"]
    )
    return {
        "error_code": code_str,
        "message": friendly,
        "suggestion": suggestion,
        "raw": f"{type(exc).__name__}: {exc}",
    }
