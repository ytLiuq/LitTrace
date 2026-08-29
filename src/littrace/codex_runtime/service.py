"""High-level LitTrace chat service backed by Codex App Server."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager, nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any

from littrace.codex_runtime.client import (
    AppServerClient,
    AppServerError,
    AppServerTurnResult,
    SteerTurnResult,
)
from littrace.codex_runtime.errors import CodexErrorCode
from littrace.codex_runtime.gateway import APP_SERVER_TOOL_NAMES
from littrace.codex_runtime.runtime import (
    CodexAppServerRuntimeManager,
    shared_runtime_manager,
)
from littrace.config import CodexHomeMode, LitTraceConfig, SandboxPolicy
from littrace.codex_runtime.errors import (
    BadRequestError,
    InternalServerError,
    UnauthorizedError,
)
from littrace.codex_runtime.rollout import RolloutRecorder, rollout_path_for
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import ChatSession
from littrace.state_db import AgentThreadBindingRecord, StateStore, state_store_from_config

DEVELOPER_INSTRUCTIONS = """\
You are the conversational research layer inside LitTrace. LitTrace Postgres state is the
canonical source of scientific truth; Codex thread history is execution memory only.

Use only tools on the `littrace` MCP server for facts and changes involving the active session.
Never infer that a filesystem file is canonical. Do not mutate files or run shell commands.
The currently supported domain mutations are `set_download_selection`, `search_papers`,
`enqueue_download`, `enqueue_parse`, and `enqueue_table_extraction`.

WORKFLOW FOR A LITERATURE-DISCOVERY REQUEST:
1. Call `search_papers` first with the topic + year_min + limit. Do NOT call
   `get_workspace_context` beforehand — its `workspace_revision: 0` on an empty
   session does not block a search.
2. After `search_papers` returns, summarise the returned `papers` array
   directly (titles, DOIs, journal, year, access type).
3. Only after a mutating tool (search, set_download_selection, enqueue_*) has
   been called should you call `get_workspace_context` to verify the new
   workspace revision; pass that revision forward on any subsequent mutation
   in the same turn.

WORKFLOW FOR ANY OTHER MUTATION:
Call `get_workspace_context` first, pass the returned workspace revision, and
use a stable idempotency key for retries of one intended change. Download, parse,
and table extraction commands only submit durable work; explain that they are
queued and use the matching job-status tool when the user asks for progress.
Other mutations must be handled by the legacy LitTrace domain workflow. Keep
evidence and paper identifiers in answers when available.

If a tool returns a successful 200 response, treat that as the authoritative
result — do not reinterpret the reply as "rejected" or "denied" because the
workspace happens to be empty.
"""


# Round 16: phrasings the upstream codex App Server uses when it cannot
# — or the model thinks it cannot — call an MCP tool. Match the reply
# against this list so ``fallback_to_legacy`` can take over.
#
# Categories:
#   1. Upstream exec-mode guard (codex 0.140-0.150 stdio path):
#      "approval policy", "requires approval", "需要批准".
#   2. ChatGPT.app bundled codex 0.149.0-alpha.4.3 model confusion
#      — the tool actually returned success:true but the model
#      misread an empty-but-valid workspace as a refusal:
#      "工具调用被拒绝 / 被系统拒绝 / 被运行环境拦截" and the
#      English variants. The model's confused phrasing is stable
#      enough that a substring match is reliable.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    # Upstream exec-mode
    "approval policy",
    "需要批准",
    "requires approval",
    # Model confused on empty-but-valid response
    "工具调用被拒绝",
    "工具调用被系统拒绝",
    "工具调用连续被拒绝",
    "工具调用被运行环境拦截",
    "工具调用被工具层拒绝",
    "工具调用被工具端拒绝",
    "工具调用刚被拒绝",
    "工具调用未获授权",
    "工具侧拒绝",
    "运行环境拦截",
    "工具层拒绝",
    "工具端拒绝",
    "tool call was rejected",
    "tool call was denied",
    "tool calls were rejected",
)


def _looks_like_refusal(reply: str) -> bool:
    """Return True if ``reply`` looks like the codex App Server
    refusing an MCP tool call (either the upstream guard or the
    model's confused misread of an empty-but-valid response)."""
    if not reply:
        return False
    lowered = reply.lower()
    if any(pattern in reply for pattern in _REFUSAL_PATTERNS):
        return True
    # English variants that only appear in lowercased form. The
    # upstream codex's "Tool call was denied..." phrasing is
    # common in the confused-model replies, so match it directly.
    return any(
        marker in lowered
        for marker in (
            "tool call rejected",
            "tool call denied",
            "tool call was rejected",
            "tool call was denied",
            "tool calls were rejected",
            "tool calls were denied",
        )
    )


class CodexAppServerChatService:
    """Bind one LitTrace session to one durable Codex thread."""

    def __init__(
        self,
        config: LitTraceConfig,
        *,
        state_store: StateStore | None = None,
        client_factory: Callable[..., AppServerClient] = AppServerClient,
        runtime_manager: CodexAppServerRuntimeManager | None = None,
    ) -> None:
        self.config = config
        self.state_store = state_store or state_store_from_config(config)
        self.client_factory = client_factory
        self.runtime_manager = runtime_manager

    async def chat(
        self,
        request: ChatRequest,
        workspace: LiteratureWorkspace,
        session: ChatSession,
        *,
        cancellation: asyncio.Event | None = None,
        on_delta=None,
        elicitation_handler=None,
    ) -> tuple[ChatResponse, LiteratureWorkspace]:
        scratch_dir = self._scratch_dir(session.session_id)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        # Optional side-channel rollout log for debugging. Opt-in via
        # config.agent_runtime.rollout_enabled; never on by default.
        recorder: RolloutRecorder | None = None
        if self.config.agent_runtime.rollout_enabled:
            recorder = RolloutRecorder(
                rollout_path_for(
                    session, base_dir=self.config.agent_runtime.rollout_dir,
                )
            )
            recorder.open()
        try:
            async with _agent_binding_lock(self.state_store, session.session_id):
                if self.runtime_manager is not None:
                    turn, latest_workspace = await self.runtime_manager.use(
                        lambda client: self._chat_with_client(
                            client, request, workspace, session, scratch_dir,
                            cancellation, recorder, on_delta,
                            elicitation_handler,
                        )
                    )
                elif self.client_factory is AppServerClient:
                    manager = self._shared_runtime_manager(
                        rollout_recorder=recorder,
                    )
                    turn, latest_workspace = await manager.use(
                        lambda client: self._chat_with_client(
                            client, request, workspace, session, scratch_dir,
                            cancellation, recorder, on_delta,
                            elicitation_handler,
                        )
                    )
                else:
                    # Custom factories are intentionally ephemeral.
                    # They typically inject their own fake client, so
                    # the recorder injection happens via client_factory
                    # closure rather than through runtime_manager.
                    client = self.client_factory(
                        self._codex_command(),
                        **self._client_options(rollout_recorder=recorder),
                    )
                    async with client:
                        turn, latest_workspace = await self._chat_with_client(
                            client, request, workspace, session, scratch_dir,
                            cancellation, recorder, on_delta,
                            elicitation_handler,
                        )
        finally:
            # Close the recorder after the App Server has returned the
            # terminal event so any straggler frames (orphan
            # notifications after turn/completed) still get appended.
            if recorder is not None:
                recorder.close()
        reply = turn.reply or "Codex App Server completed the turn without a text response."
        if _looks_like_refusal(reply):
            raise AppServerError(
                error_code=CodexErrorCode.UNAUTHORIZED,
                message=(
                    "Codex app-server refused MCP tool calls (either via "
                    "the upstream exec-mode guard or because the model "
                    "misread an empty-but-valid response). Falling back "
                    f"to LitTrace's native chat path. Upstream reply: {reply[:200]}"
                ),
            )
        # Surface the terminal status as the chat action so the route
        # layer can route the response into the right action group
        # (chat vs interrupted vs committed_transport_failure).
        if turn.status == "interrupted":
            action = "codex_app_server_interrupted"
        elif turn.status == "failed":
            action = "codex_app_server_interrupted_failed"
        else:
            action = "codex_app_server_chat"
        return (
            ChatResponse(
                reply=reply,
                action=action,
                session_id=session.session_id,
            ),
            latest_workspace,
        )

    async def _chat_with_client(
        self,
        client: AppServerClient,
        request: ChatRequest,
        workspace: LiteratureWorkspace,
        session: ChatSession,
        scratch_dir: Path,
        cancellation: asyncio.Event | None = None,
        recorder: RolloutRecorder | None = None,
        on_delta=None,
        elicitation_handler=None,
    ):
        # Round 4 P1 step 7: bind the per-thread recorder. The
        # runtime_manager keeps one client across calls, so two
        # concurrent chats on different sessions used to trample
        # each other's JSONL file. The new per-thread dict on
        # ``client._rollout_recorders`` keys by ``codex_thread_id``,
        # so binding the new chat's recorder overwrites only that
        # thread's slot. A follow-up chat on the SAME thread overwrites
        # it again, which is the desired behaviour.
        # Round 4 P2 step 10: serialise the binding read-modify-write.
        # ``session_write_lock`` already takes the
        # ``littrace:session-write:{session_id}`` advisory lock for
        # the workspace body; taking the same lock here keeps the
        # binding lookup and upsert atomic against a concurrent
        # ``service.chat`` on the same session so we never end up
        # with two ``codex_thread_id`` rows for one LitTrace
        # session_id. The lock is released as soon as this ``with``
        # block exits, so the workspace body write that follows is
        # not held under this lock.
        binding = self.state_store.get_agent_thread_binding(session.session_id)
        with self.state_store.session_write_lock(session.session_id):
            binding = self.state_store.get_agent_thread_binding(session.session_id)
        if recorder is not None and binding is not None:
            client.set_rollout_recorder(binding.codex_thread_id, recorder)
        runtime = self.config.agent_runtime
        await self._require_authentication(client)
        thread_overrides = self._thread_overrides(scratch_dir)
        binding = self.state_store.get_agent_thread_binding(session.session_id)
        thread: dict[str, Any]
        runtime_kind = self._runtime_kind()
        can_resume = (
            binding is not None
            and binding.status == "active"
            and binding.runtime_kind == runtime_kind
        )
        if can_resume:
            assert binding is not None
            thread = await client.resume_thread(
                binding.codex_thread_id,
                thread_overrides,
            )
        else:
            thread = await client.start_thread(thread_overrides)
            thread_id = str(thread.get("id") or "")
            if not thread_id:
                raise BadRequestError("thread/start did not return a thread id")
            binding = self.state_store.upsert_agent_thread_binding(
                AgentThreadBindingRecord(
                    session_id=session.session_id,
                    codex_thread_id=thread_id,
                    runtime_kind=runtime_kind,
                    runtime_version=_runtime_version(client),
                    workspace_revision=workspace.context.filters.workspace_revision,
                )
            )
            # Round 4 P1 step 6: real session_meta write. Round 2
            # commit 3 (rollout) registered the type but never had a
            # production code path emit it — the rollout tests
            # appended it manually. Now the service is the source.
            if recorder is not None:
                recorder.append(
                    type_="session_meta",
                    session_id=session.session_id,
                    codex_thread_id=thread_id,
                    runtime_kind=runtime_kind,
                    runtime_version=_runtime_version(client),
                    resume=can_resume,
                )
        thread_id = binding.codex_thread_id
        # Round 4 P1 step 5: status machine
        #   draft  --(start/resume)-->  idle
        #   idle   --(run_turn start)-->  active
        #   active --(run_turn done)---->  idle
        #   *      --(transport error)-->  systemError  (frozen)
        #   *      --(delete_chat_session)--> archived
        # Only ``archived`` and ``systemError`` reject writes (the
        # state_db CAS guards); ``draft`` + ``idle`` both accept
        # revision=0 (chat path takeover) and ``active`` is the
        # normal real-writer case.
        current_session = self.state_store.get_session_state(session.session_id)
        if binding.status in ("draft", "active") and current_session is not None:
            # first chat: promote the placeholder into idle.
            self.state_store.upsert_session_state(
                current_session.model_copy(update={"status": "idle"})
            )
        current_session = self.state_store.get_session_state(session.session_id)
        if current_session is not None:
            self.state_store.upsert_session_state(
                current_session.model_copy(update={"status": "active"})
            )
        # Round 4 P1 step 6: turn_context write — capture the
        # snapshot of thread_overrides the chat is about to send
        # to the App Server. The dict contains sandbox /
        # approvalPolicy / cwd / mcp_servers config; it is the
        # single piece of state the server saw before any tool
        # response, so the rollout log is the only authoritative
        # record of "what thread did this turn run in?".
        if recorder is not None:
            recorder.append(
                type_="turn_context",
                thread_id=thread_id,
                thread_overrides=thread_overrides,
            )
        await self._require_mcp_connection(client, thread_id)
        # Round 17: bind the per-chat elicitation handler. The
        # ``runtime_manager`` keeps one client across chats on
        # different sessions, so the previous handler (from
        # another session's TUI instance) must NOT leak into
        # this turn. ``set_elicitation_handler`` overwrites; we
        # restore the prior value in the finally block.
        client.set_elicitation_handler(elicitation_handler)
        try:
            turn = await client.run_turn(
                thread_id,
                request.message,
                timeout=runtime.turn_timeout_seconds,
                cancellation=cancellation,
                on_delta=on_delta,
            )
        except AppServerError as exc:
            # Any transport-level failure freezes the session in
            # systemError so an operator can inspect via the
            # session_state row rather than watching the chat path
            # auto-retry against a half-broken connection.
            if recorder is not None:
                recorder.append(
                    type_="system_error",
                    error_code=exc.error_code.value,
                    message=str(exc),
                    additional_details=exc.additional_details,
                )
            current_session = self.state_store.get_session_state(session.session_id)
            if current_session is not None:
                self.state_store.upsert_session_state(
                    current_session.model_copy(update={"status": "systemError"})
                )
            raise
        finally:
            # Clear the handler so a subsequent chat on this shared
            # client does not inherit the previous operator's hook.
            client.set_elicitation_handler(None)
        latest_workspace = self._latest_workspace(
            session.session_id,
            fallback=workspace,
        )
        # Turn completed cleanly — flip back to ``idle`` so a
        # subsequent chat picks up where we left off.
        current_session = self.state_store.get_session_state(session.session_id)
        if current_session is not None:
            self.state_store.upsert_session_state(
                current_session.model_copy(update={"status": "idle"})
            )
        # Round 5 compaction worker bookkeeping: bump turn_count
        # and record the latest Usage.total_tokens so the worker
        # can decide whether a thread has crossed the configured
        # threshold. Usage is optional (older fakes do not set it)
        # so the default 0 keeps the math safe.
        total_tokens = turn.usage.total_tokens if turn.usage else 0
        new_turn_count = (binding.turn_count or 0) + 1
        self.state_store.upsert_agent_thread_binding(
            binding.model_copy(
                update={
                    "workspace_revision": latest_workspace.context.filters.workspace_revision,
                    "status": "active",
                    "last_error": None,
                    "turn_count": new_turn_count,
                    "last_total_tokens": total_tokens,
                }
            )
        )
        return turn, latest_workspace

    async def steer(
        self,
        session: ChatSession,
        turn_id: str,
        text: str,
        *,
        client_user_message_id: str | None = None,
    ) -> "SteerTurnResult":
        """Forward a mid-turn input to the active App Server turn.

        Round 8 step 3: thin wrapper over
        ``AppServerClient.steer_turn`` that resolves the binding
        from the session id and reuses the shared runtime
        manager so the call lands on the same process the chat
        turn started on.
        """
        binding = self.state_store.get_agent_thread_binding(session.session_id)
        if binding is None:
            raise AppServerError(
                f"no active codex thread for session {session.session_id}"
            )
        manager = self._shared_runtime_manager()
        async with manager.use(
            lambda client: client.steer_turn(
                binding.codex_thread_id,
                turn_id,
                text,
                client_user_message_id=client_user_message_id,
            )
        ) as result:
            return result

    async def start_review(
        self,
        session: ChatSession,
        *,
        target: dict[str, Any] | None = None,
        on_review_complete=None,
    ) -> "AppServerTurnResult":
        """Kick off a codex review turn on the session's thread.

        Round 8 step 3: thin wrapper over
        ``AppServerClient.start_review`` that resolves the
        binding, installs a per-thread ``on_review_complete``
        callback on the shared runtime manager, and returns
        the typed ``AppServerTurnResult`` for the review turn.
        The callback fires from the reader loop when
        ``item/completed`` carries ``exitedReviewMode``.
        """
        binding = self.state_store.get_agent_thread_binding(session.session_id)
        if binding is None:
            raise AppServerError(
                f"no active codex thread for session {session.session_id}"
            )
        manager = self._shared_runtime_manager()
        async with manager.use(
            lambda client: _start_review_through(
                client, binding.codex_thread_id, target, on_review_complete
            )
        ) as result:
            return result

    async def cancel_turn_with_reason(
        self,
        session: ChatSession,
        turn_id: str,
        *,
        reason: str,
    ) -> bool:
        """Cancel an in-flight turn and record the reason on the binding.

        Round 8 step 3: the legacy ``cancel_current_turn`` does
        not surface *why* a turn was cancelled (user pressed
        Esc, compaction triggered, model loop detected, etc.).
        The route layer passes the reason through; we record it
        in the binding's ``last_error`` column so an operator
        can later inspect why a turn ended without
        status='completed'.

        Returns True if the App Server acknowledged
        ``turn/interrupt`` with a terminal event; False on a
        transport failure (the caller should treat the turn as
        terminated either way).
        """
        binding = self.state_store.get_agent_thread_binding(session.session_id)
        if binding is None:
            return False
        manager = self._shared_runtime_manager()
        ack = await manager.use(
            lambda client: client.cancel_current_turn(
                binding.codex_thread_id, turn_id,
            )
        )
        if ack:
            self.state_store.upsert_agent_thread_binding(
                binding.model_copy(update={"last_error": f"cancelled: {reason}"})
            )
        return ack

    def _client_options(
        self,
        *,
        rollout_recorder: RolloutRecorder | None = None,
    ) -> dict[str, Any]:
        runtime = self.config.agent_runtime
        options: dict[str, Any] = {
            "startup_timeout": runtime.startup_timeout_seconds,
            "request_timeout": runtime.request_timeout_seconds,
            "environment": self._codex_environment(),
        }
        if rollout_recorder is not None:
            options["rollout_recorder"] = rollout_recorder
        return options

    def _shared_runtime_manager(
        self,
        rollout_recorder: RolloutRecorder | None = None,
    ) -> CodexAppServerRuntimeManager:
        command = self._codex_command()
        options = self._client_options(rollout_recorder=rollout_recorder)
        key = (
            tuple(command),
            options["startup_timeout"],
            options["request_timeout"],
            tuple(sorted(options["environment"].items())),
        )
        return shared_runtime_manager(key, command, client_options=options)

    def _scratch_dir(self, session_id: str) -> Path:
        digest = sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self.config.agent_runtime.scratch_root.expanduser().resolve() / digest

    def _thread_overrides(
        self,
        scratch_dir: Path,
    ) -> dict[str, Any]:
        runtime = self.config.agent_runtime
        # codex 0.140-0.150 ``approvalPolicy`` enum is
        # ``{"untrusted", "on-request", "granular", "never"}``.
        # Round 14 (npm codex 0.149/0.150) tested "untrusted" +
        # "on-request" + "granular" — every value rejected MCP
        # tool calls in app-server (stdio) mode with
        # "MCP tool call requires approval, but approval policy
        # is never". The bug is in codex's exec-mode path
        # (``core/src/tools/network_approval.rs``).
        # Switching to the ChatGPT.app bundled codex (which has
        # ``guardian_approval`` feature enabled) lets the user
        # approve via ChatGPT.app dialogs, so we send
        # ``"on-request"`` — every MCP call triggers an
        # approval prompt in the ChatGPT.app GUI.
        approval_policy = {
            SandboxPolicy.READ_ONLY: "on-request",
            SandboxPolicy.WORKSPACE_WRITE: "on-request",
            SandboxPolicy.DANGER_FULL_ACCESS: "on-request",
        }[runtime.sandbox_policy]
        overrides: dict[str, Any] = {
            "cwd": str(scratch_dir),
            "approvalPolicy": approval_policy,
            "sandbox": runtime.sandbox_policy.value,
            "serviceName": "littrace",
            "developerInstructions": DEVELOPER_INSTRUCTIONS,
            "config": {
                "mcp_servers": {
                    runtime.mcp_server_name: self._mcp_server_config(),
                },
            },
        }
        # writableRoots only makes sense for workspace-write. Codex
        # ignores the key on the other two tiers; passing it through
        # unconditionally would imply an empty whitelist for
        # danger-full-access, which is the wrong default.
        if runtime.sandbox_policy == SandboxPolicy.WORKSPACE_WRITE:
            if runtime.writable_roots:
                overrides["writableRoots"] = [
                    str(root) for root in runtime.writable_roots
                ]
        return overrides

    def _mcp_server_config(self) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[3]
        config_path = Path(
            os.environ.get("LITTRACE_CONFIG_PATH", project_root / "config.yaml")
        ).expanduser()
        env = {
            "LITTRACE_MCP_GATEWAY": "1",
            "LITTRACE_CONFIG_PATH": str(config_path.resolve()),
            "LITTRACE_METADATA_BACKEND": self.config.metadata_store.backend,
            "LITTRACE_POSTGRES_SCHEMA": self.config.metadata_store.schema_name,
            "LITTRACE_POSTGRES_CONNECT_TIMEOUT_SECONDS": str(
                self.config.metadata_store.connect_timeout_seconds
            ),
            "PYTHONPATH": _prepend_path(
                str(project_root / "src"),
                os.environ.get("PYTHONPATH"),
            ),
        }
        if self.config.metadata_store.postgres_dsn:
            env["LITTRACE_POSTGRES_DSN"] = self.config.metadata_store.postgres_dsn
        if self.config.rag.postgres_dsn:
            env["LITTRACE_RAG_POSTGRES_DSN"] = self.config.rag.postgres_dsn
        # Round 14 CR: codex spawns the MCP server as a
        # subprocess. The mcp_server side now persists its
        # token under ``CODEX_HOME/littrace-mcp.token`` so it
        # is stable across restarts; mirror it into the
        # subprocess env so the parent LitTrace and the mcp
        # subprocess agree on the same secret without forcing
        # the operator to paste a one-shot token into the codex
        # config.
        from littrace.mcp_server import _token_path
        token_path = _token_path()
        if token_path.exists():
            env["LITTRACE_MCP_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
        return {
            "command": sys.executable,
            "args": ["-m", "littrace.mcp_server"],
            "cwd": str(project_root),
            "env": env,
            "required": True,
            "startup_timeout_sec": self.config.agent_runtime.startup_timeout_seconds,
            # Round 13 step 3: round 4 enumerated only the
            # 15 built-in tools; round 13 augments that with
            # every third-party ``littrace.mcp_servers``
            # plugin tool the installer has dropped on the
            # Python path. Built-in names always come first
            # so the App Server cannot accidentally advertise
            # a plugin shadow.
            "enabled_tools": list(self._enabled_mcp_tools()),
        }

    def _enabled_mcp_tools(self) -> list[str]:
        """Merge the built-in tool names with third-party plugin
        tools discovered via the ``littrace.mcp_servers``
        entry-point group.

        The merge is name-unique and preserves the order of
        the built-in list; plugin names are appended in
        alphabetical order so a future operator can sort
        ``littrace plugin list`` against the same set.
        """
        names = list(APP_SERVER_TOOL_NAMES)
        try:
            from littrace.marketplace import list_plugins
            for entry in list_plugins().by_group("littrace.mcp_servers"):
                if entry.name and entry.name not in names:
                    names.append(entry.name)
        except Exception:  # pragma: no cover - defensive
            log.warning(
                "failed to enumerate littrace.mcp_servers plugins",
                exc_info=True,
            )
        return names

    def _codex_command(self) -> list[str]:
        command = list(self.config.agent_runtime.codex_command)
        for key, value in self.config.agent_runtime.codex_config_overrides.items():
            command.extend(["-c", f"{key}={value}"])
        return command

    def _codex_environment(self) -> dict[str, str]:
        runtime = self.config.agent_runtime
        if runtime.codex_home_mode == CodexHomeMode.SHARED:
            return {}
        codex_home = runtime.codex_home.expanduser().resolve()
        codex_home.mkdir(parents=True, exist_ok=True)
        return {"CODEX_HOME": str(codex_home)}

    def _runtime_kind(self) -> str:
        runtime = self.config.agent_runtime
        if runtime.codex_home_mode == CodexHomeMode.SHARED:
            namespace = "shared"
        else:
            resolved = str(runtime.codex_home.expanduser().resolve())
            namespace = sha256(resolved.encode("utf-8")).hexdigest()[:12]
        return f"codex_app_server/{namespace}"

    async def _require_authentication(self, client: AppServerClient) -> None:
        account = await client.read_account(refresh_token=False)
        if account.get("requiresOpenaiAuth") is not True or account.get("account") is not None:
            return
        runtime = self.config.agent_runtime
        if runtime.codex_home_mode == CodexHomeMode.ISOLATED:
            home = runtime.codex_home.expanduser().resolve()
            raise UnauthorizedError(
                "The isolated LitTrace Codex home is not authenticated. "
                f"Set CODEX_HOME={home!s} and run `codex login` once, or explicitly "
                "set agent_runtime.codex_home_mode=shared during migration."
            )
        raise UnauthorizedError("Codex App Server is not authenticated")

    async def _require_mcp_connection(
        self,
        client: AppServerClient,
        thread_id: str,
    ) -> None:
        name = self.config.agent_runtime.mcp_server_name
        result = await client.call_mcp_tool(
            thread_id,
            name,
            "get_workspace_context",
            {},
        )
        # Round 4 P0 step 4: round 3 commit 3 wrapped gateway
        # responses in McpResponse. The App Server's in-tree call_mcp_tool
        # path returns the raw JSON-RPC ``result`` dict, so a
        # successful health probe looks like ``{"success": True, ...}``
        # while a failing one looks like ``{"success": False,
        # "error": {"code": "...", ...}}``. Check both shapes.
        if isinstance(result, dict) and result.get("success") is False:
            err = result.get("error") or {}
            raise BadRequestError(
                f"LitTrace MCP health check failed: {err.get('message') or result.get('content')}",
                additional_details=err,
            )
        if result.get("isError") is True:
            raise InternalServerError(
                f"LitTrace MCP health check failed: {result.get('content')}"
            )

    def _latest_workspace(
        self,
        session_id: str,
        *,
        fallback: LiteratureWorkspace,
    ) -> LiteratureWorkspace:
        loader = getattr(self.state_store, "get_session_state", None)
        if not callable(loader):
            return fallback
        state = loader(session_id)
        if state is None:
            return fallback
        return LiteratureWorkspace.model_validate(state.workspace_json)


def _runtime_version(client: AppServerClient) -> str | None:
    user_agent = client.initialize_result.get("userAgent")
    if isinstance(user_agent, str) and user_agent:
        return user_agent
    for line in client.stderr_tail:
        line = line.strip()
        if "codex" in line.lower():
            return line
    return None


async def _start_review_through(
    client: AppServerClient,
    thread_id: str,
    target: dict[str, Any] | None,
    on_review_complete,
) -> AppServerTurnResult:
    """Install the per-thread review-complete callback and start the review.

    Round 8 step 3 helper: the callback needs to land on the
    SAME client instance the App Server is pushing notifications
    to. ``manager.use`` already guarantees that (it hands the
    active client to the lambda), so we install the hook
    before calling ``start_review``.
    """
    client.set_review_complete_callback(thread_id, on_review_complete)
    return await client.start_review(thread_id, target=target)


def _prepend_path(value: str, existing: str | None) -> str:
    return os.pathsep.join(part for part in (value, existing) if part)


@asynccontextmanager
async def _agent_binding_lock(state_store: StateStore, session_id: str):
    factory = getattr(state_store, "agent_thread_lock", None)
    manager = factory(session_id) if callable(factory) else nullcontext()
    await asyncio.to_thread(manager.__enter__)
    try:
        yield
    finally:
        await asyncio.to_thread(manager.__exit__, None, None, None)
