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

from littrace.codex_runtime.client import AppServerClient, AppServerError
from littrace.codex_runtime.gateway import APP_SERVER_TOOL_NAMES
from littrace.codex_runtime.runtime import (
    CodexAppServerRuntimeManager,
    shared_runtime_manager,
)
from littrace.config import CodexHomeMode, LitTraceConfig
from littrace.models import ChatRequest, ChatResponse, LiteratureWorkspace
from littrace.session import ChatSession
from littrace.state_db import AgentThreadBindingRecord, StateStore, state_store_from_config

DEVELOPER_INSTRUCTIONS = """\
You are the conversational research layer inside LitTrace. LitTrace Postgres state is the
canonical source of scientific truth; Codex thread history is execution memory only.

Use only tools on the `littrace` MCP server for facts and changes involving the active session.
Never infer that a filesystem file is canonical. Do not mutate files or run shell commands.
The currently supported domain mutations are `set_download_selection`, `search_papers`,
`enqueue_download`, `enqueue_parse`, and `enqueue_table_extraction`. Call `get_workspace_context`
immediately before a mutation, pass the returned workspace revision, and use a stable idempotency
key for retries of one intended change. Search only when the user asks for literature discovery; it
atomically replaces the current ranked result workspace. Download, parse, and table extraction
commands only submit durable work; explain that they are queued and use the matching job-status
tool when the user asks for progress. Other mutations must be handled by the legacy LitTrace
domain workflow. Keep evidence and paper identifiers in answers when available.
"""


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
    ) -> tuple[ChatResponse, LiteratureWorkspace]:
        scratch_dir = self._scratch_dir(session.session_id)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        async with _agent_binding_lock(self.state_store, session.session_id):
            if self.runtime_manager is not None:
                turn, latest_workspace = await self.runtime_manager.use(
                    lambda client: self._chat_with_client(
                        client, request, workspace, session, scratch_dir
                    )
                )
            elif self.client_factory is AppServerClient:
                manager = self._shared_runtime_manager()
                turn, latest_workspace = await manager.use(
                    lambda client: self._chat_with_client(
                        client, request, workspace, session, scratch_dir
                    )
                )
            else:
                # Custom factories are intentionally ephemeral.  This keeps
                # unit/integration fakes deterministic and gives embedders an
                # explicit runtime_manager injection point for reuse.
                client = self.client_factory(
                    self._codex_command(),
                    **self._client_options(),
                )
                async with client:
                    turn, latest_workspace = await self._chat_with_client(
                        client, request, workspace, session, scratch_dir
                    )
        reply = turn.reply or "Codex App Server completed the turn without a text response."
        return (
            ChatResponse(
                reply=reply,
                action="codex_app_server_chat",
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
    ):
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
                raise AppServerError("thread/start did not return a thread id")
            binding = self.state_store.upsert_agent_thread_binding(
                AgentThreadBindingRecord(
                    session_id=session.session_id,
                    codex_thread_id=thread_id,
                    runtime_kind=runtime_kind,
                    runtime_version=_runtime_version(client),
                    workspace_revision=workspace.context.filters.workspace_revision,
                )
            )
        thread_id = binding.codex_thread_id
        await self._require_mcp_connection(client, thread_id)
        turn = await client.run_turn(
            thread_id,
            request.message,
            timeout=runtime.turn_timeout_seconds,
        )
        latest_workspace = self._latest_workspace(
            session.session_id,
            fallback=workspace,
        )
        self.state_store.upsert_agent_thread_binding(
            binding.model_copy(
                update={
                    "workspace_revision": latest_workspace.context.filters.workspace_revision,
                    "status": "active",
                    "last_error": None,
                }
            )
        )
        return turn, latest_workspace

    def _client_options(self) -> dict[str, Any]:
        runtime = self.config.agent_runtime
        return {
            "startup_timeout": runtime.startup_timeout_seconds,
            "request_timeout": runtime.request_timeout_seconds,
            "environment": self._codex_environment(),
        }

    def _shared_runtime_manager(self) -> CodexAppServerRuntimeManager:
        command = self._codex_command()
        options = self._client_options()
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
        return {
            "cwd": str(scratch_dir),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "serviceName": "littrace",
            "developerInstructions": DEVELOPER_INSTRUCTIONS,
            "config": {
                "mcp_servers": {
                    runtime.mcp_server_name: self._mcp_server_config(),
                },
            },
        }

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
        return {
            "command": sys.executable,
            "args": ["-m", "littrace.mcp_server"],
            "cwd": str(project_root),
            "env": env,
            "required": True,
            "startup_timeout_sec": self.config.agent_runtime.startup_timeout_seconds,
            "enabled_tools": list(APP_SERVER_TOOL_NAMES),
        }

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
            raise AppServerError(
                "The isolated LitTrace Codex home is not authenticated. "
                f"Set CODEX_HOME={home!s} and run `codex login` once, or explicitly "
                "set agent_runtime.codex_home_mode=shared during migration."
            )
        raise AppServerError("Codex App Server is not authenticated")

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
        if result.get("isError") is True:
            raise AppServerError(f"LitTrace MCP health check failed: {result.get('content')}")

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
        if "codex" in line.lower() and "version" in line.lower():
            return line[-200:]
    return None


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
