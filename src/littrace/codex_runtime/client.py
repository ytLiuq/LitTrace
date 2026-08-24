"""Async JSONL client for ``codex app-server``.

App Server uses a JSON-RPC-like full-duplex protocol without the
``jsonrpc: 2.0`` field.  A client must keep reading while requests are in
flight because approval and elicitation requests travel from server to
client on the same stream.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from littrace.codex_runtime.rollout import RolloutRecorder


class AppServerError(RuntimeError):
    """Base failure raised by the App Server transport."""


class AppServerProtocolError(AppServerError):
    """Malformed wire data or a JSON-RPC error response."""


@dataclass(frozen=True)
class AppServerTurnResult:
    thread_id: str
    turn_id: str
    status: str
    reply: str
    turn: dict[str, Any]


class AppServerClient:
    """One long-lived, full-duplex App Server process."""

    def __init__(
        self,
        command: list[str] | tuple[str, ...] = ("codex", "app-server"),
        *,
        startup_timeout: float = 20.0,
        request_timeout: float = 60.0,
        stream_limit: int = 8 * 1024 * 1024,
        environment: dict[str, str] | None = None,
        rollout_recorder: "RolloutRecorder | None" = None,
    ) -> None:
        if not command:
            raise ValueError("App Server command must not be empty")
        self.command = list(command)
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.stream_limit = stream_limit
        self.environment = dict(environment or {})
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._thread_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._turn_request_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._closed = False
        self.initialize_result: dict[str, Any] = {}
        # Optional side-channel log for debugging. None by default so
        # production deployments that do not opt in pay no cost.
        self._rollout_recorder = rollout_recorder

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def healthy(self) -> bool:
        """Whether the process and its response reader can accept more RPCs."""

        return bool(self.running and self._reader_task is not None and not self._reader_task.done())

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    async def start(self) -> dict[str, Any]:
        if self.running:
            return {}
        if self._closed:
            raise AppServerError("App Server client is closed")
        executable = shutil.which(self.command[0]) or self.command[0]
        try:
            self._process = await asyncio.create_subprocess_exec(
                executable,
                *self.command[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self.stream_limit,
                env={**os.environ, **self.environment},
            )
        except OSError as exc:
            raise AppServerError(
                f"Unable to start Codex App Server with {self.command!r}: {exc}"
            ) from exc
        self._reader_task = asyncio.create_task(self._read_stdout(), name="codex-appserver-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="codex-appserver-stderr")
        try:
            initialized = await self.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "littrace",
                        "title": "LitTrace",
                        "version": "0.1.0",
                    }
                },
                timeout=self.startup_timeout,
            )
            self.initialize_result = initialized
            await self.notify("initialized")
            return initialized
        except Exception:
            await self.close()
            raise

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.running:
            raise AppServerError("Codex App Server is not running")
        loop = asyncio.get_running_loop()
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            await self._write(message)
            return await asyncio.wait_for(
                future,
                timeout=self.request_timeout if timeout is None else timeout,
            )
        except TimeoutError as exc:
            raise AppServerError(f"App Server request timed out: {method}") from exc
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def start_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        result = await self.request("thread/start", params)
        return _required_object(result, "thread")

    async def resume_thread(
        self,
        thread_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(overrides or {})
        params["threadId"] = thread_id
        result = await self.request("thread/resume", params)
        return _required_object(result, "thread")

    async def list_mcp_server_status(self, thread_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"detail": "toolsAndAuthOnly"}
        if thread_id is not None:
            params["threadId"] = thread_id
        return await self.request(
            "mcpServerStatus/list",
            params,
        )

    async def call_mcp_tool(
        self,
        thread_id: str,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.request(
            "mcpServer/tool/call",
            {
                "threadId": thread_id,
                "server": server,
                "tool": tool,
                "arguments": arguments or {},
            },
        )

    async def read_account(self, *, refresh_token: bool = False) -> dict[str, Any]:
        """Return the active App Server authentication state."""

        return await self.request(
            "account/read",
            {"refreshToken": refresh_token},
        )

    async def run_turn(
        self,
        thread_id: str,
        text: str,
        *,
        timeout: float = 300.0,
        cancellation: asyncio.Event | None = None,
    ) -> AppServerTurnResult:
        queue = self._thread_queues.setdefault(thread_id, asyncio.Queue())
        self._turn_request_tasks.setdefault(thread_id, set())
        result = await self.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
            },
        )
        turn = _required_object(result, "turn")
        turn_id = _required_string(turn, "id")
        if self._rollout_recorder is not None:
            self._rollout_recorder.append(
                type_="turn_start",
                turn_id=turn_id,
                thread_id=thread_id,
                user_text=text,
            )
        completed_messages: list[str] = []
        deltas: list[str] = []

        async def wait_for_completion() -> AppServerTurnResult:
            while True:
                # select between queue and cancellation event so a caller
                # cancel doesn't have to wait for the next server frame.
                get_task = asyncio.create_task(queue.get())
                if cancellation is not None:
                    cancel_task = asyncio.create_task(cancellation.wait())
                    done, pending = await asyncio.wait(
                        {get_task, cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    if get_task in done:
                        message = get_task.result()
                    else:
                        # cancellation fired — issue graceful interrupt
                        # and wait for the server's terminal event.
                        await self.cancel_current_turn(
                            thread_id, turn_id, grace_seconds=10.0
                        )
                        # Drain whatever the server sent since we last
                        # consumed; the terminal event will arrive within
                        # grace_seconds.
                        try:
                            message = await asyncio.wait_for(
                                queue.get(),
                                timeout=10.0,
                            )
                        except TimeoutError:
                            # Transport died — guaranteed terminal event
                            # is 'failed' so the caller can distinguish
                            # from a clean interrupt.
                            return AppServerTurnResult(
                                thread_id=thread_id,
                                turn_id=turn_id,
                                status="failed",
                                reply="",
                                turn={"id": turn_id, "status": "failed"},
                            )
                else:
                    message = await get_task
                method = message.get("method")
                params = message.get("params") or {}
                if params.get("turnId") != turn_id and (
                    not isinstance(params.get("turn"), dict)
                    or params["turn"].get("id") != turn_id
                ):
                    continue
                if method == "item/agentMessage/delta":
                    delta = params.get("delta")
                    if isinstance(delta, str):
                        deltas.append(delta)
                elif method == "item/completed":
                    item = params.get("item") or {}
                    if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                        completed_messages.append(item["text"])
                elif method == "turn/completed":
                    completed_turn = _required_object(params, "turn")
                    status = str(completed_turn.get("status") or "unknown")
                    reply = "\n".join(part for part in completed_messages if part).strip()
                    if not reply:
                        reply = "".join(deltas).strip()
                    if self._rollout_recorder is not None:
                        self._rollout_recorder.append(
                            type_="turn_complete",
                            turn_id=turn_id,
                            status=status,
                            reply=reply,
                            turn=completed_turn,
                        )
                    if status not in {"completed", "interrupted"}:
                        error = completed_turn.get("error")
                        raise AppServerError(
                            f"Codex turn {turn_id} ended with status {status}: {error}"
                        )
                    return AppServerTurnResult(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status=status,
                        reply=reply,
                        turn=completed_turn,
                    )

        try:
            return await asyncio.wait_for(wait_for_completion(), timeout=timeout)
        except TimeoutError as exc:
            try:
                await self.cancel_current_turn(
                    thread_id, turn_id, grace_seconds=10.0
                )
            except AppServerError:
                pass
            raise AppServerError(f"Codex turn timed out: {turn_id}") from exc
        finally:
            self._thread_queues.pop(thread_id, None)
            pending = self._turn_request_tasks.pop(thread_id, None)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def cancel_current_turn(
        self,
        thread_id: str,
        turn_id: str,
        *,
        grace_seconds: float = 10.0,
    ) -> bool:
        """Issue ``turn/interrupt`` and wait for ``turn/completed``.

        Returns True if the server acknowledged with status='interrupted'
        within ``grace_seconds``; False if the request failed (transport
        died, app server rejected, or no terminal event arrived in time).
        The caller should treat both outcomes as "guaranteed terminal
        event with status='failed' or 'interrupted'" and stop waiting for
        turn completion themselves.
        """
        try:
            await asyncio.wait_for(
                self.request(
                    "turn/interrupt",
                    {"threadId": thread_id, "turnId": turn_id},
                    timeout=grace_seconds,
                ),
                timeout=grace_seconds,
            )
            return True
        except (AppServerError, TimeoutError):
            return False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.returncode is None:
            if process.stdin is not None and hasattr(process.stdin, "close"):
                process.stdin.close()
                wait_closed = getattr(process.stdin, "wait_closed", None)
                if wait_closed is not None:
                    try:
                        await wait_closed()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=4.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        current = asyncio.current_task()
        io_tasks = [
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not current
        ]
        if io_tasks:
            _, pending_io = await asyncio.wait(io_tasks, timeout=1.0)
            for task in pending_io:
                task.cancel()
            await asyncio.gather(*io_tasks, return_exceptions=True)
        request_tasks = [
            task for task in self._request_tasks if task is not current and not task.done()
        ]
        for task in request_tasks:
            task.cancel()
        if request_tasks:
            await asyncio.gather(*request_tasks, return_exceptions=True)
        # CPython's Proactor subprocess transport can otherwise outlive the
        # event loop after a child with stdio grandchildren exits.  Closing
        # the owning transport here prevents noisy unclosed-pipe finalizers.
        transport = getattr(process, "_transport", None) if process is not None else None
        if transport is not None:
            transport.close()
            await asyncio.sleep(0)
        # Close the side-channel rollout recorder last so any
        # pending events queued by the reader still get flushed.
        if self._rollout_recorder is not None:
            self._rollout_recorder.close()
            self._rollout_recorder = None
        self._fail_pending(AppServerError("Codex App Server client closed"))

    async def _write(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise AppServerError("Codex App Server stdin is unavailable")
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            process.stdin.write(data)
            await process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while line := await self._process.stdout.readline():
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AppServerProtocolError(
                        f"Invalid App Server JSONL frame: {line[:200]!r}"
                    ) from exc
                await self._dispatch(message)
            returncode = await self._process.wait()
            if self._closed:
                return
            detail = "\n".join(self._stderr_tail)
            raise AppServerError(
                f"Codex App Server exited with code {returncode}"
                + (f": {detail}" if detail else "")
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - background reader must wake every pending RPC
            self._fail_pending(exc)

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            self._stderr_tail.append(line.decode(errors="replace").rstrip())

    async def _dispatch(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if request_id is not None and method is None:
            future = self._pending.get(request_id)
            if future is None or future.done():
                return
            error = message.get("error")
            if error is not None:
                future.set_exception(AppServerProtocolError(str(error)))
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return
        if request_id is not None and isinstance(method, str):
            params = message.get("params") or {}
            task = asyncio.create_task(
                self._handle_server_request(request_id, method, params)
            )
            # Every handler is added to _request_tasks so close() can
            # cancel it if it has not drained by then. _turn_request_tasks
            # is a stricter per-turn subset used to drain handlers that
            # arrived during the turn — handlers that arrived AFTER the
            # turn completed (no entry in _turn_request_tasks for this
            # thread_id) only live in _request_tasks and are caught by
            # close()'s cancel-and-gather on shutdown.
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)
            thread_id = params.get("threadId")
            if isinstance(thread_id, str):
                turn_tasks = self._turn_request_tasks.get(thread_id)
                if turn_tasks is not None:
                    turn_tasks.add(task)
                    task.add_done_callback(turn_tasks.discard)
            return
        if isinstance(method, str):
            params = message.get("params") or {}
            # Round 4 P1 step 6: history-compaction events bypass
            # the per-thread queue gate. codex-harness fires these
            # notifications outside any in-flight turn and we want
            # the rollout log to record them regardless.
            if method in {"item/compactedHistory", "thread/compacted"}:
                if self._rollout_recorder is not None:
                    self._rollout_recorder.append(
                        type_="compaction",
                        method=method,
                        params=params,
                    )
                return
            thread_id = params.get("threadId")
            if isinstance(thread_id, str) and thread_id in self._thread_queues:
                # Side-channel rollout log: capture the server
                # notification before handing it to the turn consumer.
                # Recorder writes are best-effort and never raise.
                if self._rollout_recorder is not None:
                    self._rollout_recorder.append(
                        type_="event",
                        method=method,
                        params=params,
                    )
                await self._thread_queues[thread_id].put(message)

    # codex-harness exposes a 6-word approval decision vocabulary.
    # LitTrace rejects every approval server-initiated request because
    # every domain mutation is already mediated by the Mcp Gateway's
    # own CAS + idempotency check — letting codex run shell / patch
    # commands would create a second write path. The full vocabulary
    # is wired so a future deployment that wants to allow some
    # decisions can flip a flag without rewriting the handler.
    _APPROVAL_REJECTED_METHODS = frozenset(
        {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "execCommandApproval",
            "applyPatchApproval",
        }
    )

    async def _reply_approval(
        self, request_id: int | str, decision: str
    ) -> None:
        """Reply to a server-initiated approval request with one of
        the codex-harness six-word vocabulary entries
        (``accept`` / ``acceptForSession`` /
        ``acceptWithExecpolicyAmendment`` /
        ``applyNetworkPolicyAmendment`` / ``decline`` / ``cancel``).
        """
        await self._write(
            {"id": request_id, "result": {"decision": decision}}
        )

    async def _handle_server_request(
        self,
        request_id: int | str,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if method in self._APPROVAL_REJECTED_METHODS:
            # LitTrace is read-only at the App Server level. Every
            # mutation goes through the Mcp Gateway's CAS + idempotent
            # write path, so approving any of these would open a
            # second write channel.
            await self._reply_approval(request_id, "decline")
            return
        if method == "mcpServer/elicitation/request":
            # Elicitation has a 3-state vocabulary (accept / decline
            # / cancel), distinct from the 6-word approval set.
            await self._write(
                {
                    "id": request_id,
                    "result": {"action": "decline", "content": None, "_meta": None},
                }
            )
            return
        if method == "item/permissions/requestApproval":
            # Modern codex-harness folds network-policy amendment
            # into the permissions flow. Returning an empty
            # permissions set with scope=turn is the documented
            # "no extra authorization granted" response.
            await self._write(
                {
                    "id": request_id,
                    "result": {
                        "permissions": {},
                        "scope": "turn",
                    },
                }
            )
            return
        if method == "item/tool/requestUserInput":
            answers = {
                str(question.get("id")): {"answers": []}
                for question in params.get("questions", [])
                if question.get("id") is not None
            }
            await self._write({"id": request_id, "result": {"answers": answers}})
            return
        await self._write(
            {
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"LitTrace does not support server request: {method}",
                },
            }
        )

    def _fail_pending(self, exc: Exception) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(exc)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


def _required_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise AppServerProtocolError(f"App Server response is missing object field {key!r}")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise AppServerProtocolError(f"App Server response is missing string field {key!r}")
    return result
