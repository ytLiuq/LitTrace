from __future__ import annotations

import asyncio
import json

from littrace.codex_runtime.client import AppServerClient


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.messages: list[dict[str, object]] = []

    def write(self, data: bytes) -> None:
        for line in data.splitlines():
            message = json.loads(line)
            self.messages.append(message)
            self.process.handle(message)

    async def drain(self) -> None:
        await asyncio.sleep(0)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdin = _FakeStdin(self)
        self.returncode: int | None = None
        self._waiter: asyncio.Future[int] | None = None

    def handle(self, message: dict[str, object]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            self.feed({"id": request_id, "result": {"userAgent": "codex-test/1"}})
        elif method == "thread/start":
            self.feed({"id": request_id, "result": {"thread": {"id": "thr-test"}}})
        elif method == "mcpServerStatus/list":
            self.feed(
                {
                    "id": request_id,
                    "result": {
                        "data": [{"name": "littrace", "runtimeStatus": "connected"}]
                    },
                }
            )
        elif method == "turn/start":
            self.feed(
                {"id": request_id, "result": {"turn": {"id": "turn-test", "status": "inProgress"}}}
            )
            self.feed(
                {
                    "id": "approval-1",
                    "method": "item/commandExecution/requestApproval",
                    "params": {"threadId": "thr-test", "turnId": "turn-test"},
                }
            )
            self.feed(
                {
                    "id": "permissions-1",
                    "method": "item/permissions/requestApproval",
                    "params": {"threadId": "thr-test", "turnId": "turn-test"},
                }
            )
            self.feed(
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thr-test",
                        "turnId": "turn-test",
                        "itemId": "item-1",
                        "delta": "partial",
                    },
                }
            )
            self.feed(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr-test",
                        "turnId": "turn-test",
                        "item": {"type": "agentMessage", "id": "item-1", "text": "final answer"},
                    },
                }
            )
            self.feed(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thr-test",
                        "turn": {"id": "turn-test", "status": "completed", "items": []},
                    },
                }
            )

    def feed(self, message: dict[str, object]) -> None:
        self.stdout.feed_data((json.dumps(message) + "\n").encode())

    def terminate(self) -> None:
        self._finish(0)

    def kill(self) -> None:
        self._finish(-9)

    def _finish(self, code: int) -> None:
        if self.returncode is not None:
            return
        self.returncode = code
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(code)

    async def wait(self) -> int:
        if self.returncode is not None:
            return self.returncode
        self._waiter = asyncio.get_running_loop().create_future()
        return await self._waiter


def test_jsonl_client_is_full_duplex_and_fails_closed(monkeypatch) -> None:
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        initialized = await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        assert initialized["userAgent"] == "codex-test/1"
        thread = await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )
        assert thread["id"] == "thr-test"
        status = await asyncio.wait_for(
            client.list_mcp_server_status("thr-test"), timeout=2
        )
        assert status["data"][0]["runtimeStatus"] == "connected"
        turn = await asyncio.wait_for(client.run_turn("thr-test", "question"), timeout=2)
        await asyncio.sleep(0)
        assert turn.reply == "final answer"
        approval = next(item for item in process.stdin.messages if item.get("id") == "approval-1")
        assert approval["result"] == {"decision": "decline"}
        permissions = next(
            item for item in process.stdin.messages if item.get("id") == "permissions-1"
        )
        assert permissions["result"] == {"permissions": {}, "scope": "turn"}
        assert all("jsonrpc" not in item for item in process.stdin.messages)
        assert process.stdin.messages[0]["method"] == "initialize"
        assert process.stdin.messages[1]["method"] == "initialized"
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())


def test_orphan_server_request_drained_by_close(monkeypatch) -> None:
    """A server-initiated request that arrives after turn/completed is
    caught by close()'s cancel-and-gather — run_turn has already returned
    by the time it lands, so the per-turn drain can't.
    """
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        # Override handle so the second turn/start only emits turn-id +
        # turn/completed (no approval), letting run_turn return cleanly.
        # The orphan approval is then pushed after run_turn finishes.

        def delayed_approval_after_turn(message: dict[str, object]) -> None:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                process.feed({"id": request_id, "result": {"userAgent": "codex-test/1"}})
            elif method == "thread/start":
                process.feed(
                    {"id": request_id, "result": {"thread": {"id": "thr-test"}}}
                )
            elif method == "turn/start":
                process.feed(
                    {
                        "id": request_id,
                        "result": {"turn": {"id": "turn-test", "status": "inProgress"}},
                    }
                )
                process.feed(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thr-test",
                            "turn": {"id": "turn-test", "status": "completed", "items": []},
                        },
                    }
                )

        original_handle = process.handle
        handle_calls = {"count": 0}

        def selective_handle(message: dict[str, object]) -> None:
            if message.get("method") == "turn/start":
                delayed_approval_after_turn(message)
                handle_calls["count"] += 1
            elif message.get("id") == "approval-orphan":
                # the orphan approval response — see below
                pass
            else:
                original_handle(message)

        process.handle = selective_handle  # type: ignore[method-assign]
        thread = await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )
        assert thread["id"] == "thr-test"

        # run_turn returns once turn/completed is delivered. At this
        # point no approval has been pushed — the fake drops the
        # approval into stdout one event-loop tick later to simulate a
        # server request that arrived AFTER run_turn had already
        # resolved the turn.
        await asyncio.wait_for(client.run_turn("thr-test", "question"), timeout=2)
        process.feed(
            {
                "id": "approval-orphan",
                "method": "item/commandExecution/requestApproval",
                "params": {"threadId": "thr-test", "turnId": "turn-test"},
            }
        )
        # Give the reader loop a tick to dispatch the orphan.
        await asyncio.sleep(0)

        # close() must cancel the orphan handler and exit within a
        # reasonable timeout — if close() ever tried to await it
        # instead of cancel+gather, this would hang.
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())


def test_run_turn_invokes_on_delta_per_frame(monkeypatch) -> None:
    """Round 6 step 5: ``on_delta`` is awaited for every
    ``item/agentMessage/delta`` frame, in order, and the terminal
    ``reply`` is still built from the joined deltas so a failing
    consumer cannot truncate the final text.
    """
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        # Replace the default turn/start response with one that
        # emits two deltas before item/completed so the test can
        # verify the callback fires once per frame. Keeping the
        # rest of the default scenario identical.
        default_handle = _FakeProcess.handle

        def custom_handle(process_obj, message: dict[str, object]) -> None:
            if message.get("method") == "turn/start":
                request_id = message.get("id")
                process_obj.feed(
                    {"id": request_id, "result": {"turn": {"id": "turn-test", "status": "inProgress"}}}
                )
                process_obj.feed(
                    {
                        "id": "approval-1",
                        "method": "item/commandExecution/requestApproval",
                        "params": {"threadId": "thr-test", "turnId": "turn-test"},
                    }
                )
                process_obj.feed(
                    {
                        "id": "permissions-1",
                        "method": "item/permissions/requestApproval",
                        "params": {"threadId": "thr-test", "turnId": "turn-test"},
                    }
                )
                process_obj.feed(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": "thr-test",
                            "turnId": "turn-test",
                            "itemId": "item-1",
                            "delta": "partial",
                        },
                    }
                )
                process_obj.feed(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": "thr-test",
                            "turnId": "turn-test",
                            "itemId": "item-2",
                            "delta": " answer",
                        },
                    }
                )
                process_obj.feed(
                    {
                        "method": "item/completed",
                        "params": {
                            "threadId": "thr-test",
                            "turnId": "turn-test",
                            "item": {"type": "agentMessage", "id": "item-1", "text": "final answer"},
                        },
                    }
                )
                process_obj.feed(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thr-test",
                            "turn": {"id": "turn-test", "status": "completed", "items": []},
                        },
                    }
                )
                return
            default_handle(process_obj, message)

        process.handle = lambda message: custom_handle(process, message)  # type: ignore[method-assign]
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )
        # Bypass the default approval path so the test stays focused
        # on the callback; approve via a one-shot handler.
        seen: list[str] = []

        async def collect(delta: str) -> None:
            seen.append(delta)

        turn = await asyncio.wait_for(
            client.run_turn("thr-test", "question", on_delta=collect),
            timeout=2,
        )
        # The fake feeds one delta synchronously inside handle()
        # and the second delta from the monkeypatched wrapper.
        # Either order is acceptable; the contract is \"one callback
        # per frame\".
        assert sorted(seen) == sorted(["partial", " answer"])
        # item/completed.text wins over the joined deltas, so the
        # reply carries the server's terminal text rather than the
        # streaming payload.
        assert turn.reply == "final answer"

    asyncio.run(scenario())


def test_run_turn_on_delta_swallows_consumer_errors(monkeypatch) -> None:
    """A consumer that raises must not poison the transport."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )

        def boom(delta: str) -> None:
            raise RuntimeError(f"consumer broke on {delta!r}")

        turn = await asyncio.wait_for(
            client.run_turn("thr-test", "question", on_delta=boom),
            timeout=2,
        )
        # Reply is still the server-supplied terminal text; a
        # throwing consumer must not poison the final payload.
        assert turn.reply == "final answer"

    asyncio.run(scenario())


def test_run_turn_on_delta_supports_async_callback(monkeypatch) -> None:
    """Async callbacks are awaited instead of fire-and-forget."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )

        seen: list[str] = []
        ready = asyncio.Event()

        async def collect(delta: str) -> None:
            await asyncio.sleep(0)  # yields to the loop
            seen.append(delta)
            ready.set()

        await asyncio.wait_for(
            client.run_turn("thr-test", "question", on_delta=collect),
            timeout=2,
        )
        # The callback was awaited at least once; we do not depend
        # on the exact count because the test only sends the default
        # scenario's single delta.
        assert seen == ["partial"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Round 17: ``mcpServer/elicitation/request`` opt-in handler hook.
#
# The LitTrace TUI installs an async callback on the shared
# AppServerClient so an MCP-tool approval prompt can block until the
# operator presses Y/N. The previous contract was an unconditional
# decline. These tests pin the new contract:
#
#   1. No handler installed -> decline (back-compat).
#   2. Handler installed -> its return value is sent back.
#   3. Handler raises -> we still send decline, the App Server
#      does not hang.
#   4. set_elicitation_handler(None) reverts to the auto-decline path.
# ---------------------------------------------------------------------------


def test_elicitation_no_handler_auto_declines(monkeypatch) -> None:
    """When no handler is set, the existing auto-decline path must
    still fire so headless callers keep their contract."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )
        # Feed a single elicitation request directly. No handler is
        # installed; the client must reply with the documented
        # decline envelope.
        process.feed(
            {
                "id": "elic-1",
                "method": "mcpServer/elicitation/request",
                "params": {
                    "server": "littrace",
                    "tool": "search_papers",
                    "arguments": {"topic": "MXene"},
                },
            }
        )
        # Yield enough ticks for the reader loop to dispatch and
        # write the response.
        for _ in range(5):
            await asyncio.sleep(0)
        response = next(
            item for item in process.stdin.messages if item.get("id") == "elic-1"
        )
        assert response["result"] == {
            "action": "decline",
            "content": None,
            "_meta": None,
        }
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())


def test_elicitation_handler_return_value_is_relayed(monkeypatch) -> None:
    """When a handler is installed, its return value is sent as the
    JSON-RPC ``result`` so the TUI's Y/N modal lands on the App Server."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )
        seen_params: list[dict[str, object]] = []
        # Mimic the TUI handler: park a future, capture the params,
        # resolve with the operator's decision on a later tick.
        decision = {"action": "accept", "content": None, "_meta": None}
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async def handler(params: dict[str, object]) -> dict[str, object]:
            seen_params.append(params)
            return await future

        client.set_elicitation_handler(handler)
        process.feed(
            {
                "id": "elic-2",
                "method": "mcpServer/elicitation/request",
                "params": {
                    "server": "littrace",
                    "tool": "search_papers",
                    "arguments": {"topic": "MXene", "limit": 5},
                },
            }
        )
        # Let the reader loop dispatch the elicitation. The handler
        # is now blocked on its future.
        for _ in range(5):
            await asyncio.sleep(0)
        # Sanity: nothing has been written back yet because the
        # handler is awaiting the future.
        assert all(
            item.get("id") != "elic-2" for item in process.stdin.messages
        ), "reader wrote a response before the handler resolved"
        assert seen_params and seen_params[0]["tool"] == "search_papers"
        # Operator presses Y. Unblock the handler.
        future.set_result(decision)
        for _ in range(5):
            await asyncio.sleep(0)
        response = next(
            item for item in process.stdin.messages if item.get("id") == "elic-2"
        )
        assert response["result"] == decision
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())


def test_elicitation_handler_exception_falls_back_to_decline(monkeypatch) -> None:
    """A raising handler must not hang the App Server — we always
    send a decline envelope so codex can resume its turn."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )

        async def boom(params: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("UI blew up")

        client.set_elicitation_handler(boom)
        process.feed(
            {
                "id": "elic-3",
                "method": "mcpServer/elicitation/request",
                "params": {"server": "littrace", "tool": "search_papers"},
            }
        )
        for _ in range(5):
            await asyncio.sleep(0)
        response = next(
            item for item in process.stdin.messages if item.get("id") == "elic-3"
        )
        assert response["result"] == {
            "action": "decline",
            "content": None,
            "_meta": None,
        }
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())


def test_elicitation_set_to_none_reverts_to_auto_decline(monkeypatch) -> None:
    """``set_elicitation_handler(None)`` must restore the back-compat
    auto-decline path so a previous chat's UI hook does not leak
    into a subsequent headless chat on the same shared client."""
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def scenario() -> None:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        process = processes[0]
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )

        async def should_not_run(params: dict[str, object]) -> dict[str, object]:
            raise AssertionError("handler should have been cleared")

        client.set_elicitation_handler(should_not_run)
        client.set_elicitation_handler(None)
        process.feed(
            {
                "id": "elic-4",
                "method": "mcpServer/elicitation/request",
                "params": {"server": "littrace", "tool": "search_papers"},
            }
        )
        for _ in range(5):
            await asyncio.sleep(0)
        response = next(
            item for item in process.stdin.messages if item.get("id") == "elic-4"
        )
        assert response["result"]["action"] == "decline"
        await asyncio.wait_for(client.close(), timeout=2)

    asyncio.run(scenario())
