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
