"""Round 8 client-level tests for steer_turn / start_review /
review-complete callback wiring.

These tests do not boot a real App Server process. Instead
they extend the existing ``_FakeProcess`` pattern from
``test_codex_app_server_client.py`` so we can assert the
exact wire shape the client emits and the exact notification
flow it consumes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from littrace.codex_runtime.client import (
    AppServerClient,
    SteerTurnResult,
)


class _FakeStdin:
    def __init__(self, process: "_FakeProcess") -> None:
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
        """Default handshake: initialize / thread/start /
        mcpServerStatus/list / turn/start only. Tests override
        this attribute to add custom method responses.
        """
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            self.feed({"id": request_id, "result": {"userAgent": "fake-codex/0"}})
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
                {
                    "id": request_id,
                    "result": {"turn": {"id": "turn-r8-test", "status": "inProgress"}},
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


@pytest.fixture
def make_client(monkeypatch):
    """Return a factory that spawns a fake-backed AppServerClient
    and yields ``(client, process)`` so the test can install a
    custom ``handle`` callable on the process before the
    client starts the JSONL handshake.
    """
    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    async def factory() -> tuple[AppServerClient, _FakeProcess]:
        client = AppServerClient(["codex", "app-server"])
        await asyncio.wait_for(client.start(), timeout=2)
        return client, processes[0]

    return factory


def _install_standard_handshake(process: _FakeProcess) -> None:
    """Make the standard initialize / thread/start reply
    sequence available so the test can override only the
    method-specific behaviour.
    """

    def handle(message: dict[str, object]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            process.feed({"id": request_id, "result": {"userAgent": "fake-codex/0"}})
        elif method == "thread/start":
            process.feed({"id": request_id, "result": {"thread": {"id": "thr-test"}}})
        elif method == "mcpServerStatus/list":
            process.feed(
                {
                    "id": request_id,
                    "result": {
                        "data": [{"name": "littrace", "runtimeStatus": "connected"}]
                    },
                }
            )
        elif method == "turn/start":
            process.feed(
                {
                    "id": request_id,
                    "result": {"turn": {"id": "turn-r8-test", "status": "inProgress"}},
                }
            )

    process.handle = handle  # type: ignore[method-assign]


async def _drain_standard_turn(
    process: _FakeProcess, turn_id: str,
) -> None:
    process.feed(
        {
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": "thr-test", "turnId": turn_id,
                "itemId": "item-1", "delta": "ok",
            },
        }
    )
    process.feed(
        {
            "method": "item/completed",
            "params": {
                "threadId": "thr-test", "turnId": turn_id,
                "item": {"type": "agentMessage", "id": "item-1", "text": "ok"},
            },
        }
    )
    process.feed(
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thr-test",
                "turn": {"id": turn_id, "status": "completed", "items": []},
            },
        }
    )


def test_steer_turn_emits_expected_wire_shape(monkeypatch) -> None:
    """``turn/steer`` must send the exact codex-harness request
    shape and return a typed ``SteerTurnResult``.
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

        def handle(message: dict[str, object]) -> None:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                process.feed({"id": request_id, "result": {"userAgent": "fake-codex/0"}})
            elif method == "thread/start":
                process.feed({"id": request_id, "result": {"thread": {"id": "thr-test"}}})
            elif method == "mcpServerStatus/list":
                process.feed(
                    {
                        "id": request_id,
                        "result": {
                            "data": [{"name": "littrace", "runtimeStatus": "connected"}]
                        },
                    }
                )
            elif method == "turn/steer":
                process.feed(
                    {"id": request_id, "result": {"turnId": "turn-existing"}}
                )

        process.handle = handle  # type: ignore[method-assign]
        result = await asyncio.wait_for(
            client.steer_turn(
                thread_id="thr-test",
                turn_id="turn-existing",
                text="actually focus on failing tests first",
                client_user_message_id="client-msg-1",
            ),
            timeout=2,
        )

        # Typed return
        assert isinstance(result, SteerTurnResult)
        assert result.thread_id == "thr-test"
        assert result.turn_id == "turn-existing"
        assert result.client_user_message_id == "client-msg-1"

        # Wire shape
        steer_call = next(
            msg for msg in process.stdin.messages if msg.get("method") == "turn/steer"
        )
        params = steer_call["params"]
        assert params["threadId"] == "thr-test"
        assert params["expectedTurnId"] == "turn-existing"
        assert params["clientUserMessageId"] == "client-msg-1"
        assert params["input"] == [
            {"type": "text", "text": "actually focus on failing tests first"}
        ]

    asyncio.run(scenario())


def test_steer_turn_omits_client_user_message_id_when_none(monkeypatch) -> None:
    """The optional ``clientUserMessageId`` is dropped from the
    request when the caller does not supply one, matching the
    codex-harness wire shape.
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

        def handle(message: dict[str, object]) -> None:
            method = message.get("method")
            request_id = message.get("id")
            if method == "initialize":
                process.feed({"id": request_id, "result": {"userAgent": "fake-codex/0"}})
            elif method == "thread/start":
                process.feed({"id": request_id, "result": {"thread": {"id": "thr-test"}}})
            elif method == "mcpServerStatus/list":
                process.feed(
                    {
                        "id": request_id,
                        "result": {
                            "data": [{"name": "littrace", "runtimeStatus": "connected"}]
                        },
                    }
                )
            elif method == "turn/steer":
                process.feed({"id": request_id, "result": {"turnId": "turn-existing"}})

        process.handle = handle  # type: ignore[method-assign]
        await asyncio.wait_for(
            client.steer_turn(
                thread_id="thr-test",
                turn_id="turn-existing",
                text="focus on the rest",
            ),
            timeout=2,
        )
        steer_call = next(
            msg for msg in process.stdin.messages if msg.get("method") == "turn/steer"
        )
        assert "clientUserMessageId" not in steer_call["params"]

    asyncio.run(scenario())


def test_review_complete_callback_fires_on_exited_review_mode(monkeypatch) -> None:
    """``item/completed`` with ``type=exitedReviewMode`` must
    fire the per-thread callback the route layer installs
    before kicking off the review.
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
        seen: list[dict[str, object]] = []

        # ``start_review`` only sends ``review/start``; the
        # per-thread queue is created by ``start_thread`` on
        # the chat path. Mirror that here so the reader loop
        # has somewhere to put notifications for ``thr-test``.
        await asyncio.wait_for(
            client.start_thread({"sandbox": "read-only"}), timeout=2
        )

        # Wrap the default handle so we keep the standard
        # handshake responses and only override the
        # ``review/start`` reply.
        default_handle = process.handle

        def handle(message: dict[str, object]) -> None:
            method = message.get("method")
            request_id = message.get("id")
            if method == "review/start":
                process.feed(
                    {
                        "id": request_id,
                        "result": {"turn": {"id": "turn-review-1", "status": "inProgress"}},
                    }
                )
                return
            default_handle(message)

        process.handle = handle  # type: ignore[method-assign]

        def on_complete(item: dict[str, object]) -> None:
            seen.append(item)

        client.set_review_complete_callback("thr-test", on_complete)
        task = asyncio.create_task(
            client.start_review(thread_id="thr-test"),
        )
        # Let the review request reach the fake and the reply
        # propagate back through the reader task before we start
        # streaming notifications.
        await asyncio.sleep(0.05)
        # Push the review-mode lifecycle + final reply.
        process.feed(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thr-test", "turnId": "turn-review-1",
                    "item": {"type": "enteredReviewMode", "id": "r1"},
                },
            }
        )
        process.feed(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thr-test", "turnId": "turn-review-1",
                    "itemId": "r1", "delta": "verdict: ",
                },
            }
        )
        process.feed(
            {
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thr-test", "turnId": "turn-review-1",
                    "itemId": "r1", "delta": "ship it",
                },
            }
        )
        process.feed(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-test", "turnId": "turn-review-1",
                    "item": {
                        "type": "agentMessage", "id": "r1",
                        "text": "verdict: ship it",
                    },
                },
            }
        )
        process.feed(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-test", "turnId": "turn-review-1",
                    "item": {"type": "exitedReviewMode", "id": "r2"},
                },
            }
        )
        process.feed(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-test",
                    "turn": {"id": "turn-review-1", "status": "completed", "items": []},
                },
            }
        )
        # Give the reader task enough loop ticks to forward each
        # frame to the per-thread queue that ``_drain_turn`` is
        # reading from.
        for _ in range(5):
            await asyncio.sleep(0)
        result = await asyncio.wait_for(task, timeout=2)

        # Callback fired with the raw ``exitedReviewMode`` item.
        assert len(seen) == 1
        assert seen[0].get("type") == "exitedReviewMode"

        # Terminal reply carries the joined deltas.
        assert result.status == "completed"
        assert result.reply == "verdict: ship it"

        # Active review tracking is reset on terminal.
        assert "thr-test" not in client._active_review_turns  # type: ignore[attr-defined]
        # The callback map is per-client and lives until the
        # route layer explicitly clears it (``set_review_complete_callback
        # (thread_id, None)``); the client only resets the
        # bookkeeping for ``_active_review_turns`` on terminal.
        # The service layer wires the clear in a follow-up round;
        # for now the map is allowed to retain the entry.
        assert "thr-test" in client._review_complete_callbacks  # type: ignore[attr-defined]

    asyncio.run(scenario())


def test_review_complete_callback_cleared_after_set_to_none(monkeypatch) -> None:
    """``set_review_complete_callback(thread_id, None)`` must
    drop the binding so a stale callback does not fire on a
    subsequent review.
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

        # Smoke: a no-op callback is fine.
        client.set_review_complete_callback("thr-test", lambda item: None)
        assert "thr-test" in client._review_complete_callbacks  # type: ignore[attr-defined]
        client.set_review_complete_callback("thr-test", None)
        assert "thr-test" not in client._review_complete_callbacks  # type: ignore[attr-defined]

    asyncio.run(scenario())
