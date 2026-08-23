from __future__ import annotations

import asyncio
import threading
from typing import ClassVar

import pytest

from littrace.codex_runtime.client import AppServerError
from littrace.codex_runtime.runtime import CodexAppServerRuntimeManager


class _PersistentFakeClient:
    instances: ClassVar[list[_PersistentFakeClient]] = []

    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.kwargs = kwargs
        self.running = False
        self.start_count = 0
        self.close_count = 0
        type(self).instances.append(self)

    @property
    def healthy(self) -> bool:
        return self.running

    async def start(self):
        self.running = True
        self.start_count += 1
        return {"userAgent": "fake"}

    async def close(self):
        self.running = False
        self.close_count += 1


def test_manager_reuses_one_process_across_asyncio_run_calls() -> None:
    _PersistentFakeClient.instances.clear()
    manager = CodexAppServerRuntimeManager(
        ["codex", "app-server"],
        client_factory=_PersistentFakeClient,
        client_options={"request_timeout": 12.0},
    )

    async def identify(client):
        return id(client), id(asyncio.get_running_loop()), threading.get_ident()

    first = asyncio.run(manager.use(identify))
    second = asyncio.run(manager.use(identify))
    manager.close()

    assert first == second
    assert len(_PersistentFakeClient.instances) == 1
    client = _PersistentFakeClient.instances[0]
    assert client.kwargs["request_timeout"] == 12.0
    assert client.start_count == 1
    assert client.close_count == 1


def test_manager_does_not_retry_an_operation_after_transport_loss() -> None:
    _PersistentFakeClient.instances.clear()
    manager = CodexAppServerRuntimeManager(
        ["codex", "app-server"],
        client_factory=_PersistentFakeClient,
    )
    calls = 0

    async def fail_after_possible_commit(client):
        nonlocal calls
        calls += 1
        client.running = False
        raise AppServerError("transport lost")

    with pytest.raises(AppServerError, match="transport lost"):
        asyncio.run(manager.use(fail_after_possible_commit))

    async def recover(client):
        return id(client)

    asyncio.run(manager.use(recover))
    manager.close()

    assert calls == 1
    assert len(_PersistentFakeClient.instances) == 2
