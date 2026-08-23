"""Process lifecycle for the long-lived Codex App Server runtime.

Desktop entry points invoke async code with a fresh ``asyncio.run`` for each
message.  An asyncio subprocess therefore cannot be safely cached on any of
those short-lived loops.  This manager owns one daemon thread and one durable
event loop, and marshals App Server work onto that loop from either Desktop or
the API.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Awaitable, Callable, Hashable
from concurrent.futures import Future
from typing import Any, TypeVar

from littrace.codex_runtime.client import AppServerClient, AppServerError

T = TypeVar("T")
ClientOperation = Callable[[AppServerClient], Awaitable[T]]


class CodexAppServerRuntimeManager:
    """Own and reuse one App Server process across caller event loops."""

    def __init__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        client_factory: Callable[..., AppServerClient] = AppServerClient,
        client_options: dict[str, Any] | None = None,
        thread_name: str = "littrace-codex-app-server",
    ) -> None:
        self.command = list(command)
        self.client_factory = client_factory
        self.client_options = dict(client_options or {})
        self.thread_name = thread_name
        self._state_lock = threading.Lock()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client_lock: asyncio.Lock | None = None
        self._client: AppServerClient | None = None
        self._closed = False

    @property
    def running(self) -> bool:
        client = self._client
        return bool(client is not None and _client_is_healthy(client))

    async def use(self, operation: ClientOperation[T]) -> T:
        """Run ``operation`` with the shared client on its owning loop."""

        loop = self._ensure_thread()
        future: Future[T] = asyncio.run_coroutine_threadsafe(
            self._use_on_runtime_loop(operation),
            loop,
        )
        return await asyncio.wrap_future(future)

    def close(self, *, timeout: float = 10.0) -> None:
        """Stop the process and its owning event loop. Safe to call repeatedly."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
        if loop is None or thread is None:
            return
        if threading.current_thread() is thread:
            raise RuntimeError("Runtime manager cannot synchronously close its own thread")
        future = asyncio.run_coroutine_threadsafe(self._close_client(), loop)
        try:
            future.result(timeout=timeout)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=timeout)

    def _ensure_thread(self) -> asyncio.AbstractEventLoop:
        with self._state_lock:
            if self._closed:
                raise AppServerError("Codex App Server runtime manager is closed")
            if self._thread is None:
                self._started.clear()
                self._thread = threading.Thread(
                    target=self._run_event_loop,
                    name=self.thread_name,
                    daemon=True,
                )
                self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise AppServerError("Codex App Server runtime loop did not start")
        loop = self._loop
        if loop is None:
            raise AppServerError("Codex App Server runtime loop is unavailable")
        return loop

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._client_lock = asyncio.Lock()
        self._started.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _use_on_runtime_loop(self, operation: ClientOperation[T]) -> T:
        client = await self._ensure_client()
        try:
            return await operation(client)
        finally:
            # Never retry an interrupted operation: an MCP mutation may have
            # committed before the transport failed.  Dispose an unhealthy
            # process now so only the next caller gets a clean process.
            if not _client_is_healthy(client):
                await self._discard_client(client)

    async def _ensure_client(self) -> AppServerClient:
        lock = self._client_lock
        if lock is None:
            raise AppServerError("Codex App Server runtime loop is not initialized")
        async with lock:
            if self._closed:
                raise AppServerError("Codex App Server runtime manager is closed")
            if self._client is not None and _client_is_healthy(self._client):
                return self._client
            if self._client is not None:
                await self._client.close()
            client = self.client_factory(self.command, **self.client_options)
            await client.start()
            self._client = client
            return client

    async def _discard_client(self, client: AppServerClient) -> None:
        lock = self._client_lock
        if lock is None:
            return
        async with lock:
            if self._client is not client:
                return
            self._client = None
            await client.close()

    async def _close_client(self) -> None:
        lock = self._client_lock
        if lock is None:
            return
        async with lock:
            client, self._client = self._client, None
            if client is not None:
                await client.close()


def _client_is_healthy(client: AppServerClient) -> bool:
    healthy = getattr(client, "healthy", None)
    return bool(healthy if healthy is not None else client.running)


_REGISTRY_LOCK = threading.Lock()
_RUNTIME_MANAGERS: dict[Hashable, CodexAppServerRuntimeManager] = {}


def shared_runtime_manager(
    key: Hashable,
    command: list[str] | tuple[str, ...],
    *,
    client_options: dict[str, Any] | None = None,
) -> CodexAppServerRuntimeManager:
    """Return the process manager for one immutable runtime fingerprint."""

    with _REGISTRY_LOCK:
        manager = _RUNTIME_MANAGERS.get(key)
        if manager is None:
            manager = CodexAppServerRuntimeManager(
                command,
                client_options=client_options,
            )
            _RUNTIME_MANAGERS[key] = manager
        return manager


def shutdown_runtime_managers() -> None:
    """Close every process created by :func:`shared_runtime_manager`."""

    with _REGISTRY_LOCK:
        managers = list(_RUNTIME_MANAGERS.values())
        _RUNTIME_MANAGERS.clear()
    for manager in managers:
        manager.close()


atexit.register(shutdown_runtime_managers)
