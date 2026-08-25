"""Minimal ``codex app-server``-shaped fake for live SSE tests.

Speaks just enough of the JSONL protocol to drive one
``POST /chat/stream`` request end-to-end:

  * ``initialize`` → ``userAgent`` reply
  * ``thread/start`` → ``thread.id`` reply
  * ``mcpServerStatus/list`` → ``connected`` reply
  * ``turn/start`` → ``turn.id`` reply, then 3 ``item/agentMessage/delta``
    frames separated by small ``asyncio.sleep`` ticks so the SSE route
    has time to forward each one, then ``item/completed`` and
    ``turn/completed`` to close the turn.

Optional behaviour controlled via environment so the same script
covers the happy path AND the mid-stream failure path:

  * ``FAKE_CODEX_FAIL_AFTER_DELTAS=1`` — after the first delta, raise
    on the next protocol frame so the SSE route emits an ``error``
    event instead of a ``done`` event.
  * ``FAKE_CODEX_DELTAS`` — comma-separated delta strings. Default:
    ``好的,按灵敏度,排序我推荐 5 篇。``.

The fake is intentionally dumb: it answers one request at a time and
dies after ``turn/completed`` so the runtime manager tears the
process down cleanly. A second ``turn/start`` would fail with
``EOF on stdin``, which is exactly what production behaviour looks
like when a single-turn session exits — and it lets the live test
keep the runtime manager cache simple.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    line = await reader.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8").strip())


def _deltas() -> list[str]:
    raw = os.environ.get("FAKE_CODEX_DELTAS", "好的,按灵敏度,排序我推荐 5 篇。")
    return raw.split(",")


def _should_fail() -> bool:
    return os.environ.get("FAKE_CODEX_FAIL_AFTER_DELTAS", "").strip() == "1"


async def _serve_one_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        # 1) initialize
        init = await _read_message(reader)
        if init is None:
            return
        writer.write(
            (
                json.dumps(
                    {"id": init.get("id"), "result": {"userAgent": "fake-codex/0"}}
                )
                + "\n"
            ).encode("utf-8")
        )
        await writer.drain()

        # 2) initialized notification (no response expected)
        initialized = await _read_message(reader)
        if initialized is None:
            return

        # 3) loop: thread/start, mcpServerStatus/list, turn/start
        while True:
            msg = await _read_message(reader)
            if msg is None:
                return
            method = msg.get("method")
            request_id = msg.get("id")
            if method == "thread/start":
                writer.write(
                    (
                        json.dumps(
                            {
                                "id": request_id,
                                "result": {"thread": {"id": "fake-thread-1"}},
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
            elif method == "mcpServerStatus/list":
                writer.write(
                    (
                        json.dumps(
                            {
                                "id": request_id,
                                "result": {
                                    "data": [
                                        {"name": "littrace", "runtimeStatus": "connected"}
                                    ]
                                },
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
            elif method == "turn/start":
                writer.write(
                    (
                        json.dumps(
                            {
                                "id": request_id,
                                "result": {
                                    "turn": {"id": "fake-turn-1", "status": "inProgress"}
                                },
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()

                # Stream a few deltas. The client invokes on_delta
                # per frame so the SSE route pushes one delta event
                # per chunk.
                for index, delta in enumerate(_deltas()):
                    writer.write(
                        (
                            json.dumps(
                                {
                                    "method": "item/agentMessage/delta",
                                    "params": {
                                        "threadId": "fake-thread-1",
                                        "turnId": "fake-turn-1",
                                        "itemId": f"item-{index}",
                                        "delta": delta,
                                    },
                                }
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    await writer.drain()
                    if _should_fail() and index == 0:
                        # After the first delta, close the stream so
                        # the client's wait_for_completion hits
                        # the EOF / failed branch and ``turn``
                        # surfaces ``status=failed``.
                        writer.close()
                        return
                    await asyncio.sleep(0.05)

                # Close the turn cleanly. ``item/completed`` wins
                # over the joined deltas for the final reply, so
                # the SSE ``done`` event carries this text.
                full_reply = "".join(_deltas())
                writer.write(
                    (
                        json.dumps(
                            {
                                "method": "item/completed",
                                "params": {
                                    "threadId": "fake-thread-1",
                                    "turnId": "fake-turn-1",
                                    "item": {
                                        "type": "agentMessage",
                                        "id": "item-final",
                                        "text": full_reply,
                                    },
                                },
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
                writer.write(
                    (
                        json.dumps(
                            {
                                "method": "turn/completed",
                                "params": {
                                    "threadId": "fake-thread-1",
                                    "turn": {
                                        "id": "fake-turn-1",
                                        "status": "completed",
                                        "items": [],
                                    },
                                },
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                await writer.drain()
                return
            else:
                # Unknown method — just acknowledge so the client
                # does not hang.
                if request_id is not None:
                    writer.write(
                        (
                            json.dumps(
                                {"id": request_id, "result": {"ok": True}}
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                    await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    # We talk over stdin/stdout (the same channel the real
    # ``codex app-server`` uses) so the live test does not have to
    # reach for sockets — LitTrace's ``AppServerClient`` already
    # wires up stdio pipes for ``codex app-server`` subcommands.
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    writer_transport, writer_protocol = await asyncio.get_running_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_running_loop())
    await _serve_one_connection(reader, writer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (BrokenPipeError, ConnectionResetError):
        pass
