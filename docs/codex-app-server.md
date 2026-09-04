# Codex App Server runtime

LitTrace is migrating its conversational execution layer to the official
`codex app-server` JSONL protocol. The integration is a strangler: LitTrace
keeps ownership of scientific state and domain writes while Codex owns thread
history, turn execution, streaming, and tool orchestration.

## Current migration boundary

- Read-only questions and evidence synthesis can run through App Server.
- Download selection, search-only, download-only, parse-only, and table-only requests run
  through the idempotent `set_download_selection`, `search_papers`,
  `enqueue_download`, `enqueue_parse`, and `enqueue_table_extraction` MCP
  commands. All require the current workspace revision and a stable
  idempotency key.
- `enqueue_download` commits the workspace CAS, audit event, replay result, and
  immutable `download_job` row in one Postgres transaction. It never performs
  network or filesystem work in the App Server process.
- A separate LitTrace worker claims `download_job` rows with `SKIP LOCKED` and
  an expiring lease, executes the frozen paper snapshot, and records completion
  or exponential-backoff failure on the same row. Run one batch with
  `littrace jobs download run`, or run the process continuously with
  `littrace jobs download daemon`.
- `enqueue_parse` freezes paper metadata plus the registered PDF artifact
  reference and checksum. Its worker materializes bytes from object storage,
  verifies the SHA-256, runs Docling/OCR, and atomically merges results into the
  latest Postgres workspace. Changed PDFs or papers removed from the active
  context are reported as stale instead of overwriting newer state. Run it with
  `littrace jobs parse run` or `littrace jobs parse daemon`.
- Table extraction freezes a hash of every parsed source. Its worker preserves
  cells belonging to unrelated papers, replaces only the requested papers,
  and rejects results when their parsed source changed. Run it with
  `littrace jobs table run` or `littrace jobs table daemon`.
- Search combined with download or later processing, storyline generation,
  document generation, and autonomous review continue through the legacy
  LitTrace coordinator.
- `session_state.workspace_json` in Postgres is canonical. A Codex thread is
  execution memory, not a second workspace database.
- `agent_thread_bindings` maps one LitTrace session to one durable Codex thread.
  A Postgres advisory lock permits only one active turn for that session.
- Codex runs in a session-specific scratch directory with `sandbox=read-only`
  and `approvalPolicy=on-request`. LitTrace automatically approves only its
  own empty MCP tool-approval form; shell and filesystem approvals remain
  denied. Domain writes happen only through allowlisted MCP commands.
- App Server defaults to a dedicated LitTrace `CODEX_HOME`; global MCP servers,
  skills, threads, and Codex memories are not inherited.
- App Server starts a required LitTrace MCP process. It exposes only:
  `get_workspace_context`, `get_download_jobs`, `get_parse_jobs`,
  `get_table_jobs`, `search_workspace_rag`, `get_paper_status`, `get_evidence`,
  `quality_report`, `set_download_selection`, `search_papers`,
  `enqueue_download`, `enqueue_parse`, and `enqueue_table_extraction`.
- Every tool call resolves App Server's `_meta.threadId` back to the binding and
  reloads the current Postgres workspace. The gateway MCP process has no
  module-level workspace; its five mutations can only commit through the
  Postgres CAS/idempotency transaction.
- One App Server process is reused across sessions and across Desktop calls
  that each create a fresh asyncio loop. A dedicated daemon thread owns its
  durable event loop. Transport failures are never automatically retried
  because an MCP write may already have committed; the next call starts a clean
  process instead.
- The HTTP chat route does not save the workspace a second time after an App
  Server turn. MCP commands already own their Postgres transaction, while a
  read-only turn has no domain state to persist. Legacy responses retain the
  existing route-level save during migration.

## Enable the runtime

The default remains `legacy`. Configure:

Install the App Server and Postgres extras in the project environment first:

```text
pip install -e ".[mcp,rag]"
```

```yaml
agent_runtime:
  mode: codex_app_server
  codex_command: [codex, app-server]
  codex_home_mode: isolated
  codex_home: ./data/codex-home
  fallback_to_legacy: true
```

Equivalent environment flag:

```text
LITTRACE_AGENT_RUNTIME=codex_app_server
```

The isolated home needs its own one-time Codex login. On PowerShell:

```powershell
New-Item -ItemType Directory -Force .\data\codex-home | Out-Null
$env:CODEX_HOME = (Resolve-Path .\data\codex-home).Path
codex login
Remove-Item Env:CODEX_HOME
```

LitTrace never copies credentials from or edits the user's global Codex home.
For a temporary migration only, `codex_home_mode: shared` restores the old
inheritance behavior.

The real-process smoke check prints only sanitized booleans, server names, and
tool names (never account details or tokens):

```text
python scripts/smoke_codex_app_server.py --codex-home ./data/codex-home-smoke --check-littrace-mcp
```

Raw Codex config overrides can be passed without editing the user's global
configuration:

```yaml
agent_runtime:
  codex_config_overrides:
    service_tier: '"fast"'
```

Overrides are supplied as `codex app-server -c ...`; LitTrace does not modify
either the global Codex config or the managed home's `config.toml`.

## Failure behavior

With `fallback_to_legacy: true`, process startup, protocol, MCP, binding, or
configuration failures fall back to the existing coordinator and append a
`codex_app_server_fallback` warning to the response. Set it to `false` while
developing the integration to surface the original exception.

If the canonical workspace revision advanced before the failure, LitTrace does
not replay the request through legacy code: an MCP mutation may already have
committed. It returns the new Postgres workspace with a
`codex_app_server_post_commit_failure` warning, preserving exactly-once command
semantics even when response generation is interrupted.

Postgres connections use a bounded timeout (`metadata_store.connect_timeout_seconds`,
default 5 seconds), so an unavailable state database cannot indefinitely block
the full-duplex App Server connection.

## Deliberately deferred

- Durable commands/workers for storyline, document generation, and autonomous
  review.
- Streaming App Server deltas into the current HTTP/Desktop UI. ✅ done in Round 6
  via `POST /chat/stream` (text/event-stream). See "Streaming chat"
  below for the SSE protocol contract.
- Cancellation/steering UI and a multi-process pool for higher concurrency.
- Replacing the legacy route-level save with an explicit asynchronous
  filesystem/materialized-view projector after all domain writes use MCP
  commands.
- Treating Codex memory as scientific truth. Long-term facts remain in LitTrace
  state and RAG; the dedicated Codex home may retain thread execution context
  and Codex's own memory files, but LitTrace does not read them as canonical
  scientific state.

## Streaming chat (Round 6)

`POST /chat` is the buffered JSON variant. `POST /chat/stream` is its
SSE counterpart and is the recommended path for the desktop GUI and
the future Web UI so a multi-thousand-word literature review does
not appear all at once after a 30-second wait.

### Endpoint

```
POST /chat/stream
Content-Type: application/json
X-LitTrace-Session-Id: <optional; same as request.session_id>

{"message": "...", "session_id": "..."}
```

Response:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
Connection: keep-alive
```

### Event protocol

Each event is two lines plus a blank line terminator. The `event:`
line carries the SSE event name; the `data:` line carries a JSON
payload.

```
event: delta
data: {"delta": "好的，按灵敏度排序"}

event: delta
data: {"delta": "我推荐这 5 篇："}

event: done
data: {"reply": "完整的 reply 文本",
       "action": "codex_app_server_chat",
       "session_id": "...",
       "session_root": "...",
       "workspace": {...WorkspaceSummary...}}
```

Event types:

  * `delta` — one per `item/agentMessage/delta` frame received from
    the App Server. `data.delta` is the raw text fragment.
  * `done` — terminal event. `data` is a full `ChatResponse` so
    clients that only consume the SSE stream can render the final
    state without a follow-up `GET /sessions/.../messages` call.
  * `error` — emitted when the chat service raises mid-stream.
    `data.code` is a stable identifier
    (`codex_app_server_chat_failed`) and `data.message` carries the
    exception's class name plus message.

### Compatibility

`POST /chat` is unchanged. Clients that cannot consume SSE keep the
buffered JSON response, including the full workspace summary. The
desktop GUI migrates first; CLI users stay on `/chat`.

### Cancellation

Closing the HTTP response (browser tab close, `EventSource.close()`)
propagates to the chat task and cancels the underlying App Server
turn, matching the behaviour of the legacy `/chat` route's
`asyncio.Event` cancellation channel.

## Mid-turn steering and review (Round 8)

Beyond the buffered `POST /chat` and SSE `POST /chat/stream`
endpoints, LitTrace exposes the three additional codex-harness
turn-lifecycle surfaces the desktop GUI and the future Web UI
need to redirect or inspect an in-flight turn.

### `POST /chat/steer`

Append user input to a running regular turn without cancelling
it. Implemented on top of the codex-harness `turn/steer` RPC
([openai/codex#10821](https://github.com/openai/codex/pull/10821)).

Request:

```json
{
  "turn_id": "turn-existing-1",
  "text": "actually focus on the failing tests first",
  "client_user_message_id": "client-msg-r8-1",
  "session_id": "r8-smoke-steer"
}
```

Response:

```json
{
  "turn_id": "turn-existing-1",
  "thread_id": "thr-r8",
  "client_user_message_id": "client-msg-r8-1",
  "session_id": "r8-smoke-steer"
}
```

The server rejects steering against review or manual
compaction turns; the route surfaces the
`active_turn_not_steerable` error so the client can render an
actionable message instead of a generic 5xx.

### `POST /chat/review`

Kick off Codex's automated reviewer for the session's thread.
Implemented on top of the `review/start` RPC. The service layer
installs a per-thread `on_review_complete` callback BEFORE the
RPC fires so the reader loop's `item/completed` /
`exitedReviewMode` handler can flip the UI to "review complete"
without parsing the raw item stream.

Request:

```json
{
  "target": { "type": "commit", "sha": "deadbeef" },
  "session_id": "r8-smoke-review"
}
```

Response:

```json
{
  "turn_id": "turn-review-r8",
  "status": "completed",
  "review_text": "verdict: ship it",
  "exit_item": { "type": "exitedReviewMode", "id": "r1" },
  "session_id": "r8-smoke-review"
}
```

`review_text` is the joined deltas of the review's final
`agentMessage`. `exit_item` is the raw `exitedReviewMode` item
so a future CLI pipeline can inspect sub-fields without
re-parsing the assistant text.

### `POST /chat/{turn_id}/cancel`

Cancel an in-flight turn and stamp the reason on the binding.
Replaces the implicit cancellation channel on `POST /chat` with
a structured reason that operators can later inspect in
`agent_thread_bindings.last_error`.

Request:

```json
{
  "reason": "user_pressed_esc",
  "session_id": "r8-smoke-cancel"
}
```

Response:

```json
{
  "turn_id": "turn-abc",
  "reason": "user_pressed_esc",
  "acknowledged": true,
  "session_id": "r8-smoke-cancel"
}
```

`acknowledged` is `true` if the App Server replied to
`turn/interrupt` with a terminal event within the grace window,
`false` if the request failed at the transport layer (in which
case the caller should treat the turn as terminated regardless).

### Compatibility

All three routes are additive. Existing `POST /chat` and
`POST /chat/stream` clients continue to work unchanged. The
`turn/steer` and `review/start` calls require the App Server
to be configured for `codex app-server` mode
(`LITTRACE_AGENT_RUNTIME=codex_app_server`); the legacy
coordinator rejects them with 503.
