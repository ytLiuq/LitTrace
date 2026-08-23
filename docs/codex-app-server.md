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
  and `approvalPolicy=never`. Domain writes happen only through allowlisted MCP
  commands, not shell or filesystem access.
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
- Streaming App Server deltas into the current HTTP/Desktop UI.
- Cancellation/steering UI and a multi-process pool for higher concurrency.
- Replacing the legacy route-level save with an explicit asynchronous
  filesystem/materialized-view projector after all domain writes use MCP
  commands.
- Treating Codex memory as scientific truth. Long-term facts remain in LitTrace
  state and RAG; the dedicated Codex home may retain thread execution context
  and Codex's own memory files, but LitTrace does not read them as canonical
  scientific state.
