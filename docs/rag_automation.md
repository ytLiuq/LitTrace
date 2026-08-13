# LitTrace RAG Automation

LitTrace keeps RAG state isolated by `session_id`. Each session owns a
`workspace/rag/profile.json`, and pgvector tables are derived from that profile.

## One-Off Daily Job

Run all configured Sentinel watchlists, then refresh existing session RAG
indexes:

```bash
littrace rag daily
```

The job order is:

1. Run each `sessions/sentinel/<watchlist_id>/watchlist.yaml`.
2. Let Sentinel search source APIs, resolve full text, download open-access
   PDFs, parse files, and refresh the watchlist RAG index.
3. Refresh regular chat-session RAG indexes from already parsed workspace
   documents.

Use this command from cron, launchd, or another scheduler.

## Queue Operations

Inspect the embedding queue:

```bash
littrace rag jobs status
littrace rag jobs status --session SESSION_ID
littrace rag jobs status --status dead --limit 50
```

Run pending embedding jobs once:

```bash
littrace rag jobs run --limit 20
```

Requeue dead-letter embedding jobs after fixing the underlying cause:

```bash
littrace rag jobs requeue-dead --session SESSION_ID --limit 20
```

Jobs are claimed with a worker lease and `FOR UPDATE SKIP LOCKED`, so multiple
workers can run concurrently. Expired `running` leases become reclaimable.
Failures below the retry limit remain `failed` with `next_attempt_at`; failures
at or above the retry limit move to `dead` and require explicit requeue.

## Dependency Doctor

Check whether the configured metadata DB, pgvector DB, artifact store, and
embedding endpoint settings are ready:

```bash
littrace rag doctor
```

The doctor does not download papers. It is safe to run before real external E2E
tests or from deployment checks.

## Session Health Metrics

LitTrace tracks four session-level metrics that map directly to the product
goal: turning a research background into a fresh, searchable research memory.

Inspect them from the CLI:

```bash
littrace metrics session --session SESSION_ID
```

Or from the API:

```http
GET /sessions/{session_id}/metrics
X-LitTrace-Session-Id: {session_id}
```

The report includes:

```text
Discovery: 今日相关新增 N 篇
Acquisition: PDF 获取率 X%
RAG: freshness X%，stale N
Consistency: pass X%，missing N
```

`readiness` is the user-facing status:

```text
not_ready -> search_ready -> pdf_ready -> rag_ready -> analysis_ready
```

Discovery is calculated from the session's source-retrieval ledger: a paper is
counted once on the UTC day on which it was first recorded. A missing ledger is
reported as `not_measured`, never as a misleading zero. RAG freshness matches
the current artifact checksum against completed embedding jobs; `stale` is the
number of available embeddable artifacts without a current completed job.

## Long-Running Worker

For a simple local worker:

```bash
littrace rag daemon --interval-hours 24
```

Disable the startup run when a supervisor already handles the first run:

```bash
littrace rag daemon --interval-hours 24 --no-immediate-run
```

## Repo Script

The repository includes a small wrapper script:

```bash
./scripts/run_rag_daily.sh
```

Override paths with environment variables:

```bash
LITTRACE_VENV_BIN=/custom/.venv/bin LITTRACE_LOG_DIR=/custom/logs ./scripts/run_rag_daily.sh
```

## Cron Example

Run every day at 02:15:

```cron
15 2 * * * cd /path/to/LitTrace && .venv/bin/littrace rag daily >> logs/rag-daily.log 2>&1
```

## macOS launchd Example

Save as `~/Library/LaunchAgents/com.littrace.rag.daily.plist` and adjust paths:

The repo ships a ready-made template at `scripts/com.littrace.rag.daily.plist`.
Either edit that file in place or copy it into `~/Library/LaunchAgents/`.

Load it with:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.littrace.rag.daily.plist
```

## PostgreSQL Requirements

Start PostgreSQL with pgvector:

```bash
docker compose -f docker-compose.rag.yml up -d
```

Enable RAG and configure pgvector:

```yaml
rag:
  enabled: true
  backend: "pgvector"
  postgres_dsn: "postgresql://littrace:littrace@localhost:5433/littrace"
  schema_name: "littrace_rag"
  embedding_base_url: "https://api.openai.com/v1"
  embedding_api_key: "..."
```

The application creates the pgvector extension, schema, per-session table, and
HNSW vector index when it refreshes a session.
