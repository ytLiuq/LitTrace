# LitTrace RAG Automation

LitTrace keeps RAG state isolated by `user_id + session_id`. Each session owns a
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
