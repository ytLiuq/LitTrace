# LitTrace Storage Architecture

## Decision

LitTrace should use:

- Postgres for metadata, permissions, workflow state, chat messages, and memory.
- MinIO or another S3-compatible object store for durable artifacts.
- pgvector for user/session-scoped RAG indexes.
- The local filesystem only as a cache, export target, or local backend.

This makes local development possible while keeping the product path ready for
multi-user isolation and remote execution.

## Boundaries

Postgres owns small structured state:

- users, sessions, membership, and permissions
- messages and memory records
- paper metadata and source identities
- artifacts and object references
- download_task, parse_job, embedding_job, export_job
- trace and quality report summaries

MinIO owns durable blobs:

- paper PDFs
- supplementary files
- OCR outputs and parser intermediates
- structured document JSON
- workspace snapshots
- reports, exports, and benchmark traces

pgvector owns derived retrieval indexes:

- document chunks
- embeddings
- embedding model metadata
- chunk provenance back to artifact_id, revision, content_sha256, page, and section

## Object Keys

Durable artifacts are scoped by user and session:

```text
users/{user_id}/sessions/{session_id}/papers/{paper_id}/paper.pdf
users/{user_id}/sessions/{session_id}/papers/{paper_id}/supplementary/{filename}
users/{user_id}/sessions/{session_id}/structured_documents/{paper_id}.json
users/{user_id}/sessions/{session_id}/workspace/snapshots/{revision}.json
users/{user_id}/sessions/{session_id}/artifacts/{artifact_id}/{filename}
users/{user_id}/sessions/{session_id}/traces/{run_id}.jsonl
```

The prefix is an organization and tenancy boundary, not the authorization
system. Downloads must still go through database permission checks.

## Download Task Meaning

`download_task` tracks movement from an external source into object storage.
It should not mean "a local temp file exists."

Recommended status flow:

```text
planned
queued
fetching_source
auth_required
source_resolved
downloading
uploading_to_object_storage
stored
verified
failed
cancelled
```

A task is successful only after:

```text
source bytes fetched
-> object uploaded to MinIO
-> sha256/size verified
-> artifacts row committed
-> download_task points to artifact_id
```

Core fields:

```text
task_id
user_id
session_id
paper_id
source_name
source_url
doi
access_type
requires_login
status
attempt_count
last_error
target_bucket
target_object_key
artifact_id
sha256
size_bytes
created_at
updated_at
completed_at
```

## RAG Fit

RAG should be a derived index, not the source of truth.

Source of truth:

```text
MinIO artifact: PDF / structured_document / OCR JSON
Postgres metadata: paper / artifact / parse_job
```

Derived index:

```text
document_chunks
document_embeddings
```

Every embedding row must include:

```text
user_id
session_id
artifact_id
source_revision
content_sha256
embedding_model
```

Session-scoped retrieval must always filter by user and session:

```sql
WHERE user_id = :user_id
  AND session_id = :session_id
ORDER BY embedding <-> :query_embedding
LIMIT :top_k
```

Later, broader scopes can be added explicitly:

```text
session
project
user_library
team_shared
```

The first implementation should default to `session`.

## Current Compatibility Phase

The codebase now has configuration and artifact reference primitives for this
architecture, while preserving the existing local JSON/JSONL workflow.

`LocalArtifactStore` is not a new durable product storage decision. It is the
local implementation of the object-store interface, used for development,
tests, offline runs, and local cache/export compatibility. Production should
switch `object_store.backend` to `s3` with a MinIO endpoint or another
S3-compatible provider.

Default behavior:

```text
metadata_store.backend = local_json
object_store.backend = local
rag.enabled = false
```

Production-oriented configuration:

```yaml
metadata_store:
  backend: "postgres"
  postgres_dsn: "postgresql://littrace:littrace@localhost:5432/littrace"

object_store:
  backend: "s3"
  bucket: "littrace"
  endpoint_url: "http://127.0.0.1:9000"
  region: "us-east-1"

rag:
  enabled: true
  backend: "pgvector"
  postgres_dsn: "postgresql://littrace:littrace@localhost:5432/littrace"
  scope: "session"
```

## Retry Worker

Download failures should be stored as task state and retried by a background
worker. The compatibility phase uses `data/metadata/download_tasks.json` as a
local stand-in for the future Postgres `download_tasks` table.

The retry worker does not know whether the store is JSON or Postgres. It only
needs:

```text
list_retryable(limit)
upsert(task)
get(task_id)
```

This lets production replace the local store with a Postgres-backed store
without rewriting the retry loop.

The worker should not retry `auth_required` tasks automatically. Those need a
user/institution login handoff first. Normal source/network/upload failures can
move to `failed`, set `retry_after`, and be retried until `max_attempts`.

## Artifact Registry

Object storage writes must also create an artifact metadata row. The code has a
local JSON registry for compatibility and a Postgres registry for production.

The registry is scoped by:

```text
user_id
session_id
artifact_id
```

This matters because a natural artifact id like `paper_pdf:p1` may repeat in
different sessions.

Downloads should use:

```text
GET /artifacts/{artifact_id}/download-link?user_id=...&session_id=...
```

The route checks the artifact registry first, then returns a local URI or signed
object-store URL. Clients should not construct MinIO/S3 object keys directly.

Next implementation steps:

1. Add Postgres migrations for sessions, messages, memory_records, papers,
   artifacts, download_tasks, parse_jobs, document_chunks, and embeddings.
2. Add the S3/MinIO `ArtifactStore` adapter.
3. Move PDF download completion to "stored in object store + artifact row committed."
4. Add embedding jobs that update pgvector from structured document artifacts.
5. Expose download endpoints that authorize by artifact rows, then stream or
   return signed URLs.
