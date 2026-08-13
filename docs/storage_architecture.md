# LitTrace Storage Architecture

## Decision

LitTrace should use:

- Postgres for metadata, workflow state, chat messages, and memory.
- MinIO or another S3-compatible object store for durable artifacts.
- pgvector for session-scoped RAG indexes.
- The local filesystem only as a cache, export target, or local backend.

This makes local development possible while keeping session-scoped storage
portable to remote execution.

## Boundaries

Postgres owns small structured state:

- sessions
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

Durable artifacts are scoped by session.

```text
sessions/{session_id}/papers/{paper_id}/paper.pdf
sessions/{session_id}/papers/{paper_id}/supplementary/{filename}
sessions/{session_id}/structured_documents/{paper_id}.json
sessions/{session_id}/workspace/snapshots/{revision}.json
sessions/{session_id}/artifacts/{artifact_id}/{filename}
sessions/{session_id}/traces/{run_id}.jsonl
```

The session prefix is the storage and retrieval boundary.

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
session_id
artifact_id
source_revision
content_sha256
embedding_model
```

Session-scoped retrieval must always filter by session.

```sql
WHERE session_id = :session_id
ORDER BY embedding <-> :query_embedding
LIMIT :top_k
```

The first implementation is intentionally limited to `session` scope.

## Required Runtime Configuration

Postgres is the only metadata, session, task, and artifact-registry store.
`LocalArtifactStore` remains available only as a filesystem implementation of
the object-storage interface for development; it is not a metadata fallback.
Production should use an S3-compatible store such as MinIO.

```yaml
metadata_store:
  backend: "postgres"
  postgres_dsn: "postgresql://littrace:littrace@localhost:5432/littrace"

artifact_storage:
  backend: "s3"
  bucket: "littrace"
  endpoint_url: "http://127.0.0.1:9000"
  region: "us-east-1"
```

RAG configuration:

```yaml
rag:
  enabled: true
  backend: "pgvector"
  postgres_dsn: "postgresql://littrace:littrace@localhost:5432/littrace"
  scope: "session"
```

## Operational Guardrails

- API routes reject conflicting route/header session scopes.
- Artifact download URLs require a matching session scope.
- Session deletion removes registered object-store artifacts, artifact registry
  rows, pgvector chunks, and DB session state. Object deletion failures are
  returned in the delete report so operators can retry cleanup.
- Embedding jobs use leases, retry delays, dead-letter state, and explicit
  requeue commands.

## Retry Worker

Download failures should be stored as task state and retried by a background
worker. `download_tasks` is a Postgres table and the sole source for the retry
worker.

The worker should not retry `auth_required` tasks automatically. Those need a
user/institution login handoff first. Normal source/network/upload failures can
move to `failed`, set `retry_after`, and be retried until `max_attempts`.

## Artifact Registry

Object storage writes must also create a Postgres artifact metadata row.

The registry is scoped by:

```text
session_id
artifact_id
```

This matters because a natural artifact id like `paper_pdf:p1` may repeat in
different sessions.

Downloads should use:

```text
GET /artifacts/{artifact_id}/download-link?session_id=...
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
