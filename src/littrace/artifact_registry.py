from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from littrace.artifact_store import BlobRef
from littrace.config import LitTraceConfig


class ArtifactRecord(BaseModel):
    artifact_id: str
    session_id: str
    kind: str
    paper_id: str | None = None
    bucket: str | None = None
    object_key: str
    backend: str
    content_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    revision: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_blob_ref(
        cls,
        ref: BlobRef,
        *,
        artifact_id: str,
        session_id: str,
        kind: str,
        paper_id: str | None = None,
        revision: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "ArtifactRecord":
        return cls(
            artifact_id=artifact_id,
            session_id=session_id,
            kind=kind,
            paper_id=paper_id,
            bucket=ref.bucket,
            object_key=ref.object_key,
            backend=ref.backend,
            content_type=ref.content_type,
            sha256=ref.sha256,
            size_bytes=ref.size_bytes,
            revision=revision,
            metadata=metadata or {},
        )


class ArtifactRegistry(Protocol):
    def upsert(self, record: ArtifactRecord) -> ArtifactRecord:
        ...

    def get(self, artifact_id: str, *, session_id: str) -> ArtifactRecord | None:
        ...

    def find_in_session(self, artifact_id: str, *, session_id: str) -> ArtifactRecord | None:
        ...

    def list_for_session(self, *, session_id: str) -> list[ArtifactRecord]:
        ...

    def delete_for_session(self, *, session_id: str) -> int:
        ...


class PostgresArtifactRegistry:
    def __init__(self, dsn: str, *, schema_name: str = "littrace") -> None:
        self.dsn = dsn
        self.schema_name = _safe_identifier(schema_name)
        self._initialized = False

    def upsert(self, record: ArtifactRecord) -> ArtifactRecord:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema_name}.artifacts (
                    artifact_id, session_id, kind, paper_id, bucket, object_key,
                    backend, content_type, sha256, size_bytes, revision, created_at,
                    updated_at, metadata
                )
                VALUES (
                    %(artifact_id)s, %(session_id)s, %(kind)s, %(paper_id)s,
                    %(bucket)s, %(object_key)s, %(backend)s, %(content_type)s, %(sha256)s,
                    %(size_bytes)s, %(revision)s, %(created_at)s, %(updated_at)s, %(metadata)s
                )
                ON CONFLICT (session_id, artifact_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    kind = EXCLUDED.kind,
                    paper_id = EXCLUDED.paper_id,
                    bucket = EXCLUDED.bucket,
                    object_key = EXCLUDED.object_key,
                    backend = EXCLUDED.backend,
                    content_type = EXCLUDED.content_type,
                    sha256 = EXCLUDED.sha256,
                    size_bytes = EXCLUDED.size_bytes,
                    revision = EXCLUDED.revision,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                """,
                _artifact_row(record),
            )
            conn.commit()
        return record

    def get(self, artifact_id: str, *, session_id: str) -> ArtifactRecord | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT metadata
                FROM {self.schema_name}.artifacts
                WHERE artifact_id = %(artifact_id)s
                  AND session_id = %(session_id)s
                """,
                {"artifact_id": artifact_id, "session_id": session_id},
            ).fetchone()
        if row is None:
            return None
        metadata = row[0]
        if isinstance(metadata, dict) and "_record" in metadata:
            return ArtifactRecord.model_validate(metadata["_record"])
        return None

    def find_in_session(self, artifact_id: str, *, session_id: str) -> ArtifactRecord | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT metadata
                FROM {self.schema_name}.artifacts
                WHERE artifact_id = %(artifact_id)s
                  AND session_id = %(session_id)s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                {"artifact_id": artifact_id, "session_id": session_id},
            ).fetchone()
        if row is None:
            return None
        metadata = row[0]
        if isinstance(metadata, dict) and "_record" in metadata:
            return ArtifactRecord.model_validate(metadata["_record"])
        return None

    def list_for_session(self, *, session_id: str) -> list[ArtifactRecord]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT metadata
                FROM {self.schema_name}.artifacts
                WHERE session_id = %(session_id)s
                ORDER BY updated_at DESC
                """,
                {"session_id": session_id},
            ).fetchall()
        records: list[ArtifactRecord] = []
        for row in rows:
            metadata = row[0]
            if isinstance(metadata, dict) and "_record" in metadata:
                records.append(ArtifactRecord.model_validate(metadata["_record"]))
        return records

    def delete_for_session(self, *, session_id: str) -> int:
        self._ensure_schema()
        with self._connect() as conn:
            result = conn.execute(
                f"""
                DELETE FROM {self.schema_name}.artifacts
                WHERE session_id = %(session_id)s
                """,
                {"session_id": session_id},
            )
            deleted = result.rowcount if result.rowcount >= 0 else 0
            conn.commit()
        return deleted

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.artifacts (
                    artifact_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    paper_id TEXT,
                    bucket TEXT,
                    object_key TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    content_type TEXT,
                    sha256 TEXT,
                    size_bytes BIGINT,
                    revision TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL,
                    PRIMARY KEY (session_id, artifact_id)
                )
                """
            )
            conn.execute(
                f"ALTER TABLE {self.schema_name}.artifacts DROP COLUMN IF EXISTS user_id CASCADE"
            )
            conn.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS artifacts_session_artifact_uidx
                ON {self.schema_name}.artifacts (session_id, artifact_id)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS artifacts_session_idx
                ON {self.schema_name}.artifacts (session_id, updated_at)
                """
            )
            conn.commit()
        self._initialized = True

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Postgres artifact registry requires the optional storage extra: "
                "pip install -e '.[storage]'"
            ) from exc
        return psycopg.connect(self.dsn)


def artifact_registry_from_config(config: LitTraceConfig) -> ArtifactRegistry:
    from littrace.state_db import require_postgres_metadata

    dsn, schema_name = require_postgres_metadata(config.metadata_store)
    return PostgresArtifactRegistry(dsn, schema_name=schema_name)


def _artifact_row(record: ArtifactRecord) -> dict[str, object]:
    row = record.model_dump(mode="json")
    metadata = dict(record.metadata)
    metadata["_record"] = record.model_dump(mode="json")
    try:
        from psycopg.types.json import Jsonb
    except ImportError:
        row["metadata"] = json.dumps(metadata)
    else:
        row["metadata"] = Jsonb(metadata)
    return row


def _safe_identifier(value: str) -> str:
    from littrace.state_db import _safe_identifier as _si
    return _si(value)
