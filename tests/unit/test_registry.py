"""Artifact registry test — real Postgres + real schema + real upsert/list.

The factory is exercised end-to-end: the returned ``PostgresArtifactRegistry``
actually connects to the real Postgres, runs ``ensure_schema`` to create
the artifacts table, upserts a record, reads it back, and confirms the
schema_name is the one requested.
"""

from __future__ import annotations

import uuid

import pytest

from littrace.artifact_registry import (
    ArtifactRecord,
    PostgresArtifactRegistry,
    artifact_registry_from_config,
)
from littrace.artifact_store import BlobRef
from littrace.config import LitTraceConfig, MetadataStoreConfig


pytestmark = pytest.mark.unit


_REAL_DSN = "postgresql://littrace:littrace@localhost:5433/littrace"


def test_artifact_registry_factory_builds_postgres_registry():
    schema = f"littrace_test_reg_{uuid.uuid4().hex[:8]}"
    config = LitTraceConfig(
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn=_REAL_DSN,
            schema_name=schema,
        )
    )

    registry = artifact_registry_from_config(config)

    assert isinstance(registry, PostgresArtifactRegistry)
    assert registry.schema_name == schema

    # Real SQL: ensure_schema creates the artifacts table, upsert writes
    # a real row, list_for_session reads it back.
    registry._ensure_schema()
    record = ArtifactRecord.from_blob_ref(
        BlobRef(
            backend="local",
            object_key="sessions/s1/papers/p1/paper.pdf",
            content_type="application/pdf",
        ),
        artifact_id="paper_pdf:p1",
        session_id="s1",
        kind="paper_pdf",
        paper_id="p1",
    )
    registry.upsert(record)
    persisted = registry.list_for_session(session_id="s1")
    assert len(persisted) == 1
    assert persisted[0].artifact_id == "paper_pdf:p1"
    assert persisted[0].object_key == record.object_key
