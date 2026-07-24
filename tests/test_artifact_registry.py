from littrace.artifact_registry import (
    ArtifactRecord,
    LocalArtifactRegistry,
    PostgresArtifactRegistry,
    artifact_registry_from_config,
)
from littrace.artifact_store import BlobRef
from littrace.config import LitTraceConfig, MetadataStoreConfig, StorageConfig


def test_local_artifact_registry_scopes_records_by_user_and_session(tmp_path):
    registry = LocalArtifactRegistry(tmp_path / "artifacts.json")
    ref = BlobRef(backend="local", object_key="users/u1/sessions/s1/papers/p1/paper.pdf")

    registry.upsert(
        ArtifactRecord.from_blob_ref(
            ref,
            artifact_id="paper_pdf:p1",
            user_id="u1",
            session_id="s1",
            kind="paper_pdf",
            paper_id="p1",
        )
    )
    registry.upsert(
        ArtifactRecord.from_blob_ref(
            ref.model_copy(update={"object_key": "users/u1/sessions/s2/papers/p1/paper.pdf"}),
            artifact_id="paper_pdf:p1",
            user_id="u1",
            session_id="s2",
            kind="paper_pdf",
            paper_id="p1",
        )
    )

    assert registry.get("paper_pdf:p1", user_id="u1", session_id="s1").session_id == "s1"
    assert registry.get("paper_pdf:p1", user_id="u1", session_id="s2").session_id == "s2"
    assert registry.get("paper_pdf:p1", user_id="u2", session_id="s1") is None


def test_artifact_registry_factory_uses_local_json_by_default(tmp_path):
    config = LitTraceConfig(storage=StorageConfig(metadata_dir=tmp_path))

    registry = artifact_registry_from_config(config)

    assert isinstance(registry, LocalArtifactRegistry)


def test_artifact_registry_factory_builds_postgres_registry():
    config = LitTraceConfig(
        metadata_store=MetadataStoreConfig(
            backend="postgres",
            postgres_dsn="postgresql://example/littrace",
            schema_name="custom_schema",
        )
    )

    registry = artifact_registry_from_config(config)

    assert isinstance(registry, PostgresArtifactRegistry)
    assert registry.schema_name == "custom_schema"
