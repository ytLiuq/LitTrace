from littrace.artifact_registry import (
    PostgresArtifactRegistry,
    artifact_registry_from_config,
)
from littrace.config import LitTraceConfig, MetadataStoreConfig


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
