from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.retrieval.rag_profile import RagProfile


class RagChunkRecord(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    embedding: list[float]
    chunk_hash: str
    source_record_id: str | None = None
    section: str | None = None
    page: int | None = None
    table_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    score: float
    chunk_hash: str
    source_record_id: str | None = None
    section: str | None = None
    page: int | None = None
    table_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class PgvectorCollection:
    schema_name: str
    table_name: str
    embedding_dimension: int

    @property
    def qualified_table(self) -> str:
        return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"


@dataclass
class PgvectorRagStore:
    config: LitTraceConfig
    profile: RagProfile
    collection: PgvectorCollection = field(init=False)

    def __post_init__(self) -> None:
        if self.profile.backend != "pgvector" or self.config.rag.backend != "pgvector":
            raise ValueError("PgvectorRagStore requires rag.backend='pgvector'.")
        if not self.config.rag.postgres_dsn:
            raise ValueError("config.rag.postgres_dsn is required for pgvector RAG storage.")
        self.collection = PgvectorCollection(
            schema_name=self.profile.postgres_schema,
            table_name=self.profile.collection_name,
            embedding_dimension=self.profile.embedding_dimension,
        )

    def setup_sql(self) -> list[str]:
        return pgvector_setup_sql(self.profile)

    def ensure_schema(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        setup_statements = self.setup_sql()
        extension_statement, *schema_statements = setup_statements
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(extension_statement)
            connection.commit()
            register_vector(connection)
            with connection.cursor() as cursor:
                for statement in schema_statements:
                    cursor.execute(statement)
            connection.commit()

    def upsert_chunks(self, chunks: Iterable[RagChunkRecord]) -> int:
        import psycopg
        from psycopg.types.json import Jsonb
        from pgvector.psycopg import register_vector

        records = list(chunks)
        if not records:
            return 0
        self.ensure_schema()
        table = self.collection.qualified_table
        now = datetime.now(UTC)
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {table} (
                        profile_id, session_id, chunk_id, paper_id,
                        source_record_id, section, page, table_id, chunk_hash,
                        text, embedding, metadata, updated_at
                    )
                    VALUES (
                        %(profile_id)s, %(session_id)s, %(chunk_id)s, %(paper_id)s,
                        %(source_record_id)s, %(section)s, %(page)s, %(table_id)s, %(chunk_hash)s,
                        %(text)s, %(embedding)s, %(metadata)s::jsonb, %(updated_at)s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        paper_id = EXCLUDED.paper_id,
                        source_record_id = EXCLUDED.source_record_id,
                        section = EXCLUDED.section,
                        page = EXCLUDED.page,
                        table_id = EXCLUDED.table_id,
                        chunk_hash = EXCLUDED.chunk_hash,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                    """,
                    [
                        {
                            "profile_id": self.profile.profile_id,
                            "session_id": self.profile.session_id,
                            "chunk_id": chunk.chunk_id,
                            "paper_id": chunk.paper_id,
                            "source_record_id": chunk.source_record_id,
                            "section": chunk.section,
                            "page": chunk.page,
                            "table_id": chunk.table_id,
                            "chunk_hash": chunk.chunk_hash,
                            "text": chunk.text,
                            "embedding": chunk.embedding,
                            "metadata": Jsonb(chunk.metadata),
                            "updated_at": now,
                        }
                        for chunk in records
                    ],
                )
            connection.commit()
        return len(records)

    def delete_missing_chunks(self, chunk_ids: Iterable[str]) -> int:
        import psycopg
        from pgvector.psycopg import register_vector

        ids = list(chunk_ids)
        self.ensure_schema()
        table = self.collection.qualified_table
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE profile_id = %(profile_id)s
                      AND session_id = %(session_id)s
                      AND NOT (chunk_id = ANY(%(chunk_ids)s))
                    """,
                    {
                        "profile_id": self.profile.profile_id,
                        "session_id": self.profile.session_id,
                        "chunk_ids": ids,
                    },
                )
                deleted = cursor.rowcount if cursor.rowcount >= 0 else 0
            connection.commit()
        return deleted

    def delete_paper_chunks(self, paper_ids: Iterable[str]) -> int:
        import psycopg
        from pgvector.psycopg import register_vector

        ids = list(paper_ids)
        if not ids:
            return 0
        self.ensure_schema()
        table = self.collection.qualified_table
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE profile_id = %(profile_id)s
                      AND session_id = %(session_id)s
                      AND paper_id = ANY(%(paper_ids)s)
                    """,
                    {
                        "profile_id": self.profile.profile_id,
                        "session_id": self.profile.session_id,
                        "paper_ids": ids,
                    },
                )
                deleted = cursor.rowcount if cursor.rowcount >= 0 else 0
            connection.commit()
        return deleted

    def delete_session(self) -> int:
        import psycopg
        from pgvector.psycopg import register_vector

        self.ensure_schema()
        table = self.collection.qualified_table
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE profile_id = %(profile_id)s
                      AND session_id = %(session_id)s
                    """,
                    {
                        "profile_id": self.profile.profile_id,
                        "session_id": self.profile.session_id,
                    },
                )
                deleted = cursor.rowcount if cursor.rowcount >= 0 else 0
            connection.commit()
        return deleted

    def query_chunks(
        self,
        embedding: list[float],
        *,
        top_k: int | None = None,
    ) -> list[RagSearchHit]:
        import psycopg
        from pgvector.psycopg import register_vector

        self.ensure_schema()
        limit = max(1, int(top_k or self.profile.top_k))
        table = self.collection.qualified_table
        with psycopg.connect(self.config.rag.postgres_dsn) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        chunk_id, paper_id, source_record_id, section, page, table_id,
                        chunk_hash, text, metadata, 1 - (embedding <=> %(embedding)s::vector) AS score
                    FROM {table}
                    WHERE profile_id = %(profile_id)s
                      AND session_id = %(session_id)s
                    ORDER BY embedding <=> %(embedding)s::vector
                    LIMIT %(limit)s
                    """,
                    {
                        "profile_id": self.profile.profile_id,
                        "session_id": self.profile.session_id,
                        "embedding": embedding,
                        "limit": limit,
                    },
                )
                rows = cursor.fetchall()
        return [
            RagSearchHit(
                chunk_id=row[0],
                paper_id=row[1],
                source_record_id=row[2],
                section=row[3],
                page=row[4],
                table_id=row[5],
                chunk_hash=row[6],
                text=row[7],
                metadata=row[8] if isinstance(row[8], dict) else {},
                score=float(row[9]),
            )
            for row in rows
        ]


def pgvector_setup_sql(profile: RagProfile) -> list[str]:
    collection = PgvectorCollection(
        schema_name=profile.postgres_schema,
        table_name=profile.collection_name,
        embedding_dimension=profile.embedding_dimension,
    )
    table = collection.qualified_table
    dimension = int(collection.embedding_dimension)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE SCHEMA IF NOT EXISTS {quote_ident(collection.schema_name)}",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            profile_id text NOT NULL,
            session_id text NOT NULL,
            chunk_id text PRIMARY KEY,
            paper_id text NOT NULL,
            source_record_id text,
            section text,
            page integer,
            table_id text,
            chunk_hash text NOT NULL,
            text text NOT NULL,
            embedding vector({dimension}) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            updated_at timestamptz NOT NULL DEFAULT now(),
            CHECK (session_id = {sql_literal(profile.session_id)})
        )
        """,
        f"ALTER TABLE {table} DROP COLUMN IF EXISTS user_id CASCADE",
        f"CREATE INDEX IF NOT EXISTS {quote_ident(profile.collection_name + '_profile_idx')} "
        f"ON {table} (profile_id)",
        f"CREATE INDEX IF NOT EXISTS {quote_ident(profile.collection_name + '_paper_idx')} "
        f"ON {table} (paper_id)",
        f"CREATE INDEX IF NOT EXISTS {quote_ident(profile.collection_name + '_embedding_hnsw_idx')} "
        f"ON {table} USING hnsw (embedding vector_cosine_ops)",
    ]


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
