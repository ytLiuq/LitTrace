"""End-to-end integration test for the Docling -> chunk -> embedding -> pgvector pipeline.

Pipeline exercised:
    real arXiv PDF (cached under tests/integration/.cache/)
      -> DoclingOCRTool.parse_pdf
      -> build_rag_chunk_drafts
      -> OpenAICompatibleEmbeddingClient.embed_texts (DashScope text-embedding-v3)
      -> PgvectorRagStore.upsert_chunks (real Postgres)
      -> PgvectorRagStore.query_chunks (top-K cosine)

Skip conditions:
    - docling not installed (pip install -e ".[parsers]")
    - LITTRACE_RAG_POSTGRES_DSN / LITTRACE_RAG_EMBEDDING_* env vars not set
    - pgvector DSN unreachable

The arXiv PDF is downloaded once and cached under tests/integration/.cache/.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

CACHE_DIR = Path(__file__).parent / ".cache"
PDF_URL = "https://arxiv.org/pdf/2005.14120"
PDF_NAME = "arxiv_2005_14120_mxene_pvb_sensor.pdf"
ARXIV_ID = "2005.14120"

EMBED_PROVIDER = os.environ.get("LITTRACE_RAG_EMBEDDING_PROVIDER", "openai-compatible")
EMBED_BASE_URL = os.environ.get("LITTRACE_RAG_EMBEDDING_BASE_URL")
EMBED_API_KEY = os.environ.get("LITTRACE_RAG_EMBEDDING_API_KEY")
EMBED_MODEL = os.environ.get("LITTRACE_RAG_EMBEDDING_MODEL", "text-embedding-v3")
EMBED_DIM = int(os.environ.get("LITTRACE_RAG_EMBEDDING_DIMENSION", "1024"))
PG_DSN = os.environ.get("LITTRACE_RAG_POSTGRES_DSN")
COLLECTION = "littrace_e2e_docling_rag"


def _cached_pdf() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / PDF_NAME
    if target.exists() and target.stat().st_size > 1024:
        return target
    import httpx

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(PDF_URL)
        response.raise_for_status()
        target.write_bytes(response.content)
    return target


@pytest.fixture(scope="module")
def pdf_path() -> Path:
    pytest.importorskip("docling.document_converter")
    try:
        return _cached_pdf()
    except Exception as exc:
        pytest.skip(f"could not fetch {PDF_URL}: {exc.__class__.__name__}: {exc}")


@pytest.fixture(scope="module")
def rag_config():
    from littrace.config import LitTraceConfig, RagConfig

    if not PG_DSN:
        pytest.skip("LITTRACE_RAG_POSTGRES_DSN not set")
    if not EMBED_BASE_URL or not EMBED_API_KEY:
        pytest.skip("LITTRACE_RAG_EMBEDDING_BASE_URL / LITTRACE_RAG_EMBEDDING_API_KEY not set")
    return LitTraceConfig(
        rag=RagConfig(
            enabled=True,
            backend="pgvector",
            postgres_dsn=PG_DSN,
            embedding_provider=EMBED_PROVIDER,
            embedding_base_url=EMBED_BASE_URL,
            embedding_api_key=EMBED_API_KEY,
            embedding_model=EMBED_MODEL,
            embedding_dimension=EMBED_DIM,
            schema_name="littrace_rag",
            collection_prefix="littrace",
            top_k=4,
            chunk_target_tokens=700,
            chunk_overlap_tokens=120,
        )
    )


@pytest.fixture(scope="module")
def rag_profile(rag_config):
    from littrace.retrieval.rag_profile import RagProfile

    return RagProfile(
        profile_id="rag:e2e_docling_rag",
        namespace="e2e_docling_rag",
        session_id="e2e_docling_rag",
        topic="MXene piezoresistive sensor",
        backend=rag_config.rag.backend,
        postgres_schema=rag_config.rag.schema_name,
        collection_name=COLLECTION,
        embedding_provider=rag_config.rag.embedding_provider,
        embedding_model=rag_config.rag.embedding_model,
        embedding_dimension=rag_config.rag.embedding_dimension,
        chunk_target_tokens=rag_config.rag.chunk_target_tokens,
        chunk_overlap_tokens=rag_config.rag.chunk_overlap_tokens,
        top_k=rag_config.rag.top_k,
        refresh_frequency="manual",
        auto_refresh_enabled=False,
        auto_download_open_access=False,
        login_required_policy="queue_only",
        source_routes=["arxiv", "open_access"],
    )


def test_real_arxiv_pdf_full_rag_pipeline(pdf_path, rag_config, rag_profile):
    """Parse a real arXiv PDF, embed its chunks into pgvector, retrieve top-K.

    Asserts:
        - docling parses the PDF (parsed=True)
        - markdown has substantial content and at least one figure (the
          fallback path activates for fpdf2-style PDFs)
        - chunk drafts split into sections + (tables or figures)
        - embeddings round-trip back from pgvector (chunks persisted)
        - retrieval returns sensible hits for MXene-related queries
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from littrace.config import StorageConfig
    from littrace.context import add_papers
    from littrace.evidence.parsing import local_pdf_path, parse_workspace_papers
    from littrace.models import LiteratureWorkspace, PaperMetadata
    from littrace.ocr.docling_adapter import DoclingOCRTool
    from littrace.retrieval.embeddings import embedding_client_from_config
    from littrace.retrieval.pgvector_store import PgvectorRagStore, RagChunkRecord
    from littrace.retrieval.rag_refresh import build_rag_chunk_drafts

    work_dir = CACHE_DIR / "scratch"
    work_dir.mkdir(parents=True, exist_ok=True)
    rag_config.storage = StorageConfig(
        paper_library_dir=work_dir / "papers",
        sessions_dir=work_dir / "sessions",
    )

    paper = PaperMetadata(
        paper_id=f"arxiv:{ARXIV_ID}",
        title="A highly sensitive piezoresistive sensor based on MXene and polyvinyl butyral",
        year=2020,
    )
    workspace = add_papers(LiteratureWorkspace(papers={paper.paper_id: paper}), [paper])

    target = local_pdf_path(rag_config, paper)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_path.read_bytes())

    workspace, report = parse_workspace_papers(workspace, rag_config, tool=DoclingOCRTool())
    parsed = workspace.parsed_papers[paper.paper_id]
    sd = parsed.structured_document or {}

    assert parsed.parsed, f"docling parse failed: {parsed.error}"
    assert len(sd.get("markdown", "")) > 5000, "markdown suspiciously short"
    figures = sd.get("figures", [])
    assert len(figures) >= 1, "no figures parsed (markdown fallback not active?)"

    drafts = build_rag_chunk_drafts(workspace, rag_profile)
    assert len(drafts) > 5, f"expected >5 chunk drafts, got {len(drafts)}"
    sources = {d.metadata.get("source") for d in drafts if d.metadata}
    assert sources & {"section", "table", "figure_summary"}, f"missing source types: {sources}"

    embed_client = embedding_client_from_config(rag_config, rag_profile)
    embeddings = asyncio.run(embed_client.embed_texts([d.text for d in drafts]))
    assert len(embeddings) == len(drafts)

    store = PgvectorRagStore(rag_config, rag_profile)
    store.ensure_schema()
    records = [
        RagChunkRecord(
            chunk_id=drafts[i].chunk_id,
            paper_id=drafts[i].paper_id,
            text=drafts[i].text,
            chunk_hash=drafts[i].chunk_hash,
            embedding=embeddings[i],
            metadata=drafts[i].metadata,
        )
        for i in range(len(drafts))
    ]
    store.upsert_chunks(records)

    with psycopg.connect(PG_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM {rag_profile.postgres_schema}.{COLLECTION} "
                "WHERE profile_id = %s",
                (rag_profile.profile_id,),
            )
            stored = cur.fetchone()[0]
    assert stored == len(drafts), f"expected {len(drafts)} rows, got {stored}"

    for query in [
        "MXene pressure sensor sensitivity kPa-1",
        "polyvinyl butyral PVB composite fabrication",
        "wide detection limit low power consumption",
        "piezoresistive mechanism contact resistance",
    ]:
        q_vec = asyncio.run(embed_client.embed_texts([query]))[0]
        hits = store.query_chunks(q_vec, top_k=4)
        assert len(hits) > 0, f"no hits for query='{query}'"
        for hit in hits:
            assert 0.0 <= hit.score <= 1.0, f"score out of range: {hit.score}"
            assert hit.text, "hit.text is empty"
            assert hit.paper_id == paper.paper_id
            hit_id = hashlib.sha256((query + hit.text[:80]).encode()).hexdigest()[:8]
            snippet = hit.text[:140].replace("\n", " ")
            print(
                f"  q={query!r:55s}  hit={hit_id}  score={hit.score:.4f}  "
                f"src={hit.metadata.get('source', '?') if hit.metadata else '?'}  "
                f"text={snippet}…"
            )