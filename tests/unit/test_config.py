"""Configuration loading + canonical work identity decisions.

Consolidates the former ``test_config.py`` and ``test_identity.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from littrace.config import CodexHomeMode, LitTraceConfig, load_config
from littrace.config_wizard import write_config_template
from littrace.context import add_papers
from littrace.models import LiteratureWorkspace, PaperMetadata

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_config_reads_env_local_without_config_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    for name in [
        "LITTRACE_LLM_FALLBACK_MODELS",
        "LITTRACE_FALLBACK_LLM_API_KEY",
        "LITTRACE_FALLBACK_LLM_BASE_URL",
        "LITTRACE_FALLBACK_LLM_MODEL",
        "LITTRACE_RAG_ENABLED",
        "LITTRACE_RAG_BACKEND",
        "LITTRACE_RAG_POSTGRES_DSN",
        "LITTRACE_RAG_SCHEMA",
        "LITTRACE_RAG_COLLECTION_PREFIX",
        "LITTRACE_RAG_EMBEDDING_BASE_URL",
        "LITTRACE_RAG_EMBEDDING_API_KEY",
        "LITTRACE_RAG_EMBEDDING_MODEL",
        "LITTRACE_RAG_EMBEDDING_DIMENSION",
        "LITTRACE_RAG_AUTO_REFRESH",
    ]:
        monkeypatch.delenv(name, raising=False)
    Path(".env.local").write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=test-key",
                "DEEPSEEK_BASE_URL=https://example.com",
                "DEEPSEEK_MODEL=deepseek-test",
                "LITTRACE_LLM_FALLBACK_MODELS=deepseek-v4-pro,deepseek-chat",
                "LITTRACE_FALLBACK_LLM_API_KEY=fallback-key",
                "LITTRACE_FALLBACK_LLM_BASE_URL=https://fallback.example.com",
                "LITTRACE_FALLBACK_LLM_MODEL=fallback-model",
                "LITTRACE_RAG_ENABLED=true",
                "LITTRACE_RAG_BACKEND=pgvector",
                "LITTRACE_RAG_POSTGRES_DSN=postgresql://user:pass@localhost:5432/littrace",
                "LITTRACE_RAG_SCHEMA=littrace_test_rag",
                "LITTRACE_RAG_COLLECTION_PREFIX=test_littrace",
                "LITTRACE_RAG_EMBEDDING_BASE_URL=https://embeddings.example.com/v1",
                "LITTRACE_RAG_EMBEDDING_API_KEY=rag-key",
                "LITTRACE_RAG_EMBEDDING_MODEL=text-embedding-3-large",
                "LITTRACE_RAG_EMBEDDING_DIMENSION=3072",
                "LITTRACE_RAG_AUTO_REFRESH=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.llm.api_key == "test-key"
    assert config.llm.base_url == "https://example.com"
    assert config.llm.model == "deepseek-test"
    assert config.llm.fallback_models == ["deepseek-v4-pro", "deepseek-chat"]
    assert config.llm.fallback_api_key == "fallback-key"
    assert config.llm.fallback_base_url == "https://fallback.example.com"
    assert config.llm.fallback_model == "fallback-model"
    assert config.rag.enabled is True
    assert config.rag.backend == "pgvector"
    assert config.rag.postgres_dsn == "postgresql://user:pass@localhost:5432/littrace"
    assert config.rag.schema_name == "littrace_test_rag"
    assert config.rag.collection_prefix == "test_littrace"
    assert config.rag.embedding_base_url == "https://embeddings.example.com/v1"
    assert config.rag.embedding_api_key == "rag-key"
    assert config.rag.embedding_model == "text-embedding-3-large"
    assert config.rag.embedding_dimension == 3072
    assert config.rag.auto_refresh_enabled is True


def test_write_config_template_creates_yaml(tmp_path):
    target = tmp_path / "config.yaml"

    result = write_config_template(target)

    assert result.created
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "unpaywall_email" in text
    assert "browser_act_path" in text
    assert "chrome_profile_name" in text
    assert "auto_launch_chrome: false" in text
    assert "required: false" in text


def test_codex_runtime_home_is_isolated_and_env_overridable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert LitTraceConfig().agent_runtime.codex_home_mode == CodexHomeMode.ISOLATED
    monkeypatch.setenv("LITTRACE_CODEX_HOME_MODE", "shared")
    monkeypatch.setenv("LITTRACE_CODEX_HOME", str(tmp_path / "managed-home"))

    config = load_config()

    assert config.agent_runtime.codex_home_mode == CodexHomeMode.SHARED
    assert config.agent_runtime.codex_home == tmp_path / "managed-home"


# ---------------------------------------------------------------------------
# Canonical-work identity
# ---------------------------------------------------------------------------


def test_adding_doi_equivalent_records_keeps_identity_decisions():
    workspace = add_papers(
        LiteratureWorkspace(),
        [
            PaperMetadata(paper_id="index", title="Paper", doi="10.1000/example"),
            PaperMetadata(paper_id="publisher", title="Paper version", doi="10.1000/example"),
        ],
    )

    work = workspace.canonical_works["doi:10.1000/example"]
    assert sorted(work.version_paper_ids) == ["index", "publisher"]
    assert len(workspace.resolution_decisions) == 2
