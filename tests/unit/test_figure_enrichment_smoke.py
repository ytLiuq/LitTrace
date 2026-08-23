"""Smoke test for littrace.figure_enrichment. No mocks — exercises the
multi-modal figure enricher construction and basic config wiring."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_figure_enrichment_config_defaults():
    from littrace.config import FigureEnrichmentConfig

    cfg = FigureEnrichmentConfig()
    assert cfg.enabled is False
    assert cfg.prompt
    assert "JSON" in cfg.prompt or "json" in cfg.prompt


def test_enrich_parsed_figures_no_backend_returns_empty(monkeypatch):
    import asyncio

    from littrace.config import FigureEnrichmentConfig, LitTraceConfig
    from littrace.figure_enrichment import enrich_parsed_figures
    from littrace.models import ParsedPaper

    cfg = LitTraceConfig(figure_enrichment=FigureEnrichmentConfig(enabled=False))
    paper = ParsedPaper(paper_id="p1", parsed=True)
    report = asyncio.run(enrich_parsed_figures(cfg, paper))
    # disabled config — no enrichments, no rejections, no failures
    assert report.enriched == 0
    assert report.rejected == 0
    assert report.failed == 0


def test_figure_bytes_can_be_loaded_from_artifact_storage(tmp_path):
    from littrace.artifact_store import artifact_store_from_config
    from littrace.config import ArtifactStorageConfig, LitTraceConfig
    from littrace.figure_enrichment import _load_figure_bytes

    config = LitTraceConfig(
        artifact_storage=ArtifactStorageConfig(
            backend="local",
            local_root=tmp_path / "artifacts",
        )
    )
    image = b"\x89PNG\r\n\x1a\nfigure"
    ref = artifact_store_from_config(config).put_bytes(
        "figures/f1.png",
        image,
        content_type="image/png",
    )

    path, loaded, content_type = _load_figure_bytes(
        config,
        {"asset_path": None, "storage_ref": ref.model_dump(mode="json")},
    )

    assert path is not None and path.suffix == ".png"
    assert loaded == image
    assert content_type == "image/png"
