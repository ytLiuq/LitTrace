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
    from littrace.figure_enrichment import enrich_parsed_figures
    from littrace.config import LitTraceConfig, FigureEnrichmentConfig
    from littrace.models import ParsedPaper

    cfg = LitTraceConfig(figure_enrichment=FigureEnrichmentConfig(enabled=False))
    paper = ParsedPaper(paper_id="p1", parsed=True)
    report = asyncio.run(enrich_parsed_figures(cfg, paper))
    # disabled config — no enrichments, no rejections, no failures
    assert report.enriched == 0
    assert report.rejected == 0
    assert report.failed == 0