import asyncio

from littrace.config import LitTraceConfig
from littrace.figure_enrichment import enrich_parsed_figures
from littrace.llm import LLMReply
from littrace.models import EvidenceSpan, ParsedPaper


def _parsed(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"real-image-bytes")
    return ParsedPaper(
        parsed=True,
        figures=[
            {
                "figure_id": "F1",
                "caption": "A process diagram.",
                "asset_path": str(image),
                "asset_ref": "artifact://figure_image/p1/F1",
                "summary": "A process diagram.",
            }
        ],
        structured_document={"figures": []},
        sections=[
            {
                "name": "figures",
                "text": "A process diagram.",
                "evidence": EvidenceSpan(paper_id="p1", section="figures").model_dump(),
            }
        ],
    )


def test_figure_enrichment_is_disabled_by_default(tmp_path):
    parsed = _parsed(tmp_path)
    report = asyncio.run(enrich_parsed_figures(LitTraceConfig(), parsed))
    assert report.skipped == 1
    assert parsed.figures[0].get("enrichment_status") is None


def test_figure_enrichment_persists_accepted_analysis(monkeypatch, tmp_path):
    replies = iter(
        [
            (
                '{"figure_type":"process_diagram",'
                '"visual_summary":"Three fabrication routes are shown.",'
                '"observations":["The routes share a drying step."],'
                '"ocr_text":["gelatin"],"confidence":0.9}'
            ),
            (
                '{"confirmed":true,'
                '"corrected_summary":"The paper describes three fabrication routes.",'
                '"verified_observations":["The routes share a drying step."],'
                '"terminology_corrections":[],"issues":[],"confidence":0.9}'
            ),
        ]
    )

    async def fake_vision_completion(*_args, **_kwargs):
        return LLMReply(
            text=next(replies),
            used_llm=True,
        )

    monkeypatch.setattr("littrace.figure_enrichment.vision_completion", fake_vision_completion)
    config = LitTraceConfig()
    config.figure_enrichment.enabled = True
    parsed = _parsed(tmp_path)

    report = asyncio.run(enrich_parsed_figures(config, parsed))

    assert report.enriched == 1
    assert report.context_confirmed == 1
    assert parsed.figures[0]["enrichment_status"] == "accepted"
    assert parsed.figures[0]["context_confirmed"] is True
    assert "paper describes three fabrication routes" in parsed.figures[0]["summary"]
    assert parsed.structured_document["figures"] == parsed.figures

    second = asyncio.run(enrich_parsed_figures(config, parsed))
    assert second.skipped == 1
    assert second.enriched == 0
