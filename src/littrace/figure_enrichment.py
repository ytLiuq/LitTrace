"""Durable-friendly multimodal enrichment for Docling figure assets."""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from littrace.config import LitTraceConfig
from littrace.llm import vision_completion
from littrace.models import ParsedPaper


class FigureAnalysis(BaseModel):
    figure_type: str = "unknown"
    visual_summary: str = ""
    observations: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FigureContextConfirmation(BaseModel):
    confirmed: bool = False
    corrected_summary: str = ""
    verified_observations: list[str] = Field(default_factory=list)
    terminology_corrections: dict[str, str] | list[str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FigureEnrichmentReport(BaseModel):
    processed: int = 0
    enriched: int = 0
    skipped: int = 0
    rejected: int = 0
    failed: int = 0
    context_confirmed: int = 0
    warnings: list[str] = Field(default_factory=list)


async def enrich_parsed_figures(
    config: LitTraceConfig,
    parsed: ParsedPaper,
) -> FigureEnrichmentReport:
    """Analyze extracted figure files and update the parsed document in place.

    Existing successful results are reused when the image hash is unchanged.
    This makes retries idempotent and lets the durable embedding job resume
    after a process interruption without paying for completed images again.
    """
    report = FigureEnrichmentReport()
    settings = config.figure_enrichment
    if not settings.enabled:
        report.skipped = len(parsed.figures)
        return report

    figures = [figure for figure in parsed.figures if isinstance(figure, dict)]
    for figure in figures[: settings.max_figures_per_job]:
        report.processed += 1
        path_value = figure.get("asset_path")
        path = Path(str(path_value)) if path_value else None
        if path is None or not path.is_file():
            report.failed += 1
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:missing_asset"
            )
            continue

        data = path.read_bytes()
        image_sha256 = sha256(data).hexdigest()
        if (
            figure.get("enrichment_status") == "accepted"
            and figure.get("enrichment_image_sha256") == image_sha256
        ):
            report.skipped += 1
            continue

        caption = str(figure.get("caption") or "").strip()
        prompt = settings.prompt
        if caption:
            prompt += f"\nFigure caption from the paper:\n{caption}"
        reply = await vision_completion(
            config,
            image_data_url=_image_data_url(path, data),
            prompt=prompt,
            json_mode=True,
        )
        if not reply.used_llm:
            report.failed += 1
            figure["enrichment_status"] = "failed"
            figure["enrichment_error"] = reply.error or "vision_request_failed"
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:{figure['enrichment_error']}"
            )
            continue

        try:
            analysis = FigureAnalysis.model_validate(_parse_json(reply.text))
        except Exception as exc:
            report.failed += 1
            figure["enrichment_status"] = "failed"
            figure["enrichment_error"] = (
                f"invalid_response:{exc.__class__.__name__}:{str(exc)[:300]}"
            )
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:{figure['enrichment_error']}"
            )
            continue

        figure["enrichment_image_sha256"] = image_sha256
        figure["figure_type"] = analysis.figure_type
        figure["visual_summary"] = analysis.visual_summary
        figure["observations"] = analysis.observations
        figure["ocr_text"] = analysis.ocr_text
        figure["enrichment_confidence"] = analysis.confidence
        if analysis.confidence < settings.min_confidence or not analysis.visual_summary:
            figure["enrichment_status"] = "rejected"
            report.rejected += 1
            continue

        context = _figure_context(parsed, caption)
        if not context:
            figure["enrichment_status"] = "rejected"
            figure["context_confirmation_status"] = "missing_context"
            report.rejected += 1
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:missing_context"
            )
            continue

        confirmation_reply = await vision_completion(
            config,
            image_data_url=_image_data_url(path, data),
            prompt=_confirmation_prompt(caption, context, analysis),
            json_mode=True,
        )
        if not confirmation_reply.used_llm:
            report.failed += 1
            figure["enrichment_status"] = "failed"
            figure["context_confirmation_status"] = "failed"
            figure["enrichment_error"] = (
                confirmation_reply.error or "context_confirmation_failed"
            )
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:{figure['enrichment_error']}"
            )
            continue
        try:
            confirmation = FigureContextConfirmation.model_validate(
                _parse_json(confirmation_reply.text)
            )
        except Exception as exc:
            report.failed += 1
            figure["enrichment_status"] = "failed"
            figure["context_confirmation_status"] = "failed"
            figure["enrichment_error"] = (
                f"invalid_confirmation:{exc.__class__.__name__}:{str(exc)[:300]}"
            )
            report.warnings.append(
                f"figure:{figure.get('figure_id', report.processed)}:{figure['enrichment_error']}"
            )
            continue

        figure["context_confirmation"] = confirmation.model_dump(mode="json")
        figure["context_confirmation_confidence"] = confirmation.confidence
        figure["context_confirmed"] = confirmation.confirmed
        if not confirmation.confirmed or confirmation.confidence < settings.min_confidence:
            figure["enrichment_status"] = "rejected"
            figure["context_confirmation_status"] = "rejected"
            report.rejected += 1
            continue

        confirmed_analysis = analysis.model_copy(
            update={
                "visual_summary": confirmation.corrected_summary or analysis.visual_summary,
                "observations": confirmation.verified_observations or analysis.observations,
                "confidence": min(analysis.confidence, confirmation.confidence),
            }
        )
        figure["visual_summary"] = confirmed_analysis.visual_summary
        figure["observations"] = confirmed_analysis.observations
        figure["enrichment_confidence"] = confirmed_analysis.confidence
        figure["enrichment_status"] = "accepted"
        figure["context_confirmation_status"] = "confirmed"
        figure["enrichment_source"] = "multimodal_llm_context_confirmed"
        figure["summary"] = _rag_summary(figure, confirmed_analysis)
        report.context_confirmed += 1
        report.enriched += 1

    report.skipped += max(0, len(figures) - settings.max_figures_per_job)
    parsed.structured_document["figures"] = parsed.figures
    parsed.structured_document["figure_enrichment"] = report.model_dump(mode="json")
    return report


def _image_data_url(path: Path, data: bytes) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _parse_json(text: str) -> object:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removeprefix("json").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return json.loads(cleaned)


def _rag_summary(figure: dict[str, object], analysis: FigureAnalysis) -> str:
    parts: list[str] = []
    if figure.get("caption"):
        parts.append(f"Caption: {figure['caption']}")
    parts.append(f"Visual summary: {analysis.visual_summary}")
    if analysis.observations:
        parts.append("Observations: " + "; ".join(analysis.observations))
    if analysis.ocr_text:
        parts.append("Visible labels: " + "; ".join(analysis.ocr_text))
    return " ".join(parts)


def _figure_context(parsed: ParsedPaper, caption: str) -> str:
    sections = [
        str(section.get("text") or "").strip()
        for section in parsed.sections
        if isinstance(section, dict) and str(section.get("text") or "").strip()
    ]
    if not sections:
        return caption
    context = "\n\n".join(sections)
    if caption and caption in context:
        start = max(0, context.index(caption) - 2500)
        return context[start : start + 4500]
    return context[:4500]


def _confirmation_prompt(
    caption: str,
    context: str,
    analysis: FigureAnalysis,
) -> str:
    return (
        "Do a second-pass context verification of the image analysis. "
        "Compare the image, the paper caption, and the paper context. "
        "Correct terminology using the paper context. Mark confirmed=false "
        "only if a central visual claim is unsupported or conflicts with the paper. "
        "Do not reject merely because the caption uses an abbreviation or omits "
        "a full chemical name; put those fixes in terminology_corrections and "
        "return confirmed=true when the corrected analysis is supported. "
        "Return JSON only with fields: confirmed, corrected_summary, "
        "verified_observations, terminology_corrections, issues, confidence.\n\n"
        f"Caption:\n{caption or '(none)'}\n\n"
        f"Paper context:\n{context}\n\n"
        f"Initial image analysis:\n{analysis.model_dump_json()}"
    )
