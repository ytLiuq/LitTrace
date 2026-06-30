from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DownloadMode(StrEnum):
    METADATA_ONLY = "metadata_only"
    ASK_EACH_TIME = "ask_each_time"
    DOWNLOAD_OPEN_ACCESS = "download_open_access"
    DOWNLOAD_SELECTED = "download_selected"
    DOWNLOAD_ALL_ALLOWED = "download_all_allowed"


class StorageConfig(BaseModel):
    paper_library_dir: Path = Path("./data/papers")
    metadata_dir: Path = Path("./data/metadata")
    cache_dir: Path = Path("./data/cache")
    sessions_dir: Path = Path("./sessions")


class PaperDownloadConfig(BaseModel):
    mode: DownloadMode = DownloadMode.ASK_EACH_TIME
    organize_by: str = "year_doi"
    filename_template: str = "{year}_{first_author}_{short_title}_{doi_hash}.pdf"
    save_metadata_even_if_pdf_skipped: bool = True
    allow_requires_login_download: bool = True


class DoclingParserConfig(BaseModel):
    export_markdown: bool = True
    extract_tables: bool = True
    extract_figures: bool = True


class PaddleOCRParserConfig(BaseModel):
    lang: str = "en"
    use_angle_cls: bool = True
    pdf_render_scale: float = 2.0
    max_pages: int | None = None


class ParsingConfig(BaseModel):
    default_parser: str = "metadata_only"
    preferred_engines: list[str] = Field(
        default_factory=lambda: ["docling", "paddleocr", "marker", "grobid"]
    )
    docling: DoclingParserConfig = Field(default_factory=DoclingParserConfig)
    paddleocr: PaddleOCRParserConfig = Field(default_factory=PaddleOCRParserConfig)


class APIConfig(BaseModel):
    user_agent: str = "LitTrace/0.1"
    openalex_api_key: str | None = None
    unpaywall_email: str | None = None
    crossref_mailto: str | None = None
    request_timeout_seconds: float = 20.0
    enable_live_search: bool = False


class LiteratureContextDefaults(BaseModel):
    visible_to_user: bool = True
    default_year_min: int = 2023
    default_recent_year_min: int = 2023
    preferred_disciplines: list[str] = Field(default_factory=list)
    preferred_publishers: list[str] = Field(default_factory=list)
    preferred_journals: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    golden_set_dir: Path = Path("./eval/golden")
    traces_dir: Path = Path("./eval/traces")


class LitTraceConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    paper_download: PaperDownloadConfig = Field(default_factory=PaperDownloadConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    literature_context: LiteratureContextDefaults = Field(
        default_factory=LiteratureContextDefaults
    )
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path = "config.yaml") -> LitTraceConfig:
    config_path = Path(path)
    if not config_path.exists():
        return LitTraceConfig()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return LitTraceConfig.model_validate(raw)
