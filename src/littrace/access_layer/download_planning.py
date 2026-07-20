from __future__ import annotations

from littrace.access_layer.paths import (
    build_download_plan,
    paper_storage_dir,
    plan_download,
    target_pdf_path,
)
from littrace.attachments import attach_pdf_to_paper, check_download_presence
from littrace.downloads import execute_downloads

__all__ = [
    "build_download_plan",
    "attach_pdf_to_paper",
    "check_download_presence",
    "execute_downloads",
    "paper_storage_dir",
    "plan_download",
    "target_pdf_path",
]
