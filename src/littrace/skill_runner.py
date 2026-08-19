"""Backward-compatibility shim for :mod:`littrace.skill_runner`.

The 13 ``*_skill`` functions plus :class:`SearchSkillResult` now live in
the plugin-style sub-packages under :mod:`littrace.skills.<name>`. This
module re-exports them under their historical names so that the 20+
existing callers (CLI, API routes, autonomous loop, sentinel agent, MCP
server, scripts, and ``tests/unit/test_observability.py``) keep working
without any code change.

**New code should import from** ``littrace.skills.<name>.run`` **directly.**

Deprecation timeline
--------------------
This shim is intended to remain for at least one release. After the next
minor version bump it may move to a ``DeprecationWarning``-only stub or
be removed entirely; check ``docs/migration_skill_runner.md`` at that
point.
"""
from __future__ import annotations

import warnings

# Eager import: each submodule's ``__init__.py`` self-registers its skill
# in the global :class:`SkillRegistry`. Importing the package first means
# ``registry().all()`` already contains every built-in skill by the time
# callers reach for the legacy aliases below.
from littrace.skills import (  # noqa: F401  (side-effect: populates registry)
    audit_citation_links,
    build_comparison_matrices,
    build_download_plan,
    build_quality_metrics,
    build_research_document_report,
    build_research_plan,
    build_storyline_from_workspace,
    execute_downloads,
    export_session_bundle,
    extract_performance_cells,
    parse_workspace_papers,
    quality_report,
    resolve_workspace_full_text,
    search_papers,
)

# Re-exports — bind each new ``run`` to the legacy ``<name>_skill`` alias.
# These are the only symbols the 20 existing callers depend on, so the
# shim surface stays minimal.
from littrace.skills.audit_citation_links import run as audit_citation_links_skill
from littrace.skills.build_comparison_matrices import run as build_comparison_matrix_skill
from littrace.skills.build_download_plan import run as build_download_plan_skill
from littrace.skills.build_quality_metrics import run as build_quality_metrics_skill
from littrace.skills.build_research_document_report import run as build_research_report_skill
from littrace.skills.build_research_plan import run as build_research_plan_skill
from littrace.skills.build_storyline_from_workspace import run as build_storyline_skill
from littrace.skills.execute_downloads import run as execute_downloads_skill
from littrace.skills.export_session_bundle import run as export_session_bundle_skill
from littrace.skills.extract_performance_cells import run as extract_tables_skill
from littrace.skills.parse_workspace_papers import run as parse_workspace_skill
from littrace.skills.quality_report import run as build_quality_report_skill
from littrace.skills.resolve_workspace_full_text import run as resolve_workspace_full_text_skill
from littrace.skills.search_papers import run as search_papers_skill

# ``SearchSkillResult`` is a dataclass. The identity must be preserved
# across the legacy and new locations so that ``isinstance`` checks in
# callers like ``tests/unit/test_observability.py`` and ``workflow.py``
# keep working. Re-exporting from the canonical module achieves this.
from littrace.skills._helpers import SearchSkillResult


# Emit the deprecation warning once at module-import time so test runs and
# CLI invocations both notice. Use a module filter to prevent duplicate emissions.
warnings.warn(
    "littrace.skill_runner is deprecated; import from littrace.skills.<name> instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    "SearchSkillResult",
    "audit_citation_links_skill",
    "build_comparison_matrix_skill",
    "build_download_plan_skill",
    "build_quality_metrics_skill",
    "build_quality_report_skill",
    "build_research_plan_skill",
    "build_research_report_skill",
    "build_storyline_skill",
    "execute_downloads_skill",
    "export_session_bundle_skill",
    "extract_tables_skill",
    "parse_workspace_skill",
    "resolve_workspace_full_text_skill",
    "search_papers_skill",
]