"""Plugin-style skill registry for LitTrace.

Each skill lives in its own sub-package under ``littrace.skills.<name>``
with the layout::

    skills/<name>/
        __init__.py    # register() + run = re-export of run.py::run
        SKILL.md       # human / agent documentation
        run.py         # business logic; wires contract → function via run_tool

The ``__init__.py`` for each skill is auto-imported when this package is
loaded; each one calls ``register()`` to attach itself to the module-global
:func:`registry`. Third-party skills can be discovered at runtime via
:func:`discover`, which walks the ``littrace.skills`` entry-point group.

New code should import from ``littrace.skills.<name>``. The legacy
``littrace.skill_runner`` module remains as a thin shim for backward
compatibility — see ``skill_runner.py`` for the deprecation notice.
"""
from __future__ import annotations

# Eager import: each submodule's __init__.py self-registers in
# registry().all() at module import time. Ordering matters only for
# debugging — registry is dict-based and idempotent.
from littrace.skills import (
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
    review_agent,
    search_papers,
    skill_creator,
)

from littrace.skills.registry import SkillManifest, discover, registry


__all__ = [
    "SkillManifest",
    "discover",
    "registry",
]