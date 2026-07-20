from __future__ import annotations

from littrace.models import LiteratureWorkspace

WORKSPACE = LiteratureWorkspace()


def get_workspace() -> LiteratureWorkspace:
    return WORKSPACE


def set_workspace(workspace: LiteratureWorkspace) -> LiteratureWorkspace:
    global WORKSPACE
    WORKSPACE = workspace
    return WORKSPACE
