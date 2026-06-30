from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRoleSpec(BaseModel):
    name: str
    goal: str
    backstory: str
    tools: list[str] = Field(default_factory=list)


LITTRACE_CREW_ROLES = [
    AgentRoleSpec(
        name="Source Router",
        goal="Choose high-quality materials and chemistry literature sources for the query.",
        backstory="A bibliometrics-aware research librarian focused on publisher coverage and recency.",
        tools=["route_sources", "openalex_search", "crossref_search", "unpaywall_lookup"],
    ),
    AgentRoleSpec(
        name="Citation Verifier",
        goal="Ensure every paper-specific answer has a citation and a resolvable access link.",
        backstory="A meticulous citation auditor who treats unverified links as research debt.",
        tools=["citation_records_for_papers", "audit_citation_links"],
    ),
    AgentRoleSpec(
        name="Access Manager",
        goal="Plan and execute compliant PDF downloads without bypassing authentication.",
        backstory="A permissions-first archivist who separates open access from login-required content.",
        tools=["build_download_plan", "execute_downloads"],
    ),
    AgentRoleSpec(
        name="Storyline Verifier",
        goal="Constrain research narratives to evidence-backed solution-limit-response chains.",
        backstory="A skeptical materials scientist who rejects broad claims without paper-level evidence.",
        tools=["check_storyline_claims", "check_citations"],
    ),
]


def crew_role_specs() -> list[AgentRoleSpec]:
    return LITTRACE_CREW_ROLES


def build_crewai_agents():
    try:
        from crewai import Agent
    except ImportError:
        return None

    return [
        Agent(
            role=spec.name,
            goal=spec.goal,
            backstory=spec.backstory,
            verbose=False,
        )
        for spec in LITTRACE_CREW_ROLES
    ]
