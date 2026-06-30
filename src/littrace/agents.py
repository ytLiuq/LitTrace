from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRoleSpec(BaseModel):
    name: str
    goal: str
    backstory: str
    tools: list[str] = Field(default_factory=list)


class AgentRuntimeStatus(BaseModel):
    name: str
    role_layer: str
    runtime: str
    implemented: bool
    workflow_node: str | None = None
    callable_tools: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)


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
        name="Publisher Connector",
        goal="Map papers to publisher families and authorized access routes.",
        backstory="A source-aware connector that prefers DOI landing pages and known OA PDFs.",
        tools=["build_publisher_route_report", "infer_publisher_family"],
    ),
    AgentRoleSpec(
        name="PDF/OCR Parser",
        goal="Parse downloaded or user-provided PDFs into traceable text, table, and page evidence.",
        backstory="A document parser that treats page-aware evidence as the unit of trust.",
        tools=["parse_workspace_papers", "docling", "paddleocr"],
    ),
    AgentRoleSpec(
        name="Table Extractor",
        goal="Extract material performance metrics into provenance-preserving comparison matrices.",
        backstory="A careful evaluator who keeps units, snippets, and comparability warnings attached.",
        tools=["extract_performance_cells", "build_comparison_matrices", "check_performance_cells"],
    ),
    AgentRoleSpec(
        name="Research Planner",
        goal="Turn a user research question into an evidence-first multi-agent plan.",
        backstory="A workflow strategist that chooses when to retrieve, parse, compare, and narrate.",
        tools=["build_research_plan", "route_sources", "build_publisher_search_plan"],
    ),
    AgentRoleSpec(
        name="Research Writer",
        goal="Write evidence-grounded answers with citation guardrails.",
        backstory="A cautious scientific writer who removes unsupported claims before final output.",
        tools=["write_evidence_grounded_answer", "guard_citations", "remove_unsupported_sentences"],
    ),
    AgentRoleSpec(
        name="Eval Auditor",
        goal="Measure retrieval, parsing, table, storyline, and citation quality.",
        backstory="A quality engineer for research workflows and golden-set regression tests.",
        tools=["build_quality_report", "run_golden_eval", "build_agent_portfolio_report"],
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


def agent_runtime_statuses() -> list[AgentRuntimeStatus]:
    return [
        AgentRuntimeStatus(
            name="Source Router",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="plan_sources",
            callable_tools=["route_sources"],
            remaining_work=["Turn publisher search-plan URLs into optional browser-assisted retrieval."],
        ),
        AgentRuntimeStatus(
            name="Citation Verifier",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="audit_citations",
            callable_tools=["citation_records_for_papers", "audit_citation_links"],
            remaining_work=["Cache link checks and classify institution-login redirects."],
        ),
        AgentRuntimeStatus(
            name="Access Manager",
            role_layer="CrewAI role + callable tool",
            runtime="Local async tool",
            implemented=True,
            workflow_node="plan_downloads",
            callable_tools=["build_download_plan", "execute_downloads"],
            remaining_work=["Detect when the manually downloaded PDF has appeared and resume parsing."],
        ),
        AgentRuntimeStatus(
            name="Publisher Connector",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="route_publishers",
            callable_tools=["build_publisher_route_report", "infer_publisher_family"],
            remaining_work=["Parse publisher search result pages when terms and authentication allow it."],
        ),
        AgentRuntimeStatus(
            name="PDF/OCR Parser",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="parse_full_text",
            callable_tools=["parse_workspace_papers", "docling", "paddleocr"],
            remaining_work=["Expand the benchmark from session metrics to a curated golden PDF set."],
        ),
        AgentRuntimeStatus(
            name="Table Extractor",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="extract_tables",
            callable_tools=["extract_performance_cells", "build_comparison_matrices"],
            remaining_work=["Add stronger unit conversion and chemistry-specific comparability rules."],
        ),
        AgentRuntimeStatus(
            name="Research Planner",
            role_layer="CrewAI role + callable tool",
            runtime="Local deterministic planner",
            implemented=True,
            workflow_node=None,
            callable_tools=["build_research_plan"],
            remaining_work=["Learn from successful traces to prioritize steps automatically."],
        ),
        AgentRuntimeStatus(
            name="Research Writer",
            role_layer="CrewAI role + LLM tool",
            runtime="DeepSeek-compatible LLM + citation guard",
            implemented=True,
            workflow_node=None,
            callable_tools=["write_evidence_grounded_answer", "guard_citations"],
            remaining_work=["Add paragraph-level revision loops when citation guard fails."],
        ),
        AgentRuntimeStatus(
            name="Eval Auditor",
            role_layer="CrewAI role + callable tools",
            runtime="Local quality/golden-set tools",
            implemented=True,
            workflow_node=None,
            callable_tools=["build_quality_report", "run_golden_eval", "build_agent_portfolio_report"],
            remaining_work=["Expand the curated materials/chemistry golden set."],
        ),
        AgentRuntimeStatus(
            name="Storyline Verifier",
            role_layer="CrewAI role + LangGraph node",
            runtime="LangGraph",
            implemented=True,
            workflow_node="build_storyline",
            callable_tools=["build_storyline_from_workspace", "check_storyline_claims"],
            remaining_work=["Add sentence-level citation validation for generated narratives."],
        ),
    ]


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
