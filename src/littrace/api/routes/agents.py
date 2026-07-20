from __future__ import annotations

from fastapi import APIRouter

from littrace.agent_audits import AgentAuditReport, audit_parser_agent, audit_storyline_agent, audit_table_agent
from littrace.agent_interactions import AgentInteractionReport, build_agent_interaction_report
from littrace.agent_strength import AgentPortfolioReport, build_agent_portfolio_report
from littrace.agents import (
    AgentRoleSpec,
    AgentRuntimeStatus,
    agent_runtime_statuses,
    crew_role_specs,
    runtime_agent_role_specs,
)
from littrace.api.state import get_workspace
from littrace.autonomous_loop import run_autonomous_research_loop
from littrace.config import load_config
from littrace.models import AutonomousResearchLoopReport
from littrace.research_planner import ResearchPlan
from littrace.skill_runner import build_research_plan_skill

router = APIRouter(prefix="/agents")


@router.get("/crew", response_model=list[AgentRoleSpec])
def agents_crew() -> list[AgentRoleSpec]:
    return crew_role_specs()


@router.get("/runtime", response_model=list[AgentRoleSpec])
def agents_runtime() -> list[AgentRoleSpec]:
    return runtime_agent_role_specs()


@router.get("/status", response_model=list[AgentRuntimeStatus])
def agents_status() -> list[AgentRuntimeStatus]:
    return agent_runtime_statuses()


@router.get("/strength", response_model=AgentPortfolioReport)
def agents_strength() -> AgentPortfolioReport:
    return build_agent_portfolio_report(load_config(), get_workspace())


@router.get("/audits", response_model=list[AgentAuditReport])
def agents_audits() -> list[AgentAuditReport]:
    config = load_config()
    workspace = get_workspace()
    return [
        audit_parser_agent(config, workspace),
        audit_table_agent(workspace),
        audit_storyline_agent(workspace),
    ]


@router.get("/plan", response_model=ResearchPlan)
async def agents_plan(topic: str) -> ResearchPlan:
    return await build_research_plan_skill(topic, get_workspace())


@router.get("/interactions", response_model=AgentInteractionReport)
def agents_interactions() -> AgentInteractionReport:
    return build_agent_interaction_report(get_workspace())


@router.post("/autonomous-loop", response_model=AutonomousResearchLoopReport)
async def agents_autonomous_loop(
    objective: str,
    max_rounds: int = 2,
    auto_replan: bool = False,
) -> AutonomousResearchLoopReport:
    workspace = get_workspace()
    report = await run_autonomous_research_loop(
        load_config(),
        objective,
        workspace,
        max_rounds=max_rounds,
        auto_replan=auto_replan,
    )
    workspace.context.filters.autonomous_loop_report = report.model_dump(mode="json")
    return report
