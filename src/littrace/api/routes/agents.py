from __future__ import annotations

from fastapi import APIRouter

from littrace.quality_audits import QualityAuditReport, audit_parser, audit_storyline, audit_tables
from littrace.workflow_status import WorkflowStatusReport, build_workflow_status
from littrace.runtime_components import RuntimeComponentStatus, component_statuses
from littrace.api.state import get_workspace
from littrace.autonomous_loop import run_review_loop
from littrace.config import load_config
from littrace.models import ReviewLoopReport
from littrace.research_planner import ResearchPlan
from littrace.skill_runner import build_research_plan_skill

router = APIRouter(prefix="/agents")


@router.get("/components", response_model=list[RuntimeComponentStatus])
def runtime_components() -> list[RuntimeComponentStatus]:
    return component_statuses()


@router.get("/quality-audits", response_model=list[QualityAuditReport])
def quality_audits() -> list[QualityAuditReport]:
    config = load_config()
    workspace = get_workspace()
    return [
        audit_parser(config, workspace),
        audit_tables(workspace),
        audit_storyline(workspace),
    ]


@router.get("/plan", response_model=ResearchPlan)
async def agents_plan(topic: str) -> ResearchPlan:
    return await build_research_plan_skill(topic, get_workspace())


@router.get("/workflow", response_model=WorkflowStatusReport)
def workflow_status() -> WorkflowStatusReport:
    return build_workflow_status(get_workspace())


@router.post("/review-loop", response_model=ReviewLoopReport)
async def review_loop(
    objective: str,
    max_rounds: int = 2,
    auto_replan: bool = False,
) -> ReviewLoopReport:
    workspace = get_workspace()
    report = await run_review_loop(
        load_config(),
        objective,
        workspace,
        max_rounds=max_rounds,
        auto_replan=auto_replan,
    )
    workspace.context.filters.autonomous_loop_report = report.model_dump(mode="json")
    return report
