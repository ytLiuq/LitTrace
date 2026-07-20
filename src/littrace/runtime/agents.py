from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from littrace.agent_interactions import build_agent_interaction_report
from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace, PaperSearchRequest
from littrace.runtime.messages import (
    AgentArtifact,
    AgentMessage,
    AgentRunResult,
    ReActStep,
    ReActTrace,
)
from littrace.skill_runner import (
    audit_citation_links_skill,
    build_comparison_matrix_skill,
    build_download_plan_skill,
    build_quality_report_skill,
    build_research_plan_skill,
    build_research_report_skill,
    build_storyline_skill,
    extract_tables_skill,
    parse_workspace_skill,
    resolve_workspace_full_text_skill,
    search_papers_skill,
)
from littrace.tool_contracts import ToolCallContext


PLANNER_ALLOWED_TOOLS = ["build_research_plan"]
RETRIEVAL_ALLOWED_TOOLS = ["search_papers"]
ACCESS_PARSING_ALLOWED_TOOLS = [
    "resolve_workspace_full_text",
    "build_download_plan",
    "parse_workspace_papers",
]


def _planned_access_actions(intent: str) -> list[str]:
    actions: list[str] = []
    if intent in {"resolve_full_text", "access", "access_and_parse"}:
        actions.append("resolve_workspace_full_text")
    if intent in {"download_plan", "access", "access_and_parse"}:
        actions.append("build_download_plan")
    if intent in {"parse", "access_and_parse"}:
        actions.append("parse_workspace_papers")
    return actions


def _planned_retrieval_access_actions(intent: str) -> list[str]:
    actions: list[str] = []
    if intent in {"search", "retrieve", "research"}:
        actions.append("search_papers")
    if intent in {
        "resolve_full_text",
        "download_plan",
        "access",
        "parse",
        "access_and_parse",
        "research",
    }:
        access_intent = "access_and_parse" if intent == "research" else intent
        actions.extend(_planned_access_actions(access_intent))
    return actions


def _react_step(
    index: int,
    thought: str,
    action: str,
    observation: str,
    *,
    decision: str = "act",
    tool: str | None = None,
    next_action: str | None = None,
    ok: bool = True,
) -> ReActStep:
    return ReActStep(
        step_index=index,
        thought=thought,
        decision=decision,
        action=action,
        observation=observation,
        tool=tool,
        next_action=next_action,
        ok=ok,
    )


def _react_artifact(agent: str, step: ReActStep) -> AgentArtifact:
    return AgentArtifact(
        kind="react_step",
        producer=agent,
        payload=step.model_dump(mode="json"),
    )


def _with_next_actions(steps: list[ReActStep]) -> list[ReActStep]:
    updated: list[ReActStep] = []
    for index, step in enumerate(steps):
        next_action = steps[index + 1].action if index + 1 < len(steps) else "finish"
        updated.append(step.model_copy(update={"next_action": next_action}))
    return updated


def _tool_result_artifact(agent: str, result) -> AgentArtifact:
    return AgentArtifact(
        kind="tool_result",
        producer=agent,
        payload=result.model_dump(mode="json"),
    )


class RuntimeAgent(Protocol):
    name: str

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        """Run one agent turn and return produced artifacts plus workspace state."""


@dataclass(frozen=True)
class PlannerAgent:
    name: str = "planner"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        topic = str(message.payload.get("topic") or message.payload.get("objective") or "")
        try:
            plan = await build_research_plan_skill(
                topic,
                workspace,
                context=ToolCallContext(
                    caller=self.name,
                    task_id=message.task_id,
                    intent=message.intent,
                    react_step=1,
                ),
            )
        except RuntimeError as exc:
            step = _react_step(
                1,
                "The user objective should be converted into bounded evidence steps.",
                "build_research_plan",
                str(exc),
                tool="build_research_plan",
                ok=False,
            )
            return (
                AgentRunResult(
                    agent=self.name,
                    status="failed",
                    react_trace=ReActTrace(
                        allowed_tools=PLANNER_ALLOWED_TOOLS,
                        steps=_with_next_actions([step]),
                        final_observation=str(exc),
                        stop_reason="tool_failed",
                    ),
                    artifacts=[
                        _react_artifact(self.name, step),
                    ],
                    warnings=[str(exc)],
                ),
                workspace,
            )
        step = _react_step(
            1,
            "The user objective should be converted into bounded evidence steps.",
            "build_research_plan",
            f"planned {len(plan.steps)} steps",
            tool="build_research_plan",
        )
        trace = ReActTrace(
            allowed_tools=PLANNER_ALLOWED_TOOLS,
            planned_actions=["build_research_plan"],
            steps=_with_next_actions([step]),
            final_observation="research plan ready",
            stop_reason="completed",
        )
        return (
            AgentRunResult(
                agent=self.name,
                react_trace=trace,
                artifacts=[
                    _react_artifact(self.name, step),
                    AgentArtifact(
                        kind="research_plan",
                        producer=self.name,
                        payload=plan.model_dump(mode="json"),
                    ),
                ],
            ),
            workspace,
        )


@dataclass(frozen=True)
class RetrievalAgent:
    name: str = "retrieval"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        request_payload = message.payload.get("search") or message.payload
        request = PaperSearchRequest.model_validate(request_payload)
        context = ToolCallContext(
            caller=self.name,
            task_id=message.task_id,
            intent=message.intent,
            react_step=1,
        )
        try:
            search_result = await search_papers_skill(
                request,
                config,
                context=context,
            )
        except RuntimeError as exc:
            step = _react_step(
                1,
                "Search is the next evidence acquisition step.",
                "search_papers",
                str(exc),
                tool="search_papers",
                ok=False,
            )
            return (
                AgentRunResult(
                    agent=self.name,
                    status="failed",
                    react_trace=ReActTrace(
                        allowed_tools=RETRIEVAL_ALLOWED_TOOLS,
                        steps=_with_next_actions([step]),
                        final_observation=str(exc),
                        stop_reason="tool_failed",
                    ),
                    artifacts=[_react_artifact(self.name, step)],
                    warnings=[str(exc)],
                ),
                workspace,
            )
        result = search_result.result
        diagnostics_payload = (
            search_result.diagnostics.model_dump(mode="json")
            if search_result.diagnostics
            else None
        )
        step = _react_step(
            1,
            "Search is the next evidence acquisition step.",
            "search_papers",
            f"retrieved {len(result.papers)} papers",
            tool="search_papers",
            ok=True,
        )
        from littrace.context import add_ranked_candidate_papers

        workspace = add_ranked_candidate_papers(
            workspace,
            result.papers,
            request,
            active_limit=config.literature_context.active_context_limit,
        )
        return (
            AgentRunResult(
                agent=self.name,
                react_trace=ReActTrace(
                    allowed_tools=RETRIEVAL_ALLOWED_TOOLS,
                    planned_actions=["search_papers"],
                    steps=_with_next_actions([step]),
                    final_observation=f"retrieved {len(result.papers)} papers",
                    stop_reason="completed",
                ),
                artifacts=[
                    _react_artifact(self.name, step),
                    _tool_result_artifact(self.name, search_result.tool_result),
                    AgentArtifact(
                        kind="paper_search_result",
                        producer=self.name,
                        payload=result.model_dump(mode="json"),
                    ),
                    AgentArtifact(
                        kind="search_diagnostics",
                        producer=self.name,
                        payload={
                            "use_live": search_result.use_live,
                            "diagnostics": diagnostics_payload,
                        },
                    ),
                ],
            ),
            workspace,
        )


@dataclass(frozen=True)
class AccessParsingAgent:
    name: str = "access_parsing"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        artifacts: list[AgentArtifact] = []
        react_steps: list[ReActStep] = []
        intent = message.intent
        if intent in {"resolve_full_text", "access", "access_and_parse"}:
            try:
                workspace = await resolve_workspace_full_text_skill(
                    workspace,
                    config,
                    context=ToolCallContext(
                        caller=self.name,
                        task_id=message.task_id,
                        intent=message.intent,
                        react_step=len(react_steps) + 1,
                        metadata={"active_papers": len(workspace.context.active_papers)},
                    ),
                )
            except RuntimeError as exc:
                step = _react_step(
                    len(react_steps) + 1,
                    "Full-text candidates should be resolved before access or parsing.",
                    "resolve_workspace_full_text",
                    str(exc),
                    tool="resolve_workspace_full_text",
                    ok=False,
                )
                react_steps.append(step)
                artifacts.append(_react_artifact(self.name, step))
                return (
                    AgentRunResult(
                        agent=self.name,
                        status="failed",
                        react_trace=ReActTrace(
                            allowed_tools=ACCESS_PARSING_ALLOWED_TOOLS,
                            steps=_with_next_actions(react_steps),
                            final_observation=str(exc),
                            stop_reason="tool_failed",
                        ),
                        artifacts=artifacts,
                        warnings=[str(exc)],
                    ),
                    workspace,
                )
            react_steps.append(
                _react_step(
                    len(react_steps) + 1,
                    "Full-text candidates should be resolved before access or parsing.",
                    "resolve_workspace_full_text",
                    f"resolved {len(workspace.full_text_reports)} reports",
                    tool="resolve_workspace_full_text",
                )
            )
            artifacts.append(_react_artifact(self.name, react_steps[-1]))
            artifacts.append(
                AgentArtifact(
                    kind="full_text_reports",
                    producer=self.name,
                    payload={
                        key: value.model_dump(mode="json")
                        for key, value in workspace.full_text_reports.items()
                    },
                )
            )
        if intent in {"download_plan", "access", "access_and_parse"}:
            papers = [workspace.papers[pid] for pid in workspace.context.active_papers]
            try:
                plan = await build_download_plan_skill(
                    config,
                    workspace,
                    context=ToolCallContext(
                        caller=self.name,
                        task_id=message.task_id,
                        intent=message.intent,
                        react_step=len(react_steps) + 1,
                        metadata={"paper_count": len(papers)},
                    ),
                )
            except RuntimeError as exc:
                step = _react_step(
                    len(react_steps) + 1,
                    "Access planning should separate open access from login-required papers.",
                    "build_download_plan",
                    str(exc),
                    tool="build_download_plan",
                    ok=False,
                )
                react_steps.append(step)
                artifacts.append(_react_artifact(self.name, step))
                return (
                    AgentRunResult(
                        agent=self.name,
                        status="failed",
                        react_trace=ReActTrace(
                            allowed_tools=ACCESS_PARSING_ALLOWED_TOOLS,
                            steps=_with_next_actions(react_steps),
                            final_observation=str(exc),
                            stop_reason="tool_failed",
                        ),
                        artifacts=artifacts,
                        warnings=[str(exc)],
                    ),
                    workspace,
                )
            react_steps.append(
                _react_step(
                    len(react_steps) + 1,
                    "Access planning should separate open access from login-required papers.",
                    "build_download_plan",
                    f"downloadable={getattr(plan, 'downloadable_count', 0)}",
                    tool="build_download_plan",
                )
            )
            artifacts.append(
                AgentArtifact(kind="download_plan", producer=self.name, payload=plan.model_dump())
            )
        if intent in {"parse", "access_and_parse"}:
            try:
                workspace, parse_report = await parse_workspace_skill(
                    workspace,
                    config,
                    context=ToolCallContext(
                        caller=self.name,
                        task_id=message.task_id,
                        intent=message.intent,
                        react_step=len(react_steps) + 1,
                        metadata={"active_papers": len(workspace.context.active_papers)},
                    ),
                )
            except RuntimeError as exc:
                step = _react_step(
                    len(react_steps) + 1,
                    "The workspace has local PDFs or parse failures worth recording.",
                    "parse_workspace_papers",
                    str(exc),
                    tool="parse_workspace_papers",
                    ok=False,
                )
                react_steps.append(step)
                artifacts.append(_react_artifact(self.name, step))
                return (
                    AgentRunResult(
                        agent=self.name,
                        status="failed",
                        react_trace=ReActTrace(
                            allowed_tools=ACCESS_PARSING_ALLOWED_TOOLS,
                            steps=_with_next_actions(react_steps),
                            final_observation=str(exc),
                            stop_reason="tool_failed",
                        ),
                        artifacts=artifacts,
                        warnings=[str(exc)],
                    ),
                    workspace,
                )
            react_steps.append(
                _react_step(
                    len(react_steps) + 1,
                    "The workspace has local PDFs or parse failures worth recording.",
                    "parse_workspace_papers",
                    f"parsed={parse_report.get('parsed_count', 0)}",
                    tool="parse_workspace_papers",
                )
            )
            artifacts.append(_react_artifact(self.name, react_steps[-1]))
            artifacts.append(AgentArtifact(kind="parse_report", producer=self.name, payload=parse_report))
        return (
            AgentRunResult(
                agent=self.name,
                react_trace=ReActTrace(
                    allowed_tools=ACCESS_PARSING_ALLOWED_TOOLS,
                    planned_actions=_planned_access_actions(message.intent),
                    steps=_with_next_actions(react_steps),
                    final_observation="access/parsing turn complete" if react_steps else None,
                    stop_reason="completed" if react_steps else "no_action",
                ),
                artifacts=artifacts,
            ),
            workspace,
        )


@dataclass(frozen=True)
class RetrievalAccessParsingAgent:
    name: str = "retrieval_access_parsing"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        artifacts: list[AgentArtifact] = []
        react_steps: list[ReActStep] = []
        warnings: list[str] = []
        status = "completed"
        if message.intent in {"search", "retrieve", "research"}:
            retrieval_result, workspace = await RetrievalAgent().run(message, workspace, config)
            artifacts.extend(retrieval_result.artifacts)
            warnings.extend(retrieval_result.warnings)
            if retrieval_result.react_trace:
                react_steps.extend(retrieval_result.react_trace.steps)
            if retrieval_result.status != "completed":
                status = retrieval_result.status
        if message.intent in {
            "resolve_full_text",
            "download_plan",
            "access",
            "parse",
            "access_and_parse",
            "research",
        }:
            access_intent = "access_and_parse" if message.intent == "research" else message.intent
            access_message = message.model_copy(update={"intent": access_intent})
            access_result, workspace = await AccessParsingAgent().run(
                access_message,
                workspace,
                config,
            )
            artifacts.extend(access_result.artifacts)
            warnings.extend(access_result.warnings)
            if access_result.react_trace:
                offset = len(react_steps)
                react_steps.extend(
                    step.model_copy(update={"step_index": offset + step.step_index})
                    for step in access_result.react_trace.steps
                )
            if access_result.status != "completed":
                status = access_result.status
        final = "retrieval/access/parsing complete" if status == "completed" else "agent failed"
        return (
            AgentRunResult(
                agent=self.name,
                status=status,
                artifacts=artifacts,
                react_trace=ReActTrace(
                    allowed_tools=[*RETRIEVAL_ALLOWED_TOOLS, *ACCESS_PARSING_ALLOWED_TOOLS],
                    planned_actions=_planned_retrieval_access_actions(message.intent),
                    steps=_with_next_actions(react_steps),
                    final_observation=final,
                    stop_reason="completed" if status == "completed" else "tool_failed",
                ),
                warnings=warnings,
            ),
            workspace,
        )


@dataclass(frozen=True)
class EvidenceAgent:
    name: str = "evidence"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        papers = [workspace.papers[pid] for pid in workspace.context.active_papers]
        try:
            audit = await audit_citation_links_skill(
                papers,
                config,
                context=ToolCallContext(
                    caller=self.name,
                    task_id=message.task_id,
                    intent=message.intent,
                ),
            )
        except RuntimeError as exc:
            return (
                AgentRunResult(
                    agent=self.name,
                    status="failed",
                    warnings=[str(exc)],
                ),
                workspace,
            )
        return (
            AgentRunResult(
                agent=self.name,
                artifacts=[
                    AgentArtifact(
                        kind="citation_audit",
                        producer=self.name,
                        payload=audit.model_dump(mode="json"),
                    ),
                ],
            ),
            workspace,
        )


@dataclass(frozen=True)
class SynthesisAgent:
    name: str = "synthesis"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        artifacts: list[AgentArtifact] = []
        if message.intent in {"tables", "synthesize"}:
            try:
                workspace, harness = await extract_tables_skill(
                    workspace,
                    config,
                    context=ToolCallContext(
                        caller=self.name,
                        task_id=message.task_id,
                        intent=message.intent,
                    ),
                )
            except RuntimeError as exc:
                return (
                    AgentRunResult(
                        agent=self.name,
                        status="failed",
                        artifacts=artifacts,
                        warnings=[str(exc)],
                    ),
                    workspace,
                )
            matrix = build_comparison_matrix_skill(workspace)
            artifacts.extend(
                [
                    AgentArtifact(
                        kind="table_harness",
                        producer=self.name,
                        payload=harness.model_dump(mode="json"),
                    ),
                    AgentArtifact(
                        kind="comparison_matrix",
                        producer=self.name,
                        payload=matrix.model_dump(mode="json"),
                    ),
                ]
            )
        if message.intent in {"storyline", "synthesize"}:
            claims = build_storyline_skill(workspace)
            artifacts.append(
                AgentArtifact(
                    kind="storyline",
                    producer=self.name,
                    payload={"claims": [claim.model_dump(mode="json") for claim in claims]},
                )
            )
        if message.intent in {"document", "synthesize"}:
            try:
                report = await build_research_report_skill(
                    workspace,
                    config,
                    context=ToolCallContext(
                        caller=self.name,
                        task_id=message.task_id,
                        intent=message.intent,
                    ),
                )
            except RuntimeError as exc:
                return (
                    AgentRunResult(
                        agent=self.name,
                        status="failed",
                        artifacts=artifacts,
                        warnings=[str(exc)],
                    ),
                    workspace,
                )
            artifacts.append(
                AgentArtifact(
                    kind="research_document",
                    producer=self.name,
                    payload=report.model_dump(mode="json"),
                )
            )
        return AgentRunResult(agent=self.name, artifacts=artifacts), workspace


@dataclass(frozen=True)
class EvaluationAgent:
    name: str = "evaluation"

    async def run(
        self,
        message: AgentMessage,
        workspace: LiteratureWorkspace,
        config: LitTraceConfig,
    ) -> tuple[AgentRunResult, LiteratureWorkspace]:
        quality = build_quality_report_skill(config, workspace)
        metrics = dict(quality.metrics)
        interactions = build_agent_interaction_report(workspace)
        return (
            AgentRunResult(
                agent=self.name,
                artifacts=[
                    AgentArtifact(
                        kind="quality_metrics",
                        producer=self.name,
                        payload=metrics,
                    ),
                    AgentArtifact(
                        kind="agent_interactions",
                        producer=self.name,
                        payload=interactions.model_dump(mode="json"),
                    ),
                ],
            ),
            workspace,
        )


def default_runtime_agents() -> dict[str, RuntimeAgent]:
    agents: list[RuntimeAgent] = [
        PlannerAgent(),
        RetrievalAccessParsingAgent(),
        EvidenceAgent(),
        SynthesisAgent(),
        EvaluationAgent(),
    ]
    return {agent.name: agent for agent in agents}
