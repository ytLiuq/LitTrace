from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.runtime.agents import RuntimeAgent, default_runtime_agents
from littrace.runtime.messages import AgentArtifact, AgentMessage, AgentRunResult


class AgentRuntime:
    """Small in-process runtime for typed LitTrace agent turns."""

    def __init__(
        self,
        agents: dict[str, RuntimeAgent] | None = None,
        workspace: LiteratureWorkspace | None = None,
    ) -> None:
        self.agents = agents or default_runtime_agents()
        self.workspace = workspace or LiteratureWorkspace()
        self.messages: list[AgentMessage] = []
        self.artifacts: dict[str, AgentArtifact] = {}

    async def send(self, message: AgentMessage, config: LitTraceConfig) -> AgentRunResult:
        if message.receiver not in self.agents:
            return AgentRunResult(
                agent=message.receiver,
                status="failed",
                warnings=[f"Unknown runtime agent: {message.receiver}"],
            )
        self.messages.append(message)
        result, self.workspace = await self.agents[message.receiver].run(
            message,
            self.workspace,
            config,
        )
        for artifact in result.artifacts:
            self.artifacts[artifact.artifact_id] = artifact
        self.messages.extend(result.messages)
        return result

    def artifact_list(self) -> list[AgentArtifact]:
        return list(self.artifacts.values())
