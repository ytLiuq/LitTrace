from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentArtifact(BaseModel):
    """A typed artifact produced by a runtime agent."""

    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str
    producer: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ReActStep(BaseModel):
    step_index: int
    thought: str
    decision: str = "act"
    action: str
    observation: str
    tool: str | None = None
    next_action: str | None = None
    ok: bool = True


class ReActTrace(BaseModel):
    strategy: str = "bounded_react"
    max_steps: int = 4
    allowed_tools: list[str] = Field(default_factory=list)
    planned_actions: list[str] = Field(default_factory=list)
    steps: list[ReActStep] = Field(default_factory=list)
    final_observation: str | None = None
    stop_reason: str | None = None

    @property
    def completed(self) -> bool:
        return bool(self.final_observation)


class AgentMessage(BaseModel):
    """Message envelope for agent-to-agent communication."""

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    sender: str
    receiver: str
    intent: str
    payload: dict[str, Any] = Field(default_factory=dict)
    required_artifacts: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AgentRunResult(BaseModel):
    """Result envelope returned by a runtime agent."""

    agent: str
    status: str = "completed"
    artifacts: list[AgentArtifact] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    react_trace: ReActTrace | None = None
    warnings: list[str] = Field(default_factory=list)
