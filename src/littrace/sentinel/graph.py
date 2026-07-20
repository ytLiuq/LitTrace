from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.sentinel.agent import LiteratureSentinelAgent, SentinelRunResult
from littrace.sentinel.state import Watchlist


class SentinelGraphState(TypedDict, total=False):
    config: LitTraceConfig
    watchlist: Watchlist
    agent: LiteratureSentinelAgent
    result: SentinelRunResult
    workspace: LiteratureWorkspace
    summary: dict[str, object]


def build_sentinel_graph():
    graph = StateGraph(SentinelGraphState)

    async def prepare_task(state: SentinelGraphState) -> SentinelGraphState:
        agent = state["agent"]
        state["watchlist"] = agent.watchlist
        return state

    async def run_agent(state: SentinelGraphState) -> SentinelGraphState:
        result = await state["agent"].run()
        state["result"] = result
        state["workspace"] = result.workspace
        state["summary"] = result.summary.model_dump(mode="json")
        return state

    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_agent", run_agent)
    graph.set_entry_point("prepare_task")
    graph.add_edge("prepare_task", "run_agent")
    graph.add_edge("run_agent", END)
    return graph.compile()


async def run_sentinel_graph(config: LitTraceConfig, watchlist: Watchlist) -> SentinelGraphState:
    agent = LiteratureSentinelAgent(config, watchlist)
    graph = build_sentinel_graph()
    return await graph.ainvoke({"config": config, "watchlist": watchlist, "agent": agent})
