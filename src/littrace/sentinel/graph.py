from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from littrace.config import LitTraceConfig
from littrace.models import LiteratureWorkspace
from littrace.sentinel.agent import LiteratureSentinel, SentinelRunResult
from littrace.sentinel.state import Watchlist


class SentinelGraphState(TypedDict, total=False):
    config: LitTraceConfig
    watchlist: Watchlist
    sentinel: LiteratureSentinel
    result: SentinelRunResult
    workspace: LiteratureWorkspace
    summary: dict[str, object]


def build_sentinel_graph():
    graph = StateGraph(SentinelGraphState)

    async def prepare_task(state: SentinelGraphState) -> SentinelGraphState:
        sentinel = state["sentinel"]
        state["watchlist"] = sentinel.watchlist
        return state

    async def run_sentinel(state: SentinelGraphState) -> SentinelGraphState:
        result = await state["sentinel"].run()
        state["result"] = result
        state["workspace"] = result.workspace
        state["summary"] = result.summary.model_dump(mode="json")
        return state

    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_sentinel", run_sentinel)
    graph.set_entry_point("prepare_task")
    graph.add_edge("prepare_task", "run_sentinel")
    graph.add_edge("run_sentinel", END)
    return graph.compile()


async def run_sentinel_graph(config: LitTraceConfig, watchlist: Watchlist) -> SentinelGraphState:
    sentinel = LiteratureSentinel(config, watchlist)
    graph = build_sentinel_graph()
    return await graph.ainvoke(
        {"config": config, "watchlist": watchlist, "sentinel": sentinel}
    )
