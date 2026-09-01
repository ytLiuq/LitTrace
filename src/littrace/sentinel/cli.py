from __future__ import annotations

from littrace.config import LitTraceConfig
from littrace.sentinel.agent import LiteratureSentinel
from littrace.sentinel.state import Watchlist
from littrace.sentinel.storage import (
    ensure_sentinel_store,
    load_sentinel_state,
    load_watchlist,
    save_sentinel_state,
    save_watchlist,
)


def init_sentinel(config: LitTraceConfig, watchlist_id: str, topic: str) -> str:
    watchlist = Watchlist(watchlist_id=watchlist_id, topic=topic, objective=topic)
    store = ensure_sentinel_store(config, watchlist)
    save_watchlist(store, watchlist)
    save_sentinel_state(store, load_sentinel_state(store))
    return str(store.root)


async def run_sentinel(config: LitTraceConfig, watchlist_or_id, topic: str | None = None):
    """Run the daily sentinel against ``watchlist_or_id``.

    Round 17: accepts either a ``Watchlist`` instance (new callers
    that already pre-applied CLI overrides) or a bare ``str``
    watchlist id (legacy callers — the function still loads +
    updates the watchlist itself, so a CLI that doesn't have the
    override plumbing keeps working).
    """
    if isinstance(watchlist_or_id, Watchlist):
        watchlist = watchlist_or_id
    else:
        watchlist_id = watchlist_or_id
        store = ensure_sentinel_store(
            config,
            Watchlist(watchlist_id=watchlist_id, topic=topic or watchlist_id),
        )
        watchlist = load_watchlist(store)
        if topic:
            watchlist = watchlist.model_copy(update={"topic": topic, "objective": topic})
    sentinel = LiteratureSentinel(config, watchlist)
    return await sentinel.run()


def access_review(config: LitTraceConfig, watchlist_id: str, topic: str | None = None):
    store = ensure_sentinel_store(config, Watchlist(watchlist_id=watchlist_id, topic=topic or watchlist_id))
    watchlist = load_watchlist(store)
    if topic:
        watchlist = watchlist.model_copy(update={"topic": topic, "objective": topic})
    sentinel = LiteratureSentinel(config, watchlist)
    return sentinel.access_review()


async def resume_after_login(config: LitTraceConfig, watchlist_id: str, topic: str | None = None):
    store = ensure_sentinel_store(config, Watchlist(watchlist_id=watchlist_id, topic=topic or watchlist_id))
    watchlist = load_watchlist(store)
    if topic:
        watchlist = watchlist.model_copy(update={"topic": topic, "objective": topic})
    sentinel = LiteratureSentinel(config, watchlist)
    return await sentinel.resume_after_login()
