from __future__ import annotations

import asyncio

from littrace.config import LitTraceConfig
from littrace.downloads import CDPDownloadSession


def test_each_paper_keeps_its_own_tab_until_workflow_close(monkeypatch) -> None:
    browsers = []

    class FakeBrowser:
        def __init__(self, *args, **kwargs):
            self.index = len(browsers) + 1
            self.tab_id = None
            self.closed = False
            browsers.append(self)

        def connect_new_tab(self):
            self.tab_id = f"tab-{self.index}"

        def close_tab(self):
            self.closed = True
            self.tab_id = None

    monkeypatch.setattr("littrace.access_layer.cdp_core.CDPBrowser", FakeBrowser)

    async def run() -> None:
        session = CDPDownloadSession(LitTraceConfig())
        session._prepared = True
        session._available = True
        first = await session.browser_for("paper-1")
        second = await session.browser_for("paper-2")
        first.connect_new_tab()
        second.connect_new_tab()

        assert first is not second
        assert first.tab_id != second.tab_id
        assert not first.closed and not second.closed

        await session.close()
        assert first.closed and second.closed

    asyncio.run(run())
