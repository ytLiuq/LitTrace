import httpx
import pytest

from littrace.config import LLMConfig, LitTraceConfig
from littrace.llm import chat_completion


@pytest.mark.anyio
async def test_chat_completion_uses_fallback_endpoint(monkeypatch):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "primary.example.com" in str(request.url):
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"actions":["search"]}'}}]},
        )

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("littrace.llm.httpx.AsyncClient", MockAsyncClient)

    reply = await chat_completion(
        LitTraceConfig(
            llm=LLMConfig(
                api_key="primary-key",
                base_url="https://primary.example.com",
                model="primary-model",
                fallback_models=["backup-model"],
                fallback_api_key="fallback-key",
                fallback_base_url="https://fallback.example.com",
                fallback_model="fallback-model",
            )
        ),
        "Return JSON.",
        "Find papers.",
    )

    assert reply.used_llm
    assert "primary.example.com" in calls[0]
    assert any("primary.example.com" in call for call in calls[1:])
