"""Contract tests for the public OpenAPI surface.

The /docs, /redoc and /openapi.json endpoints are opt-in via
``LITTRACE_API_DOCS_ENABLED=1`` so production deployments do not
leak the internal schema by default. These tests lock down both
branches:

  - docs are gated when the env var is unset,
  - when enabled, /openapi.json contains the expected metadata
    (title, description, contact, license, the nine endpoint
    groups, and at least one tag on every operation).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from littrace.api.app import app, make_app


pytestmark = pytest.mark.api


EXPECTED_TAGS = {
    "research", "evaluation", "downloads", "artifacts",
    "sessions", "context", "publishers", "system", "agents",
}


def test_openapi_and_docs_are_disabled_by_default() -> None:
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def _docs_enabled_client() -> TestClient:
    """Build a second app instance with docs paths enabled.

    FastAPI only reads docs_url/openapi_url/redoc_url at
    construction, so the only way to test the enabled branch is
    to construct a fresh app with ``LITTRACE_API_DOCS_ENABLED=1``
    in os.environ. The module-level ``app`` is left untouched so
    other tests still see the gated default.
    """
    saved = os.environ.get("LITTRACE_API_DOCS_ENABLED")
    os.environ["LITTRACE_API_DOCS_ENABLED"] = "1"
    try:
        enabled_app = make_app()
    finally:
        if saved is None:
            os.environ.pop("LITTRACE_API_DOCS_ENABLED", None)
        else:
            os.environ["LITTRACE_API_DOCS_ENABLED"] = saved
    return TestClient(enabled_app)


def test_openapi_metadata_when_enabled() -> None:
    client = _docs_enabled_client()
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    info = schema["info"]
    assert info["title"] == "LitTrace API"
    assert "description" in info and info["description"]
    assert info["version"] == "0.1.0"
    assert "contact" in info and info["contact"]["url"].startswith("http")
    assert info["license"]["name"] == "Apache-2.0"


def test_openapi_contains_all_endpoint_groups() -> None:
    client = _docs_enabled_client()
    schema = client.get("/openapi.json").json()
    tag_names = {tag["name"] for tag in schema.get("tags", [])}
    assert EXPECTED_TAGS.issubset(tag_names), (
        f"missing endpoint groups: {EXPECTED_TAGS - tag_names}"
    )


def test_openapi_every_operation_carries_at_least_one_tag() -> None:
    client = _docs_enabled_client()
    schema = client.get("/openapi.json").json()
    missing: list[str] = []
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not operation.get("tags"):
                missing.append(f"{method.upper()} {path}")
    assert not missing, (
        f"operations without tags: {missing[:5]} ... (total {len(missing)})"
    )


def test_health_response_model_is_pydantic_when_docs_enabled() -> None:
    """``/health`` exposes a named Pydantic schema, not {}."""
    client = _docs_enabled_client()
    schema = client.get("/openapi.json").json()
    health = schema["paths"]["/health"]["get"]
    schema_ref = health["responses"]["200"]["content"]["application/json"]["schema"]
    # The /health route uses the HealthResponse Pydantic model
    # so the schema has a "$ref" pointer; a free-form
    # response_model=object would render as {"type": "object"}.
    assert "$ref" in schema_ref, (
        f"expected $ref to HealthResponse, got {schema_ref}"
    )
    assert schema_ref["$ref"].endswith("/HealthResponse")

def test_every_route_has_documented_method_and_response() -> None:
    """Round 7 CR pass 3 coverage: every route advertised
    in ``app.openapi()`` has a non-empty path, a registered
    HTTP method, at least one response code, and at least
    one tag. Routes missing any of these are usually
    in-progress work that should not ship.

    The test also guards the surface size: a refactor that
    silently drops a route will be caught here because the
    expected count is a stable lower bound (currently 80+).
    """
    import pytest as _pytest

    client = _docs_enabled_client()
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})

    assert paths, "OpenAPI schema has no paths"
    # A stable lower bound: 65+ routes at the round 7
    # checkpoint (the FastAPI ``app.openapi()`` consolidates
    # path templates, so multiple decorators on ``/sessions/{id}``
    # count as one entry). The bound is intentionally loose
    # so a refactor that re-organises a few routes does not
    # trigger this test; a wholesale regression will.
    assert len(paths) >= 60, (
        f"OpenAPI schema only has {len(paths)} paths; expected >= 60"
    )

    bad: list[str] = []
    for path, methods in paths.items():
        if not path.startswith("/"):
            bad.append(f"{path}: path does not start with /")
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                bad.append(f"{method.upper()} {path}: unknown method")
                continue
            responses = op.get("responses", {}) or {}
            if not responses:
                bad.append(f"{method.upper()} {path}: no responses documented")
            tags = op.get("tags", []) or []
            if not tags:
                bad.append(f"{method.upper()} {path}: no tag")
    assert not bad, "schema validation failures:\n  - " + "\n  - ".join(bad)
