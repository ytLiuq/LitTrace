"""Stateless LitTrace tools exposed to Codex App Server."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field

from littrace.codex_runtime.errors import (
    AppServerError,
    CodexErrorCode,
)
from littrace.config import LitTraceConfig
from littrace.evaluation.quality_report import build_quality_report
from littrace.models import LiteratureWorkspace, coerce_parsed
from littrace.retrieval.rag_search import search_workspace_rag
from littrace.state_db import AgentThreadBindingRecord, AsyncTaskRecord, StateStore


class McpError(BaseModel):
    """Mirror of codex-harness's structured error envelope."""

    code: CodexErrorCode
    message: str
    details: dict[str, Any] | None = None


class McpResponse(BaseModel):
    """Uniform ``{success, error, data, warnings}`` envelope.

    Every gateway tool returns an instance of this class. The
    serialised form (via ``model_dump(mode="json")``) is what the
    MCP stdio server writes to ``TextContent.text``. Downstream
    consumers dispatch on ``success`` first and then look at
    ``data`` / ``error`` / ``warnings``.

    ``__getitem__`` and ``__contains__`` look up keys in ``data``
    first, falling back to envelope-level fields. This keeps the
    pre-envelope test contract (``assert result["xxx"] == ...``)
    working so the rollout can be incremental.
    """

    success: bool
    data: dict[str, Any] | None = None
    error: McpError | None = None
    warnings: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        if self.data is not None and key in self.data:
            return self.data[key]
        if key in ("success", "data", "error", "warnings"):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if self.data is not None and key in self.data:
            return True
        return key in ("success", "data", "error", "warnings")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def ok_envelope(
    data: dict[str, Any], warnings: list[str] | None = None
) -> McpResponse:
    return McpResponse(
        success=True, data=data, warnings=warnings or [],
    )


def err_envelope(
    code: CodexErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> McpResponse:
    return McpResponse(
        success=False,
        error=McpError(code=code, message=message, details=details),
    )

READ_ONLY_TOOL_NAMES = (
    "get_workspace_context",
    "get_download_jobs",
    "get_parse_jobs",
    "get_table_jobs",
    "get_storyline_jobs",
    "search_workspace_rag",
    "get_paper_status",
    "get_evidence",
    "quality_report",
)
WRITE_TOOL_NAMES = (
    "set_download_selection",
    "search_papers",
    "enqueue_download",
    "enqueue_parse",
    "enqueue_table_extraction",
    "enqueue_storyline",
)
APP_SERVER_TOOL_NAMES = READ_ONLY_TOOL_NAMES + WRITE_TOOL_NAMES


def read_only_tool_specs() -> list[dict[str, Any]]:
    """MCP tool declarations kept separate from the SDK adapter for tests."""

    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        {
            "name": "get_workspace_context",
            "description": (
                "Read the canonical LitTrace workspace summary for the current Codex thread. "
                "Use this before answering questions about the active literature session."
            ),
            "inputSchema": empty_schema,
        },
        {
            "name": "get_download_jobs",
            "description": (
                "Read recent durable download jobs for the current LitTrace session, "
                "including queued, running, completed, failed, and dead states."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_parse_jobs",
            "description": (
                "Read recent durable PDF parsing jobs for the current LitTrace session, "
                "including queued, running, completed, failed, and dead states."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_table_jobs",
            "description": (
                "Read recent durable performance-table extraction jobs for the current "
                "LitTrace session."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_storyline_jobs",
            "description": (
                "Read recent durable evidence-grounded storyline jobs for the current "
                "LitTrace session."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "search_workspace_rag",
            "description": "Semantically search indexed full-text evidence in the current workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_paper_status",
            "description": "Read metadata, selection, download-resolution, and parse status for one paper.",
            "inputSchema": {
                "type": "object",
                "properties": {"paper_id": {"type": "string"}},
                "required": ["paper_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_evidence",
            "description": "Read traceable evidence records and performance cells from the workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "quality_report",
            "description": "Compute LitTrace's read-only evidence and citation quality report.",
            "inputSchema": empty_schema,
        },
    ]


def app_server_tool_specs() -> list[dict[str, Any]]:
    """All tools available to the isolated LitTrace App Server runtime."""

    return [
        *read_only_tool_specs(),
        {
            "name": "set_download_selection",
            "description": (
                "Atomically add, remove, replace, or clear the active papers selected for "
                "download. Read get_workspace_context first and pass its workspace_revision. "
                "Use one stable idempotency_key for all retries of the same intended change."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["add", "remove", "replace", "clear"],
                    },
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 100,
                        "default": [],
                    },
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["mode", "expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_papers",
            "description": (
                "Search external scholarly sources and atomically replace the canonical "
                "LitTrace workspace with the ranked result set. Read get_workspace_context "
                "first, pass its workspace_revision, and reuse one idempotency_key for retries."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "minLength": 2, "maxLength": 1000},
                    "year_min": {
                        "type": ["integer", "null"],
                        "minimum": 1800,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 40,
                    },
                    "live": {"type": "boolean", "default": True},
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["topic", "expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "enqueue_download",
            "description": (
                "Atomically enqueue a durable LitTrace download job for active papers. "
                "If paper_ids is omitted, use the current selected_for_download set. "
                "Read get_workspace_context first, pass its workspace_revision, and reuse "
                "one idempotency_key for retries of the same request. This command queues "
                "work; use get_download_jobs to observe execution."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 100,
                    },
                    "target": {
                        "type": "string",
                        "enum": ["local_and_storage", "storage_only"],
                        "default": "local_and_storage",
                    },
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "enqueue_parse",
            "description": (
                "Atomically enqueue durable PDF parsing for active papers whose PDFs are "
                "registered in LitTrace artifact storage. If paper_ids is omitted, parse "
                "all active papers. Read get_workspace_context first, pass its workspace "
                "revision, and reuse one idempotency_key for retries. This command queues "
                "work; use get_parse_jobs to observe execution."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 100,
                    },
                    "parse_strategy": {
                        "type": "string",
                        "enum": ["auto", "text_only", "ocr"],
                        "default": "auto",
                    },
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "enqueue_table_extraction",
            "description": (
                "Atomically enqueue durable performance metric and structured-table "
                "extraction from parsed active papers. If paper_ids is omitted, process all "
                "active parsed papers. Read get_workspace_context first, pass its workspace "
                "revision, and reuse one idempotency_key for retries. Use get_table_jobs for "
                "progress."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 100,
                    },
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
        {
            "name": "enqueue_storyline",
            "description": (
                "Atomically enqueue an evidence-grounded storyline job from parsed active "
                "papers. If paper_ids is omitted, use all active parsed papers. Read "
                "get_workspace_context first, pass its workspace revision, reuse one "
                "idempotency_key for retries, and use get_storyline_jobs for progress."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 100,
                    },
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 200,
                    },
                },
                "required": ["expected_revision", "idempotency_key"],
                "additionalProperties": False,
            },
        },
    ]


class LitTraceToolGateway:
    """Resolve every call from App Server thread identity to Postgres state."""

    def __init__(self, config: LitTraceConfig, state_store: StateStore) -> None:
        self.config = config
        self.state_store = state_store
        # Round 13 step 3: third-party plugins installed via
        # the ``littrace.mcp_servers`` entry-point group
        # can extend the tool catalog without modifying this
        # file. ``external_tools`` is the merged name set;
        # ``external_handlers`` maps tool name -> async
        # callable. ``apply_external_plugins`` walks the
        # marketplace discovery result and registers each
        # plugin's tools.
        self.external_tools: dict[str, dict[str, Any]] = {}
        self.external_handlers: dict[str, Any] = {}

    def register_external_tool(
        self,
        *,
        name: str,
        spec: dict[str, Any],
        handler: Any,
    ) -> None:
        """Install one third-party MCP tool into the gateway.

        ``spec`` is the JSON-Schema-shaped tool description the
        App Server uses to advertise the tool to the model;
        ``handler`` is the async callable that resolves the
        tool's call. The gateway's ``call`` method already
        delegates to ``external_handlers`` once a tool name
        is registered, so the only contract the plugin has
        to honour is ``async def handler(name, args, *,
        codex_thread_id) -> dict[str, Any]``.
        """
        if not name or not isinstance(spec, dict):
            raise ValueError("third-party tool requires a non-empty name and dict spec")
        self.external_tools[name] = spec
        self.external_handlers[name] = handler

    def list_external_tool_specs(self) -> list[dict[str, Any]]:
        """JSON-Schema list used by the App Server thread/start
        payload to advertise the registered third-party tools
        alongside the 15 built-in LitTrace tools.
        """
        return list(self.external_tools.values())

    def load_workspace(
        self,
        codex_thread_id: str,
    ) -> tuple[AgentThreadBindingRecord, LiteratureWorkspace]:
        if not codex_thread_id:
            raise PermissionError("Missing App Server _meta.threadId")
        binding = self.state_store.get_agent_thread_binding_by_thread_id(codex_thread_id)
        if binding is None or binding.status != "active":
            raise PermissionError("Codex thread is not bound to an active LitTrace session")
        state = self.state_store.get_session_state(binding.session_id)
        if state is None:
            raise LookupError(f"LitTrace session state does not exist: {binding.session_id}")
        workspace = LiteratureWorkspace.model_validate(state.workspace_json)
        return binding, workspace

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        codex_thread_id: str,
    ) -> dict[str, Any]:
        # Round 13 step 3: third-party MCP servers register
        # via the gateway's ``external_handlers`` map; the
        # built-in allowlist is still the source of truth for
        # the 15 LitTrace tools.
        if name in self.external_handlers:
            handler = self.external_handlers[name]
            return await handler(
                name, arguments or {}, codex_thread_id=codex_thread_id,
            )
        if name not in APP_SERVER_TOOL_NAMES:
            raise PermissionError(f"Tool is not in the App Server allowlist: {name}")
        args = arguments or {}
        binding, workspace = self.load_workspace(codex_thread_id)
        if name == "set_download_selection":
            return self._set_download_selection(binding, workspace, args)
        if name == "search_papers":
            return await self._search_papers(binding, workspace, args)
        if name == "enqueue_download":
            return self._enqueue_download(binding, workspace, args)
        if name == "enqueue_parse":
            return self._enqueue_parse(binding, workspace, args)
        if name == "enqueue_table_extraction":
            return self._enqueue_table_extraction(binding, workspace, args)
        if name == "enqueue_storyline":
            return self._enqueue_storyline(binding, workspace, args)
        if name == "get_workspace_context":
            return _workspace_context(binding, workspace)
        if name == "get_download_jobs":
            jobs = self.state_store.list_async_tasks(
                session_id=binding.session_id,
                kind="download_job",
                limit=_bounded_int(args.get("limit"), default=20, minimum=1, maximum=100),
            )
            return {
                "session_id": binding.session_id,
                "jobs": [_download_job_summary(job) for job in jobs],
            }
        if name == "get_parse_jobs":
            jobs = self.state_store.list_async_tasks(
                session_id=binding.session_id,
                kind="parse_job",
                limit=_bounded_int(args.get("limit"), default=20, minimum=1, maximum=100),
            )
            return {
                "session_id": binding.session_id,
                "jobs": [_parse_job_summary(job) for job in jobs],
            }
        if name == "get_table_jobs":
            jobs = self.state_store.list_async_tasks(
                session_id=binding.session_id,
                kind="table_job",
                limit=_bounded_int(args.get("limit"), default=20, minimum=1, maximum=100),
            )
            return {
                "session_id": binding.session_id,
                "jobs": [_table_job_summary(job) for job in jobs],
            }
        if name == "get_storyline_jobs":
            jobs = self.state_store.list_async_tasks(
                session_id=binding.session_id,
                kind="storyline_job",
                limit=_bounded_int(args.get("limit"), default=20, minimum=1, maximum=100),
            )
            return {
                "session_id": binding.session_id,
                "jobs": [_storyline_job_summary(job) for job in jobs],
            }
        if name == "get_paper_status":
            return _paper_status(workspace, str(args.get("paper_id") or ""))
        if name == "get_evidence":
            return _evidence(
                workspace,
                paper_id=str(args.get("paper_id") or "") or None,
                limit=_bounded_int(args.get("limit"), default=20, minimum=1, maximum=100),
            )
        if name == "quality_report":
            report = build_quality_report(self.config, workspace)
            return report.model_dump(mode="json")
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        result = await search_workspace_rag(
            self.config,
            workspace,
            query,
            top_k=_bounded_int(args.get("top_k"), default=5, minimum=1, maximum=20),
        )
        if result is None:
            return {"enabled": False, "hits": []}
        return {
            "enabled": True,
            "profile_id": result.profile.profile_id,
            "hits": [hit.model_dump(mode="json") for hit in result.hits],
        }

    def _set_download_selection(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        mode = str(arguments.get("mode") or "")
        if mode not in {"add", "remove", "replace", "clear"}:
            raise ValueError("mode must be add, remove, replace, or clear")
        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")
        raw_ids = arguments.get("paper_ids", [])
        if not isinstance(raw_ids, list):
            raise TypeError("paper_ids must be an array")
        paper_ids = list(dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip()))
        if len(paper_ids) > 100:
            raise ValueError("paper_ids must contain at most 100 items")

        active_ids = list(workspace.context.active_papers)
        if mode != "clear":
            invalid = [paper_id for paper_id in paper_ids if paper_id not in active_ids]
            if invalid:
                raise ValueError(
                    "Only active workspace papers can be selected for download: "
                    + ", ".join(invalid[:10])
                )
        previous = [
            paper_id
            for paper_id in workspace.context.selected_for_download
            if paper_id in active_ids
        ]
        if mode == "clear":
            selected: list[str] = []
        elif mode == "replace":
            selected = paper_ids
        elif mode == "add":
            selected = [*previous, *(item for item in paper_ids if item not in previous)]
        else:
            removals = set(paper_ids)
            selected = [item for item in previous if item not in removals]

        updated = workspace.model_copy(deep=True)
        updated.context.selected_for_download = selected
        updated.context.filters.workspace_revision = expected_revision + 1
        command = {
            "mode": mode,
            "paper_ids": paper_ids,
            "expected_revision": expected_revision,
        }
        result = {
            "session_id": binding.session_id,
            "tool": "set_download_selection",
            "idempotency_key": idempotency_key,
            "previous_selection": previous,
            "selected_for_download": selected,
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="set_download_selection",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
        )

    async def _search_papers(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from littrace.models import PaperSearchRequest
        from littrace.retrieval.search import build_query_variants
        from littrace.workflow import run_search_preview

        topic = str(arguments.get("topic") or "").strip()
        if not 2 <= len(topic) <= 1000:
            raise ValueError("topic must contain 2 to 1000 characters")
        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")
        limit = _strict_bounded_int(
            arguments.get("limit", 40),
            name="limit",
            minimum=1,
            maximum=100,
        )
        raw_year_min = arguments.get("year_min", 2023)
        year_min = None
        if raw_year_min is not None:
            year_min = _strict_bounded_int(
                raw_year_min,
                name="year_min",
                minimum=1800,
                maximum=datetime.now(UTC).year + 1,
            )
        live = arguments.get("live", True)
        if not isinstance(live, bool):
            raise TypeError("live must be a boolean")
        request = PaperSearchRequest(
            topic=topic,
            year_min=year_min,
            limit=limit,
            live=live,
            query_variants=build_query_variants(topic),
        )
        updated = await run_search_preview(request, self.config)
        # A search replaces the result workspace but must not erase the
        # session's accepted long-term research background.
        old_filters = workspace.context.filters
        new_filters = updated.context.filters
        for field_name in (
            "research_background",
            "research_retrieval_policy",
            "research_background_status",
            "research_background_rejection_reason",
            "research_background_set_at",
            "research_background_last_sync_at",
            "research_background_last_downloaded_count",
            "research_background_last_parsed_count",
        ):
            setattr(new_filters, field_name, getattr(old_filters, field_name))
        new_filters.workspace_revision = expected_revision + 1
        command = {
            "topic": topic,
            "year_min": year_min,
            "limit": limit,
            "live": live,
            "expected_revision": expected_revision,
        }
        active_papers = [
            updated.papers[paper_id]
            for paper_id in updated.context.active_papers
            if paper_id in updated.papers
        ]
        result = {
            "session_id": binding.session_id,
            "tool": "search_papers",
            "idempotency_key": idempotency_key,
            "topic": topic,
            "search_mode": new_filters.search_mode,
            "paper_count": len(updated.papers),
            "active_paper_count": len(active_papers),
            "papers": [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "journal": paper.journal,
                    "doi": paper.doi,
                }
                for paper in active_papers
            ],
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="search_papers",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
        )

    def _enqueue_download(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")
        target = str(arguments.get("target") or "local_and_storage")
        if target not in {"local_and_storage", "storage_only"}:
            raise ValueError("target must be local_and_storage or storage_only")

        raw_ids = arguments.get("paper_ids")
        if raw_ids is None:
            paper_ids = list(workspace.context.selected_for_download)
        elif isinstance(raw_ids, list):
            paper_ids = list(
                dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip())
            )
        else:
            raise TypeError("paper_ids must be an array")
        if not paper_ids:
            raise ValueError(
                "No papers were provided and selected_for_download is empty"
            )
        if len(paper_ids) > 100:
            raise ValueError("paper_ids must contain at most 100 items")
        active_ids = set(workspace.context.active_papers)
        invalid = [paper_id for paper_id in paper_ids if paper_id not in active_ids]
        if invalid:
            raise ValueError(
                "Only active workspace papers can be queued for download: "
                + ", ".join(invalid[:10])
            )

        command = {
            "paper_ids": paper_ids,
            "target": target,
            "expected_revision": expected_revision,
        }
        arguments_sha256 = _command_sha256(command)
        task_digest = sha256(
            f"{binding.session_id}\0enqueue_download\0{idempotency_key}".encode()
        ).hexdigest()[:32]
        task_id = f"download:{task_digest}"
        task = AsyncTaskRecord(
            task_id=task_id,
            session_id=binding.session_id,
            kind="download_job",
            artifact_id=f"download_batch:{task_digest}",
            event_type="download_requested",
            source_revision=str(expected_revision),
            content_sha256=arguments_sha256,
            result_json={
                "schema_version": "littrace.download_job.v1",
                "command": {
                    **command,
                    "papers": [
                        workspace.papers[paper_id].model_dump(mode="json")
                        for paper_id in paper_ids
                    ],
                },
            },
        )
        updated = workspace.model_copy(deep=True)
        updated.context.filters.workspace_revision = expected_revision + 1
        result = {
            "session_id": binding.session_id,
            "tool": "enqueue_download",
            "idempotency_key": idempotency_key,
            "task_id": task_id,
            "status": "queued",
            "paper_ids": paper_ids,
            "target": target,
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="enqueue_download",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
            async_task=task,
        )

    def _enqueue_parse(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        from littrace.artifact_registry import artifact_registry_from_config

        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")
        parse_strategy = str(arguments.get("parse_strategy") or "auto")
        if parse_strategy not in {"auto", "text_only", "ocr"}:
            raise ValueError("parse_strategy must be auto, text_only, or ocr")

        raw_ids = arguments.get("paper_ids")
        if raw_ids is None:
            paper_ids = list(workspace.context.active_papers)
        elif isinstance(raw_ids, list):
            paper_ids = list(
                dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip())
            )
        else:
            raise TypeError("paper_ids must be an array")
        if not paper_ids:
            raise ValueError("No active papers were provided for parsing")
        if len(paper_ids) > 100:
            raise ValueError("paper_ids must contain at most 100 items")
        active_ids = set(workspace.context.active_papers)
        invalid = [paper_id for paper_id in paper_ids if paper_id not in active_ids]
        if invalid:
            raise ValueError(
                "Only active workspace papers can be queued for parsing: "
                + ", ".join(invalid[:10])
            )

        registry = artifact_registry_from_config(self.config)
        sources: list[dict[str, object]] = []
        missing: list[str] = []
        for paper_id in paper_ids:
            record = registry.find_in_session(
                f"paper_pdf:{paper_id}",
                session_id=binding.session_id,
            )
            if record is None or not record.sha256:
                missing.append(paper_id)
                continue
            sources.append(record.model_dump(mode="json"))
        if missing:
            raise ValueError(
                "PDF artifacts must be registered before parsing: "
                + ", ".join(missing[:10])
            )

        command = {
            "paper_ids": paper_ids,
            "parse_strategy": parse_strategy,
            "expected_revision": expected_revision,
        }
        arguments_sha256 = _command_sha256(command)
        task_digest = sha256(
            f"{binding.session_id}\0enqueue_parse\0{idempotency_key}".encode()
        ).hexdigest()[:32]
        task_id = f"parse:{task_digest}"
        task = AsyncTaskRecord(
            task_id=task_id,
            session_id=binding.session_id,
            kind="parse_job",
            artifact_id=f"parse_batch:{task_digest}",
            event_type="parse_requested",
            source_revision=str(expected_revision),
            content_sha256=arguments_sha256,
            result_json={
                "schema_version": "littrace.parse_job.v1",
                "command": {
                    **command,
                    "papers": [
                        workspace.papers[paper_id].model_dump(mode="json")
                        for paper_id in paper_ids
                    ],
                    "sources": sources,
                },
            },
        )
        updated = workspace.model_copy(deep=True)
        updated.context.filters.workspace_revision = expected_revision + 1
        result = {
            "session_id": binding.session_id,
            "tool": "enqueue_parse",
            "idempotency_key": idempotency_key,
            "task_id": task_id,
            "status": "queued",
            "paper_ids": paper_ids,
            "parse_strategy": parse_strategy,
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="enqueue_parse",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
            async_task=task,
        )

    def _enqueue_table_extraction(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")

        raw_ids = arguments.get("paper_ids")
        if raw_ids is None:
            paper_ids = [
                paper_id
                for paper_id in workspace.context.active_papers
                if coerce_parsed(workspace.parsed_papers.get(paper_id)).parsed
            ]
        elif isinstance(raw_ids, list):
            paper_ids = list(
                dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip())
            )
        else:
            raise TypeError("paper_ids must be an array")
        if not paper_ids:
            raise ValueError("No parsed active papers were provided for table extraction")
        if len(paper_ids) > 100:
            raise ValueError("paper_ids must contain at most 100 items")
        active_ids = set(workspace.context.active_papers)
        invalid = [paper_id for paper_id in paper_ids if paper_id not in active_ids]
        if invalid:
            raise ValueError(
                "Only active workspace papers can be queued for table extraction: "
                + ", ".join(invalid[:10])
            )
        unparsed = [
            paper_id
            for paper_id in paper_ids
            if not coerce_parsed(workspace.parsed_papers.get(paper_id)).parsed
        ]
        if unparsed:
            raise ValueError(
                "Papers must be parsed before table extraction: "
                + ", ".join(unparsed[:10])
            )

        parsed_sha256 = {
            paper_id: _model_sha256(coerce_parsed(workspace.parsed_papers[paper_id]))
            for paper_id in paper_ids
        }
        command = {
            "paper_ids": paper_ids,
            "expected_revision": expected_revision,
        }
        arguments_sha256 = _command_sha256(command)
        task_digest = sha256(
            f"{binding.session_id}\0enqueue_table_extraction\0{idempotency_key}".encode()
        ).hexdigest()[:32]
        task_id = f"table:{task_digest}"
        task = AsyncTaskRecord(
            task_id=task_id,
            session_id=binding.session_id,
            kind="table_job",
            artifact_id=f"table_batch:{task_digest}",
            event_type="table_extraction_requested",
            source_revision=str(expected_revision),
            content_sha256=arguments_sha256,
            result_json={
                "schema_version": "littrace.table_job.v1",
                "command": {
                    **command,
                    "parsed_sha256": parsed_sha256,
                },
            },
        )
        updated = workspace.model_copy(deep=True)
        updated.context.filters.workspace_revision = expected_revision + 1
        result = {
            "session_id": binding.session_id,
            "tool": "enqueue_table_extraction",
            "idempotency_key": idempotency_key,
            "task_id": task_id,
            "status": "queued",
            "paper_ids": paper_ids,
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="enqueue_table_extraction",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
            async_task=task,
        )

    def _enqueue_storyline(
        self,
        binding: AgentThreadBindingRecord,
        workspace: LiteratureWorkspace,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        expected_revision = _strict_non_negative_int(
            arguments.get("expected_revision"),
            name="expected_revision",
        )
        idempotency_key = str(arguments.get("idempotency_key") or "").strip()
        if not 8 <= len(idempotency_key) <= 200:
            raise ValueError("idempotency_key must contain 8 to 200 characters")

        raw_ids = arguments.get("paper_ids")
        if raw_ids is None:
            paper_ids = [
                paper_id
                for paper_id in workspace.context.active_papers
                if coerce_parsed(workspace.parsed_papers.get(paper_id)).parsed
            ]
        elif isinstance(raw_ids, list):
            paper_ids = list(
                dict.fromkeys(str(item).strip() for item in raw_ids if str(item).strip())
            )
        else:
            raise TypeError("paper_ids must be an array")
        if not paper_ids:
            raise ValueError("No parsed active papers were provided for storyline generation")
        if len(paper_ids) > 100:
            raise ValueError("paper_ids must contain at most 100 items")
        active_ids = set(workspace.context.active_papers)
        invalid = [paper_id for paper_id in paper_ids if paper_id not in active_ids]
        if invalid:
            raise ValueError(
                "Only active workspace papers can be queued for storyline generation: "
                + ", ".join(invalid[:10])
            )
        unparsed = [
            paper_id
            for paper_id in paper_ids
            if not coerce_parsed(workspace.parsed_papers.get(paper_id)).parsed
        ]
        if unparsed:
            raise ValueError(
                "Papers must be parsed before storyline generation: "
                + ", ".join(unparsed[:10])
            )

        source_sha256 = {
            paper_id: _model_sha256(
                {
                    "paper": workspace.papers[paper_id].model_dump(mode="json"),
                    "parsed": coerce_parsed(
                        workspace.parsed_papers[paper_id]
                    ).model_dump(mode="json"),
                }
            )
            for paper_id in paper_ids
        }
        command = {
            "paper_ids": paper_ids,
            "expected_revision": expected_revision,
        }
        arguments_sha256 = _command_sha256(command)
        task_digest = sha256(
            f"{binding.session_id}\0enqueue_storyline\0{idempotency_key}".encode()
        ).hexdigest()[:32]
        task_id = f"storyline:{task_digest}"
        task = AsyncTaskRecord(
            task_id=task_id,
            session_id=binding.session_id,
            kind="storyline_job",
            artifact_id=f"storyline_batch:{task_digest}",
            event_type="storyline_requested",
            source_revision=str(expected_revision),
            content_sha256=arguments_sha256,
            result_json={
                "schema_version": "littrace.storyline_job.v1",
                "command": {
                    **command,
                    "source_sha256": source_sha256,
                },
            },
        )
        updated = workspace.model_copy(deep=True)
        updated.context.filters.workspace_revision = expected_revision + 1
        result = {
            "session_id": binding.session_id,
            "tool": "enqueue_storyline",
            "idempotency_key": idempotency_key,
            "task_id": task_id,
            "status": "queued",
            "paper_ids": paper_ids,
            "workspace_revision": expected_revision + 1,
        }
        return self._commit_workspace_tool(
            binding=binding,
            updated=updated,
            tool_name="enqueue_storyline",
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
            command=command,
            result=result,
            async_task=task,
        )

    def _commit_workspace_tool(
        self,
        *,
        binding: AgentThreadBindingRecord,
        updated: LiteratureWorkspace,
        tool_name: str,
        idempotency_key: str,
        expected_revision: int,
        command: dict[str, Any],
        result: dict[str, Any],
        async_task: AsyncTaskRecord | None = None,
    ) -> dict[str, Any]:
        workspace_json = updated.model_dump(mode="json")
        canonical_json = json.dumps(
            workspace_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        arguments_sha256 = _command_sha256(command)
        record = self.state_store.commit_agent_workspace_tool(
            session_id=binding.session_id,
            idempotency_key=idempotency_key,
            tool_name=tool_name,
            arguments_sha256=arguments_sha256,
            expected_revision=expected_revision,
            workspace_json=workspace_json,
            workspace_sha256=sha256(canonical_json.encode("utf-8")).hexdigest(),
            result_json=result,
            audit_event={
                "type": "agent_tool_committed",
                "tool": tool_name,
                "idempotency_key": idempotency_key,
                "arguments_sha256": arguments_sha256,
                "expected_revision": expected_revision,
                "committed_revision": expected_revision + 1,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            async_task=async_task,
        )
        return {
            **record.result_json,
            "idempotency_reused": record.reused,
        }


# Compatibility name for callers from the read-only phase of the migration.
ReadOnlyToolGateway = LitTraceToolGateway


def _workspace_context(
    binding: AgentThreadBindingRecord,
    workspace: LiteratureWorkspace,
) -> dict[str, Any]:
    active_ids = list(workspace.context.active_papers)
    papers = [workspace.papers[paper_id] for paper_id in active_ids if paper_id in workspace.papers]
    filters = workspace.context.filters
    return {
        "session_id": binding.session_id,
        "workspace_revision": filters.workspace_revision,
        "topic": filters.topic,
        "research_background": filters.research_background,
        "paper_count": len(workspace.papers),
        "active_paper_count": len(active_ids),
        "parsed_paper_count": len(workspace.parsed_papers),
        "performance_cell_count": len(workspace.performance_cells),
        "claim_count": len(workspace.claims),
        "selected_for_download": list(workspace.context.selected_for_download),
        "papers": [paper.model_dump(mode="json") for paper in papers[:100]],
        "truncated": len(papers) > 100,
    }


def _paper_status(workspace: LiteratureWorkspace, paper_id: str) -> dict[str, Any]:
    if not paper_id:
        raise ValueError("paper_id must not be empty")
    paper = workspace.papers.get(paper_id)
    if paper is None:
        raise LookupError(f"Unknown paper_id: {paper_id}")
    parsed = coerce_parsed(workspace.parsed_papers.get(paper_id, {}))
    full_text = workspace.full_text_reports.get(paper_id)
    return {
        "paper": paper.model_dump(mode="json"),
        "active": paper_id in workspace.context.active_papers,
        "excluded": paper_id in workspace.context.excluded_papers,
        "selected_for_download": paper_id in workspace.context.selected_for_download,
        "parsed": parsed.parsed,
        "parse_error": parsed.error,
        "section_count": len(parsed.sections),
        "table_count": len(parsed.tables),
        "full_text_resolution": (
            full_text.model_dump(mode="json") if full_text is not None else None
        ),
    }


def _evidence(
    workspace: LiteratureWorkspace,
    *,
    paper_id: str | None,
    limit: int,
) -> dict[str, Any]:
    evidence = [
        record
        for record in workspace.evidence_records.values()
        if paper_id is None or record.paper_id == paper_id
    ]
    cells = [
        cell
        for cell in workspace.performance_cells
        if paper_id is None or cell.paper_id == paper_id
    ]
    claims = [
        claim
        for claim in workspace.claims
        if paper_id is None
        or any(
            workspace.evidence_records.get(evidence_id) is not None
            and workspace.evidence_records[evidence_id].paper_id == paper_id
            for evidence_id in claim.evidence_ids
        )
    ]
    return {
        "paper_id": paper_id,
        "evidence": [item.model_dump(mode="json") for item in evidence[:limit]],
        "performance_cells": [item.model_dump(mode="json") for item in cells[:limit]],
        "claims": [item.model_dump(mode="json") for item in claims[:limit]],
        "truncated": any(len(items) > limit for items in (evidence, cells, claims)),
    }


def _download_job_summary(job: AsyncTaskRecord) -> dict[str, object]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    execution = payload.get("execution")
    command = command if isinstance(command, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    return {
        "task_id": job.task_id,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "paper_ids": list(command.get("paper_ids") or []),
        "target": command.get("target"),
        "downloaded_count": execution.get("downloaded_count"),
        "requires_login_count": execution.get("requires_login_count"),
        "skipped_count": execution.get("skipped_count"),
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }


def _parse_job_summary(job: AsyncTaskRecord) -> dict[str, object]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    execution = payload.get("execution")
    command = command if isinstance(command, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    report = execution.get("report")
    report = report if isinstance(report, dict) else {}
    return {
        "task_id": job.task_id,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "paper_ids": list(command.get("paper_ids") or []),
        "parse_strategy": command.get("parse_strategy"),
        "parsed_count": report.get("parsed_count"),
        "failed_count": report.get("failed_count"),
        "stale_paper_ids": list(execution.get("stale_paper_ids") or []),
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }


def _table_job_summary(job: AsyncTaskRecord) -> dict[str, object]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    execution = payload.get("execution")
    command = command if isinstance(command, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    return {
        "task_id": job.task_id,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "paper_ids": list(command.get("paper_ids") or []),
        "performance_cell_count": execution.get("performance_cell_count"),
        "structured_artifact_count": execution.get("structured_artifact_count"),
        "stale_paper_ids": list(execution.get("stale_paper_ids") or []),
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }


def _storyline_job_summary(job: AsyncTaskRecord) -> dict[str, object]:
    payload = job.result_json if isinstance(job.result_json, dict) else {}
    command = payload.get("command")
    execution = payload.get("execution")
    command = command if isinstance(command, dict) else {}
    execution = execution if isinstance(execution, dict) else {}
    return {
        "task_id": job.task_id,
        "status": job.status,
        "attempt_count": job.attempt_count,
        "paper_ids": list(command.get("paper_ids") or []),
        "storyline_claim_count": execution.get("storyline_claim_count"),
        "evidence_record_count": execution.get("evidence_record_count"),
        "stale_paper_ids": list(execution.get("stale_paper_ids") or []),
        "last_error": job.last_error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
    }


def _command_sha256(command: dict[str, Any]) -> str:
    return sha256(
        json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_sha256(value: object) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value) if value is not None else default
    except (TypeError, ValueError):
        result = default
    return min(max(result, minimum), maximum)


def _strict_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _strict_bounded_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    result = _strict_non_negative_int(value, name=name)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result
