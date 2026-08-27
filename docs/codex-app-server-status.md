# Codex App Server — current status (2026-08-27)

## TL;DR

`mode: codex_app_server` is wired up end-to-end (mcp_servers config, JSON-RPC
client, sandbox overrides, fallback refusal detection) and the App Server
itself launches cleanly. The upstream stdio exec-mode guard, however, blocks
MCP tool calls in every version we tested (codex 0.140.0 / 0.146.1 / 0.149.1
/ 0.150.1 npm + `0.149.0-alpha.4.3` bundled with ChatGPT.app).

ChatGPT.app's bundled codex + `--enable exec_permission_approvals /
enable_mcp_apps / request_permissions_tool` reaches the MCP server fine and
the tool returns `success: true`, but the model then misreads the
empty-but-valid workspace as a refusal and refuses to call any further tool.
That confused phrasing is now caught by `littrace.codex_runtime.service.
_looks_like_refusal` so `fallback_to_legacy=True` cleanly hands the turn to
LitTrace's native chat path.

## What works right now

- `mode: legacy` + `DEEPSEEK_*` (or `qwen-plus`) — LitTrace's LLM-driven chat
  coordinator. `/chat` does set research background, runs `search_papers`
  end-to-end against OpenAlex, parses PDFs, and falls back through the
  `handle_agent_chat` `except AppServerError` path when the App Server mode
  is enabled. 295 unit + api + eval tests pass.
- The MCP server still starts (`LITTRACE_MCP_GATEWAY=1`) so any external MCP
  client can drive the 15 gateway tools directly.

## What is wired up but does not yet work

- `mode: codex_app_server` reaches MCP tools, but the model treats an
  empty-but-valid response as a refusal and refuses to proceed. The fallback
  detection is in place, so a turn that hits the refusal will route to
  legacy, but the App Server's own process stays alive between turns and a
  subsequent chat can hang on a stale client. We do not recommend flipping
  the default until upstream codex fixes the empty-workspace refusal
  interpretation.

## Switch back to App Server when

- Upstream codex fixes the stdio exec-mode guard (any 0.151+ patch that lets
  the model successfully call MCP tools on an empty workspace, or that
  exposes a different protocol — `codex mcp-server` over the `exec-server`
  socket, for example).
- The `request_timeout_seconds` can be lowered enough that even a confused
  refusal reply becomes a quick 30-second timeout instead of a hanging
  connection.

## Configuration

The relevant knobs (all of these are gitignored — operator-local only):

```yaml
# config.yaml
agent_runtime:
  mode: legacy                 # or codex_app_server
  fallback_to_legacy: true     # always on
  codex_command: ["/Applications/ChatGPT.app/Contents/Resources/codex", "app-server", "--enable", "exec_permission_approvals", "--enable", "enable_mcp_apps", "--enable", "request_permissions_tool"]
  codex_home: /Users/<you>/.codex
  sandbox_policy: read-only
```

```
# .env.local (LLM intent parser + qwen-plus)
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPSEEK_MODEL=qwen-plus
```

The DEEPSEEK_* names are legacy; `_with_env_overrides` still reads them
because the qwen / dashscope keys are the actual values.
