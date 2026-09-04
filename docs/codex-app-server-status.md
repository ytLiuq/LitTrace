# Codex App Server — current status (2026-08-27)

## TL;DR

`mode: codex_app_server` is wired up end-to-end (MCP config, JSON-RPC client,
sandbox overrides, approval handling, and fallback detection). With the
ChatGPT.app bundled Codex 0.151.0-alpha.7.2, a real turn can approve and call
LitTrace MCP tools and return their workspace result. `fallback_to_legacy=true`
remains enabled for provider or network failures.

## What works right now

- `mode: legacy` + `DEEPSEEK_*` (or `qwen-plus`) — LitTrace's LLM-driven chat
  coordinator. `/chat` does set research background, runs `search_papers`
  end-to-end against OpenAlex, parses PDFs, and falls back through the
  `handle_agent_chat` `except AppServerError` path when the App Server mode
  is enabled. 295 unit + api + eval tests pass.
- The MCP server still starts (`LITTRACE_MCP_GATEWAY=1`) so any external MCP
  client can drive the 15 gateway tools directly.

## Current verification

- `mode: codex_app_server` reaches LitTrace MCP tools on Codex Desktop
  0.151.0-alpha.7.2. The client accepts the explicit LitTrace MCP approval
  elicitation and returns the tool result to the model.
- The child process strips outer Codex sandbox variables so a nested launch
  does not inherit `CODEX_SANDBOX_NETWORK_DISABLED` and reset its response
  stream. Shell and file-change approvals remain denied.
- Keep `fallback_to_legacy: true` for provider/network outages. A transport
  failure now wakes the active turn immediately instead of waiting for the
  full turn timeout.

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
