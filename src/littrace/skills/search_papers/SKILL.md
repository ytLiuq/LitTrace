# `search_papers`

合同名：`search_papers`（category=`retrieval`）。

## 输入
- `PaperSearchRequest`：检索词、topic、年份、是否走 live 等
- `LitTraceConfig`：决定 `enable_live_search` 与 LLM 路由

## 输出
- `SearchSkillResult`，含 `result / diagnostics / use_live / tool_result`

## 网络
需要。`requires_network=True`，预算 `requests: 2.0`。

## 备注
`SearchSkillResult` dataclass 在 [`_helpers.py`](../_helpers.py) 定义，并通过本子包和 `skill_runner` shim 双重 re-export，保证调用方 `isinstance` / 类型检查不变。