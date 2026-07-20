# Literature Sentinel Agent 设计方案

## 1. 场景

`Literature Sentinel Agent` 是一个长期运行的文献哨兵。它面向的不是一次性问答，而是持续监控某个研究方向，自动发现新论文、维护证据库、处理 PDF 解析、生成周期性 briefing，并把需要用户机构认证的访问任务排队给用户处理。

典型用户目标：

> 长期监控 “MXene flexible piezoresistive sensors” 方向。每天凌晨自动检索新文献，过滤重复和低相关论文，优先处理开放获取 PDF，解析成 structured documents，抽取性能指标，更新 evidence base。每周生成 briefing。如果发现高灵敏度、低迟滞、宽应变范围等关键进展，立即标记。遇到机构登录或 Cloudflare，不要自动绕过，生成用户待处理任务。

## 2. 目标

### 2.1 产品目标

- 持续监控用户关注的研究方向。
- 自动发现新文献并判断 novelty / relevance。
- 尽量自动获取开放获取全文。
- 对本地可用 PDF 使用 Docling 转为 structured documents。
- 从 structured documents 中抽取 evidence、tables、performance cells。
- 持续维护 session/workspace/evidence base。
- 遇到登录、MFA、Cloudflare、机构认证时生成用户任务，而不是自动处理账号密码。
- 周期性生成 digest、alerts、quality reports。
- 所有行为可追踪、可恢复、可评测。

### 2.2 非目标

- 不保存用户账号密码。
- 不绕过机构认证、Cloudflare 或出版社访问控制。
- 不把多 Agent 作为默认路径。
- 不让 LLM 直接从 raw PDF 写作。
- 不用 RAG-first 取代 session workspace。
- 不把所有步骤都塞进一个不可解释 prompt。

## 3. 总体架构

```text
Scheduler / CLI / API
  -> LiteratureSentinelAgent
     -> LangGraph State Machine
        -> load_state
        -> search_recent
        -> novelty_filter
        -> resolve_access
        -> download_or_queue_access
        -> parse_documents
        -> extract_evidence
        -> build_resource_pack
        -> quality_gate
        -> update_evidence_base
        -> write_digest_or_alert
        -> persist_state
  -> Session Workspace
  -> Sentinel State Store
  -> Harness Engine
  -> User Access Review Queue
```

关键原则：

```text
Agent = 持续目标 + 决策循环 + 状态恢复
Skills = 具体能力执行
Workspace = 事实状态
Harness = 质量判断
LangGraph = 条件流程与重试
```

## 4. 项目框架建议

建议新增模块：

```text
src/littrace/
  sentinel/
    __init__.py
    agent.py                 # LiteratureSentinelAgent 入口
    graph.py                 # LangGraph state machine
    state.py                 # SentinelState / Watchlist / RetryQueue
    resource_pack.py         # ResourcePack / EvidencePack
    access_queue.py          # 用户鉴权任务队列
    digest.py                # briefing / alert 生成
    storage.py               # state/evidence base 持久化
    policies.py              # novelty、relevance、retry、access 策略
```

也可以先用扁平文件 MVP：

```text
src/littrace/sentinel_agent.py
src/littrace/sentinel_state.py
src/littrace/resource_pack.py
```

等稳定后再拆目录。

## 5. 数据结构

### 5.1 Watchlist

```yaml
watchlists:
  - id: mxene_sensor
    topic: MXene flexible piezoresistive sensors
    query_variants:
      - MXene flexible piezoresistive sensor
      - MXene strain sensor
      - flexible pressure sensor MXene
    year_min: 2024
    frequency: daily
    alert_rules:
      - metric: gauge factor
        condition: unusually_high
      - metric: hysteresis
        condition: low
      - metric: strain range
        condition: wide
    preferred_sources:
      - OpenAlex
      - Crossref
      - Semantic Scholar
      - Unpaywall
```

### 5.2 SentinelState

```python
class SentinelState(BaseModel):
    schema_version: str
    watchlist_id: str
    last_run_at: str | None
    seen_paper_ids: list[str]
    rejected_paper_ids: list[str]
    retry_queue: list[RetryTask]
    access_queue: list[AccessTask]
    digest_history: list[DigestRecord]
    evidence_base_version: str
    warnings: list[str]
```

### 5.3 AccessTask

```python
class AccessTask(BaseModel):
    task_id: str
    paper_id: str
    title: str
    doi: str | None
    publisher: str | None
    landing_url: str | None
    reason: Literal[
        "requires_institution_login",
        "cloudflare_or_mfa",
        "publisher_login_required",
        "cdp_session_unavailable",
    ]
    suggested_action: str = "open_in_browser"
    retry_after_login: bool = True
    attempts: int = 0
    created_at: str
    updated_at: str
```

### 5.4 ResourcePack

`ResourcePack` 是 PDF 到写作之间的核心解耦层。

```python
class ResourcePack(BaseModel):
    schema_version: str = "littrace.resource_pack.v1"
    watchlist_id: str | None
    objective: str
    papers: list[PaperMetadata]
    citation_records: list[CitationRecord]
    full_text_report_refs: list[str]
    structured_document_refs: list[str]
    performance_cell_refs: list[str]
    comparison_matrix_refs: list[str]
    storyline_claim_refs: list[str]
    missing_evidence: list[str]
    quality_warnings: list[str]
    artifact_refs: list[ToolArtifactRef]
```

规则：

- Writer / digest generator 只读 `ResourcePack`。
- Writer 不直接访问 raw PDF。
- Writer 不直接检索新文献。
- 如果 evidence 不够，返回 `missing_evidence`，由 Agent graph 决定是否重跑 collection。

## 6. LangGraph 执行流程

### 6.1 主图

```text
START
  -> load_state
  -> load_watchlist
  -> search_recent
  -> novelty_filter
  -> resolve_access
  -> download_or_queue_access
  -> parse_documents
  -> extract_evidence
  -> build_resource_pack
  -> quality_gate
  -> update_evidence_base
  -> write_digest_or_alert
  -> persist_state
END
```

### 6.2 条件边

```text
search_recent -> persist_state
  if no new candidates

novelty_filter -> persist_state
  if no relevant or novel papers

resolve_access -> download_or_queue_access
  if open access or existing login works

resolve_access -> queue_user_access
  if institution login / MFA / Cloudflare required

download_or_queue_access -> parse_documents
  if PDFs downloaded or already exist

parse_documents -> retry_or_quarantine
  if parser failed or quality too low

quality_gate -> update_evidence_base
  if quality passed

quality_gate -> retry_or_quarantine
  if quality failed

update_evidence_base -> write_digest_or_alert
  if new important evidence found

update_evidence_base -> persist_state
  if no alert needed
```

### 6.3 Replan / Retry

可重试的失败：

- temporary network error
- publisher transient error
- PDF downloaded but corrupted
- Docling parse failed
- table extraction low confidence

不可自动重试，需用户处理：

- SSO login required
- MFA required
- Cloudflare challenge
- missing institutional entitlement
- publisher account required

## 7. 用户鉴权与机构认证交互

### 7.1 原则

- Agent 不保存账号密码。
- Agent 不自动输入用户凭据。
- Agent 不绕过 Cloudflare / MFA / SSO。
- Agent 只复用用户本地浏览器已有的合法登录态。
- 需要用户参与时生成 access queue。

### 7.2 夜间任务行为

```text
凌晨自动任务
  -> 检索新论文
  -> 解析 OA PDF
  -> 尝试使用本地 Chrome/CDP 已有登录态
  -> 如果需认证：写入 access_queue
  -> 继续处理其它论文
  -> 生成 morning summary
```

### 7.3 用户白天处理

命令建议：

```bash
littrace sentinel access-review
littrace sentinel open-access-task <task_id>
littrace sentinel resume-after-login
```

交互：

```text
发现 6 篇新论文，其中 2 篇需要机构认证：
1. ACS paper ... requires_institution_login
2. Elsevier paper ... cloudflare_or_mfa

运行 littrace sentinel open-access-task <task_id> 后，
系统会打开本地 Chrome，用户手动完成学校 SSO。
完成后运行 resume-after-login，Agent 继续下载和解析。
```

## 8. Skills 和 Tools

### 8.1 直接复用现有 Skills

来自 `skill_runner.py`：

- `search_papers_skill`
- `resolve_workspace_full_text_skill`
- `build_download_plan_skill`
- `execute_downloads_skill`
- `parse_workspace_skill`
- `extract_tables_skill`
- `audit_citation_links_skill`
- `build_comparison_matrix_skill`
- `build_storyline_skill`
- `build_quality_report_skill`
- `build_research_report_skill`
- `export_session_bundle_skill`

### 8.2 建议新增 Sentinel Skills

```text
load_watchlist_skill
load_sentinel_state_skill
save_sentinel_state_skill
deduplicate_seen_papers_skill
score_novelty_skill
score_relevance_skill
build_access_tasks_skill
build_resource_pack_skill
update_evidence_base_skill
write_digest_skill
write_alert_skill
summarize_run_skill
```

### 8.3 ToolContract 新增建议

```text
load_watchlist
save_sentinel_state
score_paper_novelty
score_paper_relevance
build_access_queue
build_resource_pack
update_evidence_base
write_sentinel_digest
write_sentinel_alert
```

每个 ToolContract 需要声明：

- input_schema
- output_schema
- requires_network
- mutates_workspace
- side_effects
- allow_in_react

## 9. Harness Engine

文献哨兵需要 Harness Engine，而不是只靠 prompt 判断质量。

### 9.1 Harness 目标

- 防止低质量 paper 进入 evidence base。
- 防止解析失败污染 structured documents。
- 防止 hallucinated digest。
- 防止重复论文重复进入 watchlist。
- 防止没有证据的 alert。

### 9.2 Harness 类型

```text
RetrievalHarness
  - candidate_count
  - duplicate_rate
  - relevance_score
  - recency_score

AccessHarness
  - open_access_rate
  - login_required_count
  - failed_download_count
  - retryable_failure_count

ParserHarness
  - docling_parsed_rate
  - markdown_chars
  - section_count
  - table_count
  - figure_count
  - parser_warnings

EvidenceHarness
  - citation_coverage
  - table_metric_coverage
  - evidence_span_count
  - low_confidence_cell_count

DigestHarness
  - every_claim_has_evidence
  - no_new_evidence_no_alert
  - missing_uncertainty_disclosure
  - citation_grounding

StateHarness
  - seen_set_consistency
  - retry_queue_consistency
  - access_queue_consistency
  - artifact_refs_exist
```

### 9.3 Gate 策略

```text
pass:
  update evidence base
  write digest

warn:
  update evidence base with warnings
  include uncertainty in digest

fail:
  quarantine paper or resource
  add retry task
  do not include in digest as evidence
```

## 10. 可观测性

### 10.1 必须记录的事件

```text
sentinel_run_started
watchlist_loaded
search_completed
candidate_deduplicated
candidate_rejected
candidate_promoted
access_resolved
access_task_created
download_completed
download_failed
parse_completed
parse_failed
evidence_extracted
quality_gate_passed
quality_gate_failed
evidence_base_updated
digest_written
alert_written
sentinel_run_completed
```

### 10.2 Trace 结构

建议每次 run 有一个 `run_id`：

```text
sentinel_runs/
  2026-07-16T02-00-00_mxene_sensor/
    run_manifest.json
    workflow_trace.json
    tool_calls.jsonl
    harness_report.json
    digest.md
    alerts.json
```

### 10.3 Tool Call Journal

现有 `ToolResult` 已有：

- tool
- contract_id
- ok
- error
- warnings
- elapsed_ms
- output_summary
- metadata

Sentinel 应该把每次 ToolResult 写入：

```text
tool_calls.jsonl
```

这样可以回答：

- 哪个 tool 最慢？
- 哪些 publisher 最常失败？
- 哪些 parser warning 最常见？
- 哪些论文反复进入 retry？

### 10.4 指标

```text
run_duration_ms
new_candidates_count
promoted_papers_count
rejected_papers_count
download_success_rate
parse_success_rate
access_task_count
retry_queue_size
digest_claim_count
evidence_coverage_rate
quality_score
```

## 11. 存储结构

建议：

```text
sentinel/
  watchlists/
    mxene_sensor.yaml
  state/
    mxene_sensor_state.json
  access_queue/
    pending_access_tasks.json
  retry_queue/
    retry_tasks.json
  evidence_base/
    papers/
    structured_documents/
    performance_cells.jsonl
    citation_records.jsonl
    resource_packs/
  runs/
    2026-07-16T02-00-00_mxene_sensor/
      run_manifest.json
      workflow_trace.json
      tool_calls.jsonl
      harness_report.json
      digest.md
      alerts.json
```

也可以把它挂到现有 session workspace：

```text
session/
  workspace/
    sentinel/
      state.json
      access_queue.json
      retry_queue.json
      evidence_base/
      runs/
```

## 12. CLI / API

### 12.1 CLI

```bash
littrace sentinel init --watchlist mxene_sensor
littrace sentinel run --watchlist mxene_sensor
littrace sentinel run-nightly
littrace sentinel status
littrace sentinel access-review
littrace sentinel open-access-task <task_id>
littrace sentinel resume-after-login
littrace sentinel digest --latest
```

### 12.2 API

```text
GET  /sentinel/watchlists
POST /sentinel/watchlists
POST /sentinel/run
GET  /sentinel/status
GET  /sentinel/access-tasks
POST /sentinel/access-tasks/{task_id}/open
POST /sentinel/resume-after-login
GET  /sentinel/runs/{run_id}
GET  /sentinel/digest/latest
```

## 13. 安全与合规

必须遵守：

- 不保存用户密码。
- 不绕过登录或访问控制。
- 用户显式处理机构认证。
- 对出版社请求做 rate limit。
- 对失败下载做 backoff。
- 明确记录访问来源和状态。
- 只下载用户有权访问或开放获取的 PDF。

建议策略：

```text
if open_access:
  download
elif local_chrome_session_has_access:
  download
elif login_required:
  create_access_task
else:
  add_retry_or_quarantine
```

## 14. MVP 路线

### Phase 1：离线 Sentinel MVP

- 新增 watchlist/state 数据结构。
- 新增 `littrace sentinel run`。
- 使用 mock/live search。
- 去重 seen papers。
- 生成 digest。
- 不做自动下载。

验收：

- 能重复运行且不重复纳入旧论文。
- 能生成 run manifest 和 digest。

### Phase 2：接入 Skills

- 接入 `search_papers_skill`。
- 接入 `resolve_workspace_full_text_skill`。
- 接入 `parse_workspace_skill`。
- 接入 `build_quality_report_skill`。

验收：

- Open access PDF 可自动进入 parse。
- structured documents 写入 evidence base。

### Phase 3：Access Queue

- 新增 `AccessTask`。
- 新增 `access-review` / `resume-after-login`。
- CDP 登录态可用时继续下载。

验收：

- 需要机构认证的论文不会阻塞整个 run。
- 用户登录后可恢复下载解析。

### Phase 4：Harness Engine

- RetrievalHarness
- AccessHarness
- ParserHarness
- EvidenceHarness
- DigestHarness
- StateHarness

验收：

- 低质量解析不会进入 evidence base。
- digest claim 有 evidence。

### Phase 5：LangGraph 条件循环

- retry loop
- quarantine path
- no-new-paper fast path
- alert path

验收：

- Graph trace 可解释每个分支。
- 每个失败都有 retry/quarantine/access_task 状态。

### Phase 6：定时运行

- cron / launchd / app automation。
- 每天凌晨自动 run。
- 早上生成 summary。

验收：

- 离线或认证失败不会破坏 state。
- 下一次 run 可恢复。

## 15. 最终建议

`Literature Sentinel Agent` 的价值不在于多一个 Agent 名字，而在于：

- 长期目标
- 后台调度
- 独立状态
- 条件循环
- 鉴权任务队列
- 自动恢复
- evidence base 维护
- 周期性 digest
- 可观测 run history

这正是 LangGraph、ToolContracts、Session Workspace、Harness Engine 可以一起发挥价值的地方。
