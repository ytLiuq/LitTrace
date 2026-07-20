# LitTrace 最优架构建议

## 结论

LitTrace 最适合走 **单 Coordinator Agent + Skills/ToolContracts + Session Workspace + Harness/Eval + Optional Review Council** 路线。

不要把主路径做成重型多 Agent，也不要把 RAG 作为当前的主记忆层。这个项目的核心价值不是“很多 Agent 互相聊天”，而是：能可靠检索论文、拿到全文、解析 PDF、抽取证据、生成可追溯的表格和研究叙事，并且能被评测体系反复检查。

一句话版本：

> 一个 Coordinator 负责理解用户和调度；Skills/ToolContracts 负责干活；Session Workspace 负责保存事实状态；Docling structured document 负责把 PDF 变成可引用证据；Harness/Eval 负责质量门；多 Agent 只作为可选审稿委员会。

## 推荐架构图

```text
User / CLI / TUI / API / Window
  -> LitTraceCoordinator
     -> Intent Confidence + Ambiguity Handling
     -> MemoryView
     -> Skill Selection
  -> Skills / ToolContracts
     -> Search
     -> Full-text Resolve
     -> Download Planning / Execution
     -> Docling PDF Parsing
     -> Table Extraction
     -> Storyline / Report
     -> Quality Report / Export
  -> Session Workspace
     -> workspace.json
     -> workspace/manifest.json
     -> workspace/artifact_index.json
     -> workspace/memory.json
     -> workspace/structured_documents/
     -> workspace/snapshots/
  -> Harness / Eval Gates
  -> Optional Multi-Agent Review Council
  -> User-facing Answer / Artifacts
```

## 为什么是这个架构

### 1. 单 Agent 更适合当前产品形态

LitTrace 的主流程有明确的用户目标和清晰的工具链：

- 检索论文
- 管理当前 literature context
- 解析 PDF
- 抽取性能表格
- 生成 storyline / report
- 检查引用、证据和质量

这些步骤更像一个研究工作台的流水线，而不是多个自治 Agent 需要长期协商的开放世界任务。单 Coordinator 能让行为更可控、状态更可调试、测试更直接。

### 2. Skills 比 Agent 更应该成为能力边界

项目真正可复用的能力不是“RetrievalAgent”或“ParserAgent”这个角色名，而是稳定的工具能力：

- `search_papers`
- `resolve_workspace_full_text`
- `build_download_plan`
- `execute_downloads`
- `parse_workspace_papers`
- `extract_performance_cells`
- `build_storyline`
- `build_quality_report`
- `export_session_bundle`

这些能力应该被 `ToolContract` 包起来，形成稳定、可测试、可审计的执行层。Agent 只负责选择和组合 Skills，不应该复制业务逻辑。

### 3. Session Workspace 比 RAG 更适合作为当前记忆层

LitTrace 处理的是一组用户明确选中的论文、PDF、表格、证据和报告。这里最重要的是结构化状态和可追溯 artifact，而不是从大规模语料中语义召回。

因此，当前不建议 RAG-first。更好的路径是：

- 用 workspace 保存当前研究上下文
- 用 Docling structured document 保存 PDF 结构化内容
- 用 artifact index 保存生成物
- 用 memory.json 保存短期任务状态、偏好、执行记录和文档摘要
- 后续如果文档数量变大，再给 structured documents 加索引检索

## 目标分层

### 1. Coordinator Layer

主文件：

- `src/littrace/coordinator.py`
- `src/littrace/chat.py`
- `src/littrace/intent.py`
- `src/littrace/intent_llm.py`

职责：

- 接收用户消息
- 解析 intent
- 给出 intent confidence
- 发现 ambiguous intent
- 暂存 pending intent
- 合并用户澄清
- 构建 `MemoryView`
- 选择 Skill 或 bounded workflow
- 决定是否调用 review council

不应该做的事：

- 不直接写检索逻辑
- 不直接写 PDF parsing 逻辑
- 不直接写表格抽取逻辑
- 不直接写 citation audit 逻辑
- 不把多 Agent 当主执行路径

### 2. Skill / ToolContract Layer

主文件：

- `src/littrace/tool_contracts.py`
- `src/littrace/skill_runner.py`

职责：

- 定义 `ToolContract`
- 定义 `ToolResult`
- 定义 `ToolExecutionPolicy`
- 统一执行、记录耗时、捕获错误、输出 summary/warnings
- 把已有函数逐步包成 Skill

建议标准：

每个重要能力都应该有：

- contract name
- version
- category
- input schema 描述
- output schema 描述
- side effects
- network requirement
- workspace mutation 标记
- 是否允许在 ReAct loop 中调用

大输出不要长期塞进 `ToolResult.output`。应该优先写 artifact，然后通过 `ToolResult.output_ref` 返回引用。

### 3. Domain Capability Layer

主文件：

- `src/littrace/search.py`
- `src/littrace/full_text.py`
- `src/littrace/access.py`
- `src/littrace/downloads.py`
- `src/littrace/parsing.py`
- `src/littrace/tables.py`
- `src/littrace/storyline.py`
- `src/littrace/document_composer.py`
- `src/littrace/quality_report.py`
- `src/littrace/export.py`

职责：

- 实现具体业务能力
- 保持函数可单测
- 不关心用户对话
- 不关心 Agent 角色
- 不直接决定产品流程

这一层是项目的“发动机”。上层 Coordinator 和 Agent 可以调用它，但不能把业务逻辑复制一份。

### 4. Session Workspace Layer

主文件：

- `src/littrace/session.py`
- `src/littrace/models.py`
- `src/littrace/runtime/memory.py`

推荐存储结构：

```text
session/
  workspace.json
  messages.jsonl
  artifacts/
  workspace/
    manifest.json
    artifact_index.json
    memory.json
    structured_documents/
    snapshots/
```

职责：

- 保存当前 active papers
- 保存 selected papers
- 保存 parsed papers
- 保存 structured document path
- 保存 table extraction 结果
- 保存 storyline/report artifacts
- 保存 pending intent
- 保存 session-scoped preferences
- 保存 workspace snapshots

Session Workspace 是事实源。Agent 记忆、报告生成、评测、导出都应该能从这里恢复。

### 5. Runtime Memory Layer

主文件：

- `src/littrace/runtime/memory.py`

建议明确分成四层：

- `WorkingMemory`：当前 turn 或下一步行动需要的状态，例如 pending intent、active paper ids、selected downloads。
- `EpisodicMemory`：发生过什么，例如 tool calls、ReAct traces、artifact events、失败记录。
- `DocumentMemory`：结构化文档摘要、section/table/figure 数量、Docling quality、document artifact 引用。
- `PreferenceMemory`：用户和 session 偏好，例如 parser preference、download selection mode、source route preference。

Coordinator 应该消费压缩后的 `MemoryView`，不要直接把完整历史和全文塞进 prompt。

### 6. PDF / Docling Structured Document Layer

主文件：

- `src/littrace/ocr/docling_adapter.py`
- `src/littrace/parsing.py`
- `tests/test_docling_adapter.py`
- `tests/test_parsing.py`

推荐方向：

- 优先使用 Docling 作为 PDF 到结构化文档的主路径
- 输出 markdown、outline、sections、tables、figures、parser report
- 把 structured document 写入 `workspace/structured_documents/`
- 在 workspace 中只保存引用和摘要
- 给真实 PDF 加集成测试，而不是只测 mock adapter

Docling structured document 是 LitTrace 的关键中间层。它连接 raw PDF 和后续的 table extraction / storyline / citation grounding。

### 7. Workflow / LangGraph Layer

主文件：

- `src/littrace/workflow.py`
- `src/littrace/workflow_parts/`

推荐定位：

> LangGraph 是 bounded workflow/state machine，不是整个产品架构。

适合 LangGraph 的场景：

- 固定 research pipeline
- search -> audit -> parse -> table -> storyline -> report
- 有条件分支的质量门
- 可回放的 workflow trace
- bounded replan

不适合 LangGraph 的场景：

- 简单聊天回合
- 每个 Skill 都硬套 graph node
- 用 graph 替代清晰的 ToolContract
- 用多 Agent 消息传递作为默认主路径

所以走单 Agent + Skills 路线后，LangGraph 仍然有用，但它的位置应该下沉为“复杂流程编排层”。

### 8. Harness / Eval Layer

主文件：

- `src/littrace/harnesses.py`
- `src/littrace/quality_report.py`
- `src/littrace/eval_api.py`
- `src/littrace/golden_eval.py`
- `src/littrace/retrieval_eval.py`
- `src/littrace/pdf_benchmark.py`
- `src/littrace/agent_audits.py`

职责：

- 检查检索质量
- 检查 citation grounding
- 检查 PDF parsing quality
- 检查 table extraction quality
- 检查 storyline evidence coverage
- 检查 hallucination risk
- 检查 tool policy compliance
- 检查 memory consistency

Harness 不应该拥有主执行流程。它应该像质量门一样读取 workspace、memory、artifacts、traces，然后输出 findings、scores 和 suggested fixes。

### 9. Optional Multi-Agent Review Council

主文件：

- `src/littrace/autonomous_loop.py`
- `src/littrace/runtime/agents.py`
- `src/littrace/runtime/orchestrator.py`
- `src/littrace/agent_interactions.py`
- `src/littrace/storyline_review.py`

推荐定位：

多 Agent 不做主路径，只做可选 review council。

适合的 reviewer 角色：

- Citation Auditor
- Table Auditor
- Storyline Skeptic
- Method Reviewer
- Access/Compliance Reviewer
- Replanning Agent

它们输出：

- critique
- severity
- evidence reference
- suggested fix
- safe replan action

它们不应该直接改 workspace，除非 Coordinator 明确允许执行某个 safe action。

## 推荐运行流程

### 普通用户回合

```text
User message
  -> Coordinator parses intent
  -> if ambiguous: store pending intent and ask clarification
  -> build MemoryView
  -> select Skill or bounded Workflow
  -> run ToolContract
  -> update Workspace
  -> write artifacts / memory
  -> run relevant harness checks
  -> return answer with warnings / citations / next actions
```

### PDF 解析回合

```text
User requests parse
  -> Coordinator selects parse_workspace_papers skill
  -> SkillRunner runs ToolContract
  -> Docling parses PDFs
  -> structured documents written to workspace/structured_documents/
  -> parser reports and quality reports written
  -> DocumentMemory updated
  -> artifact_index updated
```

### Storyline / Report 回合

```text
User requests storyline/report
  -> Coordinator builds synthesis MemoryView
  -> check active papers and document memory
  -> run table/storyline/report skills
  -> run citation and evidence harnesses
  -> optionally ask review council
  -> return grounded synthesis + artifact refs
```

### Optional Review Council 回合

```text
Draft / Report / Workspace
  -> Coordinator asks review council
  -> Review agents read workspace + memory + artifacts
  -> produce critiques and suggested fixes
  -> harnesses score findings
  -> Coordinator accepts safe fixes or asks user
```

## 建议目录边界

```text
src/littrace/
  coordinator.py              # 单 Agent 主入口：intent、ambiguity、memory view、skill selection
  chat.py                     # CLI/chat 入口，逐步瘦身为 coordinator facade
  intent.py                   # 规则 intent + confidence + ambiguity
  intent_llm.py               # LLM intent parser

  tool_contracts.py           # ToolContract / ToolResult / ToolExecutionPolicy
  skill_runner.py             # 所有核心 Skill 的统一包装层

  session.py                  # Session Workspace 持久化
  models.py                   # 共享数据模型

  runtime/
    memory.py                 # Working/Episodic/Document/Preference memory
    messages.py               # Agent message, ReActTrace, artifacts
    agents.py                 # 可选 review/runtime agents，逐步瘦身
    orchestrator.py           # 可选 agent orchestration

  ocr/
    docling_adapter.py        # Docling structured parser
    paddleocr_adapter.py      # fallback parser
    registry.py               # parser selection

  workflow.py                 # bounded workflow graph
  workflow_parts/             # graph state/routing/tracing

  harnesses.py                # deterministic checks
  quality_report.py           # quality summary
  eval_api.py                 # eval metrics
  golden_eval.py              # golden tasks

  autonomous_loop.py          # optional self-iteration / review council
```

## Agent 和 Harness 能不能同时改而不互踩

可以，但前提是边界要清楚：

- 改 `coordinator.py` / `chat.py`：影响用户主路径，要优先保持行为兼容。
- 改 `skill_runner.py` / `tool_contracts.py`：影响所有调用方，需要跑更宽的回归测试。
- 改 `runtime/agents.py` / `autonomous_loop.py`：应该只影响可选 review/self-iteration，不应破坏普通 chat/search/parse。
- 改 `harnesses.py` / `quality_report.py` / `eval_api.py`：应该只读 workspace/artifacts/memory，默认不写主状态。
- 改 `parsing.py` / `docling_adapter.py`：会影响 document memory、table/storyline，需要跑 PDF 和 parsing 相关测试。

最重要的规则：

> Agent 负责决策和审阅，Harness 负责评测和发现问题，Skill 负责执行，Workspace 负责保存事实状态。

只要遵守这个规则，agent 和 harness 可以并行演进。

## 哪些东西应该保留

核心保留：

- `LitTraceCoordinator`
- `ToolContract`
- `ToolResult`
- `ToolExecutionPolicy`
- `SkillRunner`
- `Session Workspace`
- `Runtime Memory`
- `Docling structured documents`
- `Artifact index`
- `Snapshots`
- `Harness/Eval`
- `ReActTrace`
- Optional Review Council

## 哪些东西应该减少

建议逐步减少：

- 主路径里的多 Agent handoff
- runtime agents 里重复实现的 search/parse/table/report 逻辑
- graph node 对简单函数的一层空包装
- 大对象直接塞进 chat response 或 tool result
- prompt 中隐藏的长期记忆
- 未经过 workspace/artifact index 的临时状态

不是删除多 Agent，而是把它降级为质量增强层。

## 重构路线

### Phase 1：稳定 Coordinator

- 让所有 chat turn 先经过 `LitTraceCoordinator.prepare_turn`
- intent confidence 和 ambiguity handling 在 Coordinator 层统一处理
- pending intent 存入 workspace context
- Coordinator 消费 `MemoryView`

验收标准：

- ambiguous intent 会要求澄清
- 用户澄清能合并 pending intent
- 普通 chat 行为不回退

### Phase 2：Skill-first Runtime

- 所有核心能力进入 `skill_runner.py`
- 所有核心能力拥有 `ToolContract`
- Runtime agents 只调用 Skills，不直接调用底层业务函数
- Workflow nodes 也优先调用 Skills

验收标准：

- 添加新能力时优先加 Skill，而不是新 Agent
- ToolResult 能记录 ok/error/warnings/elapsed/output_summary
- ToolExecutionPolicy 能拦截不允许的 network/workspace mutation

### Phase 3：Session Workspace / Memory 完整化

- `memory.json` 随 session 保存
- `structured_documents/` 成为 PDF 结构化输出目录
- `artifact_index.json` 覆盖报告、表格、解析产物、质量报告
- `MemoryView` 成为 Coordinator 和 Review Council 的输入

验收标准：

- 重启 session 后能恢复 workspace 和 memory
- structured document 可以被定位和审计
- memory 不依赖 prompt 隐式历史

### Phase 4：Docling 真实 PDF 测试

- 加真实 PDF fixture 或最小可公开 PDF
- 测 Docling parse path
- 测 structured document 写盘
- 测 quality report
- 测 DocumentMemory 记录

验收标准：

- 不只是 mock adapter 通过
- 能证明真实 PDF -> structured document -> workspace reference 的链路成立

### Phase 5：Harness / Eval 质量门

- citation grounding check
- parser quality check
- table extraction check
- storyline evidence coverage check
- memory consistency check
- tool policy compliance check

验收标准：

- Harness 可以解释失败原因
- quality report 能给出 actionable findings
- eval API 能用于回归测试

### Phase 6：Optional Review Council

- Review Council 读取 workspace/memory/artifacts
- 输出 critiques 和 suggested fixes
- Coordinator 决定是否执行 replan
- 默认主路径不依赖 Review Council

验收标准：

- 开启 review council 能提升报告质量
- 关闭 review council 不影响普通流程
- review agents 不直接篡改 workspace

## 测试建议

核心测试组：

```bash
.venv/bin/pytest -q \
  tests/test_coordinator.py \
  tests/test_intent.py \
  tests/test_runtime_memory.py \
  tests/test_session.py \
  tests/test_tool_contracts.py \
  tests/test_skill_runner.py
```

## 兼容模块迁移策略

为降低本轮重构的迁移风险，旧的顶层导入路径暂时保留为极薄的兼容模块。例如，
`littrace.search` 转发至 `littrace.retrieval.search`，`littrace.tables`、
`littrace.storyline` 和 `littrace.document_composer` 转发至 `littrace.evidence`
包下的实现。它们不承载业务逻辑，也不应成为新代码的导入目标。

新代码必须直接依赖新的领域包路径；外部集成可在一个完整的稳定版本周期内继续使用
旧路径。移除兼容模块前，需要先在发布说明中给出替换路径，并在下一个主版本中完成删除。
这使代码量集中在真正的领域实现，同时不破坏现有脚本、插件和测试。

PDF / Docling 测试组：

```bash
.venv/bin/pytest -q \
  tests/test_docling_adapter.py \
  tests/test_parsing.py \
  tests/test_tables.py
```

主路径回归测试组：

```bash
.venv/bin/pytest -q \
  tests/test_chat.py \
  tests/test_workflow.py \
  tests/test_api_routes.py \
  tests/test_autonomous_loop.py \
  tests/test_runtime_agents.py
```

代码质量：

```bash
.venv/bin/ruff check src tests
```

## 成功标准

这个架构算跑顺了，如果满足：

- 大多数用户回合只经过一个 Coordinator
- 所有核心能力都有 ToolContract
- Runtime agents 不再复制业务函数
- Session Workspace 可以复现状态
- Docling structured document 可检查、可引用、可评分
- Harness 能指出输出为什么不安全、不完整或证据不足
- Review Council 能提高质量，但不拥有主路径
- 添加新 Skill 不需要新增一个 Agent
- LangGraph 只用于 bounded workflow，而不是覆盖全部交互
- 不触碰 cache 也能完成主架构演进

## 最终建议

LitTrace 应该做成一个 **research-grade single-agent workbench**，而不是大而散的多 Agent 系统。

最佳长期形态是：

- 一个强 Coordinator
- 一组强 Skills
- 明确 ToolContracts
- 明确 Session Workspace
- Docling-first structured documents
- 可审计 artifacts
- deterministic harnesses
- 可选 adversarial review council

这条路线代码量更少、调试更容易、测试更稳定，也更贴合你的真实产品目标。
