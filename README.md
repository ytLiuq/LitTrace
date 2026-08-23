# LitTrace

LitTrace is a materials and chemistry research agent for traceable literature
workflows. It helps researchers search papers, manage the active literature
context, optionally download PDFs, parse full text with OCR tools, build
evidence-grounded comparison tables, and generate truthful research storylines.

LitTrace uses a **Single Coordinator + Skills/ToolContracts + Session
Workspace** architecture:

- The **Coordinator** owns user-facing intent parsing, ambiguity handling, and
  memory-view preparation.
- **Skills/ToolContracts** are the stable execution surface for search,
  download planning, PDF parsing, table extraction, storyline/report
  synthesis, exports, and quality checks.
- **Session workspace** files are the source of truth for papers, structured
  documents, artifacts, snapshots, and runtime memory.
- **LangGraph** is kept for bounded research workflows and traceable state
  transitions, not as the whole product architecture.
- Deterministic quality gates run on the main path. A temporary, read-only,
  bounded Reviewer Agent is optional for high-value reports.

## Product Principles

- The active literature context is visible and editable by the user.
- PDFs are never downloaded by surprise. Download behavior is controlled by
  configuration and explicit user selection.
- Every paper-related answer must carry citations and checked access links.
- Storylines must be grounded in evidence: what earlier papers solved, what
  limits remained, and how later papers responded.
- Performance tables must preserve provenance down to paper, page, table, row,
  column, snippet, parser, and confidence.
- Evaluation APIs are first-class so retrieval, parsing, extraction, storyline,
  and end-to-end quality can improve measurably.

## Initial Scope

- Materials/chemistry source routing for Crossref, OpenAlex, Semantic Scholar,
  Unpaywall, arXiv, and publisher links such as Wiley, ACS, Springer Nature,
  RSC, Elsevier, MDPI, and Nature Portfolio.
- Configurable paper storage folders.
- Literature context panel state for show/hide, include/exclude, pinning,
  filtering, and download selection.
- Pluggable OCR/PDF parsing tool interface.
- Harness interfaces for citations, links, tables, and storylines.
- FastAPI evaluation endpoints.

## Local Development

LitTrace is designed for **local native use**. Docker is optional for CI or
API-only batch jobs, but the main product surface is a local desktop window
that can cooperate with your browser login state and local PDF folders.

```bash
git clone https://github.com/ytLiuq/LitTrace.git
cd LitTrace
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,parsers]"
littrace setup-browser --launch
littrace doctor
littrace-window
```

Publisher-authenticated full-text access uses a local Chrome CDP connection,
not BrowserAct. The setup command discovers local Chrome profiles, reports
whether publisher cookies are already present, and can launch Chrome with a
remote debugging port:

```yaml
cdp_downloader:
  cdp_url: "http://127.0.0.1:19222"
  chrome_profile_name: "Default"
  auto_launch_chrome: true
```

You can override the CDP endpoint with `LITTRACE_CDP_URL`, the Chrome binary
with `LITTRACE_CHROME_EXECUTABLE`, the user-data directory with
`LITTRACE_CHROME_USER_DATA_DIR`, and the selected profile with
`LITTRACE_CHROME_PROFILE_NAME`. BrowserAct settings remain for the legacy
interactive login handoff path and fallback diagnostics; publisher PDF download
primarily uses the CDP downloader.
Use `littrace setup-browser --no-launch` for diagnostics without opening Chrome.

For the Codex-style local interactive app, start the native popup window:

The window is the recommended product surface. It opens a local desktop-style
chat window, keeps the literature context in a hideable side panel, and uses
separate popups for context selection, OCR/text-layer choice, and login-gated
download handoff. It is not a web app and does not require starting FastAPI.

If your Python build does not include Tk support, use a Python distribution
with Tkinter enabled, then reinstall the editable package and run
`littrace-window` again.

For the lower-level command shell:

```bash
littrace
```

Optional DeepSeek-compatible chat support is loaded from `.env.local`:

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

`.env.local` is ignored by Git. If these values are missing, the shell falls
back to deterministic local commands and help text.

Shell commands:

```text
/context
/hide-context
/show-context
/papers
/dashboard
/quality
/agents
/workflow
/quality-audits
/plan MXene sensor
/init-config
/login N
/attach N /path/to/paper.pdf
/attach-si N /path/to/si.pdf
/full-text
/backfill-dois 10.1021/acs.nanolett.5c01464
/publisher-retrieve acs MXene sensor
/check-downloads
/resume-downloads
/parse
/table
/storyline
/storyline-report
/storyline-review
/benchmark
/golden-eval
/export
/quit
```

`/golden-eval` reads real materials/chemistry tasks from `eval/golden`.
When the current chat has literature context, it also scores DOI recall,
recent-paper ratio, publisher coverage, table-metric recall, storyline evidence
coverage, citation coverage, and workflow progress.

RAG maintenance commands run outside the interactive shell:

```bash
docker compose -f docker-compose.rag.yml up -d
littrace rag daily
littrace rag refresh --session SESSION_ID
littrace rag daemon --interval-hours 24
```

See `docs/rag_automation.md` for cron and macOS launchd examples.
See `docs/evaluation.md` for literature-retrieval, chunk-level RAG, exact table
cell, and evidence-grounded task benchmarks.

### 验证真实 daily 更新

普通 `pytest` 只验证本地逻辑，不访问出版社，也不会启动 Chrome。要验证完整的
检索 -> 全文入口探测 -> CDP 下载 -> digest 落盘链路，显式运行 live smoke test：

```bash
LITTRACE_LIVE_DAILY_TESTS=1 .venv/bin/pytest -q -m live tests/test_live_daily_update.py
```

该测试需要网络和可启动的 Chrome，耗时通常数分钟；`cdp_downloader.auto_launch_chrome: true`
时 LitTrace 会自动启动自己的 CDP Chrome。登录、MFA 或 Cloudflare 阻断的论文不会被伪造
成成功，而是进入 Sentinel access queue。

Conversation examples:

```text
检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表
显示上下文
选择第 1、3 篇下载
全部下载
取消选择第 2 篇
生成发展脉络
```

For login-gated papers, LitTrace uses the CDP publisher downloader: it opens the
authorized publisher or DOI page in local Chrome, lets the user complete
Cloudflare or institutional login when needed, and then saves the PDF into the
expected LitTrace folder. `/attach N /path/to/paper.pdf` can still copy an
existing local PDF into the expected folder for that paper. `/resume-downloads`
parses ready PDFs, extracts tables, and writes session artifacts.
`/publisher-retrieve family topic` merges DOI-level publisher search results
into the active context, and `/attach-si N path` stores supplementary files
under the current session artifacts.
`/full-text` resolves DOI-level full-text candidates from Crossref links,
Unpaywall OA locations when configured, publisher landing pages, and existing
PDF URLs before download or login handoff. `/backfill-dois` adds exact DOI
metadata to the current context when keyword search misses a known target paper.

Sessions, messages, task state, and research memory are persisted in Postgres.
PDFs and derived artifacts use the configured object-storage backend. Local
working directories are disposable staging space only.

Copy `config.example.yaml` to `config.yaml` and set `storage.paper_library_dir`
before running workflows that may download PDFs.

To enable the optional Docling parser backend:

```bash
pip install -e ".[parsers]"
```

To enable RAG plus object storage support:

```bash
pip install -e ".[rag,storage]"
```

Then set one of:

```yaml
parsing:
  default_parser: "docling"
```

```yaml
parsing:
  default_parser: "paddleocr"  # "paddlerocr" is accepted as an alias
```

PaddleOCR handles raster images directly. For PDFs, LitTrace uses optional
`pypdfium2` to render pages to temporary PNG files, then runs PaddleOCR page by
page and stores page-aware evidence spans.

## Runtime Components

- `LitTrace Coordinator` is the only default agent. It owns the user turn and
  invokes retrieval, access, parsing, extraction, synthesis, and export skills
  through typed ToolContracts.
- Citation, evidence, table, storyline, publication, and evaluation checks are
  deterministic quality gates, not agents.
- `Optional Reviewer` runs only when explicitly requested for a
  high-value report. It receives a bounded evidence bundle, is read-only, has
  no search/download tools, cannot publish, and uses bounded rounds.
- `Publisher E2E` can run real golden-set DOI checks with
  `LITTRACE_RUN_PUBLISHER_E2E=1 pytest tests/test_publisher_e2e.py`; each case must
  download and parse a PDF, otherwise it fails.
- Storyline construction and the local dialogue surface are Coordinator-owned
  skills and product interfaces, not separate agents.

## API Preview

```bash
curl -X POST http://127.0.0.1:8000/search/preview \
  -H "Content-Type: application/json" \
  -d '{"topic":"MXene flexible sensor","limit":5,"live":true}'

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"检索 MXene flexible sensor 的最新论文","live":false}'

curl http://127.0.0.1:8000/context
curl -X POST "http://127.0.0.1:8000/config/init?path=config.yaml"
curl http://127.0.0.1:8000/quality
curl http://127.0.0.1:8000/citations/context
curl -X POST http://127.0.0.1:8000/citations/audit
curl -X POST http://127.0.0.1:8000/downloads/plan
curl -X POST http://127.0.0.1:8000/full-text/resolve
curl -X POST http://127.0.0.1:8000/papers/backfill-dois \
  -H "Content-Type: application/json" \
  -d '{"dois":["10.1021/acs.nanolett.5c01464"]}'
curl http://127.0.0.1:8000/eval/full-text
curl -X POST http://127.0.0.1:8000/downloads/execute \
  -H "Content-Type: application/json" \
  -d '{"paper_ids":[],"dry_run":true}'
curl -X POST http://127.0.0.1:8000/workflow/research \
  -H "Content-Type: application/json" \
  -d '{"search":{"topic":"MXene flexible sensor","live":false},"audit_citations":false,"plan_downloads":false,"parse_full_text":true,"build_storyline":true}'
curl -X POST http://127.0.0.1:8000/parse/context
curl -X POST http://127.0.0.1:8000/tables/extract
curl http://127.0.0.1:8000/tables/matrix
curl http://127.0.0.1:8000/agents/components
curl http://127.0.0.1:8000/agents/quality-audits
curl http://127.0.0.1:8000/agents/workflow
curl "http://127.0.0.1:8000/agents/plan?topic=MXene%20sensor"
curl http://127.0.0.1:8000/eval/golden
curl http://127.0.0.1:8000/publishers/routes
curl "http://127.0.0.1:8000/publishers/search-plan?topic=MXene%20sensor"
curl -X POST "http://127.0.0.1:8000/publishers/retrieve?topic=MXene%20sensor&family=acs"
curl "http://127.0.0.1:8000/publishers/browser-plan?topic=MXene%20sensor&family=acs"
curl -X POST "http://127.0.0.1:8000/publishers/enrich-html?html=<html>...</html>"
curl -X POST "http://127.0.0.1:8000/downloads/login/{paper_id}?dry_run=true"
curl -X POST http://127.0.0.1:8000/downloads/check
curl -X POST "http://127.0.0.1:8000/downloads/resume?session_id={session_id}"
curl -X POST "http://127.0.0.1:8000/papers/{paper_id}/attach-pdf?source_path=/path/to/paper.pdf"
curl -X POST "http://127.0.0.1:8000/papers/{paper_id}/attach-si?source_path=/path/to/si.pdf&session_id={session_id}"
curl http://127.0.0.1:8000/storyline/report
curl http://127.0.0.1:8000/storyline/review
curl http://127.0.0.1:8000/eval/pdf-benchmark
curl http://127.0.0.1:8000/eval/golden
```

`/citations/audit` treats `requires_login` as a valid, traceable access state:
the link resolves, but the user must authenticate through an authorized route.
