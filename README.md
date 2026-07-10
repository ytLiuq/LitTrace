# LitTrace

LitTrace is a materials and chemistry research agent for traceable literature
workflows. It helps researchers search papers, manage the active literature
context, optionally download PDFs, parse full text with OCR tools, build
evidence-grounded comparison tables, and generate truthful research storylines.

LitTrace is intentionally a **LangGraph** project:

- **LangGraph** is the primary stateful workflow engine for source routing,
  search, citation auditing, download planning, OCR parsing, and later
  storyline/table verification.
- **13 Agent roles** (defined in `agents.py`) describe the system's
  organizational structure and tool ownership. They are executed as LangGraph
  nodes or direct Python function calls—not via an external agent framework.

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
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=19222
littrace doctor
littrace-window
```

Publisher-authenticated full-text access uses a local Chrome CDP connection,
not BrowserAct. Start Chrome with a remote debugging port before running real
publisher downloads:

```yaml
cdp_downloader:
  cdp_url: "http://127.0.0.1:19222"
```

You can override the CDP endpoint with `LITTRACE_CDP_URL`.

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
/agent-flow
/agent-audits
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
coverage, citation coverage, and agent handoff progress.

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

Each shell run creates a folder under `storage.sessions_dir` (default:
`./sessions/<timestamp-id>/`) containing:

```text
workspace.json
messages.jsonl
artifacts/
```

Copy `config.example.yaml` to `config.yaml` and set `storage.paper_library_dir`
before running workflows that may download PDFs.

To enable the optional Docling parser backend:

```bash
pip install -e ".[parsers]"
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

## Agent Status

- `Source Router` routes materials/chemistry queries toward OpenAlex, Crossref,
  Unpaywall, and preferred publisher families.
- `Citation Verifier` builds citation records and audits access links.
- `Access Manager` plans compliant downloads and uses the CDP publisher
  downloader for login-gated papers. It asks the user to complete required
  Cloudflare or institutional authentication in local Chrome, then downloads
  via publisher-specific PDF routes.
- `Publisher Connector` maps papers to publisher families such as ACS, Wiley,
  Nature, MDPI, RSC, and Elsevier, then emits DOI/publisher access routes.
- `PDF/OCR Parser` requires a local PDF and uses Docling or PaddleOCR; metadata/abstract
  fallback is disabled for research answers.
- `Table Agent` extracts performance cells into evidence-preserving matrices.
- `Research Planner` turns a question into a retrieval/parse/table/storyline
  plan using the current context.
- `Research Writer` produces evidence-grounded answers behind citation guard.
- `Eval Auditor` reports agent strength, quality metrics, and golden-set status.
- `Publisher E2E` can run real golden-set DOI checks with
  `LITTRACE_RUN_PUBLISHER_E2E=1 pytest tests/test_publisher_e2e.py`; each case must
  download and parse a PDF, otherwise it fails.
- `Research Storyline Agent` builds conservative solution-limit-response chains
  from parsed evidence and refuses unsupported broad narratives.
- `Dialogue Agent` is the primary product surface: a local shell with a
  hideable literature context panel.

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
curl http://127.0.0.1:8000/agents/crew
curl http://127.0.0.1:8000/agents/status
curl http://127.0.0.1:8000/agents/strength
curl http://127.0.0.1:8000/agents/audits
curl http://127.0.0.1:8000/agents/interactions
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
