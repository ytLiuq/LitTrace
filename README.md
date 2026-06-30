# LitTrace

LitTrace is a materials and chemistry research agent for traceable literature
workflows. It helps researchers search papers, manage the active literature
context, optionally download PDFs, parse full text with OCR tools, build
evidence-grounded comparison tables, and generate truthful research storylines.

LitTrace is intentionally a **LangGraph + CrewAI** project:

- **LangGraph** is the primary stateful workflow engine for source routing,
  search, citation auditing, download planning, OCR parsing, and later
  storyline/table verification.
- **CrewAI** is the optional role layer for research-team style agents such as
  Source Router, Citation Verifier, Access Manager, and Storyline Verifier.

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn littrace.api.app:app --reload
```

For the Codex-style local agent shell:

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
/login N
/parse
/table
/storyline
/benchmark
/export
/quit
```

Conversation examples:

```text
检索 2024 年后的 AFM 和 ACS Nano，先别下载，生成性能对比表
显示上下文
选择第 1、3 篇下载
全部下载
取消选择第 2 篇
生成发展脉络
```

For login-gated papers, `/login N` opens the authorized publisher or DOI page in
the local browser and prints the exact target path where the PDF should be
saved for the current session/parser flow.

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
- `Access Manager` plans compliant downloads and marks login-required papers
  instead of bypassing authentication. Login-required papers return an
  `open_login_popup` action with the authorized URL, target path, and manual
  handoff instructions.
- `Publisher Connector` maps papers to publisher families such as ACS, Wiley,
  Nature, MDPI, RSC, and Elsevier, then emits DOI/publisher access routes.
- `PDF/OCR Parser` exposes metadata-only, Docling, and PaddleOCR tools.
- `Table Agent` extracts performance cells into evidence-preserving matrices.
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
curl http://127.0.0.1:8000/citations/context
curl -X POST http://127.0.0.1:8000/citations/audit
curl -X POST http://127.0.0.1:8000/downloads/plan
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
curl http://127.0.0.1:8000/publishers/routes
curl "http://127.0.0.1:8000/publishers/search-plan?topic=MXene%20sensor"
curl -X POST "http://127.0.0.1:8000/downloads/login/{paper_id}?dry_run=true"
curl http://127.0.0.1:8000/eval/pdf-benchmark
```

`/citations/audit` treats `requires_login` as a valid, traceable access state:
the link resolves, but the user must authenticate through an authorized route.
