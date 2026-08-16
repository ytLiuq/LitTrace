# Parsing modes

LitTrace supports two PDF-parsing modes. The default is **text_only**; switch to **OCR** only when the text layer probe reports empty or unreliable output.

## text_only (default)

Extracts the embedded text layer that modern publishers (ACS, Wiley, RSC, Elsevier, Nature, IEEE, MDPI) write into the PDF at production time. Fast (milliseconds per page), perfect fidelity for layout that the publisher already typeset.

| Aspect | Behavior |
|---|---|
| Speed | ~10–100 ms / page |
| Accuracy | 100% for modern publishers |
| Table / figure | Depends on publisher embedding structured text |
| Provenance | paper_id + page + character offset |
| When to use | Almost always — start here |

## OCR

Renders each page to an image and runs PaddleOCR / Docling on it. Slow (seconds per page), useful only when the text layer is missing or corrupt.

| Aspect | Behavior |
|---|---|
| Speed | ~1–10 s / page (PaddleOCR) / slower for Docling |
| Accuracy | ~95% en / ~85% zh, depends on font + scan quality |
| Table / figure | Docling extracts layout; PaddleOCR gives only text stream |
| Provenance | paper_id + page + image bbox |
| When to use | Scanned PDFs, pre-2000 papers, image-only PDFs |

## How to switch

- **Subnav button** — click the `[只看文字层]` / `[使用 OCR]` button (top of window) to toggle the next parse. The status line shows the current mode.
- **Chat flag** — `/parse --ocr` forces OCR; `/parse --text` forces text-only. The chat flag overrides the subnav toggle for that one invocation.
- The probe at parse time inspects the text-layer density; if it returns empty, the recommendation flips to "OCR" but the user must confirm by issuing `/parse --ocr`.

## What we **never** do

- No silent auto-promotion from `text_only` to `ocr`. The user is always in control.
- No auto-download of the source PDF — see "Auto-download is manual-only" in the main README.

## Configuration

The default parser backend is `paddleocr` (set in `config.yaml` under `parsing.default_parser`). Per-paper choice lives on the workspace's `parse_strategy` filter; the runtime honors both.