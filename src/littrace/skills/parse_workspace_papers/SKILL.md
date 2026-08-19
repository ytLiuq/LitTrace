# `parse_workspace_papers`

合同名：`parse_workspace_papers`（category=`document_parsing`）。

## 输入 / 输出
- in：`LiteratureWorkspace` + `LitTraceConfig`
- out：`(workspace, parse_report)` —— 会写回 `parsed_papers`，故 `mutates_workspace=True`

## 副作用
走本地 OCR / Docling 解析器，可能读 PDF 文件、调用 paddleocr。