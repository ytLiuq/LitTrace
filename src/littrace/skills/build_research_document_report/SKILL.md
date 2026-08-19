# `build_research_document_report`

合同名：`build_research_document_report`（category=`synthesis`）。

`provenance_outputs=["Claim","VerificationReport","ReleaseSnapshot"]`、
`quality_requirements=["claim_verifier","publication_gate"]`。

会触发完整的 claim 验证 + 发布门，输出 `ResearchDocumentReport`（含 `release_blockers`）。