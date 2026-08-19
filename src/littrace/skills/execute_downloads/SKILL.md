# `execute_downloads`

合同名：`execute_downloads`（category=`access`）。

`requires_network=True`、`side_effects=["network","filesystem"]`、`idempotent=False`。
按 `DownloadExecutionRequest` 执行实际 PDF 下载；失败不重试，调用方负责。