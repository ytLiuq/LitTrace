# `build_download_plan`

合同名：`build_download_plan`（category=`access`）。

按 `workspace.context.selected_for_download` + active papers 生成合规模块化下载计划（`DownloadPlan`）。
不实际下载，由下游 `execute_downloads` 执行。