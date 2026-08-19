# `build_quality_metrics`

合同名：`build_quality_metrics`（category=`evaluation`）。

`requires_network=False`，输出 `dict[str, object]`。
内部委托 `evaluation/quality_report.build_quality_report(...).metrics`，
因此与 `quality_report` 共享同一计算逻辑，仅 contract 维度不同。

这是补齐原 14 个契约里唯一一个缺 `*_skill` wrapper 的 skill。