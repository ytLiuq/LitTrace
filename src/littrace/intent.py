from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChatIntent:
    actions: list[str] = field(default_factory=list)
    topic: str | None = None
    year_min: int | None = None
    journals: list[str] = field(default_factory=list)
    skip_download: bool = False
    show_context: bool | None = None
    select_all_downloads: bool = False
    clear_download_selection: bool = False
    auto_replan: bool = False
    parse_strategy: str | None = None
    select_indices: list[int] = field(default_factory=list)
    deselect_indices: list[int] = field(default_factory=list)


JOURNAL_ALIASES = {
    "afm": "Advanced Functional Materials",
    "advanced functional materials": "Advanced Functional Materials",
    "am": "Advanced Materials",
    "advanced materials": "Advanced Materials",
    "acs nano": "ACS Nano",
    "nano letters": "Nano Letters",
    "nature materials": "Nature Materials",
    "mdpi": "MDPI",
}


def parse_chat_intent(message: str) -> ChatIntent:
    lowered = message.lower()
    intent = ChatIntent()

    if any(
        token in lowered
        for token in [
            "检索",
            "搜索",
            "查找",
            "调研",
            "了解",
            "研究一下",
            "找一些",
            "找几篇",
            "有哪些论文",
            "有什么文献",
            "相关工作",
            "文献调研",
            "literature",
            "survey",
            "search",
            "papers",
        ]
    ):
        intent.actions.append("search")
    if any(token in lowered for token in ["下载", "保存", "存到本地", "download"]):
        intent.actions.append("download")
    if any(token in lowered for token in ["选择", "选中", "勾选", "select"]):
        intent.actions.append("select_downloads")
    if any(token in lowered for token in ["取消选择", "取消下载", "deselect", "unselect"]):
        intent.actions.append("deselect_downloads")
    if any(token in lowered for token in ["全部下载", "全都下载", "下载全部", "download all"]):
        intent.select_all_downloads = True
        if "select_downloads" not in intent.actions:
            intent.actions.append("select_downloads")
    if any(token in lowered for token in ["清空下载", "都不下载", "clear downloads"]):
        intent.clear_download_selection = True
        if "deselect_downloads" not in intent.actions:
            intent.actions.append("deselect_downloads")
    if any(token in lowered for token in ["解析", "全文", "ocr", "parse"]):
        intent.actions.append("parse")
    if any(token in lowered for token in ["只看文字", "文字层", "不要ocr", "不ocr", "text only", "text-only"]):
        intent.parse_strategy = "text_only"
        if "parse" not in intent.actions:
            intent.actions.append("parse")
    elif any(token in lowered for token in ["使用ocr", "用ocr", "强制ocr", "ocr解析", "force ocr"]):
        intent.parse_strategy = "ocr"
        if "parse" not in intent.actions:
            intent.actions.append("parse")
    if any(token in lowered for token in ["表格", "性能", "对比", "横向比较", "指标", "benchmark", "matrix", "table"]):
        intent.actions.append("table")
    if any(
        token in lowered
        for token in [
            "故事",
            "脉络",
            "发展",
            "主要路线",
            "研究路线",
            "技术路线",
            "路线",
            "方向",
            "演进",
            "来龙去脉",
            "前后关系",
            "storyline",
            "narrative",
            "evolution",
        ]
    ):
        intent.actions.append("storyline")
    if "总结" in lowered and any(token in lowered for token in ["路线", "方向", "脉络", "这些论文", "这些文献"]):
        if "storyline" not in intent.actions:
            intent.actions.append("storyline")
    if any(token in lowered for token in ["报告", "文档", "综述", "research report", "document", "brief"]):
        intent.actions.append("document")
    if any(
        token in lowered
        for token in [
            "辩论",
            "反驳",
            "多轮",
            "修订",
            "reviewer loop",
            "autonomous",
            "debate",
            "revise",
        ]
    ):
        intent.actions.append("autonomous_review")
    if any(token in lowered for token in ["自动重规划", "自动补救", "自动优化", "auto replan"]):
        intent.auto_replan = True
        if "autonomous_review" not in intent.actions:
            intent.actions.append("autonomous_review")
    if any(token in lowered for token in ["当前文献", "参考了哪些", "context"]):
        intent.actions.append("list_context")
    if any(token in lowered for token in ["agent状态", "agent 进度", "agents status", "agent status"]):
        intent.actions.append("agent_status")
    if any(token in lowered for token in ["隐藏上下文", "隐藏文献", "hide context"]):
        intent.show_context = False
        intent.actions.append("hide_context")
    if any(token in lowered for token in ["显示上下文", "显示文献", "show context"]):
        intent.show_context = True
        intent.actions.append("show_context")
    if any(token in lowered for token in ["先别下载", "不要下载", "不下载", "skip download"]):
        intent.skip_download = True
        intent.actions = [action for action in intent.actions if action != "download"]

    indices = _extract_indices(message)
    if indices:
        if "取消" in message or "deselect" in lowered or "unselect" in lowered:
            intent.deselect_indices = indices
        elif "下载" in message or "选择" in message or "select" in lowered:
            intent.select_indices = indices

    year_match = re.search(r"(20\d{2})\s*(?:年)?(?:后|以后|之后|以来|起)?", message)
    if year_match:
        intent.year_min = int(year_match.group(1))

    for key, canonical in JOURNAL_ALIASES.items():
        if key in lowered:
            intent.journals.append(canonical)
    intent.journals = list(dict.fromkeys(intent.journals))
    intent.topic = topic_from_message(message)
    return intent


def topic_from_message(message: str) -> str:
    cleaned = re.sub(
        r"(我想|一下|请|帮我|please|search|检索|搜索|查找|调研|了解|研究一下|相关|最新|论文|文献|papers?|articles?|只保留|排除|生成|先别下载|不要下载|不下载)",
        " ",
        message,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(详细|深入|系统|研究|调研|帮我|一下)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(20\d{2})\s*(年)?(后|以后|之后|以来|起)?", " ", cleaned)
    for alias in JOURNAL_ALIASES:
        cleaned = re.sub(re.escape(alias), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ：:，,。.")
    cleaned = re.sub(r"(的|关于|有关)$", "", cleaned).strip(" ：:，,。.")
    cleaned = re.sub(r"^[的和与及，,、\s]+|[的和与及，,、\s]+$", "", cleaned).strip(" ：:，,。.")
    if not cleaned or cleaned in {"表格", "性能", "对比", "性能对比表"}:
        fallback = re.sub(
            r"(请|帮我|please|search|检索|搜索|查找|生成|先别下载|不要下载|不下载|表格|性能|对比)",
            " ",
            message,
            flags=re.IGNORECASE,
        )
        fallback = re.sub(r"(20\d{2})\s*(年)?(后|以后|之后|以来|起)?", " ", fallback)
        fallback = re.sub(r"\s+", " ", fallback).strip(" ：:，,。.")
        return fallback or message.strip()
    return cleaned or message.strip()


def _extract_indices(message: str) -> list[int]:
    indices: list[int] = []
    for match in re.finditer(r"(?:第\s*)?(\d{1,3})(?:\s*篇)?", message):
        value = int(match.group(1))
        if value > 0:
            indices.append(value)
    return list(dict.fromkeys(indices))
