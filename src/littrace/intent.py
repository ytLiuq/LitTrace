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

    if any(token in lowered for token in ["检索", "搜索", "查找", "search", "papers"]):
        intent.actions.append("search")
    if any(token in lowered for token in ["下载", "download"]):
        intent.actions.append("download")
    if any(token in lowered for token in ["解析", "全文", "ocr", "parse"]):
        intent.actions.append("parse")
    if any(token in lowered for token in ["表格", "性能", "对比", "matrix", "table"]):
        intent.actions.append("table")
    if any(token in lowered for token in ["故事", "脉络", "发展", "storyline", "narrative"]):
        intent.actions.append("storyline")
    if any(token in lowered for token in ["当前文献", "参考了哪些", "context"]):
        intent.actions.append("list_context")
    if any(token in lowered for token in ["隐藏上下文", "隐藏文献", "hide context"]):
        intent.show_context = False
        intent.actions.append("hide_context")
    if any(token in lowered for token in ["显示上下文", "显示文献", "show context"]):
        intent.show_context = True
        intent.actions.append("show_context")
    if any(token in lowered for token in ["先别下载", "不要下载", "不下载", "skip download"]):
        intent.skip_download = True
        intent.actions = [action for action in intent.actions if action != "download"]

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
        r"(请|帮我|please|search|检索|搜索|查找|最新|论文|文献|papers?|articles?|只保留|排除|生成|先别下载|不要下载|不下载)",
        " ",
        message,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(20\d{2})\s*(年)?(后|以后|之后|以来|起)?", " ", cleaned)
    for alias in JOURNAL_ALIASES:
        cleaned = re.sub(re.escape(alias), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ：:，,。.")
    return cleaned or message.strip()
