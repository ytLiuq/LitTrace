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
    confidence: float = 0.0
    ambiguous: bool = False
    ambiguity_types: list[str] = field(default_factory=list)
    ambiguity_reasons: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)


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
    if any(
        token in lowered
        for token in ["只看文字", "文字层", "不要ocr", "不ocr", "text only", "text-only"]
    ):
        intent.parse_strategy = "text_only"
        if "parse" not in intent.actions:
            intent.actions.append("parse")
    elif any(token in lowered for token in ["使用ocr", "用ocr", "强制ocr", "ocr解析", "force ocr"]):
        intent.parse_strategy = "ocr"
        if "parse" not in intent.actions:
            intent.actions.append("parse")
    if any(
        token in lowered
        for token in ["表格", "性能", "对比", "横向比较", "指标", "benchmark", "matrix", "table"]
    ):
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
    if "总结" in lowered and any(
        token in lowered for token in ["路线", "方向", "脉络", "这些论文", "这些文献"]
    ):
        if "storyline" not in intent.actions:
            intent.actions.append("storyline")
    if any(
        token in lowered
        for token in ["报告", "文档", "综述", "research report", "document", "brief"]
    ):
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
    if any(
        token in lowered for token in ["agent状态", "agent 进度", "agents status", "agent status"]
    ):
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
    _score_intent(intent, message)
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


def _score_intent(intent: ChatIntent, message: str) -> None:
    lowered = message.lower().strip()
    if lowered in {"你好", "您好", "hello", "hi", "hey"}:
        intent.confidence = 0.4
        intent.ambiguous = False
        intent.ambiguity_types = []
        intent.ambiguity_reasons = []
        intent.clarification_questions = []
        return
    score = 0.15
    reasons: list[str] = []
    types: list[str] = []
    questions: list[str] = []

    if intent.actions:
        score += min(0.45, 0.18 * len(set(intent.actions)))
    else:
        types.append("missing_action")
        reasons.append("没有识别到明确动作")
        questions.append("你想让我检索文献、解析 PDF、做表格，还是生成发展脉络？")

    if intent.topic and len(intent.topic) >= 4:
        score += 0.25
    elif any(action in intent.actions for action in ["search", "table", "storyline", "document"]):
        types.append("missing_topic")
        reasons.append("主题不够明确")
        questions.append("这个任务的研究主题或关键词是什么？")

    if intent.year_min or intent.journals:
        score += 0.08
    if (
        intent.skip_download
        or intent.parse_strategy
        or intent.select_indices
        or intent.deselect_indices
    ):
        score += 0.07

    if {"search", "parse"} <= set(intent.actions) and not any(
        token in lowered for token in ["全文", "pdf", "已下载", "本地", "parse", "解析"]
    ):
        types.append("underspecified_target")
        reasons.append("同时出现检索和解析，但没有说明解析哪些 PDF")
        questions.append("需要先检索新文献，还是解析当前 session-workspace 里的 PDF？")

    if {"select_downloads", "deselect_downloads"} <= set(intent.actions):
        types.append("conflicting_actions")
        reasons.append("同时包含选择和取消选择下载")
        questions.append("你是要选择下载，还是取消已有下载选择？")

    if intent.select_indices and not any(
        action in intent.actions
        for action in ["select_downloads", "deselect_downloads", "download"]
    ):
        types.append("underspecified_indices")
        reasons.append("给出了序号，但没有明确序号用途")
        questions.append("这些序号是要加入下载列表、取消下载，还是用于分析？")

    if lowered in {"继续", "go on", "continue", "开始", "做吧"}:
        types.append("context_dependent")
        reasons.append("短指令依赖上下文")
        questions.append("要继续上一步的哪个动作？")

    intent.confidence = round(max(0.0, min(1.0, score)), 3)
    intent.ambiguity_types = list(dict.fromkeys(types))
    intent.ambiguity_reasons = list(dict.fromkeys(reasons))
    intent.clarification_questions = list(dict.fromkeys(questions))[:3]
    intent.ambiguous = bool(intent.ambiguity_reasons) and intent.confidence < 0.72


def intent_to_payload(intent: ChatIntent) -> dict[str, object]:
    return {
        "actions": list(intent.actions),
        "topic": intent.topic,
        "year_min": intent.year_min,
        "journals": list(intent.journals),
        "skip_download": intent.skip_download,
        "show_context": intent.show_context,
        "select_all_downloads": intent.select_all_downloads,
        "clear_download_selection": intent.clear_download_selection,
        "auto_replan": intent.auto_replan,
        "parse_strategy": intent.parse_strategy,
        "select_indices": list(intent.select_indices),
        "deselect_indices": list(intent.deselect_indices),
        "confidence": intent.confidence,
        "ambiguous": intent.ambiguous,
        "ambiguity_types": list(intent.ambiguity_types),
        "ambiguity_reasons": list(intent.ambiguity_reasons),
        "clarification_questions": list(intent.clarification_questions),
    }


def intent_from_payload(payload: dict[str, object]) -> ChatIntent:
    return ChatIntent(
        actions=_string_list(payload.get("actions")),
        topic=payload.get("topic") if isinstance(payload.get("topic"), str) else None,
        year_min=payload.get("year_min") if isinstance(payload.get("year_min"), int) else None,
        journals=_string_list(payload.get("journals")),
        skip_download=bool(payload.get("skip_download")),
        show_context=payload.get("show_context")
        if isinstance(payload.get("show_context"), bool)
        else None,
        select_all_downloads=bool(payload.get("select_all_downloads")),
        clear_download_selection=bool(payload.get("clear_download_selection")),
        auto_replan=bool(payload.get("auto_replan")),
        parse_strategy=payload.get("parse_strategy")
        if payload.get("parse_strategy") in {"text_only", "ocr"}
        else None,
        select_indices=_int_list(payload.get("select_indices")),
        deselect_indices=_int_list(payload.get("deselect_indices")),
        confidence=float(payload.get("confidence") or 0.0),
        ambiguous=bool(payload.get("ambiguous")),
        ambiguity_types=_string_list(payload.get("ambiguity_types")),
        ambiguity_reasons=_string_list(payload.get("ambiguity_reasons")),
        clarification_questions=_string_list(payload.get("clarification_questions")),
    )


def merge_pending_intent(
    pending: ChatIntent, clarification: ChatIntent, message: str
) -> ChatIntent:
    merged = ChatIntent(
        actions=list(dict.fromkeys([*pending.actions, *clarification.actions])),
        topic=clarification.topic or pending.topic,
        year_min=clarification.year_min or pending.year_min,
        journals=list(dict.fromkeys([*pending.journals, *clarification.journals])),
        skip_download=pending.skip_download or clarification.skip_download,
        show_context=clarification.show_context
        if clarification.show_context is not None
        else pending.show_context,
        select_all_downloads=pending.select_all_downloads or clarification.select_all_downloads,
        clear_download_selection=pending.clear_download_selection
        or clarification.clear_download_selection,
        auto_replan=pending.auto_replan or clarification.auto_replan,
        parse_strategy=clarification.parse_strategy or pending.parse_strategy,
        select_indices=clarification.select_indices or pending.select_indices,
        deselect_indices=clarification.deselect_indices or pending.deselect_indices,
    )
    if not merged.actions and merged.topic:
        merged.actions = ["search"]
    _score_intent(merged, message)
    if pending.actions and merged.topic:
        merged.ambiguous = False
        merged.ambiguity_types = []
        merged.ambiguity_reasons = []
        merged.clarification_questions = []
        merged.confidence = max(merged.confidence, 0.76)
    return merged


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]
