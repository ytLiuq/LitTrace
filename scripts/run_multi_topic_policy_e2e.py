#!/usr/bin/env python3
"""Run real LLM policy and live retrieval checks for multiple unrelated topics."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from littrace.config import load_config
from littrace.research_background import assess_research_background
from littrace.models import PaperSearchRequest
from littrace.retrieval.search import filter_papers_by_retrieval_policy
from littrace.skill_runner import search_papers_skill


@dataclass(frozen=True)
class TopicCase:
    name: str
    background: str


CASES = (
    TopicCase("flexible_pressure", "近五年柔性薄膜压阻压力传感器的材料、微结构与长期稳定性"),
    TopicCase(
        "battery",
        "我希望系统跟踪近五年硫化物固态锂金属电池中锂负极与固态电解质界面稳定性，重点比较界面阻抗增长、枝晶抑制和循环寿命。",
    ),
    TopicCase(
        "water_treatment",
        "我希望系统跟踪近五年用于水中抗生素残留去除的光催化膜材料，重点比较可见光降解效率、通量和长期抗污染稳定性。",
    ),
)


async def main() -> None:
    config = load_config()
    config.api.enable_live_search = True
    results = []
    for case in CASES:
        assessment = await assess_research_background(case.background, config)
        if not assessment.accepted or assessment.retrieval_policy is None:
            results.append(
                {
                    "name": case.name,
                    "gate": "rejected",
                    "reason": assessment.reason,
                    "suggestions": assessment.suggestions,
                }
            )
            continue
        policy = assessment.retrieval_policy
        request = PaperSearchRequest(
            topic=policy.canonical_topic,
            query_variants=policy.query_variants,
            year_min=2021,
            limit=20,
            live=True,
        )
        search = await search_papers_skill(request, config)
        accepted, rejected = filter_papers_by_retrieval_policy(search.result.papers, policy)
        results.append(
            {
                "name": case.name,
                "gate": "accepted",
                "canonical_topic": policy.canonical_topic,
                "required_concept_groups": policy.required_concept_groups,
                "excluded_concepts": policy.excluded_concepts,
                "query_variants": policy.query_variants,
                "searched": len(search.result.papers),
                "accepted": len(accepted),
                "rejected": len(rejected),
            }
        )
        if not search.result.papers or not accepted:
            raise RuntimeError(f"{case.name}: live retrieval produced no policy-accepted papers")
    print(json.dumps({"topics": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
