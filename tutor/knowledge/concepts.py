"""Concept whitelist: the fine-grained "what knowledge does solving this need"
layer, loaded from seeds/concepts.json (the source of truth).

Partner of `tutor/knowledge/taxonomy.py`, which owns the single coarse
problem_type. A tagged problem carries both:

    problem_type = "counting"
    concepts     = ["permutation", "permutation_with_identical_elements"]

Nothing outside this list is ever accepted — an LLM that invents an id has it
dropped here, not downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

# 개념 화이트리스트 원본 파일.
CONCEPTS_PATH = Path(__file__).resolve().parent / "seeds" / "concepts.json"

# 문제 하나에 붙일 수 있는 개념 최대 개수.
MAX_CONCEPTS = 4  # a problem needs a handful of ideas, not a syllabus


# seeds/concepts.json 로드.
def _load() -> list[dict]:
    return json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))


CONCEPT_DEFS = _load()

# 개념 id → 한국어 이름.
CONCEPT_NAMES: dict[str, str] = {
    c["id"]: c["name"]
    for c in CONCEPT_DEFS
}

# 허용된 개념 id 집합. 이 밖의 값은 모두 버린다.
ALLOWED_CONCEPT_IDS = frozenset(CONCEPT_NAMES)


# 화이트리스트에 있는 개념인지.
def is_allowed_concept(concept_id: str) -> bool:
    return concept_id in ALLOWED_CONCEPT_IDS


# 모델이 만들어 낸 가짜 id를 버리고 중복 제거 후 개수를 자른다.
def normalize_concepts(concepts: list[str], limit: int = MAX_CONCEPTS) -> list[str]:
    """Drop invented / unsupported concept ids, de-duplicate, cap the count."""
    kept = list(dict.fromkeys(
        c.strip() for c in concepts
        if isinstance(c, str) and c.strip() in ALLOWED_CONCEPT_IDS
    ))
    return kept[:limit]


# 프롬프트에 넣을 개념 메뉴(영역별로 묶어서).
def concepts_for_prompt() -> str:
    """The whitelist as a menu, grouped by curriculum area."""
    by_category: dict[str, list[dict]] = {}
    for c in CONCEPT_DEFS:
        by_category.setdefault(c.get("category", "기타"), []).append(c)
    blocks = []
    for category, items in by_category.items():
        lines = "\n".join(f"- {c['id']}: {c['name']}" for c in items)
        blocks.append(f"[{category}]\n{lines}")
    return "\n".join(blocks)