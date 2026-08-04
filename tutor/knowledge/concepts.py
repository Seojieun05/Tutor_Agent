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

CONCEPTS_PATH = Path(__file__).resolve().parent / "seeds" / "concepts.json"

MAX_CONCEPTS = 4  # a problem needs a handful of ideas, not a syllabus


def _load() -> list[dict]:
    return json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))


CONCEPT_DEFS = _load()

CONCEPT_NAMES: dict[str, str] = {
    c["id"]: c["name"]
    for c in CONCEPT_DEFS
}

ALLOWED_CONCEPT_IDS = frozenset(CONCEPT_NAMES)


def is_allowed_concept(concept_id: str) -> bool:
    return concept_id in ALLOWED_CONCEPT_IDS


def normalize_concepts(concepts: list[str], limit: int = MAX_CONCEPTS) -> list[str]:
    """Drop invented / unsupported concept ids, de-duplicate, cap the count."""
    kept = list(dict.fromkeys(
        c.strip() for c in concepts
        if isinstance(c, str) and c.strip() in ALLOWED_CONCEPT_IDS
    ))
    return kept[:limit]


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