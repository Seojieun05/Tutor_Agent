"""Grok ConceptTagger: classify a recognized problem into the two whitelists.

Deliberately a separate call from the VLM recognition. The Recognizer answers
"what is written on this paper"; the tagger answers "what kind of problem is
this and what must you know to solve it". Keeping them apart means a wrong tag
can be reproduced and debugged from `problem_text` alone, without an image.

The model picks from whitelists, and Python enforces them again afterwards —
an invented id is dropped here rather than poisoning the KB or the retrieval.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from tutor.knowledge.concepts import MAX_CONCEPTS, concepts_for_prompt, normalize_concepts
from tutor.knowledge.taxonomy import (
    UNKNOWN_PROBLEM_TYPE,
    normalize_problem_type,
    problem_types_for_prompt,
)
from tutor.llm.client import LLMClient

log = logging.getLogger(__name__)


class ProblemTags(BaseModel):
    problem_type: str = UNKNOWN_PROBLEM_TYPE
    concepts: list[str] = []


def _system_prompt() -> str:
    return f"""You classify a Korean K-12 math problem for a tutoring system.

Return exactly two things:

1. problem_type — the ONE coarse type of the problem, chosen from this list:
{problem_types_for_prompt()}

2. concepts — the curriculum concepts a student actually needs to SOLVE it,
   {MAX_CONCEPTS} at most (fewer is better; [] is fine), chosen from this list:
{concepts_for_prompt()}

Hard rules:
- Use ids EXACTLY as written above. NEVER invent, translate or modify an id.
- problem_type is exactly one id; concepts is a list of 0-{MAX_CONCEPTS} ids.
- concepts are the knowledge required to solve, not keywords from the wording.
  Pick what the solution steps rely on. If the problem merely mentions a
  triangle but is solved by an equation, tag the equation concept.
- Do NOT tag solution strategies (case splitting, arrangement tricks, working
  backwards). Those are not curriculum concepts.
- If you cannot classify it confidently, use problem_type "unknown" and
  concepts []. Guessing is worse than admitting uncertainty.

Example — "GOOD와 2002를 문자와 숫자가 번갈아 오도록 배열하는 경우의 수":
  problem_type: "counting"
  concepts: ["permutation", "permutation_with_identical_elements"]
  (not "alternating_arrangement" — that is a strategy, not a concept.)

Return ONLY the JSON object."""


class ConceptTagger:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def tag(self, rec) -> ProblemTags:
        """rec: tutor.vision.recognizer.Recognition (kept untyped to avoid a
        knowledge → vision import cycle)."""
        raw = self.llm.complete_json(
            purpose="tag",
            system=_system_prompt(),
            user=self._context(rec),
            schema=ProblemTags,
        )
        tags = self.validate(raw)
        log.info(
            "tagged problem_type=%s concepts=%s (model said %s / %s)",
            tags.problem_type,
            tags.concepts,
            raw.problem_type,
            raw.concepts,
        )
        return tags

    @staticmethod
    def validate(raw: ProblemTags) -> ProblemTags:
        """Whitelist enforcement in Python — never trust the model's ids."""
        return ProblemTags(
            problem_type=normalize_problem_type(raw.problem_type),
            concepts=normalize_concepts(raw.concepts),
        )

    @staticmethod
    def _context(rec) -> str:
        parts = [f"문제: {rec.problem_text}"]
        if rec.equations:
            parts.append(f"수식: {rec.equations}")
        if rec.choices:
            parts.append(f"보기: {rec.choices}")
        if rec.diagram_conditions:
            parts.append(f"그림/조건: {rec.diagram_conditions}")
        # student_work is deliberately excluded: the tags describe the PROBLEM,
        # so they must not drift as the student writes (the problem cache and
        # the retrieval keys depend on that stability).
        return "\n".join(parts)
