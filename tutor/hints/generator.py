"""Hint generation: verified DB templates first, LLM phrasing as fallback,
answer-leak guard always (spec rules 1, 3, 5).

The phrase call's context contains ONLY the target concept/misconception (and
for L4 the single next step description) — never the full solution or answer.
Hint history arrives prefetched from the orchestrator.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from tutor.hints.guard import leaks_answer
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import MatchResult, ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.policy.engine import Action, Decision
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

FIXED_ACTIONS: dict[Action, str] = {
    Action.ASK_RECAPTURE: "문제와 지금까지 쓴 풀이가 잘 보이게 카메라에 다시 보여 줄래요?",
    Action.PROBE: "방금 쓴 줄을 소리 내어 읽어 줄래요? 어떻게 생각했는지 듣고 싶어요.",
    Action.WAIT: "",
}

_PHRASE_SYSTEM = """You are a Socratic math tutor speaking Korean to a student.
Produce exactly ONE short spoken hint (1-2 sentences) at the requested level:
L1 = a Socratic question, L2 = a concept reminder, L3 = a procedural nudge,
L4 = reveal only the given next step.

Hard rules:
- NEVER state the final answer or any result beyond the given step.
- NEVER solve the problem. Guide with questions and concepts.
- Korean 존댓말(해요체), friendly, spoken style — never 반말.
- Never repeat an earlier hint: the ones already given are listed, and the student
  may have moved on to a later step since then.
- You may look up hint templates and misconceptions in the knowledge base.

Return ONLY JSON: {"hint": "..."}"""


class PhrasedHint(BaseModel):
    hint: str


class HintGenerator:
    def __init__(self, llm: LLMClient, db: KnowledgeDB):
        self.llm = llm
        self.db = db

    def generate(
        self,
        decision: Decision,
        match: MatchResult,
        reference: ReferenceSolution | None,
        rec: Recognition,
        history: list[HintRecord],
    ) -> str:
        if decision.action in FIXED_ACTIONS:
            return FIXED_ACTIONS[decision.action]

        if reference is not None and decision.target_step > len(reference.steps):
            # every reference step is done — nothing left to hint at
            return "훌륭해요, 문제를 끝까지 풀었네요! 어떻게 구했는지 스스로 설명해 볼까요?"

        slots = self._slots(decision, match, reference)
        # Anything already said this problem is off the table: a concept-level
        # template fits every step, so reusing it verbatim is how the tutor
        # ends up asking the same question after the student has progressed.
        given = {h.hint_text for h in history if h.hint_text}

        # 1) Verified DB pedagogy first (spec rule 1) — concept/misconception
        # specific templates only; fully-generic ones stay the last resort.
        for template in self.db.hint_templates_for(
            match.concepts, decision.misconception, decision.level
        ):
            if template.concept_id is None and template.misconception_id is None:
                continue
            try:
                text = template.template_text.format(**slots)
            except (KeyError, IndexError):
                continue  # a slot we cannot fill
            if text in given:
                continue
            if reference is not None and leaks_answer(text, reference, decision.target_step):
                continue
            return text

        # 2) LLM phrasing fallback with minimal context.
        text = self._phrase(decision, match, slots, history)
        if reference is not None and leaks_answer(text, reference, decision.target_step):
            log.warning("hint leaked answer; regenerating once")
            text = self._phrase(decision, match, slots, history, stronger=True)
            if leaks_answer(text, reference, decision.target_step):
                return self._generic_fallback(decision, slots)
        return text

    def _slots(
        self,
        decision: Decision,
        match: MatchResult,
        reference: ReferenceSolution | None,
    ) -> dict[str, str]:
        slots: dict[str, str] = {}
        if match.problem is not None:
            slots.update(match.problem.parameters)
        if match.bindings:
            slots.update(match.bindings)
        if "b" in slots:
            slots.setdefault("term", slots["b"])
        if reference is not None:
            # L4 reveals only the next step's DESCRIPTION, never its expression
            # (the last step's expression is often the answer itself).
            target = next(
                (s for s in reference.steps if s.idx == decision.target_step), None
            )
            if target is not None:
                slots.setdefault("step", target.description)
        return slots

    def _phrase(
        self,
        decision: Decision,
        match: MatchResult,
        slots: dict[str, str],
        history: list[HintRecord],
        stronger: bool = False,
    ) -> str:
        parts = [
            f"힌트 레벨: L{decision.level} ({decision.action.value})",
            f"개념: {', '.join(match.concepts) or '알 수 없음'}",
        ]
        if decision.misconception:
            m = self.db.get_misconception(decision.misconception)
            parts.append(
                f"진단된 오개념: {m.description if m else decision.misconception}"
            )
        if "step" in slots:
            # Every level is aimed at the SAME target step; only L4 may say it
            # out loud. Without this the tutor asks about step 1 forever, even
            # after the student has moved on.
            if decision.level >= 4:
                parts.append(f"알려줘도 되는 다음 단계: {slots['step']}")
            else:
                parts.append(
                    f"학생이 지금 해내야 하는 단계 (절대 그대로 말하지 말 것): {slots['step']}\n"
                    "이 단계를 학생이 스스로 떠올리도록 이끄는 내용만 말하세요."
                )
        if history:
            parts.append(
                "이미 준 힌트 (반복 금지): "
                + " / ".join(h.hint_text for h in history if h.hint_text)
            )
        if stronger:
            parts.append(
                "경고: 이전 시도가 답을 노출했습니다. 어떤 수치나 최종 결과도 말하지 마세요."
            )
        result = self.llm.run_with_tools(
            purpose="phrase", system=_PHRASE_SYSTEM, user="\n".join(parts), schema=PhrasedHint
        )
        return result.hint.strip()

    def _generic_fallback(self, decision: Decision, slots: dict[str, str]) -> str:
        for template in self.db.hint_templates_for([], None, decision.level):
            if template.concept_id is None and template.misconception_id is None:
                try:
                    return template.template_text.format(**slots)
                except (KeyError, IndexError):
                    continue
        return "지금까지 한 풀이를 처음부터 소리 내어 설명해 볼까요?"
