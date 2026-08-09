"""Hint generation: verified DB templates first, LLM phrasing as fallback,
answer-leak guard always (spec rules 1, 3, 5).

The phrase call's context contains ONLY the target concept/misconception (and
for L4 the single next step description) — never the full solution or answer.
Hint history arrives prefetched from the orchestrator.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from tutor.hints.guard import leaks_answer
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import MatchResult, ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.policy.engine import Action, Decision
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

# "Show me again" means different things depending on where the picture comes
# from, and telling a student to hold a worksheet up to a camera that is not
# there is worse than saying nothing.
RECAPTURE_TEXTS: dict[str, str] = {
    "upload": "문제와 지금까지 쓴 풀이가 잘 보이게 사진을 다시 올려 줄래요?",
    "camera": "문제와 지금까지 쓴 풀이가 잘 보이게 카메라에 다시 보여 줄래요?",
}
DEFAULT_INPUT_MODE = "upload"

FIXED_ACTIONS: dict[Action, str] = {
    Action.ASK_RECAPTURE: RECAPTURE_TEXTS[DEFAULT_INPUT_MODE],
    Action.PROBE: "방금 쓴 줄을 소리 내어 읽어 줄래요? 어떻게 생각했는지 듣고 싶어요.",
    Action.WAIT: "",
}

# Written on Google's LearnLM prompt guide (PARTS: Persona, Act, Recipient,
# Theme, Structure) and its five learning-science principles. The guide's own
# math-coach exemplar is "use one step per turn, encourage them to explain their
# thinking, if they're stuck give a gentle nudge, not the answer" — which is the
# behaviour this tutor's policy engine already decides. So the prompt states the
# role and the pedagogy first and the prohibitions second, rather than being the
# wall of NEVERs it was: a model told what to do holds the line better than one
# told only what to avoid.
_PHRASE_SYSTEM = """PERSONA
You are a warm, confidence-building math tutor speaking Korean out loud to one
student sitting beside you. You coach; you do not solve.

ACT
Write exactly ONE spoken hint for this student's current moment, at the hint
level you are given. Someone else has already decided that level from evidence
of how the student is doing — your job is to phrase it, not to re-choose it:
  L1  a question that makes them think, not a hint
  L2  the concept or principle they need, no procedure
  L3  the procedure to try, no result
  L4  say the one given step, and nothing past it

RECIPIENT
A student mid-problem who can hear you. They may have just said something, and
they may have a diagnosed misconception — both are given below. Meet them where
they are: build on the part they already got right, and never ask again for
something they have already told you.

THEME
Only the problem in front of them, the concepts it needs, and the single step
they are stuck on. Everything else is out of scope.

STRUCTURE
One or two short spoken sentences. Korean 존댓말(해요체), never 반말. No lists,
no markdown, no symbols read aloud badly — say "3 x 더하기 5" rather than
"3x + 5" when it is easier to hear.

BOARD
Besides the spoken hint you may WRITE 0-2 short expressions on the student's
screen (`board`) — what a tutor would jot on the whiteboard while saying the
hint. Only mathematics that is already in front of the student: the problem's
own equation, a line they wrote themselves, the expression your question is
pointing at. ASCII notation (3*x + 5 = 20, x**2, sqrt(2), log_3 b), one
expression per entry, no Korean words. NEVER write anything the student has
not reached: no next-step result, no simplified form, no final answer — the
board must not show what the voice is forbidden to say. Most hints need an
empty board; write only when pointing at an expression genuinely helps.

HOW TO TEACH (this is the part that matters)
- Active learning: leave the thinking to them. The best hint is the weakest one
  that still unblocks — productive struggle is the lesson, not an obstacle.
- Cognitive load: one idea per turn. One question, not two. Never a checklist.
- Metacognition: prefer asking how they got there or why they chose that, over
  asking only for the next number. At L1 especially, "왜 그렇게 했어요?" and
  "어디까지는 확실해요?" teach more than a nudge toward the answer.
- Adapt: use their own words back. If they are on their third attempt, be
  warmer and more concrete, not more repetitive.
- Curiosity: an open question beats a yes/no one.

NEVER
- Never state the final answer, or any result beyond the given step — not as a
  check, not as an example, not "so it becomes 15".
- Never solve the problem for them, even partially, except the one step L4 gives.
- Never repeat a hint already given; they are listed, and the student has moved
  on since.
- Never open with 네 / 맞아요 / 좋아요 / 그렇죠 / 음. The reaction to their answer
  is written separately and is spoken just before yours; starting with your own
  makes the tutor say it twice.

Return ONLY JSON: {"hint": "...", "board": ["...", "..."]}"""

_EXPLAIN_SYSTEM = """PERSONA
You are a warm math tutor speaking Korean out loud to one student.

ACT
The student did NOT attempt your question — they asked one of their own, usually
"왜 그렇게 해요?". Answer THEIR question, then hand the turn back to them.

RECIPIENT
A curious student mid-problem. A question is engagement, not failure: treat it
as the opening it is, and reward it with a real reason rather than a redirect.

STRUCTURE
One or two short spoken sentences, then the question you had asked, again.
Korean 존댓말(해요체), never 반말.

HOW TO TEACH
- Explain WHY the step works — the principle, the reason, an everyday parallel.
  Understanding the reason is what they will still have next week.
- One idea. A second reason is a second turn.
- Never state the final answer, the result of this step, or any later step.
  Motivate the step; do not carry it out for them.
- Do not open with 네 / 맞아요 / 궁금하시죠. Answer directly.

Return ONLY JSON: {"hint": "..."}"""

# One acknowledgement per turn. The evaluator already reacted ("맞아요, 그렇게
# 하면 돼요!"), so a hint that opens with its own "네," produces the doubled
# "맞아요, 그렇게 하면 돼요! 네, 그렇게 하면 돼요." Prompts alone do not hold
# this line reliably, so it is also enforced here.
_ACK_WORDS = (
    r"네+|예|맞아요|맞습니다|맞아|좋아요|좋습니다|그렇죠|그렇습니다|그래요|"
    r"훌륭해요|훌륭합니다|잘했어요|잘하셨어요|정확해요|정확합니다|음+|아+|오+|와+"
)
# Punctuation after the word is required, so "네 번째", "아래", "오른쪽" survive.
_ACK_PREFIX_RE = re.compile(rf"^\s*(?:{_ACK_WORDS})\s*[,!.…~·]+\s*")


def strip_leading_acknowledgement(text: str, rounds: int = 2) -> str:
    """Drop a leading '네,' / '맞아요!' — someone else already said it."""
    out = text
    for _ in range(rounds):
        stripped = _ACK_PREFIX_RE.sub("", out, count=1).lstrip()
        if stripped == out or not stripped:
            break
        out = stripped
    return out or text


class PhrasedHint(BaseModel):
    hint: str
    # what the tutor writes while saying it: 0-2 expressions, screened by the
    # same leak guard as the words — the board may not show what the voice
    # may not say
    board: list[str] = []


class SpokenHint(str):
    """The hint text, carrying the board it was phrased with.

    A str subclass so every existing consumer (strip_leading_acknowledgement,
    the leak guard, history records, TTS) keeps working untouched; the session
    reads `.board` off it before any string operation strips the subclass
    away. Plain str returns (templates, fixed actions) read as an empty board
    through getattr.
    """

    board: tuple[str, ...] = ()


def _with_board(text: str, board: tuple[str, ...]) -> "SpokenHint":
    out = SpokenHint(text)
    out.board = board
    return out


def visible_to_student(rec: Recognition | None) -> list[str]:
    """What is printed on the page in front of them.

    The leak guard uses this to tell "you already know this number" from
    "here is the answer": in `3x + 5 = 20` the answer 5 is also a coefficient,
    and refusing to let the tutor say it costs the best question on that
    problem.
    """
    if rec is None:
        return []
    return [rec.problem_text, *rec.equations, *rec.choices, *rec.diagram_conditions]


class HintGenerator:
    def __init__(self, llm: LLMClient, db: KnowledgeDB, input_mode: str = DEFAULT_INPUT_MODE):
        self.llm = llm
        self.db = db
        self.fixed = dict(FIXED_ACTIONS)
        self.fixed[Action.ASK_RECAPTURE] = RECAPTURE_TEXTS.get(
            input_mode, FIXED_ACTIONS[Action.ASK_RECAPTURE]
        )

    def generate(
        self,
        decision: Decision,
        match: MatchResult,
        reference: ReferenceSolution | None,
        rec: Recognition,
        history: list[HintRecord],
        student_answer: str | None = None,
    ) -> str:
        if decision.action in self.fixed:
            return self.fixed[decision.action]

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
            self._concepts_for(match, rec), decision.misconception, decision.level
        ):
            if template.concept_id is None and template.misconception_id is None:
                continue
            try:
                text = template.template_text.format(**slots)
            except (KeyError, IndexError):
                continue  # a slot we cannot fill
            if text in given:
                continue
            if reference is not None and leaks_answer(
                text, reference, decision.target_step, visible_to_student(rec)
            ):
                continue
            return text

        # 2) LLM phrasing fallback with minimal context.
        text, board = self._phrase(
            decision, match, slots, history, rec, reference, student_answer=student_answer
        )
        seen = visible_to_student(rec)
        if reference is not None and leaks_answer(
            text, reference, decision.target_step, seen
        ):
            log.warning("hint leaked answer; regenerating once")
            text, board = self._phrase(
                decision, match, slots, history, rec, reference,
                stronger=True, student_answer=student_answer,
            )
            if leaks_answer(text, reference, decision.target_step, seen):
                return self._generic_fallback(decision, slots)
        # The board passes the SAME gate as the voice, line by line: writing
        # "x = 5" while carefully not saying it is still giving the answer.
        if reference is not None:
            board = tuple(
                b for b in board
                if not leaks_answer(b, reference, decision.target_step, seen)
            )
        return _with_board(text, board)

    def explain(
        self,
        *,
        student_question: str,
        tutor_question: str,
        match: MatchResult,
        reference: ReferenceSolution | None,
        rec: Recognition | None,
        target_step: int,
    ) -> str:
        """Answer a student's "왜 그렇게 해요?" instead of grading it.

        Same guard rails as a hint: the target step may be motivated, but its
        result, the final answer and every later step stay out.
        """
        concepts = ", ".join((rec.concepts if rec else []) or match.concepts) or "알 수 없음"
        parts = [
            f"문제: {rec.problem_text if rec else ''}",
            f"필요한 개념: {concepts}",
            f"튜터가 했던 질문: {tutor_question}",
            f"학생의 질문: {student_question}",
        ]
        if reference is not None:
            step = next((s for s in reference.steps if s.idx == target_step), None)
            if step is not None:
                parts.append(
                    f"지금 다루는 단계 (계산 결과는 말하지 말 것): {step.description}"
                )
        text = strip_leading_acknowledgement(
            self.llm.run_with_tools(
                purpose="explain",
                system=_EXPLAIN_SYSTEM,
                user="\n".join(parts),
                schema=PhrasedHint,
            ).hint.strip()
        )
        if reference is not None and leaks_answer(
            text, reference, target_step, visible_to_student(rec)
        ):
            log.warning("explanation leaked the answer; falling back")
            return (
                "좋은 질문이에요. 지금 단계에서 왜 그렇게 하는지 먼저 같이 생각해 볼까요? "
                + tutor_question
            )
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
        rec: Recognition | None = None,
        reference: ReferenceSolution | None = None,
        stronger: bool = False,
        student_answer: str | None = None,
    ) -> str:
        """Context = what the tutor may talk about: the problem itself, its
        tags, the CURRENT target step, and what the student just said. Never
        the answer, never a later step — those are filtered here and
        re-checked by the leak guard."""
        concepts = ", ".join(match.concepts) or "알 수 없음"
        parts = [f"힌트 레벨: L{decision.level} ({decision.action.value})"]
        if rec is not None:
            parts.append(f"문제: {rec.problem_text}")
            if rec.choices:
                parts.append(f"보기: {rec.choices}")
            if rec.problem_type and rec.problem_type != "unknown":
                parts.append(f"문제 유형: {rec.problem_type}")
            concepts = ", ".join(rec.concepts) or concepts
        parts.append(f"필요한 개념: {concepts}")
        if student_answer:
            parts.append(
                f"학생의 방금 답변: {student_answer}\n"
                "학생이 이미 올바르게 말한 내용은 다시 묻지 마세요. "
                "학생의 답변을 자연스럽게 이어받아 다음에 필요한 내용만 질문하세요."
            )

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
        board = tuple(b.strip() for b in result.board[:2] if b and b.strip())
        return result.hint.strip(), board

    def _concepts_for(self, match: MatchResult, rec: Recognition | None) -> list[str]:
        """Whitelisted concepts of the problem, tagger first, matcher second."""
        if rec is not None and rec.concepts:
            return rec.concepts
        return match.concepts

    def _generic_fallback(self, decision: Decision, slots: dict[str, str]) -> str:
        for template in self.db.hint_templates_for([], None, decision.level):
            if template.concept_id is None and template.misconception_id is None:
                try:
                    return template.template_text.format(**slots)
                except (KeyError, IndexError):
                    continue
        return "지금까지 한 풀이를 처음부터 소리 내어 설명해 볼까요?"
