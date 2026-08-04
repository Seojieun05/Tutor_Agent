"""Spoken-answer evaluation: did the student's reply answer the tutor's question?

This is the turn *after* a hint. It deliberately does NOT re-capture or
re-recognize the worksheet — the student answered out loud, so the only new
evidence is the transcript. One small no-tools LLM call replaces the whole
capture → VLM → match → estimate chain, which is what makes the answer turn
fast enough to feel like a conversation.

The verdict drives the existing policy through the store, with no new rules:

    CORRECT   → the pending hint worked  → target step advances → next step L1
    INCORRECT → the pending hint failed  → same step, escalate one level (L2)
    UNCLEAR   → no evidence either way   → same step, same level, re-ask
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from tutor.knowledge.models import ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.state.models import Status

log = logging.getLogger(__name__)

Verdict = Literal["CORRECT", "INCORRECT", "UNCLEAR"]
Intent = Literal["ANSWER", "QUESTION"]

_SYSTEM = """You grade a student's SPOKEN answer to a tutor's Socratic question.

You are given the problem, the reference solution steps, the question the tutor
just asked, and what the student said (speech-to-text, so expect informal
Korean, filler words and small transcription errors).

verdict:
- "CORRECT"   — the student answered the question correctly, or clearly shows
                they understand the targeted step. Accept informal or partial
                phrasing ("5를 빼면 돼요", "부호가 바뀌어요") when the idea is right.
                Do not demand exact numbers unless the question asked for them.
- "INCORRECT" — the answer is wrong, targets the wrong idea, or the student says
                they do not know / asks for more help ("모르겠어요", "힌트 더 주세요").
- "UNCLEAR"   — off-topic, unintelligible, or too vague to judge either way.
                Use this only when you genuinely cannot tell; prefer the others.

intent — decide this FIRST, it matters more than the verdict:
- "ANSWER"   — the student attempts an answer, says they do not know, or asks
               for more help / what to do next ("모르겠어요", "힌트 더 주세요").
- "QUESTION" — the student asks WHY something is done, or what a concept or
               method means ("왜 나눠요?", "이항이 뭐예요?", "왜 그렇게 해야 해요?").
               They are asking for an explanation, not attempting the step.
               Set intent "QUESTION" even if their words also contain a guess;
               a question that is not answered will be asked again.
For "QUESTION" the verdict is ignored — use "UNCLEAR" and leave feedback empty.

feedback: ONE short spoken Korean sentence reacting to the answer (친근한 반말체
금지, 존댓말). For CORRECT confirm briefly ("맞아요, 그렇게 하면 돼요!").
For INCORRECT do NOT correct it and do NOT give the next step — just acknowledge
warmly ("음, 조금 달라요."). For UNCLEAR say you did not catch it.
This is the ONLY reaction the student hears — what follows it is written
separately, so keep feedback to one clause and never continue into a hint.
NEVER state the final answer, a later step, or the result of the current step.

misconception: an id from the given list if the wrong answer matches one, else null.
status: the student's state after this answer, one of CORRECT, CALCULATION_ERROR,
CONCEPT_ERROR, PROCEDURAL_ERROR, MISREAD, STUCK.

Return ONLY the JSON object."""


class AnswerVerdict(BaseModel):
    # default ANSWER: a model that omits the field must not break the turn —
    # grading a question as an answer is recoverable, crashing is not
    intent: Intent = "ANSWER"
    verdict: Verdict
    feedback: str = ""
    misconception: str | None = None
    status: Status | None = None


class AnswerEvaluator:
    def __init__(self, llm: LLMClient, db=None):
        self.llm = llm
        self.db = db

    def evaluate(
        self,
        *,
        problem_text: str,
        reference: ReferenceSolution,
        question: str,
        target_step: int,
        transcript: str,
    ) -> AnswerVerdict:
        verdict = self.llm.complete_json(
            purpose="evaluate",
            system=_SYSTEM,
            user=self._context(problem_text, reference, question, target_step, transcript),
            schema=AnswerVerdict,
        )
        log.info(
            "answer intent=%s verdict=%s target_step=%d transcript=%r",
            verdict.intent,
            verdict.verdict,
            target_step,
            transcript[:60],
        )
        return verdict

    def _context(
        self,
        problem_text: str,
        reference: ReferenceSolution,
        question: str,
        target_step: int,
        transcript: str,
    ) -> str:
        steps = "\n".join(
            f"  {s.idx}. {s.description} → {s.expression}"
            for s in reference.steps
            if s.idx <= target_step  # never show the model steps beyond the target
        )
        parts = [
            f"문제: {problem_text}",
            f"기준 풀이 (지금 목표는 {target_step}단계):\n{steps or '  (없음)'}",
            f"튜터가 방금 한 질문: {question}",
            f"학생의 음성 답변: {transcript}",
        ]
        if self.db is not None:
            known = self.db.misconceptions_for(reference.concepts)
            if known:
                parts.append(
                    "알려진 오개념 목록: "
                    + ", ".join(f"{m.id}({m.description})" for m in known)
                )
        return "\n\n".join(parts)
