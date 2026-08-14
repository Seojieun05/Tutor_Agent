"""Spoken-answer evaluation: did the student's reply answer the tutor's question?

This is the turn *after* a hint. It deliberately does NOT re-capture or
re-recognize the worksheet — the student answered out loud, so the only new
evidence is the transcript. One small LLM call replaces the whole capture →
VLM → match → estimate chain, which is what makes the answer turn fast enough
to feel like a conversation. Its only tools are the sympy checks (compute /
check_equivalence), and the prompt gates them to the answers whose grade
actually turns on arithmetic — no KB round trips, ever.

The verdict drives the existing policy through the store, with no new rules:

    CORRECT   → the pending hint worked  → target step advances → next step L1
    INCORRECT → the pending hint failed  → same step, escalate one level (L2)
    UNCLEAR   → no evidence either way   → same step, same level, re-ask
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from tutor.knowledge.models import ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.state.models import Status

log = logging.getLogger(__name__)

Verdict = Literal["CORRECT", "INCORRECT", "UNCLEAR"]

# What a student who just said "모르겠어요" hears — warmth, never "think more".
# A module constant so the TTS cache can pre-render it like the other fixed
# lines: this is the one reaction that should never keep anyone waiting.
SURRENDER_FEEDBACK = "괜찮아요, 어려울 수 있어요. 같이 짚어 볼게요."
Intent = Literal["ANSWER", "QUESTION", "WORK_CHECK"]

_SYSTEM = """You grade a student's SPOKEN answer to a tutor's Socratic question.

You are given the problem, the reference solution steps, the question the tutor
just asked, and what the student said (speech-to-text, so expect informal
Korean, filler words and small transcription errors).

verdict:
- "CORRECT"   — the student answered the question correctly, or clearly shows
                they understand the targeted step. Accept informal or partial
                phrasing ("5를 빼면 돼요", "부호가 바뀌어요") when the idea is right.
                Do not demand exact numbers unless the question asked for them.
                A student who runs AHEAD is correct, not wrong: if the question
                asked for a slope and they give the whole tangent, they have
                done the step and more. Grade what they demonstrated, not
                whether it matches the question word for word.
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
- "WORK_CHECK" — the student asks you to LOOK at what they have written on the
               page and react to it ("이렇게 하는 거 맞아요?", "내가 쓴 거 봐 주세요",
               "여기서 뭐가 틀렸어요?", "제 풀이 좀 봐 주세요"). They are pointing at
               the worksheet, not answering. The tutor will take a fresh photo,
               so do not guess what they wrote.
For "QUESTION" and "WORK_CHECK" the verdict is ignored — use "UNCLEAR" and
leave feedback empty.

feedback: ONE short spoken Korean sentence reacting to the answer (친근한 반말체
금지, 존댓말). For CORRECT confirm briefly ("맞아요, 그렇게 하면 돼요!").
For INCORRECT do NOT correct it and do NOT give the next step — just acknowledge
warmly ("음, 조금 달라요."). For UNCLEAR say you did not catch it.
This is the ONLY reaction the student hears — what follows it is written
separately, so keep feedback to one clause and never continue into a hint.
NEVER state the final answer, a later step, or the result of the current step.

reached_step / reached_claim: only when the student answered BEYOND the step
they were asked about. `reached_step` is the highest reference step their
answer demonstrably completes, and `reached_claim` is the expression they said
that proves it, written in ASCII ("y = -2*x - 4"). The claim is checked
mechanically against that step before anything moves, so an unprovable jump
costs nothing — but a claim that does not match the step you named will simply
be ignored. Leave both out when the answer is to the step that was asked.

misconception: an id from the given list if the wrong answer matches one, else null.
status: the student's state after this answer, one of CORRECT, CALCULATION_ERROR,
CONCEPT_ERROR, PROCEDURAL_ERROR, MISREAD, STUCK.

tools (when available): `compute` and `check_equivalence` do real math. Use one
ONLY when the grade genuinely turns on arithmetic or equivalence you cannot see
at a glance — the student claims a number or a rearranged equation and you are
not certain it matches the reference step. Conceptual answers ("5를 빼면 돼요")
need no tool; obviously right or wrong answers need no tool. Every call is a
beat of silence the student sits through, so at most one or two, then decide.

Return ONLY the JSON object."""


class AnswerVerdict(BaseModel):
    # default ANSWER: a model that omits the field must not break the turn —
    # grading a question as an answer is recoverable, crashing is not
    intent: Intent = "ANSWER"
    verdict: Verdict
    feedback: str = ""
    misconception: str | None = None
    status: Status | None = None
    # A PROPOSAL, not a verdict: how far the student ran ahead, and the
    # expression that proves it. The orchestrator checks the claim against
    # that reference step with sympy before letting either of them move
    # anything — see Session._reached_step.
    reached_step: int | None = None
    reached_claim: str = ""


class AnswerEvaluator:
    def __init__(self, llm: LLMClient, db=None, second_opinion: LLMClient | None = None):
        self.llm = llm
        self.db = db
        # Consulted ONLY before telling a student they are wrong. Measured on
        # a student who answered a slope question with the whole tangent —
        # correct, and a step further than asked — the small routed model said
        # INCORRECT in 2.0s while flash and grok both said CORRECT. Grading a
        # right answer wrong is the most expensive mistake this system makes,
        # and it is the one judgement worth paying twice for.
        self.second_opinion = second_opinion

    def evaluate(
        self,
        *,
        problem_text: str,
        reference: ReferenceSolution,
        question: str,
        target_step: int,
        transcript: str,
    ) -> AnswerVerdict:
        surrendered = self._gave_up(transcript)
        if surrendered is not None:
            return surrendered
        context = self._context(problem_text, reference, question, target_step, transcript)
        verdict = self._judge(self.llm, context)
        log.info(
            "answer intent=%s verdict=%s target_step=%d transcript=%r",
            verdict.intent,
            verdict.verdict,
            target_step,
            transcript[:60],
        )
        if (
            verdict.verdict == "INCORRECT"
            and verdict.intent == "ANSWER"
            and self.second_opinion is not None
        ):
            # A correct answer stays fast: only "wrong" pays for the second
            # look, and a student who is actually wrong is the one who can
            # afford the wait.
            log.info("second opinion before telling the student they are wrong")
            try:
                better = self._judge(self.second_opinion, context)
            except Exception:  # noqa: BLE001 — the first verdict still stands
                log.exception("second opinion failed; keeping the first verdict")
                return verdict
            if better.verdict == "CORRECT":
                log.info("the second opinion overturns it: the answer was right")
                return better
        return verdict

    # "잘 모르겠는데", "모르겠어요", "힌트 주세요" — surrender, not an attempt.
    # Nothing here needs a model: the verdict is INCORRECT by construction
    # (the pending hint did not land → same step, one level up, exactly the
    # policy that already exists) and no attempt means nothing to grade wrong,
    # so the LLM judge AND the second opinion are skipped. Measured live, the
    # pair cost 2.8s to reach this same conclusion — and then reacted with
    # "조금 더 생각해 볼까요?", which tells a student who just said they cannot
    # think of anything to think harder, immediately before helping anyway.
    _SURRENDER_RE = re.compile(r"모르겠|모르는|힌트|도와|도움|어떻게 하는지|hint|help", re.I)
    # an attempt that CONTAINS doubt still gets graded: "5인 것 같은데
    # 모르겠어요" carries a value, and grading it is kinder than ignoring it
    _CARRIES_AN_ATTEMPT = re.compile(r"\d|[=+*/^√]|(?<![가-힣])[a-zA-Z](?![a-zA-Z])")

    def _gave_up(self, transcript: str) -> AnswerVerdict | None:
        text = transcript.strip()
        if not text or not self._SURRENDER_RE.search(text):
            return None
        if self._CARRIES_AN_ATTEMPT.search(text):
            return None
        log.info("surrender, not an attempt: %r — escalating without a judge", text[:40])
        return AnswerVerdict(
            intent="ANSWER",
            verdict="INCORRECT",
            feedback=SURRENDER_FEEDBACK,
        )

    def _judge(self, llm: LLMClient, context: str) -> AnswerVerdict:
        return llm.run_with_tools(
            purpose="evaluate", system=_SYSTEM, user=context, schema=AnswerVerdict
        )

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
