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
    PARTIAL   → right direction, unfinished step → same step, new L1 question
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

Verdict = Literal["CORRECT", "PARTIAL", "INCORRECT", "UNCLEAR"]

# What a student who just said "모르겠어요" hears — warmth, never "think more".
# A module constant so the TTS cache can pre-render it like the other fixed
# lines: this is the one reaction that should never keep anyone waiting.
SURRENDER_FEEDBACK = "괜찮아요, 어려울 수 있어요. 같이 짚어 볼게요."

# "잘 모르겠는데" — surrender, not an attempt. Module-level because the HINT
# GENERATOR needs the same judgement: an escalation after a surrender has no
# student error to point at, which changes which shelf the hint comes from.
_SURRENDER_WORDS_RE = re.compile(
    r"모르겠|모르는|힌트|도와|도움|어떻게 하는지|hint|help", re.I
)
_ATTEMPT_RE = re.compile(r"\d|[=+*/^√]|(?<![가-힣])[a-zA-Z](?![a-zA-Z])")


def is_surrender(text: str) -> bool:
    """They asked for help without attempting anything gradable."""
    stripped = (text or "").strip()
    return bool(
        stripped
        and _SURRENDER_WORDS_RE.search(stripped)
        and not _ATTEMPT_RE.search(stripped)
    )
# What an utterance that DIED mid-phrase hears. "응, 레이 와이 절편은" is not
# an answer to grade — the sentence never arrived — and grading the fragment
# confirmed work the student had not said. The re-asked question follows.
CUTOFF_FEEDBACK = "앗, 말이 중간에 끊긴 것 같아요."
Intent = Literal["ANSWER", "QUESTION", "WORK_CHECK"]

_SYSTEM = """You grade a student's SPOKEN answer to a tutor's Socratic question.

You are given the problem, the reference solution steps, the question the tutor
just asked, and what the student said (speech-to-text, so expect informal
Korean, filler words and small transcription errors).

verdict:
- "CORRECT"   — the student answered the question correctly AND completed what
                that question asks. Informal phrasing is fine. A method answer
                ("5를 빼면 돼요") completes a method question, but merely naming
                a method does NOT complete a step whose requested value or
                equation is still missing.
                A student who runs AHEAD is correct, not wrong: if the question
                asked for a slope and they give the whole tangent, they have
                done the step and more. Grade what they demonstrated, not
                whether it matches the question word for word.
                A step may store several labeled results at once
                ("l(0) = -4, m(0) = 10"): a student who says all the values
                ("-4하고 10이요") has completed it — do NOT hold the step
                open for not naming which label goes with which value.
- "PARTIAL"   — the student's direction, prerequisite, or intermediate result
                is mathematically right, but it does not yet complete the
                targeted step or answer the tutor's actual question. This is
                progress, not an error, and must stay on the same step.
                Example: target step is "f'(x)=2x-4, f'(1)=-2" and the student
                says "f를 미분하면 2x-4니까 거기에 1을 대입하면 될 것 같아요"
                without calculating the resulting slope → PARTIAL, not CORRECT.
                Example: target is the tangent-line equation and the student
                only explains how to find its slope → PARTIAL, not CORRECT.
                On the FINAL step this matters most: describing what to take
                as the base and height of a triangle answers the setup
                question, but until the area's VALUE is said the problem is
                not finished → PARTIAL, not CORRECT.
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

feedback: A short spoken Korean reaction to the answer (친근한 반말체 금지,
존댓말). For CORRECT confirm briefly, and when it is natural, name in a
word or two WHAT was right — the idea or value they nailed ("맞아요, 기울기를
정확히 봤어요!") — feedback that names what worked teaches more than a bare
맞아요; still one clause, still never the next step. For PARTIAL affirm the
direction without saying the step is finished ("좋아요, 제대로 접근하고 있어요.").
For INCORRECT, first say it is different and then briefly name WHAT in the
student's answer does not match the target. You may use an already-established
result, but do NOT supply the corrected value/equation or the next step. For
example, if a student gives the derivative where a tangent-line equation was
asked for: "조금 달라요. 방금 말한 식은 앞에서 구한 도함수예요." For UNCLEAR
say you did not catch it.
This is the ONLY reaction the student hears — what follows it is written
separately, so keep feedback to one clause and never continue into a hint.
NEVER state the final answer, a later step, or the result of the current step.

error_focus: only for an INCORRECT attempted answer, write one short Korean
phrase identifying the exact mismatch for the next tutor turn, without the
corrected answer (example: "2x-4를 접선의 방정식으로 혼동함; 이는 앞에서 구한
f'(x)임"). Leave it empty for CORRECT, PARTIAL, UNCLEAR, questions, work checks,
and a student who simply says they do not know.

reached_step / reached_claim: only with CORRECT, when the student answered BEYOND the step
they were asked about. `reached_step` is the highest reference step their
answer demonstrably completes, and `reached_claim` is the expression they said
that proves it, written in ASCII ("y = -2*x - 4"). The claim is checked
mechanically against that step before anything moves, so an unprovable jump
costs nothing — but a claim that does not match the step you named will simply
be ignored. Future step NAMES are supplied specifically so you can recognize a
student who ran ahead; their expressions remain hidden. For example, if asked
for the slope but the student says the whole tangent equation and a later step
is named "접선 l의 방정식 쓰기", set reached_step to that step and transcribe the
spoken equation into reached_claim. Leave both out only when the answer did not
go beyond the step that was asked.

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
    # A private hand-off to the hint generator: what the attempted answer got
    # wrong.  It is not itself spoken, and it must never contain the corrected
    # current-step result.  Keeping it separate from a broad misconception id
    # lets a one-off mix-up (derivative versus tangent equation) be addressed
    # just as specifically as an error diagnosed from the photographed work.
    error_focus: str = ""
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
        cut = self._cut_off(transcript)
        if cut is not None:
            return cut
        surrendered = self._gave_up(transcript)
        if surrendered is not None:
            return surrendered
        graded_transcript = self.normalize_transcript(
            reference, target_step, transcript
        )
        if graded_transcript != transcript:
            log.info(
                "contextual transcript normalization: %r -> %r",
                transcript,
                graded_transcript,
            )
        context = self._context(
            problem_text, reference, question, target_step, graded_transcript
        )
        verdict = self._judge(self.llm, context)
        verdict = self._downgrade_unfinished_plan(
            verdict, reference, target_step, graded_transcript
        )
        verdict = self._normalize_partial_feedback(
            verdict, reference, target_step
        )
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
                better = self._downgrade_unfinished_plan(
                    better, reference, target_step, graded_transcript
                )
                better = self._normalize_partial_feedback(
                    better, reference, target_step
                )
            except Exception:  # noqa: BLE001 — the first verdict still stands
                log.exception("second opinion failed; keeping the first verdict")
                return verdict
            if better.verdict in {"CORRECT", "PARTIAL"}:
                log.info(
                    "the second opinion overturns it: the answer was %s",
                    "complete" if better.verdict == "CORRECT" else "partially right",
                )
                return better
        return verdict

    @staticmethod
    def normalize_transcript(
        reference: ReferenceSolution, target_step: int, transcript: str
    ) -> str:
        """Repair narrow STT errors only when the target step disambiguates.

        Korean ``마이너스 이`` has arrived live as ``Minus e.``.  In isolation
        that text could mean the mathematical constant e, so it is never
        rewritten globally.  It means -2 only when the current verified step
        itself ends in -2 and the whole utterance is this short answer shape.
        The ordinary English spellings are accepted by the same narrow gate.

        Likewise, a spoken ``x는 7`` can arrive as ``y는 7``.  It is repaired
        only for a step explicitly asking for an x-coordinate, and only for a
        short variable-plus-number answer. A wrong number stays wrong.
        """
        from fractions import Fraction

        step = next((s for s in reference.steps if s.idx == target_step), None)
        if step is None:
            return transcript
        normalized = transcript
        if "x좌표" in (step.description or ""):
            short_coordinate = re.sub(
                r"[.,!?]+$", "", (transcript or "").strip().lower()
            )
            mistaken_axis = re.fullmatch(
                r"(?:y|와이)\s*(?:는|은|가|=)?\s*"
                r"(-?\d+(?:\.\d+)?)\s*(?:이에요|예요|이요|요)?",
                short_coordinate,
            )
            if mistaken_axis:
                normalized = f"x는 {mistaken_axis.group(1)}"

        tail = (step.expression or "").split("=")[-1].strip()
        try:
            target = Fraction(tail.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            return normalized
        if target != -2:
            return normalized

        short = re.sub(r"[.,!?]+$", "", normalized.strip().lower())
        if re.fullmatch(r"minus\s+(?:2|two|to|too|e|ee)", short):
            return "-2"
        return normalized

    _UNFINISHED_PLAN_RE = re.compile(
        r"(?:대입|계산|정리|구하|쓰).{0,40}(?:하면|해서|해)\s*"
        r"(?:될\s*것\s*같|되겠|돼요|볼(?:게|까|래)).*?[?.!~ ]*$"
    )

    @staticmethod
    def _final_piece_spoken(expression: str, transcript: str) -> bool | None:
        """Did the transcript say the composite step's final numeric piece?

        "f'(x) = 2*x - 4, f'(1) = -2" ends on -2; a transcript that computed
        only 2x-4 does not contain it. Only numeric tails are judged — a
        symbolic tail has no fixed spoken shape and answers None — and a
        number that merely appears anywhere counts as said.
        """
        from fractions import Fraction

        tail = expression.split("=")[-1].strip()
        try:
            target = Fraction(tail.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            return None
        # STT writes what it HEARS: the sign arrives as 마이너스 and the
        # number itself may never be a digit at all. Live, a correct "엑스는
        # 칠에서 만날 것 같은데" was graded PARTIAL because the scan below
        # found no 7 anywhere in it.
        from tutor.speech.mathspeak import with_digits

        spoken = re.sub(r"마이너스\s*", "-", with_digits(transcript or ""))
        for said in re.findall(
            r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?", spoken
        ):
            try:
                if Fraction(said.replace(" ", "")) == target:
                    return True
            except (ValueError, ZeroDivisionError):
                continue
        return False

    @staticmethod
    def _normalize_partial_feedback(
        verdict: AnswerVerdict,
        reference: ReferenceSolution,
        target_step: int,
    ) -> AnswerVerdict:
        """A PARTIAL reaction acknowledges completed work; it gives no advice."""
        if verdict.verdict != "PARTIAL":
            return verdict
        step = next((s for s in reference.steps if s.idx == target_step), None)
        expression = step.expression if step is not None else ""
        derivative = re.search(r"\b([A-Za-z])\s*['′]\s*\(\s*x\s*\)\s*=", expression)
        feedback = (
            f"맞아요, {derivative.group(1)}'(x)는 잘 구했어요."
            if derivative is not None
            else "맞아요, 여기까지는 잘했어요."
        )
        return verdict.model_copy(update={"feedback": feedback})

    @classmethod
    def _downgrade_unfinished_plan(
        cls,
        verdict: AnswerVerdict,
        reference: ReferenceSolution,
        target_step: int,
        transcript: str,
    ) -> AnswerVerdict:
        """A future-tense plan cannot complete a multi-result reference step.

        The LLM prompt carries the general distinction.  This narrow rule pins
        the live failure deterministically: saying the derivative and planning
        to substitute 1 is useful progress, but it is not yet the resulting
        slope, much less the tangent equation after it.
        """
        if verdict.verdict != "CORRECT":
            return verdict
        step = next((s for s in reference.steps if s.idx == target_step), None)
        expression = step.expression if step is not None else ""
        composite = "," in expression or expression.count("=") >= 2
        if not composite:
            return verdict
        # Two ways to be half-done with a composite step: PLAN the second
        # half out loud ("대입하면 될 것 같아요"), or simply stop after the
        # first half's result. Live: "f′부터 계산하면 좋을 것 같아. 계산하면
        # 2x-4." wore no plan tail, was graded CORRECT, and the step-2 line
        # then congratulated a slope nobody had computed. When the step's
        # FINAL piece is a plain number (-2), its absence from the transcript
        # is checkable — and absent means not done.
        #
        # A SAID final result outranks how the sentence was phrased. Live on
        # step 7 ("l(0) = -4, m(0) = 10") both intercepts arrived inside a
        # plan-shaped sentence, this gate demoted the second opinion's
        # CORRECT, and the tutor then asked which line owns which intercept —
        # a distinction no later step needs spoken.
        spoken = cls._final_piece_spoken(expression, transcript)
        if spoken:
            return verdict
        planned = cls._UNFINISHED_PLAN_RE.search(transcript.strip())
        if spoken is None and not planned:
            return verdict
        log.info("correct direction but unfinished composite step; grading PARTIAL")
        return verdict.model_copy(
            update={
                "verdict": "PARTIAL",
                "feedback": "좋아요, 제대로 접근하고 있어요.",
                "reached_step": None,
                "reached_claim": "",
            }
        )

    # The particles a Korean noun phrase hangs on mid-sentence. An utterance
    # whose LAST word is hangul ending on one of these never arrived at its
    # predicate — the VAD closed on a pause, not on a sentence — so there is
    # nothing to grade. Digits and variables are excluded first ("x는 2" ends
    # on the value, the answer shape), and the AMBIGUOUS particles are left
    # out on purpose: 이/가/과/로/고 are also how ordinary nouns end (넓이,
    # 차이, 결과, 그리고), and eating a valid answer costs more than missing
    # a fragment the UNCLEAR path would shrug at anyway.
    _DANGLING_JOSA = ("은", "는", "을", "를", "의", "와", "에", "에서", "부터", "까지")

    def _cut_off(self, transcript: str) -> AnswerVerdict | None:
        text = transcript.strip().rstrip(" ?.!…~,")
        if not text:
            return None
        last = text.split()[-1]
        if not re.fullmatch(r"[가-힣]{2,}", last):
            return None                    # a value tail, latin, or too short to call
        if not last.endswith(self._DANGLING_JOSA):
            return None
        log.info("utterance died mid-phrase: %r — re-asking, not grading", text[-24:])
        return AnswerVerdict(
            intent="ANSWER",
            verdict="UNCLEAR",             # same step, same level, re-asked
            feedback=CUTOFF_FEEDBACK,
        )

    # "잘 모르겠는데", "모르겠어요", "힌트 주세요" — surrender, not an attempt.
    # Nothing here needs a model: the verdict is INCORRECT by construction
    # (the pending hint did not land → same step, one level up, exactly the
    # policy that already exists) and no attempt means nothing to grade wrong,
    # so the LLM judge AND the second opinion are skipped. Measured live, the
    # pair cost 2.8s to reach this same conclusion — and then reacted with
    # "조금 더 생각해 볼까요?", which tells a student who just said they cannot
    # think of anything to think harder, immediately before helping anyway.
    # an attempt that CONTAINS doubt still gets graded: "5인 것 같은데
    # 모르겠어요" carries a value, and grading it is kinder than ignoring it —
    # both halves of that judgement live in module-level is_surrender now,
    # shared with the hint generator.
    _SURRENDER_RE = _SURRENDER_WORDS_RE
    _CARRIES_AN_ATTEMPT = _ATTEMPT_RE

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
        # The prompt asks for "at most one or two" tool calls; the budget now
        # says the same thing, because a small routed model does not hold that
        # line on its own. Live, grading "엑스는 칠에서 만날 것 같은데" spent
        # six rounds looking up check_equivalence — which had answered on the
        # first — and 4.6s of a student's silence. Running out is not a
        # failure: the seam asks once more with the tools withdrawn, which is
        # exactly the "then decide" the prompt already demands.
        return llm.run_with_tools(
            purpose="evaluate", system=_SYSTEM, user=context,
            schema=AnswerVerdict, max_rounds=2,
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
            if s.idx <= target_step  # future expressions stay hidden
        )
        future_steps = "\n".join(
            f"  {s.idx}. {s.description}"
            for s in reference.steps
            if s.idx > target_step
        )
        parts = [
            f"문제: {problem_text}",
            f"기준 풀이 (지금 목표는 {target_step}단계):\n{steps or '  (없음)'}",
            f"튜터가 방금 한 질문: {question}",
            f"학생의 음성 답변: {transcript}",
        ]
        if future_steps:
            parts.append(
                "이후 단계 이름 (학생이 질문보다 앞서 답했는지 판별할 때만 사용; "
                "피드백이나 힌트에서 절대 언급하지 말 것):\n" + future_steps
            )
        if self.db is not None:
            known = self.db.misconceptions_for(reference.concepts)
            if known:
                parts.append(
                    "알려진 오개념 목록: "
                    + ", ".join(f"{m.id}({m.description})" for m in known)
                )
        return "\n\n".join(parts)
