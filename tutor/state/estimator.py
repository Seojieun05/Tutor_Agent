"""Student State Estimator: rule pre-checks + LLM diff against the reference.

The estimator only RETURNS a state — the orchestrator owns all store writes
and resolves the pending hint's effectiveness via hint_was_effective.
"""

from __future__ import annotations

import json
import logging
import re

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.state.models import STATUS_RANK, StudentState
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

# [프롬프트] 학생 진단용 시스템 프롬프트. 손글씨 풀이를 기준 풀이와 대조해
# last_correct_step(어디까지 맞았는지) · status · misconception · current_step을 뽑게 한다.
# 핵심 지침: 기준 풀이는 하나의 정답 경로일 뿐이며, 확신이 없으면 UNCERTAIN을 쓰라는 것.
_SYSTEM = """You diagnose where a student is stuck by comparing their handwritten work
to a reference solution. Be conservative: use "UNCERTAIN" when unsure, never guess.

Rules:
- The reference is ONE valid path, not the only one. Work that is mathematically
  sound and consistent with the problem is correct progress even when no
  reference step resembles it — judge the mathematics on the page, not the
  resemblance to the reference.
- A previous diagnosis is context, not evidence. When the written work has
  changed, re-derive the diagnosis from what is on the page NOW; repeating an
  earlier misconception requires the CURRENT work to still show it.
- Treat the student's latest spoken response as valid evidence of understanding,
  even when the handwritten work is empty or unchanged.
- last_correct_step: the largest N such that reference steps 1..N are ALL accounted
  for — written on the page, said aloud, or made unnecessary by the student's own
  valid route. A route can replace a step's DERIVATION, never its RESULT: when a
  step's result (a value or equation such as "r**3 = 2") appears nowhere on the
  page or in the quoted speech, that step is NOT accounted for — even if the line
  that would produce it is one operation away. A later step done while an earlier
  one is still missing does NOT raise it: report the unbroken prefix, and name the
  missing step in current_step. (0 = none.)
- status: CORRECT | CALCULATION_ERROR | CONCEPT_ERROR | PROCEDURAL_ERROR | MISREAD | STUCK | UNCERTAIN.
- misconception: prefer an id from the provided misconception list when the student's
  error matches its indicators; otherwise a short free-text description or null.
- current_step: short Korean description of what the student is currently doing.

Return ONLY the JSON object."""


# 힌트가 통했는지 판정: 단계가 나아갔거나, 오개념이 풀렸거나, 상태 등급이 올랐으면 효과 있음.
def hint_was_effective(prev: StudentState, new: StudentState) -> bool:
    """Effective = step progress OR misconception resolved OR status improved."""
    if new.last_correct_step < prev.last_correct_step:
        # A nicer status word cannot make lost progress evidence of success.
        return False
    progress = new.last_correct_step > prev.last_correct_step
    resolved = prev.misconception is not None and new.misconception != prev.misconception
    improved = STATUS_RANK.get(new.status, 0) > STATUS_RANK.get(prev.status, 0)
    return progress or resolved or improved


# 학생 상태 추정기. 상태를 만들어 돌려줄 뿐이고, 저장은 세션(오케스트레이터)이 한다.
class StudentStateEstimator:
    # 진단 모델, 지식 DB, 사진 신뢰도 임계값을 받는다.
    def __init__(self, llm: LLMClient, db: KnowledgeDB, conf_threshold: float = 0.6):
        self.llm = llm
        self.db = db
        self.conf_threshold = conf_threshold

    # 진단 본체: 결정론적 사전 검사 → 기호 계산 빠른 경로 → LLM 진단 → 결정론적 사후 규칙.
    def estimate(
        self,
        *,
        rec: Recognition,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
        transcript: str | None = None,
    ) -> StudentState:
        pre = self._pre_check(
            rec,
            prev_state,
            prev_work,
            history,
            transcript,
        )
        if pre is not None:
            return pre

        quick = self._rule_based_progress(rec, reference, prev_state)
        if quick is not None:
            return self._post_rules(quick, reference, prev_state, rec, transcript)

        # complete_json, deliberately not run_with_tools: everything the
        # diagnosis may consult — the reference, the work, the misconception
        # list — is already IN the context, and live the model still spent a
        # tool round on search_domain_kb (measured: 17.6s for one estimate,
        # two rounds). A diagnosis has nothing to look up.
        state = self.llm.complete_json(
            purpose="estimate",
            system=_SYSTEM,
            user=self._build_context(rec, reference, prev_state, history, transcript),
            schema=StudentState,
        )
        if (
            state.status == "UNCERTAIN"
            and prev_state is not None
            and rec.student_work
            and rec.confidence >= self.conf_threshold
        ):
            # A model shown a previous UNCERTAIN tends to echo it — live, one
            # blurry photo early in a problem turned every later work check
            # into UNCERTAIN → probe, forever. The page is legible and there
            # IS work on it: look again with fresh eyes (no previous state, no
            # history) before giving up on a verdict.
            log.info("estimate UNCERTAIN on a legible page; retrying without bias")
            state = self.llm.complete_json(
                purpose="estimate",
                system=_SYSTEM,
                user=self._build_context(rec, reference, None, [], transcript),
                schema=StudentState,
            )
        if (
            state.status != "CORRECT"
            and prev_state is not None
            and prev_state.misconception
            and state.misconception == prev_state.misconception
            and prev_work is not None
            and rec.student_work != prev_work
        ):
            # The page CHANGED but the diagnosis repeats the old misconception
            # word for word — the other parrot (live: work corrected from r to
            # r**3, still "diagnosed" with the r-relation misconception). Same
            # medicine as the UNCERTAIN echo: one look with fresh eyes.
            log.info("old misconception repeated on changed work; retrying without bias")
            state = self.llm.complete_json(
                purpose="estimate",
                system=_SYSTEM,
                user=self._build_context(rec, reference, None, [], transcript),
                schema=StudentState,
            )
        return self._post_rules(state, reference, prev_state, rec, transcript)

    # 사진 인식과 진단을 한 번의 호출로 합칠 때 넘기는 맥락(기준 풀이·오개념 목록·직전 상태·힌트 이력).
    def vision_context(
        self,
        *,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
        transcript: str | None = None,
    ) -> str:
        """Context that lets worksheet reading and diagnosis share one call.

        This contains only information already cached for the active problem.
        The recognizer is told explicitly that none of it is page content.
        """
        misconceptions = [
            m.model_dump() for m in self.db.misconceptions_for(reference.concepts)
        ]
        parts = [
            "기준 풀이 단계:\n"
            + "\n".join(
                f"  {s.idx}. {s.description} → {s.expression}"
                for s in reference.steps
            ),
            "알려진 오개념 목록: "
            + json.dumps(misconceptions, ensure_ascii=False),
        ]
        if prev_state is not None:
            parts.append("직전 상태: " + prev_state.model_dump_json())
        if prev_work is not None:
            parts.append(
                "직전 사진의 학생 풀이 (현재 사진과 비교할 때만 사용): "
                + json.dumps(prev_work, ensure_ascii=False)
            )
        if history:
            parts.append(
                "지금까지 준 힌트: "
                + json.dumps(
                    [
                        {
                            "step": h.step,
                            "level": h.level,
                            "action": h.action,
                            "effective": h.effective,
                        }
                        for h in history
                    ],
                    ensure_ascii=False,
                )
            )
        if transcript:
            parts.append(f"학생이 직전에 말한 내용: {transcript}")
        return "\n\n".join(parts)

    # 인식 호출에 딸려 온 진단을 받아들일지 검사. 미심쩍으면 None을 돌려 별도 진단을 다시 돌린다.
    def accept_vision_estimate(
        self,
        state: StudentState | None,
        *,
        rec: Recognition,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
        transcript: str | None = None,
    ) -> StudentState | None:
        """Validate a diagnosis returned with worksheet recognition.

        Deterministic checks still outrank the model. Ambiguous diagnoses that
        the ordinary estimator would retry are declined here, causing the
        session to fall back to the existing dedicated estimate call.
        """
        pre = self._pre_check(rec, prev_state, prev_work, history, transcript)
        if pre is not None:
            return pre
        quick = self._rule_based_progress(rec, reference, prev_state)
        if quick is not None:
            return self._post_rules(quick, reference, prev_state, rec, transcript)
        if state is None:
            return None
        if (
            state.status == "UNCERTAIN"
            and prev_state is not None
            and rec.student_work
            and rec.confidence >= self.conf_threshold
        ):
            log.info(
                "inline estimate was UNCERTAIN on a legible page; using dedicated estimate"
            )
            return None
        if (
            state.status != "CORRECT"
            and prev_state is not None
            and prev_state.misconception
            and state.misconception == prev_state.misconception
            and prev_work is not None
            and rec.student_work != prev_work
        ):
            log.info(
                "inline estimate repeated an old misconception on changed work; "
                "using dedicated estimate"
            )
            return None
        return self._post_rules(state, reference, prev_state, rec, transcript)

    # --- deterministic pre-checks: no LLM call -------------------------------

    # 기준 풀이 없이도 판단되는 것들만 보는 사전 검사(사진이 안 읽힘, 빈 종이 등).
    def precheck(
        self,
        *,
        rec: Recognition,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
        transcript: str | None = None,
    ) -> StudentState | None:
        """The deterministic half of estimate(): everything decidable WITHOUT
        the reference solution — an unreadable photo, an empty page. The
        background-solve path runs this while the reference is still being
        written, so a garbled frame still ends in "다시 보여 줄래요?" rather
        than in a confident hint about garbage."""
        return self._pre_check(rec, prev_state, prev_work, history, transcript)

    # 사전 검사 본체. 여기서 상태가 정해지면 LLM은 아예 부르지 않는다.
    def _pre_check(
        self,
        rec: Recognition,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
        transcript: str | None = None,
    ) -> StudentState | None:
        if rec.confidence < self.conf_threshold or self._last_line_illegible(rec):
            base = prev_state or StudentState()
            return base.model_copy(update={"status": "UNCERTAIN"})
        if not rec.student_work and not transcript:
            attempts = prev_state.attempt_count + 1 if prev_state else 1
            return StudentState(
                current_step="아직 풀이를 시작하지 않음",
                last_correct_step=0,
                status="STUCK",
                # erased/absent work is not evidence a misconception was resolved
                misconception=prev_state.misconception if prev_state else None,
                attempt_count=attempts,
                previous_hint_effective=False if self._hint_pending(history) else None,
            )
        if (
            prev_state is not None
            and prev_work is not None
            and rec.student_work == prev_work
            and self._hint_pending(history)
            and not transcript
        ):
            # Nothing changed since the last hint: it did not help. No LLM needed.
            return prev_state.model_copy(
                update={
                    "attempt_count": prev_state.attempt_count + 1,
                    "previous_hint_effective": False,
                }
            )
        return None

    # 기호 계산 빠른 경로: 마지막 줄이 기준 단계와 같은 꼴이면 LLM 없이 진도를 확정한다.
    def _rule_based_progress(
        self,
        rec: Recognition,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
    ) -> StudentState | None:
        """Symbolic fast path: when the newest work line IS a reference step
        (or just restates the problem), progress is decidable without an LLM.
        Form comparison, not equivalence — '3*x + 5 = 20', '3*x = 15' and
        'x = 5' are one equation but DIFFERENT pedagogical steps. Returns None
        when actual diagnosis is needed."""
        if not rec.student_work:
            return None

        # 두 식이 같은 꼴인지(문제의 정의로 도함수를 대입한 경우까지 포함).
        def same(a: str, b: str) -> bool:
            # A reference may leave f'(x) symbolic while the student has
            # already inserted the derivative obtained from the problem's
            # explicit f(x) definition.  That is the same written step once
            # the independently verified substitution is applied.
            return mathnorm.equations_same_form_with_derivatives(
                a, b, rec.equations
            )

        matched = {
            step.idx
            for step in reference.steps
            if any(same(line, step.expression) for line in rec.student_work)
        }
        # A later line vouches for the earlier steps it PASSED THROUGH: a page
        # showing only "x = 5" did step 1 ("3*x = 15") in its head, and the
        # two are one equation — same solution set, different form. It vouches
        # for nothing on a different thread: g'(1) = -4 says nothing about
        # l의 방정식, however many steps further along it sits.
        implied = {
            earlier.idx
            for earlier in reference.steps
            for done in reference.steps
            if done.idx in matched and done.idx > earlier.idx
            and mathnorm.equations_equivalent(done.expression, earlier.expression)
        }
        # A fresh capture often contains only the newest line.  Correct work
        # already proven on this same problem remains part of the prefix even
        # when the camera crop no longer shows it.
        proven = set()
        if prev_state is not None:
            proven = set(range(1, min(prev_state.last_correct_step, len(reference.steps)) + 1))
        covered = matched | implied | proven
        # The PREFIX, not the peak: steps 1..N all accounted for. A page
        # showing steps 1, 3 and 4 of independent sub-results is not "step 4
        # done" — the hole at 2 is the very thing the tutor exists to notice
        # (live: l의 기울기에서 곧장 m의 기울기로, l의 방정식은 한 번도 쓰이지
        # 않은 채 확인이 지나갔다). With a hole, the newest line no longer
        # equals the prefix, so this fast path declines and the full diagnosis
        # decides whether the skip was fine or the thing to point at.
        lcs = 0
        while lcs + 1 in covered:
            lcs += 1
        last = rec.student_work[-1]
        last_step = next(
            (s.idx for s in reference.steps if same(last, s.expression)), None
        )
        if last_step is not None and last_step == lcs and lcs >= 1:
            return StudentState(
                current_step=f"기준 풀이 {lcs}단계까지 완료",
                last_correct_step=lcs,
                status="CORRECT",
                misconception=None,
            )
        if last_step is None and any(same(last, eq) for eq in rec.equations):
            # newest line just restates the problem: no progress, nothing wrong
            return StudentState(
                current_step="문제를 옮겨 적음",
                last_correct_step=lcs,
                status="STUCK",
                misconception=prev_state.misconception if prev_state else None,
            )
        return None

    # 아직 답을 못 받은 힌트가 남아 있는지.
    @staticmethod
    def _hint_pending(history: list[HintRecord]) -> bool:
        return any(h.level >= 1 and h.effective is None for h in history)

    # 마지막 풀이 줄이 흐릿해서 못 읽은 영역에 걸쳐 있는지.
    @staticmethod
    def _last_line_illegible(rec: Recognition) -> bool:
        if not rec.uncertain_regions or not rec.student_work:
            return False
        last = rec.student_work[-1]
        return any(last and last in region for region in rec.uncertain_regions)

    # --- LLM context (state/history prefetched by the orchestrator) ----------

    # 진단 모델에 넘길 사용자 메시지 조립(문제·학생 풀이·기준 풀이·오개념 목록·직전 상태·발화).
    def _build_context(
        self,
        rec: Recognition,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        history: list[HintRecord],
        transcript: str | None,
    ) -> str:
        misconceptions = [
            m.model_dump() for m in self.db.misconceptions_for(reference.concepts)
        ]
        parts = [
            f"문제: {rec.problem_text}",
            "기준 풀이 단계:\n"
            + "\n".join(
                f"  {s.idx}. {s.description} → {s.expression}" for s in reference.steps
            ),
            "학생이 쓴 풀이 (순서대로):\n"
            + "\n".join(f"  {i + 1}. {line}" for i, line in enumerate(rec.student_work)),
            "알려진 오개념 목록: " + json.dumps(misconceptions, ensure_ascii=False),
        ]
        if prev_state is not None:
            parts.append("직전 상태: " + prev_state.model_dump_json())
        if history:
            parts.append(
                "지금까지 준 힌트: "
                + json.dumps(
                    [
                        {
                            "step": h.step,
                            "level": h.level,
                            "action": h.action,
                            "effective": h.effective,
                        }
                        for h in history
                    ],
                    ensure_ascii=False,
                )
            )
        if transcript:
            parts.append(f"학생이 방금 말한 내용: {transcript}")
        return "\n\n".join(parts)

    # --- deterministic post-rules --------------------------------------------

    # 사후 규칙: 모델 판정 위에 결정론적 검사(경계 결과 확인, 산술 검사)를 덧씌운다.
    def _post_rules(
        self,
        state: StudentState,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        rec: Recognition | None = None,
        transcript: str | None = None,
    ) -> StudentState:
        state = self._arithmetic_check(state, reference, rec)
        max_step = len(reference.steps)
        clamped = min(max(state.last_correct_step, 0), max_step)
        if (
            prev_state is not None
            and state.status == "CORRECT"
            and clamped < prev_state.last_correct_step
        ):
            # CORRECT means "everything shown is sound", not "previously
            # proven work vanished".  A camera crop or VLM that returns only
            # the newest line must not send a same-problem lesson backwards.
            log.warning(
                "correct estimate regressed from step %d to %d; preserving the proven prefix",
                prev_state.last_correct_step, clamped,
            )
            clamped = min(prev_state.last_correct_step, max_step)
        if state.status == "CORRECT":
            # Only a confirming diagnosis moves the lesson forward on the
            # spot — that is the claim worth auditing. An error diagnosis
            # already keeps the tutor engaged with the same ground.
            clamped = self._frontier_result_check(
                clamped, reference, prev_state, rec, transcript
            )
        attempts = 1
        if prev_state is not None and clamped == prev_state.last_correct_step:
            attempts = prev_state.attempt_count + 1
        effective = None
        normalized = state.model_copy(update={"last_correct_step": clamped})
        if prev_state is not None:
            effective = hint_was_effective(prev_state, normalized)
        return normalized.model_copy(
            update={
                "attempt_count": attempts,
                "previous_hint_effective": effective,
            }
        )

    # 새로 인정한 단계의 결과가 종이에도 말에도 없으면, 진도를 한 칸 되돌린다.
    @staticmethod
    def _frontier_result_check(
        lcs: int,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        rec: Recognition | None,
        transcript: str | None,
    ) -> int:
        """A route may replace a step's DERIVATION, never its RESULT.

        Live on problem 12 the page showed
        "2*(a_1+a_4+a_7) = r**3*(a_1+a_4+a_7)" — one division short of
        r³ = 2 — the diagnosis credited the r³ step as done, and the tutor
        asked for a_1 while r³ was still unwritten. When the NEWLY credited
        frontier step's own final claim is on no written line and its value
        was not just spoken, the frontier retreats exactly one step. The
        steps beneath stay credited: the student's route replaced their
        derivations, and that allowance is the judge's to give.
        """
        proven = prev_state.last_correct_step if prev_state is not None else 0
        if rec is None or not rec.student_work or lcs <= proven:
            return lcs
        step = next((s for s in reference.steps if s.idx == lcs), None)
        claim = (step.expression if step is not None else "").split(",")[-1].strip()
        if "=" not in claim:
            return lcs                     # nothing checkable: trust the judge
        # Written evidence is judged by the same standards the rule-based
        # fast path credits with: plain equivalence, or the reference's
        # symbolic derivative already substituted from the problem's own
        # definitions.
        if any(
            mathnorm.equations_equivalent(line, claim)
            or mathnorm.equations_same_form_with_derivatives(
                line, claim, rec.equations
            )
            for line in rec.student_work
        ):
            return lcs
        # The result may have been SAID instead: presence of the claim's
        # numeric tail in the transcript is enough to stand aside — leniency
        # here only prevents a false retreat.
        tail = claim.split("=")[-1].strip()
        spoken = re.sub(r"마이너스\s*", "-", transcript or "")
        if tail and re.search(rf"(?<![\d.]){re.escape(tail)}(?![\d.])", spoken):
            return lcs
        log.info(
            "frontier result %r is on no written line: step %d stays open",
            claim, lcs,
        )
        return lcs - 1

    # 숫자가 틀렸으면 관대한 모델 판정보다 우선한다 — sympy로 값을 대조해 계산 오류로 못 박는다.
    @staticmethod
    def _arithmetic_check(
        state: StudentState,
        reference: ReferenceSolution,
        rec: Recognition | None,
    ) -> StudentState:
        """A wrong number outranks a lenient judge.

        When the newest work line CLAIMS a value ("f'(1) = 2-1-2+3×1" claims 2,
        "x = 5" claims 5) and the verified answer is a scalar, sympy compares
        them — and a mismatch caps the diagnosis at CALCULATION_ERROR no matter
        how correct the model judged the work to be. This is exactly the miss
        that shipped: an LLM grader waved through a substitution whose
        arithmetic was wrong, and the tutor said 맞아요 to a wrong line.
        """
        if rec is None or not rec.student_work or state.status != "CORRECT":
            return state
        if reference.final_answer.kind != "SCALAR":
            return state
        claim = mathnorm.numeric_claim(rec.student_work[-1])
        if claim is None:
            return state
        try:
            expected = float(mathnorm.parse_expression(str(reference.final_answer.value)))
        except (mathnorm.ParseError, TypeError, ValueError):
            return state
        if abs(claim - expected) <= 1e-9:
            return state
        log.warning(
            "arithmetic check: last line claims %s but the answer is %s — "
            "overriding %s with CALCULATION_ERROR",
            claim, expected, state.status,
        )
        return state.model_copy(
            update={
                "status": "CALCULATION_ERROR",
                # the wrong line is the frontier: the final step is NOT done
                "last_correct_step": min(state.last_correct_step, len(reference.steps) - 1),
                "misconception": None,
            }
        )
