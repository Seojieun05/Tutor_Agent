"""Student State Estimator: rule pre-checks + LLM diff against the reference.

The estimator only RETURNS a state — the orchestrator owns all store writes
and resolves the pending hint's effectiveness via hint_was_effective.
"""

from __future__ import annotations

import json
import logging

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.state.models import STATUS_RANK, StudentState
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

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
- last_correct_step: highest reference step index the student has completed correctly (0 = none).
- status: CORRECT | CALCULATION_ERROR | CONCEPT_ERROR | PROCEDURAL_ERROR | MISREAD | STUCK | UNCERTAIN.
- misconception: prefer an id from the provided misconception list when the student's
  error matches its indicators; otherwise a short free-text description or null.
- current_step: short Korean description of what the student is currently doing.
- You may look up misconception details in the domain knowledge base.

Return ONLY the JSON object."""


def hint_was_effective(prev: StudentState, new: StudentState) -> bool:
    """Effective = step progress OR misconception resolved OR status improved."""
    progress = new.last_correct_step > prev.last_correct_step
    resolved = prev.misconception is not None and new.misconception != prev.misconception
    improved = STATUS_RANK.get(new.status, 0) > STATUS_RANK.get(prev.status, 0)
    return progress or resolved or improved


class StudentStateEstimator:
    def __init__(self, llm: LLMClient, db: KnowledgeDB, conf_threshold: float = 0.6):
        self.llm = llm
        self.db = db
        self.conf_threshold = conf_threshold

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
            return self._post_rules(quick, reference, prev_state, rec)

        state = self.llm.run_with_tools(
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
            state = self.llm.run_with_tools(
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
            state = self.llm.run_with_tools(
                purpose="estimate",
                system=_SYSTEM,
                user=self._build_context(rec, reference, None, [], transcript),
                schema=StudentState,
            )
        return self._post_rules(state, reference, prev_state, rec)

    # --- deterministic pre-checks: no LLM call -------------------------------

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

        def same(a: str, b: str) -> bool:
            return mathnorm.equations_same_form(a, b)

        lcs = 0
        for step in reference.steps:
            if any(same(line, step.expression) for line in rec.student_work):
                lcs = max(lcs, step.idx)
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

    @staticmethod
    def _hint_pending(history: list[HintRecord]) -> bool:
        return any(h.level >= 1 and h.effective is None for h in history)

    @staticmethod
    def _last_line_illegible(rec: Recognition) -> bool:
        if not rec.uncertain_regions or not rec.student_work:
            return False
        last = rec.student_work[-1]
        return any(last and last in region for region in rec.uncertain_regions)

    # --- LLM context (state/history prefetched by the orchestrator) ----------

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

    def _post_rules(
        self,
        state: StudentState,
        reference: ReferenceSolution,
        prev_state: StudentState | None,
        rec: Recognition | None = None,
    ) -> StudentState:
        state = self._arithmetic_check(state, reference, rec)
        max_step = len(reference.steps)
        clamped = min(max(state.last_correct_step, 0), max_step)
        attempts = 1
        if prev_state is not None and clamped == prev_state.last_correct_step:
            attempts = prev_state.attempt_count + 1
        effective = None
        if prev_state is not None:
            effective = hint_was_effective(prev_state, state)
        return state.model_copy(
            update={
                "last_correct_step": clamped,
                "attempt_count": attempts,
                "previous_hint_effective": effective,
            }
        )

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
