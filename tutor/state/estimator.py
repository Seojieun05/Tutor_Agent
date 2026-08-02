"""Student State Estimator: rule pre-checks + LLM diff against the reference.

The estimator only RETURNS a state — the orchestrator owns all store writes
and resolves the pending hint's effectiveness via hint_was_effective.
"""

from __future__ import annotations

import json
import logging

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
        pre = self._pre_check(rec, prev_state, prev_work, history)
        if pre is not None:
            return pre

        state = self.llm.run_with_tools(
            purpose="estimate",
            system=_SYSTEM,
            user=self._build_context(rec, reference, prev_state, history, transcript),
            schema=StudentState,
        )
        return self._post_rules(state, reference, prev_state)

    # --- deterministic pre-checks: no LLM call -------------------------------

    def _pre_check(
        self,
        rec: Recognition,
        prev_state: StudentState | None,
        prev_work: list[str] | None,
        history: list[HintRecord],
    ) -> StudentState | None:
        if rec.confidence < self.conf_threshold or self._last_line_illegible(rec):
            base = prev_state or StudentState()
            return base.model_copy(update={"status": "UNCERTAIN"})
        if not rec.student_work:
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
        ):
            # Nothing changed since the last hint: it did not help. No LLM needed.
            return prev_state.model_copy(
                update={
                    "attempt_count": prev_state.attempt_count + 1,
                    "previous_hint_effective": False,
                }
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
    ) -> StudentState:
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
