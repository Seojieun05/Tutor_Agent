"""Pedagogical policy: a pure, rule-based state machine over hint levels L0-L4.

Ordered rules, first match wins. Invariants: first hint at a step is always
L1; escalation is exactly +1 and only after an ineffective hint (effective is
False — None does not escalate); progress resets to L1 (fading); no action can
reveal the final answer. The orchestrator prefetches state and history from
the SessionStore immediately before calling decide().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from tutor.state.models import StudentState
from tutor.store.session_store import HintRecord

Trigger = Literal["HINT_REQUEST", "STATE_UPDATE", "RECOGNITION_FAILED"]


class Action(str, Enum):
    WAIT = "WAIT"
    PROBE = "PROBE"
    SOCRATIC_QUESTION = "SOCRATIC_QUESTION"
    CONCEPT_HINT = "CONCEPT_HINT"
    PROCEDURAL_HINT = "PROCEDURAL_HINT"
    PARTIAL_STEP = "PARTIAL_STEP"
    ASK_RECAPTURE = "ASK_RECAPTURE"


LEVEL_ACTIONS: dict[int, Action] = {
    0: Action.WAIT,
    1: Action.SOCRATIC_QUESTION,
    2: Action.CONCEPT_HINT,
    3: Action.PROCEDURAL_HINT,
    4: Action.PARTIAL_STEP,
}

MAX_LEVEL = 4


@dataclass(frozen=True)
class Decision:
    action: Action
    level: int
    target_step: int
    misconception: str | None
    rationale: str


def decide(state: StudentState, history: list[HintRecord], trigger: Trigger) -> Decision:
    target = state.last_correct_step + 1

    # R10: capture/decode failure — nothing to reason about.
    if trigger == "RECOGNITION_FAILED":
        return Decision(Action.ASK_RECAPTURE, 0, target, None, "recognition failed")

    # R1/R2: unreadable state — recapture once, then probe verbally.
    if state.status == "UNCERTAIN":
        if history and history[-1].action == Action.ASK_RECAPTURE.value:
            return Decision(Action.PROBE, 0, target, None, "recapture already tried; probe")
        return Decision(Action.ASK_RECAPTURE, 0, target, None, "uncertain recognition")

    # R3/R4: passive state updates never interrupt the student (spec rule 7).
    if trigger == "STATE_UPDATE":
        return Decision(Action.WAIT, 0, target, None, "no interruption on state update")

    # HINT_REQUEST from here on.
    prior = [h for h in history if h.step == target and h.level >= 1]

    # R5-R7: first hint at this step (including CORRECT students asking ahead) is L1.
    if not prior:
        return Decision(
            LEVEL_ACTIONS[1], 1, target, state.misconception, "first hint at step: weakest"
        )

    last = prior[-1]
    if last.effective is False:
        # R8: escalate exactly one level past the strongest hint tried here.
        level = min(max(h.level for h in prior) + 1, MAX_LEVEL)
        return Decision(
            LEVEL_ACTIONS[level],
            level,
            target,
            state.misconception,
            f"hint L{last.level} ineffective: escalate to L{level}",
        )

    # R9: last hint helped (or is unresolved) but the student asks again —
    # repeat the level, don't escalate on effective=None.
    return Decision(
        LEVEL_ACTIONS[last.level],
        last.level,
        target,
        state.misconception,
        "previous hint not proven insufficient: repeat level",
    )
