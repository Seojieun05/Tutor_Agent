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

# The two things the tutor says when it cannot see the worksheet.
BLIND_ACTIONS = frozenset({Action.ASK_RECAPTURE.value, Action.PROBE.value})


@dataclass(frozen=True)
class Decision:
    action: Action
    level: int
    target_step: int
    misconception: str | None
    rationale: str


def decide(state: StudentState, history: list[HintRecord], trigger: Trigger) -> Decision:
    target = state.last_correct_step + 1

    # Are we already teaching blind? Both level-0 actions mean "the tutor cannot
    # see the worksheet", so the tail of the history says whether asking for a
    # photo has been tried and has not worked. It re-arms by itself: the moment a
    # real hint (level >= 1) is given, the camera is evidently working again.
    still_blind = bool(history) and history[-1].action in BLIND_ACTIONS

    # R10: capture/decode failure — nothing to reason about. Worth one retry;
    # past that the tutor is talking to itself, which is exactly what a camera
    # that never delivers a frame used to produce, forever.
    if trigger == "RECOGNITION_FAILED":
        if still_blind:
            return Decision(Action.PROBE, 0, target, None, "camera keeps failing; go verbal")
        return Decision(Action.ASK_RECAPTURE, 0, target, None, "recognition failed")

    # R1/R2: unreadable state — recapture once, then probe verbally.
    if state.status == "UNCERTAIN":
        if still_blind:
            return Decision(Action.PROBE, 0, target, None, "recapture already tried; probe")
        return Decision(Action.ASK_RECAPTURE, 0, target, None, "uncertain recognition")

    # R3/R4: passive state updates never interrupt the student (spec rule 7).
    if trigger == "STATE_UPDATE":
        return Decision(Action.WAIT, 0, target, None, "no interruption on state update")

    # HINT_REQUEST from here on.
    prior = [h for h in history if h.step == target and h.level >= 1]

    # Progress can happen *inside* one reference step.  A composite step such
    # as "differentiate f, then evaluate f'(1)" may need L2 for its first
    # half, but once that hint works the remaining half starts again at L1.
    # Only hints after the latest proven-helpful one belong to the active
    # escalation run; older strong support must not make the next nudge L3.
    latest_progress = next(
        (i for i in range(len(prior) - 1, -1, -1) if prior[i].effective is True),
        None,
    )
    active = prior[latest_progress + 1:] if latest_progress is not None else prior

    # R5-R7: first hint at this step (including CORRECT students asking ahead) is L1.
    if not active:
        return Decision(
            LEVEL_ACTIONS[1], 1, target, state.misconception,
            "first hint after progress: weakest" if prior else "first hint at step: weakest",
        )

    last = active[-1]
    if last.effective is False:
        # R8: escalate exactly one level past the strongest hint tried here.
        level = min(max(h.level for h in active) + 1, MAX_LEVEL)
        return Decision(
            LEVEL_ACTIONS[level],
            level,
            target,
            state.misconception,
            f"hint L{last.level} ineffective: escalate to L{level}",
        )

    # R9: the active hint is unresolved but the student asks again — repeat
    # its level. A proven-helpful hint was sliced out above and faded to L1.
    return Decision(
        LEVEL_ACTIONS[last.level],
        last.level,
        target,
        state.misconception,
        "previous hint not proven insufficient: repeat level",
    )
