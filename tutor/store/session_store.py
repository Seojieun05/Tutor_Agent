"""Per-connection session store: student state + hint history.

Hard read/write split: the orchestrator prefetches reads for LLM prompts and
is the ONLY writer. No write method is ever registered as an LLM tool.

Every HintRecord carries the problem_hash it was issued for — history is
always queried per problem, so hints for one problem can never drive
escalation on another. Switching problems resets the student state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tutor.state.models import StudentState


@dataclass(frozen=True)
class HintRecord:
    id: int
    problem_hash: str
    step: int
    level: int
    action: str
    hint_text: str
    effective: bool | None  # None until the next state estimate resolves it


class SessionStore:
    def __init__(self) -> None:
        self._state: StudentState | None = None
        self._history: list[HintRecord] = []
        self._next_id = 1

    # --- reads (prefetched by the orchestrator into LLM context) -------------

    def get_state(self) -> StudentState | None:
        return self._state.model_copy() if self._state is not None else None

    def get_history(
        self, step: int | None = None, problem_hash: str | None = None
    ) -> list[HintRecord]:
        return [
            h
            for h in self._history
            if (step is None or h.step == step)
            and (problem_hash is None or h.problem_hash == problem_hash)
        ]

    def pending_hint(self, problem_hash: str | None = None) -> HintRecord | None:
        """Latest real hint (level >= 1) for this problem whose effectiveness
        is unresolved."""
        for h in reversed(self._history):
            if problem_hash is not None and h.problem_hash != problem_hash:
                continue
            if h.level >= 1 and h.effective is None:
                return h
        return None

    # --- writes (orchestrator only) ------------------------------------------

    def set_state(self, state: StudentState) -> None:
        self._state = state.model_copy()

    def clear_state(self) -> None:
        """The student moved to a different problem: the old state is
        meaningless there. History is kept (it is filtered per problem)."""
        self._state = None

    def append_hint(
        self, *, problem_hash: str, step: int, level: int, action: str, hint_text: str
    ) -> int:
        record = HintRecord(
            id=self._next_id,
            problem_hash=problem_hash,
            step=step,
            level=level,
            action=action,
            hint_text=hint_text,
            effective=None,
        )
        self._history.append(record)
        self._next_id += 1
        return record.id

    def mark_hint_effective(self, hint_id: int, effective: bool) -> None:
        for i, h in enumerate(self._history):
            if h.id == hint_id:
                self._history[i] = replace(h, effective=effective)
                return
        raise KeyError(f"no hint record with id {hint_id}")
