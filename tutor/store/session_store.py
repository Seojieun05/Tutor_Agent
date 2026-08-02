"""Per-connection session store: student state + hint history.

Hard read/write split: the orchestrator prefetches reads for LLM prompts and
is the ONLY writer. No write method is ever registered as an LLM tool.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tutor.state.models import StudentState


@dataclass(frozen=True)
class HintRecord:
    id: int
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

    def get_history(self, step: int | None = None) -> list[HintRecord]:
        if step is None:
            return list(self._history)
        return [h for h in self._history if h.step == step]

    def pending_hint(self) -> HintRecord | None:
        """Latest real hint (level >= 1) whose effectiveness is unresolved."""
        for h in reversed(self._history):
            if h.level >= 1 and h.effective is None:
                return h
        return None

    # --- writes (orchestrator only) ------------------------------------------

    def set_state(self, state: StudentState) -> None:
        self._state = state.model_copy()

    def append_hint(self, *, step: int, level: int, action: str, hint_text: str) -> int:
        record = HintRecord(
            id=self._next_id,
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
