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


# 힌트 이력 한 줄: 어느 문제·단계·레벨에 무엇을 말했고, 그게 효과가 있었는지.
@dataclass(frozen=True)
class HintRecord:
    id: int
    problem_hash: str
    step: int
    level: int
    action: str
    hint_text: str
    effective: bool | None  # None until the next state estimate resolves it


# 연결 하나의 기억: 학생 상태 + 힌트 이력. 읽기는 여러 곳, 쓰기는 오케스트레이터만.
class SessionStore:
    # 빈 상태로 시작.
    def __init__(self) -> None:
        self._state: StudentState | None = None
        self._history: list[HintRecord] = []
        self._next_id = 1

    # --- reads (prefetched by the orchestrator into LLM context) -------------

    # 현재 학생 상태 사본.
    def get_state(self) -> StudentState | None:
        return self._state.model_copy() if self._state is not None else None

    # 힌트 이력 조회(단계·문제 해시로 거를 수 있다).
    def get_history(
        self, step: int | None = None, problem_hash: str | None = None
    ) -> list[HintRecord]:
        return [
            h
            for h in self._history
            if (step is None or h.step == step)
            and (problem_hash is None or h.problem_hash == problem_hash)
        ]

    # 아직 답을 받지 못한 '가장 최근' 힌트. 오래된 미해결 힌트는 되살리지 않는다.
    def pending_hint(self, problem_hash: str | None = None) -> HintRecord | None:
        """The latest real hint, if that latest hint is still unresolved.

        An older unresolved record is history, not a question that can come
        back from the dead after a newer question was answered.  Scanning past
        a resolved latest hint resurrected a step-1 question during step 4.
        """
        for h in reversed(self._history):
            if problem_hash is not None and h.problem_hash != problem_hash:
                continue
            if h.level >= 1:
                return h if h.effective is None else None
        return None

    # --- writes (orchestrator only) ------------------------------------------

    # 학생 상태 저장.
    def set_state(self, state: StudentState) -> None:
        self._state = state.model_copy()

    # 문제가 바뀌었을 때 상태를 비운다(이력은 문제별로 걸러 쓰므로 유지).
    def clear_state(self) -> None:
        """The student moved to a different problem: the old state is
        meaningless there. History is kept (it is filtered per problem)."""
        self._state = None

    # 힌트 이력 추가. 효과 여부는 아직 미정(None).
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

    # 힌트에 효과 여부를 기록.
    def mark_hint_effective(self, hint_id: int, effective: bool) -> None:
        for i, h in enumerate(self._history):
            if h.id == hint_id:
                self._history[i] = replace(h, effective=effective)
                return
        raise KeyError(f"no hint record with id {hint_id}")

    # 학생이 못 들은 턴이었으면 그 질문을 다시 미해결 상태로 되돌린다.
    def unresolve_hint(self, hint_id: int) -> None:
        """Put a hint back on the table, as if it had never been answered.

        For turns the student never heard: a reply cut off by a barge-in
        resolved a question that, from where the student sits, is still the
        one being asked.
        """
        for i, h in enumerate(self._history):
            if h.id == hint_id:
                self._history[i] = replace(h, effective=None)
                return
        raise KeyError(f"no hint record with id {hint_id}")

    # 만들었지만 한 번도 말하지 않은 힌트를 지운다.
    def drop_hint(self, hint_id: int) -> None:
        """Erase a hint that was generated but never spoken."""
        self._history = [h for h in self._history if h.id != hint_id]
