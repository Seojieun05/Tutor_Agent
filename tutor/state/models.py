"""Student state schema (spec: Student State section)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# 학생 상태 판정값. 맞음 / 계산 오류 / 개념 오류 / 절차 오류 / 잘못 읽음 / 막힘 / 판단 불가.
Status = Literal[
    "CORRECT",
    "CALCULATION_ERROR",
    "CONCEPT_ERROR",
    "PROCEDURAL_ERROR",
    "MISREAD",
    "STUCK",
    "UNCERTAIN",
]

# Coarse ranking used by hint_was_effective's "status improved" check.
# UNCERTAIN ranks with the errors: a blurry frame is missing information,
# never evidence of improvement.
STATUS_RANK: dict[str, int] = {
    "STUCK": 0,
    "CONCEPT_ERROR": 0,
    "CALCULATION_ERROR": 0,
    "PROCEDURAL_ERROR": 0,
    "MISREAD": 0,
    "UNCERTAIN": 0,
    "CORRECT": 1,
}


# 학생 상태 한 장: 지금 단계, 맞게 끝낸 마지막 단계, 판정, 오개념, 시도 횟수, 직전 힌트 효과.
class StudentState(BaseModel):
    current_step: str = ""
    last_correct_step: int = 0
    status: Status = "STUCK"
    misconception: str | None = None
    attempt_count: int = 1
    previous_hint_effective: bool | None = None
