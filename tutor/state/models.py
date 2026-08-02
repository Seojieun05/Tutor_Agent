"""Student state schema (spec: Student State section)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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


class StudentState(BaseModel):
    current_step: str = ""
    last_correct_step: int = 0
    status: Status = "STUCK"
    misconception: str | None = None
    attempt_count: int = 1
    previous_hint_effective: bool | None = None
