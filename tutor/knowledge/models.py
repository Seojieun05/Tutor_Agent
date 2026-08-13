"""Domain models shared across the knowledge, solver, and matching layers."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator

AnswerKind = Literal["SCALAR", "ROOT_SET", "EXPRESSION"]


class Answer(BaseModel):
    kind: AnswerKind
    value: str | list[str]


class SolutionStep(BaseModel):
    idx: int
    description: str
    expression: str


class ReferenceSolution(BaseModel):
    steps: list[SolutionStep]
    final_answer: Answer
    concepts: list[str] = []
    verified: bool = False
    origin: Literal["db", "template", "grok"] = "grok"

    @model_validator(mode="before")
    @classmethod
    def _number_the_steps(cls, data):
        """Fill in `idx` from position when the model left it out.

        A step's index IS where it sits in the list, and demanding the model
        restate it buys nothing — one solver wrote six perfectly good steps
        and lost the whole solution to six "Field required" errors. Numbering
        the model DID give is left exactly as it is: a solution that skips or
        reorders indices is saying something, and this is not the place to
        argue with it.
        """
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            for position, step in enumerate(data["steps"], start=1):
                if isinstance(step, dict) and step.get("idx") is None:
                    step["idx"] = position
        return data


class Problem(BaseModel):
    id: str
    problem_type: str
    problem_text: str
    equations: list[str]
    parameters: dict[str, str] = {}
    answer: Answer
    source: str = ""
    verified: bool = False
    template_id: str | None = None
    concepts: list[str] = []


class TemplateStep(BaseModel):
    description: str  # Korean, with {param} slots
    expression: str  # sympy template in the params, may contain '='


class Template(BaseModel):
    id: str
    problem_type: str
    pattern: str
    params: list[str]
    steps: list[TemplateStep]
    answer_kind: AnswerKind


class Misconception(BaseModel):
    id: str
    concept_id: str
    description: str
    indicators: list[str] = []


class HintTemplate(BaseModel):
    id: str
    concept_id: str | None = None
    misconception_id: str | None = None
    level: int
    template_text: str  # Korean, may contain {term}/{a}/{b}/{c}/{step} slots


class Tier(str, Enum):
    EXACT = "EXACT"
    TEMPLATE = "TEMPLATE"
    CONCEPT = "CONCEPT"
    SEMANTIC = "SEMANTIC"  # nearest by text embedding; no verified solution
    NEW = "NEW"


class MatchResult(BaseModel):
    tier: Tier
    problem: Problem | None = None
    bindings: dict[str, str] | None = None
    concepts: list[str] = []
    reference: ReferenceSolution | None = None
