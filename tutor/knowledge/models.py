"""Domain models shared across the knowledge, solver, and matching layers."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator

# 정답의 형태: 값 하나 / 근의 집합 / 기호식.
AnswerKind = Literal["SCALAR", "ROOT_SET", "EXPRESSION"]


# 최종 정답.
class Answer(BaseModel):
    kind: AnswerKind
    value: str | list[str]


# 기준 풀이의 한 단계: 번호, 한국어 설명, 그 단계를 마친 뒤의 식.
class SolutionStep(BaseModel):
    idx: int
    description: str
    expression: str


# 기준 풀이 한 벌. verified는 사람이 검증한 것만 True — 모델이 만든 풀이는 항상 False.
class ReferenceSolution(BaseModel):
    steps: list[SolutionStep]
    final_answer: Answer
    concepts: list[str] = []
    verified: bool = False
    origin: Literal["db", "template", "grok"] = "grok"

    # 모델이 단계 번호를 빠뜨렸으면 순서대로 채워 준다(번호가 없다고 풀이를 통째로 버리지 않게).
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


# DB에 저장된 문제 한 건: 본문·수식·파라미터·정답·출처·검증 여부·템플릿·개념 태그.
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


# 템플릿 문제의 단계(설명과 식에 {param} 자리가 들어간다).
class TemplateStep(BaseModel):
    description: str  # Korean, with {param} slots
    expression: str  # sympy template in the params, may contain '='


# 같은 구조·다른 숫자 문제를 만들어 내는 풀이 템플릿.
class Template(BaseModel):
    id: str
    problem_type: str
    pattern: str
    params: list[str]
    steps: list[TemplateStep]
    answer_kind: AnswerKind


# 알려진 오개념 하나: 설명과, 그것을 알아보는 단서들.
class Misconception(BaseModel):
    id: str
    concept_id: str
    description: str
    indicators: list[str] = []


# 힌트 문장 템플릿. 개념 또는 오개념에 묶이고 레벨을 가진다.
class HintTemplate(BaseModel):
    id: str
    concept_id: str | None = None
    misconception_id: str | None = None
    level: int
    template_text: str  # Korean, may contain {term}/{a}/{b}/{c}/{step} slots


# 매칭 등급: 완전 동일 / 같은 템플릿 / 같은 개념 / 임베딩 유사 / 처음 보는 문제.
class Tier(str, Enum):
    EXACT = "EXACT"
    TEMPLATE = "TEMPLATE"
    CONCEPT = "CONCEPT"
    SEMANTIC = "SEMANTIC"  # nearest by text embedding; no verified solution
    NEW = "NEW"


# 매칭 결과: 등급과, 찾아낸 문제·파라미터 바인딩·개념·기준 풀이.
class MatchResult(BaseModel):
    tier: Tier
    problem: Problem | None = None
    bindings: dict[str, str] | None = None
    concepts: list[str] = []
    reference: ReferenceSolution | None = None
