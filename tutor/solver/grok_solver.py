"""Grok Solver fallback — used only for CONCEPT/NEW tiers (spec rule 2).

Solutions are sympy machine-checked but NEVER marked verified (spec: Grok
output is not automatically verified); passing candidates are stored in the
DB as unverified for later human review.

The prompt used to say "check yourself as you go", and on the one problem in
three that took it literally the model checked a single step per round trip:
9 rounds, 13 `compute` calls, two of them spent discovering that
`f = ...; diff(f, x)` does not parse, one an exact repeat of the round before,
one asking sympy to simplify `2*x - 4`. It ran out of rounds and the whole
solve was lost — which is how a background solve failure reached a student as
"show me your worksheet". Measured over three runs of that problem: checking
as it goes, 31.8s median; verifying once at the end, 16.9s, same answer. The
same-answer part holds across all twelve runs of three problems, and the
deterministic check below still decides what is stored.
"""

from __future__ import annotations

import logging

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import ReferenceSolution
from tutor.llm.client import LLMClient
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

# [프롬프트] 기준 풀이 작성용 시스템 프롬프트. 한 단계에 한 가지 연산, 설명은 한국어,
# 4점 문제는 보통 8~12단계로 쪼개게 한다(힌트 사다리가 이 단계들을 하나씩 밟기 때문).
# 검산은 마지막에 compute 한 번만 — 단계마다 검산하면 왕복이 늘어 배경 풀이가 통째로 실패했다.
_SYSTEM = """You are a careful math solver. Solve the given problem step by step.

Rules:
- One atomic operation per step, in order. `expression` is the state AFTER the step,
  in ASCII sympy syntax (* for multiplication, ** for powers).
- A student is walked through these steps one at a time and asked a question
  about each, so a step that bundles two ideas asks too much: split it. A 4점
  problem is usually 8-12 steps.
- Write step descriptions in Korean (the student is Korean).
- final_answer.kind: "SCALAR" for a single value, "ROOT_SET" for multiple roots,
  "EXPRESSION" for a symbolic result (e.g. a derivative).
- You may search the domain knowledge base for similar solved problems.
- Solve it yourself. Use `compute` ONCE, at the end, to verify the final
  answer; do not check intermediate steps one at a time. It takes ONE sympy
  expression — `diff(x**2 - 4*x - 3, x)`, never `f = x**2 - 4*x - 3; diff(f, x)`.
  If the check contradicts your answer, fix it before returning.
- concepts: short English snake_case tags like "linear_equation".
- origin must be "grok" and verified must be false.

Return ONLY the JSON object."""


# DB에 없는 문제(CONCEPT/NEW)일 때만 부르는 풀이 생성기.
class GrokSolver:
    # 풀이 모델과 지식 DB를 받는다.
    def __init__(self, llm: LLMClient, db: KnowledgeDB):
        self.llm = llm
        self.db = db

    # 문제를 풀어 기준 풀이를 만든다. 모델이 뭐라 하든 verified=False로 못 박고,
    # sympy 기계 검산을 통과한 것만 미검증 상태로 DB에 저장한다.
    def solve(self, rec: Recognition, problem_key: str) -> ReferenceSolution:
        user = (
            f"문제: {rec.problem_text}\n"
            f"수식: {rec.equations}\n"
            f"보기: {rec.choices}\n"
            f"그림 조건: {rec.diagram_conditions}"
        )
        solution = self.llm.run_with_tools(
            purpose="solve", system=_SYSTEM, user=user, schema=ReferenceSolution
        )
        # Spec rule: solver output is never verified, whatever the model claims.
        solution = solution.model_copy(update={"verified": False, "origin": "grok"})

        machine_checked = mathnorm.verify_answer(
            rec.equations, solution.final_answer.kind, solution.final_answer.value
        )
        log.info("solver machine check for %s: %s", problem_key, machine_checked)
        if machine_checked:
            self.db.insert_unverified_solution(problem_key, solution)
        return solution
