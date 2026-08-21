"""EchoLLMClient: deterministic canned outputs, no network.

Powers echo mode (no XAI_API_KEY) so the whole websocket pipeline runs
end-to-end, and doubles as the test double — tests pass canned per-purpose
queues to script multi-turn scenarios.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Sequence

from pydantic import BaseModel

# echo 모드용 가짜 학생 풀이 진행 단계. 사진이 바뀔 때마다 한 칸씩 나아가서
# 오프라인에서도 힌트 상승·완화 시연이 된다.
# each DISTINCT image seen by `recognize` advances one stage of this scripted
# worksheet, so the h → n → h demo shows escalation AND fading offline
_WORK_STAGES = [
    ["3*x = 20 + 5"],  # sign_flip_on_move mistake
    ["3*x = 15"],  # corrected step 1
    ["3*x = 15", "x = 5"],  # solved
]

# 용도별 기본 응답. API 키 없이도 전체 파이프라인이 돌게 해 주는 미리 정해진 답들.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "recognize": {
        "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
        "equations": ["3*x + 5 = 20"],
        "choices": [],
        "diagram_conditions": [],
        "student_work": ["3*x = 20 + 5"],
        "uncertain_regions": [],
        "confidence": 0.95,
        # the merged recognize call carries the tags (whitelist-valid ids)
        "problem_type": "linear_equation",
        "concepts": ["linear_equation"],
    },
    "solve": {
        "steps": [
            {"idx": 1, "description": "양변에서 5를 빼서 x항만 남긴다", "expression": "3*x = 15"},
            {"idx": 2, "description": "양변을 x의 계수 3으로 나눈다", "expression": "x = 5"},
        ],
        "final_answer": {"kind": "SCALAR", "value": "5"},
        "concepts": ["linear_equation"],
        "verified": False,
        "origin": "grok",
    },
    "estimate": {
        "current_step": "상수항 이항",
        "last_correct_step": 0,
        "status": "CONCEPT_ERROR",
        "misconception": "sign_flip_on_move",
        "attempt_count": 1,
        "previous_hint_effective": None,
    },
    "phrase": {"hint": "지금 단계에서 어떤 등식의 성질을 쓸 수 있을지 생각해 볼까요?"},
    "tag": {"problem_type": "linear_equation", "concepts": ["linear_equation"]},
    # echo mode transcribes every utterance as "힌트 주세요", which the rules
    # already resolve — this only covers a test that forces the ambiguous path
    "intent": {"intent": "HINT_REQUEST", "reason": "echo"},
    # echo mode cannot judge an answer: stay neutral so the tutor re-asks the
    # same level instead of pretending the student was right or wrong
    "explain": {"hint": "그렇게 하면 양쪽이 똑같이 유지되기 때문이에요. 다시 한번 해 볼까요?"},
    "preflight": {"hint": "일차방정식 문제군요. 미지수의 계수와 상수항을 확인하셨나요?"},
    # echo mode teaches in words: no picture rather than a fake one
    "illustrate": {"functions": [], "caption": "", "why": "echo mode"},
    "evaluate": {
        "intent": "ANSWER",
        "verdict": "UNCLEAR",
        "feedback": "잘 못 들었어요.",
        "misconception": None,
        "status": None,
    },
}


# 실제 모델 대신 정해진 답을 돌려주는 클라이언트(오프라인 시연·테스트용).
class EchoLLMClient:
    # 테스트가 지정한 응답 큐를 받고, 호출된 용도 순서를 기록해 둔다.
    def __init__(self, responses: dict[str, list[BaseModel | dict[str, Any]]] | None = None):
        self._queues = {k: list(v) for k, v in (responses or {}).items()}
        self._image_stage: dict[str, int] = {}  # image hash → progress stage
        self.calls: list[str] = []  # purposes, in order — tests assert on this
        # The user message of each call, in the same order as `calls`. What a
        # prompt CARRIES is behaviour too: a hint that must not be written
        # blind to the rest of its ladder is testable here without a model.
        self.prompts: list[str] = []

    # 그 용도로 실제 어떤 프롬프트가 갔는지(테스트 검증용).
    def prompts_for(self, purpose: str) -> list[str]:
        """Every user message sent for one purpose, in order."""
        return [
            user for called, user in zip(self.calls, self.prompts)
            if called == purpose
        ]

    # 다음 응답 하나 꺼내기(큐가 비면 기본값, recognize는 사진에 따라 단계 진행).
    def _next(
        self, purpose: str, schema: type[BaseModel], images: Sequence[bytes] = ()
    ) -> BaseModel:
        self.calls.append(purpose)
        queue = self._queues.get(purpose)
        if queue:
            item = queue.pop(0)
            if isinstance(item, BaseModel):
                return schema.model_validate(item.model_dump())
            return schema.model_validate(item)
        if purpose == "recognize" and images:
            key = hashlib.sha1(images[0]).hexdigest()
            stage = self._image_stage.setdefault(
                key, min(len(self._image_stage), len(_WORK_STAGES) - 1)
            )
            return schema.model_validate(
                dict(_DEFAULTS["recognize"], student_work=_WORK_STAGES[stage])
            )
        if purpose in _DEFAULTS:
            return schema.model_validate(_DEFAULTS[purpose])
        raise KeyError(f"EchoLLMClient has no canned response for purpose {purpose!r}")

    # 정해진 답을 스키마에 맞춰 돌려준다.
    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[BaseModel],
    ) -> BaseModel:
        self.prompts.append(user)
        return self._next(purpose, schema, images)

    # 툴 호출 없이 같은 방식으로 응답.
    def run_with_tools(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[BaseModel],
        max_rounds: int = 6,
    ) -> BaseModel:
        self.prompts.append(user)
        return self._next(purpose, schema, images)

    # 완성된 문장을 한 번에 흘려보내 스트리밍 경로도 그대로 테스트되게 한다.
    def complete_json_stream(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[BaseModel],
        text_field: str,
        on_text_delta: Callable[[str], None],
    ) -> BaseModel:
        self.prompts.append(user)
        result = self._next(purpose, schema, images)
        value = getattr(result, text_field, "")
        if value:
            # Echo mode has no model tokens. Word-sized callbacks keep its wire
            # behaviour representative without pretending there was latency.
            for part in re.findall(r"\S+\s*", str(value)):
                on_text_delta(part)
        return result
