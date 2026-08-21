"""A model with a standby, because a quota error should not cost a lesson.

The tutor can be pointed at a better model for one job — Gemini's LearnLM
tuning for the hints, say — and that model can be exactly the one your key
cannot call today: a free-tier key lists gemini-3.1-pro-preview happily and
then answers every request with

    429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-3.1-pro

which is not a rate limit that clears in a minute, it is "not on your plan".
Without a standby that reaches the student as "잠깐 문제가 생겼어요" and a lost
turn, every turn, and the cause is invisible from where they are sitting.

So: try the preferred model, fall back to the one that has always worked, and
say so once rather than on every call. The fallback stays in effect for a
cooldown so a dead primary is paid for once a minute instead of once a turn —
a failed request still costs its round trip, and the student is waiting.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# 기본 모델이 죽었을 때, 다시 시도하기 전 예비 모델로 버티는 시간.
COOLDOWN_S = 60.0


# 원하는 모델을 먼저 쓰되, 실패하면 조용히 예비 모델로 넘어가는 래퍼.
class FallbackLLM:
    """LLMClient that prefers one model and survives it being unavailable."""

    # 기본 모델 · 예비 모델 · 로그 이름 · 쿨다운을 받는다.
    def __init__(self, primary, standby, label: str = "llm", cooldown_s: float = COOLDOWN_S):
        self.primary = primary
        self.standby = standby
        self.label = label
        self.cooldown_s = cooldown_s
        self._blocked_until = 0.0
        self._failures = 0

    # 단순 호출을 폴백 경로로 넘긴다.
    def complete_json(self, **kwargs):
        return self._call("complete_json", kwargs)

    # 툴 호출을 폴백 경로로 넘긴다.
    def run_with_tools(self, **kwargs):
        return self._call("run_with_tools", kwargs)

    # 스트리밍 호출을 폴백 경로로 넘긴다.
    def complete_json_stream(self, **kwargs):
        return self._call("complete_json_stream", kwargs)

    # --- internals ------------------------------------------------------------

    # 실제 분기: 쿨다운 중이면 예비 모델, 아니면 기본 모델을 시도하고 실패 시 예비로.
    def _call(self, method: str, kwargs):
        if time.monotonic() < self._blocked_until:
            return getattr(self.standby, method)(**kwargs)
        try:
            result = getattr(self.primary, method)(**kwargs)
        except Exception as e:  # noqa: BLE001 — any failure is the standby's cue
            self._trip(e)
            return getattr(self.standby, method)(**kwargs)
        self._failures = 0
        return result

    # 실패를 기록하고 쿨다운을 건다. 로그는 한 번만 크게 남긴다.
    def _trip(self, error: Exception) -> None:
        self._failures += 1
        self._blocked_until = time.monotonic() + self.cooldown_s
        message = str(error)
        # Log the whole thing once, then keep it short: a 429 body is 2 KB of
        # quota JSON and it is the same 2 KB every time.
        if self._failures == 1:
            hint = ""
            if "limit: 0" in message:
                hint = (
                    " — this model is not on your plan (free tier quota is 0); "
                    "enable billing or set a model your key can call"
                )
            log.error("%s failed, using the standby for %.0fs%s: %s",
                      self.label, self.cooldown_s, hint, message)
        else:
            log.warning("%s still failing (%d); standby for another %.0fs",
                        self.label, self._failures, self.cooldown_s)
