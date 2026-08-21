"""How long each model call took, and what the turn added up to.

The tutor's latency is almost entirely other people's servers, so the only
useful question is *which* call. One log line per call answers it in production:

    llm.timing INFO  phrase   4.8s  (gemini-3.6-flash)
    llm.timing INFO  estimate 2.1s  (gemini-3.6-flash)

`turn()` groups the calls a single student utterance provoked and logs the total
next to the parts, because a turn that feels slow is usually two calls that each
looked fine. It is a contextvar rather than an argument so the stages do not
have to thread a collector through the pipeline to be counted.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager

log = logging.getLogger("llm.timing")

# The turn currently being timed, if any. A contextvar and not a global: two
# sessions on one server are two independent turns.
_turn: contextvars.ContextVar[list | None] = contextvars.ContextVar("turn", default=None)

# Every model call this process has made. A stage that took no time because it
# hit a verified DB template looks identical to a fast model until you can see
# this, and telling those two apart is the whole point of measuring.
_calls = 0


# 이 프로세스가 지금까지 한 모델 호출 수.
def model_calls() -> int:
    return _calls


# 한 구간의 소요 시간을 로그로 남기고, 턴이 측정 중이면 거기에도 더한다.
def record(label: str, seconds: float, detail: str = "") -> None:
    """Log one stage, and add it to the turn if one is being timed."""
    log.info("%-9s %5.1fs  %s", label, seconds, detail)
    entries = _turn.get()
    if entries is not None:
        entries.append((label, seconds))


# with 블록의 소요 시간을 재는 구간.
@contextmanager
def stage(label: str, detail: str = ""):
    """Time a block whether or not it succeeds — a slow failure is still slow."""
    started = time.perf_counter()
    try:
        yield
    finally:
        record(label, time.perf_counter() - started, detail)


# 학생 발화 하나가 만든 모든 호출을 묶어 총합까지 로그로 남긴다.
@contextmanager
def turn(name: str):
    """Group the stages of one student utterance and log what they add up to."""
    entries: list[tuple[str, float]] = []
    token = _turn.set(entries)
    started = time.perf_counter()
    try:
        yield entries
    finally:
        _turn.reset(token)
        total = time.perf_counter() - started
        parts = " ".join(f"{label} {secs:.1f}s" for label, secs in entries)
        # The gap is the interesting number when it is large: it is everything
        # that was NOT a model call — capture, sympy, the DB, our own code.
        counted = sum(secs for _, secs in entries)
        log.info(
            "TURN %s %.1fs = %s%s",
            name, total, parts or "(no model calls)",
            f" + {total - counted:.1f}s other" if total - counted > 0.05 else "",
        )


# 로그에 찍을 모델 이름을 클라이언트에서 캐낸다.
def model_name(client) -> str:
    """What this client will ACTUALLY call.

    Asked of the client rather than read off a setting, because
    GEMINI_HINT_MODEL is set whether or not HINT_PROVIDER=gemini — labelling
    calls from the setting reports a model that never ran.
    """
    name = getattr(client, "model", None)  # GeminiClient
    if name:
        return str(name)
    settings = getattr(client, "settings", None)  # GrokClient
    if settings is not None and getattr(settings, "chat_model", None):
        return str(settings.chat_model)
    primary = getattr(client, "primary", None)  # FallbackLLM: name what it tries first
    return model_name(primary) if primary is not None else "?"


# 어떤 LLM 클라이언트든 감싸서, 호출마다 시간 로그를 남기게 만든다.
def timed(client, _unused: str = ""):
    """Wrap an LLM client so every call it serves is recorded.

    A proxy rather than an edit to each provider: Grok, Gemini and the fallback
    chain all satisfy the same two-method protocol, and the fallback has to be
    timed from the outside anyway — what the student waits through is the retry
    included, not the first attempt that failed.
    """
    if getattr(client, "_timed", False):
        return client  # vision_llm and hint_llm are often the same object

    model = model_name(client)

    # 시간 측정만 얹은 얇은 래퍼.
    class Timed:
        _timed = True

        # 감싸지 않은 속성은 원래 객체로 그대로 넘긴다.
        def __getattr__(self, name):  # keep .llm, .settings and friends reachable
            return getattr(client, name)

        # 시간을 재며 단순 호출.
        def complete_json(self, *, purpose, **kwargs):
            global _calls
            _calls += 1
            with stage(purpose, model):
                return client.complete_json(purpose=purpose, **kwargs)

        # 시간을 재며 툴 호출.
        def run_with_tools(self, *, purpose, **kwargs):
            global _calls
            _calls += 1
            with stage(purpose, model):
                return client.run_with_tools(purpose=purpose, **kwargs)

    return Timed()
