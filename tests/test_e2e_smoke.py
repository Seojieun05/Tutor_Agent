"""End-to-end smoke test: in-process websocket server + scripted client.

Exercises framing, problem caching, matching, estimation, policy escalation
and fading, the leak guard, and the orchestrator's store discipline —
all offline (EchoLLMClient, NullSpeaker).
"""

import asyncio
import re

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

WRONG = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "student_work": ["3*x = 20 + 5"],
    "confidence": 0.95,
}
PROGRESSED = dict(WRONG, student_work=["3*x = 15"])

STATE_WRONG = {
    "current_step": "상수항 이항",
    "last_correct_step": 0,
    "status": "CONCEPT_ERROR",
    "misconception": "sign_flip_on_move",
    "attempt_count": 1,
    "previous_hint_effective": None,
}
STATE_PROGRESS = {
    "current_step": "계수로 나누기",
    "last_correct_step": 1,
    "status": "CORRECT",
    "misconception": None,
    "attempt_count": 1,
    "previous_hint_effective": None,
}


class SpyStore(SessionStore):
    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def get_state(self):
        self.ops.append("get_state")
        return super().get_state()

    def get_history(self, step=None):
        self.ops.append("get_history")
        return super().get_history(step)

    def set_state(self, state):
        self.ops.append("set_state")
        super().set_state(state)

    def append_hint(self, **kw):
        self.ops.append("append_hint")
        return super().append_hint(**kw)

    def mark_hint_effective(self, hint_id, effective):
        self.ops.append(f"mark_hint_effective:{effective}")
        super().mark_hint_effective(hint_id, effective)


@pytest.fixture
def scripted_deps(db):
    llm = EchoLLMClient(
        {
            "recognize": [WRONG, WRONG, PROGRESSED],
            "estimate": [STATE_WRONG, STATE_PROGRESS],  # 2nd request uses the no-LLM pre-check
        }
    )
    speaker = NullSpeaker()
    store = SpyStore()
    deps = Deps(
        settings=Settings(capture_timeout_s=2.0),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=speaker,
        store=store,
    )
    return deps, llm, speaker, store


# >1 MiB, like a real high-quality capture — regression for the server's
# websocket max_size (1 MiB default would close the connection with 1009)
BIG_JPEG = b"\xff\xd8" + b"\x00" * (2 * 1024 * 1024)


async def request_hint(ws) -> dict:
    """Send hint_request, answer the capture_request, return hint_issued data."""
    await ws.send(make_event("hint_request", {}))
    events = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        ev = parse_event(raw)
        events.append(ev.event)
        if ev.event == "capture_request":
            await ws.send(
                encode_image(BIG_JPEG, ImageHeader(capture_id=ev.data["capture_id"]))
            )
        elif ev.event == "hint_issued":
            assert "speech_state" in events  # spoke before issuing
            return ev.data


async def test_full_hint_flow(scripted_deps):
    deps, llm, speaker, store = scripted_deps

    async def handler(ws):
        await Session(ws, deps).run()

    async with serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}", max_size=16 * 1024 * 1024) as ws:
            await ws.send(make_event("hello", {"device_id": "test"}))
            assert parse_event(await ws.recv()).event == "hello_ack"

            # 1st hint: EXACT match on lin_001, L1 Socratic question, no leak.
            issued = await request_hint(ws)
            assert issued["level"] == 1
            assert issued["action"] == "SOCRATIC_QUESTION"
            first = speaker.spoken[0]
            assert first and "x = 5" not in first
            assert not re.search(r"(?<![\d.])5(?![\d.])", first)  # answer never spoken

            # 2nd hint, same worksheet: pre-check marks the L1 hint ineffective → L2.
            issued = await request_hint(ws)
            assert issued["level"] == 2
            assert issued["action"] == "CONCEPT_HINT"
            history = store.get_history()
            assert history[0].effective is False
            assert llm.calls.count("estimate") == 1  # 2nd used the no-LLM pre-check

            # 3rd hint, progressed worksheet: pending L2 marked effective, fade to L1.
            issued = await request_hint(ws)
            assert issued["level"] == 1
            history = store.get_history()
            assert history[1].effective is True
            assert history[2].step == 2  # now targeting the next step

    # store discipline: state/history prefetched (via the store) right before
    # each decide — i.e. between set_state and append_hint.
    ops = store.ops
    for i, op in enumerate(ops):
        if op == "set_state":
            until_append = ops[i + 1 : ops.index("append_hint", i)]
            assert "get_state" in until_append and "get_history" in until_append

    # solver never ran: EXACT tier provided a verified reference (spec rule 2).
    assert "solve" not in llm.calls
