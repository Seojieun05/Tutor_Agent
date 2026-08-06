"""What the tutor does when it genuinely cannot see the worksheet.

Two different failures produced one identical symptom on real hardware — the
tutor saying "문제와 지금까지 쓴 풀이가 잘 보이게 카메라에 다시 보여 줄래요?" and
nothing else, forever:

    the frame never arrived      RECOGNITION_FAILED fired before any history
                                 rule could notice it had already asked once
    the frame arrived unreadable problem_hash is derived from what the VLM
                                 read, so a garbled read minted a fresh identity
                                 every turn — and the per-problem history that
                                 was supposed to break the loop was empty each
                                 time

Both are now one retry followed by a verbal fallback, because a tutor that
cannot see can still teach by ear. The rule these tests defend: the same
recapture sentence is never spoken twice in a row.
"""

import pytest

from tutor.config import Settings
from tutor.hints.generator import FIXED_ACTIONS, HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.policy.engine import Action
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

RECAPTURE = FIXED_ACTIONS[Action.ASK_RECAPTURE]
PROBE = FIXED_ACTIONS[Action.PROBE]

# A blurred photo does not read the same way twice: the transcription drifts,
# and with it the problem identity. This is the input that used to loop.
UNREADABLE = [
    {"problem_text": "다음 방정 ... 푸시", "equations": [], "student_work": [],
     "uncertain_regions": ["전체가 흐림"], "confidence": 0.2},
    {"problem_text": "다음 방정식을 푸시오 3x", "equations": [], "student_work": [],
     "uncertain_regions": ["대부분 읽을 수 없음"], "confidence": 0.25},
    {"problem_text": "다 방정식 푸시오", "equations": [], "student_work": [],
     "uncertain_regions": ["흐림"], "confidence": 0.15},
]


class FakeWS:
    """Just enough socket for _deliver: it sends events and nothing reads them."""

    def __init__(self):
        self.sent: list[str] = []

    async def send(self, message):
        self.sent.append(message)


@pytest.fixture
def session(db):
    llm = EchoLLMClient({"recognize": list(UNREADABLE)})
    deps = Deps(
        settings=Settings(capture_timeout_s=0.1),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(),
        store=SessionStore(),
    )
    return Session(FakeWS(), deps)


async def test_a_camera_that_never_answers_stops_repeating_itself(session):
    """No frame at all: _request_capture returns None every time."""
    session._request_capture = lambda: _none()

    await session.handle_hint_request()
    await session.handle_hint_request()
    await session.handle_hint_request()

    said = session.deps.speaker.spoken
    assert said[0] == RECAPTURE, "the first failure should ask for another photo"
    assert said[1] == PROBE, "the second should switch to teaching by voice"
    assert RECAPTURE not in said[1:], "asking again would be the old infinite loop"


async def test_an_unreadable_photo_stops_repeating_itself(session, monkeypatch):
    """Frames arrive, but every read is garbage — and reads differently each
    time, so the per-problem history cannot be what breaks the loop."""
    session._request_capture = lambda: _jpeg()

    await session.handle_hint_request()
    await session.handle_hint_request()

    said = session.deps.speaker.spoken
    assert said[0] == RECAPTURE
    assert said[1] != RECAPTURE, (
        "the second unreadable photo must not produce the same sentence: "
        "problem_hash churns on a garbled read, which used to empty the history"
    )


async def test_the_hint_history_survives_an_unreadable_photo(session):
    """The recapture has to be recorded, or nothing downstream can count it."""
    session._request_capture = lambda: _none()
    await session.handle_hint_request()

    history = session.store.get_history()
    assert len(history) == 1
    assert history[0].action == Action.ASK_RECAPTURE.value


async def _none():
    return None


async def _jpeg():
    return b"\xff\xd8" + b"blur" * 1024
