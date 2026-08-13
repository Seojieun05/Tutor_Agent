"""The Grok Solver runs while the student hears the first hint.

On a NEW problem the solver was ~25s of dead air — over half the first turn —
and nothing the first hint says needs its output: the policy opens at L1 from
the concepts regardless. So the solve becomes a background task, the first hint
is asked from the concepts, and the ANSWER turn (which grades against the
reference and genuinely cannot proceed without it) awaits the task — by which
time it has almost always finished.

The gate below stands in for those 25 seconds: the test decides when the
solver "finishes", so these scenarios are exact rather than raced.
"""

import asyncio
import json
import threading

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.intent import IntentClassifier
from tutor.speech.stt import Transcript
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

JPEG = b"\xff\xd8" + b"csat" * 2048
PCM = b"\x00\x00" * 100

# Not in the seeded KB: matching falls through, so the solver must run.
UNSEEN_PROBLEM = {
    "problem_text": "1보다 큰 두 실수 a, b가 log_a b = 3을 만족시킬 때, log_9 ab의 값은?",
    "equations": ["log_a(b) = 3"],
    "choices": ["3/8", "1/2", "5/8", "3/4", "7/8"],
    "diagram_conditions": [],
    "student_work": [],
    "uncertain_regions": [],
    "confidence": 0.97,
}


class GatedSolver:
    """The real solver behind a gate the TEST holds the key to."""

    def __init__(self, inner):
        self.inner = inner
        self.release = threading.Event()
        self.fail_next = False
        self.fail_always = False
        self.calls = 0

    def solve(self, rec, h):
        self.calls += 1
        assert self.release.wait(timeout=10), "test never released the solver"
        if self.fail_always or self.fail_next:
            self.fail_next = False
            raise RuntimeError("solver exploded")
        return self.inner.solve(rec, h)


class PhoneWS:
    """A device with a camera on the session socket (tutor/web/phone.html)."""

    def __init__(self):
        self.events: list[dict] = []
        self.session: Session | None = None

    async def send(self, raw):
        if not isinstance(raw, str):
            return
        event = json.loads(raw)
        self.events.append(event)
        if event["event"] == "capture_request":
            await self.session.on_frame(
                encode_image(JPEG, ImageHeader(capture_id=event["data"]["capture_id"]))
            )


class ScriptedTranscriber:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000) -> Transcript:
        return Transcript(text=self.texts.pop(0) if self.texts else "")


def build(db, *utterances: str, llm_responses: dict | None = None):
    responses = {"recognize": [UNSEEN_PROBLEM] * 6}
    responses.update(llm_responses or {})
    llm = EchoLLMClient(responses)
    solver = GatedSolver(GrokSolver(llm, db))
    speaker = NullSpeaker()
    ws = PhoneWS()
    deps = Deps(
        settings=Settings(capture_timeout_s=2.0),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=solver,
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(*utterances),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        classifier=IntentClassifier(),
        store=SessionStore(),
    )
    session = Session(ws, deps)
    ws.session = session
    return session, llm, speaker, ws, solver


async def test_the_first_hint_does_not_wait_for_the_solver(db):
    """The point: L1 lands while the solver is still thinking."""
    session, llm, speaker, ws, solver = build(db, "이 문제 힌트 줄래?")

    await session._handle_utterance(PCM, 16000)  # solver still gated

    assert speaker.spoken and speaker.spoken[0]              # a hint was spoken
    history = session.store.get_history(problem_hash=session.ctx.hash)
    assert history and history[-1].level == 1                # and it was L1
    assert session.ctx.reference is None                     # no reference yet
    assert session.ctx.solving is not None and not session.ctx.solving.done()
    assert llm.calls.count("estimate") == 0   # nothing to diagnose against yet

    solver.release.set()                                     # solver "finishes"
    reference = await session.ctx.reference_ready()
    assert reference.steps                                   # and lands in ctx
    assert session.ctx.reference is reference


async def test_a_spoken_answer_waits_out_the_solve_then_grades(db):
    """Grading needs the reference: the answer turn awaits the running task."""
    session, llm, speaker, ws, solver = build(
        db, "이 문제 힌트 줄래?", "5예요",
        llm_responses={
            "evaluate": [{"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
                          "misconception": None, "status": "CORRECT"}],
        },
    )
    await session._handle_utterance(PCM, 16000)              # hint; solve pending
    assert session.ctx.reference is None

    # the solver finishes a beat AFTER the answer starts waiting on it
    asyncio.get_running_loop().call_later(0.05, solver.release.set)
    await session._handle_utterance(PCM, 16000)              # "5예요"

    assert llm.calls.count("evaluate") == 1                  # graded, not dropped
    assert llm.calls.count("recognize") == 1                 # and no re-capture
    assert session.ctx.reference is not None


async def test_a_failed_solve_is_retried_on_the_answer_turn_never_re_photographed(db):
    """A dead solve must not cost the student the answer they just gave.

    This used to fall back to the capture flow, which is the wrong shape twice
    over: the student SPOKE, so nothing on the paper has changed, and if the
    phone has dropped — it had, three seconds earlier, the night this was
    found — the answer turn ends by asking for a photo nobody can take. The
    recognition already in hand is all the solver ever needed.
    """
    session, llm, speaker, ws, solver = build(
        db, "이 문제 힌트 줄래?", "5예요",
        llm_responses={
            "evaluate": [{"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
                          "misconception": None, "status": "CORRECT"}],
        },
    )
    solver.fail_next = True
    solver.release.set()                                     # fail immediately

    await session._handle_utterance(PCM, 16000)              # first hint still fine
    await asyncio.sleep(0.05)                                # let the failure land
    assert session.ctx.reference is None

    await session._handle_utterance(PCM, 16000)              # "5예요"

    assert solver.calls == 2                                 # the solve was retried...
    assert session.ctx.reference is not None                 # ...and won this time
    assert llm.calls.count("evaluate") == 1                  # so the answer WAS graded
    assert llm.calls.count("recognize") == 1                 # and no second look at the page
    assert sum(1 for e in ws.events if e["event"] == "capture_request") == 1


async def test_when_the_retry_loses_too_the_tutor_explains_instead_of_asking_for_a_photo(db):
    """Second failure: still no camera. An explanation needs no reference."""
    session, llm, speaker, ws, solver = build(db, "이 문제 힌트 줄래?", "5예요")
    solver.fail_always = True
    solver.release.set()

    await session._handle_utterance(PCM, 16000)
    await asyncio.sleep(0.05)
    await session._handle_utterance(PCM, 16000)              # "5예요" — ungradeable twice

    assert solver.calls == 2                                 # tried again, lost again
    assert llm.calls.count("evaluate") == 0                  # nothing could be graded
    assert llm.calls.count("explain") == 1                   # so the pending question is explained
    assert len(speaker.spoken) == 2                          # and the student hears a reply
    assert sum(1 for e in ws.events if e["event"] == "capture_request") == 1
    # ungraded, so the answer is still evidence for whatever comes next
    assert session.last_transcript == "5예요"


async def test_a_kb_problem_never_starts_a_background_solve(db):
    """EXACT match: the verified reference exists, so nothing changed."""
    session, llm, speaker, ws, solver = build(
        db, "이 문제 힌트 줄래?",
        llm_responses={"recognize": [{
            "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
            "equations": ["3*x + 5 = 20"],
            "choices": [], "diagram_conditions": [], "uncertain_regions": [],
            "student_work": ["3*x = 20 + 5"],
            "confidence": 0.95,
        }]},
    )

    await session._handle_utterance(PCM, 16000)

    assert session.ctx.solving is None                       # no background task
    assert session.ctx.reference is not None                 # KB's verified solution
    assert solver.calls == 0
    assert llm.calls.count("estimate") >= 0                  # diagnosis ran as before
    assert speaker.spoken


async def test_closing_the_session_cancels_the_inflight_solve(db):
    session, llm, speaker, ws, solver = build(db, "이 문제 힌트 줄래?")
    await session._handle_utterance(PCM, 16000)
    task = session.ctx.solving
    assert task is not None and not task.done()

    for t in session._tasks:                                 # what run()'s finally does
        t.cancel()
    await asyncio.gather(*session._tasks, return_exceptions=True)
    assert task.cancelled()
