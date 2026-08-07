"""'이렇게 하는 거 맞아?' — the turn that has to look again.

Three utterances, three different turns over the SAME session state. The one
that used to be broken is the middle one: a hint was pending, so any speech was
graded as an answer, and the tutor re-explained its own question at a worksheet
it had never re-read. Checking work IS re-reading the worksheet.

The camera here answers on the session's own socket, which is what a phone
running tutor/web/phone.html does: capture_request in, one JPEG back.
"""

import json

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.session import Deps, ProblemContext, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.intent import IntentClassifier
from tutor.speech.stt import Transcript
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

PHONE_JPEG = b"\xff\xd8" + b"phone" * 2048
PCM = b"\x00\x00" * 100

REFERENCE = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="양변에서 5를 뺀다", expression="3*x = 15"),
        SolutionStep(idx=2, description="양변을 3으로 나눈다", expression="x = 5"),
    ],
    final_answer=Answer(kind="SCALAR", value="5"),
    concepts=["linear_equation"],
    verified=True,
    origin="db",
)

PROBLEM = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "choices": [],
    "diagram_conditions": [],
    "uncertain_regions": [],
    "confidence": 0.95,
}


def seen(student_work: list[str]) -> dict:
    """What the VLM reads off the page this time round."""
    return dict(PROBLEM, student_work=student_work)


class PhoneWS:
    """A device with a camera on the session socket, like tutor/web/phone.html."""

    def __init__(self, jpeg: bytes | None = PHONE_JPEG):
        self.events: list[dict] = []
        self.jpeg = jpeg
        self.captures = 0
        self.session: Session | None = None

    async def send(self, raw):
        if not isinstance(raw, str):
            return
        event = json.loads(raw)
        self.events.append(event)
        if event["event"] != "capture_request":
            return
        self.captures += 1
        capture_id = event["data"]["capture_id"]
        # the phone answers on the same socket the request arrived on
        await self.session.on_frame(
            encode_image(self.jpeg, ImageHeader(capture_id=capture_id))
        )

    def event_names(self) -> list[str]:
        return [e["event"] for e in self.events]


class ScriptedTranscriber:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000) -> Transcript:
        return Transcript(text=self.texts.pop(0) if self.texts else "")


def build(db, *utterances: str, llm_responses: dict | None = None):
    llm = EchoLLMClient(llm_responses or {})
    speaker = NullSpeaker()
    ws = PhoneWS()
    deps = Deps(
        settings=Settings(capture_timeout_s=2.0),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(*utterances),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        classifier=IntentClassifier(),  # rules only: no LLM guessing in tests
        store=SessionStore(),
    )
    session = Session(ws, deps)
    ws.session = session
    session.ctx = ProblemContext(
        hash="p1",
        recognition=Recognition(**seen(["3*x = 20 + 5"])),
        match=MatchResult(tier=Tier.EXACT, concepts=["linear_equation"], reference=REFERENCE),
        reference=REFERENCE,
    )
    return session, llm, speaker, ws


def ask_l1(session: Session, step: int = 1) -> int:
    """Pretend the tutor just asked its L1 question and is waiting."""
    session.store.set_state(
        StudentState(status="CONCEPT_ERROR", last_correct_step=step - 1,
                     misconception="sign_flip_on_move")
    )
    return session.store.append_hint(
        problem_hash="p1", step=step, level=1, action="SOCRATIC_QUESTION",
        hint_text="어떤 항을 반대쪽으로 옮겨야 할까요?",
    )


# --- the fix ----------------------------------------------------------------


async def test_a_work_check_takes_a_fresh_photo_and_reads_it(db):
    """The whole point: 'is this right?' must look at what they just wrote."""
    session, llm, speaker, ws = build(
        db, "이렇게 하는 거 맞아?", llm_responses={"recognize": [seen(["3*x = 15"])]}
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1
    assert llm.calls.count("recognize") == 1
    assert llm.calls.count("evaluate") == 0  # nothing was answered, so nothing graded
    assert speaker.spoken
    # the newly written line is what the diagnosis ran on
    assert session.prev_work == ["3*x = 15"]


async def test_a_work_check_survives_a_pending_question(db):
    """A hint is pending — the old code graded this as an answer instead."""
    session, llm, speaker, ws = build(db, "여기서 뭐가 틀렸어?")
    ask_l1(session)
    assert session.store.pending_hint("p1") is not None

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1 and llm.calls.count("recognize") == 1


async def test_a_work_check_reuses_the_problem_it_is_already_on(db):
    """Same worksheet: no re-tagging, no re-matching, no second solve."""
    session, llm, _, _ = build(db, "내가 쓴 거 봐줘.")
    ask_l1(session)
    before = session.ctx

    await session._handle_utterance(PCM, 16000)

    assert session.ctx is before          # same context object
    assert session.ctx.reference is REFERENCE
    assert llm.calls.count("solve") == 0
    assert llm.calls.count("tag") == 0


async def test_correct_work_is_confirmed_and_not_hinted_at(db):
    """'풀이 맞아?' is a yes/no question. When the answer is yes, say yes.

    A hint here would push them at a step they have not reached and imply
    something was wrong — and it must not enter the hint ladder either.
    """
    session, llm, speaker, _ = build(
        db,
        "이렇게 하는 거 맞아?",
        llm_responses={
            "recognize": [seen(["3*x = 15"])],
            "estimate": [{"current_step": "이항", "last_correct_step": 1,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": True}],
        },
    )
    before = len(session.store.get_history(problem_hash="p1"))
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert speaker.spoken == ["맞아요! 이대로 하면 돼요. 또 궁금한 게 있으면 물어봐 주세요."]
    assert llm.calls.count("phrase") == 0          # no hint was generated
    # and no hint record: the ladder must not move on a question that was answered
    assert len(session.store.get_history(problem_hash="p1")) == before + 1  # only ask_l1's
    assert session.ctx is not None                 # still on the same problem


async def test_a_wrong_line_is_acknowledged_without_naming_the_mistake(db):
    session, _, speaker, _ = build(
        db,
        "풀이 맞나요?",
        llm_responses={
            "recognize": [seen(["3*x = 25"])],
            "estimate": [{"current_step": "이항", "last_correct_step": 0,
                          "status": "CONCEPT_ERROR", "misconception": "sign_flip_on_move",
                          "attempt_count": 2, "previous_hint_effective": False}],
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    spoken = speaker.spoken[0]
    assert spoken.startswith("음, 지금 쓴 줄을 같이 볼까요?")
    assert "3*x = 15" not in spoken and "x = 5" not in spoken


# --- what must NOT change ---------------------------------------------------


async def test_a_spoken_answer_still_skips_the_camera(db):
    """The latency contract: an answer is graded from the transcript alone."""
    session, llm, speaker, ws = build(
        db,
        "5예요",
        llm_responses={
            "evaluate": [{"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
                          "misconception": None, "status": "CORRECT"}]
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 0
    assert "capture_request" not in ws.event_names()
    assert llm.calls.count("recognize") == 0 and llm.calls.count("estimate") == 0
    assert llm.calls.count("evaluate") == 1


async def test_a_plain_hint_request_does_not_grow_a_reaction(db):
    """Only a work check earns the 'here is what I saw' opener.

    No pending question here: "힌트 주세요" while one IS pending is the student
    declining to answer, which the evaluator already reads as "escalate".
    """
    session, _, speaker, ws = build(db, "힌트 주세요")
    session.store.set_state(StudentState(status="STUCK"))

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1
    assert not speaker.spoken[0].startswith("네, 여기까지는")
    assert not speaker.spoken[0].startswith("음, 지금 쓴 줄을")


async def test_thinking_out_loud_is_left_alone(db):
    """Spec rule 7: do not interrupt a student who is working."""
    session, llm, speaker, ws = build(db, "음 그러니까 이제 이걸")
    session.store.set_state(StudentState(status="CORRECT", last_correct_step=1))

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 0
    assert speaker.spoken == []
    assert llm.calls == []
    assert session.last_transcript is None  # not evidence for the next turn either
    # the device is told a transcript arrived, but that no reply is coming
    transcript = next(e for e in ws.events if e["event"] == "transcript")
    assert transcript["data"]["wants_hint"] is False


# --- the evaluator's safety net ---------------------------------------------


async def test_the_evaluator_can_redirect_an_answer_to_a_work_check(db):
    """A phrasing the keywords miss still ends up looking at the page."""
    session, llm, speaker, ws = build(
        db,
        "여기 이거 어떡해요",   # no work-check keyword, so it routes as an answer
        llm_responses={
            "evaluate": [{"intent": "WORK_CHECK", "verdict": "UNCLEAR", "feedback": "",
                          "misconception": None, "status": None}],
            "recognize": [seen(["3*x = 15"])],
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert llm.calls.count("evaluate") == 1   # graded first...
    assert ws.captures == 1                   # ...then redirected to the camera
    assert llm.calls.count("recognize") == 1
    # the turn is decided by the page, not by the discarded UNCLEAR verdict:
    # the photo shows step 1 done, so that is the state and the hint that got
    # them there is resolved as having worked
    assert session.store.get_state().last_correct_step == 1
    assert session.store.get_history(problem_hash="p1")[0].effective is True
