"""The answer turn: the student replies to the tutor's question out loud.

Covers the ladder the policy is supposed to produce — correct → next step L1,
wrong → same step L2, unclear → same step, same level re-asked — and the
latency contract that makes it usable: an answer never re-captures or
re-recognizes the worksheet.
"""

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.server.session import Deps, ProblemContext, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

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

RECOGNITION = Recognition(
    problem_text="다음 일차방정식을 푸시오: 3x + 5 = 20",
    equations=["3*x + 5 = 20"],
    student_work=["3*x = 20 + 5"],
    confidence=0.95,
)


class FakeWS:
    """Collects what the server sends; raises if anything asks for a capture."""

    def __init__(self):
        self.events: list[str] = []

    async def send(self, raw):
        if isinstance(raw, str):
            self.events.append(raw)

    def event_names(self) -> list[str]:
        import json

        return [json.loads(e)["event"] for e in self.events if isinstance(e, str)]


def build_session(db, verdicts: list[dict]) -> tuple[Session, EchoLLMClient, NullSpeaker]:
    llm = EchoLLMClient({"evaluate": verdicts})
    speaker = NullSpeaker()
    deps = Deps(
        settings=Settings(),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        store=SessionStore(),
    )
    session = Session(FakeWS(), deps)
    session.ctx = ProblemContext(
        hash="p1",
        recognition=RECOGNITION,
        match=MatchResult(tier=Tier.EXACT, concepts=["linear_equation"], reference=REFERENCE),
        reference=REFERENCE,
    )
    return session, llm, speaker


def ask_l1(session: Session, step: int = 1) -> int:
    """Pretend the tutor just asked its L1 question at `step`."""
    session.store.set_state(
        StudentState(status="CONCEPT_ERROR", last_correct_step=step - 1,
                     misconception="sign_flip_on_move")
    )
    return session.store.append_hint(
        problem_hash="p1", step=step, level=1, action="SOCRATIC_QUESTION",
        hint_text="어떤 항을 반대쪽으로 옮겨야 할까요?",
    )


async def test_correct_answer_advances_to_next_step_l1(db):
    session, llm, speaker = build_session(
        db, [{"verdict": "CORRECT", "feedback": "맞아요!", "misconception": None, "status": "CORRECT"}]
    )
    hint_id = ask_l1(session)

    await session.handle_answer("5를 빼면 돼요", session.store.pending_hint("p1"))

    history = session.store.get_history(problem_hash="p1")
    assert history[0].id == hint_id and history[0].effective is True
    issued = history[-1]
    assert (issued.step, issued.level) == (2, 1)  # next step, weakest hint again
    assert session.store.get_state().last_correct_step == 1
    assert speaker.spoken and "맞아요" in speaker.spoken[0]
    # no capture, no vision, no diagnosis — only the answer evaluation ran
    assert "capture_request" not in session.ws.event_names()
    assert llm.calls.count("recognize") == 0 and llm.calls.count("estimate") == 0
    assert llm.calls.count("evaluate") == 1


async def test_wrong_answer_escalates_same_step_to_l2(db):
    session, llm, speaker = build_session(
        db,
        [{"verdict": "INCORRECT", "feedback": "음, 조금 달라요.",
          "misconception": "sign_flip_on_move", "status": "CONCEPT_ERROR"}],
    )
    ask_l1(session)

    await session.handle_answer("그냥 5를 더하면 돼요", session.store.pending_hint("p1"))

    history = session.store.get_history(problem_hash="p1")
    assert history[0].effective is False
    issued = history[-1]
    assert (issued.step, issued.level) == (1, 2)  # same step, one level stronger
    assert issued.action == "CONCEPT_HINT"
    assert session.store.get_state().last_correct_step == 0
    assert "capture_request" not in session.ws.event_names()


async def test_unclear_answer_reasks_same_level(db):
    session, llm, speaker = build_session(
        db, [{"verdict": "UNCLEAR", "feedback": "잘 못 들었어요.", "misconception": None, "status": None}]
    )
    ask_l1(session)

    await session.handle_answer("어... 음...", session.store.pending_hint("p1"))

    history = session.store.get_history(problem_hash="p1")
    assert history[0].effective is None  # unresolved: no evidence either way
    issued = history[-1]
    assert (issued.step, issued.level) == (1, 1)  # same step, same level
    # re-asked, not repeated verbatim
    assert issued.hint_text != history[0].hint_text


async def test_dont_know_escalates_like_a_wrong_answer(db):
    session, llm, _ = build_session(
        db, [{"verdict": "INCORRECT", "feedback": "괜찮아요.", "misconception": None, "status": "STUCK"}]
    )
    ask_l1(session)
    await session.handle_answer("모르겠어요", session.store.pending_hint("p1"))
    assert session.store.get_history(problem_hash="p1")[-1].level == 2


async def test_three_wrong_answers_climb_one_level_each(db):
    verdict = {"verdict": "INCORRECT", "feedback": "", "misconception": None, "status": "CONCEPT_ERROR"}
    session, _, _ = build_session(db, [dict(verdict) for _ in range(3)])
    ask_l1(session)

    levels = []
    for _ in range(3):
        await session.handle_answer("모르겠어요", session.store.pending_hint("p1"))
        levels.append(session.store.get_history(problem_hash="p1")[-1].level)
    assert levels == [2, 3, 4]  # never skips, caps at L4


async def test_feedback_that_leaks_the_answer_is_dropped(db):
    session, _, speaker = build_session(
        db,
        [{"verdict": "CORRECT", "feedback": "맞아요, 답은 x = 5예요!",
          "misconception": None, "status": "CORRECT"}],
    )
    ask_l1(session)
    await session.handle_answer("5를 빼요", session.store.pending_hint("p1"))
    assert speaker.spoken
    assert "x = 5" not in speaker.spoken[0]


async def test_utterance_without_pending_hint_is_not_an_answer(db):
    """Nothing was asked: an idle remark must not be graded."""
    session, llm, _ = build_session(db, [])
    session.store.set_state(StudentState(status="STUCK"))
    await session._handle_utterance(b"\x00\x00" * 100, 16000)
    assert llm.calls.count("evaluate") == 0
