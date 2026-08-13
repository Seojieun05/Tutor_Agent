"""A turn the student talked over never happened, as far as the ladder knows.

Live, on problem 13: the STT misheard "-2x-4" as "-2x 빼기 3", the tutor read
it back, and the student cut in to correct it. The turn it interrupted had
already graded the misheard answer INCORRECT, resolved L1 as a failure, and
generated an L2 hint — which the barge-in then dropped, unspoken. It was
recorded anyway, so the next turn escalated from it: the student went L1 → L3
having heard only L1, for two mistakes the microphone made.

A hint that was never said is not a hint that was given, and the question it
was going to replace is still the one on the table.
"""

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.policy.engine import Action, Decision
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

HINT = "거듭제곱 법칙을 떠올려 보세요."
PROBLEM = "problem-13"


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


class TalkedOver(Session):
    """A session whose every line is dropped by a barge-in already in progress
    — what BrowserSession._say does when `_interrupted` is set before it runs."""

    silenced = True

    async def _say(self, text: str):
        if self.silenced:
            return False
        return None  # spoke it (the base session returns None too)


@pytest.fixture
def session(db):
    llm = EchoLLMClient()
    deps = Deps(
        settings=Settings(filler_enabled=False),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(),
        store=SessionStore(),
    )
    return TalkedOver(FakeWS(), deps)


def decision(level: int = 2) -> Decision:
    return Decision(Action.CONCEPT_HINT, level, 3, None, "test")


def first_hint(session) -> int:
    """The L1 the student actually heard, and answered."""
    return session.store.append_hint(
        problem_hash=PROBLEM, step=3, level=1,
        action="SOCRATIC_QUESTION", hint_text="기울기를 어떻게 구할까요?",
    )


async def test_a_hint_nobody_heard_is_not_recorded(session):
    first_hint(session)

    await session._deliver(decision(), HINT, PROBLEM)

    levels = [h.level for h in session.store.get_history(problem_hash=PROBLEM)]
    assert levels == [1], "an unspoken L2 entered the history"


async def test_the_question_it_answered_goes_back_on_the_table(session):
    """The L1 was resolved as a failure by the same turn. From where the
    student sits it is still the question being asked."""
    hint_id = first_hint(session)
    session._resolve_hint(hint_id, False)
    assert session.store.pending_hint(PROBLEM) is None  # resolved

    await session._deliver(decision(), HINT, PROBLEM)

    pending = session.store.pending_hint(PROBLEM)
    assert pending is not None and pending.id == hint_id
    assert pending.effective is None


async def test_the_next_turn_asks_at_the_same_level_not_the_next_one(session):
    """The whole point: no escalation off a hint that was never given."""
    from tutor.policy.engine import decide
    from tutor.state.models import StudentState

    hint_id = first_hint(session)
    session._resolve_hint(hint_id, False)
    await session._deliver(decision(), HINT, PROBLEM)

    again = decide(
        StudentState(status="CONCEPT_ERROR", last_correct_step=2),
        session.store.get_history(problem_hash=PROBLEM),
        "HINT_REQUEST",
    )
    assert again.level == 1, again


async def test_a_hint_the_student_heard_is_recorded_as_always(session):
    """The guard is for silence, not for interruption: a line cut off halfway
    was still said, and still asked."""
    session.silenced = False
    hint_id = first_hint(session)
    session._resolve_hint(hint_id, False)

    await session._deliver(decision(), HINT, PROBLEM)

    levels = [h.level for h in session.store.get_history(problem_hash=PROBLEM)]
    assert levels == [1, 2]
    assert session.store.pending_hint(PROBLEM).level == 2


async def test_a_dropped_turn_that_resolved_nothing_leaves_history_alone(session):
    """A first hint, talked over: nothing to put back, nothing to corrupt."""
    await session._deliver(decision(1), HINT, PROBLEM)

    assert session.store.get_history(problem_hash=PROBLEM) == []
    assert session.store.pending_hint(PROBLEM) is None
