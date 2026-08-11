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


def build_session(
    db, verdicts: list[dict], client=EchoLLMClient
) -> tuple[Session, EchoLLMClient, NullSpeaker]:
    llm = client({"evaluate": verdicts})
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


class TestProblemCompletion:
    """Getting the LAST step right ends the problem, it does not hint again."""

    async def test_last_step_correct_closes_the_problem(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요, 그렇게 하면 돼요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=2)  # REFERENCE has 2 steps: this is the last one
        before = len(session.store.get_history(problem_hash="p1"))

        await session.handle_answer("양변을 3으로 나눠요", session.store.pending_hint("p1"))

        assert speaker.spoken == [
            "맞아요, 그렇게 하면 돼요! 문제를 끝까지 풀었네요! 또 모르는 문제가 있으면 알려주세요."
        ]
        # no _deliver(): no new hint record and no hint_issued event
        history = session.store.get_history(problem_hash="p1")
        assert len(history) == before
        assert "hint_issued" not in session.ws.event_names()
        assert history[-1].effective is True  # the hint that got them there worked
        # the problem is over
        assert session.ctx is None
        assert session.store.get_state() is None
        assert session.prev_work is None
        # and the page is told, so it can file the conversation on the left
        import json
        solved = next(
            e for e in map(json.loads, session.ws.events) if e["event"] == "solved"
        )
        assert solved["data"]["text"]          # the problem, for the archive title

    async def test_a_middle_step_still_continues_the_dialogue(self, db):
        session, _, speaker = build_session(
            db, [{"verdict": "CORRECT", "feedback": "맞아요!", "misconception": None,
                  "status": "CORRECT"}]
        )
        ask_l1(session, step=1)  # not the last step
        await session.handle_answer("5를 빼요", session.store.pending_hint("p1"))

        assert session.ctx is not None  # still working on it
        assert "hint_issued" in session.ws.event_names()
        assert session.store.get_history(problem_hash="p1")[-1].step == 2
        assert "또 모르는 문제가 있으면" not in speaker.spoken[0]
        assert "solved" not in session.ws.event_names()  # a middle step is not a win

    async def test_completion_feedback_that_leaks_is_dropped(self, db):
        session, _, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요, 답은 x = 5예요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=2)
        await session.handle_answer("나누면 돼요", session.store.pending_hint("p1"))

        assert speaker.spoken == ["문제를 끝까지 풀었네요! 또 모르는 문제가 있으면 알려주세요."]

    async def test_after_completion_speech_is_not_graded_as_an_answer(self, db):
        session, llm, _ = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요!", "misconception": None,
              "status": "CORRECT"}],
        )
        ask_l1(session, step=2)
        await session.handle_answer("나누면 돼요", session.store.pending_hint("p1"))

        graded = llm.calls.count("evaluate")
        await session._handle_utterance(b"\x00\x00" * 100, 16000)
        assert llm.calls.count("evaluate") == graded  # nothing left to answer


class TestOneAcknowledgementPerTurn:
    """The evaluator reacts; the hint must not react again."""

    def test_strip_only_removes_a_real_opener(self):
        from tutor.hints.generator import strip_leading_acknowledgement as strip

        assert strip("네, 그렇게 하면 돼요.") == "그렇게 하면 돼요."
        assert strip("맞아요! 이제 3을 어떻게 할까요?") == "이제 3을 어떻게 할까요?"
        assert strip("네, 맞아요! 다음은요?") == "다음은요?"   # doubled openers
        assert strip("음... 어떤 항을 옮길까요?") == "어떤 항을 옮길까요?"
        # words that merely start with an ack syllable must survive
        assert strip("네 번째 항을 보세요.") == "네 번째 항을 보세요."
        assert strip("아래쪽 식을 보세요.") == "아래쪽 식을 보세요."
        assert strip("오른쪽으로 옮겨 볼까요?") == "오른쪽으로 옮겨 볼까요?"
        assert strip("맞아요!") == "맞아요!"  # never returns empty

    async def test_the_tutor_says_it_once(self, db):
        """The reaction and the hint are now two utterances — the reaction
        plays while the hint generates — but the rule holds across them:
        one acknowledgement per turn, so the hint's own opener is stripped."""
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요, 그렇게 하면 돼요!",
              "misconception": None, "status": "CORRECT"}],
        )
        # force the LLM phrasing path and have it echo an acknowledgement back
        llm._queues["phrase"] = [{"hint": "네, GOOD 글자들은 그렇게 하면 돼요. 다음은 무엇일까요?"}]
        session.ctx.match = MatchResult(tier=Tier.NEW, concepts=["nothing_seeded"],
                                        reference=REFERENCE)
        session.ctx.recognition = Recognition(problem_text="문제")
        ask_l1(session, step=1)

        await session.handle_answer("5를 빼요", session.store.pending_hint("p1"))

        assert speaker.spoken[0] == "맞아요, 그렇게 하면 돼요!"   # the reaction, alone
        hint = speaker.spoken[1]
        assert not hint.startswith("네")                          # its opener is gone
        assert "GOOD 글자들은 그렇게 하면 돼요" in hint            # the content is kept


class TestStudentQuestions:
    """'왜 그렇게 해요?' is a question, not a wrong answer."""

    def _asking(self, feedback="", verdict="UNCLEAR"):
        return {"intent": "QUESTION", "verdict": verdict, "feedback": feedback,
                "misconception": None, "status": None}

    async def test_a_question_is_explained_not_graded(self, db):
        session, llm, speaker = build_session(db, [self._asking()])
        hint_id = ask_l1(session, step=1)
        before = session.store.get_state()

        await session.handle_answer("왜 5를 빼야 해요?", session.store.pending_hint("p1"))

        assert llm.calls == ["evaluate", "explain"]
        assert speaker.spoken and speaker.spoken[0]
        # nothing was proven: the question is still open at the same level
        history = session.store.get_history(problem_hash="p1")
        assert len(history) == 1 and history[0].id == hint_id
        assert history[0].effective is None
        assert session.store.pending_hint("p1") is not None
        assert session.store.get_state() == before  # no state change
        assert "hint_issued" not in session.ws.event_names()

    async def test_the_explanation_does_not_double_up_on_feedback(self, db):
        session, llm, speaker = build_session(db, [self._asking(feedback="네, 궁금하시죠?")])
        ask_l1(session, step=1)
        await session.handle_answer("왜 나눠요?", session.store.pending_hint("p1"))
        # a question turn speaks the explanation only — no evaluator reaction
        assert not speaker.spoken[0].startswith("네, 궁금하시죠?")

    async def test_where_did_i_go_wrong_is_answered_from_the_diagnosis(self, db):
        """The live gap: "어디가 틀렸어?" reached explain(), but explain had
        only the problem and the step — so it motivated the step in the
        abstract and never said WHERE. The diagnosis the tutor already made
        travels with the question now."""
        prompts: list[str] = []

        class Recording(EchoLLMClient):
            def run_with_tools(self, *, purpose, system, user, images=(), schema,
                               max_rounds=6):
                if purpose == "explain":
                    prompts.append(user)
                return super().run_with_tools(
                    purpose=purpose, system=system, user=user, images=images,
                    schema=schema, max_rounds=max_rounds,
                )

        session, llm, _ = build_session(db, [self._asking()], client=Recording)
        ask_l1(session, step=1)
        session.store.set_state(StudentState(
            status="CONCEPT_ERROR", last_correct_step=0,
            misconception="sign_flip_on_move",
        ))

        await session.handle_answer("어디가 틀렸어요?", session.store.pending_hint("p1"))

        assert prompts and "sign_flip_on_move" in prompts[0]

    async def test_answering_after_the_explanation_still_lands(self, db):
        session, llm, _ = build_session(
            db,
            [self._asking(),
             {"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=1)
        await session.handle_answer("왜 5를 빼요?", session.store.pending_hint("p1"))
        await session.handle_answer("5를 빼면 돼요", session.store.pending_hint("p1"))

        history = session.store.get_history(problem_hash="p1")
        assert history[0].effective is True      # the original question resolved
        assert history[-1].step == 2             # and we moved on
