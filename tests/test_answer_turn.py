"""The answer turn: the student replies to the tutor's question out loud.

Covers the ladder the policy is supposed to produce — correct → next step L1,
wrong → same step L2, unclear → same step, same level re-asked — and the
latency contract that makes it usable: an answer never re-captures or
re-recognizes the worksheet.
"""

import asyncio
import json

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


async def test_answer_that_arrives_during_turn_cleanup_is_queued(db):
    session, _, _ = build_session(db, [])
    ask_l1(session)
    pending = session.store.pending_hint("p1")
    handled = []

    async def record_answer(transcript, hint):
        handled.append((transcript, hint.id))

    session._handle_answer = record_answer
    # The previous hint has stopped speaking but has not cleared its turn yet.
    session._busy = True
    session._turn_idle.clear()

    answer = asyncio.create_task(session.handle_answer("2x-4예요", pending))
    await asyncio.sleep(0)
    assert handled == []
    assert session._turn_waiters == 1

    session._busy = False
    session._turn_idle.set()
    await answer

    assert handled == [("2x-4예요", pending.id)]
    assert session._turn_waiters == 0
    assert session._busy is False


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


async def test_a_verified_tangent_is_drawn_before_the_next_question_is_built(db):
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="접선의 기울기 구하기",
                         expression="f'(1) = -2"),
            SolutionStep(idx=2, description="l의 방정식 구하기",
                         expression="l: y = -2*x - 4"),
            SolutionStep(idx=3, description="다음 접선 구하기",
                         expression="m: y = -4*x + 10"),
            SolutionStep(idx=4, description="넓이 구하기", expression="49"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    session, _, _ = build_session(db, [{
        "verdict": "CORRECT", "feedback": "맞아요!",
        "misconception": None, "status": "CORRECT",
    }])
    session.ctx.reference = reference
    session.ctx.match = MatchResult(
        tier=Tier.EXACT, concepts=reference.concepts, reference=reference,
    )
    ask_l1(session, step=2)

    await session.handle_answer(
        "l은 y는 마이너스 2x 마이너스 4예요",
        session.store.pending_hint("p1"),
    )
    await asyncio.gather(*[t for t in session._tasks if not t.done()])

    events = [json.loads(raw) for raw in session.ws.events]
    figure_at = next(i for i, e in enumerate(events) if e["event"] == "figure")
    next_question_at = next(
        i for i, e in enumerate(events)
        if e["event"] == "stage" and e["data"]["text"] == "다음 질문을 만들고 있어요"
    )
    assert figure_at < next_question_at
    assert "-2·x - 4" in events[figure_at]["data"]["svg"]


async def test_partial_answer_stays_on_the_same_step_and_gets_a_new_question(db):
    session, llm, speaker = build_session(
        db,
        [{"verdict": "PARTIAL", "feedback": "맞아요, 방향은 잘 잡았어요.",
          "misconception": None, "status": "STUCK"}],
    )
    first = ask_l1(session)

    await session.handle_answer(
        "식을 정리하면 될 것 같아요", session.store.pending_hint("p1")
    )

    history = session.store.get_history(problem_hash="p1")
    assert history[0].id == first and history[0].effective is True
    assert (history[-1].step, history[-1].level) == (1, 1)
    state = session.store.get_state()
    assert state.last_correct_step == 0 and state.status == "STUCK"
    assert speaker.spoken and "여기까지는 잘했어요" in speaker.spoken[0]
    assert llm.calls.count("recognize") == 0 and llm.calls.count("estimate") == 0


async def test_problem_13_derivative_partial_fades_l2_and_asks_only_for_slope(db):
    """Exact 16:23 regression: f'(x)=2x-4 is real progress inside step 1.
    It must not repeat the derivative concept at L2 or fall back generically;
    the remaining question is only f'(1)."""
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*x - 4"),
            SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = u'(x)*v(x) + u(x)*v'(x)"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    session, _, speaker = build_session(
        db,
        [{"verdict": "PARTIAL", "feedback": "맞아요, 도함수를 먼저 구하면 돼요.",
          "misconception": None, "status": "STUCK"}],
    )
    session.ctx.reference = reference
    session.ctx.match = MatchResult(
        tier=Tier.EXACT, concepts=reference.concepts, reference=reference
    )
    session.store.set_state(
        StudentState(status="STUCK", last_correct_step=0, misconception=None)
    )
    old = session.store.append_hint(
        problem_hash="p1", step=1, level=2, action="CONCEPT_HINT",
        hint_text="도함수의 뜻을 떠올려 볼까요?",
    )

    await session.handle_answer(
        "그러니까 f 2분하면 2x 마이너스 4.", session.store.pending_hint("p1")
    )

    history = session.store.get_history(problem_hash="p1")
    assert next(h for h in history if h.id == old).effective is True
    latest = history[-1]
    assert (latest.step, latest.level, latest.action) == (1, 1, "SOCRATIC_QUESTION")
    assert latest.hint_text == (
        "이제 구한 f'(x)를 이용해 접선 l의 기울기를 구해 볼까요?"
    )
    assert "x = 1" not in latest.hint_text
    assert "가장 확실한 줄" not in latest.hint_text
    assert speaker.spoken and "잘 구했어요" in speaker.spoken[0]


def test_future_plan_does_not_complete_problem_13s_composite_slope_step():
    reference = ReferenceSolution(
        steps=[
            SolutionStep(
                idx=1,
                description="f'(x)로 접선 l의 기울기 구하기",
                expression="f'(x) = 2*x - 4, f'(1) = -2",
            ),
            SolutionStep(
                idx=2,
                description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                expression="l: y = -2*(x - 1) - 6 = -2*x - 4",
            ),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"],
        verified=True,
        origin="db",
    )
    llm = EchoLLMClient({"evaluate": [{
        "verdict": "CORRECT", "feedback": "맞아요, 그렇게 구하면 돼요!",
        "misconception": None, "status": "CORRECT",
    }]})

    verdict = AnswerEvaluator(llm).evaluate(
        problem_text="접선 l을 구하시오",
        reference=reference,
        question="접선 l의 기울기는 어떻게 구할까요?",
        target_step=1,
        transcript="f를 미분하면 2x-4니까 거기에 1을 대입하면 될 것 같은데?",
    )

    assert verdict.verdict == "PARTIAL"
    assert "f'(x)는 잘 구했어요" in verdict.feedback


async def test_problem_13_partial_slope_talk_never_jumps_to_product_rule(db):
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*(x - 1) - 6 = -2*x - 4"),
            SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    session, llm, speaker = build_session(
        db,
        [{"verdict": "CORRECT", "feedback": "맞아요, 그렇게 구하면 돼요!",
          "misconception": None, "status": "CORRECT"}],
    )
    session.ctx.reference = reference
    session.ctx.match = MatchResult(
        tier=Tier.EXACT, concepts=reference.concepts, reference=reference
    )
    ask_l1(session, step=2)

    await session.handle_answer(
        "f를 미분하면 2x-4니까 거기에 1을 대입하면 될 것 같은데?",
        session.store.pending_hint("p1"),
    )

    state = session.store.get_state()
    latest = session.store.get_history(problem_hash="p1")[-1]
    assert state.last_correct_step == 1
    assert (latest.step, latest.level) == (2, 1)
    assert "곱의 미분법" not in " ".join(speaker.spoken)

    # Once l itself is actually complete, step 3 may open — as a question
    # about the product structure, never as an internal step announcement.
    llm._queues.setdefault("evaluate", []).append({
        "verdict": "CORRECT", "feedback": "맞아요!",
        "misconception": None, "status": "CORRECT",
    })
    await session.handle_answer(
        "l은 y = -2x - 4예요", session.store.pending_hint("p1")
    )

    latest = session.store.get_history(problem_hash="p1")[-1]
    assert session.store.get_state().last_correct_step == 2
    assert (latest.step, latest.level) == (3, 1)
    assert latest.hint_text == \
        "이제 g(x)가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?"
    assert "곱의 미분법" not in latest.hint_text and "차례" not in latest.hint_text


async def test_problem_13_whole_tangent_answer_skips_the_repeated_line_question(db):
    """Exact 16:05 regression: the stored question was tagged step 1 even
    though it asked for step 2. The spoken tangent itself proves step 2, so the
    next question must open step 3 rather than ask for l again."""
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*(x - 1) - 6 = -2*x - 4"),
            SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"),
            SolutionStep(idx=4, description="g'(1) 계산", expression="g'(1) = -4"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    session, _, _ = build_session(
        db,
        [{"verdict": "CORRECT", "feedback": "맞아요!",
          "misconception": None, "status": "CORRECT"}],
    )
    session.ctx.reference = reference
    session.ctx.match = MatchResult(
        tier=Tier.EXACT, concepts=reference.concepts, reference=reference
    )
    ask_l1(session, step=1)

    await session.handle_answer(
        "마이너스 2x 마이너스 4, 맞아?", session.store.pending_hint("p1")
    )

    state = session.store.get_state()
    latest = session.store.get_history(problem_hash="p1")[-1]
    assert state.last_correct_step == 2
    assert (latest.step, latest.level) == (3, 1)
    assert latest.hint_text == \
        "이제 g(x)가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?"
    assert "l의 방정식" not in latest.hint_text


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


async def test_wrong_spoken_equation_is_diagnosed_before_the_review_question(db):
    class RecordingCorrectionLLM(EchoLLMClient):
        def __init__(self, responses):
            scripted = dict(responses)
            scripted["phrase"] = [{
                "hint": (
                    "방금 말한 2x 마이너스 4는 앞에서 구한 도함수예요. "
                    "지금 구하는 대상과 같은 것인지 다시 살펴볼까요?"
                )
            }]
            super().__init__(scripted)
            self.phrase_prompt = ""
            self.phrase_stream_calls = 0

        def run_with_tools(
            self, *, purpose, system, user, images=(), schema, max_rounds=6
        ):
            if purpose == "phrase":
                self.phrase_prompt = user
            return super().run_with_tools(
                purpose=purpose, system=system, user=user, images=images,
                schema=schema, max_rounds=max_rounds,
            )

        def complete_json_stream(self, **kwargs):
            if kwargs.get("purpose") == "phrase":
                self.phrase_stream_calls += 1
            return super().complete_json_stream(**kwargs)

    reference = ReferenceSolution(
        steps=[
            SolutionStep(
                idx=1,
                description="f'(x)로 접선 l의 기울기 구하기",
                expression="f'(x) = 2*x - 4, f'(1) = -2",
            ),
            SolutionStep(
                idx=2,
                description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                expression="l: y = -2*x - 4",
            ),
            SolutionStep(idx=3, description="g'(x) 쓰기", expression="g'(x) = 0"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    session, llm, speaker = build_session(
        db,
        [{
            "verdict": "INCORRECT",
            "feedback": "조금 달라요. 방금 말한 식은 앞에서 구한 도함수예요.",
            "error_focus": "2x-4를 접선의 방정식으로 혼동함; 앞에서 구한 f'(x)임",
            "misconception": None,
            "status": "PROCEDURAL_ERROR",
        }],
        client=RecordingCorrectionLLM,
    )
    session.ctx.reference = reference
    session.ctx.match = MatchResult(
        tier=Tier.NEW, concepts=reference.concepts, reference=reference,
    )
    session.store.set_state(StudentState(
        status="STUCK", last_correct_step=1, misconception=None,
    ))
    session.store.append_hint(
        problem_hash="p1", step=2, level=2, action="CONCEPT_HINT",
        hint_text=(
            "기울기가 -2인 직선은 다음과 같은 꼴로 나타낼 수 있어요. "
            "접점 (1, -6)을 이 식에 넣어 볼까요?"
        ),
    )

    await session.handle_answer(
        "그럼 2x 마이너스 4 맞아?", session.store.pending_hint("p1")
    )

    assert speaker.spoken[0] == \
        "조금 달라요. 방금 말한 식은 앞에서 구한 도함수예요."
    assert "다시 살펴볼까요" in speaker.spoken[-1]
    assert "답변 판정기가 특정한 오류" in llm.phrase_prompt
    assert "2x-4를 접선의 방정식으로 혼동" in llm.phrase_prompt
    assert llm.phrase_stream_calls == 0


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


class TestSecondOpinionBeforeSayingWrong:
    """Grading a right answer wrong is the most expensive mistake here, and
    measurably the one the small routed model makes: on a student who answered
    a slope question with the whole tangent, flash-lite said INCORRECT in 2.0s
    where flash and grok both said CORRECT. So "wrong" — and only "wrong" —
    is checked twice."""

    def _evaluator(self, first, second=None):
        from tutor.state.answer import AnswerEvaluator

        return AnswerEvaluator(
            EchoLLMClient({"evaluate": [first]}),
            None,
            second_opinion=EchoLLMClient({"evaluate": [second]}) if second else None,
        )

    def _ask(self, ev):
        return ev.evaluate(
            problem_text="p", reference=REFERENCE,
            question="기울기가 얼마일까요?", target_step=1, transcript="L은 -2x - 4",
        )

    WRONG = {"intent": "ANSWER", "verdict": "INCORRECT", "feedback": "음, 조금 달라요.",
             "misconception": None, "status": "CONCEPT_ERROR"}
    RIGHT = {"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
             "misconception": None, "status": "CORRECT"}

    def test_a_wrong_verdict_is_overturned_when_the_answer_was_right(self):
        verdict = self._ask(self._evaluator(self.WRONG, self.RIGHT))
        assert verdict.verdict == "CORRECT"
        assert verdict.feedback == "맞아요!"          # and its reaction travels with it

    def test_when_the_second_look_agrees_the_first_verdict_stands(self):
        verdict = self._ask(self._evaluator(self.WRONG, dict(self.WRONG)))
        assert verdict.verdict == "INCORRECT"
        assert verdict.status == "CONCEPT_ERROR"     # the diagnosis survives

    def test_a_correct_answer_never_pays_for_a_second_look(self):
        second = EchoLLMClient({"evaluate": [self.WRONG]})
        from tutor.state.answer import AnswerEvaluator

        ev = AnswerEvaluator(EchoLLMClient({"evaluate": [self.RIGHT]}), None,
                             second_opinion=second)
        assert self._ask(ev).verdict == "CORRECT"
        assert second.calls == []                    # not consulted at all

    def test_a_question_is_not_re_graded(self):
        """QUESTION and WORK_CHECK ignore the verdict entirely, so a second
        look would be spent on nothing."""
        second = EchoLLMClient({"evaluate": [self.RIGHT]})
        from tutor.state.answer import AnswerEvaluator

        asking = {"intent": "QUESTION", "verdict": "INCORRECT", "feedback": "",
                  "misconception": None, "status": None}
        ev = AnswerEvaluator(EchoLLMClient({"evaluate": [asking]}), None,
                             second_opinion=second)
        assert self._ask(ev).intent == "QUESTION"
        assert second.calls == []

    def test_a_broken_second_opinion_leaves_the_first_verdict_alone(self):
        from tutor.state.answer import AnswerEvaluator

        class Broken(EchoLLMClient):
            def run_with_tools(self, **kwargs):
                raise RuntimeError("no network")

        ev = AnswerEvaluator(EchoLLMClient({"evaluate": [self.WRONG]}), None,
                             second_opinion=Broken())
        assert self._ask(ev).verdict == "INCORRECT"


class TestRunningAhead:
    """A student who answers past the question is ahead, not wrong. The number
    that says so also raises the leak guard's ceiling, so it is a proposal the
    orchestrator checks rather than a verdict it accepts."""

    # Three steps, so there is room to run ahead and still stop short of the
    # last one — the whole point of the ceiling.
    LONGER = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="기울기를 구한다", expression="m = -2"),
            SolutionStep(idx=2, description="접선을 세운다", expression="y = -2*x - 4"),
            SolutionStep(idx=3, description="넓이를 구한다", expression="S = 28"),
        ],
        final_answer=Answer(kind="SCALAR", value="28"),
        concepts=["differentiation"], verified=True, origin="db",
    )

    def _verdict(self, **extra):
        return {"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
                "misconception": None, "status": "CORRECT", **extra}

    def _session(self, db, verdict):
        session, llm, speaker = build_session(db, [verdict])
        session.ctx.reference = self.LONGER
        return session, llm, speaker

    async def test_a_verified_jump_moves_the_student_forward(self, db):
        """The live case: asked for the slope, answered with the whole tangent
        — which is step 1 done and step 2 as well."""
        session, _, _ = self._session(db, self._verdict(
            reached_step=2, reached_claim="y = -2*x - 4",
        ))
        ask_l1(session, step=1)
        await session.handle_answer("L은 마이너스 2x 마이너스 4", session.store.pending_hint("p1"))

        assert session.store.get_state().last_correct_step == 2

    async def test_an_unprovable_jump_costs_nothing(self, db):
        """No claim to check means no jump — the answer still counts for the
        step that was asked."""
        session, _, _ = self._session(db, self._verdict(reached_step=2))
        ask_l1(session, step=1)
        await session.handle_answer("다 풀었어요", session.store.pending_hint("p1"))

        assert session.store.get_state().last_correct_step == 1

    async def test_a_claim_that_does_not_match_the_step_is_ignored(self, db):
        session, _, _ = self._session(db, self._verdict(
            reached_step=2, reached_claim="y = 5*x + 1",   # not step 2
        ))
        ask_l1(session, step=1)
        await session.handle_answer("y는 5x 더하기 1이요", session.store.pending_hint("p1"))

        assert session.store.get_state().last_correct_step == 1

    async def test_a_rearranged_claim_still_counts(self, db):
        """sympy decides, not string equality: the student said the same line
        a different way."""
        session, _, _ = self._session(db, self._verdict(
            reached_step=2, reached_claim="y + 2*x = -4",
        ))
        ask_l1(session, step=1)
        await session.handle_answer("y 더하기 2x는 마이너스 4요", session.store.pending_hint("p1"))

        assert session.store.get_state().last_correct_step == 2

    async def test_running_ahead_never_reaches_the_last_step(self, db):
        """The ceiling is what keeps the leak guard shut and the problem open:
        an LLM-proposed jump to the final step is clamped one short, whatever
        the claim says. What DOES close a problem early is the student
        ASSERTING the verified value at the utterance's tail ("넓이는 28이요")
        — that path is mechanical, not the judge's, and has its own tests in
        TestSayingTheAnswerEndsIt. Here the value is only mentioned in
        passing, so the clamp is all that speaks."""
        session, _, speaker = self._session(db, self._verdict(
            reached_step=3, reached_claim="S = 28",    # the final step itself
        ))
        ask_l1(session, step=1)
        await session.handle_answer(
            "28이 나오는 것 같긴 한데 확실하진 않아요", session.store.pending_hint("p1")
        )

        assert session.store.get_state().last_correct_step == 1   # not 3
        assert session.ctx is not None                            # still open
        assert "문제를 끝까지" not in " ".join(speaker.spoken)


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

        # the VALUE is said: only that closes a problem (a right method alone
        # is PARTIAL on the last step — see TestAPlanDoesNotCloseTheProblem)
        await session.handle_answer("양변을 3으로 나누면 5예요", session.store.pending_hint("p1"))

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
        await session.handle_answer("나누면 5예요", session.store.pending_hint("p1"))

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


class TestSurrenderIsNotGraded:
    """Live: "음, 잘 모르겠는데." went to the judge, the judge said INCORRECT,
    the second opinion spent 2.8s agreeing, and the feedback told a student
    who had just said they cannot think of anything to think a bit more —
    immediately before helping anyway. A surrender is not an attempt: nothing
    needs a model, and the reaction must be warmth, not a nudge."""

    class NoJudge:
        def run_with_tools(self, **kw):
            raise AssertionError("the judge was consulted for a surrender")

    def evaluator(self):
        from tutor.state.answer import AnswerEvaluator
        return AnswerEvaluator(self.NoJudge(), second_opinion=self.NoJudge())

    @pytest.mark.parametrize("said", [
        "음, 잘 모르겠는데.",
        "모르겠어요",
        "힌트 주세요",
        "어떻게 하는지 모르겠어요...",
        "미분해야 하는지 모르겠어요",   # not knowing the APPROACH is being stuck
    ])
    def test_a_surrender_escalates_without_a_judge(self, said):
        from tutor.state.answer import SURRENDER_FEEDBACK
        v = self.evaluator().evaluate(
            problem_text="p", reference=REFERENCE, question="q",
            target_step=1, transcript=said,
        )
        assert v.verdict == "INCORRECT"          # the pending hint failed
        assert v.intent == "ANSWER"
        assert v.feedback == SURRENDER_FEEDBACK  # warmth, never "think more"

    @pytest.mark.parametrize("said", [
        "5인 것 같은데 잘 모르겠어요",             # carries a value: grade it
        "x는 2 아닌가요? 모르겠네요",
    ])
    def test_an_attempt_wearing_doubt_still_reaches_the_judge(self, said):
        with pytest.raises(AssertionError, match="judge was consulted"):
            self.evaluator().evaluate(
                problem_text="p", reference=REFERENCE, question="q",
                target_step=1, transcript=said,
            )


class TestConfirmedWorkPointsForward:
    """A correct work check opens the next idea as a question, without reading
    an internal DB step label aloud like a navigation instruction."""

    def state(self, step):
        from tutor.state.models import StudentState
        return StudentState(status="CORRECT", last_correct_step=step)

    def line(self, step, reference):
        from tutor.server.session import Session
        text, _asked = Session._confirmed_line(self.state(step), reference)
        return text

    def asked(self, step, reference):
        from tutor.server.session import Session
        _text, asked = Session._confirmed_line(self.state(step), reference)
        return asked

    def test_mid_problem_names_the_next_step(self):
        from tutor.knowledge.models import Answer, ReferenceSolution, SolutionStep
        ref = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                                expression="f'(1) = -2"),
                   SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                                expression="l: y = -2*x - 4")],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["differentiation"], verified=True, origin="db",
        )
        assert self.line(1, ref) == \
            "맞아요! 여기까지 잘했어요. 이제 점 (1, -6)을 지나는 l의 방정식을 어떻게 쓰면 좋을까요?"
        assert "차례" not in self.line(1, ref)
        # the forward question is a real question, so it reports the step it
        # asked about — the caller records it, and the reply lands on a
        # pending question instead of on silence
        assert self.asked(1, ref) == 2

    def test_a_finished_problem_gets_the_congratulation(self):
        from tutor.knowledge.models import Answer, ReferenceSolution, SolutionStep
        ref = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="계산", expression="1+1")],
            final_answer=Answer(kind="SCALAR", value="2"),
            concepts=[], verified=True, origin="db",
        )
        assert "다 풀었어요" in self.line(1, ref)

    def test_a_sentence_step_falls_back_to_the_plain_verdict(self):
        from tutor.knowledge.models import Answer, ReferenceSolution, SolutionStep
        from tutor.server.session import WORK_CONFIRMED
        ref = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="도함수를 구합니다.", expression="f'")],
            final_answer=Answer(kind="SCALAR", value="1"),
            concepts=[], verified=True, origin="db",
        )
        assert self.line(0, ref) == WORK_CONFIRMED

    def test_no_reference_keeps_the_old_line(self):
        from tutor.server.session import WORK_CONFIRMED
        assert self.line(3, None) == WORK_CONFIRMED


class TestWorkingAloudIsNotParroted:
    """Live: "우선 f를 미분해야겠지. 그러면 2x-4." came back as "우선 f를
    미분해야겠지. 그러면 2x-4인지 보고 있어요." A value can be echoed; a train
    of thought cannot — the stage line follows along instead."""

    def test_the_live_utterance_is_not_a_core(self):
        from tutor.server.session import answer_core
        assert answer_core("우선 f를 미분해야겠지. 그러면 2x-4.") is None

    @pytest.mark.parametrize("said,core", [
        ("5예요", "5"),
        ("마이너스 3이요", "마이너스 3"),
        ("x는 2요", "x는 2"),
    ])
    def test_values_still_echo(self, said, core):
        from tutor.server.session import answer_core
        assert answer_core(said) == core

    @pytest.mark.parametrize("said", [
        "양변을 미분하면 2x-4가 나와서 그게 기울기예요",   # a speech ending in a value
        "일단 정리하면, 3이요",                            # comma = clause boundary
    ])
    def test_speeches_do_not(self, said):
        from tutor.server.session import answer_core
        assert answer_core(said) is None


class TestAPlanDoesNotCloseTheProblem:
    """Live on 수능 13: the last step's L1 asked what to take as the base and
    height, the student answered THAT, the judge rightly said CORRECT — and
    the problem closed with 49 never said. The last step only closes on the
    VALUE; a right plan is PARTIAL and the step stays open."""

    def test_the_value_gate_reads_speech(self):
        from tutor.knowledge.models import Answer
        from tutor.server.session import names_the_final_value
        answer = Answer(kind="SCALAR", value="49")
        assert not names_the_final_value("y절편 사이를 밑변으로 하면 돼요", answer)
        assert names_the_final_value("그러면 49예요", answer)
        assert names_the_final_value("답은 49", answer)
        # fractions survive both ways STT writes them
        frac = Answer(kind="SCALAR", value="24/7")
        assert names_the_final_value("24/7이요", frac)
        assert not names_the_final_value("일반항을 쓰면 돼요", frac)
        # a surd cannot be transcribed in a checkable shape: the judge's call
        surd = Answer(kind="SCALAR", value="3*sqrt(10)/10")
        assert names_the_final_value("루트로 나와요", surd)

    def test_component_values_are_not_mistaken_for_the_final_value(self):
        from tutor.knowledge.models import Answer
        from tutor.server.session import final_value_claim

        answer = Answer(kind="SCALAR", value="49")
        question = (
            "두 y절편 사이의 길이와 교점의 x좌표를 이용하면, "
            "삼각형의 밑변과 높이는 각각 무엇일까요?"
        )
        assert final_value_claim(
            "밑변은 14고 높이는 7", answer, question
        ) == "none"
        assert final_value_claim(
            "밑변은 14고 높이는 7이라서 곱한 뒤 나누면 돼요", answer, question
        ) == "none"
        assert final_value_claim(
            "밑변은 14고 높이는 7이고 최종 답은 49예요", answer, question
        ) == "said"

    async def test_a_right_plan_keeps_the_problem_open(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=2)            # the LAST step of REFERENCE

        await session.handle_answer(
            "양변을 3으로 나누면 돼요", session.store.pending_hint("p1")
        )

        assert session.ctx is not None                     # NOT closed
        assert "solved" not in session.ws.event_names()
        assert "끝까지 풀었네요" not in " ".join(speaker.spoken)
        assert speaker.spoken[0].startswith("맞아요!")
        state = session.store.get_state()
        assert state is not None and state.status == "STUCK"   # the partial path

    async def test_correct_base_and_height_are_confirmed_not_called_wrong(self, db):
        reference = ReferenceSolution(
            steps=[
                SolutionStep(
                    idx=1,
                    description="y절편 사이를 밑변으로 삼각형의 넓이 구하기",
                    expression="(1/2)*(10 - (-4))*7 = 49",
                ),
            ],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["triangle_area"], verified=True, origin="db",
        )
        session, _, speaker = build_session(
            db,
            [{
                "verdict": "CORRECT",
                "feedback": "맞아요, 밑변과 높이를 정확히 찾았어요!",
                "misconception": None,
                "status": "CORRECT",
            }],
        )
        session.ctx.reference = reference
        session.ctx.match = MatchResult(
            tier=Tier.NEW, concepts=reference.concepts, reference=reference,
        )
        session.store.set_state(StudentState(status="STUCK", last_correct_step=0))
        session.store.append_hint(
            problem_hash="p1", step=1, level=1, action="SOCRATIC_QUESTION",
            hint_text=(
                "두 y절편 사이의 길이와 교점의 x좌표를 이용하면, "
                "삼각형의 밑변과 높이는 각각 무엇일까요?"
            ),
        )

        await session.handle_answer(
            "밑변은 14고 높이는 7", session.store.pending_hint("p1")
        )

        assert session.ctx is not None
        assert "solved" not in session.ws.event_names()
        assert speaker.spoken[0] == "맞아요, 밑변과 높이를 정확히 찾았어요!"
        assert not speaker.spoken[0].startswith("접근 방식은 좋아요")
        latest = session.store.get_history(problem_hash="p1")[-1]
        assert (latest.step, latest.level) == (1, 1)


class TestAWrongValueNeverHearsRight:
    """Live: "밑변은 14고, 높이가 7, 곱하면 98 맞나?" — right setup, wrong
    value — was answered with 맞아요-wording twice. A number asserted as the
    result at the utterance's tail is a CLAIM; a wrong claim is INCORRECT,
    however good the approach, and the working numbers before it (14, 7)
    assert nothing."""

    def test_the_claim_reader(self):
        from tutor.knowledge.models import Answer
        from tutor.server.session import final_value_claim
        a49 = Answer(kind="SCALAR", value="49")
        assert final_value_claim("밑변은 14고, 높이가 7, 곱하면 98 맞나?", a49) == "wrong"
        assert final_value_claim("곱하면 49예요", a49) == "said"
        assert final_value_claim("y절편 차이를 밑변으로 하면 돼요", a49) == "none"
        # working numbers mid-sentence, no tail assertion: not a claim
        assert final_value_claim("밑변이 14니까 이걸 반으로 나눠 볼게요", a49) == "none"

    async def test_a_wrong_final_claim_is_incorrect_not_partial(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=2)            # the LAST step; the answer is 5

        await session.handle_answer(
            "3으로 나누면 6 맞나?", session.store.pending_hint("p1")
        )

        assert session.ctx is not None
        assert "solved" not in session.ws.event_names()
        assert speaker.spoken[0].startswith("접근 방식은 좋아요.")
        assert "맞아요" not in speaker.spoken[0]
        history = session.store.get_history(problem_hash="p1")
        assert history[0].effective is False      # the pending hint did not land


class TestAFragmentIsNotAnAnswer:
    """"응, 레이 와이 절편은" — the sentence died on a particle and the VAD
    closed on the pause. There is nothing to grade: no judge, no verdict on
    work the student never said, just a re-ask."""

    def evaluator(self):
        from tutor.state.answer import AnswerEvaluator

        class NoJudge:
            def run_with_tools(self, **kw):
                raise AssertionError("judge was consulted")

        return AnswerEvaluator(NoJudge())

    @pytest.mark.parametrize("said", [
        "응, 레이 와이 절편은",
        "그러니까 두 직선의 교점을",
        "밑변의 길이는",
    ])
    def test_a_dangling_particle_is_reasked_not_graded(self, said):
        from tutor.state.answer import CUTOFF_FEEDBACK
        v = self.evaluator().evaluate(
            problem_text="p", reference=REFERENCE, question="q",
            target_step=1, transcript=said,
        )
        assert v.verdict == "UNCLEAR"
        assert v.feedback == CUTOFF_FEEDBACK

    @pytest.mark.parametrize("said", [
        "x는 2",                     # ends on the value: the answer shape
        "마이너스 2",
        "두 값의 차이",              # a noun that merely ENDS in 이
    ])
    def test_value_tails_and_nouns_still_reach_the_judge(self, said):
        with pytest.raises(AssertionError, match="judge was consulted"):
            self.evaluator().evaluate(
                problem_text="p", reference=REFERENCE, question="q",
                target_step=1, transcript=said,
            )


class TestTheEchoDropsTheThrowatClearing:
    """"그럼 마이너스 2" must echo as "마이너스 2…", not "그럼 마이너스 2…":
    the discourse marker is the student turning to speak, not the value."""

    @pytest.mark.parametrize("said,core", [
        ("그럼 마이너스 2", "마이너스 2"),
        ("응, 그럼 5", "5"),
        ("음, 마이너스 4", "마이너스 4"),
        ("그러니까 x는 2", "x는 2"),
    ])
    def test_leading_markers_are_stripped(self, said, core):
        from tutor.server.session import answer_core
        assert answer_core(said) == core


class TestSayingTheAnswerEndsIt:
    """The student asserts the verified final value — from ANY step, the
    problem closes with a yes. A tutor who hears "정답은 5예요" mid-itinerary
    and replies with the next sub-step is grading the itinerary, not the trip.
    Only a TAIL assertion closes: "5를 빼면 돼요" merely passes through the
    number and keeps the normal ladder."""

    async def test_the_answer_asserted_early_closes_the_problem(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=1)            # NOT the last step

        await session.handle_answer("정답은 5예요", session.store.pending_hint("p1"))

        assert session.ctx is None                     # closed
        assert "solved" in session.ws.event_names()
        assert "hint_issued" not in session.ws.event_names()
        assert "끝까지 풀었네요" in " ".join(speaker.spoken)

    async def test_a_judge_that_disagrees_cannot_veto_a_verified_value(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "INCORRECT", "feedback": "음, 조금 달라요.",
              "misconception": None, "status": "STUCK"}],
        )
        ask_l1(session, step=1)

        await session.handle_answer("5 맞아요?", session.store.pending_hint("p1"))

        assert session.ctx is None
        assert "solved" in session.ws.event_names()
        spoken = " ".join(speaker.spoken)
        assert "정답이에요" in spoken and "조금 달라요" not in spoken

    async def test_a_passing_mention_does_not_close(self, db):
        session, llm, speaker = build_session(
            db,
            [{"verdict": "CORRECT", "feedback": "맞아요!",
              "misconception": None, "status": "CORRECT"}],
        )
        ask_l1(session, step=1)

        await session.handle_answer("5를 빼면 돼요", session.store.pending_hint("p1"))

        assert session.ctx is not None                 # still teaching
        assert "solved" not in session.ws.event_names()


class TestAHalfComputedCompositeIsNotDone:
    """Live: "f′부터 계산하면 좋을 것 같아. 계산하면 2x-4." wore no plan tail,
    was graded CORRECT, the target advanced, and the step-2 line then
    congratulated a slope nobody had computed. A composite step whose final
    numeric piece (-2) is absent from the transcript is PARTIAL, whatever
    the judge said — and the partial machinery then asks only for f'(1)."""

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*x - 4"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )

    def judge(self, db):
        llm = EchoLLMClient({"evaluate": [
            {"verdict": "CORRECT", "feedback": "맞아요!",
             "misconception": None, "status": "CORRECT"},
        ]})
        return AnswerEvaluator(llm, db)

    def test_the_first_half_alone_is_partial(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.REF, question="q", target_step=1,
            transcript="f프라임부터 계산하면 좋을 것 같아. 계산하면 2x-4.",
        )
        assert v.verdict == "PARTIAL"

    def test_saying_the_final_piece_completes_it(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.REF, question="q", target_step=1,
            transcript="2x-4니까 f프라임 1은 마이너스 2예요",
        )
        assert v.verdict == "CORRECT"       # 마이너스 2 reads as -2

    @pytest.mark.parametrize("transcript", ["Minus e.", "Minus two.", "Minus 2."])
    def test_context_repairs_the_short_english_minus_two_transcript(
        self, db, transcript
    ):
        v = self.judge(db).evaluate(
            problem_text="p",
            reference=self.REF,
            question="f 프라임 1은 얼마일까요?",
            target_step=1,
            transcript=transcript,
        )

        assert v.verdict == "CORRECT"

    def test_minus_e_is_not_rewritten_for_a_different_target(self, db):
        other = self.REF.model_copy(deep=True)
        other.steps[0].expression = "f'(x) = 2*x - 5, f'(1) = -3"
        v = self.judge(db).evaluate(
            problem_text="p",
            reference=other,
            question="f 프라임 1은 얼마일까요?",
            target_step=1,
            transcript="Minus e.",
        )

        assert v.verdict == "PARTIAL"

    def test_a_symbolic_tail_stays_the_judges_call(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.REF, question="q", target_step=2,
            transcript="y는 마이너스 2x 마이너스 4예요",
        )
        assert v.verdict == "CORRECT"       # step 2 ends symbolically: no gate

    INTERCEPTS = ReferenceSolution(
        steps=[SolutionStep(
            idx=1, description="두 직선 l, m의 y절편 구하기",
            expression="l(0) = -4, m(0) = 10",
        )],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=[], verified=True, origin="db",
    )

    def test_saying_the_values_needs_no_attribution(self, db):
        """Live: both intercepts arrived inside a plan-shaped sentence and the
        gate held the student at step 7, asking which line owns which — a
        distinction no later step needs. Said values outrank phrasing."""
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.INTERCEPTS, question="q",
            target_step=1,
            transcript="x에 0을 대입하면 돼요. 마이너스 4하고 10이에요.",
        )
        assert v.verdict == "CORRECT"

    def test_the_plan_alone_is_still_partial(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.INTERCEPTS, question="q",
            target_step=1,
            transcript="x에 0을 대입하면 돼요.",
        )
        assert v.verdict == "PARTIAL"

    def test_y_is_repaired_to_x_only_for_an_x_coordinate_step(self, db):
        coordinate = ReferenceSolution(
            steps=[SolutionStep(
                idx=1, description="두 직선의 교점의 x좌표 구하기",
                expression="-2*x - 4 = -4*x + 10, x = 7",
            )],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=[], verified=True, origin="db",
        )
        evaluator = self.judge(db)

        assert evaluator.normalize_transcript(coordinate, 1, "y는 7이요") == "x는 7"
        assert evaluator.normalize_transcript(coordinate, 1, "y는 5이요") == "x는 5"
        assert evaluator.normalize_transcript(self.REF, 1, "y는 7이요") == "y는 7이요"


async def test_the_context_repaired_x_reaches_the_transcript_event(db):
    from tutor.speech.stt import Transcript

    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="두 직선의 교점의 x좌표 구하기",
                         expression="-2*x - 4 = -4*x + 10, x = 7"),
            SolutionStep(idx=2, description="y절편 구하기",
                         expression="l(0) = -4, m(0) = 10"),
            SolutionStep(idx=3, description="넓이 구하기", expression="49"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=[], verified=True, origin="db",
    )
    session, _, _ = build_session(db, [{
        "verdict": "CORRECT", "feedback": "맞아요!",
        "misconception": None, "status": "CORRECT",
    }])
    session.ctx.reference = reference
    session.ctx.match = MatchResult(tier=Tier.EXACT, concepts=[], reference=reference)

    class HeardY:
        def transcribe(self, pcm, sample_rate=16000):
            return Transcript(text="y는 7이요", language="ko")

    session.deps.transcriber = HeardY()
    ask_l1(session, step=1)
    await session._handle_utterance(b"\x00\x00" * 100, 16000)

    events = [json.loads(raw) for raw in session.ws.events]
    transcript = next(e for e in events if e["event"] == "transcript")
    assert transcript["data"]["text"] == "x는 7"


class TestAForeignValueIsNeverPraised:
    """Live on problem 13, with the target still at the y-intercepts: "두 개를
    곱하면 되니까 98인가?" — base times height, the halving forgotten. The
    judge called it right, the composite-step guard softened that to PARTIAL,
    and the student heard "맞아요, 여기까지는 잘했어요" for a wrong answer.
    Running ahead to the end and landing beside it is not a last-step
    privilege, so the refusal to nod cannot be gated on the last step either.
    """

    AREA_REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="두 직선의 교점의 x좌표 구하기",
                         expression="-2*x - 4 = -4*x + 10, x = 7"),
            SolutionStep(idx=2, description="두 직선 l, m의 y절편 구하기",
                         expression="l(0) = -4, m(0) = 10"),
            SolutionStep(idx=3, description="삼각형의 넓이 구하기",
                         expression="(1/2)*(10 - (-4))*7 = 49"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["derivative_applications"], verified=True, origin="db",
    )

    def session(self, db, verdict):
        session, llm, speaker = build_session(db, [verdict])
        session.ctx.reference = self.AREA_REF
        return session, llm, speaker

    async def test_ninety_eight_at_an_earlier_step_is_incorrect(self, db):
        session, _, speaker = self.session(db, {
            "intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
            "misconception": None, "status": "CORRECT",
        })
        ask_l1(session, step=2)              # the tutor is still on y절편

        await session.handle_answer(
            "두 개를 곱하면 되니까 98인가?", session.store.pending_hint("p1")
        )

        state = session.store.get_state()
        assert state.status != "CORRECT"
        assert state.last_correct_step == 1          # nothing was proven
        assert "맞아요" not in " ".join(speaker.spoken)

    async def test_the_wrong_answer_escalates_instead_of_fading(self, db):
        session, _, _ = self.session(db, {
            "intent": "ANSWER", "verdict": "PARTIAL", "feedback": "좋아요!",
            "misconception": None, "status": "CORRECT",
        })
        pending = ask_l1(session, step=2)

        await session.handle_answer(
            "두 개를 곱하면 되니까 98인가?", session.store.pending_hint("p1")
        )

        record = next(
            h for h in session.store.get_history(problem_hash="p1") if h.id == pending
        )
        assert record.effective is False     # PARTIAL would have said True

    def test_the_values_the_problem_knows_are_not_foreign(self):
        """The guard's contract, where the last-step rules cannot muddy it.

        14 is the base, written inside the step as 10 - (-4) where no digit
        scan can see it; 49 is the answer; 7 and -4 are earlier results. Only
        98 — the product of a forgotten halving — is foreign."""
        from tutor.server.session import foreign_value_assertion

        def foreign(said, step=2):
            return foreign_value_assertion(said, self.AREA_REF, step, "", [])

        assert foreign("두 개를 곱하면 되니까 98인가?") == "98"
        assert foreign("답은 100이에요") == "100"
        assert foreign("빼면 14요") is None
        assert foreign("곱하면 49예요") is None
        assert foreign("계산하면 마이너스 4요") is None
        assert foreign("나누면 7이요") is None

    def test_an_equation_read_aloud_asserts_no_value(self):
        """"L은 마이너스 2x 마이너스 4" ends in a number without claiming one:
        the trailing term of a spoken equation is not a result."""
        from tutor.server.session import foreign_value_assertion

        assert foreign_value_assertion(
            "L은 마이너스 2x 마이너스 4", self.AREA_REF, 2, "", []
        ) is None


class TestASpokenNumberFinishesItsStep:
    """Live on problem 13 step 6: "그럼 엑스는 칠에서 만날 것 같은데" is the
    intersection, found and said. STT wrote 칠, the composite-step guard
    scanned for digits, found none, and downgraded a finished step to PARTIAL
    — which then faded the ladder back to L1 on ground already covered."""

    STEP6 = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="m의 방정식", expression="m: y = -4*x + 10"),
            SolutionStep(idx=2, description="두 직선의 교점의 x좌표 구하기",
                         expression="-2*x - 4 = -4*x + 10, x = 7"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["derivative_applications"], verified=True, origin="db",
    )

    def judge(self, db):
        llm = EchoLLMClient({"evaluate": [
            {"verdict": "CORRECT", "feedback": "맞아요!",
             "misconception": None, "status": "CORRECT"},
        ]})
        return AnswerEvaluator(llm, db)

    def test_the_spoken_seven_completes_the_step(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.STEP6, question="교점의 x좌표는?",
            target_step=2, transcript="그럼 엑스는 칠에서 만날 것 같은데.",
        )
        assert v.verdict == "CORRECT"

    def test_a_spoken_wrong_number_still_falls_short(self, db):
        v = self.judge(db).evaluate(
            problem_text="p", reference=self.STEP6, question="교점의 x좌표는?",
            target_step=2, transcript="엑스는 오에서 만날 것 같은데.",
        )
        assert v.verdict == "PARTIAL"


class TestTheClosingValueMayBeSpelled:
    """Live on the last step: the student said 49 and STT wrote 사십구, so the
    value check found no number, the answer was graded PARTIAL and a finished
    problem stayed open. The ear's own numerals were already understood in
    grading; the session's value checks now read them too."""

    from tutor.knowledge.models import Answer as _A

    ANSWER = _A(kind="SCALAR", value="49")

    @pytest.mark.parametrize("said", [
        "사십구요", "정답은 사십구", "답은 사십구예요", "49요",
    ])
    def test_the_answer_closes_however_it_was_written(self, said):
        from tutor.server.session import final_value_claim
        assert final_value_claim(said, self.ANSWER, "") == "said"

    def test_a_spelled_wrong_value_is_still_wrong(self):
        from tutor.server.session import final_value_claim
        assert final_value_claim("정답은 오십이요", self.ANSWER, "") == "wrong"

    def test_a_foreign_value_said_in_words_is_caught(self, db):
        from tutor.server.session import foreign_value_assertion
        from tutor.knowledge.db import KnowledgeDB

        reference = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="넓이",
                                expression="(1/2)*(10 - (-4))*7 = 49")],
            final_answer=self.ANSWER, concepts=[], verified=True, origin="db",
        )
        assert foreign_value_assertion(
            "두 개를 곱하면 구십팔인가?", reference, 1, "", []
        ) == "98"
