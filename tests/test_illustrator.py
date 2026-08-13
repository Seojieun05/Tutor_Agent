"""The drawing hand, which works while the tutor is already talking.

A hint takes ~8s to speak, and that is the window this runs in — so unlike a
parallel design it gets the finished sentence and draws what was said. The
tests hold the two promises that follow from the timing: the picture answers
to the sentence, and a late picture never lands on a turn that has moved on.
"""

import asyncio
import json

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.hints.illustrator import Illustrator, wants_a_picture
from tutor.knowledge.matching import Matcher
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.policy.engine import Action, Decision
from tutor.server.session import Deps, ProblemContext, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

QUAD_REF = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="인수분해한다", expression="(x - 1)*(x - 3) = 0"),
        SolutionStep(idx=2, description="근을 구한다", expression="x = 1, x = 3"),
    ],
    final_answer=Answer(kind="ROOT_SET", value=["1", "3"]),
    concepts=["quadratic_function"], verified=True, origin="db",
)
REC = Recognition(
    problem_text="이차함수 y = x**2 - 4*x + 3 의 그래프가 x축과 만나는 점은?",
    equations=["y = x**2 - 4*x + 3"],
    student_work=["x**2 - 4*x + 3 = 0"],
    concepts=["quadratic_function"], confidence=0.95,
)


class TestTheTrigger:
    """The sentence decides. A hint that never mentions a shape is not a hint
    a curve supports — and skipping it there is what keeps the call off every
    algebraic turn."""

    @pytest.mark.parametrize("hint", [
        "그래프의 개형을 떠올려 볼까요?",
        "두 곡선이 만나는 곳은 어떤 방정식의 해일까요?",
        "이 구간에서 함수가 증가하는지 감소하는지 보세요.",
        "x축과 만나는 점을 생각해 볼까요?",
    ])
    def test_a_sentence_about_shape_asks_for_a_picture(self, hint):
        assert wants_a_picture(hint)

    @pytest.mark.parametrize("hint", [
        "양변에서 5를 빼면 무엇이 남을까요?",
        "곱의 미분법을 어떻게 적용할 수 있을까요?",
        "공비가 몇 번 곱해지는지 세어 보세요.",
    ])
    def test_pure_algebra_does_not(self, hint):
        assert not wants_a_picture(hint)


class TestTheSpec:
    def test_it_reads_the_hint_and_returns_a_window(self, db):
        llm = EchoLLMClient({"illustrate": [{
            "curves": [{"expr": "y = x**2 - 4*x + 3", "label": "f",
                        "role": "scaffold"}],
            "x_min": -1, "x_max": 5,
            "caption": "x축과 만나는 곳", "why": "부호 변화를 보여줌",
        }]})
        spec = Illustrator(llm).draw(
            hint="x축과 만나는 점을 생각해 볼까요?", problem_text=REC.problem_text,
            equations=REC.equations, student_work=REC.student_work,
            board=[], misconception=None, level=1,
        )
        assert [c.expr for c in spec.curves] == ["x**2 - 4*x + 3"]  # "y =" stripped
        assert (spec.curves[0].label, spec.curves[0].role) == ("f", "scaffold")
        assert (spec.x_min, spec.x_max) == (-1, 5)
        assert spec.caption == "x축과 만나는 곳"

    def test_nothing_to_draw_is_a_valid_answer(self, db):
        llm = EchoLLMClient({"illustrate": [{"curves": [], "why": "대수로 충분"}]})
        spec = Illustrator(llm).draw(
            hint="그래프를 떠올려 볼까요?", problem_text="p", equations=[],
            student_work=[], board=[], misconception=None, level=1,
        )
        assert spec.curves == []

    def test_a_broken_call_costs_the_picture_and_nothing_else(self, db):
        class Broken(EchoLLMClient):
            def complete_json(self, **kwargs):
                raise RuntimeError("no network")

        assert Illustrator(Broken()).draw(
            hint="그래프를 볼까요?", problem_text="p", equations=[],
            student_work=[], board=[], misconception=None, level=1,
        ) is None

    def test_a_caption_that_grew_into_a_sentence_is_trimmed(self, db):
        llm = EchoLLMClient({"illustrate": [{
            "curves": [{"expr": "x**2"}],
            "caption": "이 그래프에서 x축과 만나는 부분을 보시면 방정식의 해가 보이는데 그것을 확인해 보세요",
        }]})
        spec = Illustrator(llm).draw(
            hint="그래프를 볼까요?", problem_text="p", equations=[],
            student_work=[], board=[], misconception=None, level=1,
        )
        assert len(spec.caption) <= 28


class FakeWS:
    def __init__(self):
        self.events: list[dict] = []

    async def send(self, raw):
        if isinstance(raw, str):
            self.events.append(json.loads(raw))

    def names(self) -> list[str]:
        return [e["event"] for e in self.events]


def build(db, illustrate: list[dict] | None):
    llm = EchoLLMClient({"illustrate": illustrate} if illustrate else {})
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
        illustrator=Illustrator(llm) if illustrate is not None else None,
        store=SessionStore(),
    )
    session = Session(FakeWS(), deps)
    session.ctx = ProblemContext(
        hash="p1", recognition=REC,
        match=MatchResult(tier=Tier.EXACT, concepts=["quadratic_function"],
                          reference=QUAD_REF),
        reference=QUAD_REF,
    )
    return session, llm, speaker


def a_hint(level=1):
    return Decision(Action.SOCRATIC_QUESTION, level, 1, None, "test")


class TestInTheSession:
    async def test_it_draws_while_the_hint_is_spoken(self, db):
        session, llm, speaker = build(db, [{
            "curves": [{"expr": "x**2 - 4*x + 3", "label": "f", "role": "scaffold"}],
            "x_min": -1, "x_max": 5, "caption": "x축과 만나는 곳", "why": "t",
        }])
        await session._deliver(a_hint(), "x축과 만나는 점을 생각해 볼까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

        figure = next(e for e in session.ws.events if e["event"] == "figure")
        assert figure["data"]["svg"].startswith("<svg")
        assert figure["data"]["note"] == "x축과 만나는 곳"
        assert figure["data"]["id"] == "p1"          # one canvas per problem
        assert speaker.spoken == ["x축과 만나는 점을 생각해 볼까요?"]
        # the scene is remembered, so the next turn can be told what is drawn
        assert [c.expr for c in session.ctx.scene] == ["x**2 - 4*x + 3"]
        assert session.ctx.span == (-1, 5)

    async def test_an_algebraic_hint_never_pays_for_the_call(self, db):
        session, llm, _ = build(db, [{"curves": [{"expr": "x**2"}], "why": "t"}])
        await session._deliver(a_hint(), "양변에서 5를 빼면 무엇이 남을까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

        assert "illustrate" not in llm.calls
        assert "figure" not in session.ws.names()

    async def test_a_sketch_of_the_answer_is_not_drawn(self, db):
        """A curve of the answer IS the answer: the same gate as the words."""
        session, llm, _ = build(db, [{
            "curves": [{"expr": "(x - 1)*(x - 3)"}],   # the reference's own step 1
            "why": "t",
        }])
        await session._deliver(a_hint(), "그래프의 개형을 떠올려 볼까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

        assert "figure" not in session.ws.names()

    async def test_a_picture_never_lands_on_a_problem_that_is_over(self, db):
        """It is drawn while the tutor talks, so the turn can move on
        underneath it — and a sketch of the last hint on the next problem is
        worse than no sketch at all."""
        session, llm, _ = build(db, [{
            "curves": [{"expr": "x**2 - 4*x + 3"}], "why": "t",
        }])
        task = asyncio.create_task(
            session._illustrate(a_hint(), "그래프의 개형을 볼까요?", (), "p1")
        )
        session.ctx = None                      # the student moved on
        await task

        assert "figure" not in session.ws.names()

    async def test_the_scene_grows_on_one_grid_and_wipes_its_scaffolding(self, db):
        """The behaviour the whole scene model exists for. f is drawn to find
        the tangent l; once l is on the grid f has done its job and the next
        scene simply omits it. Same canvas id throughout — the picture
        changes rather than a second picture appearing below it."""
        session, llm, _ = build(db, [
            {"curves": [{"expr": "x**2 - 4*x - 3", "label": "f", "role": "scaffold"}],
             "x_min": -2, "x_max": 6, "caption": "접점을 지나는 곡선", "why": "1"},
            {"curves": [{"expr": "-2*x - 4", "label": "l", "role": "target"}],
             "caption": "구한 접선", "why": "2"},
        ])
        await session._deliver(a_hint(), "곡선의 개형을 떠올려 볼까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])
        assert [c.label for c in session.ctx.scene] == ["f"]

        # the student says the tangent, and it replaces the curve it came from
        await session._deliver(
            a_hint(2), "이제 접선의 기울기를 어떻게 쓸 수 있을까요?", "p1",
            student_said="접선은 y = -2x - 4예요",
        )
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

        assert [c.label for c in session.ctx.scene] == ["l"]     # f is gone
        assert session.ctx.span == (-2, 6)                       # the grid held still
        figures = [e for e in session.ws.events if e["event"] == "figure"]
        assert len(figures) == 2
        assert {f["data"]["id"] for f in figures} == {"p1"}       # one canvas
        assert "-2·x - 4" in figures[-1]["data"]["svg"]

    async def test_what_the_student_said_reaches_the_drawing_hand(self, db):
        session, llm, _ = build(db, [{
            "curves": [{"expr": "-2*x - 4", "label": "l", "role": "target"}], "why": "t",
        }])
        seen: dict = {}
        inner = session.deps.illustrator.draw

        def spy(**kwargs):
            seen.update(kwargs)
            return inner(**kwargs)

        session.deps.illustrator.draw = spy
        await session._deliver(a_hint(), "접선을 그려 볼까요?", "p1",
                               student_said="기울기는 -2예요")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])
        assert seen["student_said"] == "기울기는 -2예요"

    async def test_without_an_illustrator_the_tutor_still_teaches(self, db):
        session, llm, speaker = build(db, None)
        await session._deliver(a_hint(), "그래프의 개형을 떠올려 볼까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

        assert speaker.spoken == ["그래프의 개형을 떠올려 볼까요?"]
        assert "figure" not in session.ws.names()
