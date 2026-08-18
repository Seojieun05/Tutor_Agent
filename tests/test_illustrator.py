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
from tutor.hints import plot
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


class TestUnknownParametersStayQualitative:
    def test_a_representative_value_cannot_gain_a_scale_or_legend(self, db):
        llm = EchoLLMClient({"illustrate": [{
            "curves": [{"expr": "2**x - 3", "role": "scaffold"}],
            "points": [{"x": 1, "y": -1, "label": "P"}],
            # Deliberately unsafe proposal: the general guard must override it.
            "show_scale": True,
            "show_legend": True,
            "caption": "b=2인 그래프",
        }]})
        spec = Illustrator(llm).draw(
            hint="점 P의 위치를 살펴볼까요?",
            problem_text="상수 b>1에 대하여 y=b^x-3 위의 점 P를 생각하자.",
            equations=["y = b**x - 3"],
            student_work=[], board=[], misconception=None, level=1,
        )

        assert [point.label for point in spec.points] == ["P"]
        assert not spec.show_scale and not spec.show_legend
        assert spec.caption == ""

    def test_a_fully_specified_graph_keeps_the_normal_scale(self, db):
        llm = EchoLLMClient({"illustrate": [{
            "curves": [
                {"expr": "x**2 - 4*x - 3", "label": "f"},
                {"expr": "(x**3 - 2*x)*(x**2 - 4*x - 3)", "label": "g"},
            ],
        }]})
        spec = Illustrator(llm).draw(
            hint="곡선 f와 g를 살펴볼까요?",
            problem_text="함수 f와 g의 접선을 구하는 문제",
            equations=[
                "f(x) = x**2 - 4*x - 3",
                "g(x) = (x**3 - 2*x)*f(x)",
            ],
            student_work=[], board=[], misconception=None, level=1,
        )

        assert spec.show_scale and spec.show_legend

    def test_existing_named_points_are_given_back_to_the_drawing_model(self, db):
        from tutor.hints.illustrator import PlotPoint

        class Recording(EchoLLMClient):
            def __init__(self):
                super().__init__({"illustrate": [{
                    "curves": [{"expr": "2**x - 3"}],
                    "points": [
                        {"x": 1, "y": -1, "label": "P"},
                        {"x": 1, "y": 0, "label": "Q"},
                    ],
                }]})
                self.user = ""

            def complete_json(self, **kwargs):
                self.user = kwargs["user"]
                return super().complete_json(**kwargs)

        llm = Recording()
        spec = Illustrator(llm).draw(
            hint="다음 점을 표시해 볼까요?",
            problem_text="점 P와 점 Q를 표시한다.",
            equations=["y = b**x - 3"],
            student_work=[], board=[], misconception=None, level=2,
            points=[PlotPoint(x=1, y=-1, label="P")],
            show_scale=False, show_legend=False,
        )

        assert "지금 표시된 점" in llm.user and "P" in llm.user
        assert [point.label for point in spec.points] == ["P", "Q"]


@pytest.mark.asyncio
async def test_area_equation_already_said_in_prose_is_not_printed_twice(db):
    session, _, _ = build(db, [])
    session.ws.events.clear()
    await session._send_problem(Recognition(
        problem_text="삼각형 AOC의 넓이가 8일 때 값을 구하여라.",
        equations=["Area(AOC) = 8"],
    ))

    event = next(e for e in session.ws.events if e["event"] == "problem")
    assert event["data"]["equations"] == []


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


class TestTheStudentsWindow:
    """The view controls on the page: figure_zoom / figure_pan / figure_reset
    move the window of the scene already on the board, and the choice sticks
    to the problem so the next turn's redraw keeps it."""

    async def drawn(self, db):
        session, llm, speaker = build(db, [{
            "curves": [{"expr": "x**2 - 4*x + 3", "label": "f", "role": "target"}],
            "x_min": -1, "x_max": 5, "caption": "x축과 만나는 곳", "why": "t",
        }])
        await session._deliver(a_hint(), "x축과 만나는 점을 생각해 볼까요?", "p1")
        await asyncio.gather(*[t for t in session._tasks if not t.done()])
        return session

    async def poke(self, session, event, data=None):
        await session.on_frame(json.dumps(
            {"type": "EVENT", "event": event, "data": data or {}}))
        await asyncio.gather(*[t for t in session._tasks if not t.done()])

    def figures(self, session):
        return [e for e in session.ws.events if e["event"] == "figure"]

    async def test_zoom_out_redraws_the_same_scene_wider(self, db):
        session = await self.drawn(db)
        before = len(self.figures(session))

        await self.poke(session, "figure_zoom", {"dir": 1})

        figs = self.figures(session)
        assert len(figs) == before + 1
        last = figs[-1]["data"]
        assert last["id"] == "p1"                  # same canvas, replaced
        assert last["v"] == 1 and last["user"] is True
        assert last["note"] == "x축과 만나는 곳"
        x0, x1, y0, y1 = session.ctx.view           # sticks to the problem
        assert x1 - x0 > 6                          # wider than the (-1, 5) span

    async def test_every_adjustment_gets_its_own_revision(self, db):
        # widen then narrow lands back on a window ALREADY SEEN; only the
        # revision tells the page this render is new and must be painted
        session = await self.drawn(db)
        await self.poke(session, "figure_zoom", {"dir": 1})
        await self.poke(session, "figure_zoom", {"dir": -1})
        revs = [f["data"]["v"] for f in self.figures(session) if "v" in f["data"]]
        assert revs == [0, 1, 2]     # the illustrator's own render, then ours

    async def test_pan_slides_the_window(self, db):
        session = await self.drawn(db)
        await self.poke(session, "figure_zoom", {"dir": 1})
        x0, x1, y0, y1 = session.ctx.view
        await self.poke(session, "figure_pan", {"dx": 0.5, "dy": 0})
        nx0, nx1, ny0, ny1 = session.ctx.view
        assert nx0 - x0 == pytest.approx((x1 - x0) * 0.5)
        assert (ny0, ny1) == (y0, y1)

    async def test_pan_down_shows_the_plane_below(self, db):
        # dy is screen-down, and the y axis grows up
        session = await self.drawn(db)
        await self.poke(session, "figure_pan", {"dx": 0, "dy": 0.5})
        x0, x1, y0, y1 = session.ctx.view
        auto = plot.compute_view(list(session.ctx.scene), session.ctx.span)
        assert y1 < auto[3]

    async def test_reset_hands_the_window_back(self, db):
        session = await self.drawn(db)
        await self.poke(session, "figure_zoom", {"dir": 1})
        await self.poke(session, "figure_reset")
        assert session.ctx.view is None
        assert self.figures(session)[-1]["data"]["v"] == 2

    async def test_zoom_is_clamped(self, db):
        session = await self.drawn(db)
        for _ in range(10):
            await self.poke(session, "figure_zoom", {"dir": 1})
        x0, x1, _, _ = session.ctx.view
        span = session.ctx.span
        assert (x1 - x0) <= (span[1] - span[0]) * 5.0

    async def test_zoom_with_no_scene_says_nothing(self, db):
        session, llm, speaker = build(db, [])
        await self.poke(session, "figure_zoom", {"dir": 1})
        assert "figure" not in session.ws.names()

    async def test_garbage_is_ignored(self, db):
        session = await self.drawn(db)
        before = len(session.ws.events)
        await self.poke(session, "figure_zoom", {"dir": "sideways"})
        await self.poke(session, "figure_pan", {"dx": "no", "dy": None})
        await self.poke(session, "figure_reset")     # nothing to reset
        assert len(session.ws.events) == before


TANGENT_REF = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="접선 l의 기울기 구하기", expression="f'(1) = -2"),
        SolutionStep(idx=2, description="l의 방정식 쓰기", expression="l: y = -2*x - 4"),
        SolutionStep(idx=3, description="교점 구하기", expression="-2*x - 4 = -4*x + 10, x = 7"),
    ],
    final_answer=Answer(kind="SCALAR", value="49"),
    concepts=["differentiation"], verified=True, origin="db",
)


class DrawRecorder:
    def __init__(self):
        self.kw = None

    def draw(self, **kw):
        self.kw = kw
        return None


class TestAVerifiedLineReachesTheBoard:
    """A tangent derived OUT LOUD is the student's now, and no photograph of
    the page can show it: the orchestrator passes the verified prefix of the
    reference to the drawing hand, and a newly earned CURVE opens the scene
    even when the sentence being spoken is pure algebra."""

    def with_recorder(self, db, last_correct_step):
        from tutor.state.models import StudentState
        session, llm, _ = build(db, [{"curves": [], "why": "t"}])
        session.deps.illustrator = DrawRecorder()
        session.ctx.reference = TANGENT_REF
        session.store.set_state(StudentState(
            current_step="t", last_correct_step=last_correct_step, status="CORRECT",
        ))
        return session, session.deps.illustrator

    async def test_an_earned_curve_opens_the_scene_mid_algebra(self, db):
        session, recorder = self.with_recorder(db, last_correct_step=2)

        await session._illustrate(
            Decision(Action.WAIT, 0, 3, None, "confirmed"),
            "이제 g'(x)를 구해 볼까요?", (), "p1",       # no shape word in it
        )

        assert recorder.kw is not None                  # the hand was called
        assert recorder.kw["verified"] == ["f'(1) = -2", "l: y = -2*x - 4"]

    async def test_the_earned_line_arrives_before_the_drawing_model(self, db):
        session, _ = self.with_recorder(db, last_correct_step=2)
        observed = {}

        class ObservingHand:
            def draw(inner_self, **kw):
                observed["figure_already_sent"] = "figure" in session.ws.names()
                return None

        session.deps.illustrator = ObservingHand()
        await session._illustrate(
            Decision(Action.WAIT, 0, 3, None, "confirmed"),
            "이제 g'(x)를 구해 볼까요?", (), "p1",
        )

        assert observed["figure_already_sent"] is True
        figures = [e for e in session.ws.events if e["event"] == "figure"]
        assert len(figures) == 1                         # no duplicate repaint
        assert "-2·x - 4" in figures[0]["data"]["svg"]
        assert [c.label for c in session.ctx.scene] == ["l"]

    async def test_nothing_past_the_frontier_is_handed_over(self, db):
        session, recorder = self.with_recorder(db, last_correct_step=2)
        await session._illustrate(
            Decision(Action.WAIT, 0, 3, None, "confirmed"), "그래프를 볼까요?", (), "p1",
        )
        assert "x = 7" not in " ".join(recorder.kw["verified"])

    async def test_scalar_steps_do_not_open_a_scene(self, db):
        # f'(1) = -2 is a number, not a curve: an algebraic sentence with no
        # drawable step keeps the call off the wire, exactly as before
        session, recorder = self.with_recorder(db, last_correct_step=1)
        await session._illustrate(
            Decision(Action.WAIT, 0, 2, None, "confirmed"),
            "이제 l의 방정식을 어떻게 쓰면 좋을까요?", (), "p1",
        )
        assert recorder.kw is None


class TestAnEarnedLineIsNotUpForProposal:
    """ensure_verified_targets: a verified labelled line (l:, m:) reaches the
    scene whatever the model proposed — l went undrawn twice on its judgement."""

    VERIFIED = ["f'(1) = -2", "l: y = -2*(x - 1) - 6 = -2*x - 4"]

    def test_a_missing_target_is_added_from_the_chains_final_form(self):
        from tutor.hints.illustrator import Curve, FigureSpec, ensure_verified_targets
        spec = FigureSpec(curves=[Curve(expr="x**2 - 4*x - 3", label="f")])
        out = ensure_verified_targets(spec, self.VERIFIED)
        assert [(c.label, c.expr) for c in out.curves][0] == ("l", "-2*x - 4")
        assert out.curves[0].role == "target"

    def test_a_refusing_model_still_yields_the_earned_line(self):
        from tutor.hints.illustrator import ensure_verified_targets
        out = ensure_verified_targets(None, self.VERIFIED)
        assert out is not None
        assert [c.label for c in out.curves] == ["l"]

    def test_a_line_already_drawn_is_not_duplicated(self):
        from tutor.hints.illustrator import Curve, FigureSpec, ensure_verified_targets
        spec = FigureSpec(curves=[Curve(expr="-2*x - 4", label="l", role="target")])
        out = ensure_verified_targets(spec, self.VERIFIED)
        assert len(out.curves) == 1

    def test_scalar_steps_obligate_nothing(self):
        from tutor.hints.illustrator import ensure_verified_targets
        assert ensure_verified_targets(None, ["f'(1) = -2", "g'(1) = -4"]) is None

    def test_a_model_cannot_demote_an_earned_target_to_scaffolding(self):
        from tutor.hints.illustrator import Curve, FigureSpec, ensure_verified_targets
        spec = FigureSpec(curves=[
            Curve(expr="-2*x - 4", label="l", role="scaffold")
        ])
        out = ensure_verified_targets(spec, self.VERIFIED)
        assert out.curves[0].role == "target"


class TestTheVerifiedTangentSceneHasADeterministicOrder:
    EQUATIONS = [
        "f(x) = x**2 - 4*x - 3",
        "g(x) = (x**3 - 2*x)*f(x)",
    ]
    STEPS = [
        "f'(x) = 2*x - 4, f'(1) = -2",
        "l: y = -2*(x - 1) - 6 = -2*x - 4",
        "g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)",
        "g'(1) = 1*(-6) + (-1)*(-2) = -4",
        "m: y = -4*(x - 1) + 6 = -4*x + 10",
    ]

    def advance(self, scene, frontier, focus=""):
        from tutor.hints.illustrator import FigureSpec, ensure_verified_scene
        return ensure_verified_scene(
            FigureSpec(curves=list(scene)), self.STEPS[:frontier], self.EQUATIONS,
            focus,
        )

    def test_f_then_l_then_l_plus_g_then_l_plus_m(self):
        scene = self.advance([], 1)
        assert [(c.label, c.role) for c in scene.curves] == [("f", "scaffold")]

        scene = self.advance(scene.curves, 2)
        # l appears the moment it is earned, while the dotted source f stays
        # long enough for the student to see the curve/tangent pair.
        assert [(c.label, c.role) for c in scene.curves] == [
            ("l", "target"), ("f", "scaffold")
        ]

        # As the next hint opens work on g'(x), the problem-given g replaces f
        # immediately; its derivative need not already be solved to show it.
        scene = self.advance(
            scene.curves, 2, "곱의 미분법으로 g'(x) 쓰기"
        )
        assert [(c.label, c.role) for c in scene.curves] == [
            ("l", "target"), ("g", "scaffold")
        ]
        assert "f(x)" not in scene.curves[1].expr
        assert "x**2 - 4*x - 3" in scene.curves[1].expr

        scene = self.advance(scene.curves, 3)
        assert [c.label for c in scene.curves] == ["l", "g"]

        scene = self.advance(scene.curves, 4)
        assert [c.label for c in scene.curves] == ["l", "g"]

        scene = self.advance(scene.curves, 5)
        assert [(c.label, c.role) for c in scene.curves] == [
            ("l", "target"), ("m", "target")
        ]

    async def test_each_verified_transition_is_published_immediately(self, db):
        """The local publisher, not the later drawing model, owns the flow."""
        from tutor.state.models import StudentState

        reference = ReferenceSolution(
            steps=[
                SolutionStep(idx=i + 1, description=description, expression=expression)
                for i, (description, expression) in enumerate(zip(
                    [
                        "f'(x)로 접선 l의 기울기 구하기",
                        "l의 방정식 쓰기",
                        "곱의 미분법으로 g'(x) 쓰기",
                        "g'(1) 계산",
                        "m의 방정식 쓰기",
                    ],
                    self.STEPS,
                ))
            ],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["differentiation"], verified=True, origin="db",
        )
        session, _, _ = build(db, [])
        session.ctx.reference = reference
        session.ctx.recognition = Recognition(
            problem_text="두 접선 l, m을 구하는 문제",
            equations=self.EQUATIONS,
            concepts=["differentiation"], confidence=1.0,
        )

        async def publish(frontier, target, *, focus=False):
            session.store.set_state(StudentState(
                current_step="t", last_correct_step=frontier, status="CORRECT"
            ))
            changed = await session._publish_verified_scene(
                Decision(Action.WAIT, 0, target, None, "verified transition"),
                session.ctx, "p1", focus_target=focus,
            )
            assert changed
            return [c.label for c in session.ctx.scene]

        assert await publish(1, 2) == ["f"]
        # l is sent immediately and f is still present; no wait for g or m.
        assert await publish(2, 3) == ["l", "f"]
        figure = [e for e in session.ws.events if e["event"] == "figure"][-1]
        assert "-2·x - 4" in figure["data"]["svg"]
        assert "x² - 4·x - 3" in figure["data"]["svg"]

        # The g' invitation swaps only the dotted scaffold.
        assert await publish(2, 3, focus=True) == ["l", "g"]
        # Earning m removes g and leaves the two target lines.
        assert await publish(5, 6) == ["l", "m"]
