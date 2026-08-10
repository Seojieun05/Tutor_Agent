"""Reading the worksheet is swappable; everything downstream of it is not.

The tutor kept saying "카메라에 다시 보여 줄래요?" and there was no way to tell
whether the photo had arrived, whether the model had read it, or whether the
gate in front of the model's answer was simply too strict. These tests pin the
three things that made that diagnosable: the knobs are readable from the
environment, the frame can be written to disk, and a broken vision provider
degrades to the chat model instead of taking the lesson down with it.
"""

import logging

import pytest

from tutor.config import Settings, load_settings
from tutor.llm.client import LLMError
from tutor.llm.echo import EchoLLMClient
from tutor.server.app import build_vision_llm
from tutor.server.session import Deps, Session
from tutor.vision.recognizer import Recognizer

JPEG = b"\xff\xd8" + b"frame" * 512


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A clean environment: no .env, no inherited keys."""
    for name in (
        "CAPTURE_TIMEOUT_S", "SAVE_CAPTURES_DIR", "VISION_PROVIDER",
        "GOOGLE_API_KEY", "GEMINI_VISION_MODEL", "XAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "absent.env"


class TestSettings:
    def test_capture_timeout_is_readable_from_the_environment(self, env, monkeypatch):
        # It was hardcoded at 5.0, which is under the time a phone needs to push
        # a UXGA frame over a busy 2.4 GHz link — and unreachable to the operator.
        monkeypatch.setenv("CAPTURE_TIMEOUT_S", "25")
        assert load_settings(env).capture_timeout_s == 25.0

    def test_capture_timeout_default_leaves_room_for_a_real_board(self, env):
        assert load_settings(env).capture_timeout_s >= 10.0

    def test_captures_are_not_saved_unless_asked(self, env):
        assert load_settings(env).save_captures_dir is None

    def test_save_captures_dir_is_a_path(self, env, monkeypatch, tmp_path):
        monkeypatch.setenv("SAVE_CAPTURES_DIR", str(tmp_path / "shots"))
        assert load_settings(env).save_captures_dir == tmp_path / "shots"

    def test_vision_provider_defaults_to_grok_and_is_case_insensitive(self, env, monkeypatch):
        assert load_settings(env).vision_provider == "grok"
        monkeypatch.setenv("VISION_PROVIDER", "Gemini")
        assert load_settings(env).vision_provider == "gemini"


class TestProviderChoice:
    def test_grok_is_the_default_eye(self):
        llm = EchoLLMClient()
        assert build_vision_llm(Settings(xai_api_key="k"), llm) is llm

    def test_echo_mode_never_reaches_for_a_second_provider(self):
        # No key at all is the demo path: it must not try to open a Gemini client.
        llm = EchoLLMClient()
        settings = Settings(vision_provider="gemini", google_api_key="g")
        assert settings.echo_mode
        assert build_vision_llm(settings, llm) is llm

    def test_a_missing_google_key_degrades_to_grok_instead_of_failing(self, caplog):
        llm = EchoLLMClient()
        settings = Settings(xai_api_key="k", vision_provider="gemini", google_api_key="")
        with caplog.at_level(logging.ERROR):
            assert build_vision_llm(settings, llm) is llm
        # Silently reading with the wrong model would be worse than not starting.
        assert "gemini" in caplog.text.lower()

    def test_gemini_client_refuses_to_exist_without_a_key(self):
        from tutor.llm.gemini import GeminiClient

        with pytest.raises(LLMError, match="GOOGLE_API_KEY"):
            GeminiClient(Settings(vision_provider="gemini", google_api_key=""))

    def test_only_the_recognizer_changes_eyes(self, db):
        """The solver, estimator and hint generator stay on the chat model."""
        from tutor.server.app import make_deps

        chat, eyes = EchoLLMClient(), EchoLLMClient()
        deps = make_deps(Settings(), db, chat, None, None, vision_llm=eyes)
        assert deps.recognizer.llm is eyes
        assert deps.solver.llm is chat
        assert deps.estimator.llm is chat
        assert deps.hint_gen.llm is chat


class TestEvalProvider:
    """Grading a spoken answer can move to flash: measured 3.6s against Grok's
    5.4s, and the verdict feeds the same deterministic policy either way."""

    def test_grading_stays_on_grok_by_default(self):
        from tutor.server.app import build_eval_llm

        llm = EchoLLMClient()
        assert build_eval_llm(Settings(xai_api_key="k"), llm) is llm

    def test_the_knob_reads_from_the_environment(self, env, monkeypatch):
        monkeypatch.setenv("EVAL_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_EVAL_MODEL", "gemini-3.6-flash")
        s = load_settings(env)
        assert (s.eval_provider, s.gemini_eval_model) == ("gemini", "gemini-3.6-flash")

    def test_a_missing_key_degrades_instead_of_refusing_to_start(self, caplog):
        from tutor.server.app import build_eval_llm

        llm = EchoLLMClient()
        settings = Settings(xai_api_key="k", eval_provider="gemini", google_api_key="")
        with caplog.at_level(logging.ERROR):
            assert build_eval_llm(settings, llm) is llm
        assert "EVAL_PROVIDER" in caplog.text

    def test_only_the_evaluator_changes_judge(self, db):
        """Swapping the grader must not move the solver, the diagnosis or the
        hint voice — the eval model judges, and judges only."""
        from tutor.server.app import make_deps

        llm, eval_llm = EchoLLMClient(), EchoLLMClient()
        deps = make_deps(Settings(), db, llm, None, None, eval_llm=eval_llm)
        assert deps.evaluator.llm is eval_llm


class TestEstimateProvider:
    """Diagnosing written work sits on the WORK_CHECK critical path: the
    student asked a yes/no question and waits on this one call for the
    verdict. Same knob shape as eval; the sympy arithmetic check still
    outranks whatever model runs."""

    def test_diagnosis_stays_on_grok_by_default(self):
        from tutor.server.app import build_estimate_llm

        llm = EchoLLMClient()
        assert build_estimate_llm(Settings(xai_api_key="k"), llm) is llm

    def test_the_knob_reads_from_the_environment(self, env, monkeypatch):
        monkeypatch.setenv("ESTIMATE_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_ESTIMATE_MODEL", "gemini-3.6-flash")
        s = load_settings(env)
        assert (s.estimate_provider, s.gemini_estimate_model) == (
            "gemini", "gemini-3.6-flash"
        )

    def test_a_missing_key_degrades_instead_of_refusing_to_start(self, caplog):
        from tutor.server.app import build_estimate_llm

        llm = EchoLLMClient()
        settings = Settings(
            xai_api_key="k", estimate_provider="gemini", google_api_key=""
        )
        with caplog.at_level(logging.ERROR):
            assert build_estimate_llm(settings, llm) is llm
        assert "ESTIMATE_PROVIDER" in caplog.text

    def test_only_the_estimator_changes_diagnostician(self, db):
        from tutor.server.app import make_deps

        llm, estimate_llm = EchoLLMClient(), EchoLLMClient()
        deps = make_deps(Settings(), db, llm, None, None, estimate_llm=estimate_llm)
        assert deps.estimator.llm is estimate_llm
        assert deps.evaluator.llm is llm
        assert deps.hint_gen.llm is llm
        assert deps.solver.llm is llm


class TestHintProvider:
    """What the tutor SAYS can move to Gemini for its LearnLM tuning. What
    decides how much it may say cannot move anywhere."""

    def test_hints_stay_on_grok_by_default(self):
        from tutor.server.app import build_hint_llm

        llm = EchoLLMClient()
        assert build_hint_llm(Settings(xai_api_key="k"), llm) is llm

    def test_the_hint_model_defaults_to_flash(self, env):
        # Measured: pro phrases better but takes 8-12s against flash's 4-6s,
        # which turns an 11s turn into 18s. The student is waiting through it.
        assert load_settings(env).gemini_hint_model == "gemini-3.6-flash"

    def test_the_hint_model_is_readable_from_the_environment(self, env, monkeypatch):
        monkeypatch.setenv("GEMINI_HINT_MODEL", "gemini-3.1-pro-preview")
        assert load_settings(env).gemini_hint_model == "gemini-3.1-pro-preview"

    def test_a_missing_key_degrades_instead_of_refusing_to_start(self, caplog):
        from tutor.server.app import build_hint_llm

        llm = EchoLLMClient()
        settings = Settings(xai_api_key="k", hint_provider="gemini", google_api_key="")
        with caplog.at_level(logging.ERROR):
            assert build_hint_llm(settings, llm) is llm
        assert "HINT_PROVIDER" in caplog.text

    def test_only_the_hint_generator_changes_voice(self, db):
        """Swapping the writer must not move the solver or the diagnosis: a
        model that has never seen the reference solution cannot leak it."""
        from tutor.server.app import make_deps

        chat, writer = EchoLLMClient(), EchoLLMClient()
        deps = make_deps(Settings(), db, chat, None, None, hint_llm=writer)
        assert deps.hint_gen.llm is writer
        assert deps.solver.llm is chat
        assert deps.estimator.llm is chat
        assert deps.recognizer.llm is chat

    def test_the_leak_guard_does_not_move_with_the_model(self, db):
        """The guard is deterministic and runs on whatever the writer wrote."""
        from tutor.hints.generator import HintGenerator
        from tutor.knowledge.models import (
            Answer, MatchResult, ReferenceSolution, SolutionStep, Tier,
        )
        from tutor.policy.engine import Action, Decision
        from tutor.vision.recognizer import Recognition

        class Leaker:
            """A "better" model that helpfully gives away the answer."""

            def run_with_tools(self, **kw):
                from tutor.hints.generator import PhrasedHint

                return PhrasedHint(hint="x = 5 니까 그렇게 하면 돼요.")

            complete_json = run_with_tools

        reference = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="양변에서 5를 뺀다", expression="3*x = 15")],
            final_answer=Answer(kind="SCALAR", value="5"),
            concepts=["linear_equation"], verified=True, origin="db",
        )
        text = HintGenerator(Leaker(), db).generate(
            Decision(Action.SOCRATIC_QUESTION, 1, 1, None, "t"),
            MatchResult(tier=Tier.NEW, concepts=["linear_equation"]),
            reference,
            Recognition(problem_text="3x + 5 = 20", equations=["3*x + 5 = 20"]),
            [],
        )
        assert "x = 5" not in text and "5 니까" not in text


class TestLearnLMPrompt:
    """The prompt is written on Google's LearnLM guide: PARTS (Persona, Act,
    Recipient, Theme, Structure) plus its five learning-science principles."""

    def test_it_states_the_role_before_the_prohibitions(self):
        from tutor.hints.generator import _PHRASE_SYSTEM

        # "Define the role and tone up front" — a model told what to do holds
        # the line better than one told only what to avoid.
        assert _PHRASE_SYSTEM.index("PERSONA") < _PHRASE_SYSTEM.index("NEVER")

    @pytest.mark.parametrize("part", ["PERSONA", "ACT", "RECIPIENT", "THEME", "STRUCTURE"])
    def test_every_part_of_parts_is_present(self, part):
        from tutor.hints.generator import _PHRASE_SYSTEM

        assert part in _PHRASE_SYSTEM

    def test_the_answer_is_still_forbidden(self):
        from tutor.hints.generator import _PHRASE_SYSTEM

        assert "Never state the final answer" in _PHRASE_SYSTEM

    def test_it_asks_for_one_idea_per_turn(self):
        """LearnLM's "manage cognitive load", and the guide's own math-coach
        exemplar: "use one step per turn"."""
        from tutor.hints.generator import _PHRASE_SYSTEM

        assert "One question, not two" in _PHRASE_SYSTEM


class TestFrameDump:
    def _session(self, settings, db):
        deps = Deps(
            settings=settings,
            recognizer=Recognizer(EchoLLMClient()),
            matcher=None, solver=None, estimator=None, hint_gen=None,
            transcriber=None, speaker=None,
        )
        return Session(ws=None, deps=deps)

    def test_nothing_is_written_by_default(self, tmp_path, db):
        self._session(Settings(), db)._save_capture(JPEG)
        assert not list(tmp_path.iterdir())

    def test_the_frame_lands_where_it_was_asked_to(self, tmp_path, db):
        target = tmp_path / "shots"
        self._session(Settings(save_captures_dir=target), db)._save_capture(JPEG)
        written = list(target.glob("*.jpg"))
        assert len(written) == 1
        assert written[0].read_bytes() == JPEG

    def test_an_unwritable_directory_does_not_end_the_lesson(self, tmp_path, db):
        blocked = tmp_path / "file-not-a-dir"
        blocked.write_text("in the way")
        # No exception: the student is mid-problem and a full disk is not their problem.
        self._session(Settings(save_captures_dir=blocked / "shots"), db)._save_capture(JPEG)


class TestStandby:
    """A key can list a model it has no quota for; that only shows up on the
    first real call, mid-lesson. gemini-3.1-pro-preview does exactly this on a
    free-tier key: 429 RESOURCE_EXHAUSTED, limit: 0."""

    def _pair(self, **kw):
        from tutor.llm.fallback import FallbackLLM

        class Dead:
            def complete_json(self, **_):
                raise RuntimeError("429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-3.1-pro")

            run_with_tools = complete_json

        class Alive:
            def __init__(self):
                self.calls = 0

            def complete_json(self, **_):
                self.calls += 1
                return "standby"

            run_with_tools = complete_json

        alive = Alive()
        return FallbackLLM(Dead(), alive, label="HINT_PROVIDER", **kw), alive

    def test_a_dead_primary_does_not_cost_the_turn(self):
        llm, standby = self._pair()
        assert llm.run_with_tools(purpose="phrase") == "standby"
        assert standby.calls == 1

    def test_the_reason_is_logged_in_a_form_you_can_act_on(self, caplog):
        llm, _ = self._pair()
        with caplog.at_level(logging.ERROR):
            llm.run_with_tools(purpose="phrase")
        assert "not on your plan" in caplog.text

    def test_a_dead_primary_is_paid_for_once_not_every_turn(self, caplog):
        """A failed call still costs its round trip, and a student is waiting."""
        llm, _ = self._pair(cooldown_s=60)
        tried = []
        llm.primary.complete_json = lambda **_: (tried.append(1), 1 / 0)[1]
        for _ in range(5):
            llm.run_with_tools(purpose="phrase")
        assert len(tried) <= 1, "the standby cooldown did not hold"

    def test_it_goes_back_to_the_primary_once_the_cooldown_passes(self):
        from tutor.llm.fallback import FallbackLLM

        class Flaky:
            def __init__(self):
                self.fail = True

            def run_with_tools(self, **_):
                if self.fail:
                    raise RuntimeError("boom")
                return "primary"

            complete_json = run_with_tools

        flaky = Flaky()
        llm = FallbackLLM(flaky, type("S", (), {"run_with_tools": lambda s, **k: "standby",
                                                "complete_json": lambda s, **k: "standby"})(),
                          cooldown_s=0.0)
        assert llm.run_with_tools() == "standby"
        flaky.fail = False
        assert llm.run_with_tools() == "primary"

    def test_a_healthy_primary_is_never_bypassed(self):
        from tutor.llm.fallback import FallbackLLM

        good = type("P", (), {"run_with_tools": lambda s, **k: "primary",
                              "complete_json": lambda s, **k: "primary"})()
        bad = type("S", (), {"run_with_tools": lambda s, **k: pytest.fail("standby used"),
                             "complete_json": lambda s, **k: pytest.fail("standby used")})()
        llm = FallbackLLM(good, bad)
        assert llm.run_with_tools() == "primary"


class TestVertexBackend:
    """Two doors to the same models. Which one you have is a billing fact:
    an AI Studio key spends prepaid credits and 429s with "prepayment credits
    are depleted" when they run out; a Cloud project spends its own."""

    def test_no_vertex_project_means_the_api_key_path(self, env, monkeypatch):
        monkeypatch.delenv("VERTEX_PROJECT", raising=False)
        assert load_settings(env).vertex_project == ""

    def test_the_project_is_read_from_the_environment(self, env, monkeypatch):
        monkeypatch.setenv("VERTEX_PROJECT", "gen-lang-client-0586206831")
        assert load_settings(env).vertex_project == "gen-lang-client-0586206831"

    def test_the_location_defaults_to_global(self, env):
        # Not tidiness: gemini-3.1-pro-preview is only published in `global`
        # and answers 404 in us-central1.
        assert load_settings(env).vertex_location == "global"

    def test_vertex_needs_no_api_key(self, monkeypatch):
        """ADC, not a key — so an empty GOOGLE_API_KEY must not refuse."""
        from google import genai

        from tutor.llm.gemini import GeminiClient

        seen = {}
        monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw) or object())

        client = GeminiClient(
            Settings(google_api_key="", vertex_project="p", vertex_location="global"),
            model="gemini-3.1-pro-preview",
        )
        assert seen == {"vertexai": True, "project": "p", "location": "global"}
        assert "vertex" in client.backend

    def test_a_vertex_project_wins_over_a_stale_api_key(self, monkeypatch):
        """Both configured means the project was the deliberate choice — and
        the key is the one that ran out of credits."""
        from google import genai

        from tutor.llm.gemini import GeminiClient

        seen = {}
        monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw) or object())

        GeminiClient(Settings(google_api_key="stale", vertex_project="p"))
        assert "api_key" not in seen and seen["vertexai"] is True

    def test_without_either_it_says_what_is_missing(self):
        from tutor.llm.gemini import GeminiClient

        with pytest.raises(LLMError, match="VERTEX_PROJECT"):
            GeminiClient(Settings(google_api_key="", vertex_project=""))
