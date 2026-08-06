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
        # It was hardcoded at 5.0, which is under the time a XIAO needs to push
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
        from tutor.llm.gemini_vision import GeminiVisionClient

        with pytest.raises(LLMError, match="GOOGLE_API_KEY"):
            GeminiVisionClient(Settings(vision_provider="gemini", google_api_key=""))

    def test_only_the_recognizer_changes_eyes(self, db):
        """The solver, estimator and hint generator stay on the chat model."""
        from tutor.server.app import make_deps

        chat, eyes = EchoLLMClient(), EchoLLMClient()
        deps = make_deps(Settings(), db, chat, None, None, vision_llm=eyes)
        assert deps.recognizer.llm is eyes
        assert deps.solver.llm is chat
        assert deps.estimator.llm is chat
        assert deps.hint_gen.llm is chat


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
