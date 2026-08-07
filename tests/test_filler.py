"""Talking while thinking, without talking over the answer.

A hint takes several seconds of photo, recognition, solving, diagnosis and TTS.
Silence from something that was just speaking reads as broken rather than as
thinking, so the tutor says "음, 어디 보자" into the gap — but a filler is not
free: every syllable of it delays the real answer. So the rules it has to obey
are timing rules, and they are what these tests pin down.

    slow turn   the filler plays, and the answer follows it — never over it
    fast turn   no filler at all, because there was no silence to fill
    repeated    a phrase said once is never re-synthesised, on disk or in memory
"""

import asyncio

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.policy.engine import Action, Decision
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.filler import FILLER_PHRASES, CachedSpeech, FillerBank
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

HINT = "이 식에서 5를 어떻게 없앨 수 있을까요?"


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


class SlowSpeaker(NullSpeaker):
    """Speech takes time, which is what makes the ordering observable."""

    def __init__(self, delay=0.05):
        super().__init__()
        self.delay = delay
        self.timeline: list[str] = []

    def speak(self, text):
        import time

        self.timeline.append(f"start:{text}")
        time.sleep(self.delay)
        self.timeline.append(f"end:{text}")
        super().speak(text)


@pytest.fixture
def session(db):
    llm = EchoLLMClient()
    deps = Deps(
        settings=Settings(filler_enabled=True, filler_delay_ms=10, tts_cache_dir=None),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=SlowSpeaker(),
        fillers=FillerBank(),
        store=SessionStore(),
    )
    return Session(FakeWS(), deps)


def decision():
    return Decision(Action.SOCRATIC_QUESTION, 1, 1, None, "test")


class TestTiming:
    async def test_a_slow_turn_gets_a_filler_before_the_answer(self, session):
        session._start_filler()
        await asyncio.sleep(0.08)  # the pipeline is still thinking
        await session._deliver(decision(), HINT)

        said = session.deps.speaker.spoken
        assert len(said) == 2, said
        assert said[0] in FILLER_PHRASES
        assert said[1] == HINT

    async def test_the_answer_never_starts_before_the_filler_finishes(self, session):
        """Overlapping audio is worse than silence: the student hears neither."""
        session._start_filler()
        await asyncio.sleep(0.08)
        await session._deliver(decision(), HINT)

        timeline = session.deps.speaker.timeline
        assert timeline[1].startswith("end:"), timeline
        assert timeline[2] == f"start:{HINT}", timeline

    async def test_a_fast_turn_says_nothing_extra(self, session):
        """Thinking beat the delay, so there was no silence to fill — and a
        filler here would be pure added latency."""
        session.deps.settings.filler_delay_ms = 5_000
        session._start_filler()
        await session._deliver(decision(), HINT)

        assert session.deps.speaker.spoken == [HINT]

    async def test_a_silent_turn_does_not_leave_a_filler_running(self, session):
        session._start_filler()
        await session._settle_filler()
        assert session._filler is None

    async def test_disabled_means_disabled(self, session):
        session.deps.settings.filler_enabled = False
        session._start_filler()
        await asyncio.sleep(0.05)
        await session._deliver(decision(), HINT)

        assert session.deps.speaker.spoken == [HINT]

    async def test_no_bank_means_no_filler(self, session):
        session.deps.fillers = None
        session._start_filler()
        await asyncio.sleep(0.05)
        await session._deliver(decision(), HINT)

        assert session.deps.speaker.spoken == [HINT]

    async def test_a_broken_filler_does_not_take_the_lesson_with_it(self, session):
        class Exploding:
            def pick(self):
                raise RuntimeError("no phrases today")

        session.deps.fillers = Exploding()
        session._start_filler()
        await asyncio.sleep(0.05)
        await session._deliver(decision(), HINT)

        assert session.deps.speaker.spoken == [HINT]


class TestBank:
    def test_it_does_not_repeat_itself_back_to_back(self):
        bank = FillerBank()
        picks = [bank.pick() for _ in range(30)]
        assert all(a != b for a, b in zip(picks, picks[1:]))

    def test_every_phrase_narrates_and_stays_bounded(self):
        """"~하고 있어요" narration, not a clipped "어디 보자" — but still one
        short sentence: it plays INSIDE the wait, never instead of the answer,
        and its TTS is cached so length costs nothing at speak time."""
        assert all(len(p) <= 22 for p in FILLER_PHRASES), FILLER_PHRASES
        assert all(p.endswith(("요.", "게요.")) for p in FILLER_PHRASES)

    def test_an_empty_bank_picks_nothing_rather_than_crashing(self):
        assert FillerBank(phrases=()).pick() == ""


class CountingSpeaker(NullSpeaker):
    def __init__(self):
        super().__init__(audio=b"ID3-audio")
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        return super().synthesize(text)


class TestCache:
    def test_a_repeated_phrase_is_synthesised_once(self):
        inner = CountingSpeaker()
        speech = CachedSpeech(inner, cacheable=FILLER_PHRASES)
        for _ in range(5):
            speech.speak(FILLER_PHRASES[0])
        assert inner.calls == 1
        assert speech.hits == 4 and speech.misses == 1
        assert len(inner.played) == 5, "every play must still reach the speaker"

    def test_a_hint_is_never_cached(self):
        """Hints belong to one student at one step. Caching them would grow
        without bound and could put one student's hint in another's ear."""
        inner = CountingSpeaker()
        speech = CachedSpeech(inner, cacheable=FILLER_PHRASES)
        for _ in range(3):
            speech.speak(HINT)
            speech.synthesize(HINT)
        assert inner.spoken == [HINT] * 3, "the hint must still be spoken"
        assert inner.calls == 3, "and re-synthesised every time"
        assert (speech.hits, speech.misses) == (0, 0), "the cache never saw it"

    def test_it_survives_a_restart(self, tmp_path):
        first = CountingSpeaker()
        CachedSpeech(first, FILLER_PHRASES, tmp_path, voice="eve").warm()
        assert first.calls == len(FILLER_PHRASES)

        second = CountingSpeaker()
        speech = CachedSpeech(second, FILLER_PHRASES, tmp_path, voice="eve")
        assert speech.synthesize(FILLER_PHRASES[0]) == b"ID3-audio"
        assert second.calls == 0, "the phrase was on disk from the previous run"

    def test_a_different_voice_does_not_reuse_the_old_one(self, tmp_path):
        CachedSpeech(CountingSpeaker(), FILLER_PHRASES, tmp_path, voice="eve").warm()
        other = CountingSpeaker()
        CachedSpeech(other, FILLER_PHRASES, tmp_path, voice="leo").synthesize(FILLER_PHRASES[0])
        assert other.calls == 1

    def test_warming_reports_what_is_ready(self, tmp_path):
        speech = CachedSpeech(CountingSpeaker(), FILLER_PHRASES, tmp_path)
        assert speech.warm() == len(FILLER_PHRASES)

    def test_a_tts_outage_costs_the_filler_and_nothing_else(self, tmp_path):
        class Broken(NullSpeaker):
            def synthesize(self, text):
                raise RuntimeError("tts is down")

        speech = CachedSpeech(Broken(), FILLER_PHRASES, tmp_path)
        assert speech.warm() == 0  # no exception escapes

    def test_an_unwritable_cache_directory_is_not_fatal(self, tmp_path):
        blocked = tmp_path / "file"
        blocked.write_text("in the way")
        speech = CachedSpeech(CountingSpeaker(), FILLER_PHRASES, blocked / "cache")
        assert speech.synthesize(FILLER_PHRASES[0]) == b"ID3-audio"
