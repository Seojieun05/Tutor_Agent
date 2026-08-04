"""STT quality gate: what reaches the tutor pipeline, and what gets a re-ask.

Room noise transcribed as foreign glyphs, silence and pure hesitation sounds
are not maths answers. Grading them would burn an LLM call and — worse —
resolve the pending question with evidence the student never gave.
"""

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.server.session import RETRY_PROMPTS, Deps, ProblemContext, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import Transcript, classify_transcript
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer


class TestClassifier:
    @pytest.mark.parametrize(
        "text",
        [
            "5를 빼면 돼요",
            "x는 5예요",
            "5",  # a bare number is a real answer
            "2 더하기 3",
            "삼각형 ABC의 넓이요",
            "음... 5를 빼요",  # a filler in front of real content is fine
            "I think it is five",
            "네",  # short, but an answer
        ],
    )
    def test_ok(self, text):
        assert classify_transcript(text) == "ok"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "...",
            "???",
            "谢谢你的帮助",  # Chinese: not what a Korean student said
            "ありがとうございます",
            "спасибо",
            "ㅇㅇㅇ",  # bare jamo, not speech
            "。。。",
        ],
    )
    def test_unclear(self, text):
        assert classify_transcript(text) == "unclear"

    @pytest.mark.parametrize(
        "text",
        ["음", "어...", "음, 어", "어어어", "흠...", "um", "uh, um", "Hmm.", "그, 저"],
    )
    def test_filler_only(self, text):
        assert classify_transcript(text) == "filler_only"

    def test_a_stray_glyph_does_not_reject_a_real_answer(self):
        assert classify_transcript("5를 빼면 돼요 好") == "ok"


REFERENCE = ReferenceSolution(
    steps=[SolutionStep(idx=1, description="양변에서 5를 뺀다", expression="3*x = 15")],
    final_answer=Answer(kind="SCALAR", value="5"),
    concepts=["linear_equation"],
    verified=True,
    origin="db",
)


class ScriptedTranscriber:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000):
        return Transcript(text=self.texts.pop(0) if self.texts else "", language="ko")


class FakeWS:
    def __init__(self):
        self.events = []

    async def send(self, raw):
        if isinstance(raw, str):
            self.events.append(raw)


def build_session(db, transcripts, verdicts=None):
    llm = EchoLLMClient({"evaluate": verdicts or []})
    speaker = NullSpeaker()
    deps = Deps(
        settings=Settings(),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(transcripts),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        store=SessionStore(),
    )
    session = Session(FakeWS(), deps)
    session.ctx = ProblemContext(
        hash="p1",
        recognition=Recognition(problem_text="3x + 5 = 20", equations=["3*x + 5 = 20"]),
        match=MatchResult(tier=Tier.EXACT, concepts=["linear_equation"], reference=REFERENCE),
        reference=REFERENCE,
    )
    session.store.set_state(StudentState(status="CONCEPT_ERROR"))
    session.store.append_hint(
        problem_hash="p1", step=1, level=1, action="SOCRATIC_QUESTION",
        hint_text="어떤 항을 옮겨야 할까요?",
    )
    return session, llm, speaker


PCM = b"\x00\x00" * 100


class TestGateInTheSession:
    async def test_noise_is_not_graded_and_the_question_stays_pending(self, db):
        session, llm, speaker = build_session(db, ["谢谢你"])
        await session._handle_utterance(PCM, 16000)

        assert speaker.spoken == [RETRY_PROMPTS["unclear"]]
        assert llm.calls == []  # nothing reached the pipeline
        # the L1 question is still waiting for a real answer
        assert session.store.pending_hint("p1") is not None
        # ...and the re-ask is not a hint: the ladder is untouched
        assert len(session.store.get_history(problem_hash="p1")) == 1

    async def test_filler_gets_the_gentle_prompt(self, db):
        session, llm, speaker = build_session(db, ["음... 어..."])
        await session._handle_utterance(PCM, 16000)

        assert speaker.spoken == ["괜찮아요, 이어서 말해 줄래요?"]
        assert llm.calls == []
        assert session.store.pending_hint("p1") is not None

    async def test_stt_failure_asks_again_instead_of_going_silent(self, db):
        class Broken:
            def transcribe(self, pcm, sample_rate=16000):
                raise RuntimeError("stt down")

        session, llm, speaker = build_session(db, [])
        session.deps.transcriber = Broken()
        await session._handle_utterance(PCM, 16000)

        assert speaker.spoken == [RETRY_PROMPTS["unclear"]]

    async def test_a_real_answer_still_gets_graded(self, db):
        session, llm, speaker = build_session(
            db,
            ["양변에서 5를 빼요"],
            verdicts=[{"verdict": "CORRECT", "feedback": "맞아요!",
                       "misconception": None, "status": "CORRECT"}],
        )
        await session._handle_utterance(PCM, 16000)

        assert llm.calls.count("evaluate") == 1
        assert session.store.get_history(problem_hash="p1")[0].effective is True

    async def test_rejected_utterance_is_not_kept_as_evidence(self, db):
        """A rejected transcript must not leak into the next estimate."""
        session, _, _ = build_session(db, ["ㅇㅇㅇ"])
        await session._handle_utterance(PCM, 16000)
        assert session.last_transcript is None
