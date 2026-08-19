"""What the student hears WHILE the tutor thinks.

The pipeline's wall clock is other people's servers; what this file pins down
is the part we control — that the wait is filled with something that belongs
to this turn, and that none of it ever delays or outlives the real answer:

    ANSWER        the student's words echoed back ("5라고 했네요...")
    WORK_CHECK    "네, 지금 쓴 풀이를 한번 볼게요" before the camera has moved
    new problem   the problem read back, once recognition knows it

Echo mode makes every model call instant, so these tests force the ordering
through the filler machinery itself rather than through timing luck.
"""

import asyncio
import json

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.session import (
    READOUT_CLOSERS,
    READOUT_OPENER,
    WORK_CHECK_OPENERS,
    Deps,
    ProblemContext,
    Session,
    answer_core,
    preflight_fallback,
)
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.filler import FillerBank
from tutor.speech.intent import IntentClassifier
from tutor.speech.stt import Transcript
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

JPEG = b"\xff\xd8" + b"page" * 2048
PCM = b"\x00\x00" * 100

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


class PhoneWS:
    def __init__(self):
        self.events: list[dict] = []
        self.session: Session | None = None

    async def send(self, raw):
        if not isinstance(raw, str):
            return
        event = json.loads(raw)
        self.events.append(event)
        if event["event"] == "capture_request":
            await self.session.on_frame(
                encode_image(JPEG, ImageHeader(capture_id=event["data"]["capture_id"]))
            )


class ScriptedTranscriber:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000) -> Transcript:
        return Transcript(text=self.texts.pop(0) if self.texts else "")


def build(db, *utterances: str, llm_responses: dict | None = None, delay_ms: int = 0):
    llm = EchoLLMClient(llm_responses or {})
    speaker = NullSpeaker()
    ws = PhoneWS()
    deps = Deps(
        settings=Settings(capture_timeout_s=2.0, filler_delay_ms=delay_ms),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(*utterances),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        classifier=IntentClassifier(),
        fillers=FillerBank(),
        store=SessionStore(),
    )
    session = Session(ws, deps)
    ws.session = session
    return session, llm, speaker, ws


def with_problem(session: Session) -> None:
    session.ctx = ProblemContext(
        hash="p1",
        recognition=RECOGNITION,
        match=MatchResult(tier=Tier.EXACT, concepts=["linear_equation"], reference=REFERENCE),
        reference=REFERENCE,
    )


def ask_l1(session: Session) -> None:
    session.store.set_state(
        StudentState(status="CONCEPT_ERROR", last_correct_step=0,
                     misconception="sign_flip_on_move")
    )
    session.store.append_hint(
        problem_hash="p1", step=1, level=1, action="SOCRATIC_QUESTION",
        hint_text="어떤 항을 반대쪽으로 옮겨야 할까요?",
    )


# --- the echo (ANSWER turns) -------------------------------------------------


class TestAnswerCore:
    """What gets echoed is the VALUE, never the politeness around it."""

    def test_polite_endings_are_stripped(self):
        assert answer_core("5예요") == "5"
        assert answer_core("마이너스 3이요.") == "마이너스 3"
        assert answer_core("x는 2요") == "x는 2"
        assert answer_core("8입니다") == "8"

    def test_bare_values_pass_through(self):
        assert answer_core("15") == "15"
        assert answer_core("x") == "x"

    def test_verbs_and_questions_are_not_echoed(self):
        """"양변에서 5를 빼요" quoted back sounds wrong at any phrasing —
        and a question must be answered, not repeated."""
        assert answer_core("양변에서 5를 빼요") is None
        assert answer_core("왜 5를 빼야 해요?") is None
        assert answer_core("모르겠어요") is None

    def test_long_speeches_are_not_echoed(self):
        assert answer_core("그러니까 제 생각에는 양변에서 5를 빼고 나서 3으로 나누면 될 것 같아요") is None
        assert answer_core("   ") is None


def test_echo_frames_rotate_instead_of_repeating():
    import random

    from tutor.speech.filler import ECHO_FRAMES, FillerBank

    bank = FillerBank(rng=random.Random(7))
    said = [bank.echo("5") for _ in range(6)]
    # the value is spelled: this line is heard, never read, and a digit is
    # something the TTS engine still has to interpret
    assert all("오" in s for s in said)
    assert all(any(s == f.format(v="오") for f in ECHO_FRAMES) for s in said)
    # the frame never repeats back-to-back — repetition is what sounds mechanical
    frames = [next(f for f in ECHO_FRAMES if s == f.format(v="오")) for s in said]
    assert all(a != b for a, b in zip(frames, frames[1:]))


async def test_an_answer_turn_opens_with_the_echo(db):
    from tutor.speech.filler import ECHO_FRAMES

    session, llm, speaker, ws = build(
        db, "5예요",
        llm_responses={"evaluate": [{"intent": "ANSWER", "verdict": "CORRECT",
                                     "feedback": "맞아요!", "misconception": None,
                                     "status": "CORRECT"}]},
    )
    with_problem(session)
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    # the opener is the VALUE in one of the rotating frames — never "5예요라고"
    assert speaker.spoken[0] in {f.format(v="오") for f in ECHO_FRAMES}
    assert speaker.spoken[1].startswith("맞아요!")   # then the real reply
    assert llm.calls.count("evaluate") == 1


# --- the opener (WORK_CHECK turns) ------------------------------------------


async def test_a_work_check_opens_by_saying_it_is_looking(db):
    session, llm, speaker, ws = build(db, "풀이 맞아?")
    with_problem(session)
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert speaker.spoken[0] in WORK_CHECK_OPENERS
    assert len(speaker.spoken) >= 2                  # the real turn still spoke


async def test_a_work_check_shows_its_stages_and_speaks_one_filler(db):
    """The voice diet: progress goes on the SCREEN (stage events), and the only
    filler the turn speaks is its opener. The old spoken narration
    ("풀이를 다 읽었어요…") stays retired."""
    from tutor.speech.filler import WORK_CHECK_NARRATIONS

    session, llm, speaker, ws = build(db, "풀이 맞아?")
    with_problem(session)
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)
    await asyncio.sleep(0)

    stages = [e["data"]["text"] for e in ws.events if e["event"] == "stage"]
    assert "쓴 풀이를 읽고 있어요" in stages
    assert "풀이를 살펴보고 있어요" in stages
    assert not [s for s in speaker.spoken if s in WORK_CHECK_NARRATIONS]
    fillers = [s for s in speaker.spoken if s in WORK_CHECK_OPENERS]
    assert len(fillers) == 1                                 # one filler, total


async def test_a_sloppy_reread_does_not_turn_a_work_check_into_a_readout(db):
    """The regression behind "풀이 확인이 말이 많다": handwriting re-read badly
    enough to miss the same-problem match used to take the NEW-problem branch
    mid work check — rewriting the card and stacking the 3-line readout on the
    opener. A work check speaks its ONE opener; the screen does the rest."""
    from tutor.speech.filler import FILLER_PHRASES

    session, llm, speaker, ws = build(
        db, "풀이 맞아?",
        llm_responses={"recognize": [dict(
            problem_text="방정식을 푸시오: x**5 - 7 = 0",   # not the problem on ctx
            equations=["x**5 - 7 = 0"], student_work=["x**5 = 7"], choices=[],
            diagram_conditions=[], uncertain_regions=[], confidence=0.95,
        )]},
    )
    with_problem(session)
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)
    await asyncio.sleep(0)

    assert "problem" not in [e["event"] for e in ws.events]   # the card held still
    assert READOUT_OPENER not in speaker.spoken               # nothing read back
    assert not any(s in READOUT_CLOSERS.values() for s in speaker.spoken)
    fillers = [s for s in speaker.spoken
               if s in WORK_CHECK_OPENERS or s in FILLER_PHRASES]
    assert len(fillers) == 1                                  # one filler, total


async def test_a_hint_turn_reports_reading_then_making(db):
    session, llm, speaker, ws = build(db, "이 문제 힌트 줄래?")

    await session._handle_utterance(PCM, 16000)

    stages = [e["data"]["text"] for e in ws.events if e["event"] == "stage"]
    assert stages.index("문제를 읽고 있어요") < stages.index("힌트를 만들고 있어요")


async def test_the_hints_board_reaches_the_page(db):
    """What the tutor wrote arrives as a `board` event, in display notation,
    alongside the spoken hint. Templates carry no board, so the DB pedagogy is
    emptied here to force the phrasing model — the one path that writes."""
    db.hint_templates_for = lambda *a, **k: []
    session, llm, speaker, ws = build(
        db, "이 문제 힌트 줄래?",
        llm_responses={"phrase": [
            {"hint": "어느 항을 옮기면 x만 남을까요?",
             "board": [{"expr": "3*x + 5 = 20", "note": "이 식에서 출발"}]}
        ]},
    )

    await session._handle_utterance(PCM, 16000)

    assert "phrase" in llm.calls                        # the board-writing path ran
    board = next(e for e in ws.events if e["event"] == "board")
    # eye notation for the expression, the margin note travelling beside it
    assert board["data"]["lines"] == [
        {"expr": "3·x + 5 = 20", "note": "이 식에서 출발"}
    ]
    # written as the voice starts: the board precedes the hint's bookkeeping
    names = [e["event"] for e in ws.events]
    assert names.index("board") < names.index("hint_issued")


async def test_a_new_problem_does_not_get_the_work_narration(db):
    """First sight of a problem reads the PROBLEM back; the work-check line
    ("풀이를 다 읽었어요…") belongs to a page the tutor has already met."""
    from tutor.speech.filler import WORK_CHECK_NARRATIONS

    session, llm, speaker, ws = build(db, "이 문제 힌트 줄래?")

    await session._handle_utterance(PCM, 16000)
    await asyncio.sleep(0)

    assert not [s for s in speaker.spoken if s in WORK_CHECK_NARRATIONS]


# --- the readout (new problems) ---------------------------------------------


async def test_the_problem_card_does_not_print_the_same_equation_twice(db):
    """Live: the card showed the chain equality inside the sentence AND again
    below it. The VLM writes the statement as printed (2(a+b)) and the
    equation list in ASCII (2*(a+b)), so the raw substring test never matched.
    A bare term ("a_10") is dropped too — it states nothing."""
    session, llm, speaker, ws = build(
        db, "이 문제 힌트 줄래?",
        llm_responses={"recognize": [dict(
            problem_text="12. 등비수열 {a_n}이 2(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6 을 만족시킬 때, a_10의 값은?",
            equations=[
                "2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6",   # the sentence says it
                "a_10",                                          # states nothing
                "S_n = a_1*(r**n - 1)/(r - 1)",                  # genuinely extra
            ],
            student_work=[], choices=[], diagram_conditions=[],
            uncertain_regions=[], confidence=0.95,
        )]},
    )

    await session._handle_utterance(PCM, 16000)

    card = next(e for e in ws.events if e["event"] == "problem")
    assert card["data"]["equations"] == ["Sₙ = a₁·(rⁿ - 1)/(r - 1)"]


def test_the_first_problem_of_a_kind_gets_a_generic_opening():
    assert preflight_fallback("등비수열").startswith("등비수열 문제군요.")


async def test_a_new_problem_is_told_what_kind_it_is(db):
    """The tutor no longer reads the statement back — that ran ten seconds on
    a long problem and the finished hint had to wait it out. It names the KIND
    of problem and what to check, and the card carries the wording."""
    session, llm, speaker, ws = build(db, "이 문제 힌트 줄래?")

    await session._handle_utterance(PCM, 16000)
    await asyncio.sleep(0)

    from tutor.speech.filler import FILLER_PHRASES

    opener_at = speaker.spoken.index(READOUT_OPENER)
    line_at = next(i for i, s in enumerate(speaker.spoken) if "문제군요" in s)
    assert opener_at < line_at
    # the statement itself is never spoken now
    assert not any("일차방정식을 푸시오" in s for s in speaker.spoken)
    # ONE filler, and it is the opener doing double duty
    assert speaker.spoken.count(READOUT_OPENER) == 1
    assert not any(s in FILLER_PHRASES for s in speaker.spoken)
    assert not any(s in READOUT_CLOSERS.values() for s in speaker.spoken)
    assert speaker.spoken[-1] != READOUT_OPENER      # the hint is the last word
    # the card still carries the statement, in display notation
    card = next(e for e in ws.events if e["event"] == "problem")
    assert "일차방정식" in card["data"]["text"]


async def test_the_concept_line_is_written_once_and_reused(db):
    """The point of the whole mechanism: the first problem of a kind pays a
    background call and says the generic opening; every later problem of that
    kind reads the stored line, with no call and cached audio."""
    session, llm, speaker, ws = build(db, "이 문제 힌트 줄래?")

    await session._handle_utterance(PCM, 16000)
    assert any(s.startswith("일차방정식 문제군요. 어떤 조건이") for s in speaker.spoken)
    # the background write is a task, not part of the turn
    await asyncio.gather(*[t for t in session._tasks if not t.done()])
    stored = db.preflight_line("linear_equation")
    assert stored and stored != preflight_fallback("일차방정식")

    speaker.spoken.clear()
    session.ctx = None                               # meet the same kind again
    await session.handle_hint_request()

    assert stored in speaker.spoken
    assert llm.calls.count("preflight") == 1         # written once, reused after


# --- the race the filler must always lose ------------------------------------


async def test_a_slow_filler_never_delays_a_fast_answer(db):
    """delay 5s, echo-mode turn finishes in ms: nothing extra is ever spoken."""
    session, llm, speaker, ws = build(
        db, "5예요", delay_ms=5000,
        llm_responses={"evaluate": [{"intent": "ANSWER", "verdict": "CORRECT",
                                     "feedback": "맞아요!", "misconception": None,
                                     "status": "CORRECT"}]},
    )
    with_problem(session)
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert not any("어디 보자" in s or "볼까요" in s for s in speaker.spoken)  # echo never fired
    assert speaker.spoken[0].startswith("맞아요!")           # answer came straight out
