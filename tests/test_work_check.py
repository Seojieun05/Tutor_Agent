"""'풀이 맞아?' — the turn that has to look again.

Three utterances, three different turns over the SAME session state. The one
that used to be broken is the middle one: a hint was pending, so any speech was
graded as an answer, and the tutor re-explained its own question at a worksheet
it had never re-read. Checking work IS re-reading the worksheet.

The camera here answers on the session's own socket, which is what a phone
running tutor/web/phone.html does: capture_request in, one JPEG back.
"""

import asyncio
import json

import pytest

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import Answer, MatchResult, ReferenceSolution, SolutionStep, Tier
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.session import Deps, ProblemContext, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.intent import IntentClassifier
from tutor.speech.stt import Transcript
from tutor.speech.tts import NullSpeaker
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

PHONE_JPEG = b"\xff\xd8" + b"phone" * 2048
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

PROBLEM = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "choices": [],
    "diagram_conditions": [],
    "uncertain_regions": [],
    "confidence": 0.95,
}


def seen(student_work: list[str]) -> dict:
    """What the VLM reads off the page this time round."""
    return dict(PROBLEM, student_work=student_work)


class PhoneWS:
    """A device with a camera on the session socket, like tutor/web/phone.html."""

    def __init__(self, jpeg: bytes | None = PHONE_JPEG):
        self.events: list[dict] = []
        self.jpeg = jpeg
        self.captures = 0
        self.session: Session | None = None

    async def send(self, raw):
        if not isinstance(raw, str):
            return
        event = json.loads(raw)
        self.events.append(event)
        if event["event"] != "capture_request":
            return
        self.captures += 1
        capture_id = event["data"]["capture_id"]
        # the phone answers on the same socket the request arrived on
        await self.session.on_frame(
            encode_image(self.jpeg, ImageHeader(capture_id=capture_id))
        )

    def event_names(self) -> list[str]:
        return [e["event"] for e in self.events]


class ScriptedTranscriber:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000) -> Transcript:
        return Transcript(text=self.texts.pop(0) if self.texts else "")


def build(db, *utterances: str, llm_responses: dict | None = None):
    llm = EchoLLMClient(llm_responses or {})
    speaker = NullSpeaker()
    ws = PhoneWS()
    deps = Deps(
        settings=Settings(capture_timeout_s=2.0),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(*utterances),
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        classifier=IntentClassifier(),  # rules only: no LLM guessing in tests
        store=SessionStore(),
    )
    session = Session(ws, deps)
    ws.session = session
    session.ctx = ProblemContext(
        hash="p1",
        recognition=Recognition(**seen(["3*x = 20 + 5"])),
        match=MatchResult(tier=Tier.EXACT, concepts=["linear_equation"], reference=REFERENCE),
        reference=REFERENCE,
    )
    return session, llm, speaker, ws


def ask_l1(session: Session, step: int = 1) -> int:
    """Pretend the tutor just asked its L1 question and is waiting."""
    session.store.set_state(
        StudentState(status="CONCEPT_ERROR", last_correct_step=step - 1,
                     misconception="sign_flip_on_move")
    )
    return session.store.append_hint(
        problem_hash="p1", step=step, level=1, action="SOCRATIC_QUESTION",
        hint_text="어떤 항을 반대쪽으로 옮겨야 할까요?",
    )


# --- the fix ----------------------------------------------------------------


async def test_a_work_check_takes_a_fresh_photo_and_reads_it(db):
    """The whole point: 'is this right?' must look at what they just wrote."""
    session, llm, speaker, ws = build(
        db, "풀이 봐줘.", llm_responses={"recognize": [seen(["3*x = 15"])]}
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1
    assert llm.calls.count("recognize") == 1
    assert llm.calls.count("evaluate") == 0  # nothing was answered, so nothing graded
    assert speaker.spoken
    # the newly written line is what the diagnosis ran on
    assert session.prev_work == ["3*x = 15"]
    # same problem, so the page's problem card is NOT rewritten: the VLM
    # re-reads with small wording differences, and a card that changes on
    # every "풀이 봐줘" reads as the tutor changing its mind
    assert "problem" not in ws.event_names()


async def test_a_work_check_survives_a_pending_question(db):
    """A hint is pending — the old code graded this as an answer instead.

    The utterance is a JUDGE-THIS ("풀이 봐줘"), not a WHERE-question: asking
    where a just-given verdict went wrong is a question about that verdict and
    belongs to explain(), which is a different test in test_answer_turn.
    """
    session, llm, speaker, ws = build(db, "내 풀이 좀 봐줘.")
    ask_l1(session)
    assert session.store.pending_hint("p1") is not None

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1 and llm.calls.count("recognize") == 1


async def test_a_work_check_reuses_the_problem_it_is_already_on(db):
    """Same worksheet: no re-tagging, no re-matching, no second solve."""
    session, llm, _, _ = build(db, "내가 쓴 거 봐줘.")
    ask_l1(session)
    before = session.ctx

    await session._handle_utterance(PCM, 16000)

    assert session.ctx is before          # same context object
    assert session.ctx.reference is REFERENCE
    assert llm.calls.count("solve") == 0
    assert llm.calls.count("tag") == 0


async def test_a_chain_equality_reread_stays_the_same_problem(db):
    """The 등비수열 regression: a = b = c cannot be sympy-parsed, and an
    unparseable comparison used to mean "different problem" — resetting the
    student's state and re-solving MID work check. Inconclusive now falls
    through to the (truncation-tolerant) problem text."""
    chain = "2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6"
    known = Recognition(
        problem_text="12. 등비수열 {a_n}이 2(a_1+a_4+a_7) = a_4+a_7+a_10 = 6 을 만족시킬 때, 공비를 구하시오.",
        equations=[chain],
        student_work=[],
        confidence=0.95,
    )
    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        llm_responses={
            "recognize": [dict(
                # the re-read: same problem, spacing shifted, text cut short
                problem_text="12. 등비수열 {a_n}이\n2(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6\n을 ",
                equations=["2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6"],
                student_work=["2*(a_1+a_4+a_7) = r**3 * (a_1+a_4+a_7)"],
                choices=[], diagram_conditions=[], uncertain_regions=[],
                confidence=0.98,
            )],
            "estimate": [{"current_step": "비 세우기", "last_correct_step": 1,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": True}],
        },
    )
    session.ctx = ProblemContext(
        hash=problem_hash(known),
        recognition=known,
        match=MatchResult(tier=Tier.NEW, concepts=["geometric_sequence"],
                          reference=REFERENCE),
        reference=REFERENCE,
    )
    before = session.ctx

    await session._handle_utterance(PCM, 16000)

    assert session.ctx is before                     # SAME problem, kept
    assert llm.calls.count("solve") == 0             # no re-solve
    assert llm.calls.count("estimate") == 1          # the verdict actually ran
    assert session.store.get_state() is not None     # state survived, not reset


async def test_correct_work_is_confirmed_and_not_hinted_at(db):
    """'풀이 맞아?' is a yes/no question. When the answer is yes, say yes.

    A hint here would push them at a step they have not reached and imply
    something was wrong — and it must not enter the hint ladder either.
    """
    session, llm, speaker, _ = build(
        db,
        "풀이 맞아?",
        llm_responses={
            "recognize": [seen(["3*x = 15"])],
            "estimate": [{"current_step": "이항", "last_correct_step": 1,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": True}],
        },
    )
    before = len(session.store.get_history(problem_hash="p1"))
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert speaker.spoken == ["맞아요! 이대로 하면 돼요. 또 궁금한 게 있으면 물어봐 주세요."]
    assert llm.calls.count("phrase") == 0          # no hint was generated
    # and no hint record: the ladder must not move on a question that was answered
    assert len(session.store.get_history(problem_hash="p1")) == before + 1  # only ask_l1's
    assert session.ctx is not None                 # still on the same problem


async def test_correct_so_far_does_not_resolve_the_unattempted_next_step(db):
    session, _, _, _ = build(
        db,
        "풀이 맞아?",
        llm_responses={"recognize": [seen(["3*x = 15"])]},
    )
    hint_id = ask_l1(session, step=2)  # step 1 proven; asking about step 2

    await session._handle_utterance(PCM, 16000)

    pending = session.store.pending_hint("p1")
    assert pending is not None and pending.id == hint_id
    assert pending.step == 2 and pending.effective is None


async def test_a_wrong_line_hears_the_verdict_first(db):
    """"풀이 맞아?" is a yes/no question: when the answer is no, say NO first.

    The verdict and nothing else — never WHAT is wrong; that stays the hint's
    job and the leak guard's jurisdiction.
    """
    from tutor.server.session import WORK_CHECK_WRONG

    session, _, speaker, _ = build(
        db,
        "풀이 맞나요?",
        llm_responses={
            "recognize": [seen(["3*x = 25"])],
            "estimate": [{"current_step": "이항", "last_correct_step": 0,
                          "status": "CONCEPT_ERROR", "misconception": "sign_flip_on_move",
                          "attempt_count": 2, "previous_hint_effective": False}],
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    spoken = speaker.spoken[0]
    assert spoken.startswith(WORK_CHECK_WRONG)
    assert "3*x = 15" not in spoken and "x = 5" not in spoken


async def test_a_wrong_final_line_is_never_confirmed(db):
    """The shipped miss, reproduced from the worksheet photo it shipped on.

    Problem 5: f(x) = (x+2)(2x²−x−2), f'(1) = 8. The student's product rule is
    right but their substitution line evaluates to 2, and the LLM judge graded
    the work CORRECT anyway. The arithmetic check must outrank it: sympy says
    2 ≠ 8, so the tutor may not say 맞아요 — whatever the model thought.
    """
    derivative_ref = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="곱의 미분법을 적용한다",
                         expression="2*x**2 - x - 2 + (x + 2)*(4*x - 1)"),
            SolutionStep(idx=2, description="x = 1을 대입한다", expression="8"),
        ],
        final_answer=Answer(kind="SCALAR", value="8"),
        concepts=["differentiation"],
        verified=True,
        origin="db",
    )
    photo = dict(
        problem_text="함수 f(x) = (x+2)(2x**2-x-2)에 대하여 f'(1)의 값은?",
        equations=["f(x) = (x + 2)*(2*x**2 - x - 2)"],
        choices=["6", "7", "8", "9", "10"],
        diagram_conditions=[], uncertain_regions=[],
        student_work=["f'(x) = 2*x**2 - x - 2 + (x + 2)*(4*x - 1)",
                      "f'(1) = 2 - 1 - 2 + 3 × 1"],
        confidence=0.95,
    )
    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        llm_responses={
            "recognize": [photo],
            # the lenient judge, verbatim: work graded fully correct
            "estimate": [{"current_step": "대입", "last_correct_step": 2,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": None}],
        },
    )
    # mid-problem, reference solved: the same worksheet the photo re-captures
    known = Recognition(**photo)
    session.ctx = ProblemContext(
        hash=problem_hash(known),
        recognition=known,
        match=MatchResult(tier=Tier.NEW, concepts=["differentiation"],
                          reference=derivative_ref),
        reference=derivative_ref,
    )

    await session._handle_utterance(PCM, 16000)

    spoken = " ".join(speaker.spoken)
    assert "맞아요! 이대로 하면 돼요" not in spoken       # never confirmed
    state = session.store.get_state()
    assert state.status == "CALCULATION_ERROR"           # the check overrode the judge
    assert state.last_correct_step == 1                  # the substitution is the frontier
    assert "8" not in spoken                             # and no answer leaked with it


# --- what must NOT change ---------------------------------------------------


async def test_a_spoken_answer_still_skips_the_camera(db):
    """The latency contract: an answer is graded from the transcript alone."""
    session, llm, speaker, ws = build(
        db,
        "5예요",
        llm_responses={
            "evaluate": [{"intent": "ANSWER", "verdict": "CORRECT", "feedback": "맞아요!",
                          "misconception": None, "status": "CORRECT"}]
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 0
    assert "capture_request" not in ws.event_names()
    assert llm.calls.count("recognize") == 0 and llm.calls.count("estimate") == 0
    assert llm.calls.count("evaluate") == 1


async def test_a_plain_hint_request_does_not_grow_a_reaction(db):
    """Only a work check earns the 'here is what I saw' opener.

    No pending question here: "힌트 주세요" while one IS pending is the student
    declining to answer, which the evaluator already reads as "escalate".
    """
    session, _, speaker, ws = build(db, "힌트 주세요")
    session.store.set_state(StudentState(status="STUCK"))

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 1
    assert not speaker.spoken[0].startswith("네, 여기까지는")
    assert not speaker.spoken[0].startswith("음, 지금 쓴 줄을")


async def test_thinking_out_loud_is_left_alone(db):
    """Spec rule 7: do not interrupt a student who is working."""
    session, llm, speaker, ws = build(db, "음 그러니까 이제 이걸")
    session.store.set_state(StudentState(status="CORRECT", last_correct_step=1))

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 0
    assert speaker.spoken == []
    assert llm.calls == []
    assert session.last_transcript is None  # not evidence for the next turn either
    # the device is told a transcript arrived, but that no reply is coming
    transcript = next(e for e in ws.events if e["event"] == "transcript")
    assert transcript["data"]["wants_hint"] is False


async def test_a_work_check_sticks_to_its_problem_on_an_inconclusive_reread(db):
    """The live failure: mid-string re-read drift (a comma, a dropped 보기)
    beat both the hash and the text prefix, so the work check "changed
    problem", reset state, and answered a yes/no question with an L1 hint.
    Without POSITIVE evidence of a different problem, a work check stays."""
    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        llm_responses={
            "recognize": [dict(
                problem_text="일차방정식 3x + 5 = 20 을, 참고하여 푸시오",  # drifted read
                equations=["방정식의 해 = 미지수"],                        # unparseable junk
                student_work=["3*x = 15"], choices=[],
                diagram_conditions=[], uncertain_regions=[], confidence=0.9,
            )],
            "estimate": [{"current_step": "이항", "last_correct_step": 1,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": True}],
        },
    )
    ask_l1(session)
    before = session.ctx

    await session._handle_utterance(PCM, 16000)

    assert session.ctx is before                 # no reset on a wobbly read
    assert llm.calls.count("solve") == 0
    # the verdict actually ran — "3*x = 15" matches the reference step
    # mechanically, so it needs no LLM, just the kept context
    assert any(s.startswith("맞아요! 이대로") for s in speaker.spoken)


async def test_a_work_check_still_switches_on_clear_evidence(db):
    """Stickiness is not blindness: equations that parse and DISAGREE are a
    different problem, work check or not."""
    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        llm_responses={"recognize": [dict(
            problem_text="이차방정식을 푸시오: x**2 = 4",
            equations=["x**2 = 4"], student_work=[], choices=[],
            diagram_conditions=[], uncertain_regions=[], confidence=0.95,
        )]},
    )
    ask_l1(session)
    before = session.ctx

    await session._handle_utterance(PCM, 16000)

    assert session.ctx is not before             # genuinely new problem


async def test_a_work_check_waits_for_the_reference_instead_of_hinting(db):
    """The other half of the live failure: the solver was still writing the
    reference, and the work check shrugged into a first hint from concepts.
    A yes/no question waits for its measuring stick and answers."""
    import threading

    gate = threading.Event()

    async def slow_solve():
        await asyncio.to_thread(gate.wait, 5)
        return REFERENCE

    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        llm_responses={
            "recognize": [seen(["3*x = 15", "x = 5"])],
            "estimate": [{"current_step": "완료", "last_correct_step": 2,
                          "status": "CORRECT", "misconception": None,
                          "attempt_count": 1, "previous_hint_effective": True}],
        },
    )
    ask_l1(session)
    session.ctx.reference = None                          # solver still writing
    session.ctx.solving = asyncio.create_task(slow_solve())

    turn = asyncio.create_task(session._handle_utterance(PCM, 16000))
    await asyncio.sleep(0.05)
    assert not turn.done()                                # waiting, not hinting
    gate.set()
    await turn

    spoken = " ".join(speaker.spoken)
    assert "맞아요" in spoken                              # the verdict, not an L1 question


# --- the evaluator's safety net ---------------------------------------------


async def test_the_evaluator_can_redirect_an_answer_to_a_work_check(db):
    """The net's one remaining live path: the answer-shaped fast lane swallowed
    a transcript that DOES name the work ("풀이 5 맞아?"), the evaluator sees it
    is not an attempt, and the turn goes to the camera after all."""
    session, llm, speaker, ws = build(
        db,
        "풀이 5 맞아?",   # ≤4 tokens with a digit → routed as an answer by rule
        llm_responses={
            "evaluate": [{"intent": "WORK_CHECK", "verdict": "UNCLEAR", "feedback": "",
                          "misconception": None, "status": None}],
            "recognize": [seen(["3*x = 15"])],
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert llm.calls.count("evaluate") == 1   # graded first...
    assert ws.captures == 1                   # ...then redirected to the camera
    assert llm.calls.count("recognize") == 1
    # the turn is decided by the page, not by the discarded UNCLEAR verdict:
    # the photo shows step 1 done, so that is the state and the hint that got
    # them there is resolved as having worked
    assert session.store.get_state().last_correct_step == 1
    assert session.store.get_history(problem_hash="p1")[0].effective is True


async def test_an_evaluator_work_check_without_the_words_stays_off_camera(db):
    """The user's rule, verbatim: the camera fires on "풀이 봐줘"-class phrases
    and on NOTHING else. An evaluator hunch about "여기 이거 어떡해요" gets an
    explanation of the pending question, not a photo stop."""
    session, llm, speaker, ws = build(
        db,
        "여기 이거 어떡해요",   # never names the written work
        llm_responses={
            "evaluate": [{"intent": "WORK_CHECK", "verdict": "UNCLEAR", "feedback": "",
                          "misconception": None, "status": None}],
        },
    )
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    assert ws.captures == 0
    assert "capture_request" not in ws.event_names()
    assert llm.calls.count("recognize") == 0
    assert llm.calls.count("explain") == 1    # answered with words instead
    assert speaker.spoken                     # the student still hears something
    # nothing was graded: the pending question is still waiting for its answer
    assert session.store.pending_hint("p1") is not None


# --- the hole in the middle, and the follow-through -------------------------
# Two defects from one demo run of 수능 13 (접선 l과 m). The student found l의
# 기울기, skipped l의 방정식, computed m의 기울기 — and the tutor confirmed
# "step 4 done" and pointed past the hole. Then, told "다음은 X 차례예요", the
# student asked "다음은 어떻게 해요?" and the tutor reached for the camera.


# Three INDEPENDENT sub-results, the shape of 수능 13: a (l의 기울기),
# b (l의 방정식), c (m의 기울기) — no later step is the same equation as an
# earlier one, so nothing vouches for a skipped middle.
REF3 = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="a 구하기", expression="a = 2"),
        SolutionStep(idx=2, description="b 구하기", expression="b = a + 1"),
        SolutionStep(idx=3, description="c 구하기", expression="c = 3*a"),
    ],
    final_answer=Answer(kind="SCALAR", value="6"),
    concepts=["linear_equation"],
    verified=True,
    origin="db",
)


def read_page(work: list[str]) -> Recognition:
    return Recognition(
        problem_text="2a = 4일 때 a, b = a+1, c = 3a를 구하시오",
        equations=["2*a = 4"],
        student_work=work,
        confidence=0.95,
    )


class TestTheHoleInTheMiddle:
    """last_correct_step is a PREFIX over what the page accounts for.

    A later line vouches for an earlier step only when it IS that step
    transformed (one equation, same solution set) — 'x = 5' vouches for
    '3*x = 15'. It never vouches across threads: c says nothing about b.
    """

    def test_a_skipped_step_declines_the_fast_path(self, db):
        """Steps 1 and 3 on the page, 2 missing: 'step 3 done' would walk the
        lesson right past the hole, so the symbolic path hands it to the full
        diagnosis instead of confirming."""
        est = StudentStateEstimator(EchoLLMClient({}), db)
        state = est._rule_based_progress(read_page(["a = 2", "c = 3*a"]), REF3, None)
        assert state is None

    def test_a_contiguous_page_is_still_confirmed_without_an_llm(self, db):
        est = StudentStateEstimator(EchoLLMClient({}), db)
        state = est._rule_based_progress(read_page(["a = 2", "b = a + 1"]), REF3, None)
        assert state is not None
        assert state.status == "CORRECT"
        assert state.last_correct_step == 2

    def test_a_transformed_line_still_vouches_for_its_own_step(self, db):
        """The other half of the bargain, pinned: skipping WITHIN one thread
        stays credited ('x = 5' alone is step 2 of the linear problem)."""
        est = StudentStateEstimator(EchoLLMClient({}), db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["x = 5"], confidence=0.9,
        )
        state = est._rule_based_progress(rec, REFERENCE, None)
        assert state is not None
        assert state.last_correct_step == 2

    def test_a_restated_problem_reports_the_prefix_not_the_peak(self, db):
        est = StudentStateEstimator(EchoLLMClient({}), db)
        state = est._rule_based_progress(
            read_page(["a = 2", "c = 3*a", "2*a = 4"]), REF3, None
        )
        assert state is not None
        assert state.status == "STUCK"
        assert state.last_correct_step == 1   # the hole at 2 caps it


class TestTheFollowThrough:
    """'다음은 어떻게 해요?' seconds after a confirmation must not begin with
    a camera shutter: the diagnosis it needs is the one just made."""

    async def confirmed(self, db):
        session, llm, speaker, ws = build(
            db, "풀이 맞아?",
            llm_responses={"recognize": [seen(["3*x = 15"]), seen(["3*x = 15"])]},
        )
        ask_l1(session)
        await session._handle_utterance(PCM, 16000)   # the confirmation turn
        assert ws.captures == 1
        assert session._continue_from is not None
        return session, llm, speaker, ws

    async def test_the_next_hint_request_skips_the_camera(self, db):
        session, llm, speaker, ws = await self.confirmed(db)
        spoken_before = len(speaker.spoken)

        await session.handle_hint_request()           # "다음은 어떻게 해요?"

        assert ws.captures == 1                       # no second photo
        assert llm.calls.count("recognize") == 1      # and no second read
        assert len(speaker.spoken) > spoken_before    # the next step was hinted
        hint = session.store.get_history(problem_hash="p1")[-1]
        assert hint.step == 2 and hint.level == 1     # L1 at the NEXT step

    async def test_spoken_next_step_request_uses_the_armed_follow_through(self, db):
        session, llm, _, ws = await self.confirmed(db)
        session.deps.transcriber.texts.append("응, 이 다음엔 어떻게 해야 돼?")

        await session._handle_utterance(PCM, 16000)

        assert ws.captures == 1                       # confirmation's photo only
        assert llm.calls.count("evaluate") == 0       # never graded as an answer
        hint = session.store.get_history(problem_hash="p1")[-1]
        assert hint.step == 2                         # never returned to step 1

    async def test_the_window_is_single_use(self, db):
        session, llm, speaker, ws = await self.confirmed(db)
        await session.handle_hint_request()
        assert ws.captures == 1

        await session.handle_hint_request()           # asked again, later

        assert ws.captures == 2                       # back to the real pipeline

    async def test_a_stale_window_takes_the_photo(self, db):
        import time as _time
        session, llm, speaker, ws = await self.confirmed(db)
        p_hash, _ = session._continue_from
        session._continue_from = (p_hash, _time.monotonic() - 120)

        await session.handle_hint_request()

        assert ws.captures == 2                       # expired: look again


async def test_problem_13_g_prime_confirmation_continues_to_g_prime_at_one(db):
    """The complete live regression from 2026-08-15 14:27.

    A cropped reread reported CORRECT step 1 after step 3 had already been
    proven; the follow-up question was then graded against an ancient step-1
    hint.  Neither kind of rewind is allowed.
    """
    p13_text = (
        "13. 함수 f(x) = x² - 4x - 3 에 대하여 곡선 y = f(x) 위의 점 "
        "(1, -6)에서의 접선을 l이라 하고, 함수 g(x) = (x³ - 2x) f(x)에 "
        "대하여 곡선 y = g(x) 위의 점 (1, 6)에서의 접선을 m이라 하자."
    )
    equations = [
        "f(x) = x**2 - 4*x - 3",
        "y = f(x)",
        "g(x) = (x**3 - 2*x)*f(x)",
        "y = g(x)",
    ]
    work = ["g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*(2*x - 4)"]
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*x - 4"),
            SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"),
            SolutionStep(idx=4, description="g'(1) 계산", expression="g'(1) = -4"),
            SolutionStep(idx=5, description="m의 방정식 쓰기", expression="m: y = -4*x + 10"),
            SolutionStep(idx=6, description="두 직선의 교점 구하기", expression="x = 7"),
            SolutionStep(idx=7, description="넓이 구하기", expression="49"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation", "derivative_applications", "area"],
        verified=True,
        origin="db",
    )
    recognized = dict(
        problem_text=p13_text,
        equations=equations,
        student_work=work,
        choices=[], diagram_conditions=[], uncertain_regions=[], confidence=1.0,
    )
    session, llm, speaker, ws = build(
        db,
        "풀이 맞아?",
        "응, 이 다음엔 어떻게 해야 돼?",
        llm_responses={
            "recognize": [recognized],
            # The bad live estimate: correct line, but a regressed step number.
            "estimate": [{"current_step": "g' 식을 올바르게 구함",
                          "last_correct_step": 1, "status": "CORRECT",
                          "misconception": None, "attempt_count": 1,
                          "previous_hint_effective": True}],
        },
    )
    initial = Recognition(**recognized)
    session.ctx = ProblemContext(
        hash="p13",
        recognition=initial.model_copy(update={"student_work": []}),
        match=MatchResult(tier=Tier.EXACT, concepts=reference.concepts,
                          reference=reference),
        reference=reference,
    )
    session.store.set_state(
        StudentState(status="CALCULATION_ERROR", last_correct_step=3)
    )
    session.prev_work = ["g'(1) = -2"]
    session.store.append_hint(
        problem_hash="p13", step=1, level=1,
        action="SOCRATIC_QUESTION", hint_text="오래된 질문",
    )
    current_hint = session.store.append_hint(
        problem_hash="p13", step=4, level=2,
        action="CONCEPT_HINT", hint_text="g'(1)은 어떻게 계산할까요?",
    )

    await session._handle_utterance(PCM, 16000)  # work confirmation

    state = session.store.get_state()
    assert state.status == "CORRECT" and state.last_correct_step == 3
    assert session.store.pending_hint("p13").id == current_hint
    confirmation = " ".join(speaker.spoken)
    assert "이제" in confirmation and "g 프라임 1" in confirmation
    assert "계산해 볼까요" in confirmation and "차례" not in confirmation
    assert "l의 기울기" not in confirmation
    assert session._continue_from is not None

    await session._handle_utterance(PCM, 16000)  # "이 다음엔 어떻게?"

    assert ws.captures == 1
    assert llm.calls.count("evaluate") == 0
    latest = session.store.get_history(problem_hash="p13")[-1]
    assert latest.step == 4


async def test_a_forward_question_is_on_the_record(db):
    """The confirm line asks a real question, so it leaves a pending one.

    Live: "맞아요! ... 이제 g의 1을 계산해 볼까요?" left no record, the reply
    "그럼 마이너스 4 맞나?" found no pending question, fell through to the
    intent LLM, came back NONE — and the tutor sat silent on the very answer
    it had invited."""
    session, llm, speaker, ws = build(
        db, "풀이 맞아?",
        llm_responses={"recognize": [seen(["a = 2", "b = a + 1"])]},
    )
    session.ctx.reference = REF3          # next step "c 구하기" is noun-form
    ask_l1(session)

    await session._handle_utterance(PCM, 16000)

    pending = session.store.pending_hint("p1")
    assert pending is not None
    assert pending.step == 3              # the step the line pointed at
    assert pending.hint_text.startswith("맞아요!")
    # and the reply to it is decided by rule, never by a guessing model
    from tutor.speech.intent import rule_intent
    assert rule_intent("그럼 마이너스 4 맞나?", has_problem=True,
                       has_pending=True) == "ANSWER"
