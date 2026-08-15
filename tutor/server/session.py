"""Per-connection session orchestrator.

Owns ALL SessionStore writes. Every utterance is first classified
(tutor/speech/intent.py) into the turn it asks for, because the same words
mean different things depending on whether a tutor question is pending:

HINT REQUEST / WORK CHECK (the worksheet is the evidence — the camera looks)
    capture → recognize → match (cached by problem_hash) → solver if needed →
    estimate → set_state + resolve pending hint effectiveness → prefetch
    state/history → policy.decide → generate hint (leak-guarded) → speak.

    One method serves both. "이 문제 힌트 줄래?" and "풀이 맞아?" want the
    same pipeline over a fresh photo; the only difference is that the
    second one asked something, so their question reaches the hint phrasing and
    the tutor confirms what it saw before hinting (`question=`).

ANSWER (the student is replying to the question the tutor just asked)
    evaluate transcript against the reference solution → resolve the pending
    hint → prefetch state/history → policy.decide → speak.
    No capture, no VLM, no matching: the student spoke, they did not write, so
    re-reading the worksheet only adds latency and re-recognition noise.

NONE (thinking out loud) — nothing happens. Spec rule 7.

The verdict feeds the unchanged policy rules through the store, which is what
produces the intended ladder: correct → next step L1, wrong → same step L2,
unclear → same step, same level, re-asked.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from websockets.exceptions import ConnectionClosed
from dataclasses import dataclass, field

from tutor.config import Settings
from tutor.hints.generator import (
    SENTENCE_STEP_RE,
    HintGenerator,
    guided_step_question,
    strip_leading_acknowledgement,
    visible_to_student,
)
from tutor.hints import illustrator, plot
from tutor.hints.guard import leaks_answer
from tutor.knowledge import mathnorm
from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import MatchResult, ReferenceSolution, Tier
from tutor.llm import timing
from tutor.policy.engine import Action, Decision, Trigger, decide
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioFrame, ImageFrame, ProtocolError, decode
from tutor.solver.grok_solver import GrokSolver
from tutor.speech import mathspeak
from tutor.speech.intent import IntentClassifier, refers_to_work
from tutor.speech.stt import classify_transcript
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator, hint_was_effective
from tutor.state.models import StudentState
from tutor.store.session_store import HintRecord, SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

log = logging.getLogger(__name__)

# What the tutor says instead of running the pipeline on an unusable utterance.
# Neither counts as a hint, so neither enters the hint history or the policy.
RETRY_PROMPTS = {
    "unclear": "잘 못 들었어요. 다시 한번 말해 줄래요?",
    "filler_only": "괜찮아요, 이어서 말해 줄래요?",
}

# Said after the evaluator's own "맞아요!" when the last step lands.
PROBLEM_DONE = "문제를 끝까지 풀었네요! 또 모르는 문제가 있으면 알려주세요."

# The student asked about their own work ("풀이 맞아?"), and the answer to that
# question is yes or no — not a hint. When the work IS right, say so and stop:
# a hint would push them at a step they have not reached, and the student asked
# to be checked, not helped.
WORK_CONFIRMED = "맞아요! 이대로 하면 돼요. 또 궁금한 게 있으면 물어봐 주세요."

# "풀이 맞아?" is a yes/no question, so the VERDICT comes first — before any
# hint, before any question back. Fixed text, never generated: it may say THAT
# the work is wrong, never WHAT is wrong. Naming the mistake is the hint's
# job, and the leak guard's. Statuses that carry no verdict (STUCK, a skipped
# estimate) claim none: saying 맞아요/틀려요 without evidence is worse than
# a neutral "let's look".
WORK_CHECK_WRONG = "앗, 함께 다시 생각해 볼까요?"
WORK_CHECK_REACTIONS: dict[str, str] = {
    "UNCERTAIN": "",  # ASK_RECAPTURE already says it cannot see the page
    "CALCULATION_ERROR": WORK_CHECK_WRONG,
    "CONCEPT_ERROR": WORK_CHECK_WRONG,
    "PROCEDURAL_ERROR": WORK_CHECK_WRONG,
    "MISREAD": WORK_CHECK_WRONG,
}


def _step_claim_variants(expression: str) -> list[str]:
    """Comparable claims contained in a stored step expression.

    Verified steps often keep the whole derivation on one line, for example
    ``l: y = -2*(x-1)-6 = -2*x-4``.  A student naturally says only the final
    equation.  Treat every equality pair in that chain as evidence for the
    same step, while keeping the normal strict symbolic checker underneath.
    """
    variants: list[str] = []
    for chunk in (expression or "").split(","):
        chunk = re.sub(r"^\s*[A-Za-z]\s*:\s*", "", chunk.strip())
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split("=") if part.strip()]
        if len(parts) < 2:
            variants.append(chunk)
            continue
        for left in range(len(parts) - 1):
            for right in range(left + 1, len(parts)):
                variants.append(f"{parts[left]} = {parts[right]}")
    return variants


def _claim_matches_step(claim: str, expression: str) -> bool:
    if mathnorm.equations_equivalent(claim, expression):
        return True
    return any(
        mathnorm.equations_equivalent(claim, variant)
        for variant in _step_claim_variants(expression)
    )


def _spoken_linear_claim(transcript: str, expression: str) -> str | None:
    """Conservatively recover a spoken linear expression such as ``-2x-4``."""
    text = (transcript or "").lower().replace("−", "-")
    for source, target in (
        ("마이너스", "-"), ("플러스", "+"), ("더하기", "+"), ("빼기", "-"),
        ("엑스", "x"),
    ):
        text = text.replace(source, target)
    compact = re.sub(r"\s+", "", text)
    found = re.search(r"[+-]?(?:\d+(?:\.\d+)?)?x(?:[+-]\d+(?:\.\d+)?)?", compact)
    if found is None:
        return None
    rhs = found.group(0)
    rhs = re.sub(r"([+-]?\d+(?:\.\d+)?)x", r"\1*x", rhs)
    if re.search(r"(?:^|[:\s])y\s*=", expression):
        return f"y = {rhs}"
    return rhs


WORK_CHECK_DEFAULT = "음, 지금 쓴 줄을 같이 볼까요?"

# The filler openers for a work check: fixed, so their TTS is cached and one
# plays the instant the delay elapses — while the camera is still being asked.
# A tuple because the same student asks "풀이 맞아?" many times in one lesson,
# and the identical opener every time is what makes it sound like a machine.
WORK_CHECK_OPENERS: tuple[str, ...] = (
    "네, 지금 쓴 풀이를 확인하고 있어요.",
    "좋아요, 쓴 풀이를 한번 볼게요.",
    "네, 풀이를 같이 확인해 볼게요.",
)

# The readout frame: the opener is fixed (pre-rendered TTS), so the only
# synthesis the narration waits for is the one line that is different every
# time — the problem itself.
#
# The closers are NO LONGER SPOKEN by default. The closer was the longest
# line of the turn (~6s alone), and it queued right where the hint tends to
# come ready — a narration line that has started must finish, so it routinely
# delayed the hint by seconds while adding the least information. Kept
# registered and pre-rendered so putting it back is a one-line change.
READOUT_OPENER = "좋아요, 문제 확인해 볼게요."
_READOUT_QUESTION = (
    "라는 문제군요. 범위 제한이나 최고차항의 계수, "
    "숨겨진 비율이나 대칭성이 있는지도 확인해 보셨나요?"
)
READOUT_CLOSERS: dict[bool, str] = {
    False: _READOUT_QUESTION,
    True: "이" + _READOUT_QUESTION,
}

# Spoken sentence endings that are the STUDENT'S politeness, not their answer.
# "5예요" quotes back as the stilted "5예요라고 했네요"; a teacher hears the 5,
# drops the ending, and says "5… 어디 보자". Longest first, stripped once.
_POLITE_TAILS = ("입니다", "이에요", "이예요", "예요", "에요", "이요", "요")


def answer_core(transcript: str) -> str | None:
    """The VALUE inside a spoken answer, or None when there is no clean one.

    "5예요" → "5" · "마이너스 3이요" → "마이너스 3" · "x는 2요" → "x는 2".
    None for anything that does not END in a value once the politeness is
    stripped — verbs ("양변에서 5를 빼요" → "…빼"), questions, long speeches.
    Those are exactly the utterances that sound wrong repeated back, so no
    echo is the natural choice: the generic filler takes the turn instead.

    A VALUE, strictly: one short phrase with nothing sentence-like inside it.
    The 24-character cap alone let "우선 f를 미분해야겠지. 그러면 2x-4" through
    (it ends in a digit), and the tutor parroted a student's whole train of
    thought back at them with "…인지 보고 있어요" stitched on. Working aloud
    is for the STAGE LINE to acknowledge, not the voice to repeat.
    """
    text = " ".join(transcript.split()).rstrip("?.!…, ")
    if not text or len(text) > 16:
        return None
    # sentence-internal punctuation or more than three words is speech,
    # not a value — "마이너스 3" is two words, an argument is not
    if re.search(r"[.!?…,]", text) or len(text.split()) > 3:
        return None
    for tail in _POLITE_TAILS:
        if text.endswith(tail) and len(text) > len(tail):
            text = text[: -len(tail)].rstrip()
            break
    if not text:
        return None
    ch = text[-1]
    # value-final only: a digit ("…3"), or a lone latin variable ("…x")
    if ch.isdigit() or (ch.isascii() and ch.isalpha()
                        and (len(text) == 1 or not text[-2].isascii() or not text[-2].isalpha())):
        return text
    return None


# Does this string claim anything, or is it a lone term?
_RELATES = re.compile(r"[=<>≤≥]")


def preflight_fallback(concept_name: str) -> str:
    """Said the first time a concept is met, while its own line is written.

    Generic on purpose: it names the kind of problem (which the tagger already
    knows) and asks the one question that fits every subject. From the next
    problem of this kind on, the DB has something better to say.
    """
    return f"{concept_name} 문제군요. 어떤 조건이 주어졌는지 먼저 정리해 볼까요?"


def _solve_dead(task: asyncio.Task | None) -> bool:
    """True when there is no live solve to wait for — never ran, or ran and lost."""
    return task is None or (
        task.done() and (task.cancelled() or task.exception() is not None)
    )


@dataclass
class ProblemContext:
    hash: str
    recognition: Recognition
    match: MatchResult
    # None while the background solver is still writing it (NEW problems only).
    reference: ReferenceSolution | None
    solving: asyncio.Task | None = None
    # ONE grid per problem: what is drawn on it right now, and the window it
    # is drawn in. The illustrator redeclares both every turn, which is how a
    # scaffold curve disappears once the thing it produced is on the board.
    scene: tuple = ()
    span: tuple[float, float] | None = None
    # The student's own window: (x0, x1, y0, y1), or None while the automatic
    # guess (the middle 90% of the target's values) is still in charge. The
    # first pan or zoom materializes the guess into this rectangle and every
    # adjustment after that moves THE RECTANGLE — not the guess, which would
    # recompute under their hands — and the illustrator's own redraws keep it,
    # so the next turn does not snap the window back. view_rev counts the
    # student's adjustments; the page uses it to tell a new render of the same
    # scene from a duplicate.
    view: tuple[float, float, float, float] | None = None
    view_rev: int = 0
    caption: str = ""

    def reference_if_ready(self) -> ReferenceSolution | None:
        """The reference if it exists RIGHT NOW; never waits.

        The first turn on a new problem uses this: an L1 hint can be asked from
        the concepts alone, and waiting out the solver is exactly the silence
        the background task exists to remove.
        """
        task = self.solving
        if (self.reference is None and task is not None and task.done()
                and not task.cancelled() and task.exception() is None):
            self.reference = task.result()
            self.solving = None
        return self.reference

    async def reference_ready(self) -> ReferenceSolution:
        """The reference, waiting out the background solver if it is running.

        Answer turns need it — grading a spoken answer against nothing is not
        grading — and by the time a student has heard the first hint and formed
        a reply, the solver has almost always finished anyway.
        """
        if self.reference is None and self.solving is not None:
            task, self.solving = self.solving, None
            self.reference = await task  # a failed solve raises here, on purpose
        if self.reference is None:
            raise RuntimeError("no reference solution: the background solve failed")
        return self.reference


@dataclass
class Deps:
    settings: Settings
    recognizer: Recognizer
    matcher: Matcher
    solver: GrokSolver
    estimator: StudentStateEstimator
    hint_gen: HintGenerator
    transcriber: object  # .transcribe(pcm, sample_rate) -> Transcript
    speaker: object  # .speak(text) -> None
    evaluator: AnswerEvaluator | None = None  # None → answers fall back to a hint request
    cameras: object | None = None  # CameraHub: eyes on another socket (the phone)
    fillers: object | None = None  # FillerBank: what to say while thinking
    # Illustrator: draws beside a hint while it is being spoken. None → the
    # tutor teaches in words only, which is a supported configuration.
    illustrator: object | None = None
    # what each utterance is FOR; without an LLM it still routes by rule
    classifier: IntentClassifier = field(default_factory=IntentClassifier)
    store: SessionStore = field(default_factory=SessionStore)


class Session:
    def __init__(self, ws, deps: Deps):
        self.ws = ws
        self.deps = deps
        self.store = deps.store
        self.ctx: ProblemContext | None = None
        self.prev_work: list[str] | None = None
        self.last_transcript: str | None = None
        self._captures: dict[str, asyncio.Future] = {}
        self._capture_seq = 0
        self._audio_buffers: dict[str, list[bytes]] = {}
        self._busy = False
        # An answer can finish STT while the hint turn is still doing its last
        # bookkeeping.  Dropping it at that boundary makes the tutor look as
        # though it is listening while being deaf.  Keep an explicit idle
        # signal so answers wait for that tiny overlap instead.
        self._turn_idle = asyncio.Event()
        self._turn_idle.set()
        self._turn_waiters = 0
        # Which pending hint THIS turn answered, so a turn the student never
        # hears can put it back. Per turn: cleared when one starts.
        self._resolved_hint: int | None = None
        self._tasks: set[asyncio.Task] = set()
        self._filler: asyncio.Task | None = None
        self._filler_spoke = False
        self._filler_open = False
        self._filler_lines: asyncio.Queue = asyncio.Queue()
        # Armed by a confirmed work check: (problem_hash, monotonic time). The
        # very next hint request may continue from that diagnosis instead of
        # asking the camera again — see _continue_confirmed.
        self._continue_from: tuple[str, float] | None = None

    def _spawn(self, coro) -> None:
        """Run the hint flow concurrently with the receive loop — it awaits
        capture frames that arrive through that same loop."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    await self.on_frame(raw)
                except ProtocolError as e:
                    log.warning("protocol error: %s", e)
                    try:
                        await self.ws.send(make_event("error", {"message": str(e)}))
                    except Exception:
                        break
                except Exception:
                    # a bad frame or transient backend failure must not kill
                    # the session (student state + hint history live here)
                    log.exception("frame handling failed; session continues")
        except (ConnectionClosed, OSError) as e:
            # A closed laptop lid or dropped Wi-Fi is a TCP reset, not a
            # server bug: one quiet line, not an ERROR with a traceback.
            log.info("device connection dropped (%s)", type(e).__name__)
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    async def on_frame(self, raw: bytes | str) -> None:
        if isinstance(raw, str):
            await self._on_event(raw)
            return
        frame = decode(raw)
        if isinstance(frame, ImageFrame):
            future = self._captures.get(frame.header.capture_id)
            if future is not None and not future.done():
                future.set_result(frame.jpeg)
            else:
                log.warning("unexpected IMAGE for capture_id %s", frame.header.capture_id)
        elif isinstance(frame, AudioFrame):
            await self._on_audio(frame)

    async def _on_event(self, raw: str) -> None:
        ev = parse_event(raw)
        if ev.event == "hello":
            await self.ws.send(make_event("hello_ack", {"proto": 1}))
        elif ev.event == "hint_request":
            self._spawn(self.handle_hint_request())
        elif ev.event == "capture_failed":
            capture_id = ev.data.get("capture_id", "")
            future = self._captures.get(capture_id)
            if future is not None and not future.done():
                future.set_result(None)
        elif ev.event in ("figure_zoom", "figure_pan", "figure_reset"):
            # The view controls on the board's sketch. Independent of the
            # turn: re-rendering a scene that already passed the leak gate
            # reads nothing new and says nothing, so they may run while _busy.
            self._spawn(self._adjust_view(ev.event, ev.data))
        elif ev.event == "error":
            log.warning("device error: %s", ev.data)
        else:
            log.warning("unknown event %r", ev.event)

    MAX_UTTERANCE_BYTES = 2 * 1024 * 1024  # ~64s of 16kHz/16-bit mono

    async def _on_audio(self, frame: AudioFrame) -> None:
        stream_id = frame.header.stream_id
        if stream_id not in self._audio_buffers:
            # the device speaks one utterance at a time: a new stream means
            # any unfinished previous stream was abandoned — drop it
            self._audio_buffers = {stream_id: []}
        buf = self._audio_buffers[stream_id]
        buf.append(frame.pcm)
        if sum(len(c) for c in buf) > self.MAX_UTTERANCE_BYTES:
            log.warning("utterance %s exceeded max size; dropping", stream_id)
            del self._audio_buffers[stream_id]
            return
        if not frame.header.last:
            return
        pcm = b"".join(self._audio_buffers.pop(stream_id))
        if pcm:
            # transcribe off the receive loop: a slow STT call must not stall
            # frame delivery (pending capture futures resolve through this loop)
            self._spawn(self._handle_utterance(pcm, frame.header.sample_rate))

    async def _handle_utterance(self, pcm: bytes, sample_rate: int) -> None:
        heard = ""
        try:
            transcript = await asyncio.to_thread(
                self.deps.transcriber.transcribe, pcm, sample_rate
            )
            text, heard = transcript.text, transcript.language
        except Exception:
            log.exception("STT failed; asking the student to repeat")
            text = ""  # graded as "unclear" below, so the student hears back

        # Quality gate: noise transcribed as glyphs, silence, or a hesitation
        # sound is not an answer. Ask again instead of grading it — and leave
        # any pending hint pending, so the real answer still lands on it.
        quality = classify_transcript(text, heard)
        if quality != "ok":
            log.info("utterance rejected (%s): %r", quality, text)
            await self._send_transcript(text, True)
            try:
                await self._speak(RETRY_PROMPTS[quality], final=True)
            except Exception:
                log.exception("could not ask the student to repeat")
            return

        log.info("utterance: %r", text)
        pending = (
            self.store.pending_hint(self.ctx.hash) if self.ctx is not None else None
        )
        # What is this utterance FOR? The same sentence means different turns
        # depending on what is already on the table, so the two flags decide as
        # much as the words do. Off the loop: it may cost one small LLM call.
        intent = await asyncio.to_thread(
            self.deps.classifier.classify,
            text,
            has_problem=self.ctx is not None,
            has_pending=pending is not None,
        )
        if intent == "ANSWER" and (pending is None or self.deps.evaluator is None):
            # Nothing to grade the answer against: treat it as asking for help
            # rather than dropping the student's turn on the floor.
            intent = "HINT_REQUEST"

        if intent == "NONE":
            # Not addressed to the tutor. It is not evidence either, so it must
            # not linger and be read into the next diagnosis.
            await self._send_transcript(text, False)
            return

        self.last_transcript = text
        await self._send_transcript(text, True)

        if intent == "ANSWER":
            await self.handle_answer(text, pending)
        elif intent == "WORK_CHECK":
            # Same turn as a hint request, but they asked something about what
            # they wrote — so the camera looks again and the answer is to them.
            await self.handle_hint_request(question=text)
        else:
            await self.handle_hint_request()

    async def _send_transcript(self, text: str, responding: bool) -> None:
        try:
            await self.ws.send(
                # `wants_hint` is the device's "a reply is coming, stay muted"
                # flag; it means responding, not literally the word "힌트"
                make_event("transcript", {"text": text, "wants_hint": responding})
            )
        except Exception:
            log.debug("could not send transcript event (connection gone)")

    def _preflight_line(self, rec: Recognition) -> str:
        """"등비수열 문제군요. 공비와 항 번호의 관계를 확인하셨나요?"

        Looked up by concept, never generated on the clock. The first problem
        of a kind says the generic opening and starts a background write; from
        the second on, the line is in the DB and its audio is in the TTS
        cache, so the whole thing costs nothing at speak time.
        """
        db = self.deps.hint_gen.db
        for concept in rec.concepts:
            name = db.concept_name(concept)
            if not name:
                continue  # a tag the KB does not carry a Korean name for
            line = db.preflight_line(concept)
            if line:
                self._remember_phrase(line)
                return line
            self._spawn(self._write_preflight(concept, name))
            return preflight_fallback(name)
        return ""

    async def _write_preflight(self, concept: str, name: str) -> None:
        """Write this concept's line for NEXT time. Never awaited by a turn."""
        try:
            line = await asyncio.to_thread(self.deps.hint_gen.write_preflight, name)
        except Exception:
            log.exception("could not write the preflight line for %s", concept)
            return
        if not line:
            return
        self.deps.hint_gen.db.save_preflight_line(concept, line)
        self._remember_phrase(line)
        log.info("preflight line written for %s: %s", concept, line)

    def _remember_phrase(self, text: str) -> None:
        """Register a fixed line with the TTS cache, so it is rendered once and
        replayed forever after (the same treatment the fillers get)."""
        register = getattr(self.deps.speaker, "register", None)
        if callable(register):
            register(text)

    async def _stage(self, text: str) -> None:
        """What the pipeline is doing right now, for the SCREEN, not the voice.

        The stages used to be narrated ("풀이를 다 읽었어요…"), and a turn could
        say three filler-ish lines before its first real word. Now the screen
        carries the progress and the voice keeps at most ONE filler per turn —
        the delay-gated opener — plus the problem readout, which is content.
        """
        try:
            await self.ws.send(make_event("stage", {"text": text}))
        except Exception:
            log.debug("could not send stage event (connection gone)")

    async def _send_problem(self, rec: Recognition) -> None:
        """What the tutor read off the page, for the browser's problem card.

        Notation for the eye (displayable), never the photo — and never the
        student's work or anything downstream of the solver, so nothing that
        could leak an answer travels on this event.

        The equations line exists for what the STATEMENT does not spell out (a
        formula from a figure, a condition the VLM had to lift out). An
        equation the sentence already contains is printed twice for nothing.
        """
        text = mathspeak.displayable(rec.problem_text)
        seen = mathnorm.compact(text)
        equations = []
        for raw in rec.equations:
            eq = mathspeak.displayable(raw)
            compact = mathnorm.compact(eq)
            # a bare term ("a_10") states nothing; a repeat of the sentence
            # states it twice. The VLM emits both often enough to matter.
            if not compact or not _RELATES.search(compact) or compact in seen:
                continue
            equations.append(eq)
        try:
            await self.ws.send(
                make_event("problem", {"text": text, "equations": equations})
            )
        except Exception:
            log.debug("could not send problem event (connection gone)")

    # --- capture --------------------------------------------------------------

    async def _request_capture(self) -> bytes | None:
        """The student's worksheet, from whichever eye can see it NOW.

        In camera mode a connected phone is asked FIRST: it photographs the
        page as it is at this moment. The browser's attached photo is frozen
        at paste time — and serving it while a phone was watching meant a
        student who changed their work was diagnosed on the picture from
        minutes ago, forever. The attached photo stays as the fallback when
        no phone answers; in upload mode it is the only eye by design.
        """
        timeout = self.deps.settings.capture_timeout_s
        cameras = self.deps.cameras
        phone_first = self.deps.settings.input_mode == "camera" and bool(cameras)
        started = time.monotonic()
        if phone_first:
            jpeg = await cameras.capture(timeout)
            eye = "camera device"
            if not jpeg:
                log.info("no frame from the camera device; trying the session device")
                started = time.monotonic()
                jpeg = await self._capture_from_device()
                eye = "session device"
        else:
            jpeg = await self._capture_from_device()
            eye = "session device"
            if not jpeg:
                if self.deps.settings.input_mode != "camera":
                    # INPUT_MODE=upload: the picture comes from the browser's
                    # file picker and nowhere else. Falling through to a camera
                    # device here would silently hand the tutor a different
                    # photo from the one the student just chose.
                    log.warning("no worksheet photo attached (INPUT_MODE=upload)")
                    return None
                log.warning("no eye at all: the session device has no camera and none "
                            "is connected on /camera")
                return None
        elapsed = (time.monotonic() - started) * 1000
        timing.record("capture", elapsed / 1000, eye)

        if jpeg:
            log.info("captured %d bytes from the %s in %.0f ms", len(jpeg), eye, elapsed)
            self._save_capture(jpeg)
        else:
            # The single most useful line in the log when the tutor keeps asking
            # to be shown the worksheet again: it says the VLM was never called.
            log.warning(
                "NO FRAME from the %s after %.0f ms (timeout %.0fs) — the worksheet was "
                "never seen, so recognition did not run. Raise CAPTURE_TIMEOUT_S, or check "
                "that the camera page is still open and connected.",
                eye, elapsed, timeout,
            )
        return jpeg

    def _save_capture(self, jpeg: bytes) -> None:
        """SAVE_CAPTURES_DIR=... to see what the camera actually sent."""
        directory = self.deps.settings.save_captures_dir
        if directory is None:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"capture-{int(time.time() * 1000)}.jpg"
            path.write_bytes(jpeg)
            log.info("frame saved: %s", path)
        except OSError as e:  # a full disk must not end the lesson
            log.warning("could not save the frame: %s", e)

    async def _capture_from_device(self) -> bytes | None:
        self._capture_seq += 1
        capture_id = f"cap-{self._capture_seq}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._captures[capture_id] = future
        await self.ws.send(
            make_event("capture_request", {"capture_id": capture_id, "quality": "high"})
        )
        try:
            return await asyncio.wait_for(future, timeout=self.deps.settings.capture_timeout_s)
        except asyncio.TimeoutError:
            return None
        finally:
            self._captures.pop(capture_id, None)

    # --- the main flow --------------------------------------------------------

    async def handle_hint_request(self, question: str | None = None) -> None:
        if self._busy:
            log.info("hint request ignored: already handling one")
            return
        self._busy = True
        self._turn_idle.clear()
        self._resolved_hint = None
        # a work check earns its own opener: they asked about THEIR page, and
        # "네, 지금 쓴 풀이를 확인하고 있어요" answers that before the camera has
        # moved. Rotated, because the fifth identical opener sounds like a machine.
        bank = self.deps.fillers
        # no bank → _start_filler is a no-op anyway, so the opener is moot
        opener = None
        if bank:
            # ONE filler per turn, and it announces the turn's actual job. The
            # hint turn used to spend two ("음, 살펴보고 있어요" then "좋아요,
            # 문제 확인해 볼게요" when the readout began) — the readout opener
            # is now THE opener, spoken at delay-time while the camera works,
            # and the readout itself is just the problem line.
            opener = (bank.rotate(WORK_CHECK_OPENERS, "opener") if question
                      else READOUT_OPENER)
        self._start_filler(opener)
        try:
            with timing.turn("WORK_CHECK" if question else "HINT_REQUEST"):
                await self._handle_hint_request(question)
        except Exception:
            log.exception("hint request failed")
            try:
                await self._deliver(
                    Decision(Action.PROBE, 0, 0, None, "internal error"),
                    "잠깐 문제가 생겼어요. 다시 한번 힌트를 요청해 줄래요?",
                )
            except Exception:
                log.exception("recovery delivery failed (connection likely gone)")
        finally:
            # A turn that says nothing (WAIT) still owes the filler an ending.
            try:
                await self._settle_filler()
            finally:
                self._busy = False
                self._turn_idle.set()
                await self._turn_finished()

    async def handle_answer(self, transcript: str, pending: HintRecord) -> None:
        """The student answered the tutor's question out loud."""
        if self._busy:
            # This is normally the end of the hint that asked `pending`: audio
            # has stopped, while history/illustration bookkeeping is still
            # unwinding.  The student already did their part; preserve it.
            log.info("answer queued: waiting for the current turn to finish")
            self._turn_waiters += 1
            try:
                while self._busy:
                    await self._turn_idle.wait()
            finally:
                self._turn_waiters -= 1
        self._busy = True
        self._turn_idle.clear()
        self._resolved_hint = None
        # the echo IS the acknowledgement of having heard them — and it plays
        # while the evaluator is still grading the very value it echoes
        core = answer_core(transcript)
        bank = self.deps.fillers
        self._start_filler(bank.echo(core) if core and bank is not None else None)
        try:
            with timing.turn("ANSWER"):
                await self._handle_answer(transcript, pending)
        except Exception:
            log.exception("answer handling failed")
            try:
                await self._deliver(
                    Decision(Action.PROBE, 0, pending.step, None, "internal error"),
                    "잠깐 문제가 생겼어요. 다시 한번 말해 줄래요?",
                    pending.problem_hash,
                )
            except Exception:
                log.exception("recovery delivery failed (connection likely gone)")
        finally:
            try:
                await self._settle_filler()
            finally:
                self._busy = False
                self._turn_idle.set()
                await self._turn_finished()

    async def _turn_finished(self) -> None:
        """Device-specific floor handoff after all turn bookkeeping is done."""
        return None

    async def _handle_answer(self, transcript: str, pending: HintRecord) -> None:
        ctx = self.ctx
        assert ctx is not None  # a pending hint implies a problem context
        # a graded answer moves the diagnosis: a follow-through armed by an
        # earlier confirmation would now continue from a stale picture
        self._continue_from = None

        # The reference is the measuring stick for a spoken answer, so this
        # turn cannot start without it. The wait is almost always zero: the
        # solver ran while the first hint was being spoken and heard.
        try:
            reference = await ctx.reference_ready()
        except Exception:
            # The background solve died. This used to fall back to the capture
            # flow, which is the wrong shape twice over: the student SPOKE, so
            # there is nothing new on the paper to look at, and if the phone has
            # dropped (it had, three seconds earlier) the answer turn ends by
            # asking for a photo nobody can take. The photo we already have is
            # enough to solve from, so retry from that — and if the retry loses
            # too, explain the pending question, which needs no reference.
            log.exception("no reference for the answer turn; retrying the solve")
            await self._stage("기준 풀이를 다시 계산하고 있어요")
            ctx.solving = self._start_solve(ctx.recognition, ctx.hash)
            try:
                reference = await ctx.reference_ready()
            except Exception:
                log.exception("the retried solve failed too; explaining instead")
                # never graded, so it is still evidence the next turn can read
                self.last_transcript = transcript
                await self._answer_question(ctx, pending, transcript)
                return

        # What the screen says while the judge reads. A short value is "an
        # answer being checked"; a train of thought spoken aloud is "working
        # being followed" — the same distinction that decides whether the
        # voice echoes it. The student who narrates their reasoning sees the
        # tutor following along instead of hearing themselves quoted.
        await self._stage(
            "답을 확인하고 있어요" if answer_core(transcript)
            else "말한 중간 과정이 맞는지 확인하고 있어요"
        )
        verdict = await asyncio.to_thread(
            self.deps.evaluator.evaluate,
            problem_text=ctx.recognition.problem_text,
            reference=reference,
            question=pending.hint_text,
            target_step=pending.step,
            transcript=transcript,
        )
        self.last_transcript = None  # graded; not evidence for the next turn

        if verdict.intent == "WORK_CHECK":
            if refers_to_work(transcript):
                # The answer-shaped fast path read it as a value ("풀이 5 맞아?"),
                # but they named their written work: go and look. Not the public
                # entry point — this turn already holds the busy flag.
                log.info("answer turn redirected to a work check: %r", transcript[:40])
                # ungraded, so it is still evidence — the estimator reads it too
                self.last_transcript = transcript
                await self._handle_hint_request(transcript)
                return
            # The evaluator thinks they meant their page, but they never named
            # it — and the camera fires on "풀이 봐줘" and nothing else. Explain
            # the pending question instead; if they do want their work checked,
            # the words that ask for it are one sentence away.
            log.info("evaluator suggested a work check without the words; "
                     "explaining instead: %r", transcript[:40])
            await self._answer_question(ctx, pending, transcript)
            return

        if verdict.intent == "QUESTION":
            # They asked, they did not attempt. Explain and leave the hint
            # pending: nothing was proven, so nothing escalates, nothing
            # advances, and their real answer still lands on this question.
            await self._answer_question(ctx, pending, transcript)
            return

        # Orchestrator-owned writes. The policy needs no new rules: resolving
        # the pending hint IS the signal that moves it up, down or nowhere.
        prev = self.store.get_state() or StudentState()
        if verdict.verdict == "CORRECT":
            reached = self._reached_step(verdict, reference, pending.step, transcript)
            self.store.set_state(
                prev.model_copy(
                    update={
                        "current_step": f"{reached}단계를 말로 설명함",
                        "last_correct_step": max(prev.last_correct_step, reached),
                        "status": "CORRECT",
                        "misconception": None,
                        "attempt_count": 1,
                        "previous_hint_effective": True,
                    }
                )
            )
            self._resolve_hint(pending.id, True)
            # Completion is never reached by running ahead: closing a problem
            # wipes the student's context, and one spoken sentence is not
            # enough evidence to do that. Only answering the LAST step's own
            # question ends it — _reached_step never reports that far.
            if pending.step >= len(reference.steps):
                await self._finish_problem(ctx, verdict, pending.step)
                return
        elif verdict.verdict == "PARTIAL":
            # A correct idea is not the same thing as a completed reference
            # step.  Live on problem 13, "f를 미분하면 2x-4니까 1을 대입하면
            # 될 것 같아요" was praised as CORRECT and advanced past the
            # still-missing slope/tangent work to the product rule. Keep the
            # frontier, record that the hint helped, and clear an old
            # misconception because the direction itself was sound.
            self.store.set_state(
                prev.model_copy(
                    update={
                        "current_step": f"{pending.step}단계를 말로 진행 중",
                        "status": "STUCK",
                        "misconception": None,
                        "attempt_count": prev.attempt_count + 1,
                        "previous_hint_effective": None,
                    }
                )
            )
            log.info(
                "answer partially correct: staying at step %d until the step is complete",
                pending.step,
            )
            # The hint did help even though the coarse reference step is not
            # finished. Resolve it positively so the policy fades back to L1
            # for the remaining sub-result instead of repeating L2/L3 support.
            self._resolve_hint(pending.id, True)
        elif verdict.verdict == "INCORRECT":
            self.store.set_state(
                prev.model_copy(
                    update={
                        "status": verdict.status or "CONCEPT_ERROR",
                        "misconception": verdict.misconception or prev.misconception,
                        "attempt_count": prev.attempt_count + 1,
                        "previous_hint_effective": False,
                    }
                )
            )
            self._resolve_hint(pending.id, False)
        else:
            # UNCLEAR: no evidence either way. Leaving the hint unresolved is
            # what makes the policy re-ask at the same level (R9).
            log.info("answer unclear: re-asking L%d at step %d", pending.level, pending.step)

        state = self.store.get_state() or prev
        history = self.store.get_history(problem_hash=ctx.hash)
        decision = decide(state, history, "HINT_REQUEST")
        log.info("decision after answer (%s): %s", verdict.verdict, decision)

        await self._stage("다음 질문을 만들고 있어요")
        # The reaction is ready NOW and the hint needs ~5s of model. Speaking
        # "맞아요, 그렇게 하면 돼요!" WHILE the next question is being written
        # is where the answer turn stops feeling slow: first meaningful sound
        # at evaluate-time instead of evaluate+phrase+TTS-time.
        sink = self._new_hint_stream(paused=True)
        kwargs = {"partial": verdict.verdict == "PARTIAL"}
        if sink is not None:
            kwargs["on_delta"] = sink.push
        hint_task = asyncio.create_task(asyncio.to_thread(
            self.deps.hint_gen.generate,
            decision, ctx.match, ctx.reference, ctx.recognition, history, transcript,
            **kwargs,
        ))
        try:
            spoke_feedback = await self._react(verdict.feedback, decision)
            if sink is not None:
                sink.activate()
            text = await hint_task
            streamed = await sink.finish(text) if sink is not None else False
        except Exception:
            hint_task.cancel()
            if sink is not None:
                await sink.abort()
            raise
        # read before any strip loses the subclass that carries them
        board = getattr(text, "board", ())  # before any strip loses the subclass
        if spoke_feedback and not streamed:
            # one acknowledgement per turn: the reaction already was it
            text = strip_leading_acknowledgement(text)
        # A correct answer often IS the reason the picture changes — the
        # tangent they just found is what replaces the curve it came from.
        said = transcript if verdict.verdict in {"CORRECT", "PARTIAL"} else None
        await self._deliver(
            decision, text, ctx.hash, board, said, already_spoken=streamed
        )

    @staticmethod
    def _reached_step(
        verdict, reference: ReferenceSolution, asked: int, transcript: str = ""
    ) -> int:
        """How far the student actually got, in steps we can prove.

        A student who answers the slope question with the whole tangent has
        done that step and the next one, and treating that as "wrong step"
        was punishing them for being ahead. But the number that says so also
        raises the leak guard's ceiling — everything up to the target step
        becomes sayable — so it may not be taken on the grader's word:

          · the claim is checked against that reference step with sympy, and
            an unverifiable jump simply does not happen
          · the jump stops short of the last step, so no amount of running
            ahead can make the answer sayable or close the problem
          · the next photo overrules all of it, because the estimator writes
            last_correct_step from what is actually on the page
        """
        ceiling = len(reference.steps) - 1     # never the final step
        proposed = verdict.reached_step
        claim = (verdict.reached_claim or "").strip()
        if isinstance(proposed, int) and proposed > asked:
            proposed = min(proposed, ceiling)
            step = next((s for s in reference.steps if s.idx == proposed), None)
            if step is not None and claim and step.expression and _claim_matches_step(
                claim, step.expression
            ):
                log.info("student ran ahead to step %d (%r checks out)", proposed, claim[:40])
                return proposed
            log.info("claim %r does not prove proposed step %s; checking the transcript",
                     claim[:40], proposed)

        # Gemini sometimes correctly grades the whole tangent but omits the
        # optional reached_step fields.  Recover only the narrow, mechanically
        # provable case: a spoken linear expression that matches a later step.
        for step in reversed(reference.steps):
            if step.idx <= asked or step.idx > ceiling or not step.expression:
                continue
            inferred = _spoken_linear_claim(transcript, step.expression)
            if inferred and _claim_matches_step(inferred, step.expression):
                log.info(
                    "spoken claim %r proves the student ran ahead to step %d",
                    inferred, step.idx,
                )
                return step.idx
        return asked

    async def _answer_question(
        self, ctx: ProblemContext, pending: HintRecord, question: str
    ) -> None:
        """The student asked why. Answer that, then hand the turn back.

        No store writes and no hint record: an explanation is not a hint, so
        it must not move the L1-L4 ladder or resolve the pending question.
        """
        # "어디가 틀렸어?" is answerable only with the diagnosis the tutor
        # already made — without it the explanation motivates the step in the
        # abstract and never says where the student went wrong.
        state = self.store.get_state()
        sink = self._new_hint_stream(paused=True)
        kwargs = {"on_delta": sink.push} if sink is not None else {}
        explain_task = asyncio.create_task(asyncio.to_thread(
            self.deps.hint_gen.explain,
            student_question=question,
            tutor_question=pending.hint_text,
            match=ctx.match,
            reference=ctx.reference,
            rec=ctx.recognition,
            target_step=pending.step,
            diagnosis=state,
            **kwargs,
        ))
        try:
            if sink is not None:
                await self._settle_filler()
                sink.activate()
            text = await explain_task
            streamed = await sink.finish(text) if sink is not None else False
        except BaseException:
            explain_task.cancel()
            if sink is not None:
                await sink.abort()
            raise
        log.info("explained a student question at step %d", pending.step)
        if not streamed:
            await self._speak(text, final=True)

    async def _finish_problem(self, ctx: ProblemContext, verdict, target_step: int) -> None:
        """The last step was answered correctly: congratulate and close.

        Deliberately not _deliver(): there is no next step to hint at, so a
        hint record here would sit unresolved forever and the policy would
        keep planning for a problem that is over. Speak, then drop the
        context — the next capture starts a fresh problem.
        """
        feedback = (verdict.feedback or "").strip()
        if feedback and leaks_answer(
            feedback, ctx.reference, target_step, visible_to_student(ctx.recognition)
        ):
            feedback = ""
        # The page files the problem under "해결한 문제" with its conversation.
        # An event, not a model call: which problem just closed is a fact the
        # orchestrator already holds, and facts ship as events here.
        try:
            await self.ws.send(make_event("solved", {
                "text": mathspeak.displayable(ctx.recognition.problem_text),
            }))
        except Exception:
            log.debug("could not send solved event (connection gone)")
        try:
            await self._speak(
                " ".join(part for part in (feedback, PROBLEM_DONE) if part), final=True
            )
        finally:
            log.info("problem %s solved; context closed", ctx.hash[:8])
            self.ctx = None
            self.prev_work = None
            self.last_transcript = None
            self.store.clear_state()

    async def _react(self, feedback: str | None, decision: Decision) -> bool:
        """Speak the turn's reaction NOW, while the hint is still being written.

        That concurrency is the whole point: the reaction is known seconds
        before the hint exists, and a student who has just answered deserves
        the "맞아요!" at reaction-time, not at reaction-plus-generation-time.
        Returns whether anything was said, so the hint that follows can drop
        its own opening acknowledgement — one per turn, as ever. A reaction
        that would leak the answer is dropped, not delayed.
        """
        feedback = (feedback or "").strip()
        if not feedback:
            return False
        reference = self.ctx.reference if self.ctx is not None else None
        seen = visible_to_student(self.ctx.recognition) if self.ctx is not None else []
        if reference is not None and leaks_answer(
            feedback, reference, decision.target_step, seen
        ):
            log.warning("reaction leaked the answer; staying quiet instead")
            return False
        # What _speak reports, not an assumption: a reaction dropped whole by a
        # barge-in was not heard, so the hint after it keeps its own opening.
        return await self._speak(feedback)

    # How long a confirmed work check keeps its follow-through open. Long
    # enough for "네, 다음은 어떻게 해요?" to be spoken and transcribed; short
    # enough that a page revisited after real thinking time is photographed.
    CONTINUE_WINDOW_S = 90.0

    async def _continue_confirmed(self, marker: tuple[str, float] | None) -> bool:
        """The follow-through: hint at the next step straight from a diagnosis
        made seconds ago, without asking the camera again.

        Only the turn right after a confirmed work check qualifies. The tutor
        just said "다음은 X 차례예요", the student asked to proceed, and
        nothing has had time to change on the page — a recapture would spend
        ~8 seconds of camera, VLM and estimator re-deriving the very state
        that produced the sentence they are replying to. Everything a fresh
        photo would yield — context, reference, diagnosis — is already in
        hand, so this goes straight to policy and hint. Any doubt (context
        gone, reference missing, window expired) falls back to the full
        capture pipeline; returning False costs nothing but the photo.
        """
        if marker is None or self.ctx is None:
            return False
        p_hash, stamped = marker
        if p_hash != self.ctx.hash or time.monotonic() - stamped > self.CONTINUE_WINDOW_S:
            return False
        current = self.store.get_state()
        reference = self.ctx.reference_if_ready()
        if current is None or reference is None:
            return False
        log.info(
            "follow-through after a confirmed work check: hinting at step %d "
            "without a recapture", current.last_correct_step + 1,
        )
        await self._stage("다음 단계를 짚어 보고 있어요")
        fresh_history = self.store.get_history(problem_hash=self.ctx.hash)
        decision = decide(current, fresh_history, "HINT_REQUEST")
        log.info("decision: %s", decision)
        await self._stage("힌트를 만들고 있어요")
        text, streamed = await self._generate_hint(
            decision, self.ctx.match, reference, self.ctx.recognition,
            fresh_history, None, live=True,
        )
        board = getattr(text, "board", ())
        await self._deliver(
            decision, text, self.ctx.hash, board, already_spoken=streamed
        )
        return True

    async def _handle_hint_request(self, question: str | None = None) -> None:
        """Capture → recognize → diagnose → hint, over a fresh photo.


        `question` is set when the student asked about their own work
        ("풀이 맞아?"). The pipeline is identical — that is the point:
        checking work IS re-reading the worksheet and re-diagnosing. It only
        changes what comes out: their question reaches the hint phrasing, and
        the tutor says what it saw before hinting.
        """
        marker, self._continue_from = self._continue_from, None  # single use
        if question is None and await self._continue_confirmed(marker):
            return
        await self._stage("카메라에서 사진을 받고 있어요")
        jpeg = await self._request_capture()
        state = self.store.get_state() or StudentState()
        if jpeg is None:
            cur_hash = self.ctx.hash if self.ctx is not None else ""
            decision = decide(
                state, self.store.get_history(problem_hash=cur_hash or None),
                "RECOGNITION_FAILED",
            )
            await self._deliver(decision, self.deps.hint_gen.generate(
                decision, MatchResult(tier=Tier.NEW), None, Recognition(problem_text=""), []
            ), cur_hash)
            return

        await self._stage("쓴 풀이를 읽고 있어요" if question else "문제를 읽고 있어요")
        rec = await asyncio.to_thread(self.deps.recognizer.recognize, jpeg)
        # What the VLM actually read. Nothing here comes from the solver, so no
        # answer can reach the log through this line — it is the worksheet, which
        # the student is looking at anyway.
        log.info(
            "recognized: conf=%.2f problem=%r equations=%s work=%s uncertain=%s",
            rec.confidence, rec.problem_text[:60], rec.equations,
            rec.student_work, rec.uncertain_regions,
        )
        if rec.confidence < self.deps.settings.recog_conf_threshold:
            log.warning(
                "confidence %.2f < RECOG_CONF_THRESHOLD %.2f → UNCERTAIN → the tutor will "
                "ask to see the worksheet again",
                rec.confidence, self.deps.settings.recog_conf_threshold,
            )
        elif question is None:
            # A work check never enters here, whatever the VLM read: the same
            # page re-read with sloppy handwriting can miss the same-problem
            # match, and that used to rewrite the card AND stack the 3-line
            # readout on top of the opener — four filler-ish lines before the
            # verdict the student actually asked for. On a work check the card
            # holds still, nothing is read back, and the one opener is all the
            # filler the turn speaks; the stage line ("쓴 풀이를 읽고 있어요")
            # carries the progress on screen.
            if self.ctx is None or not (
                self.ctx.hash == problem_hash(rec) or self._same_problem(rec)
            ):
                # A NEW problem reaches the page's problem card — and only a
                # new one. The VLM re-reads the same page with small wording
                # differences every turn, and a card that rewrites itself
                # mid-problem reads as the tutor changing its mind.
                await self._send_problem(rec)
                # First sight of a NEW problem: say what KIND of problem it is
                # and what to check, while the matcher and the phraser think.
                # Not the problem read back verbatim — that ran ten seconds on
                # a long statement and the finished hint had to wait it out;
                # the card on screen carries the wording instead.
                self._narrate(self._preflight_line(rec))
        # A work check holds onto its problem: the student is asking about the
        # work in front of them, and a re-read wobble must not reset the very
        # context that can grade it (the 등비수열 sessions did exactly that).
        keep = bool(question) and not self._clearly_different_problem(rec)
        ctx = await self._problem_context(rec, keep_current=keep)

        # Diagnose before helping (spec rule 4). State/history are prefetched
        # here by the orchestrator — never fetched by the model itself.
        # History is always scoped to THIS problem's hash.
        await self._stage("풀이를 살펴보고 있어요")
        prev_state = self.store.get_state()
        history = self.store.get_history(problem_hash=ctx.hash)
        reference = ctx.reference_if_ready()
        if reference is None and question and not _solve_dead(ctx.solving):
            # "풀이 맞아?" is a yes/no question and the reference is its
            # measuring stick. The hint path shrugs and opens from concepts —
            # a work check WAITS: answering a verdict question with another
            # question was this turn's live failure mode. The stage line and
            # the opener cover the wait on screen and in the ear.
            await self._stage("기준 풀이를 계산하고 있어요")
            try:
                reference = await ctx.reference_ready()
            except Exception:
                log.exception("solver died during a work check; diagnosing from concepts")
                reference = None
        if reference is None:
            # The solver is still writing the reference solution. The full
            # diagnosis compares work against that solution, so it cannot run
            # yet — but its deterministic pre-checks (unreadable photo, empty
            # page) need no reference and MUST still run: a garbled frame has
            # to end in "다시 보여 줄래요?", not in a hint about garbage. Past
            # those, the first hint on a new problem opens at L1 from the
            # concepts regardless (policy R5), and asking it NOW beats a
            # perfectly targeted question after ~25 seconds of dead air.
            new_state = self.deps.estimator.precheck(
                rec=rec, prev_state=prev_state, prev_work=self.prev_work,
                history=history, transcript=self.last_transcript,
            ) or StudentState(
                current_step="문제를 파악하는 중",
                last_correct_step=0,
                status="STUCK",
                attempt_count=prev_state.attempt_count + 1 if prev_state else 1,
            )
            log.info("solver still running: first hint from concepts alone")
        else:
            new_state = await asyncio.to_thread(
                self.deps.estimator.estimate,
                rec=rec,
                reference=reference,
                prev_state=prev_state,
                prev_work=self.prev_work,
                history=history,
                transcript=self.last_transcript,
            )
        self.prev_work = rec.student_work
        # consumed: an old utterance must not be re-read as evidence next turn
        self.last_transcript = None

        # Orchestrator-owned writes: state, then pending-hint effectiveness
        # (only a pending hint of the SAME problem can be resolved).
        pending = self.store.pending_hint(ctx.hash)
        self.store.set_state(new_state)
        log.info(
            "state: %s at step %d%s",
            new_state.status, new_state.last_correct_step,
            f" ({new_state.misconception})" if new_state.misconception else "",
        )
        # an UNCERTAIN estimate carries no evidence either way: keep the hint
        # pending rather than resolving its effectiveness on a blurry frame.
        # A skipped estimate (reference still solving) carries none either.
        if (reference is not None and pending is not None and prev_state is not None
                and new_state.status != "UNCERTAIN"):
            if (new_state.status == "CORRECT"
                    and new_state.last_correct_step < pending.step):
                # Correct SO FAR is not evidence that the pending next step was
                # attempted.  Keep that question current; resolving it exposed
                # an ancient unresolved step-1 hint underneath.
                log.info(
                    "hint L%d at step %d remains pending: correct work reaches only step %d",
                    pending.level, pending.step, new_state.last_correct_step,
                )
                effective = None
            else:
                effective = hint_was_effective(prev_state, new_state)
            # The escalation one turn later reads "hint L1 ineffective" and
            # names no evidence. This is the evidence: two states and the
            # comparison between them, at the moment it is made.
            if effective is not None:
                log.info(
                    "hint L%d at step %d: %s (was %s at step %d)",
                    pending.level, pending.step,
                    "helped" if effective else "did not help",
                    prev_state.status, prev_state.last_correct_step,
                )
                self._resolve_hint(pending.id, effective)

        # Always read state/history through the store right before the policy.
        current = self.store.get_state() or new_state

        if question and current.status == "CORRECT":
            # They asked whether their work is right, and it is. That question
            # deserves an answer, not a hint: hinting here would push them at a
            # step they have not reached and imply something was wrong.
            # Deliberately not _deliver(): no hint was given, so nothing should
            # enter the hint history or move the L1-L4 ladder.
            #
            # But the verdict alone ended the conversation. A student four
            # steps into a seven-step problem heard "이대로 하면 돼요. 또 궁금한
            # 게 있으면 물어봐 주세요" and was left standing — confirmed, and
            # abandoned. Opening the next piece of work here does not move the
            # ladder or enter history; it answers the question they are about
            # to ask without reading an internal step label aloud.
            log.info("work check at step %d: correct so far", current.last_correct_step)
            line = self._confirmed_line(current, reference)
            await self._speak(line, final=True)
            # The forward invitation often gets one reply — "그거 어떻게
            # 해요?" — and that reply must not begin with a camera shutter:
            # the diagnosis it needs is the one that was just made.
            self._continue_from = (ctx.hash, time.monotonic())
            # the confirmation is also the one moment the whole picture is
            # right so far — let the board show it
            self._spawn(self._illustrate(
                Decision(Action.WAIT, 0, current.last_correct_step + 1, None,
                         "work confirmed"),
                line, (), ctx.hash, student_said=question,
            ))
            return

        fresh_history = self.store.get_history(problem_hash=ctx.hash)
        if current.status == "UNCERTAIN" and not fresh_history:
            # An unreadable photo has no trustworthy identity: problem_hash is
            # built from what the VLM read, so a garbled read mints a new hash
            # every turn, hides the recapture we just asked for, and the tutor
            # repeats that one sentence forever. When we cannot see the problem,
            # the whole history is the right history.
            fresh_history = self.store.get_history()
        decision = decide(current, fresh_history, "HINT_REQUEST")
        log.info("decision: %s", decision)

        await self._stage("힌트를 만들고 있어요")
        # Same overlap as the answer turn: the reaction ("음, 지금 쓴 줄을 같이
        # 볼까요?") is fixed text with cached TTS, so it plays at once while the
        # hint is still being generated.
        # A plain hint has no reaction in front of it, so its safe word stream
        # can go straight to the student. Work checks deliberately keep the
        # reaction first; their generated follow-up uses the complete path.
        if question is None:
            text, streamed = await self._generate_hint(
                decision, ctx.match, reference, rec, fresh_history, question,
                live=True,
            )
            board = getattr(text, "board", ())
            await self._deliver(
                decision, text, ctx.hash, board, already_spoken=streamed
            )
            return

        sink = self._new_hint_stream(paused=True)
        kwargs = {"on_delta": sink.push} if sink is not None else {}
        hint_task = asyncio.create_task(asyncio.to_thread(
            self.deps.hint_gen.generate,
            decision, ctx.match, reference, rec, fresh_history, question,
            **kwargs,
        ))
        try:
            spoke_reaction = bool(question) and await self._react(
                self._work_reaction(current), decision
            )
            if sink is not None:
                sink.activate()
            text = await hint_task
            streamed = await sink.finish(text) if sink is not None else False
        except Exception:
            hint_task.cancel()
            if sink is not None:
                await sink.abort()
            raise
        # read before any strip loses the subclass that carries them
        board = getattr(text, "board", ())  # before any strip loses the subclass
        if spoke_reaction and not streamed:
            text = strip_leading_acknowledgement(text)
        await self._deliver(
            decision, text, ctx.hash, board, already_spoken=streamed
        )

    async def _generate_hint(
        self, decision, match, reference, rec, history, student_answer,
        *, live: bool = False,
    ) -> tuple[str, bool]:
        """Generate one hint, optionally committing safe words as they arrive.

        Device sessions have no incremental text/audio sink and naturally use
        the old complete-line path. BrowserSession overrides
        ``_new_hint_stream``; only then is ``on_delta`` attached.
        """
        # A filler may still own the speaker while the model starts writing.
        # Buffer safe deltas immediately, but do not let the final hint compete
        # with that filler for the one TTS websocket.  The model and filler
        # still run concurrently; activation is merely the audio hand-off.
        sink = self._new_hint_stream(paused=True) if live else None
        kwargs = {"on_delta": sink.push} if sink is not None else {}
        hint_task = asyncio.create_task(asyncio.to_thread(
            self.deps.hint_gen.generate,
            decision, match, reference, rec, history, student_answer,
            **kwargs,
        ))
        try:
            if sink is not None:
                await self._settle_filler()
                sink.activate()
            text = await hint_task
            if sink is None:
                return text, False
            heard = await sink.finish(text)
            return text, heard
        except BaseException:
            hint_task.cancel()
            if sink is not None:
                await sink.abort()
            raise

    def _new_hint_stream(self, paused: bool = False):
        """BrowserSession supplies the live eye/ear sink."""
        return None

    @staticmethod
    def _work_reaction(state: StudentState) -> str:
        """What the tutor says it saw, before it hints. Never what is wrong."""
        return WORK_CHECK_REACTIONS.get(state.status, WORK_CHECK_DEFAULT)

    @staticmethod
    def _confirmed_line(state: StudentState, reference) -> str:
        """The verdict, plus a question that opens the next piece of work.

        Only a noun-form step description rides in the sentence (the same rule
        the hint templates follow: a sentence cannot wear a particle), only
        its description becomes an invitation (never an expression), and the composed line
        stays at the level of an action rather than revealing its expression.
        A finished problem gets the congratulation instead.
        """
        if reference is None:
            return WORK_CONFIRMED
        nxt = next(
            (s for s in reference.steps if s.idx == state.last_correct_step + 1),
            None,
        )
        if nxt is None:
            return "맞아요! 여기까지면 다 풀었어요. 어떻게 구했는지 한번 정리해 볼까요?"
        name = nxt.description.strip().rstrip(" .!?…")
        if not name or SENTENCE_STEP_RE.search(name):
            return WORK_CONFIRMED
        return f"맞아요! 여기까지 잘했어요. {guided_step_question(name, nxt.idx)}"

    def _same_problem(self, rec: Recognition) -> bool:
        """Is this the worksheet we are already working on?

        The hash is exact by design, but the VLM re-reads the same photo with
        small wording differences, and a hash miss would reset the student's
        state — which is why a hint could loop at L1 forever. Equivalent
        equations (or identical normalized text) mean the same problem.
        """
        if self.ctx is None:
            return False
        cached = self.ctx.recognition
        if rec.equations and len(rec.equations) == len(cached.equations):
            if all(
                mathnorm.equations_equivalent(a, b, allow_scale=False)
                for a, b in zip(rec.equations, cached.equations)
            ):
                return True
            # Inconclusive is not "different": unparseable or shifted reads
            # fall through to the text. Declaring a new problem here reset the
            # student's state and orphaned their pending hint MID-PROBLEM —
            # the 등비수열 chain equality did exactly that on every re-read.
        a = mathnorm.normalize_text(rec.problem_text)
        b = mathnorm.normalize_text(cached.problem_text)
        if not a or not b:
            return False
        # The VLM truncates long problem statements at different points on
        # different reads; a read that is a clean prefix of the other (with
        # enough of it to mean something) is the same problem, shorter.
        shorter, longer = sorted((a, b), key=len)
        return len(shorter) >= 20 and longer.startswith(shorter)

    def _verified_steps(self, ctx: ProblemContext) -> list[str]:
        """Reference expressions for the steps the student has COMPLETED.

        The diagnosis already says how far they are (last_correct_step), and
        every step at or below it is theirs — written on the page, or said
        aloud and sympy-checked on the way in. The drawing hand gets this
        list because a photograph cannot show a spoken derivation: without
        it, a tangent the student earned out loud never reached the board.
        Nothing past the frontier leaves this method.
        """
        state = self.store.get_state()
        if ctx.reference is None or state is None:
            return []
        return [
            s.expression
            for s in ctx.reference.steps
            if s.idx <= state.last_correct_step and s.expression
        ]

    def _clearly_different_problem(self, rec: Recognition) -> bool:
        """POSITIVE evidence that this is another problem: equations that
        parse on both sides, compare pair by pair, and disagree. Anything
        less — a chain the parser choked on, a count that shifted, a text
        that drifted mid-string — is a re-read hiccup as easily as a new
        page. A work check stays on the problem whose work it is checking
        unless the evidence clears this bar.
        """
        if self.ctx is None:
            return False
        cached = self.ctx.recognition
        if not rec.equations or len(rec.equations) != len(cached.equations):
            return False
        both = (*rec.equations, *cached.equations)
        if any(re.search(r"[가-힣]", eq) for eq in both):
            # sympy happily parses Korean words as symbols, which would let a
            # junk read count as "clear evidence" — words are not equations
            return False
        if not all(mathnorm.parseable_claims(eq) for eq in both):
            return False
        return not all(
            mathnorm.equations_equivalent(a, b, allow_scale=False)
            for a, b in zip(rec.equations, cached.equations)
        )

    async def _problem_context(
        self, rec: Recognition, keep_current: bool = False
    ) -> ProblemContext:
        h = problem_hash(rec)
        if self.ctx is not None and (
            keep_current or self.ctx.hash == h or self._same_problem(rec)
        ):
            # Cached problem: reuse match/reference and the tags (the problem
            # did not change, only the student's work). Re-tagging here would
            # pay for an LLM call per hint and let the tags drift mid-problem.
            rec.problem_type = self.ctx.recognition.problem_type
            rec.concepts = list(self.ctx.recognition.concepts)
            self.ctx.recognition = rec
            if self.ctx.reference is None and _solve_dead(self.ctx.solving):
                # the earlier background attempt failed: this turn is the retry
                self.ctx.solving = self._start_solve(rec, h)
            return self.ctx
        if self.ctx is not None:
            # Different problem: the old student state/work/transcript are
            # meaningless here — start fresh (history stays, scoped by hash).
            log.info("problem changed (%s -> %s): resetting state", self.ctx.hash[:8], h[:8])
            self.store.clear_state()
            self.prev_work = None
            self.last_transcript = None
        # A new problem. The tags arrived with the recognition itself (one VLM
        # call does both jobs now — see tutor/vision/recognizer.py); everything
        # downstream (matching, RAG, hint phrasing) reads them off the
        # Recognition, and the cached branch above keeps them stable per problem.
        match = await asyncio.to_thread(self.deps.matcher.match, rec)
        # The one line that says whether this turn runs on curated knowledge
        # or on a model. When a rehearsal problem comes back NEW, the fix is
        # in the log two lines up: "recognized: … equations=[…]" is the exact
        # variant to add to the presolve entry.
        log.info(
            "match tier=%s problem=%s reference=%s",
            match.tier.value,
            getattr(match.problem, "id", None),
            "verified" if match.reference is not None and match.reference.verified
            else ("yes" if match.reference is not None else "none"),
        )
        reference = match.reference
        solving = None
        if reference is None:
            # CONCEPT/NEW → Grok Solver (spec rule 2) — but off the clock. At
            # ~25s it was half of a first turn, and nothing the FIRST hint says
            # needs it: the policy opens at L1 from the concepts regardless, so
            # the solver writes the reference while the student hears that hint.
            solving = self._start_solve(rec, h)
        self.ctx = ProblemContext(
            hash=h, recognition=rec, match=match, reference=reference, solving=solving
        )
        return self.ctx

    def _start_solve(self, rec: Recognition, h: str) -> asyncio.Task:
        """Write the reference solution while the student hears the first hint."""

        async def solve() -> ReferenceSolution:
            started = time.monotonic()
            result = await asyncio.to_thread(self.deps.solver.solve, rec, h)
            timing.record("solve", time.monotonic() - started, "background")
            return result

        task = asyncio.create_task(solve())
        self._tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            # retrieve the exception so a student who never answers does not
            # leave an "exception was never retrieved" corpse in the log
            if not t.cancelled() and t.exception() is not None:
                log.error("background solve for %s failed: %r", h[:8], t.exception())

        task.add_done_callback(done)
        return task

    async def _speak(self, text: str, *, final: bool = False) -> bool:
        """Say something to the student, after the filler has had its say.

        `text` stays ORIGINAL all the way to _say: the ear and the eye part
        ways there, and nowhere earlier — the hint history, the leak guard and
        the TTS cache keys all see what was actually generated.

        `final` is a LABEL, not logic: it marks the line that ends its turn
        (the hint, the confirmation, the retry prompt), so the browser can
        choreograph the mascot's return before this one and stay put through
        fillers and mid-turn reactions. Nothing on the server branches on it.

        Returns whether the student heard any of it — False only when the line
        was dropped whole, which is a barge-in that started before this one
        did. A line cut off halfway was still said.
        """
        await self._settle_filler()
        # TTS is part of the wait, and a cached phrase is not — worth telling apart.
        with timing.stage("speak"):
            return await self._say(text, final=final) is not False

    # --- filling the thinking silence ----------------------------------------

    def _start_filler(self, opener: str | None = None) -> None:
        """Begin a filler that will play only if the thinking outlasts it.

        Deliberately not awaited: this runs *beside* the pipeline, which is the
        whole point. It costs the student nothing when the answer is quick,
        because the phrase does not start until FILLER_DELAY_MS has passed with
        no answer, and _settle_filler cancels it if the answer arrives first.

        `opener` replaces the canned phrase with something that belongs to THIS
        turn — the echo of what the student just said, "쓴 풀이를 볼게요" — which
        is what makes the wait read as listening rather than as buffering.
        """
        bank = self.deps.fillers
        if bank is None or not self.deps.settings.filler_enabled or self._filler is not None:
            return
        self._filler_spoke = False
        self._filler_open = True
        self._filler_lines = asyncio.Queue()
        self._filler = asyncio.create_task(self._fill_silence(bank, opener))

    def _narrate(self, text: str) -> None:
        """Queue a mid-turn line — the problem read back once recognition knows
        it. Spoken only while the turn is still thinking: the moment the real
        answer is ready, _settle_filler drops anything not yet said."""
        if self._filler is None or not self._filler_open or not text:
            return
        self._filler_lines.put_nowait(text)

    async def _fill_silence(self, bank, opener: str | None = None) -> None:
        try:
            await asyncio.sleep(max(0.0, self.deps.settings.filler_delay_ms / 1000))
            text = opener or bank.pick()
            if text:
                # Past this point the filler owns the speaker, so _settle_filler
                # must wait for it rather than cancel it — cancelling mid-utterance
                # is how you get half a word followed by the real answer.
                self._filler_spoke = True
                log.info("filler: %s", text)
                await self._say(text)
            # Keep the floor while the turn keeps thinking: narrations pushed by
            # the pipeline (the problem readout) play here, in order.
            while True:
                line = await self._filler_lines.get()
                if line is None or not self._filler_open:
                    return
                self._filler_spoke = True
                log.info("filler (narration): %s", line[:40])
                await self._say(line)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a filler failure is not a lesson failure
            log.exception("filler playback failed; continuing in silence")

    async def _settle_filler(self) -> None:
        """Never talk over the filler, and never make the student wait for one
        that has not started — or for narrations it never got to."""
        self._filler_open = False
        task, self._filler = self._filler, None
        if task is None or task.done():
            return
        if not self._filler_spoke:
            task.cancel()  # thinking won the race: stay quiet
        else:
            self._filler_lines.put_nowait(None)  # finish the current line, skip the rest
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def _say(self, text: str, final: bool = False) -> None:
        """Say it on the machine running the server (same room as the student).

        BrowserSession overrides this to ship the audio to the device instead.
        The ear gets the spoken form here — "f 프라임 1" — and only the ear:
        nothing on this path is displayed. `final` is the turn-ending label
        from _speak; a local speaker has no mascot, so it is ignored here.
        """
        await self.ws.send(make_event("speech_state", {"state": "speaking"}))
        try:
            await asyncio.to_thread(self.deps.speaker.speak, mathspeak.speakable(text))
        finally:
            # whatever happens to TTS, never leave the device muted
            try:
                await self.ws.send(make_event("speech_state", {"state": "idle"}))
            except Exception:
                pass

    async def _illustrate(
        self, decision: Decision, hint: str, board: tuple, problem_hash: str,
        student_said: str | None = None,
    ) -> None:
        """Redraw the problem's grid beside a hint that is already being
        spoken. Never awaited by the turn: a picture is optional, and a slow
        or broken drawing hand must cost the lesson nothing."""
        ctx = self.ctx
        if self.deps.illustrator is None or ctx is None:
            return
        verified = self._verified_steps(ctx)
        if (
            not ctx.scene
            and not illustrator.wants_a_picture(hint)
            and not any(illustrator.drawable(v) for v in verified)
        ):
            # Nothing drawn yet, the sentence never mentions a shape, and the
            # student has not earned a curve worth showing. Once a grid IS
            # open it is kept current every turn, because what changes it is
            # often the student's answer, not the tutor's words — and a newly
            # verified LINE opens one even mid-algebra, because "l을 구했는데
            # l이 안 그려져 있다" is a board failing at its one job.
            return
        spec = await asyncio.to_thread(
            self.deps.illustrator.draw,
            hint=hint,
            problem_text=ctx.recognition.problem_text,
            equations=list(ctx.recognition.equations),
            student_work=list(ctx.recognition.student_work),
            board=[b.expr for b in board],
            misconception=decision.misconception,
            level=decision.level,
            scene=list(ctx.scene),
            span=ctx.span,
            student_said=student_said,
            verified=verified,
        )
        if spec is None or not spec.curves:
            return
        # The same gate as the words: a curve of the answer IS the answer.
        seen = visible_to_student(ctx.recognition)
        if ctx.reference is not None and any(
            leaks_answer(part, ctx.reference, decision.target_step, seen)
            for part in (*(c.expr for c in spec.curves), spec.caption) if part
        ):
            log.info("the sketch would leak the answer; nothing is drawn")
            return
        # A window that jumps every turn reads as a new picture rather than
        # the same board, so the previous one stands unless a new one is given.
        span = ctx.span or plot.DEFAULT_SPAN
        if spec.x_min is not None and spec.x_max is not None and spec.x_min < spec.x_max:
            span = (spec.x_min, spec.x_max)
        # through the student's own window when they have chosen one — the
        # next turn's redraw must not snap their view back to the guess
        svg = await asyncio.to_thread(
            plot.function_svg, spec.curves, span, view=ctx.view
        )
        if not svg:
            return
        # The turn moved on while we drew: a picture of the last hint landing
        # on the next problem is worse than no picture at all.
        if self.ctx is not ctx or ctx.hash != problem_hash:
            log.info("dropping a sketch whose problem is over")
            return
        if getattr(self, "_interrupted", False):
            log.info("dropping a sketch the student talked over")
            return
        gone = [c.label or c.expr for c in ctx.scene
                if c.expr not in {n.expr for n in spec.curves}]
        ctx.scene, ctx.span, ctx.caption = tuple(spec.curves), span, spec.caption
        try:
            await self.ws.send(make_event("figure", {
                # one canvas per problem: the page replaces it in place, so
                # the scene changes rather than a second picture appearing
                "id": problem_hash or "scene",
                "svg": svg,
                "of": ", ".join(c.expr for c in spec.curves),
                "note": spec.caption,
                "v": ctx.view_rev,
            }))
        except Exception:
            log.debug("could not send figure event (connection gone)")
        log.info(
            "scene: %s over %s%s — %s",
            [f"{c.label or '?'}={c.expr}({c.role})" for c in spec.curves], span,
            f", wiped {gone}" if gone else "", spec.why[:60],
        )

    async def _adjust_view(self, event: str, data: dict) -> None:
        """The same scene through the window the student is choosing.

        figure_zoom {dir: ±1} scales the rectangle about its centre (+1 shows
        more of the plane), figure_pan {dx, dy} slides it by fractions of
        itself, figure_reset hands the window back to the automatic guess.
        Nothing else changes: the curves already passed the leak gate when
        they were first drawn, and the caption is the illustrator's.
        """
        ctx = self.ctx
        if ctx is None or not ctx.scene:
            return
        span = ctx.span or plot.DEFAULT_SPAN

        if event == "figure_reset":
            if ctx.view is None:
                return
            ctx.view = None
        else:
            # their rectangle, or the automatic one the moment they touch it
            view = ctx.view
            if view is None:
                view = await asyncio.to_thread(
                    plot.compute_view, list(ctx.scene), span
                )
                if view is None:
                    return
            x0, x1, y0, y1 = view
            w, h = x1 - x0, y1 - y0
            if event == "figure_zoom":
                try:
                    direction = int(data.get("dir", 0))
                except (TypeError, ValueError):
                    return
                if not direction:
                    return
                factor = 1.4 if direction > 0 else 1 / 1.4
                # bounded against the problem's own span, so the window can
                # neither vanish into a point nor grow into empty plane
                base = (span[1] - span[0]) or 1.0
                if not (base * 0.25 <= w * factor <= base * 5.0):
                    return
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = w * factor, h * factor
                ctx.view = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            else:  # figure_pan
                try:
                    dx = max(-1.0, min(1.0, float(data.get("dx", 0))))
                    dy = max(-1.0, min(1.0, float(data.get("dy", 0))))
                except (TypeError, ValueError):
                    return
                if not dx and not dy:
                    return
                # dy is screen-down; the y axis grows up
                ctx.view = (x0 + dx * w, x1 + dx * w, y0 - dy * h, y1 - dy * h)

        ctx.view_rev += 1
        svg = await asyncio.to_thread(
            plot.function_svg, list(ctx.scene), span, view=ctx.view
        )
        if not svg or self.ctx is not ctx:
            return
        try:
            await self.ws.send(make_event("figure", {
                "id": ctx.hash or "scene",
                "svg": svg,
                "of": ", ".join(c.expr for c in ctx.scene),
                "note": ctx.caption,
                "v": ctx.view_rev,
                "user": True,          # the student adjusted their view: the
                                       # squid stays put and nothing is spoken
            }))
        except Exception:
            log.debug("could not send the adjusted figure (connection gone)")
        log.info("figure view %s -> %s", event, ctx.view or "auto")

    def _resolve_hint(self, hint_id: int, effective: bool) -> None:
        """Answer a pending hint, remembering which one — so a turn the student
        never hears can put it back."""
        self.store.mark_hint_effective(hint_id, effective)
        self._resolved_hint = hint_id

    async def _deliver(
        self, decision: Decision, text: str, problem_hash: str = "",
        board: tuple[str, ...] = (), student_said: str | None = None,
        *, already_spoken: bool = False,
    ) -> None:
        if board:
            # written as the voice starts, like a tutor's hand reaching the
            # whiteboard mid-sentence — display notation for the eye
            try:
                await self.ws.send(make_event("board", {"lines": [
                    {"expr": mathspeak.displayable(b.expr), "note": b.note}
                    for b in board
                ]}))
            except Exception:
                log.debug("could not send board event (connection gone)")
        if text:
            # The picture is drawn WHILE this is spoken. A hint takes ~8s to
            # say, which is time the drawing hand can use — and by starting
            # here it gets what no parallel design could: the finished
            # sentence, so it draws what was said instead of guessing.
            self._spawn(
                self._illustrate(decision, text, board, problem_hash, student_said)
            )
            if not already_spoken and not await self._speak(text, final=True):
                # The student talked over this whole turn, so _say skipped every
                # line of it: this hint exists only in the log. Recording it
                # would escalate the ladder for a question that was never asked
                # — measured live, a student jumped L1 to L3 having heard only
                # L1. And the hint it answered is, from where they sit, still
                # the question on the table: put it back.
                log.info("L%d not delivered (barge-in): not recorded", decision.level)
                if self._resolved_hint is not None:
                    self.store.unresolve_hint(self._resolved_hint)
                    self._resolved_hint = None
                return
        self.store.append_hint(
            problem_hash=problem_hash,
            step=decision.target_step,
            level=decision.level,
            action=decision.action.value,
            hint_text=text,
        )
        await self.ws.send(
            make_event("hint_issued", {"level": decision.level, "action": decision.action.value})
        )
