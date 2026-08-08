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
from dataclasses import dataclass, field

from tutor.config import Settings
from tutor.hints.generator import (
    HintGenerator,
    strip_leading_acknowledgement,
    visible_to_student,
)
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

# When it is not right, the tutor still has to say it looked before it hints.
# Fixed text, never generated: it may say THAT the tutor looked, never WHAT is
# wrong. Naming the mistake is the hint's job, and the leak guard's.
WORK_CHECK_REACTIONS: dict[str, str] = {
    "UNCERTAIN": "",  # ASK_RECAPTURE already says it cannot see the page
}
WORK_CHECK_DEFAULT = "음, 지금 쓴 줄을 같이 볼까요?"

# The filler opener for a work check: fixed, so its TTS is cached and it plays
# the instant the delay elapses — while the camera is still being asked.
WORK_CHECK_OPENER = "네, 지금 쓴 풀이를 확인하고 있어요."

# The readout frame: everything around the problem text is fixed, so its TTS
# is pre-rendered and the only synthesis the narration waits for is the one
# line that is different every time — the problem itself. The closer turns the
# readout into a teaching move: the things worth checking before starting.
# Two variants because Korean quotes with 라는 after a vowel, 이라는 after a
# final consonant — chosen off the LAST SPOKEN syllable of the problem line.
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
    """
    text = " ".join(transcript.split()).rstrip("?.!…, ")
    if not text or len(text) > 24:
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


def readout_of(rec: Recognition) -> list[str]:
    """The problem, read back while the tagger and phraser think — as three
    LINES, because the split is what the latency comes down to: the opener and
    the closer never change, so their audio is already rendered (CachedSpeech),
    and the problem itself is the only line that pays for TTS.

    Also the moment a misread photo gets caught: the student hears what the
    tutor believes the problem says, seconds before any hint depends on it.
    """
    text = " ".join(rec.problem_text.split())
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text)  # exam numbering: "6. "
    text = re.sub(r"\[\s*\d+\s*점\s*\]", "", text).strip()  # point tags: "[3점]"
    if not text:
        return []
    # kept as NOTATION: _say speaks it through speakable(), the browser shows
    # it through displayable() — one narration, two renderings
    if len(text) > 130:
        text = text[:130] + "…"
    # 라는/이라는 follows what the EAR will hear last, not what the page shows:
    # the particle attaches to the spoken form of the problem's final syllable.
    heard = mathspeak.speakable(text).rstrip("?.!…, \"'")
    closer = READOUT_CLOSERS[mathspeak.ends_in_consonant(heard)]
    return [READOUT_OPENER, text, closer]


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
        self._tasks: set[asyncio.Task] = set()
        self._filler: asyncio.Task | None = None
        self._filler_spoke = False
        self._filler_open = False
        self._filler_lines: asyncio.Queue = asyncio.Queue()

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
        try:
            transcript = await asyncio.to_thread(
                self.deps.transcriber.transcribe, pcm, sample_rate
            )
            text = transcript.text
        except Exception:
            log.exception("STT failed; asking the student to repeat")
            text = ""  # graded as "unclear" below, so the student hears back

        # Quality gate: noise transcribed as glyphs, silence, or a hesitation
        # sound is not an answer. Ask again instead of grading it — and leave
        # any pending hint pending, so the real answer still lands on it.
        quality = classify_transcript(text)
        if quality != "ok":
            log.info("utterance rejected (%s): %r", quality, text)
            await self._send_transcript(text, True)
            try:
                await self._speak(RETRY_PROMPTS[quality])
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

    # --- capture --------------------------------------------------------------

    async def _request_capture(self) -> bytes | None:
        """The student's worksheet, from whichever eye can see it.

        The session's own device first (a simulator image, the browser's
        attached photo). If it has no camera it answers capture_failed, and a
        phone connected on /camera — a different socket entirely — is asked
        instead. Voice and vision can therefore live on different machines.
        """
        timeout = self.deps.settings.capture_timeout_s
        started = time.monotonic()
        jpeg = await self._capture_from_device()
        eye = "session device"
        if not jpeg:
            cameras = self.deps.cameras
            if self.deps.settings.input_mode != "camera":
                # INPUT_MODE=upload: the picture comes from the browser's file
                # picker and nowhere else. Falling through to a camera device
                # here would silently hand the tutor a different photo from the
                # one the student just chose.
                log.warning("no worksheet photo attached (INPUT_MODE=upload)")
                return None
            if not cameras:
                log.warning("no eye at all: the session device has no camera and none "
                            "is connected on /camera")
                return None
            log.info("no local camera; asking a connected camera device")
            started = time.monotonic()  # the local device's answer was not the wait
            jpeg = await cameras.capture(timeout)
            eye = "camera device"
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
        # a work check earns its own opener: they asked about THEIR page, and
        # "네, 지금 쓴 풀이를 한번 볼게요" answers that before the camera has moved
        self._start_filler(WORK_CHECK_OPENER if question else None)
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
            await self._settle_filler()
            self._busy = False

    async def handle_answer(self, transcript: str, pending: HintRecord) -> None:
        """The student answered the tutor's question out loud."""
        if self._busy:
            log.info("answer ignored: a turn is already running")
            return
        self._busy = True
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
            await self._settle_filler()
            self._busy = False

    async def _handle_answer(self, transcript: str, pending: HintRecord) -> None:
        ctx = self.ctx
        assert ctx is not None  # a pending hint implies a problem context

        # The reference is the measuring stick for a spoken answer, so this
        # turn cannot start without it. The wait is almost always zero: the
        # solver ran while the first hint was being spoken and heard.
        try:
            reference = await ctx.reference_ready()
        except Exception:
            # The background solve died. Grading is impossible, but the student
            # is mid-sentence expecting a reply — fall back to the capture flow,
            # which restarts the solve (see _problem_context) and still says
            # something useful meanwhile. Not the public entry point: this turn
            # already holds the busy flag.
            log.exception("no reference for the answer turn; re-running the hint flow")
            self.last_transcript = transcript
            await self._handle_hint_request()
            return

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
            self.store.set_state(
                prev.model_copy(
                    update={
                        "current_step": f"{pending.step}단계를 말로 설명함",
                        "last_correct_step": max(prev.last_correct_step, pending.step),
                        "status": "CORRECT",
                        "misconception": None,
                        "attempt_count": 1,
                        "previous_hint_effective": True,
                    }
                )
            )
            self.store.mark_hint_effective(pending.id, True)
            if pending.step >= len(reference.steps):
                await self._finish_problem(ctx, verdict, pending.step)
                return
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
            self.store.mark_hint_effective(pending.id, False)
        else:
            # UNCLEAR: no evidence either way. Leaving the hint unresolved is
            # what makes the policy re-ask at the same level (R9).
            log.info("answer unclear: re-asking L%d at step %d", pending.level, pending.step)

        state = self.store.get_state() or prev
        history = self.store.get_history(problem_hash=ctx.hash)
        decision = decide(state, history, "HINT_REQUEST")
        log.info("decision after answer (%s): %s", verdict.verdict, decision)

        # The reaction is ready NOW and the hint needs ~5s of model. Speaking
        # "맞아요, 그렇게 하면 돼요!" WHILE the next question is being written
        # is where the answer turn stops feeling slow: first meaningful sound
        # at evaluate-time instead of evaluate+phrase+TTS-time.
        hint_task = asyncio.create_task(
            asyncio.to_thread(
                self.deps.hint_gen.generate,
                decision, ctx.match, ctx.reference, ctx.recognition, history, transcript,
            )
        )
        try:
            spoke_feedback = await self._react(verdict.feedback, decision)
            text = await hint_task
        except Exception:
            hint_task.cancel()
            raise
        if spoke_feedback:
            # one acknowledgement per turn: the reaction already was it
            text = strip_leading_acknowledgement(text)
        await self._deliver(decision, text, ctx.hash)

    async def _answer_question(
        self, ctx: ProblemContext, pending: HintRecord, question: str
    ) -> None:
        """The student asked why. Answer that, then hand the turn back.

        No store writes and no hint record: an explanation is not a hint, so
        it must not move the L1-L4 ladder or resolve the pending question.
        """
        text = await asyncio.to_thread(
            self.deps.hint_gen.explain,
            student_question=question,
            tutor_question=pending.hint_text,
            match=ctx.match,
            reference=ctx.reference,
            rec=ctx.recognition,
            target_step=pending.step,
        )
        log.info("explained a student question at step %d", pending.step)
        await self._speak(text)

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
        try:
            await self._speak(" ".join(part for part in (feedback, PROBLEM_DONE) if part))
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
        await self._speak(feedback)
        return True

    async def _handle_hint_request(self, question: str | None = None) -> None:
        """Capture → recognize → diagnose → hint, over a fresh photo.

        `question` is set when the student asked about their own work
        ("풀이 맞아?"). The pipeline is identical — that is the point:
        checking work IS re-reading the worksheet and re-diagnosing. It only
        changes what comes out: their question reaches the hint phrasing, and
        the tutor says what it saw before hinting.
        """
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
        elif self.ctx is None or not (
            self.ctx.hash == problem_hash(rec) or self._same_problem(rec)
        ):
            # First sight of a NEW problem: read it back while the tagger, the
            # matcher and the phraser think (~15s of otherwise dead air). The
            # student hears that the tutor actually saw their problem — and a
            # misread photo gets caught out loud, before any hint depends on it.
            # Three lines, queued in order: only the middle one costs TTS.
            for line in readout_of(rec):
                self._narrate(line)
        ctx = await self._problem_context(rec)

        # Diagnose before helping (spec rule 4). State/history are prefetched
        # here by the orchestrator — never fetched by the model itself.
        # History is always scoped to THIS problem's hash.
        prev_state = self.store.get_state()
        history = self.store.get_history(problem_hash=ctx.hash)
        reference = ctx.reference_if_ready()
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
        # an UNCERTAIN estimate carries no evidence either way: keep the hint
        # pending rather than resolving its effectiveness on a blurry frame.
        # A skipped estimate (reference still solving) carries none either.
        if (reference is not None and pending is not None and prev_state is not None
                and new_state.status != "UNCERTAIN"):
            self.store.mark_hint_effective(
                pending.id, hint_was_effective(prev_state, new_state)
            )

        # Always read state/history through the store right before the policy.
        current = self.store.get_state() or new_state

        if question and current.status == "CORRECT":
            # They asked whether their work is right, and it is. That question
            # deserves an answer, not a hint: hinting here would push them at a
            # step they have not reached and imply something was wrong.
            # Deliberately not _deliver(): no hint was given, so nothing should
            # enter the hint history or move the L1-L4 ladder.
            log.info("work check at step %d: correct so far", current.last_correct_step)
            await self._speak(WORK_CONFIRMED)
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

        # Same overlap as the answer turn: the reaction ("음, 지금 쓴 줄을 같이
        # 볼까요?") is fixed text with cached TTS, so it plays at once while the
        # hint is still being generated.
        hint_task = asyncio.create_task(
            asyncio.to_thread(
                self.deps.hint_gen.generate,
                decision, ctx.match, reference, rec, fresh_history, question,
            )
        )
        try:
            spoke_reaction = bool(question) and await self._react(
                self._work_reaction(current), decision
            )
            text = await hint_task
        except Exception:
            hint_task.cancel()
            raise
        if spoke_reaction:
            text = strip_leading_acknowledgement(text)
        await self._deliver(decision, text, ctx.hash)

    @staticmethod
    def _work_reaction(state: StudentState) -> str:
        """What the tutor says it saw, before it hints. Never what is wrong."""
        return WORK_CHECK_REACTIONS.get(state.status, WORK_CHECK_DEFAULT)

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
            return all(
                mathnorm.equations_equivalent(a, b, allow_scale=False)
                for a, b in zip(rec.equations, cached.equations)
            )
        return bool(rec.problem_text) and mathnorm.normalize_text(
            rec.problem_text
        ) == mathnorm.normalize_text(cached.problem_text)

    async def _problem_context(self, rec: Recognition) -> ProblemContext:
        h = problem_hash(rec)
        if self.ctx is not None and (self.ctx.hash == h or self._same_problem(rec)):
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

    async def _speak(self, text: str) -> None:
        """Say something to the student, after the filler has had its say.

        `text` stays ORIGINAL all the way to _say: the ear and the eye part
        ways there, and nowhere earlier — the hint history, the leak guard and
        the TTS cache keys all see what was actually generated.
        """
        await self._settle_filler()
        # TTS is part of the wait, and a cached phrase is not — worth telling apart.
        with timing.stage("speak"):
            await self._say(text)

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

    async def _say(self, text: str) -> None:
        """Say it on the machine running the server (same room as the student).

        BrowserSession overrides this to ship the audio to the device instead.
        The ear gets the spoken form here — "f 프라임 1" — and only the ear:
        nothing on this path is displayed.
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

    async def _deliver(self, decision: Decision, text: str, problem_hash: str = "") -> None:
        if text:
            await self._speak(text)
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
