"""Per-connection session orchestrator.

Owns ALL SessionStore writes. Two turns, deliberately asymmetric:

HINT REQUEST (the student wants help, their work is the evidence)
    capture → recognize → match (cached by problem_hash) → solver if needed →
    estimate → set_state + resolve pending hint effectiveness → prefetch
    state/history → policy.decide → generate hint (leak-guarded) → speak.

ANSWER (the student is replying to the question the tutor just asked)
    evaluate transcript against the reference solution → resolve the pending
    hint → prefetch state/history → policy.decide → speak.
    No capture, no VLM, no matching: the student spoke, they did not write, so
    re-reading the worksheet only adds latency and re-recognition noise.

The verdict feeds the unchanged policy rules through the store, which is what
produces the intended ladder: correct → next step L1, wrong → same step L2,
unclear → same step, same level, re-asked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from tutor.config import Settings
from tutor.hints.generator import HintGenerator, strip_leading_acknowledgement
from tutor.hints.guard import leaks_answer
from tutor.knowledge import mathnorm
from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.tagger import ConceptTagger
from tutor.knowledge.models import MatchResult, ReferenceSolution, Tier
from tutor.policy.engine import Action, Decision, Trigger, decide
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioFrame, ImageFrame, ProtocolError, decode
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import classify_transcript, wants_hint
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


@dataclass
class ProblemContext:
    hash: str
    recognition: Recognition
    match: MatchResult
    reference: ReferenceSolution


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
    tagger: ConceptTagger | None = None  # None → Recognition keeps "unknown"/[]
    cameras: object | None = None  # CameraHub: eyes on another socket (XIAO)
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
        self.last_transcript = text
        asks_hint = wants_hint(text)
        pending = (
            self.store.pending_hint(self.ctx.hash) if self.ctx is not None else None
        )
        # A question is on the table: whatever the student said is their answer
        # to it — including "모르겠어요", which the evaluator reads as "escalate".
        answering = (
            pending is not None
            and self.deps.evaluator is not None
            and bool(text.strip())
        )
        await self._send_transcript(text, answering or asks_hint)

        if answering:
            await self.handle_answer(text, pending)
        elif asks_hint or pending is not None:
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
        XIAO connected on /camera — a different socket entirely — is asked
        instead. Voice and vision can therefore live on different machines.
        """
        timeout = self.deps.settings.capture_timeout_s
        started = time.monotonic()
        jpeg = await self._capture_from_device()
        eye = "session device"
        if not jpeg:
            cameras = self.deps.cameras
            if not cameras:
                log.warning("no eye at all: the session device has no camera and none "
                            "is connected on /camera")
                return None
            log.info("no local camera; asking a connected camera device")
            started = time.monotonic()  # the local device's answer was not the wait
            jpeg = await cameras.capture(timeout)
            eye = "camera device"
        elapsed = (time.monotonic() - started) * 1000

        if jpeg:
            log.info("captured %d bytes from the %s in %.0f ms", len(jpeg), eye, elapsed)
            self._save_capture(jpeg)
        else:
            # The single most useful line in the log when the tutor keeps asking
            # to be shown the worksheet again: it says the VLM was never called.
            log.warning(
                "NO FRAME from the %s after %.0f ms (timeout %.0fs) — the worksheet was "
                "never seen, so recognition did not run. Raise CAPTURE_TIMEOUT_S, or check "
                "the board's serial output for the transfer time.",
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

    async def handle_hint_request(self) -> None:
        if self._busy:
            log.info("hint request ignored: already handling one")
            return
        self._busy = True
        try:
            await self._handle_hint_request()
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
            self._busy = False

    async def handle_answer(self, transcript: str, pending: HintRecord) -> None:
        """The student answered the tutor's question out loud."""
        if self._busy:
            log.info("answer ignored: a turn is already running")
            return
        self._busy = True
        try:
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
            self._busy = False

    async def _handle_answer(self, transcript: str, pending: HintRecord) -> None:
        ctx = self.ctx
        assert ctx is not None  # a pending hint implies a problem context

        verdict = await asyncio.to_thread(
            self.deps.evaluator.evaluate,
            problem_text=ctx.recognition.problem_text,
            reference=ctx.reference,
            question=pending.hint_text,
            target_step=pending.step,
            transcript=transcript,
        )
        self.last_transcript = None  # graded; not evidence for the next turn

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
            if pending.step >= len(ctx.reference.steps):
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

        text = await asyncio.to_thread(
            self.deps.hint_gen.generate,
            decision,
            ctx.match,
            ctx.reference,
            ctx.recognition,
            history,
            transcript,
        )
        await self._deliver(decision, self._with_feedback(verdict, text, decision), ctx.hash)

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
        if feedback and leaks_answer(feedback, ctx.reference, target_step):
            feedback = ""
        try:
            await self._speak(" ".join(part for part in (feedback, PROBLEM_DONE) if part))
        finally:
            log.info("problem %s solved; context closed", ctx.hash[:8])
            self.ctx = None
            self.prev_work = None
            self.last_transcript = None
            self.store.clear_state()

    def _with_feedback(self, verdict, hint: str, decision: Decision) -> str:
        """Prefix the tutor's reaction to the answer, if it leaks nothing."""
        feedback = (verdict.feedback or "").strip()
        if not feedback or not hint:
            return feedback or hint
        # The feedback IS this turn's reaction; a hint that opens with its own
        # "네," makes the tutor say it twice in one breath.
        combined = f"{feedback} {strip_leading_acknowledgement(hint)}"
        reference = self.ctx.reference if self.ctx is not None else None
        if reference is not None and leaks_answer(combined, reference, decision.target_step):
            log.warning("answer feedback leaked; dropping it")
            return hint
        return combined

    async def _handle_hint_request(self) -> None:
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
        ctx = await self._problem_context(rec)

        # Diagnose before helping (spec rule 4). State/history are prefetched
        # here by the orchestrator — never fetched by the model itself.
        # History is always scoped to THIS problem's hash.
        prev_state = self.store.get_state()
        history = self.store.get_history(problem_hash=ctx.hash)
        new_state = await asyncio.to_thread(
            self.deps.estimator.estimate,
            rec=rec,
            reference=ctx.reference,
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
        # pending rather than resolving its effectiveness on a blurry frame
        if pending is not None and prev_state is not None and new_state.status != "UNCERTAIN":
            self.store.mark_hint_effective(
                pending.id, hint_was_effective(prev_state, new_state)
            )

        # Always read state/history through the store right before the policy.
        current = self.store.get_state() or new_state
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

        text = await asyncio.to_thread(
            self.deps.hint_gen.generate, decision, ctx.match, ctx.reference, rec, fresh_history
        )
        await self._deliver(decision, text, ctx.hash)

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
            return self.ctx
        if self.ctx is not None:
            # Different problem: the old student state/work/transcript are
            # meaningless here — start fresh (history stays, scoped by hash).
            log.info("problem changed (%s -> %s): resetting state", self.ctx.hash[:8], h[:8])
            self.store.clear_state()
            self.prev_work = None
            self.last_transcript = None
        # A new problem: classify it once, then everything downstream (matching,
        # RAG, hint phrasing) reads the tags off the Recognition.
        if self.deps.tagger is not None:
            tags = await asyncio.to_thread(self.deps.tagger.tag, rec)
            rec.problem_type = tags.problem_type
            rec.concepts = tags.concepts
        match = await asyncio.to_thread(self.deps.matcher.match, rec)
        reference = match.reference
        if reference is None:  # CONCEPT/NEW → Grok Solver (spec rule 2)
            reference = await asyncio.to_thread(self.deps.solver.solve, rec, h)
        self.ctx = ProblemContext(hash=h, recognition=rec, match=match, reference=reference)
        return self.ctx

    async def _speak(self, text: str) -> None:
        """Say it on the machine running the server (XIAO setup: same room).

        BrowserSession overrides this to ship the audio to the device instead.
        """
        await self.ws.send(make_event("speech_state", {"state": "speaking"}))
        try:
            await asyncio.to_thread(self.deps.speaker.speak, text)
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
