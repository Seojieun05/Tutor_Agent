"""Per-connection session orchestrator.

Owns ALL SessionStore writes. Flow on a hint request (spec rules 4-6):
capture → recognize → match (cache by problem_hash) → solver if needed →
estimate → set_state + resolve pending hint effectiveness → prefetch
state/history → policy.decide → generate hint (leak-guarded) → speak.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import MatchResult, ReferenceSolution, Tier
from tutor.policy.engine import Action, Decision, Trigger, decide
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioFrame, ImageFrame, ProtocolError, decode
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import wants_hint
from tutor.state.estimator import StudentStateEstimator, hint_was_effective
from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognition, Recognizer

log = logging.getLogger(__name__)


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
        except Exception:
            log.exception("STT failed; utterance dropped")
            return
        log.info("utterance: %r", transcript.text)
        self.last_transcript = transcript.text
        if wants_hint(transcript.text):
            await self.handle_hint_request()

    # --- capture --------------------------------------------------------------

    async def _request_capture(self) -> bytes | None:
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

    async def _handle_hint_request(self) -> None:
        jpeg = await self._request_capture()
        state = self.store.get_state() or StudentState()
        if jpeg is None:
            decision = decide(state, self.store.get_history(), "RECOGNITION_FAILED")
            await self._deliver(decision, self.deps.hint_gen.generate(
                decision, MatchResult(tier=Tier.NEW), None, Recognition(problem_text=""), []
            ))
            return

        rec = await asyncio.to_thread(self.deps.recognizer.recognize, jpeg)
        ctx = await self._problem_context(rec)

        # Diagnose before helping (spec rule 4). State/history are prefetched
        # here by the orchestrator — never fetched by the model itself.
        prev_state = self.store.get_state()
        history = self.store.get_history()
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

        # Orchestrator-owned writes: state, then pending-hint effectiveness.
        pending = self.store.pending_hint()
        self.store.set_state(new_state)
        # an UNCERTAIN estimate carries no evidence either way: keep the hint
        # pending rather than resolving its effectiveness on a blurry frame
        if pending is not None and prev_state is not None and new_state.status != "UNCERTAIN":
            self.store.mark_hint_effective(
                pending.id, hint_was_effective(prev_state, new_state)
            )

        # Always read state/history through the store right before the policy.
        current = self.store.get_state() or new_state
        fresh_history = self.store.get_history()
        decision = decide(current, fresh_history, "HINT_REQUEST")
        log.info("decision: %s", decision)

        text = await asyncio.to_thread(
            self.deps.hint_gen.generate, decision, ctx.match, ctx.reference, rec, fresh_history
        )
        await self._deliver(decision, text)

    async def _problem_context(self, rec: Recognition) -> ProblemContext:
        h = problem_hash(rec)
        if self.ctx is not None and self.ctx.hash == h:
            # Cached problem: reuse match/reference; student work stays fresh.
            self.ctx.recognition = rec
            return self.ctx
        match = await asyncio.to_thread(self.deps.matcher.match, rec)
        reference = match.reference
        if reference is None:  # CONCEPT/NEW → Grok Solver (spec rule 2)
            reference = await asyncio.to_thread(self.deps.solver.solve, rec, h)
        self.ctx = ProblemContext(hash=h, recognition=rec, match=match, reference=reference)
        return self.ctx

    async def _deliver(self, decision: Decision, text: str) -> None:
        if text:
            await self.ws.send(make_event("speech_state", {"state": "speaking"}))
            try:
                await asyncio.to_thread(self.deps.speaker.speak, text)
            finally:
                # whatever happens to TTS, never leave the device muted
                try:
                    await self.ws.send(make_event("speech_state", {"state": "idle"}))
                except Exception:
                    pass
        self.store.append_hint(
            step=decision.target_step,
            level=decision.level,
            action=decision.action.value,
            hint_text=text,
        )
        await self.ws.send(
            make_event("hint_issued", {"level": decision.level, "action": decision.action.value})
        )
