"""Browser as the device: mic PCM in, tutor audio out, over one WebSocket.

For an SSH deployment the laptop only does capture and playback; everything
else stays on the server:

    browser mic → 16 kHz mono PCM (AUDIO frames) → Silero VAD here
      → endpointed utterance → the unchanged Session pipeline
      → TTS bytes (TTS_AUDIO frame) → browser speaker → playback_done

Which is the same conversation as simulator/voice_device.py with the VAD moved
across the socket: TurnDetector/TurnTaker are reused verbatim, so prefix
padding, onset debounce, endpointing and the four states behave identically.

Barge-in lives here. While the tutor speaks, the mic stays open on both sides
and the VAD keeps judging at a higher bar; when the student cuts in, three
things happen in this order and the order is the point:

    1. the browser is told to stop and empty its audio queue — first, because
       every millisecond after this is the tutor talking over a student
    2. the utterance in flight is abandoned: no more of it is spoken, and the
       rest of that turn says nothing
    3. the interrupting words — which began before we noticed them, and are
       already in the detector's prefix buffer — become the next turn

What does NOT happen is a rollback. A hint that was half spoken was still
given: it stays in the history, so the policy does not offer it again as if
the student had never heard it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from tutor.llm import timing
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioFrame, TtsAudioHeader, encode_tts_audio
from tutor.server.session import Deps, Session
from tutor.speech import mathspeak
from tutor.speech.turn import TurnConfig, TurnDetector, TurnState, TurnTaker

log = logging.getLogger(__name__)


class BrowserSession(Session):
    def __init__(self, ws, deps: Deps, vad=None, playback_timeout_s: float = 120.0):
        super().__init__(ws, deps)
        self.config = TurnConfig.from_settings(deps.settings)
        self.playback_timeout_s = playback_timeout_s
        self._vad = vad  # tests inject a scripted one; None → real Silero
        self.taker: TurnTaker | None = None
        self._pcm = b""  # leftover between AUDIO frames: the VAD needs exact frames
        self._playback: asyncio.Future | None = None
        self._sent_state: TurnState | None = None
        self._utterances = 0
        self._speech_seq = 0  # one turn says several lines; each gets its own stream id
        # Set when the student cuts in; every _say for the rest of that turn
        # returns without speaking. Cleared when the next turn starts.
        self._interrupted = False

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0

    async def _ensure_taker(self) -> TurnTaker:
        if self.taker is None:
            # loading Silero takes a moment: keep it off the event loop
            detector = await asyncio.to_thread(TurnDetector, self._vad, self.config)
            self.taker = TurnTaker(detector, on_barge_in=self._barged_in)
            log.info(
                "server-side VAD ready (barge-in %s)",
                "on" if self.config.barge_in else "off",
            )
        return self.taker

    # --- barge-in ------------------------------------------------------------

    def _barged_in(self) -> None:
        """The student took the floor. Called from inside the VAD, so it must
        not await — it schedules the stop and returns."""
        if self._interrupted:
            return
        self._interrupted = True
        log.info("barge-in: stopping playback and abandoning the rest of the turn")
        # Wake whoever is waiting on playback_done. The browser will never send
        # it now — it is being told to throw that audio away.
        if self._playback is not None and not self._playback.done():
            self._playback.set_result(False)
        self._spawn(self._stop_playback())

    async def _wait_for_the_floor(self, timeout_s: float = 3.0) -> None:
        """Let the abandoned turn finish unwinding before starting the new one.

        Without this the interruption is self-defeating: the old turn is still
        marked busy while it walks out of _deliver, the question the student
        just interrupted with arrives, and Session drops it as "already
        handling one" — so cutting in would lose the very thing it was for.
        It is a short wait because nothing slow is left; the speaking already
        stopped.
        """
        deadline = time.monotonic() + timeout_s
        while self._busy and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        if self._busy:
            log.warning("previous turn is still running after %.0fs", timeout_s)

    async def _stop_playback(self) -> None:
        try:
            await self.ws.send(make_event("barge_in", {}))
            await self.ws.send(make_event("speech_state", {"state": "idle"}))
        except Exception:
            log.debug("could not tell the browser to stop (connection gone)")

    async def _push_state(self) -> None:
        """Mirror the turn state to the browser (UI + its own mic gate)."""
        if self.taker is None or self.taker.state is self._sent_state:
            return
        self._sent_state = self.taker.state
        try:
            await self.ws.send(
                make_event("turn_state", {"state": self.taker.state.value})
            )
        except Exception:
            log.debug("could not send turn_state (connection gone)")

    # --- device events ------------------------------------------------------

    async def _on_event(self, raw: str) -> None:
        ev = parse_event(raw)
        if ev.event == "playback_done":
            if self._playback is not None and not self._playback.done():
                self._playback.set_result(True)
            return
        if ev.event == "hello":
            await self._ensure_taker()
            await super()._on_event(raw)
            # The browser gates its own mic too, and needs to know whether it
            # may keep it open while the tutor talks — and whether it is an eye
            # at all: with a phone on /camera the laptop never supplies a
            # photo, so offering an upload box would be offering a dead end.
            await self.ws.send(make_event("config", {
                "barge_in": self.config.barge_in,
                "input_mode": self.deps.settings.input_mode,
            }))
            await self._push_state()
            return
        await super()._on_event(raw)

    # --- mic audio: VAD here instead of on the device -----------------------

    async def _on_audio(self, frame: AudioFrame) -> None:
        taker = await self._ensure_taker()
        if frame.header.sample_rate != self.config.sample_rate:
            raise ValueError(
                f"browser must send {self.config.sample_rate} Hz mono PCM, "
                f"got {frame.header.sample_rate}"
            )

        self._pcm += frame.pcm
        size = self.config.frame_bytes
        while len(self._pcm) >= size:
            chunk, self._pcm = self._pcm[:size], self._pcm[size:]
            utterance = taker.feed(chunk, now_ms=self._now_ms())
            if utterance is not None:
                self._utterances += 1
                log.info(
                    "utterance #%d endpointed: %.1fs of PCM → pipeline",
                    self._utterances,
                    len(utterance) / 2 / self.config.sample_rate,
                )
                # off the receive loop: the pipeline awaits capture frames and
                # playback_done, which arrive through this very loop
                self._spawn(self._handle_utterance(utterance, frame.header.sample_rate))
        await self._push_state()

    async def _handle_utterance(self, pcm: bytes, sample_rate: int) -> None:
        # A fresh turn: the tutor may speak again. Cleared here rather than at
        # the barge-in, so everything still unwinding from the abandoned turn
        # stays silent until this one actually begins.
        self._interrupted = False
        await self._wait_for_the_floor()
        try:
            await super()._handle_utterance(pcm, sample_rate)
        finally:
            await self._reopen_mic()

    async def handle_hint_request(self, question: str | None = None) -> None:
        # Mid-turn utterances (the reaction, filler lines) hold PROCESSING, so
        # every turn entry owes the mic a reopen when it is over — including
        # the button-driven hint_request that never passes through an utterance.
        try:
            await super().handle_hint_request(question)
        finally:
            await self._reopen_mic()

    async def _reopen_mic(self) -> None:
        """The floor back to the student, if the turn left it in PROCESSING —
        which also covers a turn that spoke nothing at all (no hint wanted,
        STT failed, a hint already running): silence must not mean deaf."""
        if self.taker is not None and self.taker.state is TurnState.PROCESSING:
            self.taker.listen()
            await self._push_state()

    # --- tutor speech: to the browser, never to the server's speaker --------

    def _pump_tts(self, spoken: str) -> tuple[asyncio.Queue, "threading.Event", asyncio.Task]:
        """Run the blocking TTS stream in a thread, chunks into an async queue.

        None on the queue means the stream ended. The event asks the thread to
        stop pulling — httpx closes the response when the generator is dropped,
        so a barge-in stops costing bandwidth as well as attention.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def pump() -> None:
            try:
                for chunk in self.deps.speaker.synthesize_stream(spoken):
                    if stop.is_set():
                        break
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception:
                log.exception("TTS stream failed; whatever arrived still plays")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        return queue, stop, asyncio.create_task(asyncio.to_thread(pump))

    async def _say(self, text: str) -> bool:
        """Returns whether the student heard any of this line — see _speak."""
        # a hint_request can arrive before any audio (button-driven client)
        taker = await self._ensure_taker()
        if self._interrupted:
            # Already cut off in this turn. Whatever else it was going to say —
            # the closing praise, a second sentence — is not said over someone.
            log.info("skipping speech after a barge-in: %r", text[:40])
            return False
        # The ear and the eye part ways HERE: the TTS gets "f 프라임 1", the
        # transcript panel gets f'(1) — as 2·x², not as programmer ASCII.
        started = time.monotonic()
        queue, stop, pump_task = self._pump_tts(mathspeak.speakable(text))
        first = await queue.get()
        if self._interrupted:  # interrupted while the first chunk was rendering
            stop.set()
            await pump_task
            return False
        await self.ws.send(make_event("tutor_says", {"text": mathspeak.displayable(text)}))

        if first is None:  # echo mode / no TTS configured: text only, keep talking
            await pump_task
            taker.listen()
            await self._push_state()
            return

        taker.agent_speaking()
        await self._push_state()
        await self.ws.send(make_event("speech_state", {"state": "speaking"}))
        self._playback = asyncio.get_running_loop().create_future()
        # unique per SPOKEN LINE, not per student utterance: one turn now says
        # several (echo, reaction, hint), and the page groups frames by this id
        self._speech_seq += 1
        utterance_id = f"tts-{self._utterances}-{self._speech_seq}"
        audio_format = getattr(self.deps.speaker, "audio_format", "mp3")
        try:
            seq, held = 0, first
            while True:
                nxt = await queue.get()
                await self.ws.send(
                    encode_tts_audio(
                        held,
                        TtsAudioHeader(
                            utterance_id=utterance_id, format=audio_format,
                            seq=seq, last=nxt is None,
                        ),
                    )
                )
                if seq == 0:
                    # the number this whole streaming path exists to shrink:
                    # how long the student waited before the voice began
                    timing.record("speak.first", time.monotonic() - started, "first chunk")
                seq += 1
                if nxt is None or self._interrupted:
                    stop.set()
                    break
                held = nxt
            if not self._interrupted:
                await asyncio.wait_for(self._playback, timeout=self.playback_timeout_s)
        except asyncio.TimeoutError:
            log.warning("no playback_done in %.0fs; resuming anyway", self.playback_timeout_s)
        finally:
            await pump_task
            self._playback = None
            if self._interrupted:
                # The student is already speaking. Sending idle would be a lie
                # and reopening the mic would reset the turn they have started.
                await self._push_state()
                return
            try:
                await self.ws.send(make_event("speech_state", {"state": "idle"}))
            except Exception:
                log.debug("could not send speech_state idle (connection gone)")
            # With barge-in off the browser gates its own mic, so no frame will
            # arrive to expire the tail guard lazily (as it does on the
            # local-mic device): wait it out here, then reopen.
            await asyncio.sleep(self.config.tail_guard_ms / 1000)
            if self._busy:
                # Mid-turn utterance — a filler line, the reaction spoken while
                # the hint generates. The turn is not over, so the floor is not
                # the student's yet: stay in PROCESSING and let the turn's own
                # end reopen the mic. Reopening here dropped whatever they said
                # into the busy check, which is worse than a closed mic.
                taker.processing()
            else:
                taker.listen()
            await self._push_state()
