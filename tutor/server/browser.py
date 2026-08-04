"""Browser as the device: mic PCM in, tutor audio out, over one WebSocket.

For an SSH deployment the laptop only does capture and playback; everything
else stays on the server:

    browser mic → 16 kHz mono PCM (AUDIO frames) → Silero VAD here
      → endpointed utterance → the unchanged Session pipeline
      → TTS bytes (TTS_AUDIO frame) → browser speaker → playback_done

Which is the same conversation as simulator/voice_device.py with the VAD moved
across the socket: TurnDetector/TurnTaker are reused verbatim, so prefix
padding, onset debounce, endpointing and the four states behave identically.
The mic is gated for the whole PROCESSING+AGENT_SPEAKING span (browser-side
too), so barge-in is out of scope by construction.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioFrame, TtsAudioHeader, encode_tts_audio
from tutor.server.session import Deps, Session
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

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0

    async def _ensure_taker(self) -> TurnTaker:
        if self.taker is None:
            # loading Silero takes a moment: keep it off the event loop
            detector = await asyncio.to_thread(TurnDetector, self._vad, self.config)
            self.taker = TurnTaker(detector)
            log.info("server-side VAD ready for browser session")
        return self.taker

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
        try:
            await super()._handle_utterance(pcm, sample_rate)
        finally:
            # nothing was spoken (no hint wanted, STT failed, or a hint was
            # already running): reopen the mic instead of staying deaf
            if self.taker is not None and self.taker.state is TurnState.PROCESSING:
                self.taker.listen()
                await self._push_state()

    # --- tutor speech: to the browser, never to the server's speaker --------

    async def _speak(self, text: str) -> None:
        # a hint_request can arrive before any audio (button-driven client)
        taker = await self._ensure_taker()
        audio = await asyncio.to_thread(self.deps.speaker.synthesize, text)
        await self.ws.send(make_event("tutor_says", {"text": text}))

        if not audio:  # echo mode / no TTS configured: text only, keep talking
            taker.listen()
            await self._push_state()
            return

        taker.agent_speaking()
        await self._push_state()
        await self.ws.send(make_event("speech_state", {"state": "speaking"}))
        self._playback = asyncio.get_running_loop().create_future()
        try:
            await self.ws.send(
                encode_tts_audio(
                    audio,
                    TtsAudioHeader(
                        utterance_id=f"tts-{self._utterances}",
                        format=getattr(self.deps.speaker, "audio_format", "mp3"),
                    ),
                )
            )
            await asyncio.wait_for(self._playback, timeout=self.playback_timeout_s)
        except asyncio.TimeoutError:
            log.warning("no playback_done in %.0fs; resuming anyway", self.playback_timeout_s)
        finally:
            self._playback = None
            try:
                await self.ws.send(make_event("speech_state", {"state": "idle"}))
            except Exception:
                log.debug("could not send speech_state idle (connection gone)")
            # The browser gates its own mic, so no frame will arrive to expire
            # the tail guard lazily (as it does on the local-mic device): wait
            # it out here, then reopen — the browser starts sending again when
            # it sees LISTENING.
            await asyncio.sleep(self.config.tail_guard_ms / 1000)
            taker.listen()
            await self._push_state()
