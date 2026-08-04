"""Hands-free turn taking: endpointing + the four-state conversation gate.

``TurnDetector`` follows the week-3 course structure (prefix padding, debounced
onset, silence endpointing, min-speech discard) with Silero VAD in place of
webrtcvad, and counts milliseconds instead of frames so the tuning knobs stay
readable at Silero's 32 ms frame size.

``TurnTaker`` wraps it in the conversation state machine::

    LISTENING --speech onset--> USER_SPEAKING --endpoint--> PROCESSING
        ^                                                       |
        |                                                    TTS starts
        +-------- tail guard <-- AGENT_SPEAKING <----------------+

VAD runs only in LISTENING/USER_SPEAKING: while the tutor speaks, frames are
dropped without ever reaching the model, so the agent cannot hear itself.

Both classes are pure logic — the VAD is injected, so the state machine is
testable with a scripted stub and no audio hardware.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from tutor.speech.vad import frame_samples

log = logging.getLogger(__name__)


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int | None = None) -> bool: ...


@dataclass
class TurnConfig:
    sample_rate: int = 16000
    prefix_ms: int = 300  # audio kept from before onset, so no clipped first syllable
    min_speech_ms: int = 250  # shorter than this is a cough, not a turn
    silence_ms: int = 800  # trailing silence that ends the turn
    threshold: float = 0.5  # Silero speech probability
    tail_guard_ms: int = 250  # extra mute after TTS, for speaker/room decay
    # Onset debounce: SPEAKING only after N speech frames within the last W.
    onset_frames: int = 3
    onset_window: int = 5
    max_utterance_ms: int = 30_000  # force-commit guard for a turn that never ends

    @property
    def frame_samples(self) -> int:
        return frame_samples(self.sample_rate)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.sample_rate

    @property
    def prefix_frames(self) -> int:
        # round up: the padding must be at least prefix_ms, never less
        return max(1, math.ceil(self.prefix_ms / self.frame_ms))

    @classmethod
    def from_settings(cls, settings) -> "TurnConfig":
        return cls(
            sample_rate=settings.audio_sample_rate,
            prefix_ms=settings.vad_prefix_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            silence_ms=settings.vad_silence_ms,
            threshold=settings.vad_threshold,
            tail_guard_ms=settings.vad_tail_guard_ms,
        )


class TurnDetector:
    """Feed fixed-size PCM frames, get a complete utterance back when the
    speaker stops. Returns ``None`` for every frame that is not an endpoint."""

    def __init__(
        self,
        vad: VoiceActivityDetector | None = None,
        config: TurnConfig | None = None,
    ):
        self.config = config or TurnConfig()
        if vad is None:
            from tutor.speech.vad import SileroVAD

            vad = SileroVAD(self.config.threshold, self.config.sample_rate)
        self.vad = vad
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._frames: list[bytes] = []
        self._prefix: deque[bytes] = deque(maxlen=self.config.prefix_frames)
        self._onset: deque[bool] = deque(maxlen=self.config.onset_window)
        self._quiet_ms = 0.0
        self._speech_ms = 0.0
        reset = getattr(self.vad, "reset", None)
        if reset is not None:
            reset()  # Silero is stateful: a new turn starts from a clean RNN

    def feed(self, frame: bytes) -> bytes | None:
        cfg = self.config
        if len(frame) != cfg.frame_bytes:
            raise ValueError(f"expected {cfg.frame_bytes}-byte frames, got {len(frame)}")
        is_speech = self.vad.is_speech(frame, cfg.sample_rate)

        if not self.speaking:
            # LISTENING: keep recent audio (prefix padding) and debounce onset,
            # so a single noisy frame cannot open a turn.
            self._prefix.append(frame)
            self._onset.append(is_speech)
            if sum(self._onset) >= cfg.onset_frames:
                self.speaking = True
                self._frames = list(self._prefix)  # the first syllables survive
                self._speech_ms = sum(self._onset) * cfg.frame_ms
                self._quiet_ms = 0.0
                log.info("[SPEECH START]")
            return None

        # USER_SPEAKING: keep everything (pauses are part of the utterance),
        # count trailing silence, commit at silence_ms.
        self._frames.append(frame)
        if is_speech:
            self._speech_ms += cfg.frame_ms
            self._quiet_ms = 0.0
        else:
            self._quiet_ms += cfg.frame_ms

        duration_ms = len(self._frames) * cfg.frame_ms
        if self._quiet_ms < cfg.silence_ms and duration_ms < cfg.max_utterance_ms:
            return None
        if duration_ms >= cfg.max_utterance_ms:
            log.warning("[SPEECH END] max utterance length reached; committing")

        return self._commit()

    def _commit(self) -> bytes | None:
        utterance = b"".join(self._frames)
        speech_ms = self._speech_ms
        duration_s = len(self._frames) * self.config.frame_ms / 1000
        self.reset()

        if speech_ms < self.config.min_speech_ms:
            log.info("[DISCARDED] only %d ms of speech — not a turn", speech_ms)
            return None
        log.info("[SPEECH END after %.1fs -> pipeline]", duration_s)
        return utterance


class TurnState(str, Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AGENT_SPEAKING = "AGENT_SPEAKING"


class TurnTaker:
    """The conversation gate around a TurnDetector.

    The audio side is driven by ``feed()``; the pipeline side by
    ``agent_speaking()`` / ``agent_finished()`` / ``listen()``, which the caller
    wires to whatever its transport reports (here: the server's ``speech_state``
    and ``hint_issued`` events).
    """

    def __init__(
        self,
        detector: TurnDetector,
        on_change: Callable[[TurnState], None] | None = None,
    ):
        self.detector = detector
        self.config = detector.config
        self._on_change = on_change
        self._state = TurnState.LISTENING
        self._resume_at_ms: float | None = None

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def listening(self) -> bool:
        """True while microphone audio is being judged by the VAD."""
        return self._state in (TurnState.LISTENING, TurnState.USER_SPEAKING)

    def _set(self, state: TurnState) -> None:
        if state is self._state:
            return
        self._state = state
        log.info("[state] %s", state.value)
        if self._on_change is not None:
            self._on_change(state)

    def feed(self, frame: bytes, now_ms: float | None = None) -> bytes | None:
        """Returns a finished utterance, or None.

        Frames arriving while the tutor is thinking or speaking are dropped
        here — the VAD never sees them, so the agent cannot trigger on its own
        voice, and half-turns cannot survive across a response.
        """
        if now_ms is not None:
            self._maybe_resume(now_ms)
        if not self.listening:
            return None

        utterance = self.detector.feed(frame)
        if utterance is not None:
            self._set(TurnState.PROCESSING)
            return utterance
        self._set(
            TurnState.USER_SPEAKING if self.detector.speaking else TurnState.LISTENING
        )
        return None

    def processing(self) -> None:
        """The utterance is in the pipeline (STT → tutor → TTS)."""
        self._resume_at_ms = None
        self.detector.reset()
        self._set(TurnState.PROCESSING)

    def agent_speaking(self) -> None:
        """TTS started: mute the VAD until the tail guard expires."""
        self._resume_at_ms = None
        self.detector.reset()
        self._set(TurnState.AGENT_SPEAKING)

    def agent_finished(self, now_ms: float) -> None:
        """TTS finished: stay muted for tail_guard_ms, then listen again.

        The guard covers speaker/room decay, so the last word of the hint does
        not open a new user turn.
        """
        if self._state is not TurnState.AGENT_SPEAKING:
            return
        self._resume_at_ms = now_ms + self.config.tail_guard_ms

    def listen(self) -> None:
        """Back to LISTENING now (turn finished without speech, or aborted)."""
        self._resume_at_ms = None
        self.detector.reset()
        self._set(TurnState.LISTENING)

    def _maybe_resume(self, now_ms: float) -> None:
        if self._resume_at_ms is not None and now_ms >= self._resume_at_ms:
            self.listen()
