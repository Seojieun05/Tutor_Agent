# [한국어 요약] 핸즈프리 턴 전환의 심장.
# TurnDetector는 PCM 프레임을 받아 "말이 시작됐고 끝났다"를 판정해 발화 하나를 돌려주고,
# TurnTaker는 지금이 누구 차례인지(듣는 중 / 학생 말하는 중 / 생각 중 / 튜터 말하는 중)를 관리한다.
# 브라우저 클라이언트와 로컬 마이크 디바이스가 이 구현 하나를 함께 쓴다.

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from tutor.speech.vad import frame_samples

log = logging.getLogger(__name__)


# 음성 검출기 인터페이스(프레임 하나가 말인지 아닌지).
class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int | None = None) -> bool: ...


# 턴 판정 파라미터: 앞머리 여유·최소 발화·끝맺음 침묵·문턱값·꼬리 보호·바지인 기준.
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
    # Barge-in: keep listening while the tutor talks, so a student can cut in.
    # The bar is higher than a normal onset for a physical reason — the mic can
    # hear the speaker. Browser echo cancellation removes most of it, but not
    # all, so a cough or a leaked syllable must not take the floor: barge_in
    # asks for sustained speech, roughly half a second of it.
    barge_in: bool = True
    barge_in_frames: int = 8
    barge_in_window: int = 12

    # 프레임 하나의 샘플 수.
    @property
    def frame_samples(self) -> int:
        return frame_samples(self.sample_rate)

    # 프레임 하나의 바이트 수(16bit).
    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    # 프레임 하나의 길이(ms).
    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_samples / self.sample_rate

    # 말 시작 전에 남겨 둘 프레임 수(첫 음절이 잘리지 않게).
    @property
    def prefix_frames(self) -> int:
        # round up: the padding must be at least prefix_ms, never less
        return max(1, math.ceil(self.prefix_ms / self.frame_ms))

    # .env 설정에서 턴 파라미터를 만든다.
    @classmethod
    def from_settings(cls, settings) -> "TurnConfig":
        return cls(
            sample_rate=settings.audio_sample_rate,
            prefix_ms=settings.vad_prefix_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            silence_ms=settings.vad_silence_ms,
            threshold=settings.vad_threshold,
            tail_guard_ms=settings.vad_tail_guard_ms,
            barge_in=settings.barge_in,
            barge_in_frames=settings.barge_in_frames,
        )


# 프레임을 계속 먹이면, 말이 끝나는 순간 발화 하나를 통째로 돌려주는 검출기.
class TurnDetector:
    """Feed fixed-size PCM frames, get a complete utterance back when the
    speaker stops. Returns ``None`` for every frame that is not an endpoint."""

    # VAD와 설정을 받는다. 튜터가 말하는 동안에는 문턱을 높일 수 있게 onset 기준을 따로 둔다.
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
        # How hard it is to open a turn right now. None = the usual bar; the
        # taker raises it while the tutor is speaking. A mode, not state, so it
        # deliberately survives reset().
        self.onset_required: int | None = None
        self.onset_window: int | None = None
        self.reset()

    # 다음 발화를 위해 버퍼와 VAD 상태를 비운다.
    def reset(self) -> None:
        self.speaking = False
        self._frames: list[bytes] = []
        self._prefix: deque[bytes] = deque(maxlen=self.config.prefix_frames)
        self._onset: deque[bool] = deque(
            maxlen=self.onset_window or self.config.onset_window
        )
        self._quiet_ms = 0.0
        self._speech_ms = 0.0
        reset = getattr(self.vad, "reset", None)
        if reset is not None:
            reset()  # Silero is stateful: a new turn starts from a clean RNN

    # 프레임 한 개 처리. 끝점이면 완성된 발화를, 아니면 None을 돌려준다.
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
            if sum(self._onset) >= (self.onset_required or cfg.onset_frames):
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

    # 모아 둔 프레임을 발화 하나로 확정한다(너무 짧으면 버린다).
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


# 지금 누구 차례인지: 듣는 중 / 학생 말하는 중 / 생각 중 / 튜터 말하는 중.
class TurnState(str, Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AGENT_SPEAKING = "AGENT_SPEAKING"


# 상태 전이를 관리한다. 바지인(학생이 말을 끊고 들어오기)도 여기서 판정한다.
class TurnTaker:
    # 검출기와, 상태 변화·바지인 콜백을 받는다.
    def __init__(
        self,
        detector: TurnDetector,
        on_change: Callable[[TurnState], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ):
        self.detector = detector
        self.config = detector.config
        self._on_change = on_change
        # Called the moment the student takes the floor back, synchronously,
        # from inside feed(). Everything that has to stop — the audio in the
        # browser, the utterance still being spoken — stops from here.
        self._on_barge_in = on_barge_in
        self._state = TurnState.LISTENING
        self._resume_at_ms: float | None = None

    # 현재 상태.
    @property
    def state(self) -> TurnState:
        return self._state

    # 지금 마이크 소리를 판정하고 있는지.
    @property
    def listening(self) -> bool:
        """True while microphone audio is being judged by the VAD."""
        return self._state in (TurnState.LISTENING, TurnState.USER_SPEAKING)

    # 상태를 바꾸고 로그·콜백을 알린다.
    def _set(self, state: TurnState) -> None:
        if state is self._state:
            return
        self._state = state
        log.info("[state] %s", state.value)
        if self._on_change is not None:
            self._on_change(state)

    # 프레임을 상태에 맞게 처리한다. 생각 중(PROCESSING)일 때는 일부러 귀를 닫는다.
    def feed(self, frame: bytes, now_ms: float | None = None) -> bytes | None:
        """Returns a finished utterance, or None.

        Frames arriving while the tutor is THINKING are still dropped: there is
        nothing to interrupt yet, and a half-turn must not survive across a
        response. Frames arriving while the tutor is SPEAKING are judged, at a
        higher bar — that is barge-in.
        """
        if now_ms is not None:
            self._maybe_resume(now_ms)
        if self._state is TurnState.AGENT_SPEAKING and self.config.barge_in:
            return self._feed_over_the_tutor(frame)
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

    # 튜터가 말하는 동안의 처리: 더 높은 문턱을 넘는 지속된 말소리만 바지인으로 인정한다.
    def _feed_over_the_tutor(self, frame: bytes) -> bytes | None:
        """Listen while the tutor talks, and hand the floor over if cut off.

        The detector is the ordinary one, with the onset bar raised — which
        means the interrupting words are already in its prefix buffer when it
        fires. That matters: the student's question does not start at the
        moment we notice it, and a barge-in that swallows "왜" and hears
        "그렇게 해요?" is worse than no barge-in.
        """
        utterance = self.detector.feed(frame)
        if not self.detector.speaking:
            return utterance  # None, unless a short burst just committed
        log.info("[BARGE-IN] the student took the floor")
        self.detector.onset_required = None
        self.detector.onset_window = None
        self._resume_at_ms = None
        self._set(TurnState.USER_SPEAKING)
        if self._on_barge_in is not None:
            self._on_barge_in()
        return utterance

    # 발화가 파이프라인으로 넘어갔다(STT → 튜터 → TTS).
    def processing(self) -> None:
        """The utterance is in the pipeline (STT → tutor → TTS)."""
        self._resume_at_ms = None
        self._normal_onset()
        self._set(TurnState.PROCESSING)

    # TTS 시작. 바지인이 켜져 있으면 더 높은 문턱으로 계속 듣는다.
    def agent_speaking(self) -> None:
        """TTS started. With barge-in on, keep listening at a higher bar; with
        it off, this is a mute until the tail guard expires."""
        self._resume_at_ms = None
        self.detector.onset_required = self.config.barge_in_frames
        self.detector.onset_window = self.config.barge_in_window
        self.detector.reset()
        self._set(TurnState.AGENT_SPEAKING)

    # TTS 끝. 방 울림이 새 발화로 잡히지 않게 꼬리 보호 시간만큼 더 기다린다.
    def agent_finished(self, now_ms: float) -> None:
        """TTS finished: stay muted for tail_guard_ms, then listen again.

        The guard covers speaker/room decay, so the last word of the hint does
        not open a new user turn.
        """
        if self._state is not TurnState.AGENT_SPEAKING:
            return
        self._resume_at_ms = now_ms + self.config.tail_guard_ms

    # 다시 듣기 상태로.
    def listen(self) -> None:
        """Back to LISTENING now (turn finished without speech, or aborted)."""
        self._resume_at_ms = None
        self._normal_onset()
        self._set(TurnState.LISTENING)

    # 문턱을 평상시 수준으로 되돌린다.
    def _normal_onset(self) -> None:
        self.detector.onset_required = None
        self.detector.onset_window = None
        self.detector.reset()

    # 꼬리 보호 시간이 지났으면 듣기로 복귀.
    def _maybe_resume(self, now_ms: float) -> None:
        if self._resume_at_ms is not None and now_ms >= self._resume_at_ms:
            self.listen()
