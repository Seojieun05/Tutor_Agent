"""Silero VAD wrapper: one speech probability per fixed-size PCM frame.

Silero is an RNN — it carries state across frames, so ``reset()`` between turns
matters (a new utterance must not inherit the tail of the previous one).  It
also accepts only one frame size per rate: 512 samples at 16 kHz (32 ms),
256 at 8 kHz.

``torch``/``silero-vad`` are imported lazily so the server, the simulator, and
the test suite keep running without the ``voice`` extra installed.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Frame sizes Silero accepts, by sample rate.
FRAME_SAMPLES = {16000: 512, 8000: 256}

INSTALL_HINT = 'Silero VAD not installed — pip install -e ".[voice]"'


def frame_samples(sample_rate: int) -> int:
    try:
        return FRAME_SAMPLES[sample_rate]
    except KeyError:
        raise ValueError(
            f"Silero VAD supports {sorted(FRAME_SAMPLES)} Hz, got {sample_rate}"
        ) from None


class SileroVAD:
    """Speech/no-speech per frame, with the same ``is_speech`` shape webrtcvad
    has — the TurnDetector only needs that much and stays swappable."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples(sample_rate)
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise RuntimeError(INSTALL_HINT) from e
        self._torch = torch
        self._model = load_silero_vad()
        log.info(
            "Silero VAD ready (%d Hz, %d-sample frames, threshold %.2f)",
            sample_rate,
            self.frame_samples,
            threshold,
        )

    def speech_prob(self, frame: bytes) -> float:
        expected = self.frame_samples * 2
        if len(frame) != expected:
            raise ValueError(
                f"Silero needs exactly {self.frame_samples} samples "
                f"({expected} bytes) at {self.sample_rate} Hz, got {len(frame)}"
            )
        audio = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(audio), self.sample_rate).item())

    def is_speech(self, frame: bytes, sample_rate: int | None = None) -> bool:
        return self.speech_prob(frame) >= self.threshold

    def reset(self) -> None:
        self._model.reset_states()
