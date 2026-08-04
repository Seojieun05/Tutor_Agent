"""Speech-to-text via the xAI /v1/stt endpoint (multipart, no model name)."""

from __future__ import annotations

import io
import logging
import wave

import httpx
from pydantic import BaseModel

from tutor.config import Settings

log = logging.getLogger(__name__)

HINT_KEYWORDS = ("힌트", "도와", "모르겠", "hint", "help")


class Transcript(BaseModel):
    text: str
    language: str = "ko"
    confidence: float = 1.0


def wants_hint(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in HINT_KEYWORDS)


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, bits: int = 16, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class XaiTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        wav = pcm_to_wav(pcm, sample_rate)
        resp = httpx.post(
            f"{self.settings.xai_base_url.rstrip('/')}/stt",
            headers={"Authorization": f"Bearer {self.settings.xai_api_key}"},
            data={"language": "ko"},
            files={"file": ("utterance.wav", wav, "audio/wav")},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = self._extract_text(data)
        log.info("STT transcript: %r", text)
        return Transcript(text=text, language=self.settings.tutor_language)

    @staticmethod
    def _extract_text(data) -> str:
        if isinstance(data, str):
            return data
        for key in ("text", "transcript", "transcription"):
            if isinstance(data.get(key), str):
                return data[key]
        segments = data.get("segments")
        if isinstance(segments, list):
            return " ".join(
                s.get("text", "") for s in segments if isinstance(s, dict)
            ).strip()
        return ""


class EchoTranscriber:
    """No-key mode: every utterance counts as a hint request."""

    def __init__(self, settings: Settings | None = None):
        pass

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        return Transcript(text="힌트 주세요", language="ko")
