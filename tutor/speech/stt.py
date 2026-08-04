"""Speech-to-text via the xAI /v1/stt endpoint (multipart, no model name)."""

from __future__ import annotations

import io
import logging
import re
import wave
from typing import Literal

import httpx
from pydantic import BaseModel

from tutor.config import Settings

log = logging.getLogger(__name__)

HINT_KEYWORDS = ("힌트", "도와", "모르겠", "hint", "help")

# --- transcript quality gate --------------------------------------------------

TranscriptQuality = Literal["ok", "unclear", "filler_only"]

# Room noise transcribed as speech is usually a stray CJK/Cyrillic glyph or a
# hesitation sound. Grading either as a maths answer is worse than asking again.
_HANGUL_SYLLABLES = (0xAC00, 0xD7A3)
_FOREIGN_RATIO = 0.2  # a stray glyph is tolerable; a sentence of them is not

_FILLERS = frozenset(
    {
        # Korean hesitation sounds (elongation is collapsed before lookup)
        "음", "어", "아", "으", "엄", "그", "저", "흠", "에", "허", "오",
        # English, for a bilingual STT
        "um", "uh", "er", "erm", "hmm", "hm", "mm", "ah", "oh", "eh",
    }
)


class Transcript(BaseModel):
    text: str
    language: str = "ko"
    confidence: float = 1.0


def wants_hint(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in HINT_KEYWORDS)


def _is_hangul(ch: str) -> bool:
    return _HANGUL_SYLLABLES[0] <= ord(ch) <= _HANGUL_SYLLABLES[1]


def _is_expected(ch: str) -> bool:
    """Korean, English or a digit — what a Korean maths answer is made of."""
    return _is_hangul(ch) or ("a" <= ch.lower() <= "z") or ch.isdigit()


def _collapse(token: str) -> str:
    """'어어어' → '어', so elongated hesitations still read as fillers."""
    return re.sub(r"(.)\1+", r"\1", token)


def classify_transcript(text: str) -> TranscriptQuality:
    """Is this worth running the tutor pipeline on?

    "unclear"     — empty, punctuation only, lone jamo, or mostly characters
                    that are neither Korean, English nor digits (STT noise).
    "filler_only" — real speech, but only hesitation sounds.
    "ok"          — everything else, including a bare number like "5".
    """
    stripped = text.strip()
    if not stripped:
        return "unclear"

    expected = sum(1 for ch in stripped if _is_expected(ch))
    # letters from another script (Chinese, Japanese, Cyrillic, bare jamo …)
    foreign = sum(1 for ch in stripped if ch.isalpha() and not _is_expected(ch))
    if expected == 0:
        return "unclear"
    if foreign and foreign / (expected + foreign) > _FOREIGN_RATIO:
        return "unclear"

    tokens = [t for t in re.split(r"[^\w]+", stripped) if t]
    if tokens and all(_collapse(t).lower() in _FILLERS for t in tokens):
        return "filler_only"
    return "ok"


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
