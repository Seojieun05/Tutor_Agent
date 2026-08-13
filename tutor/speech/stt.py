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


def _english_words(text: str) -> int:
    """Word-shaped Latin tokens. 'x' and the 'x' inside '2x' are not words."""
    return sum(1 for t in re.split(r"[^A-Za-z]+", text) if len(t) >= 2)


def classify_transcript(text: str, heard_language: str = "") -> TranscriptQuality:
    """Is this worth running the tutor pipeline on?

    `heard_language` is what the STT says the AUDIO was, not what we asked for
    — the two disagreeing is the whole point of passing it. Empty means no
    evidence either way, and the language rule below stays out of it.

    "unclear"     — empty, punctuation only, lone jamo, mostly characters that
                    are neither Korean, English nor digits (STT noise), or
                    Korean speech that came back as English words.
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

    # Heard Korean, wrote English: the model translated instead of transcribing.
    # A student who talked over the tutor had 4.1s of mixed audio come back as
    # "The equation is minus 2x squared" — graded INCORRECT, and the ladder
    # escalated twice against someone who had said the right thing in Korean.
    # The endpoint reports the language it HEARD (measured: English audio comes
    # back "en" even when the request asks for "ko"), so a bilingual student's
    # real English answer arrives as "en" and is not caught here. Numbers
    # survive too: "y = -2x - 4" contains no English WORDS.
    if (
        heard_language == "ko"
        and not any(_is_hangul(ch) for ch in stripped)
        and _english_words(stripped) >= 2
    ):
        return "unclear"
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
        # What it HEARD, not what we asked for. The request already says
        # language=ko and the endpoint still answers "en" for English audio, so
        # this field is evidence rather than an echo — and it is the only way to
        # tell a bilingual student's English answer from Korean speech that came
        # back translated. Absent (or a plain-string response) → no evidence.
        heard = data.get("language", "") if isinstance(data, dict) else ""
        if heard and heard != self.settings.tutor_language:
            log.info("STT heard %s, not %s", heard, self.settings.tutor_language)
        log.info("STT transcript: %r", text)
        return Transcript(text=text, language=str(heard or ""))

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
