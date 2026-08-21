"""Speech-to-text via the xAI /v1/stt endpoint (multipart, no model name).

THE ENDPOINT DETECTS THE LANGUAGE AND CANNOT BE TOLD. Measured against the
live API: the same audio sent with language=ko, ja, en, zh, with no language
at all, and under `lang`, `source_language`, `audio_language`,
`language_code`, `detect_language`, `task` and a deliberately invented field
all came back byte-identical, and /v1/audio/transcriptions and its neighbours
are 404. The field we send is a report, never an instruction.

That matters because of what a maths student says out loud. Read a line of
working aloud — "마이너스 이 엑스 마이너스 삼이요" — and almost every syllable is a
loanword or a Sino-Korean numeral, which is to say it is very nearly the same
sound as the Japanese for the same thing. The detector picks ja and writes
perfectly good phonetics in the wrong script: マイナス二エックスマイナス三よ.
Measured over 28 spoken answers, 7 to 10 came back Japanese, including a bare
"네". Everything with ordinary Korean grammar in it — 답은…입니다, 기울기가…
나왔어요 — was detected correctly every time.

So the repair is to put some ordinary Korean in the audio. A fixed carrier
phrase is spliced in front of the utterance, the detector hears Korean and
decodes the WHOLE buffer as Korean, and the carrier is cut back off using the
word timings the response already carries. With the carrier, all 28 were
detected as Korean and none was lost to the cut.

It is a RETRY rather than something done to every utterance: the carrier
slightly colours the word right after the seam (루트 → 로트, 파이 → 화이), which
is not worth paying on the three quarters of turns that never needed it.
"""

from __future__ import annotations

import io
import logging
import re
import wave
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from tutor.config import Settings

log = logging.getLogger(__name__)

# 캐리어(운반) 문장: 발화 앞에 붙여 STT가 한국어로 인식하게 만드는 고정 음성.
# 인식 후에는 단어 타이밍을 이용해 다시 잘라 낸다.
# The words in carrier_ko.wav, kept here so a leaked fragment can be recognised
# and taken off the front. 1.4s of 16 kHz mono, ending in a short silence.
CARRIER_PATH = Path(__file__).with_name("carrier_ko.wav")
CARRIER_TEXT = "자, 대답할게요."
CARRIER_RATE = 16000

# 힌트 요청으로 볼 낱말들.
HINT_KEYWORDS = ("힌트", "도와", "모르겠", "hint", "help")

# --- transcript quality gate --------------------------------------------------

TranscriptQuality = Literal["ok", "unclear", "filler_only"]

# Room noise transcribed as speech is usually a stray CJK/Cyrillic glyph or a
# hesitation sound. Grading either as a maths answer is worse than asking again.
# 한글 음절 코드 범위.
_HANGUL_SYLLABLES = (0xAC00, 0xD7A3)
# 이 비율을 넘게 낯선 문자면 잘못된 문자로 받아 적힌 것으로 본다.
_FOREIGN_RATIO = 0.2  # a stray glyph is tolerable; a sentence of them is not

# 어·음 같은 망설임 소리 목록. 이것뿐이면 답이 아니다.
_FILLERS = frozenset(
    {
        # Korean hesitation sounds (elongation is collapsed before lookup)
        "음", "어", "아", "으", "엄", "그", "저", "흠", "에", "허", "오",
        # English, for a bilingual STT
        "um", "uh", "er", "erm", "hmm", "hm", "mm", "ah", "oh", "eh",
    }
)


# STT 결과: 받아 적은 말과, 엔진이 판단한 언어.
class Transcript(BaseModel):
    text: str
    language: str = "ko"
    confidence: float = 1.0


# 힌트를 달라는 말인지(간단한 낱말 검사).
def wants_hint(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in HINT_KEYWORDS)


# 한글 음절인지.
def _is_hangul(ch: str) -> bool:
    return _HANGUL_SYLLABLES[0] <= ord(ch) <= _HANGUL_SYLLABLES[1]


# 수학 답변에 나올 법한 문자인지.
def _is_expected(ch: str) -> bool:
    """Korean, English or a digit — what a Korean maths answer is made of."""
    return _is_hangul(ch) or ("a" <= ch.lower() <= "z") or ch.isdigit()


# 비교용으로 토큰을 단순화.
def _collapse(token: str) -> str:
    """'어어어' → '어', so elongated hesitations still read as fillers."""
    return re.sub(r"(.)\1+", r"\1", token)


# 영어 단어 꼴 토큰 수.
def _english_words(text: str) -> int:
    """Word-shaped Latin tokens. 'x' and the 'x' inside '2x' are not words."""
    return sum(1 for t in re.split(r"[^A-Za-z]+", text) if len(t) >= 2)


# 낯선 문자 비율.
def _foreign_ratio(text: str) -> float:
    """How much of this is written in a script the student does not use.

    Korean, English and digits are what a Korean maths answer is made of;
    kana, kanji and Cyrillic are either room noise or — far more often — the
    right sounds written in the wrong alphabet.
    """
    stripped = text.strip()
    expected = sum(1 for ch in stripped if _is_expected(ch))
    foreign = sum(1 for ch in stripped if ch.isalpha() and not _is_expected(ch))
    if not (expected or foreign):
        return 0.0
    return foreign / (expected + foreign)


# 한국어가 다른 문자(가나 등)로 받아 적혔는지 = 캐리어를 붙여 다시 물어볼 만한지.
def wrong_script(text: str) -> bool:
    """Worth asking the endpoint again with a Korean phrase in front of it."""
    return bool(text.strip()) and _foreign_ratio(text) > _FOREIGN_RATIO


# 한국어를 영어 문장으로 번역해 버렸는지(한글이 하나도 없고 영어 단어가 둘 이상).
def english_sentence(text: str) -> bool:
    """Korean audio written as English WORDS — the model translated.

    The kana failure writes the right sounds in the wrong alphabet;
    this one rewrites the meaning in another language ("The slope is
    minus 2"), and Latin letters are not foreign to a maths answer, so
    _foreign_ratio never sees it. The tell is the absence: no hangul at
    all, plus at least two word-shaped Latin tokens. "y = -2x - 4" has
    no English words and stays untouched, and a transcript with any
    Korean in it is Korean with vocabulary, not a translation.
    """
    stripped = text.strip()
    return (
        bool(stripped)
        and not any(_is_hangul(ch) for ch in stripped)
        and _english_words(stripped) >= 2
    )


# 이 발화를 파이프라인에 태울 만한지 판정: ok / unclear / filler_only.
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

    if not any(_is_expected(ch) for ch in stripped):
        return "unclear"
    # letters from another script (Chinese, Japanese, Cyrillic, bare jamo …)
    if _foreign_ratio(stripped) > _FOREIGN_RATIO:
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


# 원시 PCM을 WAV 바이트로 감싼다(업로드용).
def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, bits: int = 16, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# 캐리어 음성 PCM을 한 번만 읽어 캐시.
@lru_cache(maxsize=1)
def carrier_pcm() -> bytes:
    """The Korean phrase spliced in front of a mis-detected utterance."""
    with wave.open(str(CARRIER_PATH), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (CARRIER_RATE, 1, 2):
            raise ValueError(f"{CARRIER_PATH.name} must be 16 kHz mono 16-bit")
        return w.readframes(w.getnframes())


# 앞머리에 새어 나온 캐리어 문장 조각을 잘라 낸다.
def _strip_carrier(text: str) -> str:
    """Take the carrier off the front if the timings let a piece of it through."""
    bare = re.sub(r"[^\w]", "", CARRIER_TEXT)
    seen, kept = "", 0
    for i, ch in enumerate(text):
        if re.match(r"\w", ch):
            seen += ch
            if not bare.startswith(seen):
                return text
            kept = i + 1
            if seen == bare:
                return text[kept:].lstrip(" ,.…")
    return text


# 캐리어가 끝난 시점 이후의 단어만 남긴다. 경계에 걸친 단어는 학생 것으로 본다.
def _after(data: dict, seconds: float) -> str:
    """What was said after the carrier ended.

    A word is kept when it ENDS after the seam, not when it starts there: the
    splice is sample-exact, so nothing of the student's can end before it,
    while a word that straddles the join is theirs. Erring this way leaks a
    carrier fragment at worst; erring the other way eats the first word of the
    answer, and for "네" that is the whole answer.
    """
    words = data.get("words") if isinstance(data, dict) else None
    if not isinstance(words, list) or not words:
        return _strip_carrier(XaiTranscriber._extract_text(data))
    kept = [
        str(w.get("text", ""))
        for w in words
        if isinstance(w, dict) and float(w.get("end", 0) or 0) > seconds
    ]
    return _strip_carrier(" ".join(kept).strip())


# xAI /v1/stt를 쓰는 음성 인식기.
class XaiTranscriber:
    # 설정(키·주소)을 받는다.
    def __init__(self, settings: Settings):
        self.settings = settings

    # 오디오를 multipart로 올려 인식 결과 JSON을 받는다.
    def _post(self, pcm: bytes, sample_rate: int) -> dict:
        resp = httpx.post(
            f"{self.settings.xai_base_url.rstrip('/')}/stt",
            headers={"Authorization": f"Bearer {self.settings.xai_api_key}"},
            # Sent for the record, not for effect — see the module docstring:
            # the endpoint gives the same answer whatever this says.
            data={"language": self.settings.tutor_language},
            files={"file": ("utterance.wav", pcm_to_wav(pcm, sample_rate), "audio/wav")},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {"text": data}

    # 발화 인식. 결과가 엉뚱한 문자로 나오면 캐리어를 붙여 한 번 더 시도한다.
    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        data = self._post(pcm, sample_rate)
        text = self._extract_text(data)
        # What it HEARD, not what we asked for — the request cannot ask. It is
        # the only way to tell a bilingual student's English answer from Korean
        # speech that came back in the wrong alphabet.
        heard = str(data.get("language", "") or "")
        if heard and heard != self.settings.tutor_language:
            log.info("STT heard %s, not %s", heard, self.settings.tutor_language)

        if wrong_script(text) or english_sentence(text):
            # both failures have the same repair: some ordinary Korean in the
            # audio makes the decoder TRANSCRIBE instead of translating
            again = self._with_carrier(pcm, sample_rate)
            if again is not None:
                text, heard = again
        log.info("STT transcript: %r", text)
        return Transcript(text=text, language=heard)

    # 캐리어를 앞에 붙여 재요청하고, 응답에서 그 부분을 잘라 낸다. 실패하면 None.
    def _with_carrier(self, pcm: bytes, sample_rate: int) -> tuple[str, str] | None:
        """Ask again with a Korean phrase in front, and cut the phrase back off.

        None when this cannot be done or did not help, so the caller keeps the
        first answer and the quality gate deals with it as it always has.
        """
        if sample_rate != CARRIER_RATE:
            log.info("no carrier for %d Hz audio; keeping the first transcript", sample_rate)
            return None
        try:
            head = carrier_pcm()
            data = self._post(head + pcm, sample_rate)
        except Exception:  # noqa: BLE001 — the first transcript is still there
            log.warning("carrier retry failed; keeping the first transcript", exc_info=True)
            return None
        text = _after(data, len(head) / 2 / CARRIER_RATE)
        heard = str(data.get("language", "") or "")
        if wrong_script(text) or english_sentence(text):
            # still the wrong alphabet, or still English through a Korean
            # carrier — the latter is a student who really spoke English,
            # and their first transcript stands
            log.info("carrier retry still came back %s: %r", heard or "?", text)
            return None
        log.info("carrier retry recovered %s: %r", heard or "?", text)
        return text, heard

    # 응답 JSON 형태가 어떻든 텍스트를 꺼낸다.
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


# 키 없는 모드: 모든 발화를 힌트 요청으로 친다.
class EchoTranscriber:
    """No-key mode: every utterance counts as a hint request."""

    # 설정은 받지만 쓰지 않는다.
    def __init__(self, settings: Settings | None = None):
        pass

    # 항상 같은 문장을 돌려준다.
    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        return Transcript(text="힌트 주세요", language="ko")
