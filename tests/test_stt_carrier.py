"""Korean maths read aloud comes back as Japanese, and what is done about it.

The /v1/stt endpoint detects the language and cannot be told (measured: the
same audio under language=ko/ja/en/zh, under six other field names and under
no field at all is byte-identical). "마이너스 이 엑스 마이너스 삼이요" is very nearly
the Japanese for the same thing, so the detector picks ja and writes correct
phonetics in katakana. The fix is to splice a Korean phrase in front, let the
detector hear Korean, and cut the phrase back off with the word timings.

These tests use the real carrier asset — its length is what the cut is
measured against — but never the network.
"""

import wave

import pytest

from tutor.config import Settings
from tutor.speech import stt
from tutor.speech.stt import (
    CARRIER_PATH,
    CARRIER_RATE,
    XaiTranscriber,
    _after,
    _strip_carrier,
    carrier_pcm,
    wrong_script,
)

# what the endpoint actually returned for that utterance, before the fix
KATAKANA = "マイナス二エックスマイナス三よ"
# and after: the carrier, then the answer, with the timings it came back with
CARRIER_SECONDS = 1.404


def words(*pairs):
    return [{"text": t, "start": s, "end": e} for t, s, e in pairs]


class TestWrongScript:
    @pytest.mark.parametrize(
        "text",
        [KATAKANA, "いいよ", "ね", "サミオ", "シビオ", "プラスマイナスルートオ", "三エックスマイナス二"],
    )
    def test_the_measured_failures_all_ask_for_a_retry(self, text):
        assert wrong_script(text)

    @pytest.mark.parametrize(
        "text",
        [
            "마이너스 2x 마이너스 3이요",
            "네",
            "답은 마이너스 삼입니다",
            "5",
            "y = -3x + 3",
            # a bilingual student's real English answer is not a wrong script
            "I think the slope is minus three",
            # one stray glyph in a Korean sentence is noise, not a mis-detection
            "5를 빼면 돼요 好",
            "",
        ],
    )
    def test_what_must_not_trigger_one(self, text):
        assert not wrong_script(text)


class TestCarrierAsset:
    def test_it_is_the_format_the_splice_assumes(self):
        with wave.open(str(CARRIER_PATH), "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)

    def test_its_length_is_what_the_cut_measures_against(self):
        seconds = len(carrier_pcm()) / 2 / CARRIER_RATE
        assert seconds == pytest.approx(CARRIER_SECONDS, abs=0.02)


class TestCut:
    def test_the_student_starts_where_the_carrier_ends(self):
        data = {
            "text": "자, 대답할게요. 마이너스 2x 마이너스 3이요.",
            "words": words(
                ("자,", 0.04, 0.18), ("대답할게요.", 0.6, 1.2),
                ("마이너스", 1.5, 2.0), ("2x", 2.1, 2.4), ("마이너스", 2.5, 2.9),
                ("3이요.", 3.0, 3.4),
            ),
        }
        assert _after(data, CARRIER_SECONDS) == "마이너스 2x 마이너스 3이요."

    def test_a_one_word_answer_survives(self):
        data = {"text": "자, 대답할게요. 네.",
                "words": words(("자,", 0.04, 0.18), ("대답할게요.", 0.6, 1.2), ("네.", 1.5, 1.7))}
        assert _after(data, CARRIER_SECONDS) == "네."

    def test_a_word_straddling_the_seam_belongs_to_the_student(self):
        # the splice is sample-exact, so nothing of the carrier's can end late;
        # dropping this would eat the answer
        data = {"words": words(("대답할게요.", 0.6, 1.2), ("삼이요.", 1.39, 1.8))}
        assert _after(data, CARRIER_SECONDS) == "삼이요."

    def test_no_timings_falls_back_to_taking_the_phrase_off_the_front(self):
        assert _after({"text": "자, 대답할게요. 마이너스 삼."}, CARRIER_SECONDS) == "마이너스 삼."

    def test_a_leaked_fragment_is_taken_off_too(self):
        data = {"words": words(("자, 대답할게요.", 0.04, 1.42), ("네.", 1.5, 1.7))}
        assert _after(data, CARRIER_SECONDS) == "네."

    @pytest.mark.parametrize(
        "text,want",
        [
            ("자, 대답할게요. 네", "네"),
            ("자 대답할게요 마이너스 삼", "마이너스 삼"),          # spelled without punctuation
            ("자, 대답할게요.", ""),                          # the student said nothing
            ("자, 대답이요. 네", "자, 대답이요. 네"),            # not the carrier: left alone
            ("마이너스 삼", "마이너스 삼"),                     # nothing to strip
        ],
    )
    def test_stripping_the_phrase(self, text, want):
        assert _strip_carrier(text) == want


def transcriber(monkeypatch, plain, carried):
    """/v1/stt in the two moods it has: detect ja, or detect ko with a carrier."""
    calls = []

    def post(self, pcm, sample_rate):
        calls.append(len(pcm))
        return carried if len(calls) > 1 else plain

    monkeypatch.setattr(XaiTranscriber, "_post", post)
    return XaiTranscriber(Settings()), calls


class TestRetry:
    def test_katakana_is_asked_again_with_the_carrier_and_recovered(self, monkeypatch):
        t, calls = transcriber(
            monkeypatch,
            {"text": KATAKANA, "language": "ja"},
            {"text": "자, 대답할게요. 마이너스 2x 마이너스 3이요.", "language": "ko",
             "words": words(("자,", 0.04, 0.18), ("대답할게요.", 0.6, 1.2),
                            ("마이너스", 1.5, 2.0), ("2x", 2.1, 2.4),
                            ("마이너스", 2.5, 2.9), ("3이요.", 3.0, 3.4))},
        )
        got = t.transcribe(b"\0\0" * 16000)
        assert got.text == "마이너스 2x 마이너스 3이요."
        assert got.language == "ko"
        # the second call carried the whole utterance plus the phrase
        assert len(calls) == 2
        assert calls[1] == calls[0] + len(carrier_pcm())

    def test_a_korean_transcript_is_not_asked_twice(self, monkeypatch):
        t, calls = transcriber(monkeypatch, {"text": "답은 마이너스 삼입니다", "language": "ko"}, {})
        assert t.transcribe(b"\0\0" * 16000).text == "답은 마이너스 삼입니다"
        assert len(calls) == 1

    def test_an_english_answer_is_not_asked_twice(self, monkeypatch):
        t, calls = transcriber(monkeypatch, {"text": "the slope is minus three", "language": "en"}, {})
        assert t.transcribe(b"\0\0" * 16000).text == "the slope is minus three"
        assert len(calls) == 1

    def test_a_retry_that_does_not_help_keeps_the_first_answer(self, monkeypatch):
        t, _ = transcriber(
            monkeypatch,
            {"text": KATAKANA, "language": "ja"},
            {"text": "자, 대답할게요. マイナス三", "language": "ja",
             "words": words(("자,", 0.04, 0.18), ("대답할게요.", 0.6, 1.2),
                            ("マイナス三", 1.5, 2.0))},
        )
        got = t.transcribe(b"\0\0" * 16000)
        assert got.text == KATAKANA      # and the quality gate will re-ask
        assert got.language == "ja"

    def test_a_retry_that_raises_keeps_the_first_answer(self, monkeypatch):
        calls = []

        def boom(self, pcm, sample_rate):
            calls.append(len(pcm))
            if len(calls) > 1:
                raise RuntimeError("endpoint down")
            return {"text": KATAKANA, "language": "ja"}

        monkeypatch.setattr(XaiTranscriber, "_post", boom)
        assert XaiTranscriber(Settings()).transcribe(b"\0\0" * 16000).text == KATAKANA

    def test_audio_at_another_rate_is_left_alone(self, monkeypatch):
        t, calls = transcriber(monkeypatch, {"text": KATAKANA, "language": "ja"}, {})
        assert t.transcribe(b"\0\0" * 8000, sample_rate=8000).text == KATAKANA
        assert len(calls) == 1      # no carrier to splice at 8 kHz


def test_the_recovered_transcript_passes_the_quality_gate():
    assert stt.classify_transcript("마이너스 2x 마이너스 3이요.", "ko") == "ok"
    assert stt.classify_transcript(KATAKANA, "ja") == "unclear"
