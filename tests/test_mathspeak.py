"""What the tutor's mouth does to notation.

The reference cases are the two problems on the worksheet this was built
against: a product-rule derivative and a pair of logs. If speakable() reads
those two aloud sensibly, it earns its place in front of the TTS.
"""

from tutor.speech.mathspeak import ends_in_consonant, speakable


class TestTheWorksheetReadsAloud:
    def test_the_derivative_problem(self):
        said = speakable("함수 f(x) = (x+2)(2x**2-x-2)에 대하여 f'(1)의 값은?")
        assert "프라임 1" in said                       # f'(1)
        assert "제곱" in said and "**" not in said      # 2x**2
        assert "곱하기" in said                         # )( adjacency
        assert "(" not in said and ")" not in said
        assert "=" not in said

    def test_the_log_problem(self):
        said = speakable("log_a b = 3, log_3 b/a = 1/2 을 만족시킬 때, log_9 ab의 값은?")
        assert "밑이 a인 로그 b" in said
        assert "a분의 b" in said                        # b/a, denominator first
        assert "2분의 1" in said                        # 1/2
        assert "log" not in said

    def test_a_student_line_with_a_times_sign(self):
        said = speakable("f'(1) = 2-1-2+3×1")
        assert "프라임 1" in said
        assert "곱하기" in said and "×" not in said
        assert "빼기" in said

    def test_a_negative_number_is_minus_not_subtraction(self):
        assert "마이너스 1" in speakable("f'(1) = -1 + 9")
        assert speakable("x = -3").endswith("마이너스 3")

    def test_equals_picks_the_korean_particle(self):
        assert "x는 5" in speakable("x = 5")            # x — vowel-ish, 는
        assert "1은 8" in speakable("f'(1) = 8")        # 일 — final consonant, 은


class TestTheScreenGetsNotation:
    """displayable(): the same boundary as speakable(), split by destination —
    the transcript panel shows 2·x², never '2 x 제곱' and never '2*x**2'."""

    def test_ascii_powers_become_print_notation(self):
        from tutor.speech.mathspeak import displayable

        assert displayable("f(x) = (x+2)(2*x**2-x-2)") == "f(x) = (x+2)(2·x²-x-2)"
        assert displayable("x**3 - x**10") == "x³ - x^10"
        assert displayable("sqrt(2)") == "√(2)"

    def test_primes_and_equals_stay_as_written(self):
        from tutor.speech.mathspeak import displayable

        said = displayable("f'(1) = 2*3")
        assert "f'(1)" in said and "=" in said and "2·3" in said

    def test_plain_korean_is_untouched(self):
        from tutor.speech.mathspeak import displayable

        for text in ["맞아요! 이대로 하면 돼요.", "어떤 항을 옮겨야 할까요?"]:
            assert displayable(text) == text


class TestItLeavesSpeechAlone:
    def test_the_fixed_phrases_are_untouched(self):
        """These are TTS cache keys: a changed byte is a cache miss."""
        from tutor.hints.generator import FIXED_ACTIONS
        from tutor.server.session import (
            PROBLEM_DONE,
            RETRY_PROMPTS,
            WORK_CHECK_OPENER,
            WORK_CONFIRMED,
        )
        from tutor.speech.filler import FILLER_PHRASES

        for phrase in [PROBLEM_DONE, WORK_CHECK_OPENER, WORK_CONFIRMED,
                       *RETRY_PROMPTS.values(), *FILLER_PHRASES,
                       *(t for t in FIXED_ACTIONS.values() if t)]:
            assert speakable(phrase) == phrase, phrase

    def test_plain_korean_hints_pass_through(self):
        for text in ["어떤 항을 반대쪽으로 옮겨야 할까요?",
                     "5라고 했네요. 한번 볼게요.",
                     "양변에서 무엇을 빼면 좋을까요?"]:
            assert speakable(text) == text


class TestParticleHelper:
    def test_hangul_and_digits(self):
        assert ends_in_consonant("일")           # ㄹ
        assert not ends_in_consonant("오")
        assert ends_in_consonant("1")            # 일
        assert not ends_in_consonant("5")        # 오
        assert not ends_in_consonant("")
