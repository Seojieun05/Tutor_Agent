"""What the tutor's mouth does to notation.

The reference cases are the two problems on the worksheet this was built
against: a product-rule derivative and a pair of logs. If speakable() reads
those two aloud sensibly, it earns its place in front of the TTS.
"""

import pytest

from tutor.speech.mathspeak import displayable, ends_in_consonant, speakable


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


class TestLatexReadsAloud:
    """The hint model writes LaTeX at the student; the mouth must not spell it.

    The reference cases are the two parabolas that broke live:
    y = x^2 + 3 and y = -\\frac{1}{5}x^2 + 3.
    """

    def test_a_caret_power_is_spoken(self):
        said = speakable("y = x^2 + 3")
        assert said == "y는 x 제곱 더하기 3"

    def test_a_latex_fraction_reads_denominator_first(self):
        said = speakable("y = -\\frac{1}{5}x^2 + 3")
        assert "5분의 1" in said
        assert "마이너스" in said                       # the sign survives the frac
        assert "x 제곱" in said
        assert "frac" not in said and "\\" not in said and "{" not in said

    def test_braced_exponents_and_delimiters(self):
        said = speakable("$x^{2} \\cdot \\sqrt{2}$")
        assert "제곱" in said and "곱하기" in said and "루트 2" in said
        assert "$" not in said and "{" not in said

    def test_a_variable_exponent_is_spoken(self):
        assert "x의 n제곱" in speakable("x^n = 8")

    def test_a_sequence_index_is_spoken_without_the_underscore(self):
        said = speakable("a_4 = a_1 + 3")
        assert "_" not in said
        assert "a 4" in said and "a 1" in said


class TestTheScreenGetsNotation:
    """displayable(): the same boundary as speakable(), split by destination —
    the transcript panel shows 2·x², never '2 x 제곱' and never '2*x**2'."""

    def test_ascii_powers_become_print_notation(self):
        from tutor.speech.mathspeak import displayable

        assert displayable("f(x) = (x+2)(2*x**2-x-2)") == "f(x) = (x+2)(2·x²-x-2)"
        assert displayable("x**3 - x**10") == "x³ - x¹⁰"   # raised, not careted
        assert displayable("x^n = 8") == "xⁿ = 8"
        assert displayable("sqrt(2)") == "√(2)"

    def test_sequence_indices_sink(self):
        from tutor.speech.mathspeak import displayable

        assert displayable("a_4 + a_5 = 7") == "a₄ + a₅ = 7"
        assert displayable("x_n = 2*x_1") == "xₙ = 2·x₁"
        # an index unicode cannot sink stays whole, not half-set
        assert displayable("a_b = 1") == "a_b = 1"

    def test_a_general_term_sinks_whole(self):
        """a_{n+1} is the general term of a sequence. Sinking only the "n"
        leaves "aₙ + 1", which is a sum — a different claim entirely."""
        from tutor.speech.mathspeak import displayable

        assert displayable("a_{n+1} = a_n * r") == "aₙ₊₁ = aₙ·r"
        assert displayable("a_{n+k} = a_n * r**k") == "aₙ₊ₖ = aₙ·rᵏ"
        # a subscript unicode cannot carry stays braced rather than half-sunk
        assert displayable("a_{2i} = 0") == "a_2i = 0"

    def test_primes_and_equals_stay_as_written(self):
        from tutor.speech.mathspeak import displayable

        said = displayable("f'(1) = 2*3")
        assert "f'(1)" in said and "=" in said and "2·3" in said

    def test_logs_sink_their_base(self):
        from tutor.speech.mathspeak import displayable

        assert displayable("log_3 (b/a) = 1/2") == "log₃ (b/a) = 1/2"
        assert displayable("log_a b = 3") == "logₐ b = 3"
        assert displayable("\\log_{2} x") == "log₂ x"
        assert displayable("log_10 x = 2") == "log₁₀ x = 2"
        # a base with no unicode subscript stays as written, not half-sunk
        assert displayable("log_b a = 1") == "log_b a = 1"

    def test_latex_becomes_print_notation(self):
        from tutor.speech.mathspeak import displayable

        assert displayable("y = x^2 + 3") == "y = x² + 3"
        # the fraction is parenthesized because a term follows it
        assert displayable("y = -\\frac{1}{5}x^2 + 3") == "y = -(1/5)x² + 3"
        assert displayable("\\frac{1}{2}") == "1/2"
        assert displayable("\\frac{x+1}{2} = 3") == "(x+1)/2 = 3"
        assert displayable("$\\sqrt{2} \\cdot x^{10}$") == "√(2)·x¹⁰"

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
            READOUT_CLOSERS,
            READOUT_OPENER,
            RETRY_PROMPTS,
            WORK_CHECK_DEFAULT,
            WORK_CHECK_OPENERS,
            WORK_CHECK_REACTIONS,
            WORK_CONFIRMED,
        )
        from tutor.speech.filler import FILLER_PHRASES, WORK_CHECK_NARRATIONS

        for phrase in [PROBLEM_DONE, WORK_CONFIRMED, WORK_CHECK_DEFAULT,
                       *WORK_CHECK_OPENERS, *WORK_CHECK_NARRATIONS,
                       *(t for t in WORK_CHECK_REACTIONS.values() if t),
                       READOUT_OPENER, *READOUT_CLOSERS.values(),
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


class TestTheEarReadsSpokenNumerals:
    """STT writes what it hears. A correct "그럼 엑스는 칠에서 만날 것 같은데"
    reached the grader with no digit in it, the composite-step check found no
    7, and a finished step was graded PARTIAL. Reading 칠 as 7 is the fix —
    but the same syllables build 구해요, 넓이, 삼각형 and 칠판, so a numeral
    counts only when it starts a word and something grammatical follows it."""

    @pytest.mark.parametrize("said,expected", [
        ("그럼 엑스는 칠에서 만날 것 같은데.", "7"),
        ("마이너스 이요", "2"),
        ("이십사 나누기 칠이요", "24"),
        ("답은 구십팔인가?", "98"),
        ("f 프라임 일은 마이너스 이예요", "1"),
        ("십이요", "12"),
    ])
    def test_a_spoken_number_becomes_a_digit(self, said, expected):
        from tutor.speech.mathspeak import with_digits
        assert expected in with_digits(said)

    @pytest.mark.parametrize("said", [
        "먼저 도함수를 구해요",          # 구
        "삼각형의 넓이를 구하면 돼요",   # 삼, 이
        "이 문제 어떻게 풀어요?",        # the demonstrative
        "칠판에 쓸까요?",                # 칠
        "사용한 공식이 뭐예요?",         # 사, 이
        "일단 정리해 볼게요",            # 일
        "구하는 값이 뭐죠",              # 구, 이
        "이차방정식이요",
        "이항하면 돼요",
        "일이 많아요",
    ])
    def test_ordinary_words_are_left_alone(self, said):
        from tutor.speech.mathspeak import with_digits
        assert with_digits(said) == said


class TestTheVoiceLeavesNothingToInterpret:
    """A number the engine still has to read is a number it can read wrong.
    Live, the confirmation "정답은 49, 맞는지 확인해 볼게요" came out as 사만
    구, so a value quoted back by voice is spelled before it is sent."""

    @pytest.mark.parametrize("value,said", [
        (0, "영"), (4, "사"), (10, "십"), (11, "십일"), (20, "이십"),
        (49, "사십구"), (100, "백"), (249, "이백사십구"), (2026, "이천이십육"),
        (-2, "마이너스 이"),
    ])
    def test_the_reading_a_teacher_says(self, value, said):
        from tutor.speech.mathspeak import sino_korean
        assert sino_korean(value) == said

    def test_a_quoted_value_is_spelled(self):
        from tutor.speech.mathspeak import spell_numbers
        assert spell_numbers("정답은 49") == "정답은 사십구"
        assert spell_numbers("마이너스 3") == "마이너스 삼"

    @pytest.mark.parametrize("text", [
        "y = x^2 + 3",        # speakable reads this; it needs the digits
        "24/7",
        "f'(1) = 8",
        "3.5",                # a decimal has its own reading
        "12345",              # past 천, left to the engine
    ])
    def test_mathematics_and_oddities_are_left_alone(self, text):
        from tutor.speech.mathspeak import spell_numbers
        assert spell_numbers(text) == text

    def test_the_echo_spells_what_it_quotes(self):
        import random
        from tutor.speech.filler import FillerBank
        said = FillerBank(rng=random.Random(3)).echo("정답은 49")
        assert "사십구" in said and "49" not in said


class TestAParticleIsNotADenominator:
    """\w matches Korean, so the fraction rule swallowed the particle after
    the denominator: problem 12's answer read as "7이요분의 24"."""

    @pytest.mark.parametrize("text,said", [
        ("답은 24/7이요", "답은 7분의 24이요"),
        ("24/7이에요", "7분의 24이에요"),
        ("a/b는", "b분의 a는"),
        ("3/4 입니다", "4분의 3 입니다"),
    ])
    def test_the_particle_stays_outside_the_fraction(self, text, said):
        assert speakable(text) == said


class TestAParticleIsNotAPause:
    """`)` becomes a pause where the grouping was — but "f(x)와 곱하고" is one
    noun phrase, and the pause landed between the value and the particle
    hanging off it: "에프 엑스, 와 곱하고". A closing paren followed straight
    by hangul closes a name, not a group."""

    @pytest.mark.parametrize("text,said", [
        ("f(x)와 곱하고", "f x와 곱하고"),
        ("g(x)의 식을 써 볼까요?", "g x의 식을 써 볼까요?"),
        ("f(1)은 얼마일까요?", "f 1은 얼마일까요?"),
    ])
    def test_the_particle_stays_attached(self, text, said):
        assert speakable(text) == said

    def test_a_real_grouping_still_pauses(self):
        said = speakable("(x + 1)(x - 1) = 0")
        assert ", " in said                 # the group boundary is heard
        assert "(" not in said and ")" not in said

    def test_a_function_call_alone_is_math_enough(self):
        """Without this the commonest question the tutor asks never reached
        the reader, and the engine got the parens to do as it liked with."""
        assert speakable("f(x)를 미분해 볼까요?") == "f x를 미분해 볼까요?"

    @pytest.mark.parametrize("text", [
        "어떤 항을 반대쪽으로 옮겨야 할까요?",
        "5라고 했네요. 한번 볼게요.",
    ])
    def test_plain_korean_is_still_untouched(self, text):
        assert speakable(text) == text


class TestAPointIsReadAsAPoint:
    """Live: "점 (1, 6)을 지나는 m의 방정식" was read as "점 일 육" — two
    numbers with nothing between them, which is not a point. The sentence
    never even reached the reader: a bare coordinate carried none of the
    signals that say "there is mathematics here"."""

    def test_the_comma_is_heard(self):
        assert speakable("이제 점 (1, 6)을 지나는 m의 방정식은?") == (
            "이제 점 1 콤마 6을 지나는 m의 방정식은?"
        )

    def test_a_negative_coordinate_is_a_sign_not_a_subtraction(self):
        assert "마이너스 6" in speakable("점 (1, -6)에서의 접선")
        assert "빼기" not in speakable("점 (1, -6)에서의 접선")

    def test_the_screen_still_shows_notation(self):
        assert displayable("점 (1, 6)을 지나는") == "점 (1, 6)을 지나는"

    def test_a_grouping_is_not_mistaken_for_a_point(self):
        said = speakable("(x + 1)(x - 1) = 0")
        assert "콤마" not in said


class TestTheParticleHearsThroughAParen:
    """"f(1) = -2" is read "에프 일은 마이너스 이": the 1 decides the particle,
    not the ")" written after it. The rule only ever saw one character."""

    @pytest.mark.parametrize("text,expected", [
        ("f(1) = -2", "f 1은 마이너스 2"),
        ("g(2) = 3", "g 2는 3"),
        ("x = 5", "x는 5"),
    ])
    def test_the_particle_follows_the_sound(self, text, expected):
        assert speakable(text) == expected


class TestAMinusNeedsSomethingToSubtractFrom:
    """Live, the L2 for line l: "기울기가 -2인 직선은…" was read "기울기가
    빼기 2인 직선은". The sign was granted only after ^, =, ( or a comma, so a
    minus that followed Korean prose became a subtraction — and the sentence
    never even reached the reader, because a signed number was not on the
    list of things that mean "there is mathematics here"."""

    @pytest.mark.parametrize("text,expected", [
        ("기울기가 -2인 직선", "기울기가 마이너스 2인 직선"),
        ("기울기 -2와 점", "기울기 마이너스 2와 점"),
        ("x = -3", "x는 마이너스 3"),
    ])
    def test_after_prose_it_is_a_sign(self, text, expected):
        assert speakable(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("2*x - 4", "2 곱하기 x 빼기 4"),
        ("3*x**2 - 2", "3 곱하기 x 제곱 빼기 2"),
        ("y = -2*x - 4", "y는 마이너스 2 곱하기 x 빼기 4"),
    ])
    def test_after_an_operand_it_is_a_subtraction(self, text, expected):
        assert speakable(text) == expected

    def test_the_whole_live_line(self):
        said = speakable("기울기가 -2인 직선은 다음과 같은 꼴로 나타낼 수 있어요. "
                         "접점 (1, -6)을 이 식에 넣어 볼까요?")
        assert "기울기가 마이너스 2인" in said
        assert "1 콤마 마이너스 6" in said
        assert "빼기" not in said
