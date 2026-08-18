from tutor.knowledge.mathnorm import (
    compute_answer,
    equations_equivalent,
    equations_same_form,
    equations_same_form_with_derivatives,
    expressions_equivalent,
    identity_text,
    instantiate,
    match_template,
    normalize_text,
    texts_identical_enough,
    verify_answer,
)


def test_normalize_text():
    assert normalize_text("다음을  푸시오: 3×x + 5 = 20!") == normalize_text(
        "다음을 푸시오: 3*x + 5 = 20"
    )


def test_normalize_text_superscripts_survive_nfkc():
    # NFKC alone would fold '²' into a bare '2' ("x2"); the math map must win
    assert normalize_text("x² - 9 = 0") == normalize_text("x**2 - 9 = 0")


class TestIdentityText:
    """Live: the typed presolve entry said "y = aˣ - 2" where the VLM wrote
    "y=a^x-2" — same printed problem, different rendering. identity_text
    erases exactly what a faithful read may change (superscript rendering,
    power notation, overlines, spacing) and nothing that identifies the
    problem (numbers, conditions, words)."""

    def test_the_photograph_reads_as_the_typed_entry(self):
        typed = "상수 a (a>1)에 대하여 곡선 y = aˣ - 2 위의 점"
        photographed = "상수 a(a>1)에 대하여 곡선 y=a^x-2 위의 점"
        assert identity_text(typed) == identity_text(photographed)

    def test_power_notations_converge(self):
        assert identity_text("y = a**x - 2") == identity_text("y=a^x-2")
        assert identity_text("x² - 9") == identity_text("x^2 - 9")

    def test_overline_calls_read_as_the_bare_segment(self):
        assert identity_text("AB = BC 이고") == identity_text(
            "overline(AB) = overline(BC) 이고"
        )

    def test_a_changed_number_is_a_different_problem(self):
        assert identity_text("넓이가 8일 때") != identity_text("넓이가 12일 때")


class TestTextsIdenticalEnough:
    """Live, capture two: the same printed "상수 a(a>1)" arrived as
    "실수 a(a>1)" at conf 1.00 — a syllable of ink. Hangul substitutions up
    to the cap are forgiven; digits, symbols and length never are."""

    def test_a_misread_syllable_is_forgiven(self):
        assert texts_identical_enough(
            "10. 상수 a (a>1)에 대하여 곡선 y = aˣ - 2 위의 점",
            "10. 실수 a(a>1)에 대하여 곡선 y=a^x-2 위의 점",
        )

    def test_a_changed_digit_never_is(self):
        assert not texts_identical_enough("넓이가 8일 때", "넓이가 9일 때")
        assert not texts_identical_enough("넓이가 8일 때", "넓이가 12일 때")

    def test_the_budget_is_two_syllables(self):
        assert texts_identical_enough("상수 예각 삼각형", "실수 둔각 삼각형")
        assert not texts_identical_enough("상수 예각 방정식", "실수 둔각 부등식")

    def test_empty_prose_identifies_nothing(self):
        assert not texts_identical_enough("", "")


class TestEquivalence:
    def test_identical(self):
        assert equations_equivalent("3*x + 5 = 20", "3*x + 5 = 20")

    def test_implicit_multiplication_and_caret(self):
        assert equations_equivalent("3x + 5 = 20", "3*x + 5 = 20")
        assert equations_equivalent("x^2 - 9 = 0", "x**2 - 9 = 0")

    def test_swapped_sides(self):
        assert equations_equivalent("20 = 3*x + 5", "3*x + 5 = 20")

    def test_scalar_multiple(self):
        assert equations_equivalent("6*x + 10 = 40", "3*x + 5 = 20")

    def test_variable_rename(self):
        assert equations_equivalent("3*y + 5 = 20", "3*x + 5 = 20")

    def test_rearranged(self):
        assert equations_equivalent("3*x = 15", "3*x - 15 = 0")

    def test_overline_is_notation_not_a_function(self):
        # the VLM transcribes a printed segment bar as overline(AB); the
        # stored equation writes the bare length
        assert equations_equivalent("overline(AB) = overline(BC)", "AB = BC")

    def test_not_equivalent(self):
        assert not equations_equivalent("3*x + 5 = 20", "3*x + 5 = 21")
        assert not equations_equivalent("3*x + 5 = 20", "x**2 = 4")

    def test_chain_equalities_compare_claim_by_claim(self):
        """a = b = c is two claims, not a parse error — the 등비수열 shape
        that used to reset the student's problem on every re-read."""
        chain = "2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6"
        assert equations_equivalent(chain, "2*(a_1+a_4+a_7) = a_4+a_7+a_10 = 6")
        assert not equations_equivalent(chain, "2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 7")
        # a chain and a single equation are different claim counts
        assert not equations_equivalent(chain, "a_4 + a_7 + a_10 = 6")

    def test_derivative_forms(self):
        assert equations_equivalent("d/dx(x**3 + 2*x)", "Derivative(2*x + x**3, x)")
        assert not equations_equivalent("Derivative(x**3, x)", "Derivative(x**2, x)")

    def test_derivative_sum_notation_not_swallowed(self):
        # a greedy d/dx(...) rewrite would capture through the middle ')'
        assert equations_equivalent(
            "d/dx(x^3) + d/dx(2x)", "Derivative(x**3, x) + Derivative(2*x, x)"
        )

    def test_schoolbook_prime_notation_is_parseable(self):
        assert equations_same_form("g'(x) = 3*x", "g′(x) = 3*x")
        assert not equations_same_form("g'(x) = 3*x", "g'(x) = 2*x")

    def test_derivative_substitution_uses_explicit_function_definition(self):
        student = "g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*(2*x - 4)"
        reference = "g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"
        definitions = [
            "f(x) = x**2 - 4*x - 3",
            "g(x) = (x**3 - 2*x)*f(x)",
        ]

        assert not equations_same_form(student, reference)
        assert equations_same_form_with_derivatives(student, reference, definitions)
        assert not equations_same_form_with_derivatives(
            student.replace("2*x - 4", "2*x + 4"), reference, definitions
        )
        assert not equations_same_form_with_derivatives(
            "f'(x) = f'(x)", "f'(x) = 2*x - 4", definitions
        )

    def test_same_form_distinguishes_rearrangement_steps(self):
        # equivalent equations, but different written steps
        assert not equations_same_form("3*x + 5 = 20", "3*x = 15")
        assert not equations_same_form("3*x = 15", "x = 5")
        # same step, cosmetic differences
        assert equations_same_form("3x = 15", "3*x = 15")
        assert equations_same_form("15 = 3*x", "3*x = 15")
        assert equations_same_form("x = 5", "x = 5.0")

    def test_strict_mode_rejects_scalar_multiple(self):
        assert not equations_equivalent("6*x + 10 = 40", "3*x + 5 = 20", allow_scale=False)
        # ...but the SAME equation with swapped sides is still exact
        assert equations_equivalent("20 = 3*x + 5", "3*x + 5 = 20", allow_scale=False)

    def test_expression_vs_equation_mismatch(self):
        assert not equations_equivalent("3*x + 5", "3*x + 5 = 0")


class TestTemplateMatch:
    def test_linear(self):
        assert match_template("4*x + 1 = 13", "a*x + b = c", ["a", "b", "c"]) == {
            "a": "4",
            "b": "1",
            "c": "13",
        }

    def test_linear_negative_b(self):
        m = match_template("2*x - 7 = 3", "a*x + b = c", ["a", "b", "c"])
        assert m == {"a": "2", "b": "-7", "c": "3"}

    def test_quadratic_with_zero_b(self):
        m = match_template("x**2 - 9 = 0", "x**2 + b*x + c = 0", ["b", "c"])
        assert m == {"b": "0", "c": "-9"}

    def test_derivative_power(self):
        m = match_template("Derivative(7*x**3, x)", "Derivative(a*x**n, x)", ["a", "n"])
        assert m == {"a": "7", "n": "3"}

    def test_no_match_nonnumeric(self):
        assert match_template("y*x + 1 = 13", "a*x + b = c", ["a", "b", "c"]) is None

    def test_no_match_wrong_shape(self):
        assert match_template("x**2 - 9 = 0", "a*x + b = c", ["a", "b", "c"]) is None


def test_instantiate():
    assert instantiate("a*x = c - b", {"a": "4", "b": "1", "c": "13"}) == "4*x = 12"


class TestAnswers:
    def test_compute_scalar(self):
        assert compute_answer("4*x + 1 = 13", "SCALAR") == "3"

    def test_compute_root_set(self):
        assert compute_answer("x**2 - 5*x + 6 = 0", "ROOT_SET") == ["2", "3"]

    def test_compute_expression(self):
        assert compute_answer("Derivative(5*x**2, x)", "EXPRESSION") == "10*x"

    def test_verify_scalar(self):
        assert verify_answer(["3*x + 5 = 20"], "SCALAR", "5")
        assert not verify_answer(["3*x + 5 = 20"], "SCALAR", "4")

    def test_verify_scalar_rejects_partial_root(self):
        # 3 satisfies x**2-9=0 but is not the complete answer
        assert not verify_answer(["x**2 - 9 = 0"], "SCALAR", "3")

    def test_verify_root_set(self):
        assert verify_answer(["x**2 - 9 = 0"], "ROOT_SET", ["-3", "3"])
        assert not verify_answer(["x**2 - 9 = 0"], "ROOT_SET", ["3"])

    def test_verify_root_set_decimal_forms(self):
        assert verify_answer(["x**2 - 9 = 0"], "ROOT_SET", ["3.0", "-3.0"])

    def test_verify_expression(self):
        assert verify_answer(["Derivative(x**3 + 2*x, x)"], "EXPRESSION", "3*x**2 + 2")
        assert not verify_answer(["Derivative(x**3 + 2*x, x)"], "EXPRESSION", "3*x**3 + 2")


def test_expressions_equivalent_rewrites():
    assert expressions_equivalent("2*x + 3*x**2", "3*x**2 + 2*x")
    assert not expressions_equivalent("2*x + 3*x**2", "3*x**2 + 2")
