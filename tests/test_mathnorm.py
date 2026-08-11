from tutor.knowledge.mathnorm import (
    compute_answer,
    equations_equivalent,
    equations_same_form,
    expressions_equivalent,
    instantiate,
    match_template,
    normalize_text,
    verify_answer,
)


def test_normalize_text():
    assert normalize_text("다음을  푸시오: 3×x + 5 = 20!") == normalize_text(
        "다음을 푸시오: 3*x + 5 = 20"
    )


def test_normalize_text_superscripts_survive_nfkc():
    # NFKC alone would fold '²' into a bare '2' ("x2"); the math map must win
    assert normalize_text("x² - 9 = 0") == normalize_text("x**2 - 9 = 0")


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
