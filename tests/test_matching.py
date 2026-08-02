from tutor.knowledge.matching import Matcher, problem_hash, tag_concepts
from tutor.knowledge.models import Tier
from tutor.vision.recognizer import Recognition


def rec(text: str, equations: list[str], work: list[str] | None = None) -> Recognition:
    return Recognition(problem_text=text, equations=equations, student_work=work or [])


def test_hash_excludes_student_work():
    a = rec("풀어라: 3x + 5 = 20", ["3*x + 5 = 20"], work=[])
    b = rec("풀어라: 3x + 5 = 20", ["3*x + 5 = 20"], work=["3*x = 15", "x = 5"])
    assert problem_hash(a) == problem_hash(b)


def test_hash_includes_choices_and_diagram():
    a = rec("문제", ["3*x + 5 = 20"])
    b = Recognition(problem_text="문제", equations=["3*x + 5 = 20"], choices=["1) 5", "2) 6"])
    c = Recognition(
        problem_text="문제", equations=["3*x + 5 = 20"], diagram_conditions=["각 B = 90"]
    )
    assert len({problem_hash(a), problem_hash(b), problem_hash(c)}) == 3


def test_tagger():
    assert tag_concepts(rec("풀어라", ["3*x + 5 = 20"])) == ("linear_equation", {"linear_equation"})
    assert tag_concepts(rec("풀어라", ["x**2 - 5*x + 6 = 0"]))[0] == "quadratic_equation"
    assert tag_concepts(rec("미분하시오", ["Derivative(x**3, x)"]))[0] == "derivative"


class TestCascade:
    def test_exact_by_equivalence(self, db):
        m = Matcher(db).match(rec("일차방정식 문제", ["3x + 5 = 20"]))
        assert m.tier == Tier.EXACT
        assert m.problem.id == "lin_001"
        assert m.reference.verified is True
        assert m.reference.origin == "db"
        assert m.reference.final_answer.value == "5"

    def test_exact_swapped_sides_still_exact(self, db):
        m = Matcher(db).match(rec("일차방정식", ["20 = 3*x + 5"]))
        assert m.tier == Tier.EXACT
        assert m.problem.id == "lin_001"

    def test_scalar_multiple_is_template_not_exact(self, db):
        # 6x+10=40 is lin_001 scaled by 2: its parameters differ, so reusing
        # lin_001's stored steps/hints (subtract 5, divide by 3) would be wrong
        m = Matcher(db).match(rec("일차방정식", ["6*x + 10 = 40"]))
        assert m.tier == Tier.TEMPLATE
        assert m.bindings == {"a": "6", "b": "10", "c": "40"}
        assert m.reference.final_answer.value == "5"

    def test_template_on_unseen_equation(self, db):
        m = Matcher(db).match(rec("일차방정식을 풀어라", ["4*x + 1 = 13"]))
        assert m.tier == Tier.TEMPLATE
        assert m.bindings == {"a": "4", "b": "1", "c": "13"}
        assert m.reference.verified is True
        assert m.reference.origin == "template"
        assert m.reference.final_answer.value == "3"
        assert m.reference.steps[0].expression == "4*x = 12"

    def test_template_derivative(self, db):
        m = Matcher(db).match(rec("미분하시오", ["Derivative(7*x**3, x)"]))
        assert m.tier == Tier.TEMPLATE
        assert m.reference.final_answer.kind == "EXPRESSION"
        assert m.reference.final_answer.value == "21*x**2"

    def test_concept_on_non_monic_quadratic(self, db):
        m = Matcher(db).match(rec("이차방정식을 풀어라", ["2*x**2 - 8 = 0"]))
        assert m.tier == Tier.CONCEPT
        assert m.reference is None
        assert "quadratic_equation" in m.concepts

    def test_new_on_out_of_domain(self, db):
        m = Matcher(db).match(rec("사과 3개와 배 2개의 가격 비교", []))
        assert m.tier == Tier.NEW
        assert m.reference is None
