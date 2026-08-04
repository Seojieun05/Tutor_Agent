from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import Tier
from tutor.vision.recognizer import Recognition


def rec(
    text: str,
    equations: list[str],
    work: list[str] | None = None,
    problem_type: str = "unknown",
    concepts: list[str] | None = None,
) -> Recognition:
    """A Recognition as it reaches the Matcher: already tagged by the
    ConceptTagger (the matcher no longer infers tags itself)."""
    return Recognition(
        problem_text=text,
        equations=equations,
        student_work=work or [],
        problem_type=problem_type,
        concepts=concepts or [],
    )


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


def test_hash_ignores_tags():
    """Tags describe the problem, so they must not change its identity."""
    untagged = rec("문제", ["3*x + 5 = 20"])
    tagged = rec("문제", ["3*x + 5 = 20"], problem_type="linear_equation",
                 concepts=["linear_equation"])
    assert problem_hash(untagged) == problem_hash(tagged)


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
        m = Matcher(db).match(
            rec("미분하시오", ["Derivative(7*x**3, x)"], problem_type="derivative")
        )
        assert m.tier == Tier.TEMPLATE
        assert m.reference.final_answer.kind == "EXPRESSION"
        assert m.reference.final_answer.value == "21*x**2"

    def test_template_narrowed_by_problem_type(self, db):
        """A wrong-type template must not be tried before the right one."""
        m = Matcher(db).match(
            rec("일차방정식", ["4*x + 1 = 13"], problem_type="linear_equation")
        )
        assert m.tier == Tier.TEMPLATE
        assert m.bindings == {"a": "4", "b": "1", "c": "13"}

    def test_concept_tier_uses_the_tagged_concepts(self, db):
        m = Matcher(db).match(
            rec(
                "이차방정식을 풀어라",
                ["2*x**2 - 8 = 0"],
                problem_type="quadratic_equation",
                concepts=["quadratic_equation"],
            )
        )
        assert m.tier == Tier.CONCEPT
        assert m.reference is None
        assert "quadratic_equation" in m.concepts

    def test_untagged_problem_cannot_reach_the_concept_tier(self, db):
        """No tags, no concept retrieval — the matcher never guesses them."""
        m = Matcher(db).match(rec("이차방정식을 풀어라", ["2*x**2 - 8 = 0"]))
        assert m.tier == Tier.NEW

    def test_new_on_out_of_domain(self, db):
        m = Matcher(db).match(rec("사과 3개와 배 2개의 가격 비교", []))
        assert m.tier == Tier.NEW
        assert m.reference is None


class FakeSemantic:
    def __init__(self, hits):
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query, limit=5):
        self.queries.append(query)
        return self.hits[:limit]


class TestSemanticTier:
    def test_semantic_catches_what_concepts_miss(self, db):
        """The imported corpus is tagged with curriculum codes, not whitelist
        ids, so text similarity is the only way to reach it."""
        stored = db.all_problems()[0]
        semantic = FakeSemantic([(stored, 0.93)])
        m = Matcher(db, semantic=semantic).match(
            rec("비슷하게 생긴 새로운 문제", [], problem_type="word_problem")
        )
        assert m.tier == Tier.SEMANTIC
        assert m.problem.id == stored.id
        assert m.reference is None  # similar wording is not a verified solution
        assert semantic.queries == ["비슷하게 생긴 새로운 문제"]

    def test_weak_similarity_falls_through_to_new(self, db):
        semantic = FakeSemantic([(db.all_problems()[0], 0.42)])
        m = Matcher(db, semantic=semantic).match(rec("전혀 다른 문제", []))
        assert m.tier == Tier.NEW

    def test_exact_still_wins_over_semantic(self, db):
        semantic = FakeSemantic([(db.all_problems()[0], 0.99)])
        m = Matcher(db, semantic=semantic).match(rec("일차방정식", ["3x + 5 = 20"]))
        assert m.tier == Tier.EXACT
        assert semantic.queries == []  # never even consulted

    def test_broken_index_does_not_break_matching(self, db):
        class Exploding:
            def search(self, query, limit=5):
                raise RuntimeError("index missing")

        m = Matcher(db, semantic=Exploding()).match(rec("아무 문제", []))
        assert m.tier == Tier.NEW
