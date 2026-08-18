from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import Answer, Problem, ReferenceSolution, SolutionStep, Tier
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

    def test_graph_labels_do_not_hide_a_verified_function_problem(self, db):
        base = rec(
            "함수 f와 g의 접선을 구하시오",
            ["f(x) = x**2 - 4*x - 3", "g(x) = (x**3 - 2*x)*f(x)"],
            problem_type="derivative_applications",
            concepts=["differentiation"],
        )
        problem = Problem(
            id="known-functions",
            problem_type="derivative_applications",
            problem_text=base.problem_text,
            equations=base.equations,
            answer=Answer(kind="SCALAR", value="49"),
            source="test",
            verified=True,
            concepts=["differentiation"],
        )
        db.insert_problem(
            problem,
            normalized_text=base.problem_text,
            text_hash=problem_hash(base),
            verified=True,
        )
        db.insert_solution(
            problem.id,
            ReferenceSolution(
                steps=[SolutionStep(idx=1, description="미분", expression="f'(1) = -2")],
                final_answer=problem.answer,
                concepts=problem.concepts,
                verified=True,
                origin="db",
            ),
            verified=True,
        )
        vision_read = base.model_copy(update={"equations": [
            "f(x) = x**2 - 4*x - 3",
            "y = f(x)",
            "g(x) = (x**3 - 2*x)*f(x)",
            "y = g(x)",
        ]})

        match = Matcher(db).match(vision_read)

        assert match.tier == Tier.EXACT
        assert match.problem.id == "known-functions"
        assert match.reference.verified is True

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


class TestTheProseIsTheIdentity:
    """Live problem 10: the VLM read the printed choices and its own equation
    list, every composite hash missed, and the match fell to SEMANTIC — which
    never carries a solution — so the verified reference sat on the shelf
    while a generic concept line misled the student. Identical normalized
    prose, with each stored equation found among the recognized ones, IS the
    same problem."""

    TEXT = (
        "10. 상수 a (a>1)에 대하여 곡선 y = aˣ - 2 위의 점 중 제1사분면에 있는 "
        "점 A를 지나고 y축에 평행한 직선이 x축과 만나는 점을 B라 하자. "
        "삼각형 AOC의 넓이가 8일 때, a × OB 의 값은? [4점]"
    )

    def stored(self, db, problem_id="presolved-t10"):
        from tutor.knowledge.mathnorm import normalize_text

        problem = Problem(
            id=problem_id, problem_type="exponential_function",
            problem_text=self.TEXT, equations=["y = a**x - 2"],
            answer=Answer(kind="SCALAR", value="2**(5/2)"),
            source="warm_kb", verified=True,
            concepts=["exponential_function"],
        )
        db.insert_problem(
            problem,
            normalized_text=normalize_text(self.TEXT),
            text_hash=problem_hash(
                rec(self.TEXT, ["y = a**x - 2"])
            ),
            verified=True,
        )
        db.insert_solution(problem_id, ReferenceSolution(
            steps=[SolutionStep(idx=1, description="점근선 확인",
                                expression="y = a**x - 2 의 점근선은 y = -2")],
            final_answer=problem.answer, concepts=problem.concepts,
            verified=True, origin="db",
        ), verified=True)

    def test_choices_and_extra_equations_cannot_hide_it(self, db):
        self.stored(db)
        live = Recognition(
            problem_text=self.TEXT,                    # the same printed prose
            equations=["y = a^x - 2", "AB = BC"],      # re-read, extra condition
            choices=["① 2^(13/6)", "② 2^(7/3)", "③ 2^(5/2)"],
        )
        m = Matcher(db).match(live)
        assert m.tier == Tier.EXACT
        assert m.problem.id == "presolved-t10"
        assert m.reference is not None and m.reference.verified is True

    def test_a_changed_number_in_the_prose_is_a_different_problem(self, db):
        # Same equations, one number changed in the prose (넓이 8 → 12): a
        # lookalike whose answer differs. The extra recognized condition keeps
        # the signature rung out, exactly like the live capture.
        self.stored(db)
        live = Recognition(
            problem_text=self.TEXT.replace("넓이가 8", "넓이가 12"),
            equations=["y = a^x - 2", "AB = BC"],
        )
        assert Matcher(db).match(live).tier != Tier.EXACT

    def test_short_prose_whose_equations_differ_stays_unmatched(self, db):
        # The parameters of this problem live in its EQUATION, not its prose:
        # identical wording around a different equation is a different problem.
        from tutor.knowledge.mathnorm import normalize_text

        problem = Problem(
            id="short-prose", problem_type="linear_equation",
            problem_text="다음 방정식을 풀어라.", equations=["3*x + 5 = 20"],
            answer=Answer(kind="SCALAR", value="5"), source="test",
            verified=True, concepts=["linear_equation"],
        )
        db.insert_problem(
            problem,
            normalized_text=normalize_text("다음 방정식을 풀어라."),
            text_hash="th-short-prose", verified=True,
        )
        db.insert_solution("short-prose", ReferenceSolution(
            steps=[SolutionStep(idx=1, description="이항", expression="3*x = 15")],
            final_answer=problem.answer, concepts=problem.concepts,
            verified=True, origin="db",
        ), verified=True)
        live = Recognition(
            problem_text="다음 방정식을 풀어라.", equations=["7*x - 2 = 12"]
        )
        m = Matcher(db).match(live)
        assert m.tier != Tier.EXACT


class TestASemanticHitVerifiesItself:
    """Live: problem 10 came back SEMANTIC at 0.976 with its verified
    reference on the shelf — the photographed prose was the stored prose in
    another rendering ("y=a^x-2" for "y = aˣ - 2", overline(AB) for AB), so
    no hash could see it, and a generic concept line took the first hint.
    Embeddings propose; rendering-blind prose identity plus the stored
    equations found among the recognized ones verifies; only then does the
    stored solution serve."""

    TYPED = (
        "10. 상수 a (a>1)에 대하여 곡선 y = aˣ - 2 위의 점 중 제1사분면에 있는 "
        "점 A를 지나고 y축에 평행한 직선이 x축과 만나는 점을 B, 곡선 y = aˣ - 2 의 "
        "점근선과 만나는 점을 C라 하자. AB = BC 이고 삼각형 AOC의 넓이가 8일 때, "
        "a × OB 의 값은? (단, O는 원점이다.) [4점]"
    )
    PHOTOGRAPHED = (
        "10. 상수 a(a>1)에 대하여 곡선 y=a^x-2 위의 점 중 제1사분면에 있는 "
        "점 A를 지나고 y축에 평행한 직선이 x축과 만나는 점을 B, 곡선 y=a^x-2의 "
        "점근선과 만나는 점을 C라 하자. AB=BC 이고 삼각형 AOC의 넓이가 8일 때, "
        "a×OB 의 값은? (단, O는 원점이다.) [4점]"
    )
    # the equation list exactly as the live recognizer returned it
    READ = [
        "y = a**x - 2",
        "overline(AB) = overline(BC)",
        "Area(AOC) = 8",
        "a * overline(OB)",
    ]

    def stored(self, db):
        from tutor.knowledge.mathnorm import normalize_text

        first = None
        for pid, eqs in [
            ("presolved-s10", ["y = a**x - 2", "AB = BC"]),
            ("presolved-s10-v1", ["y = a**x - 2"]),
        ]:
            problem = Problem(
                id=pid, problem_type="exponential_function",
                problem_text=self.TYPED, equations=eqs,
                answer=Answer(kind="SCALAR", value="2**(5/2)"),
                source="warm_kb", verified=True,
                concepts=["exponential_function"],
            )
            first = first or problem
            db.insert_problem(
                problem,
                normalized_text=normalize_text(self.TYPED),
                text_hash=f"th-{pid}", verified=True,
            )
            db.insert_solution(pid, ReferenceSolution(
                steps=[SolutionStep(idx=1, description="점근선 확인",
                                    expression="y = a**x - 2 의 점근선은 y = -2")],
                final_answer=problem.answer, concepts=problem.concepts,
                verified=True, origin="db",
            ), verified=True)
        return first

    def test_the_live_capture_now_serves_the_stored_solution(self, db):
        hit = self.stored(db)
        live = Recognition(
            problem_text=self.PHOTOGRAPHED, equations=list(self.READ),
            choices=["① 2^(13/6)", "② 2^(7/3)", "③ 2^(5/2)"],
            problem_type="exponential_function",
            concepts=["exponential_function", "logarithm"],
        )
        m = Matcher(db, semantic=FakeSemantic([(hit, 0.976)])).match(live)
        assert m.tier == Tier.EXACT
        assert m.problem.id.startswith("presolved-s10")
        assert m.reference is not None and m.reference.verified is True

    def test_a_lookalike_with_a_changed_number_stays_semantic(self, db):
        hit = self.stored(db)
        live = Recognition(
            problem_text=self.PHOTOGRAPHED.replace("넓이가 8", "넓이가 12"),
            equations=list(self.READ),
        )
        m = Matcher(db, semantic=FakeSemantic([(hit, 0.97)])).match(live)
        assert m.tier == Tier.SEMANTIC
        assert m.reference is None

    def test_prose_alone_cannot_promote_when_the_equations_disagree(self, db):
        # every stored variant demands its equations be seen in the capture:
        # same prose around an unreadable/absent curve equation stays proposal
        hit = self.stored(db)
        live = Recognition(problem_text=self.PHOTOGRAPHED, equations=[])
        m = Matcher(db, semantic=FakeSemantic([(hit, 0.97)])).match(live)
        assert m.tier == Tier.SEMANTIC
        assert m.reference is None

    def test_a_misread_syllable_still_promotes(self, db):
        # capture two, verbatim: printed 상수 arrived as 실수 (conf 1.00) and
        # the equation list carried the a > 1 condition instead of the area
        hit = self.stored(db)
        live = Recognition(
            problem_text=self.PHOTOGRAPHED.replace("상수", "실수"),
            equations=[
                "y = a**x - 2",
                "a > 1",
                "overline(AB) = overline(BC)",
                "a * overline(OB)",
            ],
        )
        m = Matcher(db, semantic=FakeSemantic([(hit, 0.963)])).match(live)
        assert m.tier == Tier.EXACT
        assert m.problem.id.startswith("presolved-s10")
        assert m.reference is not None and m.reference.verified is True


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
