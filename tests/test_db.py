from tutor.knowledge.models import ReferenceSolution, SolutionStep, Answer


def test_seeded_content(db):
    assert len(db.all_problems()) == 6
    assert set(db.concepts()) == {"linear_equation", "quadratic_equation", "differentiation"}
    assert len(db.templates()) == 3


def test_verified_solution_lookup(db):
    solution = db.verified_solution("lin_001")
    assert solution is not None
    assert solution.verified is True
    assert solution.steps[-1].expression == "x = 5"


def test_unverified_solution_not_returned(db):
    grok = ReferenceSolution(
        steps=[SolutionStep(idx=1, description="d", expression="x = 1")],
        final_answer=Answer(kind="SCALAR", value="1"),
        origin="grok",
        verified=True,  # even if the model claims it
    )
    db.insert_unverified_solution("some_new_problem", grok)
    assert db.verified_solution("some_new_problem") is None


def test_hint_templates_misconception_first(db):
    hints = db.hint_templates_for(["linear_equation"], "sign_flip_on_move", level=1)
    assert hints[0].misconception_id == "sign_flip_on_move"
    generic = db.hint_templates_for(["linear_equation"], None, level=1)
    assert all(h.misconception_id is None for h in generic)


def test_misconceptions_for_concaccording(db):
    ids = {m.id for m in db.misconceptions_for(["differentiation"])}
    assert ids == {"exponent_not_decremented", "constant_term_kept"}
