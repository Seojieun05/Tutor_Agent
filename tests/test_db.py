import sqlite3

from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import ReferenceSolution, SolutionStep, Answer


def test_seeded_content(db):
    from tutor.knowledge.concepts import ALLOWED_CONCEPT_IDS

    assert len(db.all_problems()) == 6
    # seeds/concepts.json is the concept whitelist: all of it lands in the DB,
    # so retrieval and the tagger agree on the same vocabulary
    assert set(db.concepts()) == set(ALLOWED_CONCEPT_IDS)
    assert {"linear_equation", "quadratic_equation", "differentiation"} <= set(db.concepts())
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


def test_prewritten_hint_stores_speech_and_board_together(db):
    db.save_prewritten_hint(
        "p", 2, 2, "다음과 같은 꼴로 나타낼 수 있어요.",
        [{"expr": "y - y_1 = -2*(x - x_1)", "note": "점-기울기형"}],
    )

    artifact = db.prewritten_hint_artifact("p", 2, 2)

    assert artifact is not None
    assert artifact.text == "다음과 같은 꼴로 나타낼 수 있어요."
    assert [(b.expr, b.note) for b in artifact.board] == [
        ("y - y_1 = -2*(x - x_1)", "점-기울기형")
    ]
    # Text-only callers such as the forward-invitation path remain compatible.
    assert db.prewritten_hint("p", 2, 2) == artifact.text


def test_existing_prewritten_hint_table_gains_an_empty_board(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE prewritten_hints ("
        "problem_id TEXT NOT NULL, step INTEGER NOT NULL, level INTEGER NOT NULL, "
        "hint_text TEXT NOT NULL, PRIMARY KEY (problem_id, step, level))"
    )
    conn.execute("INSERT INTO prewritten_hints VALUES ('old', 1, 1, '기존 힌트')")
    conn.commit()
    conn.close()

    migrated = KnowledgeDB(path)
    artifact = migrated.prewritten_hint_artifact("old", 1, 1)

    assert artifact is not None
    assert artifact.text == "기존 힌트"
    assert artifact.board == ()
