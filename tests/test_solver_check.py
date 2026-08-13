from tutor.knowledge.models import ReferenceSolution
from tutor.llm.echo import EchoLLMClient
from tutor.solver.grok_solver import GrokSolver
from tutor.vision.recognizer import Recognition

REC = Recognition(problem_text="풀어라: 3x + 5 = 20", equations=["3*x + 5 = 20"])

GOOD = {
    "steps": [
        {"idx": 1, "description": "이항", "expression": "3*x = 15"},
        {"idx": 2, "description": "나눗셈", "expression": "x = 5"},
    ],
    "final_answer": {"kind": "SCALAR", "value": "5"},
    "concepts": ["linear_equation"],
    "verified": True,  # the model lies; the solver must override
    "origin": "grok",
}

BAD = dict(GOOD, final_answer={"kind": "SCALAR", "value": "4"})


UNNUMBERED = {
    "steps": [
        {"description": "이항", "expression": "3*x = 15"},
        {"description": "나눗셈", "expression": "x = 5"},
    ],
    "final_answer": {"kind": "SCALAR", "value": "5"},
}


def test_steps_the_model_forgot_to_number_are_numbered_by_position(db):
    """A step's index IS its position, and a solver that omits it has not
    made a mistake worth throwing six good steps away for — which is what
    happened when the whole solution died of `idx: Field required`."""
    llm = EchoLLMClient({"solve": [UNNUMBERED]})
    solution = GrokSolver(llm, db).solve(REC, "hash-3")
    assert [s.idx for s in solution.steps] == [1, 2]  # 1-based, as everything downstream reads


def test_numbering_the_model_did_give_is_left_alone(db):
    """A solution that numbers its own steps is saying something: 1, 3, 7 is
    not this validator's business to renumber."""
    sparse = dict(UNNUMBERED, steps=[
        {"idx": 1, "description": "이항", "expression": "3*x = 15"},
        {"idx": 7, "description": "나눗셈", "expression": "x = 5"},
    ])
    llm = EchoLLMClient({"solve": [sparse]})
    solution = GrokSolver(llm, db).solve(REC, "hash-4")
    assert [s.idx for s in solution.steps] == [1, 7]


def test_machine_checked_solution_stays_unverified(db):
    llm = EchoLLMClient({"solve": [GOOD]})
    solution = GrokSolver(llm, db).solve(REC, "hash-1")
    assert isinstance(solution, ReferenceSolution)
    assert solution.verified is False  # spec: Grok output never auto-verified
    assert solution.origin == "grok"
    # stored as an unverified candidate only
    assert db.verified_solution("hash-1") is None


def test_failed_machine_check_not_stored(db):
    llm = EchoLLMClient({"solve": [BAD]})
    solution = GrokSolver(llm, db).solve(REC, "hash-2")
    assert solution.verified is False
    rows = db._conn.execute(
        "SELECT COUNT(*) FROM solutions WHERE problem_id = ?", ("hash-2",)
    ).fetchone()[0]
    assert rows == 0
