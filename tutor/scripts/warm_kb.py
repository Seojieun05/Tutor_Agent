"""Fill the knowledge DB ahead of a lesson, so the first turn is not the slow one.

    python -m tutor.scripts.warm_kb                 # concept lines + their audio
    python -m tutor.scripts.warm_kb --solve 12      # ...and pre-solve a problem

Two kinds of warming, both writing to data/knowledge.db (which survives
restarts) and data/tts_cache:

  concept lines   "등비수열 문제군요. 첫째항과 공비의 조건을 먼저 확인하셨나요?"
                  The tutor writes one per concept the first time it meets it;
                  doing it here means the FIRST problem of a kind already has
                  the good line instead of the generic opening.

  a solved problem  A problem stored with a machine-checked solution matches
                  EXACT, so the solver never runs for it and a work check can
                  be graded the moment the photo is read. Worth it for a
                  problem you know you are about to show.

THIS SPENDS REAL API CALLS: one per concept, plus TTS for each line.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from tutor.config import load_settings
from tutor.console import say, soften_stdout
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.mathnorm import normalize_text, verify_answer
from tutor.knowledge.models import Answer, Problem, ReferenceSolution

# The high-school and 수능 range: what a demo or a study session actually
# lands on. Grade-school concepts stay unwarmed — the tutor writes their line
# the first time it meets one, which is the mechanism working as designed.
COMMON_CONCEPTS = [
    # 수열
    "sequence", "arithmetic_sequence", "geometric_sequence", "sequence_sum",
    "sigma_notation", "mathematical_induction", "sequence_limit", "infinite_series",
    # 식과 방정식
    "polynomial", "polynomial_operations", "factorization", "remainder_theorem",
    "factor_theorem", "linear_equation", "quadratic_equation", "higher_degree_equation",
    "simultaneous_linear_equations", "quadratic_inequality", "absolute_value",
    # 함수
    "function", "linear_function", "quadratic_function", "composite_function",
    "inverse_function", "rational_function", "radical_function",
    # 지수·로그
    "exponent", "logarithm", "exponential_function", "logarithmic_function",
    # 삼각
    "trigonometric_ratio", "trigonometric_function", "sine_rule", "cosine_rule",
    # 미적분
    "function_limit", "continuity", "differentiation", "derivative_rules",
    "derivative_applications", "indefinite_integral", "definite_integral",
    "integral_applications", "fundamental_theorem_of_calculus",
    # 확률과 통계
    "counting_principle", "permutation", "combination", "probability",
    "conditional_probability", "independent_events", "random_variable",
    "binomial_distribution", "normal_distribution", "statistical_estimation",
    # 기하
    "coordinate_geometry", "line_equation", "circle_equation", "pythagorean_theorem",
    "similarity", "vector", "dot_product",
    # 논리
    "set", "set_operations", "proposition", "necessary_sufficient_condition",
]

# Problems worth having solved before the lesson starts, by the name you call
# them. Equations are what EXACT matching keys on, so they must be written the
# way the VLM reads them off the page.
PRESOLVE = {
    "12": {
        "problem_type": "geometric_sequence",
        "problem_text": (
            "12. 등비수열 {a_n}이 2(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6 을 "
            "만족시킬 때, a_10의 값은? [4점]"
        ),
        # EXACT matching compares equation lists pairwise and demands the same
        # LENGTH, and this problem is read both ways in practice: the chain as
        # one line, or split at the middle equals. Both are the same problem
        # and both are stored, or the demo depends on which way the VLM felt.
        "equations": [
            ["2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10 = 6"],
            ["2*(a_1 + a_4 + a_7) = a_4 + a_7 + a_10", "a_4 + a_7 + a_10 = 6"],
        ],
        "concepts": ["geometric_sequence"],
        "answer": Answer(kind="SCALAR", value="24/7"),
    },
}


def build(settings):
    from tutor.hints.generator import HintGenerator
    from tutor.server.app import build_shared, wrap_with_cache
    from tutor.speech.tts import XaiSpeaker

    db = KnowledgeDB(settings.db_path)
    shared = build_shared(settings)
    speaker = (
        wrap_with_cache(settings, XaiSpeaker(settings))
        if not settings.echo_mode
        else None
    )
    return db, shared.solve_llm, HintGenerator(shared.hint_llm, db), speaker


def warm_concepts(db, gen, speaker, concept_ids: list[str], workers: int = 5) -> list[str]:
    todo = []
    for cid in concept_ids:
        name = db.concept_name(cid)
        if name is None:
            say(f"  ? {cid}: not a concept in this DB, skipped")
        elif db.preflight_line(cid):
            say(f"  = {cid}: already written")
        else:
            todo.append((cid, name))
    if not todo:
        return []

    say(f"writing {len(todo)} concept lines...")
    written: list[str] = []

    def one(item):
        cid, name = item
        try:
            return cid, gen.write_preflight(name)
        except Exception as e:  # noqa: BLE001 — one bad concept is not the run
            say(f"  ! {cid}: {e}")
            return cid, ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, line in pool.map(one, todo):
            if not line:
                continue
            db.save_preflight_line(cid, line)
            written.append(line)
            say(f"  + {cid}: {line}")

    if speaker is not None and written:
        say(f"rendering {len(written)} lines to the TTS cache...")
        speaker.register(*written)
        ready = speaker.warm(written)
        say(f"  {ready} rendered (hits {speaker.hits}, new {speaker.misses})")
    return written


def presolve(db, llm, key: str) -> bool:
    from tutor.knowledge.matching import problem_hash
    from tutor.solver.grok_solver import GrokSolver
    from tutor.vision.recognizer import Recognition

    spec = PRESOLVE.get(key)
    if spec is None:
        say(f"no pre-solve entry named {key!r}; known: {sorted(PRESOLVE)}")
        return False

    variants = spec["equations"]
    rec = Recognition(
        problem_text=spec["problem_text"],
        equations=variants[0],
        problem_type=spec["problem_type"],
        concepts=spec["concepts"],
    )
    say(f"solving {key}...")
    solution = GrokSolver(llm, db).solve(rec, f"presolved-{key}")
    say(f"  answer: {solution.final_answer.kind} {solution.final_answer.value}")
    for step in solution.steps:
        say(f"    {step.idx}. {step.description} | {step.expression}")

    # The gate is agreement with the answer written down here, not the sympy
    # check: substituting into a chain equality cannot confirm this kind of
    # problem, and a check that cannot run is not a verdict.
    expected = spec["answer"]
    if (solution.final_answer.kind, str(solution.final_answer.value)) != (
        expected.kind, str(expected.value)
    ):
        say(f"  ! solver says {solution.final_answer.value}, expected {expected.value}"
            " — NOT stored. Fix the expected answer or re-run.")
        return False
    if verify_answer(variants[0], expected.kind, expected.value):
        say("  sympy agrees with the answer too")

    reference = ReferenceSolution(
        steps=solution.steps, final_answer=expected,
        concepts=spec["concepts"], verified=True, origin="db",
    )
    for i, equations in enumerate(variants):
        pid = f"presolved-{key}" if i == 0 else f"presolved-{key}-v{i}"
        shape = Recognition(
            problem_text=spec["problem_text"], equations=equations,
            problem_type=spec["problem_type"], concepts=spec["concepts"],
        )
        db.insert_problem(
            Problem(
                id=pid,
                problem_type=spec["problem_type"],
                problem_text=spec["problem_text"],
                equations=equations,
                answer=expected,
                source="warm_kb",
                verified=True,
                concepts=spec["concepts"],
            ),
            normalized_text=normalize_text(spec["problem_text"]),
            text_hash=problem_hash(shape),
            verified=True,
        )
        db.insert_solution(pid, reference, verified=True)
        say(f"  stored {pid}: {len(equations)} equation(s)")
    say(f"  {key} is verified KB now — the solver will not run for it again")
    return True


def main(argv: list[str] | None = None) -> int:
    soften_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", action="append", default=[],
                    help=f"pre-solve a known problem ({sorted(PRESOLVE)})")
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="concept ids to write lines for (default: the common set)")
    args = ap.parse_args(argv)

    settings = load_settings()
    if settings.echo_mode:
        say("echo mode: no API key loaded, nothing to warm")
        return 1
    db, llm, gen, speaker = build(settings)

    warm_concepts(db, gen, speaker,
                  args.concepts if args.concepts is not None else COMMON_CONCEPTS)
    for key in args.solve:
        presolve(db, llm, key)
    say(f"knowledge DB: {settings.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
