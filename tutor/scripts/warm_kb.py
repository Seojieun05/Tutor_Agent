"""Fill the knowledge DB ahead of a lesson, so the first turn is not the slow one.

    python -m tutor.scripts.warm_kb                 # concept lines + their audio
    python -m tutor.scripts.warm_kb --solve 12      # ...and pre-store a problem
    python -m tutor.scripts.warm_kb --solve all --concepts   # every entry, no lines

Two kinds of warming, both writing to data/knowledge.db (which survives
restarts) and data/tts_cache:

  concept lines   "등비수열 문제군요. 첫째항과 공비의 조건을 먼저 확인하셨나요?"
                  The tutor writes one per concept the first time it meets it;
                  doing it here means the FIRST problem of a kind already has
                  the good line instead of the generic opening.

  a stored problem  A problem stored with a checked solution matches EXACT, so
                  the solver never runs for it and a work check can be graded
                  the moment the photo is read. Worth it for a problem you
                  know you are about to show.

The problems themselves live in data/presolve.json — next to the database they
are loaded into, not in this file. They are lesson material, like the DB and
the captures: what the tutor is being prepared for on one machine is not part
of the program. An entry:

    {"problem_type": "...", "problem_text": "...",
     "equations": [["..."], ["...", "..."]],      // one list per VLM reading
     "concepts": ["..."],
     "answer": {"kind": "SCALAR", "value": "24/7"},
     "steps": [{"description": "...", "expression": "..."}, ...]}   // optional

With "steps" the entry is CURATED: the solution is stored as written (spending
no API call) and the answer is the entry's own. Without them the solver writes
the steps and the run refuses to store unless the solver's answer agrees with
the entry's. The optional top-level "hint_templates" list seeds concept-level
hint lines the tutor can speak without any model call:

    {"id": "...", "concept_id": "...", "level": 1, "template_text": "... {step} ..."}

Concept lines and TTS SPEND REAL API CALLS; curated problems do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tutor.config import PROJECT_ROOT, load_settings
from tutor.console import say, soften_stdout
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.mathnorm import normalize_text, verify_answer
from tutor.knowledge.models import (
    Answer,
    HintTemplate,
    Problem,
    ReferenceSolution,
    SolutionStep,
)

PRESOLVE_PATH = PROJECT_ROOT / "data" / "presolve.json"

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

def load_presolve() -> dict:
    """The lesson material, from beside the DB it fills. Missing file → {}."""
    if not PRESOLVE_PATH.exists():
        return {}
    data = json.loads(PRESOLVE_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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


def presolve(db, llm, key: str, entries: dict) -> bool:
    from tutor.knowledge.matching import problem_hash
    from tutor.vision.recognizer import Recognition

    spec = entries.get(key)
    if spec is None:
        say(f"no pre-solve entry named {key!r}; known: {sorted(entries)}")
        return False

    variants = spec["equations"]
    expected = Answer.model_validate(spec["answer"])

    if spec.get("steps"):
        # CURATED: the steps were written by hand from a worked solution, so
        # there is nothing to ask a model and nothing that can disagree.
        steps = [
            SolutionStep(idx=i + 1, **s) for i, s in enumerate(spec["steps"])
        ]
        say(f"storing {key} (curated, {len(steps)} steps)...")
    else:
        from tutor.solver.grok_solver import GrokSolver

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
        # The gate is agreement with the answer written down here, not the
        # sympy check: substituting into a chain equality cannot confirm this
        # kind of problem, and a check that cannot run is not a verdict.
        if (solution.final_answer.kind, str(solution.final_answer.value)) != (
            expected.kind, str(expected.value)
        ):
            say(f"  ! solver says {solution.final_answer.value}, expected "
                f"{expected.value} — NOT stored. Fix the expected answer or re-run.")
            return False
        steps = solution.steps
    if verify_answer(variants[0], expected.kind, expected.value):
        say("  sympy agrees with the answer too")

    reference = ReferenceSolution(
        steps=steps, final_answer=expected,
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
        # replace, not append: re-running after editing the entry must leave
        # ONE solution, or verified_solution picks whichever row it meets first
        db._conn.execute("DELETE FROM solutions WHERE problem_id = ?", (pid,))
        db.insert_solution(pid, reference, verified=True)
        # Hints written against OLD steps are stale the moment the steps
        # change. Curated per-problem lines are restored immediately; --hints
        # may fill every remaining gap with model-written lines later.
        db.clear_prewritten_hints(pid)
        for hint in spec.get("prewritten_hints", []):
            db.save_prewritten_hint(
                pid,
                int(hint["step"]),
                int(hint["level"]),
                str(hint["text"]),
                hint.get("board", []),
            )
        say(f"  stored {pid}: {len(equations)} equation(s)")
    say(f"  {key} is verified KB now — the solver will not run for it again")
    return True


def seed_hint_templates(db, entries: dict) -> int:
    """Concept-level hint lines from the presolve file, idempotently.

    A hint the DB can answer is a hint no model is asked to write — the
    template path in HintGenerator.generate runs before the phrase call, so
    every line seeded here turns a measured ~7s of phrasing into nothing.

    Quality gate at the door, not at speak time: an L1 that announces the
    step ("X 차례예요") is the navigation voice the generator would refuse
    anyway — the whole first batch of L1 templates was seeded in that shape
    and sat in the DB as dead rows, silently skipped on every turn. Refusing
    it HERE, loudly, keeps the file honest.
    """
    from tutor.hints.generator import announces_step

    n = 0
    for spec in entries.get("hint_templates", []):
        template = HintTemplate.model_validate(spec)
        if template.level == 1 and announces_step(template.template_text):
            say(f"  REFUSED {template.id}: an L1 must ask, not announce a step")
            continue
        db.insert_hint_template(template)
        n += 1
    if n:
        say(f"seeded {n} hint template(s)")
    return n


def prewrite_hints(db, gen, entries: dict, workers: int = 6) -> int:
    """L1/L2 for every step of every stored presolve problem, ahead of time.

    The live phrasing path with the clock removed (HintGenerator.prewrite):
    same LearnLM prompt, same model, same screens — run here so the lesson
    serves model-written lines at template price, and so every line can be
    READ before a student hears one (they print below). Idempotent: rows
    that already exist are kept; editing a problem's steps clears its rows
    (see presolve) and this refills them.
    """
    from tutor.vision.recognizer import Recognition

    jobs = []          # (key, rec, reference, step_idx, level) — one per line
    targets = {}       # key → every variant pid the line is stored under
    for key in sorted(k for k in entries if k != "hint_templates"):
        spec = entries[key]
        pids = [
            f"presolved-{key}" if i == 0 else f"presolved-{key}-v{i}"
            for i in range(len(spec.get("equations", [])))
        ]
        problem = next(
            (p for p in db.all_problems() if p.id == pids[0]), None
        ) if pids else None
        reference = db.verified_solution(pids[0]) if pids else None
        if problem is None or reference is None:
            say(f"  ? {key}: not stored yet (run --solve first), skipped")
            continue
        targets[key] = (problem, pids)
        rec = Recognition(
            problem_text=spec["problem_text"], equations=spec["equations"][0],
            problem_type=spec.get("problem_type", "unknown"),
            concepts=spec.get("concepts", []),
        )
        for step in reference.steps:
            for level in (1, 2):
                if db.prewritten_hint(pids[0], step.idx, level):
                    continue           # already written (and maybe hand-edited)
                jobs.append((key, rec, reference, step.idx, level))

    if not jobs:
        say("prewritten hints: nothing to write")
        return 0
    say(f"writing {len(jobs)} step hint(s)...")

    def one(job):
        key, rec, reference, step_idx, level = job
        problem, _pids = targets[key]
        try:
            return job, gen.prewrite(
                problem=problem, reference=reference, rec=rec,
                step_idx=step_idx, level=level,
            )
        except Exception as e:  # noqa: BLE001 — one bad line is not the run
            say(f"  ! {key} step{step_idx} L{level}: {e}")
            return job, None

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (key, _rec, _reference, step_idx, level), line in pool.map(one, jobs):
            if not line:
                say(f"  ! {key} step{step_idx} L{level}: screened out, ladder covers it")
                continue
            _problem, pids = targets[key]
            for pid in pids:
                db.save_prewritten_hint(pid, step_idx, level, line)
            n += 1
            say(f"  + {key} step{step_idx} L{level}: {line}")
    say(f"prewritten hints: {n} written")
    return n


def extend_semantic_index(db, stored_ids: list[str]) -> None:
    """Append the newly stored problems to the embedding index, if it exists.

    EXACT matching does not need this; it is the safety net underneath it —
    if the VLM's reading wobbles past every stored variant, the SEMANTIC tier
    should still surface the right problem instead of a middle-school
    lookalike. Appending is cheap (a handful of passages); a full rebuild is
    tutor.scripts.build_embeddings.
    """
    from tutor.retrieval.semantic import DEFAULT_INDEX_PATH

    if not DEFAULT_INDEX_PATH.exists() or not stored_ids:
        return
    try:
        import numpy as np
        from tutor.retrieval.semantic import MODEL_NAME, get_embedding_model
    except Exception as e:  # noqa: BLE001 — the index is optional, so is this
        say(f"  (semantic index not extended: {e})")
        return

    data = np.load(DEFAULT_INDEX_PATH)
    ids = data["ids"].astype(str)
    have = set(ids)
    todo = [pid for pid in stored_ids if pid not in have]
    if not todo:
        return
    by_id = {p.id: p for p in db.all_problems()}
    passages = []
    for pid in todo:
        p = by_id[pid]
        text = p.problem_text
        if p.equations:
            text += "\n수식: " + " ; ".join(p.equations)
        passages.append("passage: " + text)
    model = get_embedding_model(MODEL_NAME)
    fresh = model.encode(
        passages, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    np.savez(
        DEFAULT_INDEX_PATH,
        ids=np.concatenate([ids, np.array(todo)]),
        embeddings=np.concatenate([data["embeddings"].astype(np.float32), fresh]),
    )
    say(f"  semantic index: +{len(todo)} → {len(ids) + len(todo)} passages")


def main(argv: list[str] | None = None) -> int:
    soften_stdout()
    entries = load_presolve()
    known = sorted(k for k in entries if k != "hint_templates")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", action="append", default=[],
                    help=f"pre-store a problem from {PRESOLVE_PATH.name} "
                         f"({known or 'none found'}), or 'all'")
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="concept ids to write lines for (default: the common "
                         "set; pass with no ids to skip)")
    ap.add_argument("--hints", action="store_true",
                    help="pre-write L1/L2 lines for every stored presolve "
                         "problem step (spends API; idempotent)")
    args = ap.parse_args(argv)

    settings = load_settings()
    curated_only = bool(args.solve) and all(
        entries.get(k, {}).get("steps") for k in
        (known if "all" in args.solve else args.solve)
    )
    if settings.echo_mode and not curated_only:
        say("echo mode: no API key loaded, nothing to warm")
        return 1

    if curated_only and args.concepts == [] and not args.hints:
        # nothing here needs a model or a voice: skip the whole server build
        db, llm, gen, speaker = KnowledgeDB(settings.db_path), None, None, None
    else:
        db, llm, gen, speaker = build(settings)

    if gen is not None:
        warm_concepts(db, gen, speaker,
                      args.concepts if args.concepts is not None else COMMON_CONCEPTS)
    seed_hint_templates(db, entries)
    keys = known if "all" in args.solve else args.solve
    stored: list[str] = []
    for key in keys:
        if presolve(db, llm, key, entries):
            stored.append(f"presolved-{key}")
            stored.extend(
                f"presolved-{key}-v{i}"
                for i in range(1, len(entries[key]["equations"]))
            )
    if args.hints and gen is not None:
        prewrite_hints(db, gen, entries)
    extend_semantic_index(db, stored)
    say(f"knowledge DB: {settings.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
