"""Seed the knowledge DB from seeds/*.json — sympy-verifying every answer first.

A seed whose answer fails symbolic verification is REJECTED (never inserted as
verified), keeping spec rule 1 honest: the DB contains only verified knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path

from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.mathnorm import normalize_text, verify_answer
from tutor.knowledge.models import (
    Answer,
    HintTemplate,
    Misconception,
    Problem,
    ReferenceSolution,
    SolutionStep,
    Template,
)

SEEDS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "seeds"


def seed_database(db: KnowledgeDB, seeds_dir: Path = SEEDS_DIR) -> list[tuple[str, bool, str]]:
    """Returns a report: (seed id, verified?, reason)."""
    from tutor.knowledge.matching import problem_hash
    from tutor.vision.recognizer import Recognition

    report: list[tuple[str, bool, str]] = []

    for c in json.loads((seeds_dir / "concepts.json").read_text(encoding="utf-8")):
        db.insert_concept(c["id"], c["name"])
        report.append((c["id"], True, "concept"))

    for t in json.loads((seeds_dir / "templates.json").read_text(encoding="utf-8")):
        db.insert_template(Template.model_validate(t))
        report.append((t["id"], True, "template"))

    for path in sorted((seeds_dir / "problems").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        answer = Answer.model_validate(raw["answer"])
        ok = verify_answer(raw["equations"], answer.kind, answer.value)
        if not ok:
            report.append((raw["id"], False, "REJECTED: answer failed sympy verification"))
            continue
        problem = Problem.model_validate({k: v for k, v in raw.items() if k != "solution"})
        rec = Recognition(
            problem_text=problem.problem_text, equations=problem.equations
        )
        db.insert_problem(
            problem,
            normalized_text=normalize_text(problem.problem_text),
            text_hash=problem_hash(rec),
            verified=True,
        )
        solution = ReferenceSolution(
            steps=[SolutionStep.model_validate(s) for s in raw["solution"]["steps"]],
            final_answer=answer,
            concepts=problem.concepts,
            verified=True,
            origin="db",
        )
        db.insert_solution(problem.id, solution, verified=True)
        report.append((problem.id, True, "problem + solution verified"))

    for m in json.loads(
        (seeds_dir / "pedagogy" / "misconceptions.json").read_text(encoding="utf-8")
    ):
        db.insert_misconception(Misconception.model_validate(m))
        report.append((m["id"], True, "misconception"))

    for h in json.loads(
        (seeds_dir / "pedagogy" / "hint_templates.json").read_text(encoding="utf-8")
    ):
        db.insert_hint_template(HintTemplate.model_validate(h))
        report.append((h["id"], True, "hint template"))

    return report


def main() -> None:
    from tutor.config import load_settings

    settings = load_settings()
    db = KnowledgeDB(settings.db_path)
    report = seed_database(db)
    rejected = [r for r in report if not r[1]]
    print(f"seeded {settings.db_path}:")
    for seed_id, ok, reason in report:
        print(f"  {'OK ' if ok else 'FAIL'} {seed_id:<24} {reason}")
    print(f"{len(report) - len(rejected)} entries verified, {len(rejected)} rejected")


if __name__ == "__main__":
    main()
