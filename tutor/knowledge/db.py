"""SQLite-backed Domain Knowledge DB.

JSON files under seeds/ are the human-editable source of truth; seed_db.py
verifies them with sympy before inserting with verified=1. Grok-generated
solutions are stored via insert_unverified_solution and never auto-verified.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tutor.knowledge.models import (
    Answer,
    HintTemplate,
    Misconception,
    Problem,
    ReferenceSolution,
    SolutionStep,
    Template,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS templates (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS problems (
    id TEXT PRIMARY KEY, problem_type TEXT NOT NULL, problem_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL, text_hash TEXT NOT NULL, body TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0, template_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_problems_hash ON problems(text_hash);
CREATE TABLE IF NOT EXISTS solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, problem_id TEXT NOT NULL,
    body TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0, origin TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS problem_concepts (problem_id TEXT NOT NULL, concept_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS misconceptions (id TEXT PRIMARY KEY, concept_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hint_templates (
    id TEXT PRIMARY KEY, concept_id TEXT, misconception_id TEXT,
    level INTEGER NOT NULL, template_text TEXT NOT NULL
);
"""


class KnowledgeDB:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- inserts (used by the seeder and the solver candidate store) ---------

    def insert_concept(self, concept_id: str, name: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO concepts VALUES (?, ?)", (concept_id, name))
        self._conn.commit()

    def insert_template(self, template: Template) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO templates VALUES (?, ?)",
            (template.id, template.model_dump_json()),
        )
        self._conn.commit()

    def insert_problem(
        self, problem: Problem, normalized_text: str, text_hash: str, verified: bool
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO problems VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                problem.id,
                problem.problem_type,
                problem.problem_text,
                normalized_text,
                text_hash,
                problem.model_dump_json(),
                int(verified),
                problem.template_id,
            ),
        )
        self._conn.execute("DELETE FROM problem_concepts WHERE problem_id = ?", (problem.id,))
        for c in problem.concepts:
            self._conn.execute("INSERT INTO problem_concepts VALUES (?, ?)", (problem.id, c))
        self._conn.commit()

    def insert_solution(
        self, problem_id: str, solution: ReferenceSolution, verified: bool
    ) -> None:
        self._conn.execute(
            "INSERT INTO solutions (problem_id, body, verified, origin) VALUES (?, ?, ?, ?)",
            (problem_id, solution.model_dump_json(), int(verified), solution.origin),
        )
        self._conn.commit()

    def insert_unverified_solution(self, problem_id: str, solution: ReferenceSolution) -> None:
        self.insert_solution(problem_id, solution.model_copy(update={"verified": False}), False)

    def insert_misconception(self, m: Misconception) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO misconceptions VALUES (?, ?, ?)",
            (m.id, m.concept_id, m.model_dump_json()),
        )
        self._conn.commit()

    def insert_hint_template(self, h: HintTemplate) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO hint_templates VALUES (?, ?, ?, ?, ?)",
            (h.id, h.concept_id, h.misconception_id, h.level, h.template_text),
        )
        self._conn.commit()

    # --- queries -------------------------------------------------------------

    def concepts(self) -> dict[str, str]:
        return dict(self._conn.execute("SELECT id, name FROM concepts"))

    def find_by_text_hash(self, text_hash: str) -> Problem | None:
        row = self._conn.execute(
            "SELECT body FROM problems WHERE text_hash = ? AND verified = 1", (text_hash,)
        ).fetchone()
        return Problem.model_validate_json(row[0]) if row else None

    def all_problems(self, verified_only: bool = True) -> list[Problem]:
        q = "SELECT body FROM problems" + (" WHERE verified = 1" if verified_only else "")
        return [Problem.model_validate_json(r[0]) for r in self._conn.execute(q)]

    def templates(self) -> list[Template]:
        return [
            Template.model_validate_json(r[0])
            for r in self._conn.execute("SELECT body FROM templates")
        ]

    def get_template(self, template_id: str) -> Template | None:
        row = self._conn.execute(
            "SELECT body FROM templates WHERE id = ?", (template_id,)
        ).fetchone()
        return Template.model_validate_json(row[0]) if row else None

    def verified_solution(self, problem_id: str) -> ReferenceSolution | None:
        row = self._conn.execute(
            "SELECT body FROM solutions WHERE problem_id = ? AND verified = 1 LIMIT 1",
            (problem_id,),
        ).fetchone()
        return ReferenceSolution.model_validate_json(row[0]) if row else None

    def misconceptions_for(self, concepts: list[str]) -> list[Misconception]:
        if not concepts:
            return []
        q = f"SELECT body FROM misconceptions WHERE concept_id IN ({','.join('?' * len(concepts))})"
        return [Misconception.model_validate_json(r[0]) for r in self._conn.execute(q, concepts)]

    def get_misconception(self, misconception_id: str) -> Misconception | None:
        row = self._conn.execute(
            "SELECT body FROM misconceptions WHERE id = ?", (misconception_id,)
        ).fetchone()
        return Misconception.model_validate_json(row[0]) if row else None

    def hint_templates_for(
        self,
        concepts: list[str],
        misconception_id: str | None = None,
        level: int | None = None,
    ) -> list[HintTemplate]:
        """Misconception-specific templates first, then concept-level ones."""
        rows = self._conn.execute(
            "SELECT id, concept_id, misconception_id, level, template_text FROM hint_templates"
        ).fetchall()
        out = [
            HintTemplate(
                id=r[0], concept_id=r[1], misconception_id=r[2], level=r[3], template_text=r[4]
            )
            for r in rows
        ]
        if level is not None:
            out = [h for h in out if h.level == level]
        matched = [h for h in out if misconception_id and h.misconception_id == misconception_id]
        generic = [
            h
            for h in out
            if h.misconception_id is None and (not concepts or h.concept_id in concepts or h.concept_id is None)
        ]
        return matched + generic

    def problems_by_concepts(self, concepts: set[str]) -> list[tuple[Problem, float]]:
        """Verified problems with Jaccard overlap of concept tags, best first."""
        scored = []
        for p in self.all_problems():
            tags = set(p.concepts)
            union = tags | concepts
            if not union:
                continue
            overlap = len(tags & concepts) / len(union)
            if overlap > 0:
                scored.append((p, overlap))
        return sorted(scored, key=lambda t: -t[1])

    def close(self) -> None:
        self._conn.close()


def answer_to_json(answer: Answer) -> str:
    return json.dumps(answer.model_dump())


__all__ = [
    "KnowledgeDB",
    "Answer",
    "Problem",
    "ReferenceSolution",
    "SolutionStep",
    "Template",
    "Misconception",
    "HintTemplate",
]
