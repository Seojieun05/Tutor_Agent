"""SQLite-backed Domain Knowledge DB.

JSON files under seeds/ are the human-editable source of truth; seed_db.py
verifies them with sympy before inserting with verified=1. Grok-generated
solutions are stored via insert_unverified_solution and never auto-verified.
"""

from __future__ import annotations

from dataclasses import dataclass
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
CREATE INDEX IF NOT EXISTS idx_problem_concepts ON problem_concepts(concept_id);
CREATE TABLE IF NOT EXISTS misconceptions (id TEXT PRIMARY KEY, concept_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hint_templates (
    id TEXT PRIMARY KEY, concept_id TEXT, misconception_id TEXT,
    level INTEGER NOT NULL, template_text TEXT NOT NULL
);
-- One line per concept, said BEFORE the student starts: what kind of problem
-- this is and what to check first. Deliberately not a hint_template with
-- level 0 — level 0 already means WAIT/PROBE to the policy, and this is not a
-- rung on that ladder. Written once per concept and reused for every problem
-- of that kind, which is what keeps it free at speak time.
CREATE TABLE IF NOT EXISTS concept_preflight (
    concept_id TEXT PRIMARY KEY, line TEXT NOT NULL
);
-- A hint written AHEAD OF TIME for one problem's one step at one level —
-- phrased at warm time by the same model and prompt the live path uses,
-- screened by the same guards, and readable by a human before any lesson.
-- The runtime serves it verbatim: model quality at template price.
CREATE TABLE IF NOT EXISTS prewritten_hints (
    problem_id TEXT NOT NULL, step INTEGER NOT NULL, level INTEGER NOT NULL,
    hint_text TEXT NOT NULL, board_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (problem_id, step, level)
);
"""


@dataclass(frozen=True)
class StoredBoardLine:
    expr: str
    note: str = ""


@dataclass(frozen=True)
class PrewrittenHint:
    """A reviewed hint artifact: what is said and what is written."""

    text: str
    board: tuple[StoredBoardLine, ...] = ()


class KnowledgeDB:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Backfill equations_sig for DBs written before it existed.

        Pure string work (no sympy), so even a 16k-problem import is a second
        of startup, once."""
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(problems)")}
        if "equations_sig" not in columns:
            self._conn.execute("ALTER TABLE problems ADD COLUMN equations_sig TEXT")
        hint_columns = {
            r[1] for r in self._conn.execute("PRAGMA table_info(prewritten_hints)")
        }
        if "board_json" not in hint_columns:
            self._conn.execute(
                "ALTER TABLE prewritten_hints "
                "ADD COLUMN board_json TEXT NOT NULL DEFAULT '[]'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_problems_eqsig ON problems(equations_sig)"
        )
        rows = self._conn.execute(
            "SELECT id, body FROM problems WHERE equations_sig IS NULL"
        ).fetchall()
        if rows:
            from tutor.knowledge.mathnorm import equations_signature

            self._conn.executemany(
                "UPDATE problems SET equations_sig = ? WHERE id = ?",
                [
                    (equations_signature(json.loads(body).get("equations", [])), pid)
                    for pid, body in rows
                ],
            )
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
        from tutor.knowledge.mathnorm import equations_signature

        self._conn.execute(
            "INSERT OR REPLACE INTO problems "
            "(id, problem_type, problem_text, normalized_text, text_hash, body, "
            " verified, template_id, equations_sig) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                problem.id,
                problem.problem_type,
                problem.problem_text,
                normalized_text,
                text_hash,
                problem.model_dump_json(),
                int(verified),
                problem.template_id,
                equations_signature(problem.equations),
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

    def concept_name(self, concept_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT name FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        return row[0] if row else None

    def preflight_line(self, concept_id: str) -> str | None:
        """The "이런 문제군요, 이것부터 보셨나요?" line for this concept."""
        row = self._conn.execute(
            "SELECT line FROM concept_preflight WHERE concept_id = ?", (concept_id,)
        ).fetchone()
        return row[0] if row else None

    def save_preflight_line(self, concept_id: str, line: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO concept_preflight VALUES (?, ?)",
            (concept_id, line.strip()),
        )
        self._conn.commit()

    def prewritten_hint(self, problem_id: str, step: int, level: int) -> str | None:
        """The line written for exactly this problem, step and level — if any."""
        artifact = self.prewritten_hint_artifact(problem_id, step, level)
        return artifact.text if artifact is not None else None

    def prewritten_hint_artifact(
        self, problem_id: str, step: int, level: int
    ) -> PrewrittenHint | None:
        """The reviewed spoken line and its optional whiteboard writing."""
        row = self._conn.execute(
            "SELECT hint_text, board_json FROM prewritten_hints "
            "WHERE problem_id = ? AND step = ? AND level = ?",
            (problem_id, step, level),
        ).fetchone()
        if row is None:
            return None
        try:
            raw_board = json.loads(row[1] or "[]")
            board = tuple(
                StoredBoardLine(
                    expr=str(item.get("expr", "")).strip(),
                    note=str(item.get("note", "")).strip(),
                )
                for item in raw_board
                if isinstance(item, dict) and str(item.get("expr", "")).strip()
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            # A hand-edited legacy row may be malformed. The spoken hint is
            # still usable; unsafe or unreadable writing simply stays off.
            board = ()
        return PrewrittenHint(text=row[0], board=board)

    def save_prewritten_hint(
        self, problem_id: str, step: int, level: int, text: str,
        board=(),
    ) -> None:
        lines = []
        for item in board or ():
            if isinstance(item, dict):
                expr, note = item.get("expr", ""), item.get("note", "")
            else:
                expr, note = getattr(item, "expr", ""), getattr(item, "note", "")
            expr = str(expr).strip()
            if expr:
                lines.append({"expr": expr, "note": str(note).strip()})
        self._conn.execute(
            "INSERT OR REPLACE INTO prewritten_hints "
            "(problem_id, step, level, hint_text, board_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (problem_id, step, level, text.strip(), json.dumps(lines, ensure_ascii=False)),
        )
        self._conn.commit()

    def clear_prewritten_hints(self, problem_id: str) -> None:
        """Steps changed → every line written against them is stale."""
        self._conn.execute(
            "DELETE FROM prewritten_hints WHERE problem_id = ?", (problem_id,)
        )
        self._conn.commit()

    def find_by_text_hash(self, text_hash: str) -> Problem | None:
        row = self._conn.execute(
            "SELECT body FROM problems WHERE text_hash = ? AND verified = 1", (text_hash,)
        ).fetchone()
        return Problem.model_validate_json(row[0]) if row else None

    def find_by_normalized_text(self, normalized: str, limit: int = 5) -> list[Problem]:
        """Verified problems whose stored prose equals this normalized text.

        The identity rung behind find_by_text_hash: printed choices and a
        re-read equation list change the composite hash on every capture,
        while the prose itself stays the problem.
        """
        if not normalized:
            return []
        rows = self._conn.execute(
            "SELECT body FROM problems WHERE normalized_text = ? AND verified = 1 LIMIT ?",
            (normalized, limit),
        )
        return [Problem.model_validate_json(r[0]) for r in rows]

    def all_problems(self, verified_only: bool = True) -> list[Problem]:
        q = "SELECT body FROM problems" + (" WHERE verified = 1" if verified_only else "")
        return [Problem.model_validate_json(r[0]) for r in self._conn.execute(q)]

    def problems_by_signature(self, signature: str, limit: int = 20) -> list[Problem]:
        """Verified problems whose equations use the same numbers/variables.

        The EXACT tier's candidate set: an indexed lookup instead of a scan
        over every stored problem (which cost ~6.5s once a dataset was imported).
        """
        if not signature:
            return []
        rows = self._conn.execute(
            "SELECT body FROM problems WHERE equations_sig = ? AND verified = 1 LIMIT ?",
            (signature, limit),
        )
        return [Problem.model_validate_json(r[0]) for r in rows]

    def has_pedagogy(self) -> bool:
        """Does the DB carry the hint templates the tutor speaks from?"""
        return bool(
            self._conn.execute("SELECT 1 FROM hint_templates LIMIT 1").fetchone()
        )

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

    PROBLEM_TYPE_BONUS = 0.1

    def problems_by_concepts(
        self, concepts: set[str], limit: int = 50, problem_type: str | None = None
    ) -> list[tuple[Problem, float]]:
        """Verified problems scored by concept overlap, best first.

        Concept overlap (Jaccard) is the score; sharing the coarse
        problem_type adds a small bonus, so two problems needing the same
        ideas rank above one that merely shares a tag. The join does the
        filtering: only problems that share at least one tag are deserialized,
        so an imported dataset does not turn this into a full-table scan."""
        if not concepts:
            return []
        placeholders = ",".join("?" * len(concepts))
        rows = self._conn.execute(
            f"""
            SELECT p.body,
                   COUNT(*) AS shared,
                   (SELECT COUNT(*) FROM problem_concepts a WHERE a.problem_id = p.id) AS total
            FROM problems p
            JOIN problem_concepts pc ON pc.problem_id = p.id
            WHERE pc.concept_id IN ({placeholders}) AND p.verified = 1
            GROUP BY p.id
            ORDER BY shared DESC
            LIMIT ?
            """,
            (*concepts, limit),
        ).fetchall()
        scored = []
        for body, shared, total in rows:
            union = total + len(concepts) - shared
            if union <= 0:
                continue
            problem = Problem.model_validate_json(body)
            score = shared / union
            if problem_type and problem_type != "unknown" and problem.problem_type == problem_type:
                score += self.PROBLEM_TYPE_BONUS
            scored.append((problem, score))
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
