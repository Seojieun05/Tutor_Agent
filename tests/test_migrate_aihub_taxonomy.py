"""AI Hub legacy taxonomy → whitelist migration.

The LLM step is stubbed: what matters here is that one mapping per standard is
reused across every problem holding it, that all five storage places end up in
step, and that nothing outside the whitelists survives.
"""

import json
import sqlite3

import pytest

from tutor.knowledge.concepts import ALLOWED_CONCEPT_IDS, CONCEPT_NAMES, MAX_CONCEPTS
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import Answer, Problem, ReferenceSolution, SolutionStep
from tutor.knowledge.taxonomy import ALLOWED_PROBLEM_TYPES, UNKNOWN_PROBLEM_TYPE
from tutor.scripts.migrate_aihub_taxonomy import (
    CONCEPT_TO_PROBLEM_TYPE,
    UNMAPPED_CONCEPTS,
    StandardMapper,
    StandardMapping,
    derive_problem_type,
    migrate,
)

STANDARDS = {
    "curriculum:2015:9수학03-01": (
        "[9수학03-01] 삼각형의 합동 조건을 이해한다.",
        ["triangle", "congruence"],
    ),
    "curriculum:2022:9수학03-01": (  # the same content in the newer curriculum
        "[9수학03-01] 삼각형의 합동을 설명할 수 있다.",
        ["triangle", "congruence"],
    ),
    "curriculum:2015:6수학01-05": (
        "[6수학01-05] 분수의 나눗셈을 할 수 있다.",
        ["fraction_operations", "division"],
    ),
    "curriculum:2015:4수학05-01": (
        "[4수학05-01] 자료를 표로 나타낼 수 있다.",
        [],  # nothing on the whitelist fits: stays empty on purpose
    ),
}


class StubLLM:
    """Answers from the table above; counts calls so reuse is measurable."""

    def __init__(self):
        self.calls: list[str] = []

    def complete_json(self, *, purpose, system, user, images=(), schema):
        self.calls.append(user)
        for name, concepts in STANDARDS.values():
            if name in user:
                return StandardMapping(concepts=concepts)
        return StandardMapping(concepts=[])


def _problem(pid, problem_type, concepts) -> Problem:
    return Problem(
        id=pid,
        problem_type=problem_type,
        problem_text=f"문제 {pid}",
        equations=["3*x = 15"],
        answer=Answer(kind="SCALAR", value="5"),
        source="AIHub …",
        verified=True,
        concepts=concepts,
    )


@pytest.fixture
def legacy_db(tmp_path):
    """A miniature of the imported corpus: many problems, few standards."""
    path = tmp_path / "knowledge.db"
    db = KnowledgeDB(path)
    for cid, (name, _) in STANDARDS.items():
        db.insert_concept(cid, name)

    layout = [
        # (id, legacy problem_type, legacy concepts)
        ("aihub_t1", "geometry", ["curriculum:2015:9수학03-01", "curriculum:2022:9수학03-01"]),
        ("aihub_t2", "math_problem", ["curriculum:2015:9수학03-01"]),
        ("aihub_t3", "math_problem", ["curriculum:2022:9수학03-01"]),
        ("aihub_f1", "division", ["curriculum:2015:6수학01-05"]),
        ("aihub_f2", "math_problem", ["curriculum:2015:6수학01-05"]),
        # nothing maps: legacy type is whitelisted, so it survives
        ("aihub_d1", "data_handling", ["curriculum:2015:4수학05-01"]),
        # nothing maps and the legacy type is not whitelisted → unknown
        ("aihub_d2", "math_problem", ["curriculum:2015:4수학05-01"]),
    ]
    for pid, ptype, concepts in layout:
        problem = _problem(pid, ptype, concepts)
        db.insert_problem(problem, normalized_text=pid, text_hash=pid, verified=True)
        db.insert_solution(
            pid,
            ReferenceSolution(
                steps=[SolutionStep(idx=1, description="…", expression="x = 5")],
                final_answer=problem.answer,
                concepts=concepts,
                verified=True,
                origin="db",
            ),
            verified=True,
        )
    db.close()
    return path


def rows(path, query, *args):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


# --- the deterministic half -------------------------------------------------


def test_type_table_only_uses_whitelisted_ids():
    assert set(CONCEPT_TO_PROBLEM_TYPE) <= ALLOWED_CONCEPT_IDS
    assert set(CONCEPT_TO_PROBLEM_TYPE.values()) <= ALLOWED_PROBLEM_TYPES
    assert UNMAPPED_CONCEPTS <= ALLOWED_CONCEPT_IDS


def test_every_concept_is_mapped_or_explicitly_unmapped():
    """A new concept in seeds/concepts.json must be classified, not forgotten."""
    assert set(CONCEPT_TO_PROBLEM_TYPE) | UNMAPPED_CONCEPTS == ALLOWED_CONCEPT_IDS


def test_problem_type_follows_the_concepts():
    assert derive_problem_type(["triangle", "congruence"], "math_problem") == (
        "geometry",
        "from_concepts",
    )
    # a clear majority decides, the odd one out does not
    assert derive_problem_type(
        ["fraction_operations", "fraction", "division"], "math_problem"
    ) == ("fraction", "from_concepts")


def test_a_whitelisted_legacy_type_survives_when_concepts_are_silent():
    assert derive_problem_type([], "geometry") == ("geometry", "kept_legacy")
    assert derive_problem_type(["pattern"], "division") == ("division", "kept_legacy")


def test_ambiguity_falls_back_and_then_to_unknown():
    # a 1-1 tie is not a decision
    assert derive_problem_type(["triangle", "permutation"], "math_problem") == (
        UNKNOWN_PROBLEM_TYPE,
        "unknown",
    )
    assert derive_problem_type([], "math_problem") == (UNKNOWN_PROBLEM_TYPE, "unknown")


# --- the LLM step is per standard, not per problem --------------------------


def test_one_call_per_standard_then_cached(legacy_db, tmp_path):
    llm = StubLLM()
    cache = tmp_path / "map.json"
    stats = migrate(legacy_db, cache_path=cache, workers=2, llm=llm)

    assert stats.problems_scanned == 7
    assert len(llm.calls) == len(STANDARDS) == 4  # NOT one per problem
    assert stats.llm_calls == 4

    # a second run re-reads the cache instead of the model
    again = StandardMapper(llm, cache)
    assert again.cache["curriculum:2015:9수학03-01"] == ["triangle", "congruence"]
    again.map_all([(cid, name) for cid, (name, _) in STANDARDS.items()])
    assert len(llm.calls) == 4


def test_cache_is_invalidated_when_the_whitelist_changes(tmp_path):
    cache = tmp_path / "map.json"
    cache.write_text(json.dumps({"_fingerprint": "stale", "curriculum:x": ["triangle"]}))
    assert StandardMapper(StubLLM(), cache).cache == {}


def test_llm_failure_costs_only_that_standard(legacy_db, tmp_path):
    class Boom(StubLLM):
        def complete_json(self, **kw):
            if "6수학01-05" in kw["user"]:
                raise RuntimeError("xAI 500")
            return super().complete_json(**kw)

    stats = migrate(legacy_db, cache_path=tmp_path / "m.json", workers=1, llm=Boom())
    assert stats.llm_failures == 1
    # the geometry standards still mapped, so those problems still migrated
    assert stats.types_after["geometry"] == 3


# --- the rewrite ------------------------------------------------------------


def test_all_five_places_are_rewritten_together(legacy_db, tmp_path):
    migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())

    # 1. the column
    types = dict(rows(legacy_db, "SELECT id, problem_type FROM problems"))
    assert types["aihub_t2"] == "geometry"
    # 2. the Problem body
    body = json.loads(rows(legacy_db, "SELECT body FROM problems WHERE id='aihub_t2'")[0][0])
    assert body["problem_type"] == "geometry"
    # 3. concepts inside that body
    assert body["concepts"] == ["triangle", "congruence"]
    # 4. the join table
    join = [r[0] for r in rows(
        legacy_db, "SELECT concept_id FROM problem_concepts WHERE problem_id='aihub_t2'"
    )]
    assert sorted(join) == ["congruence", "triangle"]
    # 5. the solution body
    solution = json.loads(
        rows(legacy_db, "SELECT body FROM solutions WHERE problem_id='aihub_t2'")[0][0]
    )
    assert solution["concepts"] == ["triangle", "congruence"]


def test_no_legacy_id_survives_anywhere(legacy_db, tmp_path):
    stats = migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())

    assert stats.leftover_join_rows == 0
    assert stats.leftover_bodies == 0
    assert stats.off_whitelist_types == 0
    assert rows(legacy_db, "SELECT COUNT(*) FROM concepts WHERE id LIKE 'curriculum:%'")[0][0] == 0
    # ...and the whitelist ids that are now in use got their names
    names = dict(rows(legacy_db, "SELECT id, name FROM concepts"))
    assert names["triangle"] == CONCEPT_NAMES["triangle"]


def test_shared_standard_gives_every_problem_the_same_tags(legacy_db, tmp_path):
    migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())
    tags = {
        pid: sorted(r[0] for r in rows(
            legacy_db, "SELECT concept_id FROM problem_concepts WHERE problem_id=?", pid
        ))
        for pid in ("aihub_t1", "aihub_t2", "aihub_t3")
    }
    assert tags["aihub_t1"] == tags["aihub_t2"] == tags["aihub_t3"]


def test_unmappable_standard_leaves_the_problem_untagged_not_wrong(legacy_db, tmp_path):
    stats = migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())
    types = dict(rows(legacy_db, "SELECT id, problem_type FROM problems"))
    assert types["aihub_d1"] == "data_handling"  # legacy type was whitelisted
    assert types["aihub_d2"] == UNKNOWN_PROBLEM_TYPE  # "math_problem" was not
    assert stats.problems_without_concepts == 2
    assert rows(
        legacy_db, "SELECT COUNT(*) FROM problem_concepts WHERE problem_id='aihub_d1'"
    )[0][0] == 0


def test_content_is_untouched(legacy_db, tmp_path):
    before = rows(legacy_db, "SELECT id, problem_text, verified, body FROM problems ORDER BY id")
    sol_before = rows(legacy_db, "SELECT problem_id, body, verified FROM solutions ORDER BY id")
    migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())
    after = rows(legacy_db, "SELECT id, problem_text, verified, body FROM problems ORDER BY id")
    sol_after = rows(legacy_db, "SELECT problem_id, body, verified FROM solutions ORDER BY id")

    for (pid, text, ver, body), (pid2, text2, ver2, body2) in zip(before, after):
        assert (pid, text, ver) == (pid2, text2, ver2)
        old, new = json.loads(body), json.loads(body2)
        for field in ("id", "problem_text", "equations", "answer", "source", "verified"):
            assert old[field] == new[field]
    for (pid, body, ver), (pid2, body2, ver2) in zip(sol_before, sol_after):
        assert (pid, ver) == (pid2, ver2)
        old, new = json.loads(body), json.loads(body2)
        assert old["steps"] == new["steps"]
        assert old["final_answer"] == new["final_answer"]
        assert old["verified"] == new["verified"]


def test_concept_count_is_capped(legacy_db, tmp_path):
    """A problem with many standards must not accumulate a syllabus."""

    class Chatty(StubLLM):
        def complete_json(self, **kw):
            return StandardMapping(concepts=sorted(ALLOWED_CONCEPT_IDS)[:8])

    migrate(legacy_db, cache_path=tmp_path / "m.json", llm=Chatty())
    counts = rows(
        legacy_db, "SELECT problem_id, COUNT(*) FROM problem_concepts GROUP BY 1"
    )
    assert all(n <= MAX_CONCEPTS for _, n in counts)


# --- safety -----------------------------------------------------------------


def test_dry_run_writes_nothing_but_reports_everything(legacy_db, tmp_path):
    before = rows(legacy_db, "SELECT id, problem_type, body FROM problems ORDER BY id")
    stats = migrate(legacy_db, dry_run=True, cache_path=tmp_path / "m.json", llm=StubLLM())

    assert rows(legacy_db, "SELECT id, problem_type, body FROM problems ORDER BY id") == before
    assert stats.problems_scanned == 7
    assert stats.solutions_updated == 7  # would change
    assert stats.types_after["geometry"] == 3
    assert list(legacy_db.parent.glob("*.bak-*")) == []  # no backup on a preview


def test_backup_is_taken_before_writing(legacy_db, tmp_path):
    migrate(legacy_db, cache_path=tmp_path / "m.json", llm=StubLLM())
    backups = list(legacy_db.parent.glob("knowledge.db.bak-*"))
    assert len(backups) == 1
    # the copy still holds the legacy taxonomy: it is the provenance archive
    assert rows(backups[0], "SELECT COUNT(*) FROM concepts WHERE id LIKE 'curriculum:%'")[0][0] == 4
    assert rows(
        backups[0], "SELECT COUNT(*) FROM problem_concepts WHERE concept_id LIKE 'curriculum:%'"
    )[0][0] > 0


def test_rerunning_is_a_no_op(legacy_db, tmp_path):
    cache = tmp_path / "m.json"
    migrate(legacy_db, cache_path=cache, llm=StubLLM())
    snapshot = rows(legacy_db, "SELECT id, problem_type, body FROM problems ORDER BY id")

    llm = StubLLM()
    stats = migrate(legacy_db, cache_path=cache, llm=llm)
    assert stats.problems_scanned == 0  # nothing carries the legacy taxonomy now
    assert llm.calls == []
    assert rows(legacy_db, "SELECT id, problem_type, body FROM problems ORDER BY id") == snapshot


def test_no_llm_leaves_the_db_alone_but_still_reports(legacy_db, tmp_path):
    stats = migrate(legacy_db, dry_run=True, cache_path=tmp_path / "empty.json", allow_llm=False)
    assert stats.llm_calls == 0
    assert stats.standards_mapped == 0
    assert stats.problems_scanned == 7
