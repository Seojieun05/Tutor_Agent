"""Migrate imported AI Hub problems onto the ConceptTagger whitelists.

The import (``import_aihub.py``) tagged problems with the source's own taxonomy:
a keyword-guessed ``problem_type`` ("math_problem", "division", ...) and one
``curriculum:2015:...`` / ``curriculum:2022:...`` concept per achievement
standard. Neither exists in the whitelists that ``tutor/knowledge/taxonomy.py``
and ``tutor/knowledge/concepts.py`` now define, so retrieval by concept cannot
match a freshly tagged problem against the imported corpus.

Cost is why this is a script and not a re-tag: 16k problems, but only a few
hundred DISTINCT achievement standards, and every problem sharing a standard
shares its concepts.

    unique achievement standards  →  one Grok call each  →  cached on disk
                                  →  reused by every problem holding it

So the LLM sees ~382 short texts instead of 16,246 problems, and a second run
(or a --dry-run) costs nothing. ``problem_type`` needs no model at all: it is
derived from the mapped concepts through the table below, and falls back to the
legacy value only when that value is itself a whitelisted type.

Everything the request calls out is rewritten together — the column, both JSON
bodies, and the join table — inside one transaction:

    problems.problem_type · problems.body(problem_type, concepts)
    problem_concepts · solutions.body(concepts)

Problem ids, text, equations, answers, solution steps and verified flags are
never touched, and data/problem_embeddings.npz is left alone (it is keyed by
problem id, which does not change).

Usage::

    # preview, no writes, no API calls beyond uncached standards
    python -m tutor.scripts.migrate_aihub_taxonomy --dry-run

    # map the standards and rewrite the DB (backs it up first)
    python -m tutor.scripts.migrate_aihub_taxonomy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from tutor.config import load_settings
from tutor.knowledge.concepts import (
    ALLOWED_CONCEPT_IDS,
    CONCEPT_NAMES,
    MAX_CONCEPTS,
    concepts_for_prompt,
    normalize_concepts,
)
from tutor.knowledge.models import Problem, ReferenceSolution
from tutor.knowledge.taxonomy import (
    ALLOWED_PROBLEM_TYPES,
    UNKNOWN_PROBLEM_TYPE,
    normalize_problem_type,
)

log = logging.getLogger(__name__)

# 예전 방식의 개념 id 접두사와, 매핑 결과 캐시 파일.
LEGACY_CONCEPT_PREFIX = "curriculum:"
DEFAULT_CACHE = Path("data/aihub_concept_map.json")


# --------------------------------------------------------------------------
# concept → problem_type: the deterministic half of the migration.
#
# One coarse type per fine concept, so a problem's type follows from what it
# actually needs. Concepts that do not imply a type on their own are listed in
# UNMAPPED below rather than guessed — they contribute nothing and the legacy
# value gets its chance instead.
# --------------------------------------------------------------------------
# 개념 → 큰 유형 대응표. 유형은 모델 없이 이 표만으로 정한다.
CONCEPT_TO_PROBLEM_TYPE: dict[str, str] = {
    # 수와 연산
    "natural_number": "arithmetic",
    "place_value": "arithmetic",
    "addition": "addition_subtraction",
    "subtraction": "addition_subtraction",
    "multiplication": "multiplication",
    "division": "division",
    "mixed_operations": "arithmetic",
    "factors_multiples": "number_theory",
    "greatest_common_divisor": "number_theory",
    "least_common_multiple": "number_theory",
    "prime_factorization": "number_theory",
    "fraction": "fraction",
    "fraction_operations": "fraction",
    "decimal": "decimal",
    "decimal_operations": "decimal",
    "recurring_decimal": "decimal",
    "ratio": "ratio_proportion",
    "rate": "ratio_proportion",
    "proportion": "ratio_proportion",
    "proportional_distribution": "ratio_proportion",
    "integer": "arithmetic",
    "rational_number": "arithmetic",
    "real_number": "arithmetic",
    "irrational_number": "exponent_root",
    "square_root": "exponent_root",
    "exponent": "exponent_root",
    "logarithm": "exponent_root",
    "complex_number": "complex_number",
    # 문자와 식
    "equality_equivalence": "equation",
    "algebraic_expression": "algebraic_expression",
    "polynomial": "algebraic_expression",
    "polynomial_operations": "algebraic_expression",
    "polynomial_identity": "identity_remainder",
    "remainder_theorem": "identity_remainder",
    "factor_theorem": "identity_remainder",
    "factorization": "factorization",
    "linear_equation": "linear_equation",
    "quadratic_equation": "quadratic_equation",
    "higher_degree_equation": "equation",
    "simultaneous_linear_equations": "system_of_equations",
    "equation_system": "system_of_equations",
    "linear_inequality": "inequality",
    "quadratic_inequality": "inequality",
    "inequality_system": "inequality",
    "direct_proportion": "ratio_proportion",
    "inverse_proportion": "ratio_proportion",
    # 함수
    "function": "function",
    "linear_function": "linear_function",
    "quadratic_function": "quadratic_function",
    "composite_function": "function",
    "inverse_function": "function",
    "rational_function": "function",
    "radical_function": "function",
    "exponential_function": "function",
    "logarithmic_function": "function",
    "trigonometric_function": "trigonometry",
    "sequence": "sequence",
    "arithmetic_sequence": "sequence",
    "geometric_sequence": "sequence",
    "sequence_sum": "sequence",
    "sigma_notation": "sequence",
    "mathematical_induction": "sequence",
    # 집합과 명제
    "set": "set_logic",
    "set_operations": "set_logic",
    "proposition": "set_logic",
    "necessary_sufficient_condition": "set_logic",
    "proof_by_contradiction": "set_logic",
    # 기하
    "point_line_plane": "geometry",
    "line_segment_ray": "geometry",
    "angle": "geometry",
    "perpendicular_parallel": "geometry",
    "triangle": "geometry",
    "triangle_properties": "geometry",
    "quadrilateral": "geometry",
    "quadrilateral_properties": "geometry",
    "polygon": "geometry",
    "circle": "geometry",
    "circle_properties": "geometry",
    "congruence": "geometry",
    "symmetry": "geometry",
    "geometric_construction": "geometry",
    "similarity": "geometry",
    "pythagorean_theorem": "geometry",
    "prism_pyramid": "geometry",
    "cylinder_cone_sphere": "geometry",
    "space_geometry": "geometry",
    "line_plane_relationships": "geometry",
    "vector": "geometry",
    "vector_operations": "geometry",
    "vector_components": "geometry",
    "dot_product": "geometry",
    "trigonometric_ratio": "trigonometry",
    "sine_rule": "trigonometry",
    "cosine_rule": "trigonometry",
    # 좌표
    "coordinate_plane": "coordinate_geometry",
    "distance_between_points": "coordinate_geometry",
    "internal_division": "coordinate_geometry",
    "coordinate_geometry": "coordinate_geometry",
    "circle_equation": "coordinate_geometry",
    "geometric_translation": "coordinate_geometry",
    "space_coordinates": "coordinate_geometry",
    "conic_section": "coordinate_geometry",
    "parabola": "coordinate_geometry",
    "ellipse": "coordinate_geometry",
    "hyperbola": "coordinate_geometry",
    # 측정
    "length": "measurement",
    "time": "measurement",
    "mass": "measurement",
    "capacity": "measurement",
    "angle_measure": "measurement",
    "perimeter": "measurement",
    "area": "measurement",
    "circle_circumference": "measurement",
    "circle_area": "measurement",
    "surface_area": "measurement",
    "volume": "measurement",
    # 자료와 가능성
    "data_classification": "data_handling",
    "table": "data_handling",
    "pictograph_reading": "data_handling",
    "bar_graph": "data_handling",
    "line_graph": "data_handling",
    "strip_graph": "data_handling",
    "pie_chart": "data_handling",
    "mean": "statistics",
    "frequency_distribution": "statistics",
    "relative_frequency": "statistics",
    "representative_value": "statistics",
    "dispersion": "statistics",
    "variance_standard_deviation": "statistics",
    "box_plot": "statistics",
    "scatter_plot": "statistics",
    "correlation": "statistics",
    "random_variable": "statistics",
    "discrete_random_variable": "statistics",
    "continuous_random_variable": "statistics",
    "probability_distribution": "statistics",
    "binomial_distribution": "statistics",
    "normal_distribution": "statistics",
    "population_sample": "statistics",
    "sample_mean": "statistics",
    "statistical_estimation": "statistics",
    "confidence_interval": "statistics",
    "population_proportion_estimation": "statistics",
    "counting_principle": "counting",
    "permutation": "counting",
    "combination": "counting",
    "permutation_with_identical_elements": "counting",
    "repeated_permutation": "counting",
    "repeated_combination": "counting",
    "binomial_theorem": "counting",
    "possibility": "probability",
    "probability": "probability",
    "complementary_event": "probability",
    "mutually_exclusive_events": "probability",
    "conditional_probability": "probability",
    "independent_events": "probability",
    "independent_trials": "probability",
    # 미적분
    "function_limit": "limit",
    "continuity": "limit",
    "sequence_limit": "limit",
    "infinite_series": "limit",
    "differentiation": "derivative",
    "derivative_rules": "derivative",
    "derivative_applications": "derivative",
    "indefinite_integral": "integral",
    "definite_integral": "integral",
    "integral_techniques": "integral",
    "integral_applications": "integral",
    "fundamental_theorem_of_calculus": "integral",
    # legacy_2015 leftovers
    "fraction_properties": "fraction",
    "pictograph_drawing": "data_handling",
    "pentagon_hexagon_classification": "geometry",
    "construct_equal_angle": "geometry",
    "gcd_lcm_properties": "number_theory",
    "external_division": "coordinate_geometry",
    "line_equation": "coordinate_geometry",
    "circular_permutation": "counting",
    "continuous_random_variable_integral_relation": "statistics",
}

# Concepts that genuinely do not pin down a coarse type: 규칙성/대응 appear in
# every area, |x| and 증명 are techniques, 행렬 has no whitelisted type.
# 어떤 유형에도 넣지 않는 개념들.
UNMAPPED_CONCEPTS = frozenset(
    {"pattern", "correspondence", "absolute_value", "proof", "matrix", "matrix_operations"}
)


# 매핑된 개념들로 큰 유형을 정한다(안 되면 화이트리스트에 있는 기존 값만 인정).
def derive_problem_type(concepts: list[str], legacy: str) -> tuple[str, str]:
    """Return (problem_type, why). Deterministic — no model involved.

    A clear majority among the mapped concepts wins. With no signal (or a tie)
    the legacy value is kept, but only if it is itself whitelisted: that is how
    "geometry"/"division" survive while "math_problem" becomes "unknown".
    """
    votes = Counter(
        CONCEPT_TO_PROBLEM_TYPE[c] for c in concepts if c in CONCEPT_TO_PROBLEM_TYPE
    )
    if votes:
        ranked = votes.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            return ranked[0][0], "from_concepts"
    fallback = normalize_problem_type(legacy)
    if fallback != UNKNOWN_PROBLEM_TYPE:
        return fallback, "kept_legacy"
    return UNKNOWN_PROBLEM_TYPE, "unknown"


# --------------------------------------------------------------------------
# achievement standard → whitelisted concepts (the one LLM step, cached)
# --------------------------------------------------------------------------


# 성취기준 하나에 대한 개념 매핑 결과.
class StandardMapping(BaseModel):
    concepts: list[str] = []


# [프롬프트] 성취기준 한 건을 개념 화이트리스트로 옮기는 매핑용. 애매하면 빈 목록이 낫다고 못 박는다.
def _system_prompt() -> str:
    return f"""You map ONE Korean curriculum achievement standard (성취기준) to the
curriculum concepts a student needs in order to solve problems carrying it.

Choose at most {MAX_CONCEPTS} ids (fewer is better; [] is fine) from this list:
{concepts_for_prompt()}

Hard rules:
- Use ids EXACTLY as written above. NEVER invent, translate or modify an id.
- Tag what the standard is ABOUT — the knowledge it teaches — not the wording.
- Prefer the most specific ids that still cover the whole standard. If the
  standard is about 이차방정식, tag quadratic_equation, not just algebraic_expression.
- Do NOT tag solution strategies or teaching activities (설명하기, 탐구하기).
- If nothing on the list fits, return []. Guessing is worse than nothing: an
  empty mapping leaves the problem searchable by text, a wrong one poisons
  concept retrieval for every problem sharing this standard.

Return ONLY the JSON object."""


# 서로 다른 성취기준마다 한 번씩만 모델을 부르고 결과를 디스크에 캐시한다(16k 문제 → 수백 회 호출).
class StandardMapper:
    """One Grok call per distinct standard, memoised on disk.

    The cache is keyed by concept id and fingerprinted with the whitelist, so
    editing seeds/concepts.json invalidates it instead of silently reusing
    mappings that point at ids which no longer exist.
    """

    # 캐시 경로와 화이트리스트 지문을 잡는다(개념 목록이 바뀌면 캐시가 자동 무효화).
    def __init__(self, llm, cache_path: Path, allow_llm: bool = True):
        self.llm = llm
        self.cache_path = cache_path
        self.allow_llm = allow_llm
        # sha1, not hash(): PYTHONHASHSEED is randomised per process, which
        # would throw the cache away (and re-spend the API budget) every run
        self.fingerprint = hashlib.sha1(
            "\n".join(sorted(ALLOWED_CONCEPT_IDS)).encode("utf-8")
        ).hexdigest()[:16]
        self.cache: dict[str, list[str]] = {}
        self.calls = 0
        self.failures = 0
        self._load()

    # 캐시 읽기.
    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if data.get("_fingerprint") != self.fingerprint:
            log.warning("concept whitelist changed since the cache was written; remapping")
            return
        self.cache = {k: v for k, v in data.items() if not k.startswith("_")}
        log.info("loaded %d cached standard mappings from %s", len(self.cache), self.cache_path)

    # 캐시 저장.
    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_fingerprint": self.fingerprint, **self.cache}
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # 성취기준 하나를 개념 목록으로 매핑(모델 호출).
    def _map_one(self, concept_id: str, name: str) -> list[str]:
        raw = self.llm.complete_json(
            purpose="tag",  # same whitelist job as the tagger, same latency knob
            system=_system_prompt(),
            user=f"성취기준: {name}",
            schema=StandardMapping,
        )
        return normalize_concepts(raw.concepts)  # whitelist enforced in Python

    # 여러 성취기준을 병렬로 매핑한다.
    def map_all(self, standards: list[tuple[str, str]], workers: int = 4) -> dict[str, list[str]]:
        todo = [(cid, name) for cid, name in standards if cid not in self.cache]
        log.info(
            "%d standard(s): %d cached, %d to map%s",
            len(standards),
            len(standards) - len(todo),
            len(todo),
            "" if self.allow_llm else " (--no-llm: leaving them empty)",
        )
        if not todo or not self.allow_llm:
            return {cid: self.cache.get(cid, []) for cid, _ in standards}

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._map_one, cid, name): cid for cid, name in todo}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    self.cache[cid] = future.result()
                    self.calls += 1
                except Exception as e:
                    # one bad standard must not cost the other 381 their results
                    self.failures += 1
                    log.warning("mapping failed for %s: %s", cid, e)
                done += 1
                if done % 25 == 0:
                    log.info("mapped %d/%d standards", done, len(todo))
                    self.save()
        self.save()
        return {cid: self.cache.get(cid, []) for cid, _ in standards}


# --------------------------------------------------------------------------
# the migration itself
# --------------------------------------------------------------------------


# 마이그레이션 통계.
@dataclass
class Stats:
    problems_scanned: int = 0
    problems_changed: int = 0
    standards: int = 0
    standards_mapped: int = 0  # produced at least one whitelist concept
    llm_calls: int = 0
    llm_failures: int = 0
    type_source: Counter = field(default_factory=Counter)
    types_before: Counter = field(default_factory=Counter)
    types_after: Counter = field(default_factory=Counter)
    concepts_before: int = 0
    concepts_after: int = 0
    problems_without_concepts: int = 0
    solutions_updated: int = 0
    legacy_concept_rows_removed: int = 0
    whitelist_concept_rows_added: int = 0
    # post-commit verification: all three must be 0
    leftover_join_rows: int = -1
    leftover_bodies: int = -1
    off_whitelist_types: int = -1


# 손대기 전에 DB를 복사해 둔다.
def backup_db(db_path: Path) -> Path:
    """Consistent copy via sqlite's backup API (safe even if a server is up)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = db_path.with_name(f"{db_path.name}.bak-{stamp}")
    src_conn = sqlite3.connect(str(db_path))
    dest_conn = sqlite3.connect(str(dest))
    try:
        with dest_conn:
            src_conn.backup(dest_conn)
    finally:
        src_conn.close()
        dest_conn.close()
    log.info("backup written: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# DB에 남아 있는 옛 방식 개념 id 목록.
def legacy_standards(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Distinct achievement standards actually referenced by a problem."""
    rows = conn.execute(
        """
        SELECT c.id, c.name
        FROM concepts c
        WHERE c.id LIKE ? AND EXISTS (
            SELECT 1 FROM problem_concepts pc WHERE pc.concept_id = c.id
        )
        ORDER BY c.id
        """,
        (LEGACY_CONCEPT_PREFIX + "%",),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# 마이그레이션 후 상태 점검.
def verify(conn: sqlite3.Connection, stats: Stats) -> None:
    """Re-read the DB and prove the legacy taxonomy is gone from every place."""
    like = LEGACY_CONCEPT_PREFIX + "%"
    stats.leftover_join_rows = conn.execute(
        "SELECT COUNT(*) FROM problem_concepts WHERE concept_id LIKE ?", (like,)
    ).fetchone()[0]
    stats.leftover_bodies = conn.execute(
        "SELECT COUNT(*) FROM problems WHERE body LIKE ?",
        (f'%"{LEGACY_CONCEPT_PREFIX}%',),
    ).fetchone()[0]
    placeholders = ",".join("?" * len(ALLOWED_PROBLEM_TYPES))
    stats.off_whitelist_types = conn.execute(
        f"SELECT COUNT(*) FROM problems WHERE problem_type NOT IN ({placeholders})",
        tuple(sorted(ALLOWED_PROBLEM_TYPES)),
    ).fetchone()[0]


# 바꿔야 할 문제들을 뽑는다.
def _target_problems(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Problems still carrying the legacy taxonomy (id, problem_type, body)."""
    return conn.execute(
        """
        SELECT DISTINCT p.id, p.problem_type, p.body
        FROM problems p
        JOIN problem_concepts pc ON pc.problem_id = p.id
        WHERE pc.concept_id LIKE ?
        ORDER BY p.id
        """,
        (LEGACY_CONCEPT_PREFIX + "%",),
    ).fetchall()


# 본체: 유형 컬럼·문제 본문·풀이 본문·개념 조인 테이블을 한 트랜잭션에서 함께 고친다.
def migrate(
    db_path: Path,
    *,
    dry_run: bool = False,
    cache_path: Path = DEFAULT_CACHE,
    workers: int = 4,
    allow_llm: bool = True,
    keep_legacy_concepts: bool = False,
    llm=None,
) -> Stats:
    stats = Stats()

    if dry_run:  # read-only handle: a preview cannot write even by accident
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        backup_db(db_path)
        conn = sqlite3.connect(str(db_path))

    try:
        standards = legacy_standards(conn)
        stats.standards = len(standards)
        if llm is None and allow_llm:
            llm = _grok_client()
        mapper = StandardMapper(llm, cache_path, allow_llm=allow_llm and llm is not None)
        mapping = mapper.map_all(standards, workers=workers)
        stats.llm_calls, stats.llm_failures = mapper.calls, mapper.failures
        stats.standards_mapped = sum(1 for v in mapping.values() if v)

        rows = _target_problems(conn)
        stats.problems_scanned = len(rows)
        used_concepts: set[str] = set()

        for problem_id, legacy_type, body in rows:
            problem = Problem.model_validate_json(body)
            stats.types_before[legacy_type] += 1
            stats.concepts_before += len(problem.concepts)

            # union in standard order, deduped and capped by the whitelist rules
            concepts = normalize_concepts(
                [c for legacy in problem.concepts for c in mapping.get(legacy, [])]
                + [c for c in problem.concepts if c in ALLOWED_CONCEPT_IDS]
            )
            problem_type, why = derive_problem_type(concepts, legacy_type)

            stats.types_after[problem_type] += 1
            stats.type_source[why] += 1
            stats.concepts_after += len(concepts)
            if not concepts:
                stats.problems_without_concepts += 1
            used_concepts.update(concepts)
            if concepts != problem.concepts or problem_type != problem.problem_type:
                stats.problems_changed += 1

            # 5: solution bodies carry their own copy of the concepts.
            # Read even on a dry run, so the preview counts what would change.
            stale_solutions = [
                (sid, ReferenceSolution.model_validate_json(sbody))
                for sid, sbody in conn.execute(
                    "SELECT id, body FROM solutions WHERE problem_id = ?", (problem_id,)
                ).fetchall()
            ]
            stale_solutions = [(sid, s) for sid, s in stale_solutions if s.concepts != concepts]
            stats.solutions_updated += len(stale_solutions)

            if dry_run:
                continue

            updated = problem.model_copy(
                update={"problem_type": problem_type, "concepts": concepts}
            )
            # 1 + 2: the column and the Problem body stay in step
            conn.execute(
                "UPDATE problems SET problem_type = ?, body = ? WHERE id = ?",
                (problem_type, updated.model_dump_json(), problem_id),
            )
            # 4: the join table the concept retrieval actually reads
            conn.execute("DELETE FROM problem_concepts WHERE problem_id = ?", (problem_id,))
            conn.executemany(
                "INSERT INTO problem_concepts VALUES (?, ?)",
                [(problem_id, c) for c in concepts],
            )
            for sid, solution in stale_solutions:
                conn.execute(
                    "UPDATE solutions SET body = ? WHERE id = ?",
                    (solution.model_copy(update={"concepts": concepts}).model_dump_json(), sid),
                )

        stats.whitelist_concept_rows_added = len(used_concepts)
        legacy_rows = conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE id LIKE ?", (LEGACY_CONCEPT_PREFIX + "%",)
        ).fetchone()[0]

        if not dry_run:
            # the id → name dictionary the tools read, now for the ids in use
            for concept_id in sorted(used_concepts):
                conn.execute(
                    "INSERT OR REPLACE INTO concepts VALUES (?, ?)",
                    (concept_id, CONCEPT_NAMES[concept_id]),
                )
            if not keep_legacy_concepts:
                # no problem references them any more; the backup keeps the
                # AI Hub standard provenance if it is ever needed again
                conn.execute(
                    "DELETE FROM concepts WHERE id LIKE ?", (LEGACY_CONCEPT_PREFIX + "%",)
                )
                stats.legacy_concept_rows_removed = legacy_rows
            conn.commit()
            verify(conn, stats)
        else:
            stats.legacy_concept_rows_removed = 0 if keep_legacy_concepts else legacy_rows
    finally:
        conn.close()

    return stats


# 매핑에 쓸 모델 클라이언트.
def _grok_client():
    settings = load_settings()
    if settings.echo_mode:
        log.warning(
            "no XAI_API_KEY: standards can only come from the cache "
            "(the run will otherwise leave every problem untagged)"
        )
        return None
    from tutor.llm.client import GrokClient

    # complete_json() never touches the tool registry, and building one would
    # load the RAG embedding model for a job that does no retrieval.
    return GrokClient(settings, registry=None)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


# 결과 보고서 출력.
def print_report(stats: Stats, dry_run: bool) -> None:
    print(f"\n[{'DRY RUN' if dry_run else 'MIGRATED'}] AI Hub legacy taxonomy → whitelists")
    print(f"  achievement standards : {stats.standards} "
          f"({stats.standards_mapped} mapped to ≥1 concept, "
          f"{stats.llm_calls} Grok calls, {stats.llm_failures} failed)")
    print(f"  problems scanned      : {stats.problems_scanned} "
          f"({stats.problems_changed} changed)")
    print(f"  concept tags          : {stats.concepts_before} → {stats.concepts_after}"
          f"  ({stats.problems_without_concepts} problems left with none)")
    label = "solutions to update  " if dry_run else "solutions updated    "
    print(f"  {label} : {stats.solutions_updated}")
    print(f"  concepts table        : +{stats.whitelist_concept_rows_added} whitelist, "
          f"-{stats.legacy_concept_rows_removed} legacy")

    print("\n  problem_type source:")
    for source in ("from_concepts", "kept_legacy", "unknown"):
        print(f"    {source:<14}: {stats.type_source.get(source, 0)}")

    if stats.leftover_join_rows >= 0:
        ok = (stats.leftover_join_rows == stats.leftover_bodies == stats.off_whitelist_types == 0)
        print(f"\n  verification (re-read from disk): {'PASS' if ok else 'FAIL'}")
        print(f"    legacy ids in problem_concepts : {stats.leftover_join_rows}")
        print(f"    legacy ids in problem bodies   : {stats.leftover_bodies}")
        print(f"    off-whitelist problem_type     : {stats.off_whitelist_types}")

    print("\n  problem_type before → after:")
    keys = sorted(set(stats.types_before) | set(stats.types_after),
                  key=lambda k: -(stats.types_after.get(k, 0) + stats.types_before.get(k, 0)))
    print(f"    {'type':<22}{'before':>8}{'after':>8}")
    for key in keys[:20]:
        before, after = stats.types_before.get(key, 0), stats.types_after.get(key, 0)
        flag = "" if key in ALLOWED_PROBLEM_TYPES else "  (legacy)"
        print(f"    {key:<22}{before:>8}{after:>8}{flag}")


# 커맨드라인 옵션 정의.
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=None, help="SQLite DB (default: DB_PATH setting)")
    p.add_argument("--dry-run", action="store_true", help="report only; opens the DB read-only")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="standard → concepts cache")
    p.add_argument("--workers", type=int, default=4, help="parallel Grok calls")
    p.add_argument("--no-llm", action="store_true", help="use only cached mappings")
    p.add_argument(
        "--keep-legacy-concepts",
        action="store_true",
        help="leave curriculum:* rows in the concepts table (they stay unreferenced)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


# 커맨드라인 진입점(--dry-run으로 미리 볼 수 있다).
def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    db_path = args.db or load_settings().db_path
    if not Path(db_path).exists():
        raise SystemExit(f"DB not found: {db_path}")

    stats = migrate(
        Path(db_path),
        dry_run=args.dry_run,
        cache_path=args.cache,
        workers=args.workers,
        allow_llm=not args.no_llm,
        keep_legacy_concepts=args.keep_legacy_concepts,
    )
    print_report(stats, args.dry_run)


if __name__ == "__main__":
    main()
