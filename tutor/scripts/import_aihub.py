"""Import AI Hub '수학 교과 문제 풀이과정 데이터' into Tutor_Agent's KnowledgeDB.

Expected layout (directories may contain nested files)::

    datasets/raw/aihub_math/TL/**/*.json
    datasets/raw/aihub_math/TS/**/*.png

The AI Hub JSON already contains problem/answer/explanation text descriptions and
curriculum achievement standards.  This importer keeps the source text, maps
achievement-standard codes to concept ids, and extracts SymPy-readable math where
possible.

Verification policy (AI Hub only):
    AI Hub is an officially labelled corpus, so its answers/explanations are
    treated as a trusted source and stored with ``verified=True``.  The SymPy
    ``mathnorm.verify_answer`` check still runs whenever an answer and equation
    are machine-checkable, and its result is recorded in the problem ``source``
    string (``sympy=VERIFIED|FAILED|UNCHECKED``) plus the import summary, but a
    SymPy failure or an uncheckable answer never demotes an AI Hub item to
    ``verified=False``.  Seed data and Grok-generated solutions keep their own
    (unchanged) verification policy — see ``seed_db.py`` and ``grok_solver.py``.

Example::

    python -m tutor.scripts.import_aihub \
      --labels datasets/raw/aihub_math/TL \
      --images datasets/raw/aihub_math/TS \
      --db data/knowledge.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tutor.config import load_settings
from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import Answer, Problem, ReferenceSolution, SolutionStep
from tutor.knowledge.matching import problem_hash
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

AIHUB_SOURCE = "AIHub 수학 교과 문제 풀이과정 데이터"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# AI Hub ships human-labelled answers/explanations, so the corpus itself is the
# trust anchor here; SymPy is only a secondary audit signal (see module docstring).
AIHUB_TRUSTED = True

SYMPY_VERIFIED = "VERIFIED"  # mathnorm proved the answer against an equation
SYMPY_FAILED = "FAILED"  # mathnorm had something to check and it did not hold
SYMPY_UNCHECKED = "UNCHECKED"  # nothing machine-checkable (text/choice answer, no equation)

PROBLEM_CLASSES = {"문항(텍스트)", "문항(이미지)"}
ANSWER_CLASSES = {"정답(텍스트)", "정답(이미지)"}
SOLUTION_CLASSES = {"해설(텍스트)", "해설(이미지)"}

_ACHIEVEMENT_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")
_LATEX_SPAN_RE = re.compile(r"\$(.+?)\$", re.DOTALL)
_ASCII_MATH_RE = re.compile(
    r"(?<![\w가-힣])([A-Za-z0-9_.()]+(?:\s*(?:\*\*|[+\-*/=^])\s*[A-Za-z0-9_.()]+)+)"
)


@dataclass(frozen=True)
class AchievementStandard:
    year: str
    code: str
    description: str

    @property
    def concept_id(self) -> str:
        return f"curriculum:{self.year}:{self.code}"

    @property
    def concept_name(self) -> str:
        return f"[{self.code}] {self.description}".strip()


@dataclass
class AnswerAnalysis:
    """The Answer to store plus the outcome of the secondary SymPy audit."""

    answer: Answer
    equations: list[str]
    sympy_check: str


@dataclass
class ConvertedItem:
    problem: Problem
    solution: ReferenceSolution
    verified: bool
    sympy_check: str
    concepts: list[AchievementStandard]
    image_path: Path | None


@dataclass
class ImportStats:
    scanned: int = 0
    imported: int = 0
    verified: int = 0
    unverified: int = 0
    failed: int = 0
    missing_images: int = 0
    # Secondary SymPy audit; these never change how an item is stored.
    sympy_verified: int = 0
    sympy_failed: int = 0
    sympy_unchecked: int = 0


def _clean(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _descriptions(payload: dict, class_names: set[str]) -> list[str]:
    """Return deduplicated text_description values in source order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in payload.get("learning_data_info") or []:
        if item.get("class_name") not in class_names:
            continue
        for info in item.get("class_info_list") or []:
            text = _clean(info.get("text_description"))
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _achievement_standards(payload: dict) -> list[AchievementStandard]:
    src = payload.get("source_data_info") or {}
    out: list[AchievementStandard] = []
    seen: set[tuple[str, str]] = set()
    for year in ("2009", "2015", "2022"):
        values = src.get(f"{year}_achievement_standard") or []
        if isinstance(values, str):
            values = [values]
        for raw in values:
            raw = _clean(raw)
            if not raw:
                continue
            m = _ACHIEVEMENT_RE.match(raw)
            if m:
                code, description = _clean(m.group(1)), _clean(m.group(2))
            else:
                # Keep unexpected standards deterministic rather than discarding them.
                code = "unknown-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
                description = raw
            key = (year, code)
            if key not in seen:
                seen.add(key)
                out.append(AchievementStandard(year, code, description))
    return out


def _infer_problem_type(problem_text: str, standards: list[AchievementStandard]) -> str:
    text = " ".join([problem_text, *(s.description for s in standards)])
    rules = (
        # Division standards often mention multiplication as the inverse operation,
        # so check the more specific division vocabulary first.
        (("나눗셈", "나누기", "몫", "나머지"), "division"),
        (("곱셈", "곱하기"), "multiplication"),
        (("덧셈", "뺄셈"), "addition_subtraction"),
        (("분수",), "fraction"),
        (("소수",), "decimal"),
        (("방정식",), "equation"),
        (("부등식",), "inequality"),
        (("함수",), "function"),
        (("미분", "도함수"), "differentiation"),
        (("적분",), "integration"),
        (("확률",), "probability"),
        (("통계", "자료", "막대그래프", "그림그래프"), "data_handling"),
        (("삼각형", "사각형", "다각형", "원", "각", "도형", "평행", "수직"), "geometry"),
        (("길이", "들이", "무게", "넓이", "부피", "시간", "단위"), "measurement"),
    )
    for keywords, problem_type in rules:
        if any(k in text for k in keywords):
            return problem_type
    return "math_problem"


def _replace_simple_latex_fraction(text: str) -> str:
    # Repeatedly handle simple/non-nested \frac{a}{b}; nested cases simply remain
    # unparsed and therefore stay unverified rather than being guessed.
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"(\1)/(\2)", text)
    return text


def _latex_to_ascii(span: str) -> str:
    s = span.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    # Common blank/shape placeholders in elementary worksheets. Keep each shape
    # stable so equations such as \square \div 9 = 7 remain solvable.
    s = s.replace("\\square", "x").replace("\\Box", "x")
    s = s.replace("□", "x").replace("\\bigcirc", "y").replace("\\triangle", "z")
    s = _replace_simple_latex_fraction(s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\^\s*\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^\s*([A-Za-z0-9.+-]+)", r"**\1", s)
    s = re.sub(r"\\(?:quad|qquad|,|;|!|:|circ)", " ", s)
    # Unit/text annotations should not become fake SymPy variables (e.g. cm -> c*m).
    s = re.sub(r"\\(?:mathrm|text)\s*\{[^{}]*\}", "", s)
    # Formatting wrappers can keep their mathematical contents.
    s = re.sub(r"\\(?:mathbf|mathit|underline)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("&", " ")
    # Unknown LaTeX commands are safer to reject later than to invent semantics.
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _candidate_math_strings(text: str) -> list[str]:
    candidates: list[str] = []

    for span in _LATEX_SPAN_RE.findall(text):
        # AI Hub often packs several worked steps into one span. Strip array
        # wrappers and split on visual separators/arrows before SymPy parsing.
        span = re.sub(r"\\begin\s*\{array\}\s*\{[^{}]*\}", " ", span)
        span = re.sub(r"\\end\s*\{array\}", " ", span)
        span = span.replace("\\hline", " ")
        pieces = re.split(
            r"\\(?:quad|qquad|rightarrow|Rightarrow|leftarrow|uparrow)\b|\\\\|,",
            span,
        )
        for piece in pieces:
            ascii_math = _latex_to_ascii(piece)
            if ascii_math:
                candidates.append(ascii_math)

    # Some labels contain already-ASCII equations outside $...$. Remove LaTeX
    # spans first so command names such as ``\square`` are not mistaken for
    # ASCII variables/equations.
    outside_latex = _LATEX_SPAN_RE.sub(" ", text)
    candidates.extend(_clean(x) for x in _ASCII_MATH_RE.findall(outside_latex))

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip(" ,.;:")
        if not candidate or candidate in seen:
            continue
        try:
            mathnorm.parse_equation(candidate)
        except mathnorm.ParseError:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _numeric_literal(answer_text: str) -> str | None:
    """Return a conservative scalar literal, never evaluate a worked expression.

    We intentionally accept simple numbers/fractions/sqrt/pi forms but reject
    operators such as ``25*21`` because in a multiple-choice item that may be the
    requested expression itself rather than the value 525.
    """
    texts: list[str] = []
    spans = _LATEX_SPAN_RE.findall(answer_text)
    if spans:
        texts.extend(_latex_to_ascii(span) for span in spans)
    else:
        texts.append(_clean(answer_text))

    allowed = re.compile(r"^[\s()+\-0-9./sqrtpi]+$")
    for text in texts:
        text = text.strip()
        if not text or not allowed.fullmatch(text):
            continue
        # Addition/multiplication in an answer may denote the requested expression.
        # A leading sign is okay; other +, *, ** are not treated as scalar literals.
        body = text[1:] if text[:1] in "+-" else text
        if "+" in body or "*" in body:
            continue
        try:
            expr = mathnorm.parse_expression(text)
        except mathnorm.ParseError:
            continue
        if not expr.free_symbols and expr.is_number is True:
            return str(expr)
    return None


def _normalized_literal(text: str) -> str:
    return re.sub(r"[\s()]", "", text)


def _auditable_equations(equations: list[str]) -> list[str]:
    """Relations SymPy can actually solve for the answer (one unknown, one '=')."""
    out: list[str] = []
    for equation in equations:
        if "=" not in equation:
            continue
        try:
            residual, is_eq = mathnorm.parse_equation(equation)
        except mathnorm.ParseError:
            continue
        if is_eq and len(residual.free_symbols) == 1:
            out.append(equation)
    return out


def _auditable_expressions(
    equations: list[str], answer_value: str
) -> tuple[list[int], bool]:
    """Indices of candidates that are a real computation of the answer.

    Extraction picks up worksheet noise (bare numbers, ``1L``, LaTeX/CSS tokens
    such as ``text-align`` that parse as a subtraction) and multiple-choice
    options that merely restate the answer.  Auditing against those produces
    fake passes (an option equal to the answer "proves" itself) and fake
    failures (a sibling option is not a claim about this answer), so only
    closed-form arithmetic that is not a verbatim restatement counts as
    machine-checkable.  The second return value flags that the answer literal
    itself appeared among the candidates.
    """
    answer_literal = _normalized_literal(answer_value)
    out: list[int] = []
    restated = False
    for i, candidate in enumerate(equations):
        if "=" in candidate:
            continue
        if _normalized_literal(candidate) == answer_literal:
            restated = True
            continue
        if not re.search(r"[+\-*/^]", candidate):
            continue
        try:
            expr = mathnorm.parse_expression(candidate)
        except mathnorm.ParseError:
            continue
        if not expr.free_symbols and expr.is_number is True:
            out.append(i)
    return out, restated


def _analyze_answer(equations: list[str], answer_text: str) -> AnswerAnalysis:
    """Build the Answer model and run the secondary SymPy audit where possible.

    The returned ``sympy_check`` is diagnostic only: AI Hub items are stored with
    ``verified=AIHUB_TRUSTED`` regardless of the outcome.
    """
    answer_text = _clean(answer_text)
    numeric = _numeric_literal(answer_text)

    if numeric is not None:
        equation_forms = _auditable_equations(equations)
        expression_idx, answer_restated = _auditable_expressions(equations, numeric)

        if equation_forms and mathnorm.verify_answer(equation_forms, "SCALAR", numeric):
            return AnswerAnalysis(
                Answer(kind="SCALAR", value=numeric), equations, SYMPY_VERIFIED
            )

        # Arithmetic questions such as "7*8" are verifiable as EXPRESSION.
        for i in expression_idx:
            candidate = equations[i]
            if mathnorm.verify_answer([candidate], "EXPRESSION", numeric):
                ordered = [candidate, *equations[:i], *equations[i + 1 :]]
                return AnswerAnalysis(
                    Answer(kind="EXPRESSION", value=numeric), ordered, SYMPY_VERIFIED
                )

        # Only call it a failure when something machine-checkable was actually
        # checked; otherwise the item simply is not auditable by SymPy.  A
        # candidate list that merely echoes the answer (multiple choice) has no
        # derivation to contradict, so it is unchecked rather than failed.
        if equation_forms or (expression_idx and not answer_restated):
            check = SYMPY_FAILED
        else:
            check = SYMPY_UNCHECKED
        return AnswerAnalysis(Answer(kind="SCALAR", value=numeric), equations, check)

    # The current Tutor_Agent AnswerKind has no TEXT/CHOICE variant.  Preserve
    # the original answer losslessly as an EXPRESSION until that model is
    # extended; SymPy cannot audit it, which is not a verification failure.
    return AnswerAnalysis(
        Answer(kind="EXPRESSION", value=answer_text), equations, SYMPY_UNCHECKED
    )


def _solution_steps(explanations: list[str]) -> list[SolutionStep]:
    steps: list[SolutionStep] = []
    seen_exprs: set[str] = set()
    idx = 1

    for explanation in explanations:
        expressions = _candidate_math_strings(explanation)
        if expressions:
            for expression in expressions:
                if expression in seen_exprs:
                    continue
                seen_exprs.add(expression)
                steps.append(
                    SolutionStep(
                        idx=idx,
                        description=explanation,
                        expression=expression,
                    )
                )
                idx += 1
        elif explanation:
            # Text/geometry explanations are still useful to an LLM even when
            # there is no SymPy-readable expression.
            steps.append(SolutionStep(idx=idx, description=explanation, expression=""))
            idx += 1
    return steps


def _index_images(images_dir: Path | None) -> dict[str, Path]:
    if images_dir is None:
        return {}
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    out: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out.setdefault(path.stem, path)
    return out


def _source_string(
    payload: dict,
    source_name: str,
    image_path: Path | None,
    images_dir: Path | None,
    sympy_check: str,
) -> str:
    raw = payload.get("raw_data_info") or {}
    src = payload.get("source_data_info") or {}
    parts = [
        AIHUB_SOURCE,
        f"source={source_name}",
        f"school={_clean(raw.get('school'))}",
        f"grade={_clean(raw.get('grade'))}",
        f"semester={_clean(raw.get('semester'))}",
        f"revision={_clean(raw.get('revision_year'))}",
        f"difficulty={_clean(src.get('level_of_difficulty'))}",
        f"format={_clean(src.get('types_of_problems'))}",
        # Audit trail: the item is trusted because of its source, this records
        # what SymPy independently had to say about it.
        f"sympy={sympy_check}",
    ]
    if image_path is not None:
        try:
            image_value = str(image_path.relative_to(images_dir)) if images_dir else image_path.name
        except ValueError:
            image_value = image_path.name
        parts.append(f"image={image_value}")
    return "|".join(parts)


def convert_json(
    payload: dict, *, image_map: dict[str, Path], images_dir: Path | None = None
) -> ConvertedItem:
    raw = payload.get("raw_data_info") or {}
    src = payload.get("source_data_info") or {}
    source_name = _clean(src.get("source_data_name"))
    if not source_name:
        raise ValueError("missing source_data_info.source_data_name")

    problem_parts = _descriptions(payload, PROBLEM_CLASSES)
    if not problem_parts:
        raise ValueError("no problem text/image text_description")
    problem_text = "\n".join(problem_parts)

    answer_parts = _descriptions(payload, ANSWER_CLASSES)
    answer_text = "\n".join(answer_parts)
    explanations = _descriptions(payload, SOLUTION_CLASSES)

    standards = _achievement_standards(payload)
    analysis = _analyze_answer(_candidate_math_strings(problem_text), answer_text)
    answer, equations = analysis.answer, analysis.equations
    concept_ids = [standard.concept_id for standard in standards]

    # Officially labelled corpus -> trusted; the SymPy audit result is carried
    # alongside for reporting but does not gate `verified`.
    verified = AIHUB_TRUSTED

    image_path = image_map.get(source_name)
    source = _source_string(
        payload, source_name, image_path, images_dir, analysis.sympy_check
    )

    problem = Problem(
        id=f"aihub_{source_name}",
        problem_type=_infer_problem_type(problem_text, standards),
        problem_text=problem_text,
        equations=equations,
        parameters={},
        answer=answer,
        source=source,
        verified=verified,
        template_id=None,
        concepts=concept_ids,
    )
    solution = ReferenceSolution(
        steps=_solution_steps(explanations),
        final_answer=answer,
        concepts=concept_ids,
        verified=verified,
        origin="db",
    )
    return ConvertedItem(
        problem=problem,
        solution=solution,
        verified=verified,
        sympy_check=analysis.sympy_check,
        concepts=standards,
        image_path=image_path,
    )


def _replace_db_solution(
    db: KnowledgeDB, problem_id: str, solution: ReferenceSolution, verified: bool
) -> None:
    """Keep re-imports idempotent with the current KnowledgeDB schema.

    ``solutions`` has an autoincrement id and no unique problem/origin key, while
    ``insert_problem`` already upserts.  Remove the previous DB-origin solution
    for this AI Hub problem before inserting the replacement.
    """
    conn = getattr(db, "_conn", None)
    if conn is None:
        raise RuntimeError("KnowledgeDB no longer exposes its SQLite connection; add a replace_solution API")
    conn.execute("DELETE FROM solutions WHERE problem_id = ? AND origin = 'db'", (problem_id,))
    conn.commit()
    db.insert_solution(problem_id, solution, verified=verified)


def import_aihub(
    *,
    labels_dir: Path,
    images_dir: Path | None,
    db_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
    log_every: int = 100,
) -> ImportStats:
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    label_paths = sorted(labels_dir.rglob("*.json"))
    if limit is not None:
        label_paths = label_paths[: max(limit, 0)]
    image_map = _index_images(images_dir)

    stats = ImportStats()
    db = None if dry_run else KnowledgeDB(db_path)
    try:
        for path in label_paths:
            stats.scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                item = convert_json(payload, image_map=image_map, images_dir=images_dir)

                if images_dir is not None and item.image_path is None:
                    stats.missing_images += 1

                if db is not None:
                    for concept in item.concepts:
                        db.insert_concept(concept.concept_id, concept.concept_name)

                    rec = Recognition(
                        problem_text=item.problem.problem_text,
                        equations=item.problem.equations,
                    )
                    db.insert_problem(
                        item.problem,
                        normalized_text=mathnorm.normalize_text(item.problem.problem_text),
                        text_hash=problem_hash(rec),
                        verified=item.verified,
                    )
                    _replace_db_solution(
                        db,
                        item.problem.id,
                        item.solution,
                        item.verified,
                    )

                stats.imported += 1
                if item.verified:
                    stats.verified += 1
                else:
                    stats.unverified += 1

                if item.sympy_check == SYMPY_VERIFIED:
                    stats.sympy_verified += 1
                elif item.sympy_check == SYMPY_FAILED:
                    stats.sympy_failed += 1
                    # Trusted source wins; keep the ids reachable under --verbose
                    # (a corpus-wide import would drown in per-item warnings).
                    log.debug(
                        "sympy check failed for %s (stored verified=%s, trusted source)",
                        item.problem.id,
                        item.verified,
                    )
                else:
                    stats.sympy_unchecked += 1
            except Exception as exc:
                stats.failed += 1
                log.exception("failed to import %s: %s", path, exc)

            if log_every > 0 and stats.scanned % log_every == 0:
                log.info(
                    "progress scanned=%d imported=%d verified=%d unverified=%d "
                    "failed=%d sympy(ok/fail/n-a)=%d/%d/%d",
                    stats.scanned,
                    stats.imported,
                    stats.verified,
                    stats.unverified,
                    stats.failed,
                    stats.sympy_verified,
                    stats.sympy_failed,
                    stats.sympy_unchecked,
                )
    finally:
        if db is not None:
            db.close()

    if stats.sympy_failed:
        log.warning(
            "%d imported item(s) contradict the sympy audit but stay verified "
            "(trusted source); rerun with --verbose to list their ids",
            stats.sympy_failed,
        )
    return stats


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import AI Hub math curriculum problems into KnowledgeDB")
    p.add_argument("--labels", type=Path, required=True, help="TL directory containing JSON labels")
    p.add_argument("--images", type=Path, default=None, help="TS directory containing source images")
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite DB path (default: current Tutor_Agent DB_PATH setting)",
    )
    p.add_argument("--limit", type=int, default=None, help="Import only the first N JSON files")
    p.add_argument("--dry-run", action="store_true", help="Parse/verify without writing to SQLite")
    p.add_argument("--log-every", type=int, default=100, help="Progress log interval; 0 disables")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    settings = load_settings()
    db_path = args.db or settings.db_path
    stats = import_aihub(
        labels_dir=args.labels,
        images_dir=args.images,
        db_path=db_path,
        limit=args.limit,
        dry_run=args.dry_run,
        log_every=args.log_every,
    )

    mode = "DRY RUN" if args.dry_run else "IMPORTED"
    print(f"[{mode}] AI Hub math -> {db_path}")
    print(f"  scanned       : {stats.scanned}")
    print(f"  imported      : {stats.imported}")
    print(f"  verified      : {stats.verified} (trusted source: {AIHUB_SOURCE})")
    print(f"  unverified    : {stats.unverified}")
    print(f"  failed        : {stats.failed}")
    print(f"  missing images: {stats.missing_images}")
    print("  sympy audit (does not affect verified):")
    print(f"    verified    : {stats.sympy_verified}")
    print(f"    failed      : {stats.sympy_failed}")
    print(f"    unchecked   : {stats.sympy_unchecked}")


if __name__ == "__main__":
    main()
