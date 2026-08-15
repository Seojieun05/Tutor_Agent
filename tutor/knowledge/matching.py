"""Matching cascade: EXACT → TEMPLATE → CONCEPT → SEMANTIC → NEW.

Equivalence is symbolic (sympy), never embedding-based — embeddings only enter
at the SEMANTIC tier, after the exact/structural ones have had their say, and
they never produce a "verified" solution. TEMPLATE hits are recomputed and
sympy-verified before being treated as verified knowledge.

The problem's tags (`rec.problem_type`, `rec.concepts`) come from the Grok
ConceptTagger and are whitelisted; this module no longer guesses them from
regexes over the problem text.
"""

from __future__ import annotations

import hashlib
import logging
import re

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import (
    Answer,
    MatchResult,
    ReferenceSolution,
    SolutionStep,
    Template,
    Tier,
)
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

_FUNCTION_DEF = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*x\s*\)\s*=\s*(.+?)\s*$"
)
_GRAPH_ALIAS = re.compile(
    r"^\s*y\s*=\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*x\s*\)\s*$"
)


def _without_graph_aliases(equations: list[str]) -> list[str]:
    """Drop ``y=f(x)`` only when the same read also defines ``f(x)=...``.

    Vision models often return both the function definition and the curve
    label from the prose.  The label adds no condition, but its presence made
    a verified two-equation problem look like a new four-equation problem.
    """
    defined = set()
    for equation in equations:
        match = _FUNCTION_DEF.match(equation)
        if match and match.group(2).strip() != "y":
            defined.add(match.group(1))
    return [
        equation
        for equation in equations
        if not (
            (alias := _GRAPH_ALIAS.match(equation))
            and alias.group(1) in defined
        )
    ]


def problem_hash(rec: Recognition) -> str:
    """Hash of the problem identity: text + equations + choices + diagram
    conditions. student_work is excluded so new work lines keep the cache."""
    parts = [mathnorm.normalize_text(rec.problem_text)]
    for eq in rec.equations:
        try:
            residual, _ = mathnorm.parse_equation(eq)
            parts.append(str(residual))
        except mathnorm.ParseError:
            parts.append(mathnorm.normalize_text(eq))
    parts.extend(mathnorm.normalize_text(c) for c in rec.choices)
    parts.extend(mathnorm.normalize_text(d) for d in rec.diagram_conditions)
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


class Matcher:
    CONCEPT_OVERLAP_THRESHOLD = 0.5
    SEMANTIC_THRESHOLD = 0.85  # cosine on multilingual-e5; below this is noise

    def __init__(self, db: KnowledgeDB, semantic=None):
        self.db = db
        self.semantic = semantic  # SemanticRetriever | None (index optional)

    def match(self, rec: Recognition) -> MatchResult:
        ptype = rec.problem_type or "unknown"
        concepts = set(rec.concepts)

        exact = self._match_exact(rec)
        if exact is not None:
            return exact

        template = self._match_template(rec, ptype, concepts)
        if template is not None:
            return template

        concept = self._match_concept(rec, ptype, concepts)
        if concept is not None:
            return concept

        semantic = self._match_semantic(rec, concepts)
        if semantic is not None:
            return semantic

        return MatchResult(tier=Tier.NEW, concepts=sorted(concepts))

    # --- EXACT ---------------------------------------------------------------

    def _match_exact(self, rec: Recognition) -> MatchResult | None:
        candidate = self.db.find_by_text_hash(problem_hash(rec))
        equations = _without_graph_aliases(rec.equations)
        if candidate is None and equations != rec.equations:
            candidate = self.db.find_by_text_hash(
                problem_hash(rec.model_copy(update={"equations": equations}))
            )
        if candidate is None and equations:
            # Indexed candidates only: same numbers and variables, which is a
            # necessary condition for the strict equivalence checked below.
            signature = mathnorm.equations_signature(equations)
            for p in self.db.problems_by_signature(signature):
                stored = _without_graph_aliases(p.equations)
                if len(stored) != len(equations):
                    continue
                # strict: a scalar multiple is a DIFFERENT problem — its
                # parameters (and hint slot values) differ; TEMPLATE handles it
                if all(
                    mathnorm.equations_equivalent(a, b, allow_scale=False)
                    for a, b in zip(equations, stored)
                ):
                    candidate = p
                    break
        if candidate is None:
            return None
        solution = self.db.verified_solution(candidate.id)
        if solution is None:
            return None
        return MatchResult(
            tier=Tier.EXACT,
            problem=candidate,
            concepts=candidate.concepts,
            reference=solution.model_copy(update={"origin": "db", "verified": True}),
        )

    # --- TEMPLATE ------------------------------------------------------------

    def _match_template(
        self, rec: Recognition, ptype: str, concepts: set[str]
    ) -> MatchResult | None:
        # problem_type narrows the candidates: a quadratic pattern cannot be
        # the right template for a counting problem even if the algebra fits.
        templates = self.db.templates()
        if ptype != "unknown":
            preferred = [t for t in templates if t.problem_type == ptype]
            templates = preferred or templates
        for eq in rec.equations:
            for template in templates:
                bindings = mathnorm.match_template(eq, template.pattern, template.params)
                if bindings is None:
                    continue
                reference = self._instantiate(template, bindings, eq, concepts)
                if reference is not None:
                    return MatchResult(
                        tier=Tier.TEMPLATE,
                        bindings=bindings,
                        concepts=sorted(concepts) or [template.problem_type],
                        reference=reference,
                    )
        return None

    def _instantiate(
        self, template: Template, bindings: dict[str, str], equation: str, concepts: set[str]
    ) -> ReferenceSolution | None:
        try:
            instantiated_eq = mathnorm.instantiate(template.pattern, bindings)
            value = mathnorm.compute_answer(instantiated_eq, template.answer_kind)
            if not mathnorm.verify_answer([equation], template.answer_kind, value):
                return None
            steps = [
                SolutionStep(
                    idx=i + 1,
                    description=s.description.format(**bindings),
                    expression=mathnorm.instantiate(s.expression, bindings),
                )
                for i, s in enumerate(template.steps)
            ]
        except (mathnorm.ParseError, KeyError, IndexError):
            return None
        return ReferenceSolution(
            steps=steps,
            final_answer=Answer(kind=template.answer_kind, value=value),
            concepts=sorted(concepts) or [template.problem_type],
            verified=True,
            origin="template",
        )

    # --- CONCEPT -------------------------------------------------------------

    def _match_concept(
        self, rec: Recognition, ptype: str, concepts: set[str]
    ) -> MatchResult | None:
        if not concepts:
            return None
        # concept overlap is the score; sharing the problem_type is a bonus
        scored = self.db.problems_by_concepts(concepts, problem_type=ptype)
        if not scored or scored[0][1] < self.CONCEPT_OVERLAP_THRESHOLD:
            return None
        return MatchResult(
            tier=Tier.CONCEPT,
            problem=scored[0][0],
            concepts=sorted(concepts),
            reference=None,  # solver fills it, unverified
        )

    # --- SEMANTIC ------------------------------------------------------------

    def _match_semantic(self, rec: Recognition, concepts: set[str]) -> MatchResult | None:
        """Last resort before NEW: the nearest problem by text embedding.

        Reaches the imported corpus, whose tags predate the whitelist, so
        concept overlap cannot find it. Similar wording is weak evidence, so
        this never carries a solution — the solver still runs.
        """
        if self.semantic is None or not rec.problem_text:
            return None
        try:
            hits = self.semantic.search(rec.problem_text, limit=1)
        except Exception:
            log.exception("semantic retrieval failed; falling through to NEW")
            return None
        if not hits or hits[0][1] < self.SEMANTIC_THRESHOLD:
            return None
        problem, score = hits[0]
        log.info("semantic match %s (score %.3f)", problem.id, score)
        return MatchResult(
            tier=Tier.SEMANTIC,
            problem=problem,
            concepts=sorted(concepts) or problem.concepts,
            reference=None,  # solver fills it, unverified
        )
