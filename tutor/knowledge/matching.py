"""Matching cascade: EXACT → TEMPLATE → CONCEPT → NEW.

Equivalence is symbolic (sympy), never embedding-based. TEMPLATE hits are
recomputed and sympy-verified before being treated as verified knowledge.
"""

from __future__ import annotations

import hashlib
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


def tag_concepts(rec: Recognition) -> tuple[str, set[str]]:
    """Rule-based tagger: infer problem_type + concept tags from equations/text."""
    text = rec.problem_text
    for eq in rec.equations:
        pre = eq.replace(" ", "")
        if "Derivative" in pre or re.search(r"d/d[a-zA-Z]", pre):
            return "derivative", {"differentiation"}
        try:
            residual, is_eq = mathnorm.parse_equation(eq)
        except mathnorm.ParseError:
            continue
        if residual.has(*[s for s in residual.free_symbols]):
            try:
                var = next(iter(residual.free_symbols))
                degree = residual.as_poly(var).degree() if residual.as_poly(var) else 0
            except Exception:
                continue
            if is_eq and degree == 2:
                return "quadratic_equation", {"quadratic_equation"}
            if is_eq and degree == 1:
                return "linear_equation", {"linear_equation"}
    if "미분" in text or "도함수" in text:
        return "derivative", {"differentiation"}
    if "이차방정식" in text:
        return "quadratic_equation", {"quadratic_equation"}
    if "방정식" in text:
        return "linear_equation", {"linear_equation"}
    return "unknown", set()


class Matcher:
    CONCEPT_OVERLAP_THRESHOLD = 0.5

    def __init__(self, db: KnowledgeDB):
        self.db = db

    def match(self, rec: Recognition) -> MatchResult:
        ptype, concepts = tag_concepts(rec)

        exact = self._match_exact(rec)
        if exact is not None:
            return exact

        template = self._match_template(rec, ptype, concepts)
        if template is not None:
            return template

        concept = self._match_concept(rec, concepts)
        if concept is not None:
            return concept

        return MatchResult(tier=Tier.NEW, concepts=sorted(concepts))

    # --- EXACT ---------------------------------------------------------------

    def _match_exact(self, rec: Recognition) -> MatchResult | None:
        candidate = self.db.find_by_text_hash(problem_hash(rec))
        if candidate is None and rec.equations:
            # Indexed candidates only: same numbers and variables, which is a
            # necessary condition for the strict equivalence checked below.
            signature = mathnorm.equations_signature(rec.equations)
            for p in self.db.problems_by_signature(signature):
                if len(p.equations) != len(rec.equations):
                    continue
                # strict: a scalar multiple is a DIFFERENT problem — its
                # parameters (and hint slot values) differ; TEMPLATE handles it
                if all(
                    mathnorm.equations_equivalent(a, b, allow_scale=False)
                    for a, b in zip(rec.equations, p.equations)
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
        for eq in rec.equations:
            for template in self.db.templates():
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

    def _match_concept(self, rec: Recognition, concepts: set[str]) -> MatchResult | None:
        if not concepts:
            return None
        scored = self.db.problems_by_concepts(concepts)
        if not scored or scored[0][1] < self.CONCEPT_OVERLAP_THRESHOLD:
            return None
        return MatchResult(
            tier=Tier.CONCEPT,
            problem=scored[0][0],
            concepts=sorted(concepts),
            reference=None,  # solver fills it, unverified
        )
