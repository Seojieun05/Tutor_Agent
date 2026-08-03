"""Domain KB search tool: the only tool the LLM can call. Read-only."""

from __future__ import annotations

from typing import Any

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.retrieval.semantic import SemanticRetriever

KB_KINDS = ("problems", "solutions", "concepts", "misconceptions", "hint_templates")


class DomainKBTool:
    def __init__(self, db: KnowledgeDB):
        self.db = db
        self.semantic = SemanticRetriever(db)

    def search(
        self,
        kind: str,
        query: str = "",
        concepts: list[str] | None = None,
        misconception_id: str | None = None,
        level: int | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        concepts = concepts or []
        if kind == "concepts":
            return {"concepts": [{"id": k, "name": v} for k, v in self.db.concepts().items()]}
        if kind == "problems":
            needle = mathnorm.normalize_text(query) if query else ""

            # 1. 기존 lexical 검색
            lexical_hits = [
                p
                for p in self.db.all_problems()
                if needle
                and needle in mathnorm.normalize_text(p.problem_text)
            ]

            if lexical_hits:
                return {
                    "problems": [
                        {
                            "id": p.id,
                            "problem_type": p.problem_type,
                            "problem_text": p.problem_text,
                            "equations": p.equations,
                            "concepts": p.concepts,
                            "retrieval": "lexical",
                        }
                        for p in lexical_hits[:limit]
                    ]
                }

            # 2. concept 검색
            if concepts:
                concept_hits = self.db.problems_by_concepts(
                    set(concepts)
            )

                if concept_hits:
                    return {
                        "problems": [
                            {
                                "id": p.id,
                                "problem_type": p.problem_type,
                                "problem_text": p.problem_text,
                                "equations": p.equations,
                                "concepts": p.concepts,
                                "retrieval": "concept",
                                "score": score,
                            }
                            for p, score in concept_hits[:limit]
                        ]
                    }

            # 3. semantic fallback
            if query:
                semantic_hits = self.semantic.search(
                    query,
                    limit=limit,
                )

                return {
                    "problems": [
                        {
                            "id": p.id,
                            "problem_type": p.problem_type,
                            "problem_text": p.problem_text,
                            "equations": p.equations,
                            "concepts": p.concepts,
                            "retrieval": "semantic",
                            "score": score,
                        }
                        for p, score in semantic_hits
                    ]
                }

            return {"problems": []}
        if kind == "solutions":
            if not query:
                return {"error": "solutions search needs query=<problem_id>"}
            solution = self.db.verified_solution(query)
            if solution is None:
                return {"solutions": []}
            return {"solutions": [solution.model_dump()]}
        if kind == "misconceptions":
            if misconception_id:
                m = self.db.get_misconception(misconception_id)
                items = [m] if m else []
            else:
                items = self.db.misconceptions_for(concepts)
            return {"misconceptions": [m.model_dump() for m in items[:limit] if m]}
        if kind == "hint_templates":
            items = self.db.hint_templates_for(concepts, misconception_id, level)
            return {"hint_templates": [h.model_dump() for h in items[:limit]]}
        return {"error": f"unknown kind {kind!r}"}
