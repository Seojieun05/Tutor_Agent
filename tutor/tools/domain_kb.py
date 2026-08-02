"""Domain KB search tool: the only tool the LLM can call. Read-only."""

from __future__ import annotations

from typing import Any

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB

KB_KINDS = ("problems", "solutions", "concepts", "misconceptions", "hint_templates")


class DomainKBTool:
    def __init__(self, db: KnowledgeDB):
        self.db = db

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
            hits = [
                p
                for p in self.db.all_problems()
                if not needle
                or needle in mathnorm.normalize_text(p.problem_text)
                or (set(p.concepts) & set(concepts))
            ]
            return {
                "problems": [
                    {
                        "id": p.id,
                        "problem_type": p.problem_type,
                        "problem_text": p.problem_text,
                        "equations": p.equations,
                        "concepts": p.concepts,
                    }
                    for p in hits[:limit]
                ]
            }
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
