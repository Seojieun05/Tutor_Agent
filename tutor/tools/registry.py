"""Tool registry with per-purpose allowlists.

The LLM only ever sees read-only tools; session state and hint history are
prefetched by the orchestrator into the prompt, never fetched by the model.
`phrase` is kind-scoped away from anything containing solutions or answers.
"""

from __future__ import annotations

import logging
from typing import Any

from tutor.knowledge.db import KnowledgeDB
from tutor.tools.domain_kb import DomainKBTool

log = logging.getLogger(__name__)

KB_KINDS_BY_PURPOSE: dict[str, frozenset[str]] = {
    "recognize": frozenset(),
    # answer grading: the orchestrator hands it the reference solution and the
    # misconception list directly, so it needs no tools (and no extra round trip)
    "evaluate": frozenset(),
    "solve": frozenset({"problems", "solutions", "concepts"}),
    "estimate": frozenset({"misconceptions"}),
    "phrase": frozenset({"hint_templates", "misconceptions"}),
}


class ToolRegistry:
    def __init__(self, db: KnowledgeDB):
        self.kb = DomainKBTool(db)

    def allowed_kinds(self, purpose: str) -> frozenset[str]:
        return KB_KINDS_BY_PURPOSE.get(purpose, frozenset())

    def openai_tools(self, purpose: str) -> list[dict[str, Any]]:
        kinds = self.allowed_kinds(purpose)
        if not kinds:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_domain_kb",
                    "description": (
                        "Search the verified domain knowledge base. "
                        f"Allowed kinds for this call: {sorted(kinds)}."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": sorted(kinds)},
                            "query": {
                                "type": "string",
                                "description": "Free text, or a problem id for kind=solutions.",
                            },
                            "concepts": {"type": "array", "items": {"type": "string"}},
                            "misconception_id": {"type": "string"},
                            "level": {"type": "integer", "minimum": 1, "maximum": 4},
                        },
                        "required": ["kind"],
                    },
                },
            }
        ]

    def dispatch(self, purpose: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        kinds = self.allowed_kinds(purpose)
        if name != "search_domain_kb" or not kinds:
            log.warning("blocked tool call %s for purpose %s", name, purpose)
            return {"error": f"tool {name!r} is not allowed for purpose {purpose!r}"}
        kind = args.get("kind")
        if kind not in kinds:
            log.warning("blocked kind %s for purpose %s", kind, purpose)
            return {"error": f"kind {kind!r} is not allowed for purpose {purpose!r}"}
        try:
            return self.kb.search(
                kind=kind,
                query=str(args.get("query", "") or ""),
                concepts=list(args.get("concepts") or []),
                misconception_id=args.get("misconception_id"),
                level=args.get("level"),
            )
        except Exception as e:  # tool errors go back to the model, never crash the loop
            log.exception("domain KB tool failed")
            return {"error": f"tool failed: {e}"}
