"""Tool registry with per-purpose allowlists.

Two tool families, both read-only: `search_domain_kb` over the knowledge DB,
and the sympy checks (`compute`, `check_equivalence`) for purposes whose
correctness turns on arithmetic. Session state and hint history are prefetched
by the orchestrator into the prompt, never fetched by the model. `phrase` is
kind-scoped away from anything containing solutions or answers — and gets no
math tools either, so it cannot compute its way to the answer it must not say.
"""

from __future__ import annotations

import logging
from typing import Any

from tutor.knowledge.db import KnowledgeDB
from tutor.tools import mathtools
from tutor.tools.domain_kb import DomainKBTool

log = logging.getLogger(__name__)

KB_KINDS_BY_PURPOSE: dict[str, frozenset[str]] = {
    "recognize": frozenset(),
    # answer grading: the orchestrator hands it the reference solution and the
    # misconception list directly, so no KB lookups (it does get the sympy
    # checks below — equivalence is a judgement, prefetching cannot replace it)
    "evaluate": frozenset(),
    # concept tagging: the whitelists are in the prompt, so there is nothing
    # to look up — and a KB lookup could only tempt it to invent ids
    "tag": frozenset(),
    # deciding what an utterance is FOR: it reads one transcript and two flags,
    # and it runs before every turn — a tool round trip would cost the latency
    # the rule fast-path exists to save
    "intent": frozenset(),
    "solve": frozenset({"problems", "solutions", "concepts"}),
    "estimate": frozenset({"misconceptions"}),
    "phrase": frozenset({"hint_templates", "misconceptions"}),
    # answering a student's "why?": same reach as phrasing a hint — concepts and
    # misconceptions, never solutions
    "explain": frozenset({"hint_templates", "misconceptions"}),
}

MATH_TOOLS_BY_PURPOSE: dict[str, frozenset[str]] = {
    # the solver checks its own steps while writing them; the after-the-fact
    # machine check in grok_solver still runs and still decides what is stored
    "solve": frozenset({"compute", "check_equivalence"}),
    # grading a spoken answer is sometimes an equivalence judgement ("3x가 15"
    # against a reference step "x = 5"), not a wording one. The prompt tells
    # the model to call only when genuinely unsure — every round trip is a
    # beat of silence the student sits through.
    "evaluate": frozenset({"compute", "check_equivalence"}),
}

_MATH_TOOL_DEFS: dict[str, dict[str, Any]] = {
    "compute": {
        "type": "function",
        "function": {
            "name": "compute",
            "description": (
                "Evaluate/simplify a math expression, or solve a one-variable "
                "equation. sympy syntax: * for multiplication, ** for powers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '3*5 + 2' or '3*x + 5 = 20'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "check_equivalence": {
        "type": "function",
        "function": {
            "name": "check_equivalence",
            "description": (
                "Check whether two equations (or two bare expressions) are "
                "mathematically equivalent: same solution set / same value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                },
                "required": ["a", "b"],
            },
        },
    },
}


class ToolRegistry:
    def __init__(self, db: KnowledgeDB):
        self.kb = DomainKBTool(db)

    def allowed_kinds(self, purpose: str) -> frozenset[str]:
        return KB_KINDS_BY_PURPOSE.get(purpose, frozenset())

    def allowed_math_tools(self, purpose: str) -> frozenset[str]:
        return MATH_TOOLS_BY_PURPOSE.get(purpose, frozenset())

    def openai_tools(self, purpose: str) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        kinds = self.allowed_kinds(purpose)
        if kinds:
            tools.append(
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
            )
        tools.extend(
            _MATH_TOOL_DEFS[name] for name in sorted(self.allowed_math_tools(purpose))
        )
        return tools

    def dispatch(self, purpose: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in _MATH_TOOL_DEFS:
            if name not in self.allowed_math_tools(purpose):
                log.warning("blocked tool call %s for purpose %s", name, purpose)
                return {"error": f"tool {name!r} is not allowed for purpose {purpose!r}"}
            if name == "compute":
                return mathtools.compute(str(args.get("expression") or ""))
            return mathtools.check_equivalence(
                str(args.get("a") or ""), str(args.get("b") or "")
            )
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
