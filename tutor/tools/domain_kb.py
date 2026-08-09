"""Domain KB search tool. Read-only, like every tool the LLM can call
(the full roster per purpose lives in tutor/tools/registry.py)."""

from __future__ import annotations

from typing import Any

import logging
import threading
import time

from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB

log = logging.getLogger(__name__)

KB_KINDS = ("problems", "solutions", "concepts", "misconceptions", "hint_templates")


def load_semantic_retriever(db: KnowledgeDB):
    """Semantic search if this machine has it, None if it does not.

    The embedding index (data/problem_embeddings.npz) and sentence-transformers
    are an optional extra: a laptop running the server for a demo should not
    have to download torch and copy a 22 MB index before the tutor will start.
    Everything except the SEMANTIC retrieval tier works without them.
    """
    try:
        from tutor.retrieval.semantic import SemanticRetriever

        return SemanticRetriever(db)
    except ImportError as e:
        log.warning("semantic search off (pip install -e '.[rag]' to enable): %s", e)
    except FileNotFoundError as e:
        log.warning(
            "semantic search off (no embedding index; "
            "build it with python -m tutor.scripts.build_embeddings): %s", e
        )
    except Exception:
        log.exception("semantic search off: the index could not be loaded")
    return None


class LazySemantic:
    """The embedding index, loaded off the startup path.

    Importing torch and sentence-transformers and reading the 22 MB index takes
    about 15 seconds, and it used to happen before the socket was even open —
    so `python server.py` looked hung, and a student opening the page early got
    a connection refused. None of it is needed to accept a connection, and most
    of it is not needed for the first hint either: SEMANTIC is the fourth tier
    of the matching cascade, behind EXACT, TEMPLATE and CONCEPT.

    Loading in a thread and *waiting only when someone actually searches* keeps
    both properties: the server is up in a second, and nobody silently loses a
    retrieval tier. Every caller is already on a worker thread (matcher.match
    and the tool loop both run under asyncio.to_thread), so blocking here
    blocks nothing else.
    """

    def __init__(self, db: KnowledgeDB, timeout_s: float = 90.0):
        self.timeout_s = timeout_s
        self._ready = threading.Event()
        self._inner = None
        self._started = time.monotonic()
        threading.Thread(
            target=self._load, args=(db,), name="semantic-index", daemon=True
        ).start()

    def _load(self, db: KnowledgeDB) -> None:
        try:
            self._inner = load_semantic_retriever(db)
        finally:
            self._ready.set()
            if self._inner is not None:
                log.info(
                    "semantic index ready after %.1fs", time.monotonic() - self._started
                )

    def search(self, *args, **kwargs):
        if not self._ready.is_set():
            waited = time.monotonic() - self._started
            log.info("waiting for the semantic index (%.1fs so far)", waited)
            if not self._ready.wait(self.timeout_s):
                # Better a weaker retrieval tier than a student waiting forever.
                log.warning("semantic index still not ready; searching without it")
                return []
        return self._inner.search(*args, **kwargs) if self._inner is not None else []


class DomainKBTool:
    def __init__(self, db: KnowledgeDB, semantic_in_background: bool = True):
        self.db = db
        self.semantic = (
            LazySemantic(db) if semantic_in_background else load_semantic_retriever(db)
        )

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

            # 3. semantic fallback (only where the index is installed)
            if query and self.semantic is not None:
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
