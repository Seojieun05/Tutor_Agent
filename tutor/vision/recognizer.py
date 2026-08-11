"""VLM problem + student-work recognition, tags included.

One call answers both "what is written on this paper" and "what kind of problem
is this". Classification used to be a second round trip (ConceptTagger) for
debuggability, and cost a measured ~7s on every new problem — half of it spent
re-sending what the VLM had just read. The separation that actually matters is
kept: the whitelists are enforced in PYTHON after the call, so an invented id
still dies here and never reaches the KB or the retrieval, and a wrong tag can
still be reproduced offline via tutor.knowledge.tagger.ConceptTagger, which
remains the standalone version of the same job.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from tutor.config import Settings
from tutor.knowledge.concepts import MAX_CONCEPTS, concepts_for_prompt, normalize_concepts
from tutor.knowledge.taxonomy import (
    UNKNOWN_PROBLEM_TYPE,
    normalize_problem_type,
    problem_types_for_prompt,
)
from tutor.llm.client import LLMClient

log = logging.getLogger(__name__)

# Something is being related or operated on — as opposed to "a_10", which is
# the quantity the question asks for, not a claim about it.
_OPERATOR = re.compile(r"[=<>≤≥+\-*/^√]")


class Recognition(BaseModel):
    problem_text: str
    equations: list[str] = []
    choices: list[str] = []  # multiple-choice options, if any
    diagram_conditions: list[str] = []  # facts read off a figure, e.g. "각 B = 90도"
    student_work: list[str] = []  # ordered lines as written
    uncertain_regions: list[str] = []
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Filled by the SAME call, but never trusted from it: Recognizer.recognize
    # re-validates both against the whitelists before anything downstream reads
    # them, so an invented id becomes "unknown"/dropped rather than a KB key.
    problem_type: str = UNKNOWN_PROBLEM_TYPE
    concepts: list[str] = []


def _system_prompt() -> str:
    return f"""You read a photo of a math worksheet. Two jobs, one answer:
transcribe exactly what is written, then classify the problem.

TRANSCRIPTION rules:
- Separate the printed/original problem from the student's handwritten work.
- problem_text: the problem statement as written (Korean or English).
- equations: every equation/expression of the PROBLEM in ASCII math, e.g. "3*x + 5 = 20",
  "Derivative(x**3 + 2*x, x)" for d/dx expressions. Use * for multiplication and ** for powers.
- choices: multiple-choice options if present, else [].
- diagram_conditions: facts stated by a figure/diagram (e.g. "angle B = 90"), else [].
- student_work: each handwritten line as a separate ASCII math string, top to bottom.
  Transcribe the student's work faithfully, INCLUDING their mistakes. Do not correct it.
- uncertain_regions: describe every part you cannot read confidently. Never guess silently.
- confidence: your overall reading confidence in [0, 1].

CLASSIFICATION rules:
- problem_type — the ONE coarse type of the problem, chosen from this list:
{problem_types_for_prompt()}
- concepts — the curriculum concepts a student actually needs to SOLVE it,
  {MAX_CONCEPTS} at most (fewer is better; [] is fine), chosen from this list:
{concepts_for_prompt()}
- Use ids EXACTLY as written above. NEVER invent, translate or modify an id.
- concepts are the knowledge required to solve, not keywords from the wording.
  Pick what the solution steps rely on. If the problem merely mentions a
  triangle but is solved by an equation, tag the equation concept.
- Do NOT tag solution strategies (case splitting, arrangement tricks, working
  backwards). Those are not curriculum concepts.
- If you cannot classify confidently, use problem_type "unknown" and concepts [].
  Guessing is worse than admitting uncertainty. A low READING confidence does
  not force "unknown": classify whatever you did read clearly.

Return ONLY the JSON object."""


class Recognizer:
    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        # Optional so tests and scripts can build a recognizer with nothing but
        # a model. Without settings the frame is sent exactly as photographed.
        self.settings = settings

    def recognize(self, jpeg: bytes) -> Recognition:
        jpeg = self._framed(jpeg)
        rec = self.llm.complete_json(
            purpose="recognize",
            system=_system_prompt(),
            user="Transcribe and classify this worksheet photo into the JSON schema.",
            images=[jpeg],
            schema=Recognition,
        )
        # A lone term is not an equation. The model lists the thing being
        # asked about ("a_10") beside the real ones, and downstream that term
        # is a third equation that never matches anything: EXACT compares
        # lists pairwise and by LENGTH, so one stray entry pushes a known
        # problem down to CONCEPT and starts a solver run for nothing.
        # Operator-free is the test, not relation-free — "x**2 - 4" from a
        # factorization problem is an equation list entry worth keeping.
        kept = [e for e in rec.equations if _OPERATOR.search(e)]
        if kept != rec.equations:
            log.info("dropped %d bare term(s) from equations: %s",
                     len(rec.equations) - len(kept),
                     [e for e in rec.equations if e not in kept])
            rec.equations = kept

        # Whitelist enforcement in Python — never trust the model's ids.
        said_type, said_concepts = rec.problem_type, rec.concepts
        rec.problem_type = normalize_problem_type(rec.problem_type)
        rec.concepts = normalize_concepts(rec.concepts)
        if (rec.problem_type, rec.concepts) != (said_type, said_concepts):
            log.info(
                "tags normalized: problem_type=%s concepts=%s (model said %s / %s)",
                rec.problem_type, rec.concepts, said_type, said_concepts,
            )
        return rec

    def _framed(self, jpeg: bytes) -> bytes:
        """Crop the desk away before the model spends its tile budget on it."""
        settings = self.settings
        if settings is None or not (settings.worksheet_roi or settings.auto_crop):
            return jpeg
        from tutor.vision import framing

        return framing.prepare_for_reading(
            jpeg,
            roi=settings.worksheet_roi,
            auto=settings.auto_crop,
            target_px=settings.crop_target_px,
        )
