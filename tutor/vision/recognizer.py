"""Grok VLM problem + student-work recognition (spec: Grok VLM only for MVP)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tutor.config import Settings
from tutor.llm.client import LLMClient


class Recognition(BaseModel):
    problem_text: str
    equations: list[str] = []
    choices: list[str] = []  # multiple-choice options, if any
    diagram_conditions: list[str] = []  # facts read off a figure, e.g. "각 B = 90도"
    student_work: list[str] = []  # ordered lines as written
    uncertain_regions: list[str] = []
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Filled by tutor.knowledge.tagger.ConceptTagger AFTER recognition, never
    # by the VLM: reading the paper and classifying the maths are separate jobs
    # (and separately debuggable). Both come from the whitelists.
    problem_type: str = "unknown"
    concepts: list[str] = []


_SYSTEM = """You read a photo of a math worksheet. Transcribe exactly what is written.

Rules:
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

Return ONLY the JSON object."""


class Recognizer:
    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        # Optional so tests and scripts can build a recognizer with nothing but
        # a model. Without settings the frame is sent exactly as photographed.
        self.settings = settings

    def recognize(self, jpeg: bytes) -> Recognition:
        jpeg = self._framed(jpeg)
        return self.llm.complete_json(
            purpose="recognize",
            system=_SYSTEM,
            user="Transcribe this worksheet photo into the JSON schema.",
            images=[jpeg],
            schema=Recognition,
        )

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
