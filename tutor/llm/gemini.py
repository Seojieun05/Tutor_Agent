"""Gemini behind the same seam as Grok, for the two jobs it is better at.

Nothing here is all-or-nothing: each job picks its own model, and the rest of
the pipeline never learns which one answered.

    VISION_PROVIDER=gemini   reads the worksheet
    HINT_PROVIDER=gemini     writes what the tutor says

The second one is the interesting choice. Google fine-tuned Gemini on LearnLM,
their learning-science work, specifically so that pedagogical system
instructions — "act as a supportive math tutor", "give a nudge, not the
answer" — are followed as behaviour rather than as a style request. That is
exactly the instruction this tutor gives, and exactly the instruction a general
model tends to drift out of by helpfully finishing the problem.

`run_with_tools` falls through to `complete_json`, which is honest rather than
lazy. The purposes routed here — recognize, phrase, explain — are the ones this
codebase deliberately feeds by prefetching: the orchestrator puts the state,
the history, the target step and the misconception into the prompt rather than
letting the model go looking (CLAUDE.md, "Do not rely on autonomous tool
calls"). A tool loop here would be dead code pretending to be a feature.

What does NOT move with the model: the policy that chose the hint level, and
the leak guard that checks the result. Both are deterministic and both still
run. Changing the writer cannot change how much is given away.
"""

from __future__ import annotations

import logging
from typing import Sequence, TypeVar

from pydantic import BaseModel

from tutor.config import Settings
from tutor.llm.client import LLMError, parse_into

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiClient:
    """An LLMClient backed by one Gemini model. Import is lazy so a missing
    google-genai costs nothing until someone actually asks for this provider."""

    def __init__(self, settings: Settings, model: str | None = None, role: str = "gemini"):
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise LLMError(
                f"{role} needs the google-genai package: "
                "pip install -e \".[gemini]\""
            ) from e

        self.settings = settings
        self.model = model or settings.gemini_vision_model or DEFAULT_MODEL
        self._genai = genai
        self._client = self._connect(settings, role)
        log.info("%s: gemini %s via %s", role, self.model, self.backend)

    def _connect(self, settings: Settings, role: str):
        """Vertex AI if a project is configured, the AI Studio key otherwise.

        Two doors to the same models, and which one you have matters:
        an AI Studio key bills prepaid credits and answers 429 "prepayment
        credits are depleted" once they run out, while a Cloud project can
        spend its own (often free) credits. The model ids are the same; the
        location is not — gemini-3.1-pro-preview lives in `global` on Vertex and
        answers 404 in us-central1.
        """
        from google import genai

        if settings.vertex_project:
            self.backend = f"vertex ai ({settings.vertex_project}/{settings.vertex_location})"
            # Credentials come from ADC: `gcloud auth application-default login`.
            return genai.Client(
                vertexai=True,
                project=settings.vertex_project,
                location=settings.vertex_location,
            )
        if not settings.google_api_key:
            raise LLMError(
                f"{role} needs either GOOGLE_API_KEY in .env "
                "(https://aistudio.google.com/apikey) or VERTEX_PROJECT plus "
                "`gcloud auth application-default login`."
            )
        self.backend = "ai studio"
        return genai.Client(api_key=settings.google_api_key)

    def complete_json(self, *, purpose, system, user, images=(), schema: type[M]) -> M:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg") for jpeg in images
        ]
        parts.append(types.Part.from_text(text=user))
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                ),
            )
        except Exception as e:  # noqa: BLE001 — one seam, one error type
            raise LLMError(f"{purpose}: gemini call failed: {e}") from e

        # .parsed is the SDK's own validation of response_schema; the text
        # fallback covers the versions and refusals where it comes back None.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise LLMError(f"{purpose}: gemini returned nothing to parse")
        try:
            return parse_into(schema, text)
        except Exception as e:  # noqa: BLE001 — same seam
            raise LLMError(f"{purpose}: gemini output failed validation: {e}") from e

    def run_with_tools(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[M],
        max_rounds: int = 6,
    ) -> M:
        """No tools: `recognize` is the only purpose routed here, and it has none."""
        return self.complete_json(
            purpose=purpose, system=system, user=user, images=images, schema=schema
        )
