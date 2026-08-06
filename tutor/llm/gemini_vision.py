"""Gemini as a second pair of eyes, behind the same seam as Grok.

Reading a photo and teaching from it are different jobs, and only the first one
is a vision problem. So this swaps the *recognizer's* model and nothing else:
the solver, the estimator, the hint generator, STT and TTS all stay on Grok,
and the pedagogy never learns which model read the worksheet.

    VISION_PROVIDER=gemini python server.py

It implements complete_json properly and lets run_with_tools fall through to it,
which is honest rather than lazy: the only purpose that reaches this client is
`recognize`, and recognize is the one purpose with an empty tool allowlist
(tutor/tools/registry.py). A tool loop here would be dead code pretending to
be a feature.
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


class GeminiVisionClient:
    """LLMClient for image reading only. Import is lazy so a missing google-genai
    costs nothing until someone actually asks for this provider."""

    def __init__(self, settings: Settings):
        if not settings.google_api_key:
            raise LLMError(
                "VISION_PROVIDER=gemini needs GOOGLE_API_KEY in .env "
                "(https://aistudio.google.com/apikey)."
            )
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise LLMError(
                "VISION_PROVIDER=gemini needs the google-genai package: "
                "pip install -e \".[vision-gemini]\""
            ) from e

        self.settings = settings
        self.model = settings.gemini_vision_model
        self._genai = genai
        self._client = genai.Client(api_key=settings.google_api_key)
        log.info("vision provider: gemini (%s)", self.model)

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
