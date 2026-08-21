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

`run_with_tools` runs a real function-calling loop only when a registry is
attached AND it offers tools for the purpose — today that is `evaluate` with
the sympy checks (EVAL_PROVIDER=gemini). Everything else falls through to
`complete_json`, which is honest rather than lazy: recognize, phrase and
explain are deliberately fed by prefetching — the orchestrator puts the state,
the history, the target step and the misconception into the prompt rather than
letting the model go looking (CLAUDE.md, "Do not rely on autonomous tool
calls"), so no registry is attached to those clients and no loop pretends to
be a feature there.

What does NOT move with the model: the policy that chose the hint level, and
the leak guard that checks the result. Both are deterministic and both still
run. Changing the writer cannot change how much is given away.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel

from tutor.config import Settings
from tutor.llm.client import JsonStringFieldStream, LLMError, parse_into

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# 모델을 지정하지 않았을 때 쓰는 기본 Gemini 모델.
DEFAULT_MODEL = "gemini-3.6-flash"


# Grok과 똑같은 인터페이스를 가진 Gemini 클라이언트. 파이프라인은 누가 답했는지 알지 못한다.
class GeminiClient:
    """An LLMClient backed by one Gemini model. Import is lazy so a missing
    google-genai costs nothing until someone actually asks for this provider."""

    # 역할(role)·모델 이름·툴 레지스트리를 받아 접속을 준비한다.
    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        role: str = "gemini",
        registry=None,
    ):
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise LLMError(
                f"{role} needs the google-genai package: "
                "pip install -e \".[gemini]\""
            ) from e

        self.settings = settings
        self.model = model or settings.gemini_vision_model or DEFAULT_MODEL
        self.registry = registry  # None → run_with_tools falls through, toolless
        self._genai = genai
        self._client = self._connect(settings, role)
        log.info("%s: gemini %s via %s", role, self.model, self.backend)

    # AI Studio 키 또는 Vertex(ADC) 중 가능한 경로로 붙는다.
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

    # 한 번 호출해 JSON 하나를 받는다(이미지 첨부 가능).
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

    # 지정한 문자열 필드를 오는 대로 흘려보내면서, 전체 JSON은 따로 검증한다.
    def complete_json_stream(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[M],
        text_field: str,
        on_text_delta: Callable[[str], None],
    ) -> M:
        """Stream one field from Gemini's structured JSON response."""
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg") for jpeg in images
        ]
        parts.append(types.Part.from_text(text=user))
        extractor = JsonStringFieldStream(text_field)
        text = ""
        try:
            responses = self._client.models.generate_content_stream(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                ),
            )
            for response in responses:
                delta = getattr(response, "text", None) or ""
                if not delta:
                    continue
                text += delta
                visible = extractor.feed(delta)
                if visible:
                    on_text_delta(visible)
        except Exception as e:  # noqa: BLE001 — fallback seam handles provider failures
            raise LLMError(f"{purpose}: gemini streaming call failed: {e}") from e
        if not text.strip():
            raise LLMError(f"{purpose}: gemini returned nothing to parse")
        try:
            return parse_into(schema, text)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"{purpose}: gemini streamed output failed validation: {e}") from e

    # 툴이 붙어 있는 용도(evaluate 등)만 실제 함수 호출 루프를 돈다. 나머지는 단순 호출로 내려간다.
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
        """Function-calling loop when the registry offers tools; fallthrough
        otherwise. Any failure raises LLMError, which is FallbackLLM's cue to
        rerun the whole call on the Grok standby (whose loop has the same
        tools) — so a Gemini quirk costs a retry, never the turn."""
        tools = self.registry.openai_tools(purpose) if self.registry is not None else []
        if not tools:
            return self.complete_json(
                purpose=purpose, system=system, user=user, images=images, schema=schema
            )
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"]["description"],
                parameters=t["function"]["parameters"],
            )
            for t in tools
        ]
        parts = [
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg") for jpeg in images
        ]
        parts.append(types.Part.from_text(text=user))
        contents = [types.Content(role="user", parts=parts)]
        # tools and response_schema do not combine reliably, so the schema
        # rides in the system instruction and the final text is validated here
        instruction = (
            f"{system}\n\nRespond with ONLY a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        config = types.GenerateContentConfig(
            system_instruction=instruction,
            tools=[types.Tool(function_declarations=declarations)],
            temperature=0,
        )
        for round_no in range(1, max_rounds + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as e:  # noqa: BLE001 — one seam, one error type
                raise LLMError(f"{purpose}: gemini call failed: {e}") from e
            calls = list(getattr(response, "function_calls", None) or [])
            if not calls:
                if round_no > 1:
                    log.info("%s: gemini answered after %d rounds", purpose, round_no)
                text = getattr(response, "text", None) or ""
                if not text.strip():
                    raise LLMError(f"{purpose}: gemini returned nothing to parse")
                try:
                    return parse_into(schema, text)
                except Exception as e:  # noqa: BLE001 — same seam
                    # One retry with the error in hand, exactly as the grok
                    # seam does. Without it, tool-loop calls got a single shot
                    # at valid JSON where every other call got two, and a
                    # solution good enough to grade against died of a missing
                    # field it could have added on request.
                    log.warning(
                        "%s: invalid JSON from gemini, retrying once (%s)", purpose, e
                    )
                    contents.append(response.candidates[0].content)
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(
                                text=f"Your previous response did not match the schema: {e}. "
                                     "Respond again with ONLY the corrected JSON object."
                            )],
                        )
                    )
                    try:
                        retry = self._client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=instruction, temperature=0
                            ),
                        )
                        return parse_into(schema, getattr(retry, "text", None) or "")
                    except Exception as e2:  # noqa: BLE001 — same seam
                        raise LLMError(
                            f"{purpose}: gemini output failed validation twice: {e2}"
                        ) from e2
            log.info(
                "%s round %d: gemini looking up %s",
                purpose, round_no, ", ".join(c.name for c in calls),
            )
            contents.append(response.candidates[0].content)
            results = [
                types.Part.from_function_response(
                    name=call.name,
                    response=self.registry.dispatch(purpose, call.name, dict(call.args or {})),
                )
                for call in calls
            ]
            contents.append(types.Content(role="user", parts=results))
        # Out of rounds, same as the grok seam: ask once more with the tools
        # withdrawn rather than discarding every round trip that got us here.
        log.warning(
            "%s: gemini out of tool rounds after %d; asking for the answer without tools",
            purpose, max_rounds,
        )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text="No more tool calls are available. Answer now, using what "
                         "you already worked out — do not request another tool call."
                )],
            )
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=instruction, temperature=0
                ),
            )
            return parse_into(schema, getattr(response, "text", None) or "")
        except Exception as e:  # noqa: BLE001 — one seam, one error type
            raise LLMError(f"{purpose}: gemini ran out of tool rounds: {e}") from e
