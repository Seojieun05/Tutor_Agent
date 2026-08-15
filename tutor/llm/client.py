"""Grok LLM seam: every model call in the pipeline goes through this interface.

GrokClient talks to the xAI OpenAI-compatible /chat/completions endpoint with
one multimodal CHAT_MODEL. EchoLLMClient (tutor/llm/echo.py) implements the
same interface with canned outputs for echo mode and tests.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Callable, Protocol, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from tutor.config import Settings
from tutor.llm import timing
from tutor.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# Purposes the student waits on mid-conversation. `solve` and `recognize` are
# excluded: a wrong reference solution or a misread worksheet costs far more
# than the seconds saved.
FAST_PURPOSES = frozenset({"evaluate", "phrase", "estimate", "tag", "explain"})


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[M],
    ) -> M: ...

    def run_with_tools(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        images: Sequence[bytes] = (),
        schema: type[M],
        max_rounds: int = 6,
    ) -> M: ...

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
    ) -> M: ...


class JsonStringFieldStream:
    """Incrementally decode one JSON string field from arbitrary SSE chunks.

    The model still returns the complete structured object for validation. This
    small parser only exposes characters inside (for example) ``"hint"`` while
    that object is arriving; JSON syntax and later fields such as ``board``
    never enter the student-facing stream.
    """

    _ESCAPES = {
        '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
        "n": "\n", "r": "\r", "t": "\t",
    }

    def __init__(self, field: str):
        self.field = field
        self._source = ""
        self._pos = 0
        self._started = False
        self._done = False
        self._escaped = False
        self._unicode = ""

    def feed(self, chunk: str) -> str:
        if not chunk or self._done:
            return ""
        self._source += chunk
        if not self._started:
            match = re.search(rf'"{re.escape(self.field)}"\s*:\s*"', self._source)
            if match is None:
                # Keep a bounded suffix: enough for a key split across chunks,
                # without retaining an arbitrarily large prefix before it.
                if len(self._source) > 256:
                    self._source = self._source[-256:]
                return ""
            self._started = True
            self._pos = match.end()

        out: list[str] = []
        while self._pos < len(self._source) and not self._done:
            ch = self._source[self._pos]
            self._pos += 1
            if self._unicode:
                if ch.lower() not in "0123456789abcdef":
                    # Invalid JSON will be handled by the normal schema retry;
                    # do not expose a malformed escape as student text.
                    self._done = True
                    break
                self._unicode += ch
                if len(self._unicode) == 5:  # 'u' + four hex digits
                    out.append(chr(int(self._unicode[1:], 16)))
                    self._unicode = ""
                    self._escaped = False
                continue
            if self._escaped:
                if ch == "u":
                    self._unicode = "u"
                elif ch in self._ESCAPES:
                    out.append(self._ESCAPES[ch])
                    self._escaped = False
                else:
                    self._done = True
                continue
            if ch == "\\":
                self._escaped = True
            elif ch == '"':
                self._done = True
            else:
                out.append(ch)
        return "".join(out)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_into(schema: type[M], text: str) -> M:
    return schema.model_validate_json(_strip_fences(text))


class GrokClient:
    def __init__(self, settings: Settings, registry: ToolRegistry):
        from openai import OpenAI

        self.settings = settings
        self.registry = registry
        self._client = OpenAI(api_key=settings.xai_api_key, base_url=settings.xai_base_url)

    def _user_content(self, user: str, images: Sequence[bytes]):
        if not images:
            return user
        content: list[dict] = [{"type": "text", "text": user}]
        for jpeg in images:
            b64 = base64.b64encode(jpeg).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        return content

    def _schema_reminder(self, schema: type[BaseModel]) -> str:
        return (
            "Respond with ONLY a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )

    def _tuning(self, purpose: str) -> dict:
        """Per-purpose latency knobs.

        grok-4.5 spends most of its wall clock on hidden reasoning tokens; for
        the conversational purposes that roughly halves the turn for no quality
        loss (measured: 9.6s → 4.3s on an answer evaluation). Correctness-
        critical purposes (solve) keep the model's default effort.
        """
        effort = self.settings.llm_reasoning_effort
        if effort and purpose in FAST_PURPOSES:
            return {"reasoning_effort": effort}
        return {}

    def complete_json(self, *, purpose, system, user, images=(), schema):
        messages = [
            {"role": "system", "content": f"{system}\n\n{self._schema_reminder(schema)}"},
            {"role": "user", "content": self._user_content(user, images)},
        ]
        return self._final_json(purpose, messages, schema)

    def complete_json_stream(
        self, *, purpose, system, user, images=(), schema, text_field, on_text_delta
    ):
        """Validate a normal JSON response while exposing one string field live."""
        messages = [
            {"role": "system", "content": f"{system}\n\n{self._schema_reminder(schema)}"},
            {"role": "user", "content": self._user_content(user, images)},
        ]
        extractor = JsonStringFieldStream(text_field)
        content = ""
        stream = self._client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            stream=True,
            **self._tuning(purpose),
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            content += delta
            visible = extractor.feed(delta)
            if visible:
                on_text_delta(visible)
        messages.append({"role": "assistant", "content": content})
        return self._parse_or_retry(purpose, messages, content, schema)

    def run_with_tools(self, *, purpose, system, user, images=(), schema, max_rounds=6):
        tools = self.registry.openai_tools(purpose)
        if not tools:
            return self.complete_json(
                purpose=purpose, system=system, user=user, images=images, schema=schema
            )
        messages = [
            {"role": "system", "content": f"{system}\n\n{self._schema_reminder(schema)}"},
            {"role": "user", "content": self._user_content(user, images)},
        ]
        for round_no in range(1, max_rounds + 1):
            # Every round is a whole round trip, so a purpose that keeps looking
            # things up costs a multiple of what its one log line suggests.
            with timing.stage(f"{purpose}.r{round_no}"):
                resp = self._client.chat.completions.create(
                    model=self.settings.chat_model,
                    messages=messages,
                    tools=tools,
                    temperature=0,
                    **self._tuning(purpose),
                )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                if round_no > 1:
                    log.info("%s: answered after %d rounds", purpose, round_no)
                messages.append({"role": "assistant", "content": msg.content or ""})
                return self._parse_or_retry(purpose, messages, msg.content or "", schema)
            log.info(
                "%s round %d: looking up %s",
                purpose, round_no,
                ", ".join(tc.function.name for tc in msg.tool_calls),
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.registry.dispatch(purpose, tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        # Out of rounds — but not out of work. Everything the model looked up
        # is in `messages`, and raising here discarded all of it: a solver that
        # had done six of eight steps died with nothing, and the student heard
        # the failure as a request for a photo. So spend one more call with the
        # tools withdrawn and ask for the answer from what it already has. A
        # half-checked reference beats no reference, and the deterministic
        # checks downstream (the solver's machine check, the leak guard) are
        # unchanged by how the model got here.
        log.warning(
            "%s: out of tool rounds after %d; asking for the answer without tools",
            purpose, max_rounds,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "No more tool calls are available. Answer now, using what you "
                    "already worked out — do not request another tool call."
                ),
            }
        )
        with timing.stage(f"{purpose}.last"):
            return self._final_json(purpose, messages, schema)

    def _final_json(self, purpose: str, messages: list, schema: type[M]) -> M:
        resp = self._client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            **self._tuning(purpose),
        )
        content = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        return self._parse_or_retry(purpose, messages, content, schema)

    def _parse_or_retry(self, purpose: str, messages: list, content: str, schema: type[M]) -> M:
        try:
            return parse_into(schema, content)
        except ValidationError as e:
            log.warning("%s: invalid JSON from model, retrying once (%s)", purpose, e)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response did not match the schema: {e}. "
                        "Respond again with ONLY the corrected JSON object."
                    ),
                }
            )
            resp = self._client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            retry_content = resp.choices[0].message.content or ""
            try:
                return parse_into(schema, retry_content)
            except ValidationError as e2:
                raise LLMError(f"{purpose}: model output failed validation twice: {e2}") from e2
