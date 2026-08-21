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

# 학생이 대화 중 기다리는 용도들 — 여기에만 reasoning_effort를 낮춰 응답을 앞당긴다.
# Purposes the student waits on mid-conversation. `solve` and `recognize` are
# excluded: a wrong reference solution or a misread worksheet costs far more
# than the seconds saved.
FAST_PURPOSES = frozenset({"evaluate", "phrase", "estimate", "tag", "explain"})


# 모델 호출 실패.
class LLMError(Exception):
    pass


# 모든 모델 호출이 지나가는 인터페이스. Grok · Gemini · Echo가 이 모양을 똑같이 구현한다.
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


# 스트리밍으로 들어오는 JSON에서 특정 문자열 필드(hint)의 내용만 실시간으로 뽑아낸다.
# JSON 문법이나 board 같은 다른 필드는 학생에게 나가는 스트림에 절대 섞이지 않는다.
class JsonStringFieldStream:
    """Incrementally decode one JSON string field from arbitrary SSE chunks.

    The model still returns the complete structured object for validation. This
    small parser only exposes characters inside (for example) ``"hint"`` while
    that object is arriving; JSON syntax and later fields such as ``board``
    never enter the student-facing stream.
    """

    # JSON 문자열 이스케이프 처리표.
    _ESCAPES = {
        '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
        "n": "\n", "r": "\r", "t": "\t",
    }

    # 어떤 필드를 뽑을지 정하고 파서 상태를 초기화.
    def __init__(self, field: str):
        self.field = field
        self._source = ""
        self._pos = 0
        self._started = False
        self._done = False
        self._escaped = False
        self._unicode = ""

    # 새 조각을 넣고, 그 필드 안쪽의 '보여도 되는 글자'만 돌려준다.
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


# 모델이 코드펜스로 감싼 JSON을 벗겨 낸다.
def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


# 응답 문자열을 지정한 스키마로 검증·파싱.
def parse_into(schema: type[M], text: str) -> M:
    return schema.model_validate_json(_strip_fences(text))


# xAI(OpenAI 호환) 엔드포인트에 붙는 실제 클라이언트.
class GrokClient:
    # 설정과 툴 레지스트리를 받아 OpenAI 호환 클라이언트를 만든다.
    def __init__(self, settings: Settings, registry: ToolRegistry):
        from openai import OpenAI

        self.settings = settings
        self.registry = registry
        self._client = OpenAI(api_key=settings.xai_api_key, base_url=settings.xai_base_url)

    # 사용자 메시지 구성. 이미지가 있으면 base64 data URL로 함께 싣는다.
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

    # 응답 스키마를 시스템 메시지 끝에 덧붙여 JSON 형태를 강제한다.
    def _schema_reminder(self, schema: type[BaseModel]) -> str:
        return (
            "Respond with ONLY a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )

    # 용도별 지연 시간 조절: 대화용 호출만 추론 강도를 낮춘다(solve·recognize는 그대로).
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

    # 툴 없이 한 번 호출해 JSON 하나를 받는다.
    def complete_json(self, *, purpose, system, user, images=(), schema):
        messages = [
            {"role": "system", "content": f"{system}\n\n{self._schema_reminder(schema)}"},
            {"role": "user", "content": self._user_content(user, images)},
        ]
        return self._final_json(purpose, messages, schema)

    # JSON 전체는 검증용으로 받되, 지정한 문자열 필드는 오는 대로 흘려보낸다(말하기를 앞당기려고).
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

    # 툴(sympy 계산·KB 검색)을 쓸 수 있는 호출. 라운드 수를 제한해 침묵이 길어지지 않게 한다.
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

    # 마지막 응답을 스키마로 파싱해 돌려준다.
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

    # 파싱 실패 시 형식을 다시 알려 주고 한 번 더 물어본다.
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
