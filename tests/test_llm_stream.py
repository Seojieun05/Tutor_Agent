"""The student-facing slice of a structured Grok SSE response."""

from types import SimpleNamespace

import pytest

from tutor.hints.generator import PhrasedHint
from tutor.llm.client import JsonStringFieldStream


def test_hint_field_is_decoded_across_arbitrary_json_chunks():
    parser = JsonStringFieldStream("hint")
    chunks = [
        '{"hi', 'nt":"첫째 ', '줄을 \\"그대로\\" ',
        '볼까요?","board":[{"expr":"x = 5"}]}',
    ]
    out = "".join(parser.feed(chunk) for chunk in chunks)
    assert out == '첫째 줄을 "그대로" 볼까요?'
    assert "x = 5" not in out


def test_unicode_escape_waits_until_all_four_digits_arrive():
    parser = JsonStringFieldStream("hint")
    assert parser.feed('{"hint":"\\u') == ""
    assert parser.feed("c548") == "안"
    assert parser.feed('녕하세요"}') == "녕하세요"


def test_gemini_structured_chunks_use_the_same_hint_stream():
    pytest.importorskip("google.genai")
    from tutor.llm.gemini import GeminiClient

    class Models:
        def generate_content_stream(self, **kwargs):
            return iter([
                SimpleNamespace(text='{"hint":"첫째 '),
                SimpleNamespace(text='줄을 볼까요?","board":[]}'),
            ])

    client = GeminiClient.__new__(GeminiClient)
    client.model = "fake-gemini"
    client._client = SimpleNamespace(models=Models())
    out = []
    result = client.complete_json_stream(
        purpose="phrase",
        system="s",
        user="u",
        schema=PhrasedHint,
        text_field="hint",
        on_text_delta=out.append,
    )
    assert "".join(out) == "첫째 줄을 볼까요?"
    assert result.hint == "첫째 줄을 볼까요?"


def test_provider_fallback_preserves_the_streaming_interface():
    from tutor.llm.fallback import FallbackLLM

    class Dead:
        def complete_json_stream(self, **kwargs):
            raise RuntimeError("quota")

    class Alive:
        def complete_json_stream(self, **kwargs):
            kwargs["on_text_delta"]("안전한 힌트")
            return "ok"

    out = []
    llm = FallbackLLM(Dead(), Alive(), cooldown_s=60)
    assert llm.complete_json_stream(on_text_delta=out.append) == "ok"
    assert out == ["안전한 힌트"]
