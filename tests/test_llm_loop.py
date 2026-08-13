"""GrokClient tool-calling loop against a fake OpenAI transport."""

import json

import pytest
from pydantic import BaseModel

from tutor.config import Settings
from tutor.llm.client import GrokClient, LLMError
from tutor.tools.registry import ToolRegistry


class Out(BaseModel):
    answer: str


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = type("F", (), {"name": name, "arguments": arguments})()

    def model_dump(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                msg = outer.script.pop(0)
                choice = type("C", (), {"message": msg})()
                return type("R", (), {"choices": [choice]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


def make_grok(db, script) -> GrokClient:
    grok = object.__new__(GrokClient)
    grok.settings = Settings(xai_api_key="test-key")
    grok.registry = ToolRegistry(db)
    grok._client = FakeClient(script)
    return grok


def test_tool_loop_dispatches_and_returns_final_json(db):
    grok = make_grok(
        db,
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "c1",
                        "search_domain_kb",
                        json.dumps({"kind": "misconceptions", "concepts": ["linear_equation"]}),
                    )
                ]
            ),
            FakeMessage(content='{"answer": "done"}'),
        ],
    )
    result = grok.run_with_tools(
        purpose="estimate", system="s", user="u", schema=Out
    )
    assert result.answer == "done"
    # the tool result was fed back to the model
    tool_msgs = [
        m
        for req in grok._client.requests
        for m in req["messages"]
        if m.get("role") == "tool"
    ]
    assert tool_msgs and "sign_flip_on_move" in tool_msgs[0]["content"]


def test_blocked_kind_returns_error_to_model(db):
    grok = make_grok(
        db,
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("c1", "search_domain_kb", json.dumps({"kind": "solutions", "query": "lin_001"}))
                ]
            ),
            FakeMessage(content='{"answer": "ok"}'),
        ],
    )
    grok.run_with_tools(purpose="phrase", system="s", user="u", schema=Out)
    tool_msgs = [
        m
        for req in grok._client.requests
        for m in req["messages"]
        if m.get("role") == "tool"
    ]
    assert "error" in tool_msgs[0]["content"]
    assert "x = 5" not in tool_msgs[0]["content"]


def _endless_lookup(n: int) -> list:
    """A model that keeps checking and never answers."""
    return [
        FakeMessage(
            tool_calls=[FakeToolCall(f"c{i}", "compute", json.dumps({"expression": "1+1"}))]
        )
        for i in range(n)
    ]


def test_running_out_of_rounds_asks_once_more_without_tools(db):
    """Out of rounds is not out of work.

    This used to raise, and the discarded transcript was a whole background
    solve: six round trips of real arithmetic thrown away, surfacing to the
    student as an ungradeable answer. The last call keeps the transcript and
    withdraws the tools, so the model answers from what it already has.
    """
    grok = make_grok(db, _endless_lookup(3) + [FakeMessage(content='{"answer": "49"}')])

    result = grok.run_with_tools(
        purpose="solve", system="s", user="u", schema=Out, max_rounds=3
    )

    assert result.answer == "49"
    sent = grok._client.requests[-1]
    assert "tools" not in sent                      # the tools are gone...
    assert any(                                     # ...and the model was told so
        m.get("role") == "user" and "No more tool calls" in (m.get("content") or "")
        for m in sent["messages"]
    )
    # everything the model looked up is still in front of it
    assert sum(1 for m in sent["messages"] if m.get("role") == "tool") == 3


def test_the_last_call_still_raises_if_the_model_will_not_answer(db):
    """A model that answers nothing is still a failed call — one retry, then out."""
    grok = make_grok(
        db,
        _endless_lookup(2) + [FakeMessage(content="not json"), FakeMessage(content="nope")],
    )
    with pytest.raises(LLMError):
        grok.run_with_tools(
            purpose="solve", system="s", user="u", schema=Out, max_rounds=2
        )


def test_invalid_json_retries_once_then_raises(db):
    grok = make_grok(
        db,
        [
            FakeMessage(content="not json"),
            FakeMessage(content="still not json"),
        ],
    )
    with pytest.raises(LLMError):
        grok.run_with_tools(purpose="recognize", system="s", user="u", schema=Out)


def test_code_fenced_json_accepted(db):
    grok = make_grok(db, [FakeMessage(content='```json\n{"answer": "fenced"}\n```')])
    result = grok.complete_json(purpose="recognize", system="s", user="u", schema=Out)
    assert result.answer == "fenced"
