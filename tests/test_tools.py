"""Per-purpose allowlist matrix: each purpose gets exactly its tools/kinds."""

import pytest

from tutor.tools.registry import KB_KINDS_BY_PURPOSE, ToolRegistry


@pytest.fixture
def registry(db):
    return ToolRegistry(db)


def test_allowlist_matrix():
    assert KB_KINDS_BY_PURPOSE["recognize"] == frozenset()
    assert KB_KINDS_BY_PURPOSE["solve"] == {"problems", "solutions", "concepts"}
    assert KB_KINDS_BY_PURPOSE["estimate"] == {"misconceptions"}
    assert KB_KINDS_BY_PURPOSE["phrase"] == {"hint_templates", "misconceptions"}


def test_recognize_gets_no_tools(registry):
    assert registry.openai_tools("recognize") == []
    result = registry.dispatch("recognize", "search_domain_kb", {"kind": "problems"})
    assert "error" in result


def test_phrase_cannot_reach_solutions_or_answers(registry):
    result = registry.dispatch("phrase", "search_domain_kb", {"kind": "solutions", "query": "lin_001"})
    assert "error" in result
    result = registry.dispatch("phrase", "search_domain_kb", {"kind": "problems"})
    assert "error" in result
    # the schema exposed to the model does not even list those kinds
    schema = registry.openai_tools("phrase")[0]["function"]["parameters"]
    assert set(schema["properties"]["kind"]["enum"]) == {"hint_templates", "misconceptions"}


def test_phrase_allowed_kinds_work(registry):
    result = registry.dispatch(
        "phrase",
        "search_domain_kb",
        {"kind": "hint_templates", "concepts": ["linear_equation"], "level": 1},
    )
    assert result["hint_templates"]
    result = registry.dispatch(
        "phrase", "search_domain_kb", {"kind": "misconceptions", "misconception_id": "sign_flip_on_move"}
    )
    assert result["misconceptions"][0]["id"] == "sign_flip_on_move"


def test_solve_can_read_solutions(registry):
    result = registry.dispatch("solve", "search_domain_kb", {"kind": "solutions", "query": "lin_001"})
    assert result["solutions"][0]["final_answer"]["value"] == "5"


def test_estimate_only_misconceptions(registry):
    ok = registry.dispatch(
        "estimate", "search_domain_kb", {"kind": "misconceptions", "concepts": ["linear_equation"]}
    )
    assert ok["misconceptions"]
    blocked = registry.dispatch("estimate", "search_domain_kb", {"kind": "solutions", "query": "lin_001"})
    assert "error" in blocked


def test_unknown_tool_blocked(registry):
    result = registry.dispatch("solve", "write_anything", {})
    assert "error" in result
