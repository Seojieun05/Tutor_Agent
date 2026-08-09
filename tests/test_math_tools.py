"""sympy check tools: mathtools functions, registry scoping, and the loop."""

import json

from pydantic import BaseModel

from tutor.tools import mathtools
from tutor.tools.registry import ToolRegistry

from test_llm_loop import FakeMessage, FakeToolCall, make_grok


class Out(BaseModel):
    answer: str


# --- mathtools ----------------------------------------------------------------


def test_compute_evaluates_arithmetic():
    assert mathtools.compute("3*5 + 2") == {"value": "17"}


def test_compute_solves_one_variable_equation():
    assert mathtools.compute("3*x + 5 = 20") == {"roots": ["5"]}


def test_compute_rejects_multi_variable_equation():
    result = mathtools.compute("x + y = 3")
    assert "error" in result and "exactly one variable" in result["error"]


def test_compute_reports_parse_errors():
    assert "error" in mathtools.compute("3*(x +")
    assert "error" in mathtools.compute("")


def test_check_equivalence_equations_scalar_multiple():
    # same claim: 3x = 15 and x = 5 have the same solution set
    assert mathtools.check_equivalence("3*x = 15", "x = 5") == {"equivalent": True}
    assert mathtools.check_equivalence("3*x = 15", "x = 4") == {"equivalent": False}


def test_check_equivalence_expressions():
    assert mathtools.check_equivalence("2*(x + 3)", "2*x + 6") == {"equivalent": True}
    assert mathtools.check_equivalence("2*(x + 3)", "2*x + 3") == {"equivalent": False}


def test_check_equivalence_mixed_shapes_never_equivalent():
    result = mathtools.check_equivalence("3*x = 15", "3*x")
    assert result["equivalent"] is False and "note" in result


def test_check_equivalence_reports_parse_errors():
    assert "error" in mathtools.check_equivalence("3*(x +", "x = 5")
    assert "error" in mathtools.check_equivalence("", "x = 5")


# --- registry scoping ----------------------------------------------------------


def test_solve_and_evaluate_get_math_tools(db):
    registry = ToolRegistry(db)
    solve_names = {t["function"]["name"] for t in registry.openai_tools("solve")}
    assert {"search_domain_kb", "compute", "check_equivalence"} == solve_names
    # evaluate: math tools only — its KB context is prefetched by the orchestrator
    eval_names = {t["function"]["name"] for t in registry.openai_tools("evaluate")}
    assert {"compute", "check_equivalence"} == eval_names


def test_phrase_gets_no_math_tools(db):
    # phrase must not be able to compute its way to the answer it must not say
    registry = ToolRegistry(db)
    names = {t["function"]["name"] for t in registry.openai_tools("phrase")}
    assert names == {"search_domain_kb"}
    blocked = registry.dispatch("phrase", "compute", {"expression": "3*x + 5 = 20"})
    assert "error" in blocked


def test_dispatch_routes_math_tools(db):
    registry = ToolRegistry(db)
    assert registry.dispatch("solve", "compute", {"expression": "3*x + 5 = 20"}) == {
        "roots": ["5"]
    }
    assert registry.dispatch(
        "evaluate", "check_equivalence", {"a": "3*x = 15", "b": "x = 5"}
    ) == {"equivalent": True}


# --- the tool loop, end to end ---------------------------------------------------


def test_solver_loop_feeds_compute_result_back(db):
    grok = make_grok(
        db,
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall(
                        "c1", "compute", json.dumps({"expression": "3*x + 5 = 20"})
                    )
                ]
            ),
            FakeMessage(content='{"answer": "checked"}'),
        ],
    )
    result = grok.run_with_tools(purpose="solve", system="s", user="u", schema=Out)
    assert result.answer == "checked"
    tool_msgs = [
        m
        for req in grok._client.requests
        for m in req["messages"]
        if m.get("role") == "tool"
    ]
    assert tool_msgs and json.loads(tool_msgs[0]["content"]) == {"roots": ["5"]}


def test_evaluate_loop_offers_math_tools(db):
    grok = make_grok(db, [FakeMessage(content='{"answer": "graded"}')])
    result = grok.run_with_tools(purpose="evaluate", system="s", user="u", schema=Out)
    assert result.answer == "graded"
    # evaluate used to be a no-tools complete_json call; now the request
    # carries the sympy tools (and only those)
    sent = grok._client.requests[0]["tools"]
    assert {t["function"]["name"] for t in sent} == {"compute", "check_equivalence"}
