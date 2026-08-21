"""sympy behind a tool call: `compute` and `check_equivalence`.

Lets the model verify its own arithmetic mid-generation instead of only being
checked after the fact. Advisory only — the deterministic post-checks (the
solver's machine check, the leak guard) still run, so a tool the model forgot
to call weakens nothing.

Never raises: unparseable input goes back to the model as {"error": ...},
which it can read and correct, where an exception would cost the whole turn.
"""

from __future__ import annotations

from typing import Any

import sympy

from tutor.knowledge import mathnorm


# sympy로 식 하나를 계산한다(모델이 스스로 검산할 때 쓰는 툴).
def compute(expression: str) -> dict[str, Any]:
    """Simplify a bare expression, or solve a one-variable equation.

    "3*5 + 2"      → {"value": "17"}
    "3*x + 5 = 20" → {"roots": ["5"]}
    """
    if not expression.strip():
        return {"error": "compute needs an expression"}
    try:
        residual, is_eq = mathnorm.parse_equation(expression)
        if not is_eq:
            return {"value": str(sympy.simplify(residual.doit()))}
        variables = residual.free_symbols
        if len(variables) != 1:
            return {
                "error": "solving needs exactly one variable, got "
                + str(sorted(str(v) for v in variables))
            }
        roots = sympy.solve(residual, next(iter(variables)))
        return {"roots": [str(r) for r in roots]}
    except mathnorm.ParseError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 — tool errors go back to the model
        return {"error": f"compute failed: {e}"}


# 두 식이 수학적으로 같은지 판정한다(말로 한 답 채점의 핵심 툴).
def check_equivalence(a: str, b: str) -> dict[str, Any]:
    """Are two equations (or two bare expressions) the same mathematical claim?

    Equations compare by solution set, scalar multiples included — "3*x = 15"
    and "x = 5" are the same claim. Expressions compare by simplified
    difference. An equation and a bare expression are never equivalent.
    """
    if not a.strip() or not b.strip():
        return {"error": "check_equivalence needs both a and b"}
    # parse first so the model hears "cannot parse", not a misleading False
    for s in (a, b):
        try:
            mathnorm.parse_equation(s)
        except mathnorm.ParseError as e:
            return {"error": str(e)}
    is_eq_a, is_eq_b = "=" in a, "=" in b
    if is_eq_a != is_eq_b:
        return {
            "equivalent": False,
            "note": "one input is an equation and the other a bare expression",
        }
    if is_eq_a:
        return {"equivalent": mathnorm.equations_equivalent(a, b)}
    return {"equivalent": mathnorm.expressions_equivalent(a, b)}
