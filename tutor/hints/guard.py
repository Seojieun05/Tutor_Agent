"""Answer-leak guard: pure, typed, symbolic (spec rule 3).

SCALAR: any number/expression in the hint equal to the answer value.
ROOT_SET: any individual root (or the set) appearing in the hint.
EXPRESSION: sympy symbolic equivalence — catches rewrites like
"2*x + 3*x**2" for the answer "3*x**2 + 2*x".
All kinds also reject verbatim reference steps beyond the target step.
"""

from __future__ import annotations

import re

import sympy

from tutor.knowledge import mathnorm
from tutor.knowledge.models import ReferenceSolution

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ASSIGN_RE = re.compile(r"[a-zA-Z]\s*=\s*([-\d./*+^() a-zA-Z]+)")
_EXPR_RE = re.compile(r"[-\d./*+^() a-zA-Z_]{3,}")


def _candidates(text: str) -> tuple[list[str], list[str]]:
    """Return (numeric candidates, expression candidates) found in the hint."""
    # exponent digits are structure, not values: 'x**2' must not count as a 2
    deexponented = re.sub(r"(\*\*|\^)\s*-?\d+", "", text)
    numbers = _NUMBER_RE.findall(deexponented)
    exprs = [m.group(1).strip() for m in _ASSIGN_RE.finditer(text)]
    for m in _EXPR_RE.finditer(text):
        s = m.group(0).strip()
        has_letter = re.search(r"[a-zA-Z]", s)
        has_op = re.search(r"[*/+\-^]", s)
        if has_letter and (has_op or re.search(r"\d\s*[a-zA-Z]", s)):
            # '3*x + 2' but also implicit multiplication like '10x'
            exprs.append(s)
        elif not has_letter and has_op and re.search(r"\d", s):
            # pure-numeric compound like '15/3' — evaluates to a value
            numbers.append(s)
    return numbers, exprs


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return False


def leaks_answer(text: str, reference: ReferenceSolution, target_step: int) -> bool:
    if not text:
        return False
    answer = reference.final_answer
    numbers, exprs = _candidates(text)

    values = answer.value if isinstance(answer.value, list) else [answer.value]
    for value in values:
        value = str(value)
        if answer.kind in ("SCALAR", "ROOT_SET"):
            if any(_numeric_equal(n, value) for n in numbers):
                return True
            if any(_numeric_equal(e, value) for e in exprs):
                return True
        else:  # EXPRESSION
            if any(mathnorm.expressions_equivalent(e, value) for e in exprs):
                return True
            # a constant answer can also leak as a bare number
            if _NUMBER_RE.fullmatch(value.strip()) and any(
                _numeric_equal(n, value) for n in numbers
            ):
                return True

    # Reference steps beyond the target step must not appear (levels < 4 get
    # their target step filtered upstream; the guard is belt-and-braces).
    squeezed = re.sub(r"\s+", "", text)
    for step in reference.steps:
        if step.idx <= target_step:
            continue
        step_squeezed = re.sub(r"\s+", "", step.expression)
        if step_squeezed and step_squeezed in squeezed:
            return True
        for e in exprs:
            if "=" not in step.expression and mathnorm.expressions_equivalent(
                e, step.expression
            ):
                return True
    return False
