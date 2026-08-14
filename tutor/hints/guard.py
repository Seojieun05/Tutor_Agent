"""Answer-leak guard: pure, typed, symbolic (spec rule 3).

SCALAR: any number/expression in the hint equal to the answer value.
ROOT_SET: any individual root (or the set) appearing in the hint.
EXPRESSION: sympy symbolic equivalence — catches rewrites like
"2*x + 3*x**2" for the answer "3*x**2 + 2*x".
All kinds also reject verbatim reference steps beyond the target step.

One thing is deliberately NOT a leak: a bare number the student can already
read on their own worksheet. In `3x + 5 = 20` the answer is 5 and so is a
coefficient, so "5를 어떻게 없앨까요?" — the most natural L1 question there is —
used to be rejected as giving the answer away, and the tutor fell back to a
generic template on exactly the problems where it had something useful to say.
Telling someone a number they are looking at is not telling them anything.

That excuse is narrow on purpose, and three things override it:
  - saying it AS the answer   "답은 5" / "x = 5" / "5입니다"
  - computing it              "15/3", which is the step, not the given
  - an expression equal to it "x = 20/4"
Only a bare mention of an already-visible number goes free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import sympy

from tutor.knowledge import mathnorm
from tutor.knowledge.models import ReferenceSolution

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ASSIGN_RE = re.compile(r"[a-zA-Z]\s*=\s*([-\d./*+^() a-zA-Z]+)")
_EXPR_RE = re.compile(r"[-\d./*+^() a-zA-Z_]{3,}")

# "the answer is N", in the forms a Korean tutor actually says it. A number in
# any of these is being presented as the result, whether or not it is visible
# in the problem.
_ANNOUNCE_RES = (
    re.compile(r"(?:답|정답|해|근|결과|값)\s*(?:은|는|이|가|으로|로)?\s*[^.!?\n]{0,6}?(-?\d+(?:\.\d+)?)"),
    re.compile(r"[a-zA-Z]\s*(?:는|은|이|가|=)\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:입니다|이에요|예요|이야|이죠|죠|이 돼요|가 돼요|이에|에요)"),
)


@dataclass
class _Candidates:
    bare: list[str] = field(default_factory=list)      # plain literals in prose
    computed: list[str] = field(default_factory=list)  # '15/3' — a calculation
    exprs: list[str] = field(default_factory=list)     # '3*x + 2'
    assigned: list[str] = field(default_factory=list)  # RHS of 'x = ...'
    announced: list[str] = field(default_factory=list) # stated as the answer


def _candidates(text: str) -> _Candidates:
    """Everything in the hint that could carry a value, sorted by how it says it."""
    found = _Candidates()
    # exponent digits are structure, not values: 'x**2' must not count as a 2
    deexponented = re.sub(r"(\*\*|\^)\s*-?\d+", "", text)
    found.bare = _NUMBER_RE.findall(deexponented)
    found.assigned = [m.group(1).strip() for m in _ASSIGN_RE.finditer(text)]
    found.exprs = list(found.assigned)
    for pattern in _ANNOUNCE_RES:
        found.announced.extend(m.group(1) for m in pattern.finditer(deexponented))
    for m in _EXPR_RE.finditer(text):
        s = m.group(0).strip()
        has_letter = re.search(r"[a-zA-Z]", s)
        has_op = re.search(r"[*/+\-^]", s)
        if has_letter and (has_op or re.search(r"\d\s*[a-zA-Z]", s)):
            # '3*x + 2' but also implicit multiplication like '10x'
            found.exprs.append(s)
        elif not has_letter and has_op and re.search(r"\d", s):
            # pure-numeric compound like '15/3' — evaluates to a value
            found.computed.append(s)
    return found


def _visible_numbers(given: Iterable[str]) -> list[str]:
    """The numbers the student can read off the problem in front of them."""
    numbers: list[str] = []
    for chunk in given:
        if not chunk:
            continue
        numbers.extend(_NUMBER_RE.findall(re.sub(r"(\*\*|\^)\s*-?\d+", "", str(chunk))))
    return numbers


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return False


def leaks_answer(
    text: str,
    reference: ReferenceSolution,
    target_step: int,
    given: Iterable[str] = (),
) -> bool:
    """Would saying this give the answer away?

    `given` is what the student can already see — the problem text, its
    equations, its choices. Numbers in there are theirs already; pass it and a
    hint may name them. Omit it and the guard stays strictly literal.
    """
    if not text:
        return False
    answer = reference.final_answer
    found = _candidates(text)
    visible = _visible_numbers(given)

    def says_the_value(value: str) -> bool:
        # Presented as the result, computed, or written as an expression:
        # all leaks regardless of what is printed on the worksheet.
        if any(_numeric_equal(n, value) for n in found.announced):
            return True
        if any(_numeric_equal(n, value) for n in found.computed):
            return True
        if any(_numeric_equal(e, value) for e in found.exprs):
            return True
        if not any(_numeric_equal(n, value) for n in found.bare):
            return False
        # A bare mention. Only excusable if it is already on their page.
        return not any(_numeric_equal(v, value) for v in visible)

    values = answer.value if isinstance(answer.value, list) else [answer.value]
    for value in values:
        value = str(value)
        if answer.kind in ("SCALAR", "ROOT_SET"):
            if says_the_value(value):
                return True
        else:  # EXPRESSION
            if any(mathnorm.expressions_equivalent(e, value) for e in found.exprs):
                return True
            # a constant answer can also leak as a bare number
            if _NUMBER_RE.fullmatch(value.strip()) and says_the_value(value):
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
        # An equation step is checked PIECE BY PIECE. This used to skip any
        # step containing "=" entirely — and nearly every step is written as
        # one ("m: y = -13x + 19") — so the RHS of a step the student had not
        # reached sailed through as long as it was spelled differently. That
        # is exactly how a finished tangent line ended up drawn on the board
        # while the student was still deriving it: the curve expression
        # "-13*x + 19" is not the string "y=-13x+19", but it IS its right side.
        for piece in step.expression.split("="):
            piece = piece.strip()
            # a bare name ("y", "m") carries no content; comparing it would
            # flag every hint that mentions the variable
            if not piece or not re.search(r"[*/+\-^]|\d\s*[a-zA-Z]|[a-zA-Z]\s*\d", piece):
                continue
            if any(mathnorm.expressions_equivalent(e, piece) for e in found.exprs):
                return True
    return False
