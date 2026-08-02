"""sympy primitives: normalization, equivalence, template matching, answer verification.

Equations are strings like "3*x + 5 = 20" or bare expressions like
"Derivative(x**3 + 2*x, x)" (also accepted as "d/dx(x**3 + 2*x)").
"""

from __future__ import annotations

import re
import unicodedata

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_DERIV_PREFIX = re.compile(r"d\s*/\s*d([a-zA-Z])\s*\(")
_UNICODE_MATH = {"×": "*", "÷": "/", "−": "-", "²": "**2", "³": "**3", "√": "sqrt"}


class ParseError(Exception):
    pass


def normalize_text(s: str) -> str:
    # map math glyphs BEFORE NFKC — NFKC folds '²' to a plain '2'
    for k, v in _UNICODE_MATH.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s+\-*/^=().,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _rewrite_derivatives(s: str) -> str:
    """d/dx(...) → Derivative(..., x), with balanced-paren scanning so trailing
    text like 'd/dx(x^3) + d/dx(2x)' is not swallowed by a greedy match."""
    while True:
        m = _DERIV_PREFIX.search(s)
        if m is None:
            return s
        depth, i = 1, m.end()
        while i < len(s) and depth:
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
            i += 1
        if depth:  # unbalanced parens: leave it, parsing will fail loudly
            return s
        inner = s[m.end() : i - 1]
        s = f"{s[:m.start()]}Derivative({inner}, {m.group(1)}){s[i:]}"


def _preprocess(s: str) -> str:
    for k, v in _UNICODE_MATH.items():
        s = s.replace(k, v)
    return _rewrite_derivatives(s).strip()


def parse_expression(s: str) -> sympy.Expr:
    try:
        expr = parse_expr(_preprocess(s), transformations=_TRANSFORMS)
    except Exception as e:
        raise ParseError(f"cannot parse {s!r}: {e}") from e
    if not isinstance(expr, sympy.Expr):
        raise ParseError(f"{s!r} did not parse to an expression (got {type(expr).__name__})")
    return expr


def parse_equation(s: str) -> tuple[sympy.Expr, bool]:
    """Return (residual expression, is_equation). For 'lhs = rhs' the residual is lhs - rhs."""
    s = _preprocess(s)
    if "=" in s:
        parts = s.split("=")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ParseError(f"malformed equation {s!r}")
        return parse_expression(parts[0]) - parse_expression(parts[1]), True
    return parse_expression(s), False


def _canonical(expr: sympy.Expr) -> sympy.Expr:
    """Rename free symbols to a canonical sequence so 3*y+5 == 3*x+5."""
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    return expr.subs(
        {s: sympy.Symbol(f"_v{i}") for i, s in enumerate(symbols)}, simultaneous=True
    )


def equations_equivalent(a: str, b: str, allow_scale: bool = True) -> bool:
    """allow_scale=True treats scalar multiples (6x+10=40 vs 3x+5=20) as
    equivalent. allow_scale=False accepts only the same equation (side swap
    included) — used by the EXACT matching tier, where reusing a stored
    solution's parameters for a scaled equation would produce wrong hints."""
    try:
        ra, ea = parse_equation(a)
        rb, eb = parse_equation(b)
    except ParseError:
        return False
    if ea != eb:
        return False
    ra, rb = _canonical(ra), _canonical(rb)
    if sympy.simplify(ra - rb) == 0:
        return True
    if ea and rb != 0:  # scalar multiples only make sense for equations
        try:
            ratio = sympy.simplify(sympy.cancel(ra / rb))
        except Exception:
            return False
        if ratio.is_number is not True or ratio == 0:
            return False
        return True if allow_scale else ratio == -1
    return False


def equations_same_form(a: str, b: str) -> bool:
    """Same WRITTEN equation, up to per-side simplification and side swap.

    All correct rearrangements of one equation share a proportional residual,
    so residual equivalence cannot tell '3*x + 5 = 20' from '3*x = 15'. This
    compares the two sides structurally: '15 = 3*x' matches '3*x = 15', but
    '3*x + 5 = 20' does not — they are different pedagogical steps.
    """
    try:
        sa, sb = _split_sides(a), _split_sides(b)
        if (sa is None) != (sb is None):
            return False
        if sa is None:
            return sympy.simplify(
                parse_expression(a).doit() - parse_expression(b).doit()
            ) == 0
        la, ra = (parse_expression(s) for s in sa)
        lb, rb = (parse_expression(s) for s in sb)

        def eq(x: sympy.Expr, y: sympy.Expr) -> bool:
            return sympy.simplify(x - y) == 0

        return (eq(la, lb) and eq(ra, rb)) or (eq(la, rb) and eq(ra, lb))
    except (ParseError, Exception):
        return False


def _split_sides(s: str) -> tuple[str, str] | None:
    s = _preprocess(s)
    if "=" not in s:
        return None
    lhs, _, rhs = s.partition("=")
    return lhs, rhs


def match_template(equation: str, pattern: str, params: list[str]) -> dict[str, str] | None:
    """Match a recognized equation against a parameterized pattern (e.g. 'a*x + b = c').

    Returns numeric bindings {param: value-as-string} or None. Sides of an
    equation are matched pairwise (also with sides swapped); bare expressions
    (e.g. Derivative patterns) are matched whole.
    """
    try:
        eq_sides = _split_sides(equation)
        pat_sides = _split_sides(pattern)
        if (eq_sides is None) != (pat_sides is None):
            return None
        wilds = {p: sympy.Wild(p, properties=[lambda k: k.is_number is True]) for p in params}

        def to_pattern(s: str) -> sympy.Expr:
            expr = parse_expression(s)
            return expr.subs({sympy.Symbol(p): w for p, w in wilds.items()})

        def try_match(target: sympy.Expr, pat: sympy.Expr) -> dict | None:
            """Bindings for the wilds this side mentions (each must be numeric)."""
            m = target.match(pat)
            if m is None:
                return None
            out = {}
            for p, w in wilds.items():
                if w in m:
                    if m[w].is_number is not True:
                        return None
                    out[p] = m[w]
            return out

        if eq_sides is None:
            whole = try_match(parse_expression(equation), to_pattern(pattern))
            if whole is None or set(whole) != set(params):
                return None
            return _stringify(whole)
        el, er = (parse_expression(x) for x in eq_sides)
        pl, pr = (to_pattern(x) for x in pat_sides)
        for tl, tr in ((el, er), (er, el)):
            ml = try_match(tl, pl)
            if ml is None:
                continue
            mr = try_match(tr, pr)
            if mr is None:
                continue
            merged = {**ml}
            conflict = any(p in ml and p in mr and ml[p] != mr[p] for p in params)
            if conflict:
                continue
            merged.update(mr)
            if set(merged) == set(params):
                return _stringify(merged)
        return None
    except ParseError:
        return None


def _stringify(bindings: dict | None) -> dict[str, str] | None:
    if bindings is None:
        return None
    return {k: str(v) for k, v in bindings.items()}


def instantiate(template_str: str, bindings: dict[str, str]) -> str:
    """Substitute numeric bindings into an expression/equation template string."""
    subs = {sympy.Symbol(k): sympy.sympify(v) for k, v in bindings.items()}

    def one(side: str) -> str:
        return str(sympy.simplify(parse_expression(side).subs(subs)))

    sides = _split_sides(template_str)
    if sides is None:
        return one(template_str)
    return f"{one(sides[0])} = {one(sides[1])}"


def _single_var(expr: sympy.Expr) -> sympy.Symbol:
    symbols = expr.free_symbols
    if len(symbols) != 1:
        raise ParseError(f"expected exactly one variable, got {symbols}")
    return next(iter(symbols))


def compute_answer(equation: str, kind: str) -> str | list[str]:
    """Compute the answer of an instantiated equation for the given answer kind."""
    residual, is_eq = parse_equation(equation)
    if kind == "EXPRESSION":
        if is_eq:
            raise ParseError("EXPRESSION answers need a bare expression, not an equation")
        return str(sympy.simplify(residual.doit()))
    var = _single_var(residual)
    roots = sympy.solve(residual, var)
    if kind == "SCALAR":
        if len(roots) != 1:
            raise ParseError(f"SCALAR answer expects one root, got {roots}")
        return str(roots[0])
    if kind == "ROOT_SET":
        return sorted(str(r) for r in roots)
    raise ParseError(f"unknown answer kind {kind!r}")


def verify_answer(equations: list[str], kind: str, value: str | list[str]) -> bool:
    """Check a claimed answer against the problem's equations with sympy."""
    try:
        if kind == "EXPRESSION":
            expr, is_eq = parse_equation(equations[0])
            if is_eq:
                return False
            return sympy.simplify(expr.doit() - parse_expression(str(value))) == 0
        eq_residuals = [parse_equation(e) for e in equations if "=" in _preprocess(e)]
        if not eq_residuals:
            return False
        if kind == "SCALAR":
            # the claimed value must be the COMPLETE solution set — a single
            # root of x**2-9=0 is not a correct SCALAR answer
            v = sympy.sympify(str(value))
            for residual, _ in eq_residuals:
                var = _single_var(residual)
                roots = sympy.solve(residual, var)
                if len(roots) != 1 or sympy.simplify(roots[0] - v) != 0:
                    return False
            return True
        if kind == "ROOT_SET":
            residual, _ = eq_residuals[0]
            var = _single_var(residual)
            roots = list(sympy.solve(residual, var))
            claimed = [sympy.sympify(str(v)) for v in value]
            if len(roots) != len(claimed):
                return False
            # numeric pairing, so "3.0" matches the Integer root 3
            remaining = list(roots)
            for c in claimed:
                for r in remaining:
                    if sympy.simplify(r - c) == 0:
                        remaining.remove(r)
                        break
                else:
                    return False
            return True
        return False
    except (ParseError, Exception):
        return False


def expressions_equivalent(a: str, b: str) -> bool:
    try:
        ea, eb = parse_expression(a), parse_expression(b)
        return sympy.simplify(ea.doit() - eb.doit()) == 0
    except (ParseError, Exception):
        return False
