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

# sympy 파서 옵션: 생략된 곱셈(2x)과 ^ 거듭제곱을 받아들인다.
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# d/dx( … ) 표기를 잡는 정규식.
_DERIV_PREFIX = re.compile(r"d\s*/\s*d([a-zA-Z])\s*\(")
# f'(1) 같은 프라임 호출 표기.
_PRIME_CALL = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*(['′]+)\s*\(\s*([A-Za-z0-9_.+-]+)\s*\)"
)
# f(x) = … 함수 정의.
_FUNCTION_DEFINITION = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+?)\s*$"
)
# 함수 호출 꼴.
_FUNCTION_CALL = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\s*\(")
# 유니코드 수학 기호 → ASCII 대응표.
_UNICODE_MATH = {"×": "*", "÷": "/", "−": "-", "²": "**2", "³": "**3", "√": "sqrt"}


# 식을 파싱하지 못했을 때.
class ParseError(Exception):
    pass


# 텍스트 정규화. 저장된 text_hash가 여기서 나오므로 결과가 절대 바뀌면 안 된다.
def normalize_text(s: str) -> str:
    # map math glyphs BEFORE NFKC — NFKC folds '²' to a plain '2'
    for k, v in _UNICODE_MATH.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s+\-*/^=().,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# An exam's segment bar is notation, not a function: overline(AB) IS AB. The
# VLM transcribes printed AB-with-a-bar this way, a stored equation writes the
# bare length, and left alone SymPy reads a function call — every equivalence
# against the stored form then silently fails.
# overline(AB) 표기.
_OVERLINE_CALL = re.compile(r"\boverline\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)")

# Superscript POWERS a printed page uses and NFKC folds flat: aˣ must read
# a^x, not ax — the bare fold loses the power and with it the problem's shape.
# 위첨자 숫자 → ** 표기.
_SUPERSCRIPT_POWERS = {
    "ˣ": "^x", "ʸ": "^y", "ⁿ": "^n",
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
}


# 표기 방식에 무관한 본문 동일성 키. 같은 인쇄물을 다르게 읽어도 같은 문자열이 되게 한다.
def identity_text(s: str) -> str:
    """Rendering-blind prose identity: the same printed problem, however read.

    normalize_text must stay byte-stable — every stored text_hash was minted
    with it — so the folds a PHOTOGRAPH needs live here instead. Live, a typed
    presolve entry said "y = aˣ - 2" where the VLM wrote "y=a^x-2": the only
    differences a faithful read introduces are superscript rendering, power
    notation, overline calls and spacing, so exactly those are erased. Numbers
    and conditions survive untouched — a lookalike with one changed value
    still reads as a different problem.
    """
    for k, v in _SUPERSCRIPT_POWERS.items():
        s = s.replace(k, v)
    s = _OVERLINE_CALL.sub(r"\1", s)
    s = normalize_text(s).replace("**", "^")
    return re.sub(r"\s+", "", s)


# 한글 한두 글자 오인식은 눈감아 주는 본문 동일 판정(숫자·기호·길이는 정확히 같아야 한다).
def texts_identical_enough(a: str, b: str, max_hangul_substitutions: int = 2) -> bool:
    """identity_text equality, forgiving OCR-grade syllable slips.

    Live, the same printed page read "상수 a(a>1)" on one capture and
    "실수 a(a>1)" on the next — conf 1.00 both times: one syllable of ink,
    zero difference in the mathematics. Same-position hangul substitutions
    up to the cap are forgiven; digits, latin letters, symbols and the
    length itself must match exactly — those are the problem's parameters
    and structure, not ink.
    """
    ia, ib = identity_text(a), identity_text(b)
    if ia == ib:
        return bool(ia)
    if len(ia) != len(ib):
        return False
    substitutions = 0
    for ca, cb in zip(ia, ib):
        if ca == cb:
            continue
        if not ("가" <= ca <= "힣" and "가" <= cb <= "힣"):
            return False
        substitutions += 1
        if substitutions > max_hangul_substitutions:
            return False
    return True


# d/dx(...)를 sympy의 Derivative(..., x)로 바꾼다(괄호 짝을 세면서).
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


# f'(1) 같은 표기를 sympy가 이해하는 형태로 바꾼다.
def _rewrite_prime_calls(s: str) -> str:
    """Turn schoolbook ``f'(x)`` into a parseable atomic symbol.

    SymPy reads an apostrophe as the start of a Python string, so a perfectly
    ordinary worksheet line such as ``g'(x) = ...`` used to make every
    equivalence check silently return False.  The application is intentionally
    atomic: without a supplied definition, ``f'(x)`` is an unknown value, not
    something SymPy is allowed to invent a derivative for.
    """

    def replacement(match: re.Match[str]) -> str:
        name, primes, argument = match.groups()
        order = len(primes)
        suffix = "prime" if order == 1 else f"prime{order}"
        safe_argument = re.sub(r"[^A-Za-z0-9_]", "_", argument)
        return f"{name}_{suffix}_at_{safe_argument}"

    return _PRIME_CALL.sub(replacement, s)


# 파싱 전 공통 전처리(유니코드 정리 + 도함수·프라임 표기 변환).
def _preprocess(s: str) -> str:
    for k, v in _UNICODE_MATH.items():
        s = s.replace(k, v)
    s = _OVERLINE_CALL.sub(r"\1", s)
    return _rewrite_prime_calls(_rewrite_derivatives(s)).strip()


# 문자열을 sympy 식으로. 실패하면 ParseError.
def parse_expression(s: str) -> sympy.Expr:
    try:
        expr = parse_expr(_preprocess(s), transformations=_TRANSFORMS)
    except Exception as e:
        raise ParseError(f"cannot parse {s!r}: {e}") from e
    if not isinstance(expr, sympy.Expr):
        raise ParseError(f"{s!r} did not parse to an expression (got {type(expr).__name__})")
    return expr


# A line that CLAIMS a value: a bare symbol or a function evaluated at a
# number on the left, e.g. "x", "f'(1)", "g(2)". "3*x" is not a claim — it is
# an equation still being solved, and comparing it to the final answer would
# flag every correct intermediate step.
# 값을 '주장하는' 좌변 꼴(x, f'(1), g(2)). 3*x 같은 건 아직 푸는 중이라 주장이 아니다.
_CLAIM_LHS = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\s*['′]*\s*(?:\(\s*-?\d+(?:\.\d+)?\s*\))?$")


# 이 줄이 주장하는 숫자를 뽑는다. 모델 판단이 아니라 산술로 오답을 잡는 근거.
def numeric_claim(line: str) -> float | None:
    """The number this work line claims, when it makes an arithmetic claim.

    "f'(1) = 2-1-2+3×1" claims 2. "x = 5" claims 5. A chained
    "f'(1) = -1+9 = 8" claims its LAST segment, 8 — that is the student's
    conclusion. None when the line is not that shape (symbolic right side,
    no equals, unparseable): absence of a claim, never a guess.

    This is what lets a wrong final line be caught by ARITHMETIC rather than
    by a model's judgement: the claim is evaluated with sympy and compared to
    the verified answer, and 2 ≠ 8 does not depend on anyone's reasoning.
    """
    if "=" not in line:
        return None
    parts = [p.strip() for p in _preprocess(line).split("=")]
    if len(parts) < 2 or not all(parts):
        return None
    if not _CLAIM_LHS.match(parts[0]):
        return None
    try:
        # primes survive _preprocess; strip them for the RHS check only —
        # the LHS shape is all we need to know about the left side
        value = parse_expression(parts[-1])
        if value.free_symbols:
            return None
        return float(value)
    except (ParseError, TypeError, ValueError):
        return None


# 등식을 (좌변-우변, 등식여부)로 바꾼다.
def parse_equation(s: str) -> tuple[sympy.Expr, bool]:
    """Return (residual expression, is_equation). For 'lhs = rhs' the residual is lhs - rhs."""
    s = _preprocess(s)
    if "=" in s:
        parts = s.split("=")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ParseError(f"malformed equation {s!r}")
        return parse_expression(parts[0]) - parse_expression(parts[1]), True
    return parse_expression(s), False


# 변수 이름을 표준화해 3*y+5와 3*x+5를 같게 본다.
def _canonical(expr: sympy.Expr) -> sympy.Expr:
    """Rename free symbols to a canonical sequence so 3*y+5 == 3*x+5."""
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    return expr.subs(
        {s: sympy.Symbol(f"_v{i}") for i, s in enumerate(symbols)}, simultaneous=True
    )


# 콤마로 이어진 여러 식을 각각 (잔차식, 등식여부) 쌍으로.
def _residual_pairs(s: str) -> list[tuple[sympy.Expr, bool]]:
    """One (residual, is_equation) per claim in the string.

    A chain equality — 2(a₁+a₄+a₇) = a₄+a₇+a₁₀ = 6, the bread and butter of
    수열 problems — is N-1 adjacent claims, not a parse error. parse_equation
    stays two-sided (its callers substitute answers into ONE equation); the
    chain unrolling lives here, where comparing claims is the whole job.
    """
    s = _preprocess(s)
    parts = [p.strip() for p in s.split("=")]
    if any(not p for p in parts):
        raise ParseError(f"malformed equation {s!r}")
    if len(parts) == 1:
        return [(parse_expression(parts[0]), False)]
    exprs = [parse_expression(p) for p in parts]
    return [(x - y, True) for x, y in zip(exprs, exprs[1:])]


# 두 잔차식이 같은지(배수 허용 여부는 인자로).
def _residuals_equivalent(
    ra: sympy.Expr, rb: sympy.Expr, is_eq: bool, allow_scale: bool
) -> bool:
    ra, rb = _canonical(ra), _canonical(rb)
    if sympy.simplify(ra - rb) == 0:
        return True
    if is_eq and rb != 0:  # scalar multiples only make sense for equations
        try:
            ratio = sympy.simplify(sympy.cancel(ra / rb))
        except Exception:
            return False
        if ratio.is_number is not True or ratio == 0:
            return False
        return True if allow_scale else ratio == -1
    return False


# 공백·곱셈기호만 지운 값싼 비교용 문자열. 같은 줄을 다르게 쓴 것인지 볼 때 쓴다.
def compact(s: str) -> str:
    """Whitespace and multiplication signs removed — enough to tell "the same
    line, rewritten" from "a different claim", at none of the cost or reach of
    symbolic equivalence. The VLM writes 2(a+b) where the model writes
    2*(a+b), and that difference is all this has to survive.
    """
    return re.sub(r"[\s·*]", "", s)


# 이 줄을 기계적으로 비교할 수 있는지. 다른 문제라는 증거와 비교 실패를 구분하려고 쓴다.
def parseable_claims(s: str) -> bool:
    """Can this line be compared mechanically at all? The session uses this to
    tell "the equations DISAGREE" (positive evidence of a different problem)
    from "the comparison never happened" (a re-read hiccup)."""
    try:
        _residual_pairs(s)
        return True
    except Exception:  # noqa: BLE001 — sympy throws many things on junk input
        return False


# 두 식이 수학적으로 같은지. allow_scale=False면 배수까지는 인정하지 않는다(EXACT 등급용).
def equations_equivalent(a: str, b: str, allow_scale: bool = True) -> bool:
    """allow_scale=True treats scalar multiples (6x+10=40 vs 3x+5=20) as
    equivalent. allow_scale=False accepts only the same equation (side swap
    included) — used by the EXACT matching tier, where reusing a stored
    solution's parameters for a scaled equation would produce wrong hints."""
    try:
        pa = _residual_pairs(a)
        pb = _residual_pairs(b)
    except ParseError:
        return False
    if len(pa) != len(pb):
        return False
    return all(
        ea == eb and _residuals_equivalent(ra, rb, ea, allow_scale)
        for (ra, ea), (rb, eb) in zip(pa, pb)
    )


# 숫자 토큰.
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")


# 값싼 인덱스 키: 식을 이루는 숫자와 변수들. EXACT 후보를 좁힐 때 쓴다.
def equations_signature(equations: list[str]) -> str:
    """Cheap index key: the numbers and variables an equation is made of.

    A *necessary* condition for strict equivalence — two equations whose
    residuals are equal (or negated: a swapped side) are built from the same
    literals and variables, in any order. Lets the EXACT tier pull a handful of
    candidates out of SQL instead of running sympy over every stored problem.
    Purely lexical, so it never parses and never fails.
    """
    parts = []
    for eq in equations:
        text = _preprocess(eq)
        numbers = sorted(
            {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER_TOKEN_RE.findall(text)}
        )
        variables = sorted(set(re.findall(r"(?<![A-Za-z_])[a-zA-Z](?![A-Za-z_])", text)))
        parts.append(",".join(numbers) + "#" + ",".join(variables))
    return "|".join(parts)


# '쓰인 모양'이 같은지. 3*x+5=20과 3*x=15는 수학적으로 같아도 다른 교육 단계라 구분한다.
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


# 문제에 적힌 함수 정의로부터 안전하게 유도되는 f'(x) 치환 목록.
def derivative_substitutions(equations: list[str]) -> dict[tuple[str, str], str]:
    """Derive safe ``f'(x)`` substitutions from explicit function definitions.

    Only definitions whose right side contains no other function call qualify.
    Thus ``f(x) = x**2 - 4*x - 3`` yields ``f'(x) = 2*x - 4``, while
    ``g(x) = h(x)*f(x)`` is deliberately skipped: treating ``f(x)`` as a
    constant symbol there would manufacture a wrong derivative.
    """
    substitutions: dict[tuple[str, str], str] = {}
    for equation in equations:
        match = _FUNCTION_DEFINITION.match(_preprocess(equation))
        if match is None:
            continue
        function, variable, rhs = match.groups()
        if _FUNCTION_CALL.search(rhs):
            continue
        try:
            expression = parse_expression(rhs)
            derivative = sympy.simplify(sympy.diff(expression, sympy.Symbol(variable)))
        except (ParseError, Exception):
            continue
        substitutions[(function, variable)] = str(derivative)
    return substitutions


# 문제가 증명해 주는 도함수 치환까지 적용한 뒤의 같은 모양 판정.
def equations_same_form_with_derivatives(
    a: str, b: str, definitions: list[str]
) -> bool:
    """Same written step after applying derivatives proven by the problem.

    This is contextual equivalence, not a license to collapse arbitrary
    pedagogical steps.  It handles the common pair
    ``...*f'(x)`` and ``...*(2*x - 4)`` only when the page also provides a
    definition from which SymPy independently derives that substitution.
    """
    substitutions = derivative_substitutions(definitions)

    def reflexive(text: str) -> bool:
        lhs, separator, rhs = text.partition("=")
        return bool(
            separator
            and "=" not in rhs
            and expressions_equivalent(lhs.strip(), rhs.strip())
        )

    # A definition can turn f'(x)=f'(x) into the computed derivative on both
    # sides.  It remains a tautology, not evidence that the reference step was
    # performed, so never match it against a non-reflexive claim.
    if reflexive(a) != reflexive(b):
        return False

    def apply_to_claim(text: str) -> str:
        for (function, variable), value in substitutions.items():
            pattern = re.compile(
                rf"\b{re.escape(function)}\s*['′]\s*"
                rf"\(\s*{re.escape(variable)}\s*\)"
            )
            text = pattern.sub(f"({value})", text)
        return text

    def apply(text: str) -> str:
        # Preserve the left-hand subject of an equation.  Substituting there
        # would turn the non-progress line "f'(x) = f'(x)" into the computed
        # derivative and falsely make it look like a completed step.  The
        # contextual rewrite is for a derivative used inside the student's
        # claimed result, which is the right-hand side (or a bare expression).
        lhs, separator, rhs = text.partition("=")
        if not separator:
            return apply_to_claim(text)
        return lhs + separator + apply_to_claim(rhs)

    return equations_same_form(apply(a), apply(b))


# 등식을 좌변·우변 문자열로 쪼갠다.
def _split_sides(s: str) -> tuple[str, str] | None:
    s = _preprocess(s)
    if "=" not in s:
        return None
    lhs, _, rhs = s.partition("=")
    return lhs, rhs


# 식이 템플릿 패턴에 맞는지 보고, 맞으면 파라미터 바인딩을 돌려준다.
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


# 바인딩 값을 문자열로.
def _stringify(bindings: dict | None) -> dict[str, str] | None:
    if bindings is None:
        return None
    return {k: str(v) for k, v in bindings.items()}


# 템플릿 문자열에 바인딩을 대입해 실제 식으로.
def instantiate(template_str: str, bindings: dict[str, str]) -> str:
    """Substitute numeric bindings into an expression/equation template string."""
    subs = {sympy.Symbol(k): sympy.sympify(v) for k, v in bindings.items()}

    def one(side: str) -> str:
        return str(sympy.simplify(parse_expression(side).subs(subs)))

    sides = _split_sides(template_str)
    if sides is None:
        return one(template_str)
    return f"{one(sides[0])} = {one(sides[1])}"


# 식에 변수가 정확히 하나면 그 변수를.
def _single_var(expr: sympy.Expr) -> sympy.Symbol:
    symbols = expr.free_symbols
    if len(symbols) != 1:
        raise ParseError(f"expected exactly one variable, got {symbols}")
    return next(iter(symbols))


# 식을 풀어 정답 값을 계산한다(종류에 따라 값/근의 집합/식).
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


# 주어진 답이 실제로 식을 만족하는지 sympy로 검산. 저장 여부를 가르는 마지막 관문.
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


# 등호 없는 두 식이 같은지.
def expressions_equivalent(a: str, b: str) -> bool:
    try:
        ea, eb = parse_expression(a), parse_expression(b)
        return sympy.simplify(ea.doit() - eb.doit()) == 0
    except (ParseError, Exception):
        return False
