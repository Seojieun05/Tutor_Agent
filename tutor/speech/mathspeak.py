"""ASCII and LaTeX math, said the way a Korean teacher says it.

TTS reads "f'(1) = 2x**2 - x" as punctuation soup, and the hint model likes to
write "y = -\\frac{1}{5}x^2 + 3" on top of that. Everything the tutor SPEAKS
passes through speakable(), which rewrites the notations a K-12 worksheet
actually contains — powers, primes, logs, fractions, the four operators —
into the spoken forms: "f 프라임 1", "2 x 제곱 빼기 x", "5분의 1", "밑이 3인
로그". Parentheses become commas: a pause where the grouping was.

Deliberately conservative: text with no math notation comes back UNCHANGED,
byte for byte. The fixed phrases ("잘 못 들었어요…") are cache keys for
pre-rendered TTS, and rewriting them would be both wrong and expensive.
"""

from __future__ import annotations

import re

# Digits as read aloud: does the Korean word end in a consonant? "일(1)" does,
# "오(5)" does not — it decides 이라고/라고, 은/는.
_DIGIT_HAS_FINAL = {"0": True, "1": True, "2": False, "3": True, "4": False,
                    "5": False, "6": True, "7": True, "8": True, "9": False}

# If none of these appear, the text has no math notation worth touching.
# The backslash is LaTeX: the VLM and the hint model both write \frac{1}{5}.
# a_4 (수열 항) counts too: a letter directly joined to an underscore.
_MATH_SIGNAL = re.compile(r"[*^=/×÷'′²³√\\]|[A-Za-z]_\w|\blog|\bsqrt|\bDerivative")

# \frac{1}{5}, \dfrac{x+1}{2} — the args stay brace-free on a K-12 worksheet
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def _latex(s: str, *, spoken: bool) -> str:
    """LaTeX, as the models actually emit it, down to the ASCII the rest of
    this module reads. \frac goes straight to its destination form — 분모-first
    Korean for the ear, num/den for the eye — because once it is a bare `1/5x`
    the denominator's edge is lost and "5x분의 1" comes out wrong.
    """
    if "\\" not in s and "$" not in s and "{" not in s:
        return s
    s = re.sub(r"\$+", "", s)                    # $…$ / $$…$$ delimiters
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[\(\)\[\]]", "", s)           # \( \) \[ \] delimiters
    s = re.sub(r"\\[,;!: ]", " ", s)             # spacing commands
    s = re.sub(r"\\(log|ln|sin|cos|tan)(?![A-Za-z])", r"\1", s)  # \log_3 → log_3
    s = s.replace("\\cdot", "*").replace("\\times", "×").replace("\\div", "÷")
    s = s.replace("\\pi", "파이" if spoken else "π")
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    if not spoken:
        # a_{n+1} must sink WHOLE, before the brace strip below turns it into
        # "a_n + 1" — the general term of a sequence is written this way and
        # means something quite different from a sum
        def sunk(m: re.Match) -> str:
            inner = m.group(1).strip()
            if not inner or not all(ch in _SUB_SRC for ch in inner):
                return m.group(0)
            return inner.translate(_SUBSCRIPT)

        s = re.sub(r"_\{([^{}]*)\}", sunk, s)
    if spoken:
        s = _FRAC.sub(lambda m: f" {m.group(2).strip()}분의 {m.group(1).strip()} ", s)
    else:
        def frac(m: re.Match) -> str:
            num, den = m.group(1).strip(), m.group(2).strip()
            wrap = lambda a: a if re.fullmatch(r"\w+", a) else f"({a})"  # noqa: E731
            body = f"{wrap(num)}/{wrap(den)}"
            # -1/5x² misreads as 1/(5x²): parenthesize when a term follows
            follows = re.match(r"\s*[0-9A-Za-z(√\\]", s[m.end():])
            return f"({body})" if follows else body

        s = _FRAC.sub(frac, s)
    # ^{10} → ^10, log_{3} → log_3: whatever survives, unwrapped
    return s.replace("{", "").replace("}", "")


def ends_in_consonant(text: str) -> bool:
    """Does the LAST spoken syllable close on a consonant (받침)?"""
    if not text:
        return False
    ch = text[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    return _DIGIT_HAS_FINAL.get(ch, False)


def _topic_particle(text: str) -> str:
    return "은" if ends_in_consonant(text.rstrip()) else "는"


def _powers(s: str) -> str:
    def power_word(exp: str) -> str:
        return {"2": " 제곱", "3": " 세제곱"}.get(exp, f"의 {exp}제곱")

    s = re.sub(r"\s*(?:\*\*|\^)\s*(\d+)", lambda m: power_word(m.group(1)), s)
    s = re.sub(r"\s*\^\s*([A-Za-z])\b", r"의 \1제곱", s)  # x^n → "x의 n제곱"
    s = s.replace("²", " 제곱").replace("³", " 세제곱")
    return s


def _primes(s: str) -> str:
    # f'(1) → "f 프라임 1"; g''(x) → "g 더블 프라임 x". The argument keeps its
    # own parentheses removed — "에프 프라임 괄호 일" helps nobody.
    def prime(m: re.Match) -> str:
        name, ticks, arg = m.group(1), m.group(2), m.group(3)
        word = "더블 프라임" if len(ticks) >= 2 else "프라임"
        return f"{name} {word} {arg.strip()}"

    return re.sub(r"\b([A-Za-z])\s*(['′]+)\s*\(\s*([^()]*?)\s*\)", prime, s)


def _logs(s: str) -> str:
    # log_a(b), log_3 b/a, log_9 ab — "밑이 a인 로그 b"
    def log(m: re.Match) -> str:
        return f"밑이 {m.group(1)}인 로그 {m.group(2).strip()}"

    s = re.sub(r"\blog_?\{?(\w+)\}?\s*\(\s*([^()]+?)\s*\)", log, s)
    s = re.sub(r"\blog_?\{?(\w+)\}?\s+(\S+)", log, s)
    return s


def _fractions(s: str) -> str:
    # Korean reads the denominator first: 1/2 → "2분의 1", b/a → "a분의 b".
    return re.sub(r"\b(\w+)\s*/\s*(\w+)\b", r"\2분의 \1", s)


def _operators(s: str) -> str:
    s = re.sub(r"\bsqrt\s*\(\s*([^()]+?)\s*\)", r"루트 \1", s)
    s = re.sub(r"√\s*", "루트 ", s)
    s = s.replace(")(", ") 곱하기 (")  # adjacency IS multiplication
    s = re.sub(r"\s*[*×]\s*", " 곱하기 ", s)
    s = re.sub(r"\s*÷\s*", " 나누기 ", s)
    s = re.sub(r"\s*\+\s*", " 더하기 ", s)
    # minus: sign when it opens an expression, subtraction between terms
    s = re.sub(r"(^|[=(,]\s*)[-−]\s*", r"\1마이너스 ", s)
    s = re.sub(r"\s*[-−]\s*", " 빼기 ", s)
    return s


def _equals(s: str) -> str:
    def eq(m: re.Match) -> str:
        left = m.group(1)
        return f"{left}{_topic_particle(left)} "

    return re.sub(r"(\S)\s*=\s*", eq, s)


def _parens(s: str) -> str:
    # A pause where the grouping was — "괄호 열고" is for dictation, not tutoring.
    s = re.sub(r"\s*\(\s*", " ", s)
    s = re.sub(r"\s*\)\s*", ", ", s)
    s = re.sub(r"\s*,\s*,+", ",", s)         # collapse stacked pauses
    s = re.sub(r",\s*(?=[.?!]|$)", "", s)    # no pause right before the end
    return s


# Indices and exponents as print sets them: dropped below or raised above the
# line, as actual unicode characters — no HTML needed, so they survive any
# text surface. Unicode has subscripts for all digits but only some letters,
# and superscripts for all digits and most letters (no q): anything that
# cannot be moved cleanly stays as written rather than coming out half-set.
# the operators too, because a general term is written a_{n+1}: without them
# the brace strip turns that into "a_n + 1" — a different claim entirely
_SUB_SRC = "0123456789aehklmnopstx+-=()"
_SUBSCRIPT = str.maketrans(_SUB_SRC, "₀₁₂₃₄₅₆₇₈₉ₐₑₕₖₗₘₙₒₚₛₜₓ₊₋₌₍₎")
_SUP_SRC = "0123456789abcdefghijklmnoprstuvwxyz"
_SUPERSCRIPT = str.maketrans(_SUP_SRC, "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")


def _display_logs(s: str) -> str:
    def lowered(m: re.Match) -> str:
        base = m.group(1)
        if not all(ch in _SUB_SRC for ch in base):
            return m.group(0)
        return "log" + base.translate(_SUBSCRIPT)

    return re.sub(r"\blog_\{?(\w+)\}?", lowered, s)


def _display_indices(s: str) -> str:
    """a_4 → a₄, x_n → xₙ — the sequence-term notation a worksheet lives on.

    Single-letter bases only, anchored on a word boundary: log_3 and friends
    have a letter before theirs, so function names never lose their tails.
    """
    def lowered(m: re.Match) -> str:
        sub = m.group(2)
        if not all(ch in _SUB_SRC for ch in sub):
            return m.group(0)
        return m.group(1) + sub.translate(_SUBSCRIPT)

    # the lookahead keeps a_2i whole: sinking only the 2 would read as "a sub
    # two, times i", which is not what was written
    return re.sub(r"\b([A-Za-z])_\{?([0-9]+|[A-Za-z])\}?(?![A-Za-z0-9])", lowered, s)


def _display_powers(s: str) -> str:
    """x**2 → x², x^10 → x¹⁰, x^n → xⁿ — raised, not careted."""
    def raised(m: re.Match) -> str:
        exp = m.group(1)
        if not all(ch in _SUP_SRC for ch in exp):
            return "^" + exp
        return exp.translate(_SUPERSCRIPT)

    return re.sub(r"(?:\*\*|\^)\{?([0-9]+|[A-Za-z])\}?", raised, s)


def displayable(text: str) -> str:
    """The text as it should be SEEN — real notation, not spoken Korean.

    The transcript panel is the one place programmer ASCII can be improved
    into print notation: 2*x**2 becomes 2·x², sqrt becomes √, log_3 becomes
    log₃. Identity when there is no math, exactly like speakable() — the two
    are the same boundary split by destination: displayable() for the eye,
    speakable() for the ear.
    """
    if not text or not _MATH_SIGNAL.search(text):
        return text
    s = _latex(text, spoken=False)
    s = _display_powers(s)     # x**2 → x², x^10 → x¹⁰, x^n → xⁿ
    s = re.sub(r"\s*\*\s*", "·", s)
    s = s.replace("sqrt(", "√(")
    s = _display_logs(s)       # before the generic pass: log_3 keeps its name
    return _display_indices(s)


def speakable(text: str) -> str:
    """The text as it should be SPOKEN. Identity when there is no math in it."""
    if not text or not _MATH_SIGNAL.search(text):
        return text
    s = _latex(text, spoken=True)
    s = _primes(s)      # before parens: they consume the argument's ()
    s = _logs(s)        # before fractions: log_3 b/a keeps its argument whole
    # a_4 → "a 4": a teacher says the index, not the underscore. After _logs,
    # whose own underscores are already spoken as "밑이 …인 로그".
    s = re.sub(r"\b([A-Za-z])_\{?(\w+)\}?", r"\1 \2", s)
    s = _powers(s)
    s = _fractions(s)
    s = _operators(s)   # before equals: "= -1" wants 마이너스 first
    s = _equals(s)
    s = _parens(s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()
