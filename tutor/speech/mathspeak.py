"""ASCII math, said the way a Korean teacher says it.

TTS reads "f'(1) = 2x**2 - x" as punctuation soup. Everything the tutor SPEAKS
passes through speakable(), which rewrites the notations a K-12 worksheet
actually contains — powers, primes, logs, fractions, the four operators —
into the spoken forms: "f 프라임 1", "2 x 제곱 빼기 x", "2분의 1", "밑이 3인
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
_MATH_SIGNAL = re.compile(r"[*^=/×÷'′²³√]|\blog|\bsqrt|\bDerivative")


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


def speakable(text: str) -> str:
    """The text as it should be SPOKEN. Identity when there is no math in it."""
    if not text or not _MATH_SIGNAL.search(text):
        return text
    s = text
    s = _primes(s)      # before parens: they consume the argument's ()
    s = _logs(s)        # before fractions: log_3 b/a keeps its argument whole
    s = _powers(s)
    s = _fractions(s)
    s = _operators(s)   # before equals: "= -1" wants 마이너스 first
    s = _equals(s)
    s = _parens(s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()
