"""Hint generation: verified DB templates first, LLM phrasing as fallback,
answer-leak guard always (spec rules 1, 3, 5).

The phrase call's context contains ONLY the target concept/misconception (and
for L4 the single next step description) — never the full solution or answer.
Hint history arrives prefetched from the orchestrator.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from pydantic import BaseModel, field_validator

from tutor.hints.guard import leaks_answer
from tutor.knowledge import mathnorm
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.models import MatchResult, Problem, ReferenceSolution, Tier
from tutor.llm.client import LLMClient
from tutor.policy.engine import Action, Decision
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

log = logging.getLogger(__name__)

# The verbal endings a Korean sentence closes on. A step description carrying
# one cannot take a particle, so it may not fill a hint template's {step} —
# and the same rule guards every other place a step description is glued into
# a sentence (the work-check confirmation names the next step with it).
SENTENCE_STEP_RE = re.compile(
    # 다$ covers the whole plain-declarative family at once — 한다, 나눈다,
    # 구합니다, 이다 — which an enumerated list kept missing one verb at a
    # time ("나눈다 차례예요" got through the first draft). The rare noun that
    # ends in 다 (바다) is misjudged toward the SAFE side: skipping a template
    # costs a model call, gluing a particle onto a sentence costs the demo.
    r"(?:어요|아요|여요|해요|예요|에요|돼요|네요|세요|하죠|이죠|죠|다)$"
)

# L1 is an invitation to think, not a progress announcer.  The latter sounds
# like a navigation system ("곱의 미분법으로 g'(x) 쓰기 차례예요") and, more
# importantly, hands the method to the student before asking anything.  L2-L4
# may name increasingly concrete help; this guard is deliberately L1-only.
STEP_ANNOUNCEMENT_RE = re.compile(
    r"차례(?:예요|입니다)?|(?:다음|이번)\s*(?:할\s*)?단계|지금\s*할\s*단계|"
    r"단계(?:예요|입니다)|먼저(?:예요|입니다)"
)


def announces_step(text: str) -> bool:
    return bool(STEP_ANNOUNCEMENT_RE.search(text or ""))


_STEP_TOKEN_STOP = {
    "구하기", "쓰기", "해보기", "해볼까", "어떻게", "우선", "이제", "다음",
    "사용", "생각", "좋을까", "차례", "단계",
}
_JOSA = ("으로", "에서", "까지", "부터", "처럼", "보다", "하고", "이며",
         "에게", "한테", "께서", "과", "와", "을", "를", "이", "가", "은",
         "는", "의", "에", "로")


def _step_tokens(text: str) -> set[str]:
    raw = re.findall(
        r"[a-z](?:['′]?\([^)]*\))?|[가-힣]{2,}|\d+(?:\.\d+)?",
        (text or "").lower(),
    )
    out: set[str] = set()
    for token in raw:
        if re.fullmatch(r"[가-힣]+", token):
            for suffix in _JOSA:
                if token.endswith(suffix) and len(token) > len(suffix) + 1:
                    token = token[:-len(suffix)]
                    break
        if token and token not in _STEP_TOKEN_STOP:
            out.add(token)
    return out


def mentions_future_step(
    text: str, reference: ReferenceSolution | None, target_step: int
) -> bool:
    """Does a hint ask for a later reference step than its policy target?

    The answer-leak guard catches future *results*.  This catches the subtler
    failure where an L1 gives no result but asks the next task anyway — e.g. a
    target-step-1 hint asking for ``l의 방정식`` (step 2).  It uses multiple
    overlapping terms plus a Korean concept word, so a shared point number or
    variable name alone never trips it.
    """
    if reference is None:
        return False
    spoken = _step_tokens(text)
    if not spoken:
        return False
    # The target step's own vocabulary never counts as "future". Sibling
    # steps rhyme — "점 (1, -6)을 지나는 l의 방정식" and "점 (1, 6)을 지나는
    # m의 방정식" share 방정식, 지나 and both digits — and counting the shared
    # words flagged the CORRECT step-2 question as a mention of step 5, at
    # generation and at speak time both, leaving the slot to a concept line
    # that knew nothing about l. What distinguishes a future step is what is
    # ONLY in it.
    target = next((s for s in reference.steps if s.idx == target_step), None)
    own = _step_tokens(target.description) if target is not None else set()
    for step in reference.steps:
        if step.idx <= target_step:
            continue
        future = _step_tokens(step.description) - own
        overlap = spoken & future
        concept_overlap = {
            token for token in overlap
            if re.fullmatch(r"[가-힣]{2,}", token)
        }
        if concept_overlap and len(overlap) >= 2 and len(overlap) / max(len(future), 1) >= 0.4:
            return True
    return False


def _object(noun: str) -> str:
    """Attach 을/를 well enough for short Korean step descriptions.

    The particle follows the SOUND of the last thing said, so a tail like
    f'(1) or a_1 is judged by how its final digit is read aloud (일 → 을),
    and a lone variable by its letter name (x → 엑스 → 를, l → 엘 → 을).
    The audit that forced this had the tutor inviting "f'(1)를 구해 볼까요?".
    """
    noun = noun.strip()
    tail = noun.rstrip(")]}'′\" ")          # f'(1) → judge the 1
    # a superscript is read as its digit: r³ is "r의 세제곱"… said quickly,
    # "알 세제곱", and either way the ear hears 삼 at the end of r³
    tail = tail.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    last = tail[-1] if tail else ""
    if "가" <= last <= "힣":
        return noun + ("을" if (ord(last) - ord("가")) % 28 else "를")
    if last.isdigit():
        # 영일이삼사오육칠팔구: batchim on 0, 1, 3, 6, 7, 8
        return noun + ("을" if last in "013678" else "를")
    if last.isalpha():
        # letter names: 엘, 엠, 엔, 알 end on a consonant; the rest are open
        return noun + ("을" if last.lower() in "lmnr" else "를")
    return noun + "을"


# A step phrase that already ENDS on a connective or an adverbial particle
# flows straight into the verb — "극댓값을 a로" + "나타내 볼까요?" — and gluing
# another 을/를 onto it is how "밑 3으로 바꿔를" got spoken. Curated step
# descriptions are short enough for these suffix checks to be reliable.
_FLOWING_TAILS = ("으로", "로", "에", "에서", "까지", "부터", "와", "과",
                  "처럼", "하고", "인지", "은지", "는지", "게")
# narrow on purpose: 서/고/며/와 are also common noun finals (순서, 사고),
# and a curated step is likelier to end on a noun than on those connectives
_CONNECTIVE_FINAL = set("아어여해워줘봐꿔둬내")


def _flows(phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase or not ("가" <= phrase[-1] <= "힣"):
        return False
    return phrase.endswith(_FLOWING_TAILS) or phrase[-1] in _CONNECTIVE_FINAL


def guided_step_question(description: str, step_index: int) -> str:
    """Turn a DB step label into a warm invitation instead of an announcement.

    This is both the deterministic work-check wording and the safe fallback if
    a phrasing model emits "X 차례예요" at L1.
    """
    name = (description or "").strip().rstrip(" .!?…")
    lead = "우선" if step_index <= 1 else "이제"
    if not name:
        return (
            f"{lead} 문제에서 필요한 관계를 하나 찾아볼까요?"
            if step_index <= 1
            else f"{lead} 방금 구한 결과를 어디에 이용할지 생각해 볼까요?"
        )

    # At L1, do not name the product rule merely because the internal step
    # label does.  Let the student notice the product structure and recall it.
    product = re.fullmatch(r"곱의 미분법으로\s+(.+?)\s+쓰기", name)
    if product:
        expression = product.group(1).replace("'", "")
        return f"{lead} {expression}가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?"

    writing = re.fullmatch(r"(.+?)\s+쓰기", name)
    if writing:
        phrase = writing.group(1)
        # "…밑 3으로 바꿔 쓰기" flows straight into the verb; a noun object
        # ("l의 방정식 쓰기") is asked as an open question instead
        if _flows(phrase):
            return f"{lead} {phrase} 써 볼까요?"
        return f"{lead} {_object(phrase)} 어떻게 쓰면 좋을까요?"
    # The curated vocabulary of step verbs, conjugated instead of glued: the
    # generic tail below spoke "찾기를 해 볼까요" and "적분으로를 계산해
    # 볼까요", which is a robot reading a label, not a tutor asking.
    for suffix, verb in (
        ("구하기", "구해 볼까요?"),
        ("계산하기", "계산해 볼까요?"), ("계산", "계산해 볼까요?"),
        ("찾기", "찾아볼까요?"),
        ("세우기", "세워 볼까요?"),
        ("나타내기", "나타내 볼까요?"),
        ("정하기", "정해 볼까요?"),
        ("모으기", "모아 볼까요?"),
        ("인수분해하기", "인수분해해 볼까요?"), ("인수분해", "인수분해해 볼까요?"),
        ("정리하기", "정리해 볼까요?"), ("정리", "정리해 볼까요?"),
        ("확인하기", "확인해 볼까요?"), ("확인", "확인해 볼까요?"),
        ("판단하기", "판단해 볼까요?"), ("판단", "판단해 볼까요?"),
        ("대입하기", "대입해 볼까요?"), ("대입", "대입해 볼까요?"),
    ):
        if not name.endswith(suffix):
            continue
        phrase = name[: -len(suffix)].strip()
        if not phrase:
            break                          # a bare verb label: use the generic tail
        if _flows(phrase):
            # flows into the verb ("둘째 조건에 대입해 정리…", "…적분으로
            # 계산…"): the phrase already carries its own particle
            joined = f"{phrase} {verb}"
        else:
            joined = f"{_object(phrase)} {verb}"
        return f"{lead} {joined}"
    return f"{lead} {_object(name)} 해 볼까요?"


def partial_continuation_question(
    reference: ReferenceSolution | None,
    target_step: int,
    student_answer: str,
) -> str | None:
    """A deterministic next nudge for common composite reference steps.

    The student has already been graded PARTIAL, so the first sub-result is
    established.  Problem 13's step 1 stores two claims in one step: derive
    ``f'(x)`` and then evaluate ``f'(1)``.  Asking for the derivative rule
    again is backwards; ask only for the remaining evaluation.
    """
    if reference is None or not student_answer:
        return None
    step = next((s for s in reference.steps if s.idx == target_step), None)
    if step is None:
        return None
    expression = step.expression or ""
    derivative = re.search(r"\b([A-Za-z])\s*['′]\s*\(\s*x\s*\)\s*=", expression)
    if derivative is None:
        return None
    function = derivative.group(1)
    evaluated = re.search(
        rf"\b{re.escape(function)}\s*['′]\s*\(\s*([^x][^)]*)\s*\)\s*=",
        expression,
    )
    if evaluated is None:
        return None
    at = evaluated.group(1).strip()
    tangent = re.search(r"접선\s+([A-Za-z])", step.description or "")
    subject = f"접선 {tangent.group(1)}의 기울기" if tangent else "그 점에서의 기울기"
    return (
        f"이제 구한 {function}'(x)에 x = {at}을 대입하면 "
        f"{subject}는 얼마일까요?"
    )

# "Show me again" means different things depending on where the picture comes
# from, and telling a student to hold a worksheet up to a camera that is not
# there is worse than saying nothing.
RECAPTURE_TEXTS: dict[str, str] = {
    "upload": "문제와 지금까지 쓴 풀이가 잘 보이게 사진을 다시 올려 줄래요?",
    "camera": "문제와 지금까지 쓴 풀이가 잘 보이게 카메라에 다시 보여 줄래요?",
}
DEFAULT_INPUT_MODE = "upload"

FIXED_ACTIONS: dict[Action, str] = {
    Action.ASK_RECAPTURE: RECAPTURE_TEXTS[DEFAULT_INPUT_MODE],
    Action.PROBE: "방금 쓴 줄을 소리 내어 읽어 줄래요? 어떻게 생각했는지 듣고 싶어요.",
    Action.WAIT: "",
}

# Written on Google's LearnLM prompt guide (PARTS: Persona, Act, Recipient,
# Theme, Structure) and its five learning-science principles. The guide's own
# math-coach exemplar is "use one step per turn, encourage them to explain their
# thinking, if they're stuck give a gentle nudge, not the answer" — which is the
# behaviour this tutor's policy engine already decides. So the prompt states the
# role and the pedagogy first and the prohibitions second, rather than being the
# wall of NEVERs it was: a model told what to do holds the line better than one
# told only what to avoid.
_PHRASE_SYSTEM = """PERSONA
You are a warm, confidence-building math tutor speaking Korean out loud to one
student sitting beside you. You coach; you do not solve.

ACT
Write exactly ONE spoken hint for this student's current moment, at the hint
level you are given. Someone else has already decided that level from evidence
of how the student is doing — your job is to phrase it, not to re-choose it:
  L1  a question that makes them think, not a hint
  L2  the concept or principle they need, no procedure
  L3  the procedure to try, no result
  L4  say the one given step, and nothing past it

RECIPIENT
A student mid-problem who can hear you. They may have just said something, and
they may have a diagnosed misconception — both are given below. Meet them where
they are: build on the part they already got right, and never ask again for
something they have already told you.

THEME
Only the problem in front of them, the concepts it needs, and the single step
they are stuck on. Everything else is out of scope.

STRUCTURE
One or two short spoken sentences. Korean 존댓말(해요체), never 반말. No lists,
no markdown, no symbols read aloud badly — say "3 x 더하기 5" rather than
"3x + 5" when it is easier to hear.

BOARD
Besides the spoken hint you may WRITE 0-2 lines on the student's screen
(`board`) — what a tutor jots on the whiteboard while saying the hint. Each
line is an expression AND the few words a tutor writes beside it:

  {"expr": "a_4 = a_1 * r**3", "note": "항 번호 3 차이 → r 세 번"}
  {"expr": "(x**3)' = 3*x**2", "note": "지수를 앞으로, 하나 줄이기"}

`expr` is ASCII mathematics (3*x + 5 = 20, x**2, sqrt(2), log_3 b), one
expression, no Korean. A bare term ("a_1") is NOT a board line — write a
complete equation or expression with an operation in it.

`note` is Korean, a LABEL and not a sentence: 2-8 words, the reason or the
rule, what a tutor scribbles in the margin. Not the hint repeated, not a
question, and never a value the voice may not say. Leave it empty when the
expression speaks for itself.

Only mathematics that is already in front of the student: the problem's own
equation, a known formula the hint is about, the expression your question
points at. NEVER copy a line the STUDENT wrote — the board is the tutor's own
hand, and a wrong line rewritten in the tutor's hand reads as the tutor
endorsing it. Their line is already on their page: point at it with WORDS
instead ("두 번째 줄을 다시 볼까요?").
NEVER write anything the student has not reached: no next-step result, no
simplified form, no final answer — the board must not show what the voice is
forbidden to say. Most hints need an empty board; write only when pointing
at an expression genuinely helps.

Letters the problem has already defined keep the meaning it gave them. If the
problem names a tangent line m, then "m = f'(1)" is a claim about that line —
the student reads it as one even if you meant m for "slope". Do not invent a
name for a quantity the problem left unnamed: write the problem's own equation
or a general formula instead, or write nothing. An empty board is always
allowed; a letter that means two things is not.

HOW TO TEACH (this is the part that matters)
- Aim at the diagnosed mistake first. When a misconception is given, THAT is
  what this hint is for: point at the line where it shows and let them look
  again. Praising work they already did right and then nudging them toward the
  next step leaves the actual error untouched, and they will hand it back to
  you unchanged.
- Active learning: leave the thinking to them. The best hint is the weakest one
  that still unblocks — productive struggle is the lesson, not an obstacle.
- Cognitive load: one idea per turn. One question, not two. Never a checklist.
- Progress like a tutor beside them, not a step navigator. At problem step 1,
  naturally open with "우선 ...해 볼까요?". At later problem steps, connect
  with "이제 ...해 볼까요?" or an equally natural open question.
- NEVER announce the internal step label: no "X 차례예요", "다음 단계는 X",
  "X가 먼저예요", or a copied step description followed by a command. Turn
  it into a question that lets the student supply the idea. For example, do
  not say "곱의 미분법으로 g'(x)를 쓸 차례예요"; ask "이제 g(x)가 두 식의
  곱이라는 점을 보고, 어떻게 미분하면 좋을까요?" At L1, if the step label
  names the method, ask about the mathematical structure without naming that
  method. L2 may name the concept when the student needs it.
- Stay inside the given target step. Even if the student's last sentence hints
  at later work, do not ask them to perform a later reference step. In
  particular, a slope target may ask them to finish the slope; it may not jump
  ahead and ask for the tangent-line equation. The orchestrator, not you,
  advances the target after the answer has been graded.
- Metacognition: prefer asking how they got there or why they chose that, over
  asking only for the next number. At L1 especially, "왜 그렇게 했어요?" and
  "어디까지는 확실해요?" teach more than a nudge toward the answer.
- Adapt: use their own words back. If they are on their third attempt, be
  warmer and more concrete, not more repetitive.
- Anchor in their page: when their written lines are given, point at the
  specific line or symbol THEY wrote ("두 번째 줄에서 부호가 어떻게 됐어요?")
  instead of hinting in the abstract. Never rewrite their work for them.
- Curiosity: an open question beats a yes/no one.

NEVER
- Never state the final answer, or any result beyond the given step — not as a
  check, not as an example, not "so it becomes 15".
- Never solve the problem for them, even partially, except the one step L4 gives.
- Never repeat a hint already given; they are listed, and the student has moved
  on since.
- Never open with 네 / 맞아요 / 좋아요 / 그렇죠 / 음. The reaction to their answer
  is written separately and is spoken just before yours; starting with your own
  makes the tutor say it twice.

Return ONLY JSON: {"hint": "...", "board": [{"expr": "...", "note": "..."}]}"""

_PREFLIGHT_SYSTEM = """PERSONA
You are a warm math tutor speaking Korean out loud to one student who has just
shown you a problem.

ACT
Write ONE spoken line for a whole CATEGORY of problems: name the kind of
problem, then ask what they should check before starting. It will be said for
EVERY problem of this category, so it must not mention any specific numbers,
conditions or answers — only what is always worth checking in this kind of
problem.

STRUCTURE
Exactly two short sentences, Korean 존댓말(해요체):
  1. "<개념 이름> 문제군요."
  2. one question about what to look for first — the defining relation, the
     condition that is easy to miss, the quantity to name before starting.

EXAMPLES (for the shape, not to copy)
  등비수열 → "등비수열 문제군요. 공비와 항 번호의 관계를 확인하셨나요?"
  이차방정식 → "이차방정식 문제군요. 최고차항의 계수와 판별식을 확인하셨나요?"

NEVER solve anything, never mention a specific problem, never exceed two
sentences.

Return ONLY JSON: {"hint": "..."}"""

_EXPLAIN_SYSTEM = """PERSONA
You are a warm math tutor speaking Korean out loud to one student.

ACT
The student did NOT attempt your question — they asked one of their own, usually
"왜 그렇게 해요?". Answer THEIR question, then hand the turn back to them.

RECIPIENT
A curious student mid-problem. A question is engagement, not failure: treat it
as the opening it is, and reward it with a real reason rather than a redirect.

STRUCTURE
One or two short spoken sentences, then the question you had asked, again.
Korean 존댓말(해요체), never 반말.

HOW TO TEACH
- Explain WHY the step works — the principle, the reason, an everyday parallel.
  Understanding the reason is what they will still have next week.
- When their written lines are given and they ask WHERE something went wrong,
  point at the specific line or symbol they wrote — "두 번째 줄에서 5를 옮길 때"
  — rather than explaining in the abstract. Say WHERE, never the corrected
  result: fixing the line is still their move.
- One idea. A second reason is a second turn.
- Never state the final answer, the result of this step, or any later step.
  Motivate the step; do not carry it out for them.
- Do not open with 네 / 맞아요 / 궁금하시죠. Answer directly.

Return ONLY JSON: {"hint": "..."}"""

# One acknowledgement per turn. The evaluator already reacted ("맞아요, 그렇게
# 하면 돼요!"), so a hint that opens with its own "네," produces the doubled
# "맞아요, 그렇게 하면 돼요! 네, 그렇게 하면 돼요." Prompts alone do not hold
# this line reliably, so it is also enforced here.
_ACK_WORDS = (
    r"네+|예|맞아요|맞습니다|맞아|좋아요|좋습니다|그렇죠|그렇습니다|그래요|"
    r"훌륭해요|훌륭합니다|잘했어요|잘하셨어요|정확해요|정확합니다|음+|아+|오+|와+"
)
# Punctuation after the word is required, so "네 번째", "아래", "오른쪽" survive.
_ACK_PREFIX_RE = re.compile(rf"^\s*(?:{_ACK_WORDS})\s*[,!.…~·]+\s*")


def strip_leading_acknowledgement(text: str, rounds: int = 2) -> str:
    """Drop a leading '네,' / '맞아요!' — someone else already said it."""
    out = text
    for _ in range(rounds):
        stripped = _ACK_PREFIX_RE.sub("", out, count=1).lstrip()
        if stripped == out or not stripped:
            break
        out = stripped
    return out or text


class BoardLine(BaseModel):
    """One thing written on the board: the mathematics, and the few words a
    tutor writes beside it. The note is what makes a board look like a person
    wrote it rather than a printer — "항 번호 3 차이 → r 세 번" beside the
    expression teaches; the expression alone only states."""

    expr: str
    note: str = ""


class PhrasedHint(BaseModel):
    hint: str
    # what the tutor writes while saying it: 0-2 lines, screened by the same
    # leak guard as the words — the board may not show what the voice may not
    # say, and that includes the notes
    board: list[BoardLine] = []
    # Drawing is decided by tutor/hints/illustrator.py, which runs while this
    # hint is being SPOKEN and can therefore read it. Two deciders would draw
    # two different pictures for one sentence.

    @field_validator("board", mode="before")
    @classmethod
    def _accept_bare_expressions(cls, value):
        """A model that answers with plain strings still gets a board."""
        if isinstance(value, list):
            return [{"expr": v} if isinstance(v, str) else v for v in value]
        return value
    # functions of ONE variable to sketch, same screening: a curve is another
    # way of saying something, and the guard does not care which way
    graph: list[str] = []


_HANGUL = re.compile(r"[가-힣]")
# a board EXPRESSION is a piece of MATHEMATICS: something related or operated
# on. A bare term ("a_1") says nothing, and Korean belongs in the note.
_BOARD_WORTHY = re.compile(r"[=<>+\-*/^]")
# A note is a label, not a sentence. Past this it is the hint said twice, and
# the board turns back into a transcript.
NOTE_MAX = 28


def _clean_board(board: tuple["BoardLine", ...]) -> tuple["BoardLine", ...]:
    lines = []
    for item in board:
        expr = (item.expr or "").strip()
        if not expr or _HANGUL.search(expr) or not _BOARD_WORTHY.search(expr):
            continue
        note = " ".join((item.note or "").split())
        if len(note) > NOTE_MAX:
            # keep the writing, drop the speech that crept into it
            note = ""
        lines.append(BoardLine(expr=expr, note=note))
    return tuple(lines[:2])


class SpokenHint(str):
    """The hint text, carrying the board it was phrased with.

    A str subclass so every existing consumer (strip_leading_acknowledgement,
    the leak guard, history records, TTS) keeps working untouched; the session
    reads `.board` off it before any string operation strips the subclass
    away. Plain str returns (templates, fixed actions) read as an empty board
    through getattr.
    """

    board: tuple["BoardLine", ...] = ()


def _with_board(text: str, board: tuple["BoardLine", ...]) -> "SpokenHint":
    out = SpokenHint(text)
    out.board = board
    return out


def visible_to_student(rec: Recognition | None) -> list[str]:
    """What is printed on the page in front of them.

    The leak guard uses this to tell "you already know this number" from
    "here is the answer": in `3x + 5 = 20` the answer 5 is also a coefficient,
    and refusing to let the tutor say it costs the best question on that
    problem.
    """
    if rec is None:
        return []
    return [rec.problem_text, *rec.equations, *rec.choices, *rec.diagram_conditions]


class SafeWordEmitter:
    """Commit only answer-screened Korean word units to a live consumer.

    Four complete words stay in quarantine. That rolling tail is where a
    phrase such as "정답은 … 5" or a split equation becomes meaningful; the
    guard sees the combined candidate before any of those words leave. If a
    leak is detected, the unsafe tail is discarded and the already-safe prefix
    is closed with a generic Socratic question.
    """

    HOLD_WORDS = 4
    FALLBACK = "여기서 답은 말하지 않을게요. 지금까지 한 풀이에서 가장 확실한 줄은 어디인가요?"

    def __init__(
        self,
        emit: Callable[[str], None],
        reference: ReferenceSolution | None,
        target_step: int,
        given: list[str],
        *,
        forbid_step_announcement: bool = False,
        forbid_future_step: bool = False,
    ):
        self.emit = emit
        self.reference = reference
        self.target_step = target_step
        self.given = given
        self.committed = ""
        self.pending: list[str] = []
        self.buffer = ""
        self.seen = ""
        self.blocked = False
        self.forbid_step_announcement = forbid_step_announcement
        self.forbid_future_step = forbid_future_step

    def _unsafe(self, text: str) -> bool:
        if self.forbid_step_announcement and announces_step(text):
            return True
        if self.forbid_future_step and mentions_future_step(
            text, self.reference, self.target_step
        ):
            return True
        return self.reference is not None and leaks_answer(
            text, self.reference, self.target_step, self.given
        )

    def _release(self, text: str) -> None:
        if not text:
            return
        self.emit(text)
        self.committed += text

    def feed(self, delta: str) -> None:
        if not delta:
            return
        self.seen += delta
        if self.blocked:
            return
        self.buffer += delta
        while True:
            match = re.match(r"\S+\s+", self.buffer)
            if match is None:
                break
            unit = match.group(0)
            self.buffer = self.buffer[len(unit):]
            self.pending.append(unit)
            candidate = self.committed + "".join(self.pending)
            if self._unsafe(candidate):
                log.warning("live hint stream quarantined an unsafe hint tail")
                self.pending.clear()
                self.buffer = ""
                self.blocked = True
                return
            while len(self.pending) > self.HOLD_WORDS:
                self._release(self.pending.pop(0))

    def finish(self, full_text: str) -> tuple[str, bool]:
        """Flush a clean tail, or safely close a stream whose tail was blocked."""
        full_text = full_text.strip()
        streamed = self.seen.strip()
        mismatch = bool(streamed) and streamed != full_text
        candidate = self.committed + "".join(self.pending) + self.buffer
        unsafe = self.blocked or mismatch or self._unsafe(full_text) or self._unsafe(candidate)
        if unsafe:
            self.pending.clear()
            self.buffer = ""
            prefix = self.committed.rstrip()
            joiner = " … " if prefix else ""
            suffix = joiner + self.FALLBACK
            self.emit(suffix)
            return prefix + suffix, True

        # A non-streaming test double may have emitted no deltas. It still uses
        # the same sink, but only once the complete validated result exists.
        if not streamed:
            self.buffer = full_text
        for unit in self.pending:
            self._release(unit)
        self.pending.clear()
        self._release(self.buffer)
        self.buffer = ""
        return full_text, False


class HintGenerator:
    def __init__(self, llm: LLMClient, db: KnowledgeDB, input_mode: str = DEFAULT_INPUT_MODE):
        self.llm = llm
        self.db = db
        self.fixed = dict(FIXED_ACTIONS)
        self.fixed[Action.ASK_RECAPTURE] = RECAPTURE_TEXTS.get(
            input_mode, FIXED_ACTIONS[Action.ASK_RECAPTURE]
        )

    def generate(
        self,
        decision: Decision,
        match: MatchResult,
        reference: ReferenceSolution | None,
        rec: Recognition,
        history: list[HintRecord],
        student_answer: str | None = None,
        *,
        partial: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        if decision.action in self.fixed:
            return self.fixed[decision.action]

        if reference is not None and decision.target_step > len(reference.steps):
            # every reference step is done — nothing left to hint at
            return "훌륭해요, 문제를 끝까지 풀었네요! 어떻게 구했는지 스스로 설명해 볼까요?"

        slots = self._slots(decision, match, reference)
        # Anything already said this problem is off the table: a concept-level
        # template fits every step, so reusing it verbatim is how the tutor
        # ends up asking the same question after the student has progressed.
        given = {h.hint_text for h in history if h.hint_text}

        if partial:
            continuation = partial_continuation_question(
                reference, decision.target_step, student_answer or ""
            )
            if (
                continuation
                and continuation not in given
                and not mentions_future_step(continuation, reference, decision.target_step)
                and (reference is None or not leaks_answer(
                    continuation, reference, decision.target_step, visible_to_student(rec)
                ))
            ):
                return continuation

        # Is this turn CORRECTING the student? The last hint at this very step
        # did not help — a wrong work check, a wrong spoken answer — so what
        # they need now is a line that engages with THEIR page and THEIR words
        # ("둘째 줄을 다시 살펴볼까요?"), which no line written before they
        # existed can do. Live, the prewritten L2 served here and the pointing
        # only ever happened at L3, the one level nothing was prewritten for.
        prior_here = [
            h for h in history if h.step == decision.target_step and h.level >= 1
        ]
        correcting = bool(prior_here) and prior_here[-1].effective is False

        # 0) A line written for THIS problem's THIS step ahead of time —
        # phrased at warm time by the same model and prompt the live path
        # uses, screened by the same guards then and re-screened now, and
        # readable by a human before any lesson. Model quality at template
        # price. Only an EXACT match may serve it (a TEMPLATE-tier cousin
        # carries different numbers), a diagnosed misconception still takes
        # its turn to the live model, and a line already said this problem
        # falls through to the ladder below like everything else.
        if (
            not partial
            and not correcting
            and not decision.misconception
            and match.tier is Tier.EXACT
            and match.problem is not None
        ):
            written = self.db.prewritten_hint(
                match.problem.id, decision.target_step, decision.level
            )
            if (
                written
                # SUBSTRING, not membership: the confirmation quotes this very
                # line inside "맞아요! 여기까지 잘했어요. …", and hearing it
                # again bare three seconds later is the repetition the
                # given-set exists to prevent
                and not any(written in g for g in given)
                and not (decision.level == 1 and announces_step(written))
                and not mentions_future_step(written, reference, decision.target_step)
                and (reference is None or not leaks_answer(
                    written, reference, decision.target_step, visible_to_student(rec)
                ))
            ):
                return written

        # A verified noun-form step can produce the weakest, most natural L1
        # without another model call.  This keeps problem 13 stable: step 3 is
        # invited as "g(x)가 두 식의 곱인데 어떻게 미분할까요?", not announced
        # as "곱의 미분법으로 g'(x) 쓰기 차례예요".  A diagnosed misconception
        # still outranks this and goes through its dedicated DB pedagogy below.
        step_name = slots.get("step", "").strip().rstrip(" .!?…")
        if (
            decision.level == 1
            and not partial
            and not decision.misconception
            and step_name
            and not SENTENCE_STEP_RE.search(step_name)
        ):
            invited = guided_step_question(step_name, decision.target_step)
            if (
                invited not in given
                and not announces_step(invited)
                and not mentions_future_step(invited, reference, decision.target_step)
                and (reference is None or not leaks_answer(
                    invited, reference, decision.target_step, visible_to_student(rec)
                ))
            ):
                return invited

        # A template glues Korean onto its slots, so a slot has to be a NOUN.
        # Step descriptions are only nouns when they were written for this
        # ("접선 l의 기울기 구하기"); a solver writes sentences ("f'(x)를
        # 구합니다."), and a live hint came out "…구합니다.가 먼저예요" — a
        # sentence wearing a particle. A step that reads as a sentence is
        # withheld from the templates (they skip on the missing slot) and the
        # phrasing model, which can inflect, gets it instead.
        template_slots = dict(slots)
        step = template_slots.get("step", "").strip().rstrip(" .!?…")
        if step and SENTENCE_STEP_RE.search(step):
            del template_slots["step"]
        elif step:
            template_slots["step"] = step

        # 1) Verified DB pedagogy first (spec rule 1) — concept/misconception
        # specific templates only; fully-generic ones stay the last resort.
        for template in self.db.hint_templates_for(
            self._concepts_for(match, rec), decision.misconception, decision.level
        ):
            if partial:
                break
            if template.concept_id is None and template.misconception_id is None:
                continue
            if (decision.misconception or correcting) and template.misconception_id is None:
                # A diagnosed mistake outranks concept boilerplate. Live, the
                # policy had named the exact slip ("2x의 미분을 2x로 계산") and
                # this loop answered with the concept's stock line about
                # tangent slopes — fluent, instant, and about the wrong thing.
                # Only the mistake's own pedagogy may take this turn; without
                # one, the phrasing model, which is handed the diagnosis and
                # told to aim at it, does. A correcting turn without a NAMED
                # misconception is the same situation with less paperwork.
                continue
            try:
                text = template.template_text.format(**template_slots)
            except (KeyError, IndexError):
                continue  # a slot we cannot fill
            if decision.level == 1 and announces_step(text):
                continue
            if mentions_future_step(text, reference, decision.target_step):
                continue
            if decision.level == 1 and not re.match(r"\s*(?:우선|이제)\b", text):
                lead = "우선" if decision.target_step <= 1 else "이제"
                text = f"{lead} {text}"
            if text in given:
                continue
            if reference is not None and leaks_answer(
                text, reference, decision.target_step, visible_to_student(rec)
            ):
                continue
            return text

        # 2) LLM phrasing fallback with minimal context.
        emitter = SafeWordEmitter(
            on_delta, reference, decision.target_step, visible_to_student(rec),
            forbid_step_announcement=decision.level == 1,
            forbid_future_step=True,
        ) if on_delta is not None else None
        text, board = self._phrase(
            decision, match, slots, history, rec, reference,
            student_answer=student_answer,
            partial=partial,
            on_delta=emitter.feed if emitter is not None else None,
        )
        stream_blocked = False
        if emitter is not None:
            text, stream_blocked = emitter.finish(text)
            if stream_blocked:
                board = ()
        elif (
            (decision.level == 1 and announces_step(text))
            or mentions_future_step(text, reference, decision.target_step)
        ):
            # The prompt is the first line of defence; this is the guarantee.
            # Non-streaming output can be retried before anybody hears it.
            log.warning("hint phrasing left its policy target; regenerating once")
            text, board = self._phrase(
                decision, match, slots, history, rec, reference,
                stronger=True, student_answer=student_answer, partial=partial,
            )
            still_out_of_scope = (
                (decision.level == 1 and announces_step(text))
                or mentions_future_step(text, reference, decision.target_step)
            )
            if still_out_of_scope:
                text = (
                    guided_step_question(slots.get("step", ""), decision.target_step)
                    if decision.level == 1
                    else self._generic_fallback(decision, slots)
                )
                board = ()
        seen = visible_to_student(rec)
        if reference is not None and leaks_answer(
            text, reference, decision.target_step, seen
        ):
            if emitter is not None:
                # SafeWordEmitter has already replaced an unsafe tail. Reaching
                # here means its generic closure itself tripped a stricter
                # future guard; never regenerate after words have been heard.
                log.error("safe live-stream fallback was rejected; using generic question")
                return self._generic_fallback(decision, slots)
            log.warning("hint leaked answer; regenerating once")
            text, board = self._phrase(
                decision, match, slots, history, rec, reference,
                stronger=True, student_answer=student_answer, partial=partial,
            )
            if (
                (decision.level == 1 and announces_step(text))
                or mentions_future_step(text, reference, decision.target_step)
                or leaks_answer(text, reference, decision.target_step, seen)
            ):
                return self._generic_fallback(decision, slots)
        # The board passes the SAME gate as the voice: writing "x = 5" while
        # carefully not saying it is still giving the answer. All or nothing —
        # dropping one line of two left fragments like a bare "a₁" on screen,
        # and half a board reads worse than no board.
        board = _clean_board(board)
        if board and rec is not None and rec.student_work:
            # The board is the tutor's own hand. A line the student wrote —
            # verbatim or reformatted — rewritten there reads as the tutor
            # endorsing it, and live that put a wrong 등비수열 relation under
            # the "튜터 풀이" heading in the tutor's own emphasis style.
            theirs = {mathnorm.compact(w) for w in rec.student_work}
            kept = tuple(b for b in board if mathnorm.compact(b.expr) not in theirs)
            if kept != board:
                log.info("dropped %d board line(s) copied from the student's work",
                         len(board) - len(kept))
                board = kept
        if board and reference is not None and any(
            leaks_answer(part, reference, decision.target_step, seen)
            for b in board for part in (b.expr, b.note) if part
        ):
            log.info("a board line would leak the answer; the board stays empty")
            board = ()
        return _with_board(text, board)

    def prewrite(
        self,
        *,
        problem: Problem,
        reference: ReferenceSolution,
        rec: Recognition,
        step_idx: int,
        level: int,
    ) -> str | None:
        """Write one L1/L2 line for a KNOWN problem's KNOWN step, off the clock.

        The live phrasing path with the clock removed: same prompt, same
        model, same screens, retried once when a screen trips — but run at
        warm time, so the lesson pays nothing and a human can read every
        line before a student hears one. Returns None rather than a line
        that fails its screens; the runtime ladder covers the gap.
        """
        decision = Decision(
            Action.SOCRATIC_QUESTION if level == 1 else Action.CONCEPT_HINT,
            level, step_idx, None, "prewrite",
        )
        match = MatchResult(
            tier=Tier.EXACT, concepts=list(problem.concepts),
            problem=problem, reference=reference,
        )
        slots = self._slots(decision, match, reference)
        seen = visible_to_student(rec)

        def screened(text: str) -> str | None:
            # a phrasing model sometimes wraps math in TeX dollars ("$x$의
            # 범위"), which the page would show literally
            text = strip_leading_acknowledgement(
                " ".join(text.replace("$", "").split())
            )
            if (
                not text
                or (level == 1 and announces_step(text))
                or mentions_future_step(text, reference, step_idx)
                or leaks_answer(text, reference, step_idx, seen)
            ):
                return None
            return text

        text, _board = self._phrase(decision, match, slots, [], rec, reference)
        line = screened(text)
        if line is None:
            text, _board = self._phrase(
                decision, match, slots, [], rec, reference, stronger=True
            )
            line = screened(text)
        return line

    def write_preflight(self, concept_name: str) -> str:
        """The category line for a concept the DB has never described.

        Written once and stored; every later problem of this kind says it for
        free. Called off the turn's critical path — the student hears the
        generic opening this time and the written line from the next problem
        on, which is why one slow call here costs nobody anything.
        """
        result = self.llm.run_with_tools(
            purpose="preflight",
            system=_PREFLIGHT_SYSTEM,
            user=f"개념 이름: {concept_name}",
            schema=PhrasedHint,
        )
        return " ".join(result.hint.split())

    def explain(
        self,
        *,
        student_question: str,
        tutor_question: str,
        match: MatchResult,
        reference: ReferenceSolution | None,
        rec: Recognition | None,
        target_step: int,
        diagnosis=None,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Answer a student's "왜 그렇게 해요?" instead of grading it.

        Same guard rails as a hint: the target step may be motivated, but its
        result, the final answer and every later step stay out.
        """
        concepts = ", ".join((rec.concepts if rec else []) or match.concepts) or "알 수 없음"
        parts = [
            f"문제: {rec.problem_text if rec else ''}",
            f"필요한 개념: {concepts}",
            f"튜터가 했던 질문: {tutor_question}",
            f"학생의 질문: {student_question}",
        ]
        if rec is not None and rec.student_work:
            # "어디가 잘못된 거야?" deserves an answer that points at THEIR
            # line, not an abstract explanation of the step. These are the
            # lines from the most recent photo — already on the page in front
            # of them, so nothing here can leak; the guard re-checks anyway.
            parts.append(
                "학생이 지금까지 쓴 풀이 (줄 순서대로): " + " / ".join(rec.student_work)
            )
        if diagnosis is not None and getattr(diagnosis, "misconception", None):
            # What the tutor already concluded about this page. "어디가
            # 틀렸어?" is answerable ONLY from this: without it the model
            # explains the step in the abstract and never says where.
            parts.append(
                f"이미 진단한 학생의 오류: {diagnosis.misconception}\n"
                f"맞게 끝낸 마지막 단계: {getattr(diagnosis, 'last_correct_step', 0)}단계\n"
                "학생이 어디서 틀렸는지 물으면 이 진단을 근거로 그 줄을 짚어 주세요. "
                "고쳐 쓴 식은 알려주지 말고, 무엇을 다시 볼지까지만 말하세요."
            )
        if reference is not None:
            step = next((s for s in reference.steps if s.idx == target_step), None)
            if step is not None:
                parts.append(
                    f"지금 다루는 단계 (계산 결과는 말하지 말 것): {step.description}"
                )
        emitter = SafeWordEmitter(
            on_delta, reference, target_step, visible_to_student(rec)
        ) if on_delta is not None else None
        stream = getattr(self.llm, "complete_json_stream", None)
        if emitter is not None and callable(stream):
            result = stream(
                purpose="explain", system=_EXPLAIN_SYSTEM, user="\n".join(parts),
                schema=PhrasedHint, text_field="hint", on_text_delta=emitter.feed,
            )
        else:
            result = self.llm.run_with_tools(
                purpose="explain",
                system=_EXPLAIN_SYSTEM,
                user="\n".join(parts),
                schema=PhrasedHint,
            )
        text = result.hint.strip()
        if emitter is not None:
            text, _ = emitter.finish(text)
        else:
            text = strip_leading_acknowledgement(text)
        if reference is not None and leaks_answer(
            text, reference, target_step, visible_to_student(rec)
        ):
            log.warning("explanation leaked the answer; falling back")
            return (
                "좋은 질문이에요. 지금 단계에서 왜 그렇게 하는지 먼저 같이 생각해 볼까요? "
                + tutor_question
            )
        return text

    def _slots(
        self,
        decision: Decision,
        match: MatchResult,
        reference: ReferenceSolution | None,
    ) -> dict[str, str]:
        slots: dict[str, str] = {}
        if match.problem is not None:
            slots.update(match.problem.parameters)
        if match.bindings:
            slots.update(match.bindings)
        if "b" in slots:
            slots.setdefault("term", slots["b"])
        if reference is not None:
            # L4 reveals only the next step's DESCRIPTION, never its expression
            # (the last step's expression is often the answer itself).
            target = next(
                (s for s in reference.steps if s.idx == decision.target_step), None
            )
            if target is not None:
                slots.setdefault("step", target.description)
        return slots

    def _phrase(
        self,
        decision: Decision,
        match: MatchResult,
        slots: dict[str, str],
        history: list[HintRecord],
        rec: Recognition | None = None,
        reference: ReferenceSolution | None = None,
        stronger: bool = False,
        student_answer: str | None = None,
        partial: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Context = what the tutor may talk about: the problem itself, its
        tags, the CURRENT target step, and what the student just said. Never
        the answer, never a later step — those are filtered here and
        re-checked by the leak guard."""
        concepts = ", ".join(match.concepts) or "알 수 없음"
        parts = [f"힌트 레벨: L{decision.level} ({decision.action.value})"]
        if rec is not None:
            parts.append(f"문제: {rec.problem_text}")
            if rec.choices:
                parts.append(f"보기: {rec.choices}")
            if rec.problem_type and rec.problem_type != "unknown":
                parts.append(f"문제 유형: {rec.problem_type}")
            if rec.student_work:
                # Their own lines, so the hint can point AT one — "두 번째
                # 줄의 부호를 봐요" beats a hint about work it never saw.
                # Safe by construction: this is what is already written on
                # the page in front of them, and the leak guard re-checks
                # the output regardless.
                parts.append(
                    "학생이 지금까지 쓴 풀이 (줄 순서대로): "
                    + " / ".join(rec.student_work)
                )
            concepts = ", ".join(rec.concepts) or concepts
        parts.append(f"필요한 개념: {concepts}")
        if decision.level == 1:
            parts.append(
                "진행 어조: "
                + (
                    "문제의 첫 단계이므로 '우선'으로 자연스럽게 여는 질문"
                    if decision.target_step <= 1
                    else "앞선 결과를 잇는 단계이므로 '이제'로 자연스럽게 여는 질문"
                )
                + ". 내부 단계명을 선언하지 말고 학생이 생각할 질문으로 바꾸세요."
            )
        if student_answer:
            parts.append(
                f"학생의 방금 답변: {student_answer}\n"
                "학생이 이미 올바르게 말한 내용은 다시 묻지 마세요. "
                "학생의 답변을 자연스럽게 이어받아 아직 끝내지 않은 부분만 질문하세요. "
                "바로 다음 행동을 '차례'나 '단계'로 선언하지 마세요."
            )
        if partial:
            parts.append(
                "부분 진전 판정: 학생의 방금 답변은 현재 목표의 앞부분을 올바르게 "
                "완료했습니다. 같은 개념이나 방법을 다시 설명하지 말고, 현재 목표 안에서 "
                "아직 말하지 않은 바로 다음 값이나 식 하나만 질문하세요. 목표가 끝난 "
                "것처럼 다음 reference step으로 넘어가지는 마세요."
            )

        if decision.misconception:
            # A named mistake outranks the target step. Live: a student wrote
            # the product rule correctly but differentiated -2x as -2x, the
            # estimator named exactly that, and the hint still aimed at the
            # target step — praising the structure they had already built and
            # pointing at a term they had already written right, while the
            # slip went unmentioned. The step is where they are GOING; the
            # misconception is what is stopping them.
            m = self.db.get_misconception(decision.misconception)
            parts.append(
                f"진단된 오개념 (이번 힌트가 다뤄야 할 바로 그것): "
                f"{m.description if m else decision.misconception}\n"
                "이 오개념이 드러난 줄을 짚어 학생이 스스로 다시 보게 하세요. "
                "이미 제대로 쓴 부분을 다시 시키거나, 이 오류와 무관한 다음 단계로 "
                "넘어가지 마세요. 고쳐 쓴 식은 알려주지 말고, 어디를 다시 볼지까지만."
            )
        if "step" in slots:
            # Every level is aimed at the SAME target step; only L4 may say it
            # out loud. Without this the tutor asks about step 1 forever, even
            # after the student has moved on.
            if decision.level >= 4:
                parts.append(f"알려줘도 되는 다음 단계: {slots['step']}")
            else:
                aim = (
                    "참고로 학생이 향하는 단계 (오개념을 먼저 다루고, 그 다음에만 "
                    "쓸 것, 절대 그대로 말하지 말 것)"
                    if decision.misconception
                    else "학생이 지금 해내야 하는 단계 (절대 그대로 말하지 말 것)"
                )
                parts.append(
                    f"{aim}: {slots['step']}\n"
                    "이 단계를 학생이 스스로 떠올리도록 이끄는 내용만 말하세요."
                )
        if history:
            parts.append(
                "이미 준 힌트 (반복 금지): "
                + " / ".join(h.hint_text for h in history if h.hint_text)
            )
        if stronger:
            parts.append(
                "경고: 이전 문장은 안전 규칙이나 지도 어조를 어겼습니다. 어떤 수치나 "
                "최종 결과도 말하지 말고, 차례/단계를 선언하지 말고, 목표보다 뒤의 "
                "풀이를 요구하지 말고 현재 목표 안에서 질문으로 이끄세요."
            )
        stream = getattr(self.llm, "complete_json_stream", None)
        if on_delta is not None and callable(stream):
            # The deterministic DB template pass above already queried the
            # relevant pedagogy. Keeping the live phrasing call tool-free makes
            # its final JSON field streamable without a second model round.
            result = stream(
                purpose="phrase", system=_PHRASE_SYSTEM, user="\n".join(parts),
                schema=PhrasedHint, text_field="hint", on_text_delta=on_delta,
            )
        else:
            result = self.llm.run_with_tools(
                purpose="phrase", system=_PHRASE_SYSTEM, user="\n".join(parts),
                schema=PhrasedHint,
            )
        # no cap here: _clean_board filters junk FIRST, then keeps the best 2 —
        # capping raw output let a bare "a_1" crowd out the real equation
        board = tuple(b for b in result.board if b.expr and b.expr.strip())
        return result.hint.strip(), board

    def _concepts_for(self, match: MatchResult, rec: Recognition | None) -> list[str]:
        """Whitelisted concepts of the problem, tagger first, matcher second."""
        if rec is not None and rec.concepts:
            return rec.concepts
        return match.concepts

    def _generic_fallback(self, decision: Decision, slots: dict[str, str]) -> str:
        for template in self.db.hint_templates_for([], None, decision.level):
            if template.concept_id is None and template.misconception_id is None:
                try:
                    return template.template_text.format(**slots)
                except (KeyError, IndexError):
                    continue
        return "지금까지 한 풀이를 처음부터 소리 내어 설명해 볼까요?"
