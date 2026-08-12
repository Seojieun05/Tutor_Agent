"""The tutor's drawing hand, which runs while the tutor is already talking.

A hint takes ~4.6s to write and ~8s to speak. The picture that supports it
does not have to exist when the voice starts — a teacher talks first and draws
while talking — so this runs in the gap the speech opens, and gets something
no parallel design could give it: the finished hint, word for word. It draws
what was actually said instead of guessing what will be.

Because it reads the sentence, it can also decline in the common case. The
trigger is the hint's own language: a sentence that never mentions shape is
not a sentence a curve supports.

It is fed the same diet as the phrasing model — problem, work, diagnosis, and
what is already on the board — and deliberately NOT the reference solution.
The leak guard still screens the result, but a model that never saw the answer
cannot draw it by accident.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from tutor.llm.client import LLMClient

log = logging.getLogger(__name__)

# Words that mean the hint is talking about a SHAPE. The picture exists to
# support a sentence, so the sentence has to be asking for one — this keeps
# the call (and the drawing) off every turn that is pure algebra.
_VISUAL = re.compile(
    r"그래프|개형|곡선|그림|그려|증가|감소|증감|교점|만나|넓이|면적|영역|"
    r"축|대칭|기울기|접선|극값|극대|극소|위로|아래로|볼록|오목|범위|구간"
)


def wants_a_picture(hint: str) -> bool:
    return bool(_VISUAL.search(hint or ""))


class FigureSpec(BaseModel):
    """What to draw, and what to be careful not to show while drawing it."""

    functions: list[str] = []          # one variable each, ASCII, y= stripped
    x_min: float | None = None
    x_max: float | None = None
    caption: str = ""                  # short Korean label, like a board note
    why: str = ""                      # logged, never shown to the student


_SYSTEM = """PERSONA
You are the drawing hand of a Korean math tutor. The tutor has just said a
hint out loud and is still speaking; you decide what appears on the board
beside it.

ACT
Return a sketch that supports THAT SENTENCE — not the problem in general, not
the solution. If the sentence does not need a picture, return an empty
`functions` list, which is the right answer most of the time.

WHAT YOU MAY DRAW
1-2 functions of ONE variable, written in x with ASCII notation
("x**2 - 4*x + 3", "2*x - 3"). Only functions the student can already see:
the problem's own function, a function the hint itself names. Never a
function that only exists in the solution.

THE WINDOW
Choose x_min and x_max so the thing the tutor is TALKING ABOUT is visible and
fills the frame. If the hint is about a sign change near the origin, a window
of [-1, 5] teaches; [-10, 10] hides it. Leave both null only when no range is
better than any other.

CAPTION
Optionally 2-8 Korean words naming what to look at ("부호가 바뀌는 구간",
"두 그래프가 만나는 곳"). A label, not a sentence, and never a value.

NEVER
- Never a number that answers the problem: no coordinates of an intersection,
  no extremum, no area. The sketch shows SHAPE; the numbers stay the
  student's to find.
- Never a function the hint did not talk about.
- When in doubt, return nothing. A wrong picture costs more than no picture.

Return ONLY JSON:
{"functions": ["..."], "x_min": null, "x_max": null, "caption": "...", "why": "..."}"""


class Illustrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def draw(
        self,
        *,
        hint: str,
        problem_text: str,
        equations: list[str],
        student_work: list[str],
        board: list[str],
        misconception: str | None,
        level: int,
    ) -> FigureSpec | None:
        parts = [
            f"튜터가 방금 말한 힌트: {hint}",
            f"문제: {problem_text}",
            f"문제의 식: {equations}",
        ]
        if student_work:
            parts.append(f"학생이 쓴 풀이: {' / '.join(student_work)}")
        if board:
            # so a second hint extends the first picture instead of replacing it
            parts.append(f"이미 칠판에 적힌 것: {' / '.join(board)}")
        if misconception:
            parts.append(f"진단된 오개념: {misconception}")
        parts.append(f"힌트 레벨: L{level}")
        try:
            spec = self.llm.complete_json(
                purpose="illustrate",
                system=_SYSTEM,
                user="\n".join(parts),
                schema=FigureSpec,
            )
        except Exception:  # noqa: BLE001 — a missing picture is not a failed turn
            log.exception("illustrator failed; the hint stands on its own")
            return None
        spec.functions = [
            f.split("=", 1)[1].strip() if "=" in f else f.strip()
            for f in spec.functions
            if f and f.strip()
        ][:2]
        spec.caption = " ".join(spec.caption.split())[:28]
        return spec
