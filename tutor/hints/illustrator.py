"""The tutor's drawing hand, which works while the tutor is already talking.

A hint takes ~4.6s to write and ~8s to speak. The picture that supports it
does not have to exist when the voice starts — a teacher talks first and draws
while talking — so this runs in the gap the speech opens, and gets something
no parallel design could give it: the finished hint, word for word, and what
the student just said.

It draws a SCENE, not a picture. One grid per problem, redeclared every turn:
the model is shown what is currently on it and returns what should be on it
now. That framing is what lets a board behave like a real one — the parabola
that was needed to find a tangent is wiped once the tangent is there, because
the model simply stops listing it. Nothing here diffs or animates; the scene
is a statement of the present, and the page replaces the canvas in place.

Which means the model has to know what the problem is FOR. A curve that only
exists to derive something else is scaffolding; the thing the question asks
about is the target. That distinction is the whole judgement, and it is the
model's, because only it can read the question.

It is fed the same diet as the phrasing model — problem, work, diagnosis, and
what is already drawn and written — and deliberately NOT the reference
solution. The leak guard still screens the result, but a model that never saw
the answer cannot draw it by accident.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from tutor.llm.client import LLMClient

log = logging.getLogger(__name__)

# Words that mean the hint is talking about a SHAPE. Only used to OPEN a
# scene: once a grid exists it stays up to date every turn, because the thing
# that changes it is often the student's answer rather than the tutor's words.
_VISUAL = re.compile(
    r"그래프|개형|곡선|그림|그려|접선|기울기|증가|감소|증감|교점|만나|넓이|면적|"
    r"영역|축|대칭|극값|극대|극소|위로|아래로|볼록|오목|범위|구간|둘러싸"
)


def wants_a_picture(hint: str) -> bool:
    return bool(_VISUAL.search(hint or ""))


class Curve(BaseModel):
    """One line on the grid, and why it is still there."""

    expr: str                                    # one variable, ASCII, in x
    label: str = ""                              # what the problem calls it: l, m, f, g
    role: Literal["target", "scaffold"] = "scaffold"


class FigureSpec(BaseModel):
    """The whole grid as it should look after this turn."""

    curves: list[Curve] = []
    x_min: float | None = None
    x_max: float | None = None
    caption: str = ""                            # short Korean label, like a board note
    why: str = ""                                # logged, never shown to the student


_SYSTEM = """PERSONA
You are the drawing hand of a Korean math tutor. The tutor has just spoken and
is still speaking; you decide what the board looks like NOW.

ACT
Return the WHOLE scene — every curve that should be on the grid after this
moment, not the change since last time. You are shown what is currently drawn;
repeat a curve to keep it, omit it to wipe it. There is ONE grid per problem
and it is redrawn in place, so curves accumulate on the same axes instead of
piling up as separate pictures.

TARGET vs SCAFFOLD — this is the judgement that matters
Read what the QUESTION actually asks for. Curves that exist only to derive
something else are `scaffold`; the objects the question is about are `target`.

  "곡선 y=f(x) 위의 점에서의 접선 l, ... 두 직선 l, m과 y축으로 둘러싸인
   도형의 넓이는?"  →  l and m are targets. f and g are scaffolds: they exist
   to produce l and m, and the area the question asks for does not involve
   them at all.

Wipe a scaffold as soon as the thing it produced is on the grid. Once the
tangent l is drawn, f has done its job and only crowds the picture — omit it.
Keep a scaffold while it is still being used, and never wipe a target.

WHAT YOU MAY DRAW
Up to 4 functions of ONE variable, ASCII in x ("x**2 - 4*x - 3", "-2*x - 4").
Only what the student can already see or has already established: the
problem's own functions, and a line THEY have just found correctly. Never a
function that only exists in the solution they have not reached.

LABEL each curve with the name the problem uses (l, m, f, g) when it has one.

THE WINDOW
Choose x_min and x_max so what the tutor is talking about fills the frame, and
keep the previous window when the scene grows — a grid that jumps every turn
is a new picture, not the same board. Widen only when a new curve needs room.

CAPTION
Optionally 2-8 Korean words naming what to look at ("두 접선이 만드는 영역").
A label, not a sentence, and never a value.

NEVER
- Never a number that answers the problem: no intersection coordinates, no
  extremum, no area.
- Never a curve the conversation has not reached.
- When in doubt, return the scene unchanged. A wrong picture costs more than
  a still one.

Return ONLY JSON:
{"curves": [{"expr": "...", "label": "...", "role": "target|scaffold"}],
 "x_min": null, "x_max": null, "caption": "...", "why": "..."}"""


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
        scene: list[Curve] | None = None,
        span: tuple[float, float] | None = None,
        student_said: str | None = None,
    ) -> FigureSpec | None:
        parts = [f"튜터가 방금 말한 힌트: {hint}"]
        if student_said:
            # what the student just established is usually WHY the scene changes
            parts.append(f"학생이 방금 맞게 말한 것: {student_said}")
        parts += [f"문제: {problem_text}", f"문제의 식: {equations}"]
        if student_work:
            parts.append(f"학생이 쓴 풀이: {' / '.join(student_work)}")
        if board:
            parts.append(f"칠판에 적힌 식: {' / '.join(board)}")
        if scene:
            drawn = " / ".join(
                f"{c.label or '이름없음'}: {c.expr} ({c.role})" for c in scene
            )
            parts.append(f"지금 그려져 있는 것: {drawn}")
            if span:
                parts.append(f"지금 창: [{span[0]}, {span[1]}]")
        else:
            parts.append("지금 그려져 있는 것: 없음 (빈 격자)")
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
        kept = []
        for c in spec.curves:
            expr = (c.expr or "").strip()
            if "=" in expr:                       # "y = 2*x - 3" and "l: ..." alike
                expr = expr.split("=", 1)[1].strip()
            if expr:
                kept.append(Curve(expr=expr, label=(c.label or "").strip()[:4],
                                  role=c.role))
        spec.curves = kept[:4]
        spec.caption = " ".join(spec.caption.split())[:28]
        return spec
