"""The tutor's drawing hand, which works while the tutor is already talking.

A hint takes ~4.6s to write and ~8s to speak. The picture that supports it
does not have to exist when the voice starts — a teacher talks first and draws
while talking — so this runs in the gap the speech opens, and gets something
no parallel design could give it: the finished hint, word for word, and what
the student just said.

It draws a SCENE, not a picture. One grid per problem, redeclared every turn:
the model is shown what is currently on it and returns what should be on it
now. A source curve stays beside the tangent just earned from it; it is wiped
when the lesson turns to a different source curve, or when that second tangent
is complete. Nothing here diffs or animates; the scene is a statement of the
present, and the page replaces the canvas in place.

Which means the model has to know what the problem is FOR. A curve that only
exists to derive something else is scaffolding; the thing the question asks
about is the target. That distinction is the whole judgement, and it is the
model's, because only it can read the question.

It is fed the same diet as the phrasing model — problem, work, diagnosis, and
what is already drawn and written — and deliberately NOT the reference
solution, with one carve-out: the VERIFIED PREFIX, the steps the orchestrator
has already graded as done. A tangent the student derived out loud is theirs
now, and a drawing hand that only believes the photograph refuses to draw the
very line the lesson just earned. The leak guard still screens the result,
and everything past the student's frontier stays unseen.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Literal

from pydantic import BaseModel

from tutor.knowledge import mathnorm
from tutor.llm.client import LLMClient

log = logging.getLogger(__name__)

# Words that mean the hint is talking about a SHAPE. Only used to OPEN a
# scene: once a grid exists it stays up to date every turn, because the thing
# that changes it is often the student's answer rather than the tutor's words.
# 힌트 문장이 그림을 부르는 말(그래프·접선·좌표 …)을 담고 있는지 보는 정규식.
_VISUAL = re.compile(
    r"그래프|개형|곡선|그림|그려|접선|기울기|증가|감소|증감|교점|만나|넓이|면적|"
    r"영역|축|대칭|극값|극대|극소|위로|아래로|볼록|오목|범위|구간|둘러싸"
)


# 이 힌트가 그림을 필요로 하는 말투인지.
def wants_a_picture(hint: str) -> bool:
    return bool(_VISUAL.search(hint or ""))


# A verified step whose expression IS a curve — y = …, f(x) = …, f'(x) = … —
# is a reason to open a scene even when the current sentence is pure algebra:
# the student just earned a line the board can show. Sequence and scalar
# steps (a_1*r³ = 2, g'(1) = -4) are not.
# y = … 또는 f(x) = … 처럼 그릴 수 있는 꼴인지.
_CURVE_LIKE = re.compile(r"\by\s*=|[A-Za-z]['′]?\s*\(\s*x\s*\)\s*=")


# 이 식을 곡선으로 그릴 수 있는지.
def drawable(expression: str) -> bool:
    return bool(_CURVE_LIKE.search(expression or ""))


# "l: y = -2*(x - 1) - 6 = -2*x - 4" — a verified, LABELLED line. The last
# piece of the chain is the finished form the board draws.
# "l: y = …" 처럼 이름 붙은 직선.
_LABELLED_LINE = re.compile(r"^\s*([A-Za-z])\s*:\s*y\s*=\s*(.+)$")


# 검증된 이름 붙은 직선(l, m)은 모델 판단과 무관하게 반드시 칠판에 올린다.
def ensure_verified_targets(
    spec: FigureSpec | None, verified: list[str] | None
) -> FigureSpec | None:
    """A verified labelled line (l:, m:) is on the board, full stop.

    The model decides the REST of the scene; whether an earned target line
    appears stopped being its call the second time l went undrawn. Anything
    in `verified` has been graded as the student's own result, so drawing it
    can reveal nothing — and the leak screen downstream still runs.
    """
    obligates = []
    for item in verified or []:
        m = _LABELLED_LINE.match(item.strip())
        if m is None:
            continue
        expr = m.group(2).split("=")[-1].strip()
        if expr:
            obligates.append(Curve(expr=expr, label=m.group(1), role="target"))
    if not obligates:
        return spec
    if spec is None:
        spec = FigureSpec()
    # If the model mentioned an earned line but mislabeled it as scaffolding
    # (or rewrote its expression), the verified version wins. Otherwise the
    # target filter below could silently drop l until another target appears.
    for required in obligates:
        for index, existing in enumerate(spec.curves):
            same_expr = existing.expr.replace(" ", "") == required.expr.replace(" ", "")
            same_label = bool(required.label and existing.label == required.label)
            if same_expr or same_label:
                spec.curves[index] = required
                break
    have = {c.expr.replace(" ", "") for c in spec.curves}
    labels = {c.label for c in spec.curves if c.label}
    missing = [
        c for c in obligates
        if c.expr.replace(" ", "") not in have and c.label not in labels
    ]
    # earned targets survive the 4-curve cap ahead of scenery
    spec.curves = (missing + spec.curves)[:4]
    return spec


# 격자 위의 곡선 하나: 식 · 이름 · 역할(문제가 묻는 대상 target / 유도용 scaffold).
class Curve(BaseModel):
    """One line on the grid, and why it is still there."""

    expr: str                                    # one variable, ASCII, in x
    label: str = ""                              # what the problem calls it: l, m, f, g
    role: Literal["target", "scaffold"] = "scaffold"


# 이름 붙은 점 하나. 좌표값은 화면에 절대 찍지 않는다.
class PlotPoint(BaseModel):
    """A named construction point; its coordinates are never printed."""

    x: float
    y: float
    label: str


# 이번 턴이 끝난 뒤 칠판이 어떤 모습이어야 하는지 — 장면 전체의 선언.
class FigureSpec(BaseModel):
    """The whole grid as it should look after this turn."""

    curves: list[Curve] = []
    points: list[PlotPoint] = []
    x_min: float | None = None
    x_max: float | None = None
    show_scale: bool = True
    show_legend: bool = True
    caption: str = ""                            # short Korean label, like a board note
    why: str = ""                                # logged, never shown to the student


# 도함수 정의 f'(x) = … 를 잡는 정규식.
_DERIVATIVE_DEFINITION = re.compile(
    r"(?:^|,)\s*([A-Za-z])\s*['′]\s*\(\s*x\s*\)\s*="
)
# 함수 정의 f(x) = … 를 잡는 정규식.
_FUNCTION_DEFINITION = re.compile(
    r"^\s*([A-Za-z])\s*\(\s*x\s*\)\s*=\s*(.+)$"
)

# y = … 형태.
_Y_DEFINITION = re.compile(r"^\s*y\s*=\s*(.+)$", re.IGNORECASE)


# 아직 정해지지 않은 상수를 임의의 값으로 대신 그린 상황인지(그럴 땐 눈금·범례를 숨긴다).
def _has_unresolved_plot_parameter(
    curves: list[Curve], equations: list[str], verified: list[str] | None,
) -> bool:
    """Whether a plotted shape replaced an undetermined symbol by a sample.

    This is deliberately structural, not problem-specific.  A symbolic family
    such as ``y=b**x-3`` cannot be sampled until the drawing hand chooses a
    legal representative for ``b``.  That internal choice must never look like
    a value the student derived, so its numeric scale and legend are hidden.
    Fully specified functions (including the f/g curves in a tangent problem)
    have no unresolved symbol and keep their normal scale.
    """
    definitions = {
        match.group(1): match.group(2).strip()
        for equation in equations
        if (match := _FUNCTION_DEFINITION.match(equation or ""))
    }
    y_sources = [
        match.group(1).strip()
        for equation in equations
        if (match := _Y_DEFINITION.match(equation or ""))
    ]
    known_text = " ".join([*(equations or []), *(verified or [])])

    for curve in curves:
        source = _expanded_problem_function(curve.label, equations) if curve.label else None
        if source is None and y_sources:
            source = y_sources[0]
        if not source:
            continue
        try:
            parsed = mathnorm.parse_expression(source)
        except Exception:
            continue
        parameters = {
            str(symbol) for symbol in parsed.free_symbols
            if str(symbol) != "x" and str(symbol) not in definitions
        }
        unresolved = {
            name for name in parameters
            if not re.search(rf"(?<![A-Za-z0-9_']){re.escape(name)}\s*=", known_text)
        }
        if unresolved:
            return True
    return False


# g(x)=…f(x) 처럼 다른 정의를 참조하는 함수를 실제로 그릴 수 있는 식으로 펼친다.
def _expanded_problem_function(name: str, equations: list[str]) -> str | None:
    """Return a plottable RHS, expanding definitions such as g(x)=...f(x)."""
    definitions = {}
    for equation in equations:
        match = _FUNCTION_DEFINITION.match(equation or "")
        if match:
            definitions[match.group(1)] = match.group(2).strip()
    rhs = definitions.get(name)
    if not rhs:
        return None
    for _ in range(len(definitions)):
        changed = False
        for dependency, value in definitions.items():
            if dependency == name:
                continue
            pattern = rf"\b{re.escape(dependency)}\s*\(\s*x\s*\)"
            replaced = re.sub(pattern, f"({value})", rhs)
            changed = changed or replaced != rhs
            rhs = replaced
        if not changed:
            break
    return rhs


# 검증된 단계들만으로 장면을 결정론적으로 구성한다(모델을 기다리지 않는 부분).
def ensure_verified_scene(
    spec: FigureSpec | None,
    verified: list[str] | None,
    equations: list[str] | None,
    focus_step: str = "",
) -> FigureSpec | None:
    """Deterministically stage a tangent lesson from its verified prefix.

    The latest completed derivative definition chooses the dotted source
    curve. The first tangent is shown BESIDE that curve, so the student sees
    what they just derived. When the next derivative becomes the focus, its
    source curve replaces the old scaffold. Completing the second tangent
    retires the scaffold and leaves the two target lines. This yields
    f → f+l → l+g → l+m without asking a model to infer lesson state.
    """
    out = ensure_verified_targets(spec, verified)
    if out is None:
        out = FigureSpec()

    latest_derivative: tuple[int, str] | None = None
    latest_target = -1
    for index, expression in enumerate(verified or []):
        derivative = _DERIVATIVE_DEFINITION.search(expression or "")
        if derivative:
            latest_derivative = (index, derivative.group(1))
        if _LABELLED_LINE.match((expression or "").strip()):
            latest_target = index

    focused_derivative = re.search(
        r"([A-Za-z])\s*['′]\s*\(\s*x\s*\)", focus_step or ""
    )

    # No tangent-lesson signal: preserve the illustrator's proposed scene.
    if latest_derivative is None and latest_target < 0 and focused_derivative is None:
        return out if out.curves else spec

    target_curves = [curve for curve in out.curves if curve.role == "target"]
    target_order = []
    for expression in verified or []:
        labelled = _LABELLED_LINE.match((expression or "").strip())
        if labelled and labelled.group(1) not in target_order:
            target_order.append(labelled.group(1))
    by_label = {curve.label: curve for curve in target_curves if curve.label}
    targets = [by_label[label] for label in target_order if label in by_label]
    targets.extend(curve for curve in target_curves if curve not in targets)
    function_name = None
    if len(target_order) >= 2:
        # The pair of target tangents is now the object the problem uses. The
        # second source curve has finished its job, but neither target leaves.
        function_name = None
    elif focused_derivative is not None:
        # The next task is to work with g'(x): g is printed in the problem and
        # may be shown before its derivative has been completed. This is how
        # the board changes from f to g as the lesson changes tangents.
        function_name = focused_derivative.group(1)
    elif latest_derivative is not None:
        # Keep f after l is earned; seeing the curve and its tangent together
        # is the closure of that subproblem. Once g'(x) is the newest completed
        # derivative this naturally chooses g instead. The two-target branch
        # above removes g as soon as m is earned.
        function_name = latest_derivative[1]
    if function_name is not None:
        rhs = _expanded_problem_function(function_name, equations or [])
        if rhs:
            targets.append(Curve(
                expr=rhs, label=function_name, role="scaffold"
            ))
    out.curves = targets[:4]
    return out


# [프롬프트] 그리기 모델용 시스템 프롬프트.
# 핵심 판단은 target(문제가 묻는 대상) vs scaffold(유도용 보조) 구분이고,
# 학생이 아직 도달하지 않은 결과는 절대 그리지 말라는 것이 가장 강한 금지다.
# 장면 전체를 매 턴 다시 선언하게 해서, 칠판 하나가 문제 하나를 계속 따라간다.
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

Keep a source curve when its first tangent has just been earned, so the student
sees them together. When the lesson turns from f to finding the tangent of g,
replace f with g but keep l. Once the tangent m is earned, remove g and keep
both target lines l and m. Never wipe an earned target.

WHAT YOU MAY DRAW
Up to 4 functions of ONE variable, ASCII in x ("x**2 - 4*x - 3", "-2*x - 4").
Only what the student can already see or has already established: the
problem's own functions, and a line THEY have just found correctly. Never a
function that only exists in the solution they have not reached.

"Established" means WRITTEN — in their work or on the tutor's board, quoted
to you above — or VERIFIED: lines listed under 튜터가 검증한 식 are steps the
student has already completed and the tutor has machine-checked, including
answers said out loud that no photograph of the page can show. Those are
theirs now; draw the ones the scene needs. You are able to derive a finished
result yourself; doing so and drawing it is the worst failure this job has.
A tangent line the student is still deriving, drawn finished on the board,
IS the answer, and "its equation is complete" is a judgement you are never
allowed to make from your own working — only from theirs or the verified list.

LABEL each curve with the name the problem uses (l, m, f, g) when it has one.

AN UNKNOWN PARAMETER IS NOT AN ANSWER
Sometimes the problem gives a FAMILY of curves but has not determined its
parameter yet, for example b>1 and y=b**x-3. To sketch its qualitative shape,
choose any legal representative internally and put that numeric expression in
`expr` so it can be rendered. For an undetermined exponential base constrained
to be greater than 1, prefer 2 as the conventional representative unless the
given conditions exclude it. But the student has NOT found that parameter:
- set `show_scale` to false and `show_legend` to false;
- do not state the representative value in the caption or anywhere else;
- use the picture only for shape and relative placement.
Once the student has actually established the parameter in the verified work,
draw the real curve and the normal scale may return. Do NOT hide the scale for
a fully specified graph. In particular, ordinary polynomial curves and tangent
lines whose formulas are already given or verified keep `show_scale: true`.

NAMED CONSTRUCTION POINTS
You may place named points already defined in the problem, such as A, B and C.
Their x/y fields are private drawing coordinates: NEVER print those coordinates
or use them in the caption. Add at most ONE new named point per turn, in the
order the statement defines them, and repeat every current point to keep it.
Place them consistently with the stated geometry. A point whose location is an
unreached result may be positioned only as an unscaled qualitative sketch, not
as a measurable coordinate.

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
 "points": [{"x": 0, "y": 0, "label": "A"}],
 "x_min": null, "x_max": null, "show_scale": true, "show_legend": true,
 "caption": "...", "why": "..."}"""


# 그리는 손. 튜터가 말하는 동안 돌기 때문에 턴을 지연시키지 않는다.
class Illustrator:
    # 그리기 모델을 받는다.
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # 완성된 힌트와 지금 칠판 상태를 보여 주고, 이번 턴의 장면(FigureSpec)을 받아 온다.
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
        points: list[PlotPoint] | None = None,
        span: tuple[float, float] | None = None,
        show_scale: bool = True,
        show_legend: bool = True,
        student_said: str | None = None,
        verified: list[str] | None = None,
    ) -> FigureSpec | None:
        parts = [f"튜터가 방금 말한 힌트: {hint}"]
        if student_said:
            # what the student just established is usually WHY the scene changes
            parts.append(f"학생이 방금 맞게 말한 것: {student_said}")
        parts += [f"문제: {problem_text}", f"문제의 식: {equations}"]
        if student_work:
            parts.append(f"학생이 쓴 풀이: {' / '.join(student_work)}")
        if verified:
            # the orchestrator's word, not the model's guess: these steps are
            # DONE — graded from the page or from a spoken, sympy-checked
            # claim. This is what lets l be drawn after it was derived aloud.
            parts.append(
                "튜터가 검증한 식 (학생이 이미 끝낸 단계 — established로 취급): "
                + " / ".join(verified)
            )
        if board:
            parts.append(f"칠판에 적힌 식: {' / '.join(board)}")
        if scene or points:
            drawn = " / ".join(
                f"{c.label or '이름없음'}: {c.expr} ({c.role})" for c in (scene or [])
            )
            parts.append(f"지금 그려져 있는 것: {drawn}")
            if points:
                parts.append(
                    "지금 표시된 점 (좌표는 말하지 말고 모두 유지): "
                    + ", ".join(point.label for point in points)
                )
            parts.append(
                f"현재 눈금: {'표시' if show_scale else '숨김'}, "
                f"현재 범례: {'표시' if show_legend else '숨김'}"
            )
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
        allowed = set(re.findall(
            r"(?<![A-Za-z])([A-Z])(?![A-Za-z])",
            " ".join([problem_text, *(verified or [])]),
        ))
        clean_points = []
        for point in spec.points[:8]:
            label = (point.label or "").strip().upper()[:1]
            if not label or label not in allowed:
                continue
            try:
                x, y = float(point.x), float(point.y)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                clean_points.append(PlotPoint(x=x, y=y, label=label))
        spec.points = clean_points
        spec.caption = " ".join(spec.caption.split())[:28]
        if _has_unresolved_plot_parameter(spec.curves, equations, verified):
            # The prompt asks for this; the code makes the safety property
            # unconditional even if a drawing model forgets one of the flags.
            spec.show_scale = False
            spec.show_legend = False
            spec.caption = ""
        return spec
