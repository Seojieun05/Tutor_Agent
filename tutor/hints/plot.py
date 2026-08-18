"""A function, drawn the way a tutor sketches one on the board.

Server-side SVG on purpose. sympy is already here to parse and evaluate the
expression, so the page needs no plotting library, no external CDN and no API
key — the figure arrives as markup the browser only has to display. It is also
the honest place for the leak guard to see a figure before the student does.

Deliberately a SKETCH, not a plot: axes, a light grid, the curve. No marked
roots, no shaded areas, no coordinates called out — a curve with its extremum
labelled is the answer written in ink, and the whole point of the board is to
show the shape while the student finds the numbers.
"""

from __future__ import annotations

import logging
import math
from html import escape

from tutor.knowledge import mathnorm
from tutor.speech import mathspeak

log = logging.getLogger(__name__)

# Drawn at the size it is actually shown at. The board is the width of the
# page's centre column, and a 520-wide sketch stretched to fill it scaled its
# own tick labels and legend up with it until they crowded the curve.
WIDTH, HEIGHT = 900, 400
PAD = 34                      # room for the axis numbers
SAMPLES = 241                 # odd, so x = 0 is sampled exactly
DEFAULT_SPAN = (-6.0, 6.0)
# Past this the curve is a vertical wall, not a shape: clip and break the line
# so an asymptote reads as an asymptote instead of a spike across the frame.
MAX_ABS_Y = 1e4
# How much taller than wide the picture may be in UNITS before the x-axis is
# given more room. The drawing area is about 2.5:1, so this is already a tall
# picture; past it the two axes stop being comparable and the grid reads as a
# mistake — which is how a live sketch ended up counting by 1 across and 50 up.
ASPECT_CAP = 3.0
# What the question is about is drawn like it matters; what only exists to
# get there is drawn like a construction line, so a glance sorts them.
TARGETS = ("var(--accent-deep)", "var(--err)")
SCAFFOLDS = ("var(--dim)", "var(--dim)")


def _sample(expr_text: str, x0: float, x1: float) -> list[tuple[float, float | None]]:
    """(x, y) with y=None where the function has nothing to draw."""
    expr = mathnorm.parse_expression(expr_text)
    free = expr.free_symbols
    if len(free) > 1:
        raise ValueError(f"{expr_text!r} has more than one variable")
    if not free:
        raise ValueError(f"{expr_text!r} is a constant, not a curve")
    import sympy

    fn = sympy.lambdify(next(iter(free)), expr, "math")
    points: list[tuple[float, float | None]] = []
    for i in range(SAMPLES):
        x = x0 + (x1 - x0) * i / (SAMPLES - 1)
        try:
            y = float(fn(x))
        except Exception:  # noqa: BLE001 — a hole in the domain is a gap, not a crash
            y = float("nan")
        points.append((x, None if not math.isfinite(y) else y))
    return points


def _window(series: list[list[tuple[float, float | None]]]) -> tuple[float, float]:
    """A y-range that shows the shape rather than one spike.

    The middle 90% of the sampled values decides the window, so a single
    asymptote cannot flatten the interesting part into a horizontal line.
    """
    ys = sorted(y for s in series for _, y in s if y is not None and abs(y) < MAX_ABS_Y)
    if not ys:
        return -1.0, 1.0
    lo, hi = ys[len(ys) // 20], ys[-1 - len(ys) // 20]
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    margin = (hi - lo) * 0.12
    return lo - margin, hi + margin


def _widen_to_aspect(
    span: tuple[float, float], y0: float, y1: float
) -> tuple[float, float]:
    """Give the x-axis room when the y-range towers over it.

    Two tangent lines over x ∈ (-2, 4) span 30 in y and 6 in x, which the tick
    chooser answers honestly and unreadably: 1 on one axis, 5 on the other,
    and one live scene came out 1 against 50. Widening x is the safe side of
    the fix — nothing already drawn leaves the frame.
    """
    x0, x1 = span
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= ASPECT_CAP * width:
        return span
    want = height / ASPECT_CAP
    middle = (x0 + x1) / 2
    return middle - want / 2, middle + want / 2


def _ticks(lo: float, hi: float, want: int = 8) -> list[float]:
    span = hi - lo
    if span <= 0:
        return []
    raw = span / want
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=mag)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float) -> str:
    return f"{v:g}" if abs(v) >= 1e-4 or v == 0 else f"{v:.1e}"


# Roughly what a legend row occupies. Only used to pick a corner, so it is a
# generous guess rather than a measurement.
LEGEND_W, LEGEND_ROW = 300.0, 14.0


def _legend_svg(rows, series, px, py) -> str:
    """The legend, in whichever corner the curves leave emptiest.

    A fixed corner is wrong for a sketch whose shape changes every problem:
    at the top it is the first thing lost when a full-height figure scrolls,
    and at the bottom two descending tangents ran straight through it. So
    count how many sampled points fall in each corner and take the quietest,
    preferring the bottom, which is where the board scrolls to.
    """
    if not rows:
        return ""
    height = len(rows) * LEGEND_ROW + 10
    corners = [                                   # bottom first: see above
        ("start", PAD, HEIGHT - PAD - height),
        ("end", WIDTH - PAD - LEGEND_W, HEIGHT - PAD - height),
        ("start", PAD, PAD),
        ("end", WIDTH - PAD - LEGEND_W, PAD),
    ]

    def crowding(left: float, top: float) -> int:
        return sum(
            1
            for _, points in series
            for x, y in points
            if y is not None
            and left <= px(x) <= left + LEGEND_W
            and top <= py(y) <= top + height
        )

    anchor, left, top = min(corners, key=lambda c: crowding(c[1], c[2]))
    x = left if anchor == "start" else left + LEGEND_W
    return "".join(
        f'<text x="{x:.0f}" y="{top + 12 + i * LEGEND_ROW:.0f}" text-anchor="{anchor}" '
        f'class="legend" fill="{colour}">{text}</text>'
        for i, (text, colour) in enumerate(rows)
    )


def _sample_all(curves, window: tuple[float, float]):
    out: list[tuple[object, list[tuple[float, float | None]]]] = []
    for curve in list(curves)[:4]:
        text = curve if isinstance(curve, str) else curve.expr
        try:
            out.append((curve, _sample(text, *window)))
        except Exception as e:  # noqa: BLE001 — an unplottable hint still speaks
            log.info("not plotting %r: %s", text, e)
    return out


def compute_view(
    curves, span: tuple[float, float] = DEFAULT_SPAN, zoom: float = 1.0
) -> tuple[float, float, float, float] | None:
    """The window function_svg would draw: (x0, x1, y0, y1), or None.

    Factored out of the drawing so the session can hold onto it: the moment
    the student pans or zooms, this guess becomes THEIR window, and their
    next adjustment has to start from the rectangle actually on screen rather
    than from a recomputation that may have moved under them.

    `zoom` scales the guess: >1 shows more of the plane, <1 moves closer. The
    x-span is scaled before sampling (a wider window has to be sampled to be
    judged) and the y-window after it is chosen, because the y-window is
    derived from the samples and would otherwise re-trim whatever the zoom
    let in.
    """
    zoom = min(4.0, max(0.4, float(zoom or 1.0)))
    if zoom != 1.0:
        mid = (span[0] + span[1]) / 2
        half = (span[1] - span[0]) / 2 * zoom
        span = (mid - half, mid + half)
    series = _sample_all(curves, span)
    if not series:
        return None

    def is_target(curve) -> bool:
        return not isinstance(curve, str) and getattr(curve, "role", "") == "target"

    # The window belongs to what the question is ABOUT. g(x) over x ∈ (-2, 4)
    # swings through 200 while the tangent lines the student is looking for
    # cover 30, and letting the construction line vote flattens both of them
    # into the same horizontal smear. A scaffold running off the top of the
    # frame is what a sketch on a real board looks like.
    aimed = [s for curve, s in series if is_target(curve)]
    y0, y1 = _window(aimed or [s for _, s in series])
    if zoom != 1.0:
        cy, hy = (y0 + y1) / 2, (y1 - y0) / 2 * zoom
        y0, y1 = cy - hy, cy + hy

    widened = _widen_to_aspect(span, y0, y1)
    if widened != span:
        log.info("widening x to %.1f..%.1f for a y-range of %.1f", *widened, y1 - y0)
        span = widened
    return (span[0], span[1], y0, y1)


def function_svg(
    curves, span: tuple[float, float] = DEFAULT_SPAN, zoom: float = 1.0,
    view: tuple[float, float, float, float] | None = None,
    points=(), show_scale: bool = True, show_legend: bool = True,
) -> str | None:
    """Markup for the scene, or None when nothing in it can be drawn.

    `curves` are objects with .expr and optionally .label/.role, or plain
    expression strings. `view` overrides the automatic window entirely — it
    is the student's own rectangle, chosen with the pan and zoom controls,
    and it is drawn verbatim: sampled over its x-range, clipped to its
    y-range, no re-guessing.
    """
    if view is None:
        view = compute_view(curves, span, zoom)
        if view is None:
            return None
    x0, x1, y0, y1 = view
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    series = _sample_all(curves, (x0, x1))
    if not series:
        return None
    px = lambda x: PAD + (x - x0) / (x1 - x0) * (WIDTH - 2 * PAD)      # noqa: E731
    py = lambda y: HEIGHT - PAD - (y - y0) / (y1 - y0) * (HEIGHT - 2 * PAD)  # noqa: E731

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="함수의 그래프" class="fnplot">'
    ]
    if show_scale:
        grid = []
        for t in _ticks(x0, x1):
            grid.append(f'<line x1="{px(t):.1f}" y1="{PAD}" x2="{px(t):.1f}" y2="{HEIGHT - PAD}"/>')
        for t in _ticks(y0, y1):
            grid.append(f'<line x1="{PAD}" y1="{py(t):.1f}" x2="{WIDTH - PAD}" y2="{py(t):.1f}"/>')
        parts.append(f'<g class="grid">{"".join(grid)}</g>')

    axes = []
    if y0 <= 0 <= y1:
        axes.append(f'<line x1="{PAD}" y1="{py(0):.1f}" x2="{WIDTH - PAD}" y2="{py(0):.1f}"/>')
    if x0 <= 0 <= x1:
        axes.append(f'<line x1="{px(0):.1f}" y1="{PAD}" x2="{px(0):.1f}" y2="{HEIGHT - PAD}"/>')
    parts.append(f'<g class="axes">{"".join(axes)}</g>')

    if show_scale:
        labels = []
        baseline = py(0) if y0 <= 0 <= y1 else HEIGHT - PAD
        for t in _ticks(x0, x1):
            if abs(t) > 1e-9:
                labels.append(
                    f'<text x="{px(t):.1f}" y="{baseline + 13:.1f}" text-anchor="middle">{_fmt(t)}</text>'
                )
        left = px(0) if x0 <= 0 <= x1 else PAD
        for t in _ticks(y0, y1):
            if abs(t) > 1e-9:
                labels.append(
                    f'<text x="{left - 5:.1f}" y="{py(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>'
                )
        parts.append(f'<g class="ticks">{"".join(labels)}</g>')

    targets = scaffolds = 0
    legend: list[tuple[str, str]] = []
    for i, (curve, sampled_points) in enumerate(series):
        text = curve if isinstance(curve, str) else curve.expr
        label = "" if isinstance(curve, str) else (curve.label or "")
        target = not isinstance(curve, str) and getattr(curve, "role", "") == "target"
        if target:
            colour = TARGETS[targets % len(TARGETS)]
            width, dash = "2.4", ""
            targets += 1
        else:
            colour = SCAFFOLDS[scaffolds % len(SCAFFOLDS)]
            width, dash = "1.5", ' stroke-dasharray="5 4"'
            scaffolds += 1

        segments, run = [], []
        for x, y in sampled_points:
            if y is None or not (y0 - (y1 - y0) <= y <= y1 + (y1 - y0)):
                if len(run) > 1:
                    segments.append(run)
                run = []
                continue
            run.append(f"{px(x):.1f},{py(y):.1f}")
        if len(run) > 1:
            segments.append(run)
        for run in segments:
            parts.append(
                f'<polyline points="{" ".join(run)}" fill="none" '
                f'stroke="{colour}" stroke-width="{width}"{dash} stroke-linecap="round"/>'
            )
        # The legend is read, so it gets the eye's notation like everything
        # else on screen: x², ·, √ rather than programmer ASCII.
        name = f"{escape(label)}: y = " if label else "y = "
        legend.append((f"{name}{escape(mathspeak.displayable(text))}", colour))

    if show_legend:
        parts.append(_legend_svg(legend, series, px, py))
    marked = []
    for point in points or ():
        try:
            x, y = float(point.x), float(point.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        label = escape(str(getattr(point, "label", ""))[:4])
        marked.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" '
            f'fill="var(--accent-deep)" stroke="var(--board-bg-hi)" stroke-width="2"/>'
        )
        if label:
            marked.append(
                f'<text x="{px(x) + 9:.1f}" y="{py(y) - 7:.1f}" '
                f'class="point-label" fill="var(--text)">{label}</text>'
            )
    if marked:
        parts.append(f'<g class="points">{"".join(marked)}</g>')
    parts.append("</svg>")
    return "".join(parts)
