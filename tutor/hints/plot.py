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

WIDTH, HEIGHT = 520, 300
PAD = 28                      # room for the axis numbers
SAMPLES = 241                 # odd, so x = 0 is sampled exactly
DEFAULT_SPAN = (-6.0, 6.0)
# Past this the curve is a vertical wall, not a shape: clip and break the line
# so an asymptote reads as an asymptote instead of a spike across the frame.
MAX_ABS_Y = 1e4
CURVES = ("var(--accent-deep)", "var(--err)")


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


def function_svg(
    expressions: list[str], span: tuple[float, float] = DEFAULT_SPAN
) -> str | None:
    """Markup for one or two curves, or None when nothing can be drawn."""
    series: list[tuple[str, list[tuple[float, float | None]]]] = []
    for text in expressions[:2]:
        try:
            series.append((text, _sample(text, *span)))
        except Exception as e:  # noqa: BLE001 — an unplottable hint still speaks
            log.info("not plotting %r: %s", text, e)
    if not series:
        return None

    x0, x1 = span
    y0, y1 = _window([s for _, s in series])
    px = lambda x: PAD + (x - x0) / (x1 - x0) * (WIDTH - 2 * PAD)      # noqa: E731
    py = lambda y: HEIGHT - PAD - (y - y0) / (y1 - y0) * (HEIGHT - 2 * PAD)  # noqa: E731

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="함수의 그래프" class="fnplot">'
    ]
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

    for i, (text, points) in enumerate(series):
        segments, run = [], []
        for x, y in points:
            if y is None or not (y0 - (y1 - y0) <= y <= y1 + (y1 - y0)):
                if len(run) > 1:
                    segments.append(run)
                run = []
                continue
            run.append(f"{px(x):.1f},{py(y):.1f}")
        if len(run) > 1:
            segments.append(run)
        colour = CURVES[i % len(CURVES)]
        for run in segments:
            parts.append(
                f'<polyline points="{" ".join(run)}" fill="none" '
                f'stroke="{colour}" stroke-width="2.2" stroke-linecap="round"/>'
            )
        # the legend is read, so it gets the eye's notation like everything
        # else on screen: x², ·, √ rather than programmer ASCII
        parts.append(
            f'<text x="{WIDTH - PAD}" y="{PAD - 10 + i * 15:.0f}" text-anchor="end" '
            f'class="legend" fill="{colour}">y = {escape(mathspeak.displayable(text))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
