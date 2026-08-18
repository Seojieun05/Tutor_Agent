"""The tutor's sketch of a function.

Server-side SVG: sympy already parses and evaluates here, so the page needs no
plotting library. A sketch, not a plot — axes, a grid, the curve, and never a
marked root or a shaded area, because a labelled extremum is the answer in ink.
"""

import re

from tutor.hints.plot import function_svg


def numbers_in(svg: str, tag: str) -> list[float]:
    body = re.search(rf'<g class="{tag}">(.*?)</g>', svg, re.S)
    return [float(v) for v in re.findall(r'="(-?\d+\.?\d*)"', body.group(1))] if body else []


class TestItDraws:
    def test_a_parabola_becomes_a_curve(self):
        svg = function_svg(["x**2 - 4*x + 3"])
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert 'class="grid"' in svg and 'class="axes"' in svg
        points = re.findall(r'<polyline points="([^"]+)"', svg)
        assert points, "no curve drawn"
        assert len(points[0].split()) > 50          # sampled, not a straight line

    def test_a_target_and_a_scaffold_are_told_apart_at_a_glance(self):
        """What the question asks for is drawn like it matters; what only
        exists to get there is drawn like a construction line."""
        from tutor.hints.illustrator import Curve

        svg = function_svg([
            Curve(expr="x**2 - 4*x - 3", label="f", role="scaffold"),
            Curve(expr="-2*x - 4", label="l", role="target"),
        ])
        assert "stroke-dasharray" in svg              # the scaffold is dashed
        assert svg.count("stroke-dasharray") < svg.count("<polyline")
        assert "f: y = " in svg and "l: y = " in svg  # named as the problem names them
        strokes = set(re.findall(r'stroke="(var\(--[a-z-]+\))"', svg))
        assert len(strokes) == 2

    def test_two_bare_expressions_still_draw(self):
        svg = function_svg(["x**2", "2*x + 1"])
        assert len(re.findall(r"<polyline", svg)) >= 2

    def test_the_legend_names_the_function_in_print_notation(self):
        assert "y = x²" in function_svg(["x**2"])          # not "x**2"
        assert "y = 2·x + 1" in function_svg(["2*x + 1"])

    def test_caret_and_implicit_multiplication_parse(self):
        assert function_svg(["3x^2 - 1"]) is not None

    def test_a_pole_breaks_the_line_instead_of_spiking(self):
        """1/x across zero is two strokes, not one wall through the frame."""
        svg = function_svg(["1/x"])
        assert len(re.findall(r"<polyline", svg)) >= 2


class TestItRefuses:
    def test_two_variables_are_not_a_curve(self):
        assert function_svg(["x + y"]) is None

    def test_a_constant_is_not_a_curve(self):
        assert function_svg(["5"]) is None

    def test_nonsense_does_not_raise(self):
        assert function_svg(["!!!"]) is None

    def test_one_bad_expression_does_not_lose_the_good_one(self):
        svg = function_svg(["x + y", "x**2"])
        assert svg is not None and "y = x²" in svg
        assert len(re.findall(r"<polyline", svg)) >= 1

    def test_at_most_two_curves(self):
        svg = function_svg(["x", "x**2", "x**3"])
        assert "y = x**3" not in svg


class TestItStaysASketch:
    def test_nothing_is_marked_on_the_curve(self):
        """No plotted points, no shading: the shape is the hint, the numbers
        are still the student's to find."""
        svg = function_svg(["x**2 - 4*x + 3"])
        assert "<circle" not in svg and "<path" not in svg
        assert "fill=\"none\"" in svg

    def test_the_window_survives_an_asymptote(self):
        """A single spike must not flatten the interesting part: the y window
        comes from the middle of the samples, not from the extremes."""
        svg = function_svg(["1/(x - 0.01)"])
        ticks = re.findall(r'<text[^>]*>(-?\d+\.?\d*)</text>', svg)
        assert ticks and max(abs(float(t)) for t in ticks) < 1000

    def test_named_construction_points_can_be_shown_without_a_scale(self):
        from tutor.hints.illustrator import Curve, PlotPoint

        svg = function_svg(
            [Curve(expr="2**x - 2")],
            span=(-1, 3),
            points=[
                PlotPoint(x=2, y=2, label="A"),
                PlotPoint(x=2, y=0, label="B"),
                PlotPoint(x=2, y=-2, label="C"),
            ],
            show_scale=False,
            show_legend=False,
        )

        assert svg.count("<circle") == 3
        assert all(f">{label}</text>" in svg for label in "ABC")
        assert 'class="axes"' in svg
        assert 'class="grid"' not in svg and 'class="ticks"' not in svg
        assert 'class="legend"' not in svg


def tick_step(svg: str) -> tuple[float, float]:
    """(x step, y step) as drawn. The x labels sit under the axis, the y
    labels to its left, so their anchors tell them apart."""
    xs = sorted(float(v) for v in re.findall(
        r'<text x="[\d.]+" y="[\d.]+" text-anchor="middle">(-?[\d.]+)</text>', svg))
    ys = sorted(float(v) for v in re.findall(
        r'<text x="[\d.-]+" y="[\d.]+" text-anchor="end">(-?[\d.]+)</text>', svg))
    return xs[1] - xs[0], ys[1] - ys[0]


class TestTheAxesStayComparable:
    """Live, on problem 13: x counted by 1 and y by 50, because g(x) swings
    through 200 over the six units the tangent lines live in."""

    SCENE = None  # built per-test; Curve is imported lazily like the other tests

    def scene(self, *specs):
        from tutor.hints.illustrator import Curve

        return [Curve(expr=e, label=n, role=r) for e, n, r in specs]

    def test_a_scaffold_does_not_set_the_window(self):
        """The window belongs to what the question is about. A construction
        line running off the top of the frame is what a sketch looks like."""
        svg = function_svg(
            self.scene(
                ("-2*x - 4", "l", "target"),
                ("(x**3 - 2*x)*(x**2 - 4*x - 3)", "g", "scaffold"),
            ),
            (-2.0, 4.0),
        )
        x_step, y_step = tick_step(svg)
        assert y_step <= 4 * x_step, f"y counts by {y_step} where x counts by {x_step}"

    def test_a_tall_scene_widens_x_instead_of_squashing_y(self):
        """Two tangents cover 30 in y over 6 in x. Widening is the safe side:
        nothing already drawn leaves the frame."""
        scene = self.scene(
            ("-2*x - 4", "l", "target"),
            ("-4*x + 10", "m", "target"),
        )
        svg = function_svg(scene, (-2.0, 4.0))
        x_step, y_step = tick_step(svg)
        assert y_step <= 4 * x_step
        xs = [float(v) for v in re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="middle">(-?[\d.]+)</text>', svg)]
        assert min(xs) < -2 and max(xs) > 4, "the x window was not widened"

    def test_a_scene_that_is_already_square_is_left_alone(self):
        svg = function_svg(self.scene(("x", "y=x", "target")), (-4.0, 4.0))
        xs = [float(v) for v in re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="middle">(-?[\d.]+)</text>', svg)]
        assert max(xs) <= 4 and min(xs) >= -4

    def test_scaffolds_alone_still_get_a_window(self):
        """No target in the scene: everything votes, as before."""
        svg = function_svg(self.scene(("x**2 - 4*x - 3", "f", "scaffold")), (-2.0, 6.0))
        assert "<polyline" in svg


class TestTheStudentsOwnWindow:
    """The automatic window is a guess — the middle 90% of the target's
    values — and a curve whose interesting part it cut off looked simply
    broken, with no way to say "wider". `zoom` is that way: it scales the
    window itself, so what was clipped is actually re-sampled and drawn."""

    def x_extent(self, svg: str) -> float:
        xs = [float(v) for v in re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="middle">(-?[\d.]+)</text>', svg)]
        return max(xs) - min(xs)

    def y_extent(self, svg: str) -> float:
        ys = [float(v) for v in re.findall(
            r'<text x="[\d.-]+" y="[\d.]+" text-anchor="end">(-?[\d.]+)</text>', svg)]
        return max(ys) - min(ys)

    def test_zooming_out_shows_more_of_both_axes(self):
        near = function_svg(["x**3 - 3*x**2 + 2"], span=(-3, 3))
        wide = function_svg(["x**3 - 3*x**2 + 2"], span=(-3, 3), zoom=2.0)
        assert self.x_extent(wide) > self.x_extent(near)
        assert self.y_extent(wide) > self.y_extent(near)

    def test_zooming_in_moves_closer(self):
        near = function_svg(["x**3 - 3*x**2 + 2"], span=(-3, 3), zoom=0.5)
        base = function_svg(["x**3 - 3*x**2 + 2"], span=(-3, 3))
        assert self.x_extent(near) < self.x_extent(base)

    def test_the_wider_window_is_really_sampled(self):
        # the extra plane must carry curve, not blank margin: the widened span
        # is re-sampled, so the polyline reaches x values the near window
        # never contained
        wide = function_svg(["x**2"], span=(-2, 2), zoom=2.0)
        xs = [float(v) for v in re.findall(
            r'<text x="[\d.]+" y="[\d.]+" text-anchor="middle">(-?[\d.]+)</text>', wide)]
        assert max(xs) >= 3          # the (-2,2) span could never label x=3

    def test_zoom_is_clamped_not_trusted(self):
        # a runaway zoom must not sample the plane to death: 1000 is 4
        assert function_svg(["x**2"], span=(-2, 2), zoom=1000) ==                function_svg(["x**2"], span=(-2, 2), zoom=4.0)
