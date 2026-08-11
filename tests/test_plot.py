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

    def test_two_curves_get_two_colours(self):
        svg = function_svg(["x**2", "2*x + 1"])
        strokes = set(re.findall(r'stroke="(var\(--[a-z-]+\))"', svg))
        assert len(strokes) == 2

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
