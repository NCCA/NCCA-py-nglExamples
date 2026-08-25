"""Headless unit tests for the pure-maths easing functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from easing import EASING_FUNCTIONS, ease_out_bounce, get_source


@pytest.mark.parametrize("name", EASING_FUNCTIONS.keys())
def test_endpoints(name):
    """Every easing function must map 0 -> 0 and 1 -> 1."""
    ease = EASING_FUNCTIONS[name]
    assert ease(0.0) == pytest.approx(0.0, abs=1e-9)
    assert ease(1.0) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "name",
    [n for n in EASING_FUNCTIONS if n.startswith("InOut")],
)
def test_in_out_symmetry(name):
    """InOut variants are point-symmetric about (0.5, 0.5)."""
    ease = EASING_FUNCTIONS[name]
    for t in [0.1, 0.25, 0.4]:
        assert ease(t) + ease(1.0 - t) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "name",
    [n for n in EASING_FUNCTIONS if not any(k in n for k in ("Back", "Elastic"))],
)
def test_range_without_overshoot(name):
    """All families except back/elastic stay within [0, 1]."""
    ease = EASING_FUNCTIONS[name]
    for i in range(101):
        v = ease(i / 100.0)
        assert -1e-9 <= v <= 1.0 + 1e-9


def test_overshoot_families_do_overshoot():
    """Back and elastic are supposed to leave [0, 1] mid-curve."""
    for name in ["In Back", "Out Back", "In Elastic", "Out Elastic"]:
        ease = EASING_FUNCTIONS[name]
        values = [ease(i / 200.0) for i in range(201)]
        assert min(values) < -1e-3 or max(values) > 1.0 + 1e-3, name


@pytest.mark.parametrize("name", EASING_FUNCTIONS.keys())
def test_get_source_is_self_contained(name):
    """The algorithm view source must define the function and include any
    constants or helpers it references."""
    src = get_source(name)
    assert f"def {EASING_FUNCTIONS[name].__name__}(" in src
    if "Back" in name:
        assert "_C1 = 1.70158" in src
    if "Elastic" in name:
        assert "_C4" in src
    if "Bounce" in name:
        assert "def ease_out_bounce(" in src


def test_bounce_segments_continuous():
    """Out-bounce must be continuous across its piecewise boundaries."""
    for boundary in [1.0 / 2.75, 2.0 / 2.75, 2.5 / 2.75]:
        below = ease_out_bounce(boundary - 1e-9)
        above = ease_out_bounce(boundary + 1e-9)
        assert below == pytest.approx(above, abs=1e-6)
