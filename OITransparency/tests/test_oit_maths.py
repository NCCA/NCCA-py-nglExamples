"""Headless unit tests for the weighted blended OIT reference maths.

composite_pixel in oit_common.py mirrors exactly what the accumulation and
composite shaders compute, so these tests pin down the properties the GPU
version relies on -- above all that the result is independent of fragment
order, which is the entire point of the technique.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from oit_common import back_to_front, composite_pixel, over_composite, weight

OPAQUE = np.array([0.4, 0.4, 0.4])

FRAGMENTS = [
    ((1.0, 0.1, 0.1), 0.45, -6.0),
    ((0.1, 0.8, 0.2), 0.45, -8.2),
    ((0.2, 0.4, 1.0), 0.30, -7.1),
    ((1.0, 0.55, 0.1), 0.60, -9.5),
]


# ----------------------------------------------------------------------
# the weight function
# ----------------------------------------------------------------------
def test_weight_is_positive_and_scales_with_alpha():
    assert weight(-5.0, 0.5) > 0.0
    assert np.isclose(weight(-5.0, 1.0), 2.0 * weight(-5.0, 0.5))


def test_weight_decreases_with_distance():
    # nearer fragments must dominate the weighted average
    assert weight(-1.0, 0.5) > weight(-10.0, 0.5) > weight(-100.0, 0.5)


def test_weight_only_depends_on_depth_magnitude():
    assert weight(-3.0, 0.5) == weight(3.0, 0.5)


def test_weight_is_clamped():
    assert weight(-1e-6, 1.0) <= 3e3
    assert weight(-1e6, 1.0) >= 1e-2 * 1.0


# ----------------------------------------------------------------------
# compositing
# ----------------------------------------------------------------------
def test_no_fragments_returns_opaque():
    assert np.allclose(composite_pixel(OPAQUE, []), OPAQUE)


def test_single_fragment_matches_over_exactly():
    frag = FRAGMENTS[0]
    expected = over_composite(OPAQUE, [frag])
    assert np.allclose(composite_pixel(OPAQUE, [frag]), expected)


def test_order_independence():
    """The defining property: every permutation gives the same pixel."""
    reference = composite_pixel(OPAQUE, FRAGMENTS)
    for perm in itertools.permutations(FRAGMENTS):
        assert np.allclose(composite_pixel(OPAQUE, list(perm)), reference)


def test_transmittance_is_a_product_of_one_minus_alpha():
    # two 0.5-alpha layers of the same colour let 25% of the background through
    black = np.zeros(3)
    frags = [((0.0, 0.0, 0.0), 0.5, -5.0), ((0.0, 0.0, 0.0), 0.5, -5.0)]
    result = composite_pixel(np.ones(3), frags)
    assert np.allclose(result, np.ones(3) * 0.25 + black * 0.75)


def test_oit_approximates_sorted_over():
    """With well separated depths OIT should land nearer the correctly
    sorted OVER composite than the wrongly (reverse) sorted one."""
    frags_btf = [
        ((0.1, 0.8, 0.2), 0.6, -50.0),  # far, drawn first in correct order
        ((1.0, 0.1, 0.1), 0.6, -2.0),  # near
    ]
    correct = over_composite(OPAQUE, frags_btf)
    wrong = over_composite(OPAQUE, list(reversed(frags_btf)))
    oit = composite_pixel(OPAQUE, frags_btf)
    assert np.linalg.norm(oit - correct) < np.linalg.norm(oit - wrong)


# ----------------------------------------------------------------------
# the per-object sort used by comparison mode 2
# ----------------------------------------------------------------------
def test_back_to_front_sorts_furthest_first():
    def translation(z):
        m = np.identity(4)
        m[3, 2] = z
        return m

    mvs = [translation(0.5), translation(-2.0), translation(1.5)]
    assert back_to_front(mvs) == [1, 0, 2]
