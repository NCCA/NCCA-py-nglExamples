"""Headless tests for the PostProcessChain tonemap maths (numpy-only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tonemap_maths import aces_fitted, apply_gamma, reinhard  # noqa: E402

CURVES = {"reinhard": reinhard, "aces_fitted": aces_fitted}


class TestMapsZeroToZero:
    def test_reinhard(self):
        assert reinhard(0.0) == 0.0

    def test_aces_fitted(self):
        assert aces_fitted(0.0) == 0.0


class TestMonotonic:
    def test_reinhard_monotonic(self):
        xs = np.linspace(0.0, 50.0, 500)
        ys = reinhard(xs)
        assert np.all(np.diff(ys) >= 0.0)

    def test_aces_fitted_monotonic(self):
        xs = np.linspace(0.0, 50.0, 500)
        ys = aces_fitted(xs)
        assert np.all(np.diff(ys) >= -1e-12)


class TestLargeInputBounded:
    def test_reinhard_large_input_at_most_one(self):
        assert reinhard(1000.0) <= 1.0
        assert reinhard(1e6) < 1.0  # asymptotic, never quite reaches 1

    def test_aces_fitted_large_input_at_most_one(self):
        assert aces_fitted(1000.0) <= 1.0
        assert aces_fitted(1e6) <= 1.0


class TestNeverNegative:
    def test_curves_never_negative_for_nonnegative_input(self):
        xs = np.linspace(0.0, 20.0, 200)
        for curve in CURVES.values():
            assert np.all(curve(xs) >= 0.0)


class TestVectorInput:
    def test_reinhard_accepts_rgb_triples(self):
        rgb = np.array([[8.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        out = reinhard(rgb)
        assert out.shape == rgb.shape
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0])

    def test_aces_fitted_accepts_rgb_triples(self):
        rgb = np.array([[8.0, 4.0, 0.5], [0.0, 0.0, 0.0]])
        out = aces_fitted(rgb)
        assert out.shape == rgb.shape
        assert np.all(out <= 1.0)


class TestApplyGamma:
    def test_zero_stays_zero(self):
        assert apply_gamma(0.0) == 0.0

    def test_one_stays_one(self):
        np.testing.assert_allclose(apply_gamma(1.0), 1.0)

    def test_darkens_midtones_when_encoding(self):
        # c ** (1/2.2) > c for 0 < c < 1, i.e. gamma encoding brightens
        # mid-tones relative to linear (this is the *encode* direction).
        assert apply_gamma(0.5) > 0.5
