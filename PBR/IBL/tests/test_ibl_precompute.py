"""Tests for the numpy-only IBL precompute maths (no GL/Qt imports).

Two stages are tested here (see PBR/IBL/README.md for the full record of
what shipped vs. was cut):

1. the diffuse irradiance map (cosine-weighted hemisphere convolution of a
   source cubemap)
2. the BRDF split-sum LUT (Karis 2013 UE4 course notes)

The prefiltered specular mip chain (stage 2 of the brief's four) was cut
for time -- there is no test for it because there is no code for it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ibl_precompute import compute_brdf_lut, generate_irradiance_map  # noqa: E402


def _uniform_cubemap(colour: tuple[float, float, float], size: int = 32) -> list:
    """Six faces, every texel the same flat colour -- lets us assert the
    convolution of a constant environment reproduces that same constant."""
    face = np.empty((size, size, 3), dtype=np.float64)
    face[:] = colour
    return [face.copy() for _ in range(6)]


class TestIrradianceMap:
    def test_uniform_environment_irradiance_equals_its_colour(self):
        faces = _uniform_cubemap((0.6, 0.3, 0.1))
        irradiance = generate_irradiance_map(faces, out_size=8)

        target = np.array([0.6, 0.3, 0.1])
        assert len(irradiance) == 6
        for face in irradiance:
            assert face.shape == (8, 8, 3)
            np.testing.assert_allclose(
                face, np.broadcast_to(target, face.shape), atol=0.05
            )

    def test_output_size_matches_request(self):
        faces = _uniform_cubemap((1.0, 1.0, 1.0), size=16)
        irradiance = generate_irradiance_map(faces, out_size=4)
        for face in irradiance:
            assert face.shape == (4, 4, 3)


class TestBRDFLut:
    def test_lut_shape(self):
        lut = compute_brdf_lut(size=32, sample_count=64)
        assert lut.shape == (32, 32, 2)

    def test_values_in_valid_range(self):
        lut = compute_brdf_lut(size=32, sample_count=64)
        assert np.all(lut >= -1e-3)
        assert np.all(lut <= 1.0 + 1e-3)

    def test_energy_conservation_a_plus_b_below_one(self):
        lut = compute_brdf_lut(size=32, sample_count=64)
        total = lut[..., 0] + lut[..., 1]
        assert np.all(total <= 1.05)

    def test_mirror_corner_normal_incidence_zero_roughness(self):
        """At NdotV == 1, roughness == 0 the surface is a perfect mirror:
        every GGX sample degenerates to H == N == V, so the split-sum
        integral collapses to A == 1, B == 0 exactly (see the derivation
        in ibl_precompute.py's docstring). Roughness is the LUT's second
        axis running low -> high, so index 0 is (near) roughness == 0."""
        lut = compute_brdf_lut(size=32, sample_count=64)
        a, b = lut[-1, 0]
        assert a == pytest.approx(1.0, abs=0.02)
        assert b == pytest.approx(0.0, abs=0.02)

    def test_a_decreases_with_roughness_at_normal_incidence(self):
        """Known LUT shape: at fixed NdotV, rougher surfaces scatter more
        of the reflected energy away from the ideal mirror direction, so
        the A (scale) term falls off as roughness rises."""
        lut = compute_brdf_lut(size=32, sample_count=64)
        column = lut[-1, :, 0]  # NdotV == 1 row across all roughness values
        assert column[0] > column[-1]
