from dataclasses import dataclass

import pytest
from bake_settings import BakeSettings, expected_shapes, prefilter_key


def test_defaults_match_the_historic_constants():
    s = BakeSettings()
    assert s.env_size == 512
    assert s.irradiance_size == 32
    assert s.prefilter_size == 128
    assert s.prefilter_mips == 5
    assert s.lut_size == 512
    assert s.prefilter_samples == 1024
    assert s.brdf_samples == 1024
    assert s.irradiance_sample_delta == pytest.approx(0.025)


def test_roughness_spans_zero_to_one_across_the_mip_chain():
    s = BakeSettings(prefilter_mips=5)
    assert s.roughness_levels() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert s.roughness_for_mip(0) == 0.0
    assert s.roughness_for_mip(4) == 1.0


def test_two_mips_is_the_minimum_and_still_spans_the_range():
    s = BakeSettings(prefilter_mips=2)
    s.validate()
    assert s.roughness_levels() == [0.0, 1.0]


def test_one_mip_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="prefilter_mips"):
        BakeSettings(prefilter_mips=1).validate()


def test_non_power_of_two_size_is_rejected():
    with pytest.raises(ValueError, match="env_size"):
        BakeSettings(env_size=500).validate()


def test_mip_chain_may_not_shrink_below_one_texel():
    # 8 with 5 mips would need an eighth of a texel at the last level
    with pytest.raises(ValueError, match="prefilter_size"):
        BakeSettings(prefilter_size=8, prefilter_mips=5).validate()


def test_zero_samples_is_rejected():
    with pytest.raises(ValueError, match="prefilter_samples"):
        BakeSettings(prefilter_samples=0).validate()


def test_sample_delta_must_be_positive_and_sane():
    with pytest.raises(ValueError, match="irradiance_sample_delta"):
        BakeSettings(irradiance_sample_delta=0.0).validate()


def test_meta_round_trip_preserves_every_field():
    s = BakeSettings(env_size=256, prefilter_mips=3, prefilter_samples=64)
    assert BakeSettings.from_meta({"settings": s.to_meta()}) == s


def test_from_meta_falls_back_to_v1_shape_when_settings_absent():
    # A schema v1 file has no settings block; it was always baked at defaults.
    assert BakeSettings.from_meta({"source": "old.exr"}) == BakeSettings.legacy_v1()


def test_legacy_v1_matches_the_historic_shape():
    # Documents the fixed shape every v1 file was actually baked at. Literal
    # numbers on purpose, as a record of that shape -- but see the test below
    # for the one that actually catches legacy_v1() drifting off it.
    legacy = BakeSettings.legacy_v1()
    assert legacy.env_size == 512
    assert legacy.irradiance_size == 32
    assert legacy.prefilter_size == 128
    assert legacy.prefilter_mips == 5
    assert legacy.lut_size == 512


def test_legacy_v1_does_not_track_the_defaults():
    # A literal-values test alone won't catch legacy_v1() regressing to a
    # bare cls(): today's defaults happen to equal the v1 shape, so a revert
    # would pass silently until some unrelated future default change. Move
    # the defaults ourselves so a revert fails right here, right now.
    @dataclass(frozen=True)
    class Drifted(BakeSettings):
        env_size: int = 1024
        prefilter_mips: int = 3

    assert Drifted().env_size == 1024  # the drift really took effect
    legacy = Drifted.legacy_v1()
    assert legacy.env_size == 512
    assert legacy.prefilter_mips == 5


def test_expected_shapes_tracks_the_settings():
    s = BakeSettings(
        env_size=64,
        irradiance_size=16,
        prefilter_size=32,
        prefilter_mips=3,
        lut_size=128,
    )
    shapes = expected_shapes(s)
    assert shapes["env"] == (6, 64, 64, 4)
    assert shapes["irradiance"] == (6, 16, 16, 4)
    assert shapes["brdf_lut"] == (128, 128, 2)
    assert shapes[prefilter_key(0)] == (6, 32, 32, 4)
    assert shapes[prefilter_key(2)] == (6, 8, 8, 4)
    assert prefilter_key(2) == "prefilter_2"
