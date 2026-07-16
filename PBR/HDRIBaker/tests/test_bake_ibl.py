"""End-to-end bake tests. These need a real GPU adapter, so they skip when
wgpu cannot get a device (headless CI, no drivers).
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")


@pytest.fixture(scope="module")
def gpu():
    import wgpu.utils

    try:
        wgpu.utils.get_default_device()
    except Exception as err:  # noqa: BLE001 - any adapter failure means skip
        pytest.skip(f"no usable wgpu device: {err}")


def _tiny_equirect() -> np.ndarray:
    # A 64x32 gradient is enough to prove the plumbing without a slow bake.
    height, width = 32, 64
    ramp = np.linspace(0.0, 4.0, width, dtype=np.float32)
    return np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2)


def test_bake_honours_non_default_sizes(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings, expected_shapes

    settings = BakeSettings(
        env_size=64,
        irradiance_size=8,
        prefilter_size=16,
        prefilter_mips=3,
        lut_size=32,
        prefilter_samples=16,
        brdf_samples=16,
        irradiance_sample_delta=0.5,
    )
    maps = bake_maps(_tiny_equirect(), settings, source="test.exr")

    for key, shape in expected_shapes(settings).items():
        assert maps[key].shape == shape, key
    assert maps["meta"]["settings"] == settings.to_meta()


def test_baked_maps_survive_a_save_load_round_trip(gpu, tmp_path):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings
    from ibl_maps import load_maps, save_maps

    settings = BakeSettings(
        env_size=64,
        irradiance_size=8,
        prefilter_size=16,
        prefilter_mips=3,
        lut_size=32,
        prefilter_samples=16,
        brdf_samples=16,
        irradiance_sample_delta=0.5,
    )
    maps = bake_maps(_tiny_equirect(), settings, source="test.exr")
    path = tmp_path / "baked.npz"
    save_maps(maps, path)

    loaded = load_maps(path)
    assert loaded["settings"] == settings
    assert loaded["irradiance"].shape == (6, 8, 8, 4)


def test_bake_records_the_roughness_the_mips_were_baked_at(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    maps = bake_maps(
        _tiny_equirect(),
        BakeSettings(
            env_size=32,
            prefilter_size=16,
            prefilter_mips=2,
            lut_size=32,
            irradiance_size=8,
            prefilter_samples=16,
            brdf_samples=16,
            irradiance_sample_delta=0.5,
        ),
    )
    assert maps["meta"]["prefilter_roughness"] == [0.0, 1.0]


def test_omitting_settings_bakes_at_the_defaults(gpu):
    # The GUI always passes settings, but the default must stay a real bake --
    # a 512 env at 1024 samples, so this one is slow by nature.
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings, expected_shapes

    maps = bake_maps(_tiny_equirect())
    for key, shape in expected_shapes(BakeSettings()).items():
        assert maps[key].shape == shape, key


def test_sample_count_changes_the_prefiltered_result(gpu):
    # The whole point of exposing SAMPLE_COUNT: it must actually reach the
    # shader. A 4-sample GGX prefilter cannot match a 512-sample one.
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings, prefilter_key

    base = dict(
        env_size=64,
        irradiance_size=8,
        prefilter_size=16,
        prefilter_mips=3,
        lut_size=32,
        irradiance_sample_delta=0.5,
    )
    noisy = bake_maps(
        _tiny_equirect(), BakeSettings(**base, prefilter_samples=4, brdf_samples=16)
    )
    clean = bake_maps(
        _tiny_equirect(), BakeSettings(**base, prefilter_samples=512, brdf_samples=16)
    )
    # mip 2 is the roughest level, where sampling noise shows up most
    a = np.asarray(noisy[prefilter_key(2)], np.float32)
    b = np.asarray(clean[prefilter_key(2)], np.float32)
    assert not np.allclose(a, b, atol=1e-3)


def test_brdf_sample_count_changes_the_lut(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    base = dict(
        env_size=32,
        irradiance_size=8,
        prefilter_size=16,
        prefilter_mips=2,
        lut_size=64,
        irradiance_sample_delta=0.5,
    )
    coarse = bake_maps(
        _tiny_equirect(), BakeSettings(**base, brdf_samples=4, prefilter_samples=16)
    )
    fine = bake_maps(
        _tiny_equirect(), BakeSettings(**base, brdf_samples=512, prefilter_samples=16)
    )
    a = np.asarray(coarse["brdf_lut"], np.float32)
    b = np.asarray(fine["brdf_lut"], np.float32)
    assert not np.allclose(a, b, atol=1e-3)


def test_irradiance_sample_delta_changes_the_irradiance(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    base = dict(
        env_size=64,
        irradiance_size=8,
        prefilter_size=16,
        prefilter_mips=2,
        lut_size=32,
        prefilter_samples=16,
        brdf_samples=16,
    )
    coarse = bake_maps(
        _tiny_equirect(), BakeSettings(**base, irradiance_sample_delta=0.5)
    )
    fine = bake_maps(
        _tiny_equirect(), BakeSettings(**base, irradiance_sample_delta=0.05)
    )
    a = np.asarray(coarse["irradiance"], np.float32)
    b = np.asarray(fine["irradiance"], np.float32)
    assert not np.allclose(a, b, atol=1e-3)


# Face order of a cube texture's array layers: +X, -X, +Y, -Y, +Z, -Z.
_POS_Y, _NEG_Y, _POS_Z = 2, 3, 4


def _sky_and_ground_equirect() -> np.ndarray:
    """A panorama whose top half (the sky) is red and bottom half (the ground)
    is blue. Row 0 of an equirect is straight up, the last row straight down.
    """
    height, width = 32, 64
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[: height // 2, :, 0] = 1.0  # sky: red
    image[height // 2 :, :, 2] = 1.0  # ground: blue
    return image


def _dominant_channel(face: np.ndarray) -> int:
    return int(np.argmax(np.asarray(face, np.float32).mean(axis=(0, 1))[:3]))


def test_bake_does_not_swap_the_poles(gpu):
    """The +Y face must be the sky and -Y the ground.

    The equirect maths and the capture projection each flip vertically. On the
    four side faces those flips cancel, so a sign error there hides; on the
    poles they do not, and the ceiling ends up underfoot.
    """
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    maps = bake_maps(
        _sky_and_ground_equirect(),
        BakeSettings(
            env_size=32,
            irradiance_size=8,
            prefilter_size=16,
            prefilter_mips=2,
            lut_size=32,
            prefilter_samples=16,
            brdf_samples=16,
        ),
    )
    env = np.asarray(maps["env"], np.float32)

    assert _dominant_channel(env[_POS_Y]) == 0, "+Y (up) should be the red sky"
    assert _dominant_channel(env[_NEG_Y]) == 2, "-Y (down) should be the blue ground"


def test_bake_keeps_side_faces_upright(gpu):
    """A side face must have sky along its top rows, not its bottom ones.

    This is the other half of the pair: negating only the equirect v mapping
    fixes the poles but leaves every side face upside down.
    """
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    maps = bake_maps(
        _sky_and_ground_equirect(),
        BakeSettings(
            env_size=32,
            irradiance_size=8,
            prefilter_size=16,
            prefilter_mips=2,
            lut_size=32,
            prefilter_samples=16,
            brdf_samples=16,
        ),
    )
    face = np.asarray(maps["env"], np.float32)[_POS_Z]
    top, bottom = face[:8], face[-8:]

    assert _dominant_channel(top) == 0, "top of a side face should be sky"
    assert _dominant_channel(bottom) == 2, "bottom of a side face should be ground"
