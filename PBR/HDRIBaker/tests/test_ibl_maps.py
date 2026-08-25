import json

import numpy as np
import pytest
from bake_settings import BakeSettings, prefilter_key
from ibl_maps import SCHEMA_VERSION, load_maps, save_maps


def _fake_maps(settings: BakeSettings | None = None) -> dict:
    settings = settings or BakeSettings()
    maps = {
        "env": np.zeros((6, settings.env_size, settings.env_size, 4), np.float16),
        "irradiance": np.full(
            (6, settings.irradiance_size, settings.irradiance_size, 4), 0.25, np.float16
        ),
        "brdf_lut": np.zeros((settings.lut_size, settings.lut_size, 2), np.float16),
        "meta": {"source": "test.exr", "settings": settings.to_meta()},
    }
    for mip in range(settings.prefilter_mips):
        size = settings.prefilter_size >> mip
        maps[prefilter_key(mip)] = np.zeros((6, size, size, 4), np.float16)
    return maps


def test_round_trip_preserves_arrays_and_meta(tmp_path):
    path = tmp_path / "maps.npz"
    save_maps(_fake_maps(), path)
    loaded = load_maps(path)

    assert loaded["irradiance"].shape == (6, 32, 32, 4)
    assert loaded["irradiance"].dtype == np.float16
    np.testing.assert_allclose(loaded["irradiance"], 0.25)
    assert loaded[prefilter_key(4)].shape == (6, 8, 8, 4)
    assert loaded["meta"]["source"] == "test.exr"
    assert loaded["meta"]["schema_version"] == SCHEMA_VERSION


def test_round_trip_at_non_default_settings(tmp_path):
    settings = BakeSettings(
        env_size=64,
        irradiance_size=16,
        prefilter_size=32,
        prefilter_mips=3,
        lut_size=128,
        prefilter_samples=32,
        brdf_samples=16,
        irradiance_sample_delta=0.5,
    )
    path = tmp_path / "small.npz"
    save_maps(_fake_maps(settings), path)
    loaded = load_maps(path)

    assert loaded["settings"] == settings
    assert loaded["env"].shape == (6, 64, 64, 4)
    assert loaded["brdf_lut"].shape == (128, 128, 2)
    assert loaded[prefilter_key(2)].shape == (6, 8, 8, 4)
    assert prefilter_key(3) not in loaded


def test_saved_meta_records_the_settings_and_roughness(tmp_path):
    settings = BakeSettings(prefilter_mips=3, prefilter_samples=64)
    path = tmp_path / "maps.npz"
    save_maps(_fake_maps(settings), path)
    loaded = load_maps(path)

    assert loaded["meta"]["settings"]["prefilter_samples"] == 64
    assert loaded["meta"]["prefilter_roughness"] == [0.0, 0.5, 1.0]


def test_v1_file_without_settings_still_loads_at_legacy_shape(tmp_path):
    # Files baked before schema v2 have no settings block but are known to be
    # the old fixed shape; they must keep working.
    path = tmp_path / "v1.npz"
    maps = _fake_maps()
    arrays = {k: v for k, v in maps.items() if k != "meta"}
    arrays["meta"] = np.array(
        json.dumps({"source": "old.exr", "schema_version": 1, "prefilter_mips": 5})
    )
    np.savez_compressed(path, **arrays)

    loaded = load_maps(path)
    assert loaded["settings"] == BakeSettings.legacy_v1()
    assert loaded["env"].shape == (6, 512, 512, 4)


def test_load_rejects_missing_array(tmp_path):
    path = tmp_path / "broken.npz"
    maps = _fake_maps()
    del maps["irradiance"]
    arrays = {k: v for k, v in maps.items() if k != "meta"}
    arrays["meta"] = np.array(json.dumps({"settings": BakeSettings().to_meta()}))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="irradiance"):
        load_maps(path)


def test_load_rejects_array_that_contradicts_its_own_settings(tmp_path):
    path = tmp_path / "lying.npz"
    maps = _fake_maps()
    # claim 256 but store 512
    maps["meta"]["settings"] = BakeSettings(env_size=256).to_meta()
    save_maps(maps, path)
    with pytest.raises(ValueError, match="env"):
        load_maps(path)


def test_load_rejects_settings_block_that_is_internally_invalid(tmp_path):
    path = tmp_path / "bad_settings.npz"
    maps = _fake_maps()
    arrays = {k: v for k, v in maps.items() if k != "meta"}
    arrays["meta"] = np.array(
        json.dumps({"settings": {**BakeSettings().to_meta(), "prefilter_mips": 1}})
    )
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="prefilter_mips"):
        load_maps(path)


def test_load_rejects_non_npz_file_with_clear_message(tmp_path):
    # A stray non-.npz path (e.g. someone passing a .py to --maps) should give
    # an actionable message, not numpy's confusing "pickled data" fallback.
    path = tmp_path / "not_maps.py"
    path.write_text("print('hello')\n")
    with pytest.raises(ValueError, match="not a NumPy .npz IBL maps file"):
        load_maps(path)
