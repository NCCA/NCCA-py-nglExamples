import ibl_maps
import numpy as np
import pytest
from ibl_maps import load_maps, prefilter_key, save_maps


def _fake_maps() -> dict:
    maps = {
        "env": np.zeros((6, 512, 512, 4), np.float16),
        "irradiance": np.full((6, 32, 32, 4), 0.25, np.float16),
        "brdf_lut": np.zeros((512, 512, 2), np.float16),
        "meta": {"source": "test.exr", "prefilter_mips": ibl_maps.PREFILTER_MIPS},
    }
    for mip in range(ibl_maps.PREFILTER_MIPS):
        size = ibl_maps.PREFILTER_SIZE >> mip
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
    assert loaded["meta"]["prefilter_mips"] == ibl_maps.PREFILTER_MIPS


def test_load_rejects_missing_array(tmp_path):
    path = tmp_path / "broken.npz"
    maps = _fake_maps()
    del maps["irradiance"]
    # save raw so the file exists but is incomplete
    np.savez_compressed(path, **{k: v for k, v in maps.items() if k != "meta"})
    with pytest.raises(ValueError, match="irradiance"):
        load_maps(path)
