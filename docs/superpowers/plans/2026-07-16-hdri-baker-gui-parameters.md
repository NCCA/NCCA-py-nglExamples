# HDRI Baker GUI Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the IBL bake's map sizes and sample-quality parameters as GUI controls in `hdri_baker.py`, carrying them through a new `.npz` schema so a baked file describes its own shape.

**Architecture:** A `BakeSettings` dataclass becomes the single source of truth for every bake parameter. It flows in one direction: GUI → `bake_maps()` → GPU uniforms → the `meta` block of the saved `.npz` → `load_maps()` validation → `hdri_demo.py`'s texture upload. The module-level constants in `ibl_maps.py` stop being the definition of a map set's shape and become mere defaults that seed the GUI. Shader parameters that are currently WGSL `const` move into the existing capture-uniform structs.

**Tech Stack:** Python 3.13, PySide6 (widgets), wgpu-py (headless bake), numpy (readback + `.npz`), pytest, ruff.

## Global Constraints

- **uv only.** Every command is `uv run …`. Never invoke `python`/`pytest` bare.
- **Scope: `PBR/HDRIBaker/` only.** `PBR/HDRI/main.py` and `PBR/HDRI/HDRIWebGPU.py` keep their own private copies of the size constants and are explicitly NOT touched by this plan.
- **Schema version 2.** `ibl_maps.SCHEMA_VERSION` goes 1 → 2. Version 1 files must still load (there are existing `ibl_maps.npz` and `San.npz` files on disk baked at v1).
- **Sizes are powers of two.** Every size parameter must be a power of two; `prefilter_mips` must be ≥ 2 (the roughness step divides by `mips - 1`).
- **Float16 storage.** Arrays stay `float16` to match `BAKE_FORMAT = rgba16float` / `LUT_FORMAT = rg16float`. Do not change formats.
- **WGSL uniform alignment.** A uniform struct's size must be a multiple of 16 bytes, and `mat4x4<f32>` members align to 16. Every numpy dtype mirroring a WGSL struct must have an `itemsize` that is a multiple of 16.
- **Docs voice.** Any README or docstring prose written in this plan uses the `jon-writing-style` skill.
- **Commit style.** Conventional commits. Never commit to `main`. Work stays on the current `agent/hdri-ibl-baker` branch in this worktree.

## Deliberately out of scope

Pre-bake image controls (exposure, environment rotation, highlight clamp, source downsample) and preview-only controls (tonemap operator, mip/face inspector) were discussed and are **not** in this plan. They are cheap and self-contained and belong in a follow-up plan once the schema below is settled.

## File Structure

| File | Responsibility |
|---|---|
| `PBR/HDRIBaker/bake_settings.py` | **Create.** `BakeSettings` dataclass, validation, `to_meta()`/`from_meta()`, `expected_shapes()`. Pure Python, no GPU, no Qt. |
| `PBR/HDRIBaker/ibl_maps.py` | **Modify.** Constants become `DEFAULT_*`. `save_maps`/`load_maps` become settings-driven, with a v1 compatibility path. |
| `PBR/HDRIBaker/bake_ibl.py` | **Modify.** `bake_maps(image, settings, source)`. Sizes and sample counts come from `settings` and reach the GPU as uniforms. |
| `PBR/HDRIBaker/shaders/Prefilter.wgsl` | **Modify.** `SAMPLE_COUNT` and the hardcoded `resolution = 512.0` become uniform members. |
| `PBR/HDRIBaker/shaders/Irradiance.wgsl` | **Modify.** `sampleDelta` becomes a uniform member. |
| `PBR/HDRIBaker/shaders/BRDF.wgsl` | **Modify.** Gains its first bind group so `SAMPLE_COUNT` can be a uniform. |
| `PBR/HDRIBaker/shaders/PBR.wgsl` | **Modify.** `MAX_REFLECTION_LOD` becomes a uniform member (it is `prefilter_mips - 1`). |
| `PBR/HDRIBaker/hdri_demo.py` | **Modify.** Sizes come from the loaded file's `meta`, not from imported constants. |
| `PBR/HDRIBaker/hdri_baker.py` | **Modify.** Adds the settings panel, a bake-timing readout, and passes `BakeSettings` to `bake_maps`. |
| `PBR/HDRIBaker/tests/test_bake_settings.py` | **Create.** Validation + meta round-trip + `expected_shapes`. |
| `PBR/HDRIBaker/tests/test_ibl_maps.py` | **Modify.** Settings-driven round trip, v1 back-compat, shape rejection. |
| `PBR/HDRIBaker/tests/test_bake_ibl.py` | **Create.** GPU-marked end-to-end bake at non-default settings. |
| `PBR/HDRIBaker/README.md` | **Modify.** Document the controls and the v2 schema. |

---

### Task 1: `BakeSettings` — the single source of truth

**Files:**
- Create: `PBR/HDRIBaker/bake_settings.py`
- Test: `PBR/HDRIBaker/tests/test_bake_settings.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BakeSettings` frozen dataclass with fields `env_size: int = 512`, `irradiance_size: int = 32`, `prefilter_size: int = 128`, `prefilter_mips: int = 5`, `lut_size: int = 512`, `prefilter_samples: int = 1024`, `brdf_samples: int = 1024`, `irradiance_sample_delta: float = 0.025`
  - `BakeSettings.validate(self) -> None` — raises `ValueError`
  - `BakeSettings.roughness_for_mip(self, mip: int) -> float`
  - `BakeSettings.roughness_levels(self) -> list[float]`
  - `BakeSettings.to_meta(self) -> dict`
  - `BakeSettings.from_meta(meta: dict) -> BakeSettings` (classmethod)
  - `BakeSettings.legacy_v1() -> BakeSettings` (classmethod)
  - `expected_shapes(settings: BakeSettings) -> dict[str, tuple[int, ...]]`
  - `prefilter_key(mip: int) -> str` — moves here from `ibl_maps.py`

- [ ] **Step 1: Write the failing test**

Create `PBR/HDRIBaker/tests/test_bake_settings.py`:

```python
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


def test_expected_shapes_tracks_the_settings():
    s = BakeSettings(env_size=64, irradiance_size=16, prefilter_size=32,
                     prefilter_mips=3, lut_size=128)
    shapes = expected_shapes(s)
    assert shapes["env"] == (6, 64, 64, 4)
    assert shapes["irradiance"] == (6, 16, 16, 4)
    assert shapes["brdf_lut"] == (128, 128, 2)
    assert shapes[prefilter_key(0)] == (6, 32, 32, 4)
    assert shapes[prefilter_key(2)] == (6, 8, 8, 4)
    assert prefilter_key(2) == "prefilter_2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest PBR/HDRIBaker/tests/test_bake_settings.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'bake_settings'`.

- [ ] **Step 3: Write minimal implementation**

Create `PBR/HDRIBaker/bake_settings.py`:

```python
"""Every knob the IBL bake has, in one place.

The baker used to hardcode its map sizes and sample counts as module
constants, which meant a saved ``.npz`` was only readable by a build that
happened to agree with it. A ``BakeSettings`` travels with the bake instead:
the GUI builds one, the bake obeys it, and it is written into the file's
``meta`` block so a loader can size its textures from the file rather than
from an import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

_SIZE_FIELDS = ("env_size", "irradiance_size", "prefilter_size", "lut_size")


def prefilter_key(mip: int) -> str:
    """Name of the prefilter array for roughness mip ``mip`` (0 = mirror)."""
    return f"prefilter_{mip}"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class BakeSettings:
    """Sizes and sample counts for one bake. Defaults reproduce the v1 maps."""

    env_size: int = 512
    irradiance_size: int = 32
    prefilter_size: int = 128
    prefilter_mips: int = 5
    lut_size: int = 512
    prefilter_samples: int = 1024
    brdf_samples: int = 1024
    irradiance_sample_delta: float = 0.025

    def validate(self) -> None:
        """Raise ``ValueError`` if any field would produce a broken bake."""
        for name in _SIZE_FIELDS:
            value = getattr(self, name)
            if not _is_power_of_two(value):
                raise ValueError(f"{name} must be a power of two, got {value}")
        # roughness_for_mip divides by (mips - 1), and one mip cannot span
        # a roughness range anyway.
        if self.prefilter_mips < 2:
            raise ValueError(f"prefilter_mips must be >= 2, got {self.prefilter_mips}")
        if self.prefilter_size >> (self.prefilter_mips - 1) < 1:
            raise ValueError(
                f"prefilter_size {self.prefilter_size} is too small for "
                f"{self.prefilter_mips} mips (the last level would be under a texel)"
            )
        for name in ("prefilter_samples", "brdf_samples"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not 0.0 < self.irradiance_sample_delta <= 1.0:
            raise ValueError(
                "irradiance_sample_delta must be in (0, 1], got "
                f"{self.irradiance_sample_delta}"
            )

    def roughness_for_mip(self, mip: int) -> float:
        """Roughness baked into prefilter mip ``mip``: 0 at the top, 1 at the last."""
        return mip / (self.prefilter_mips - 1)

    def roughness_levels(self) -> list[float]:
        return [self.roughness_for_mip(m) for m in range(self.prefilter_mips)]

    def to_meta(self) -> dict:
        """A JSON-safe dict of every field, for the ``.npz`` meta block."""
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: dict) -> "BakeSettings":
        """Rebuild settings from a file's meta block.

        A schema v1 file has no ``settings`` block; it can only ever have been
        baked at the constants of the day, so fall back to those.
        """
        block = meta.get("settings")
        if block is None:
            return cls.legacy_v1()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in known})

    @classmethod
    def legacy_v1(cls) -> "BakeSettings":
        """The fixed shape every schema v1 file was baked at."""
        return cls()


def expected_shapes(settings: BakeSettings) -> dict[str, tuple[int, ...]]:
    """Array name -> required shape for a map set baked at ``settings``."""
    shapes: dict[str, tuple[int, ...]] = {
        "env": (6, settings.env_size, settings.env_size, 4),
        "irradiance": (6, settings.irradiance_size, settings.irradiance_size, 4),
        "brdf_lut": (settings.lut_size, settings.lut_size, 2),
    }
    for mip in range(settings.prefilter_mips):
        size = settings.prefilter_size >> mip
        shapes[prefilter_key(mip)] = (6, size, size, 4)
    return shapes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest PBR/HDRIBaker/tests/test_bake_settings.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Lint**

```bash
uv run ruff check PBR/HDRIBaker/bake_settings.py PBR/HDRIBaker/tests/test_bake_settings.py
uv run ruff format PBR/HDRIBaker/bake_settings.py PBR/HDRIBaker/tests/test_bake_settings.py
```

Expected: `All checks passed!` and formatting reported unchanged or reformatted.

- [ ] **Step 6: Commit**

```bash
git add PBR/HDRIBaker/bake_settings.py PBR/HDRIBaker/tests/test_bake_settings.py
git commit -m "feat(hdri): add BakeSettings as the single source of truth for bake parameters"
```

---

### Task 2: Schema v2 — the file describes its own shape

**Files:**
- Modify: `PBR/HDRIBaker/ibl_maps.py` (whole file — it is 81 lines)
- Modify: `PBR/HDRIBaker/tests/test_ibl_maps.py` (whole file)

**Interfaces:**
- Consumes: `BakeSettings`, `expected_shapes`, `prefilter_key` from Task 1.
- Produces:
  - `SCHEMA_VERSION = 2`
  - `DEFAULT_ENV_SIZE = 512`, `DEFAULT_IRRADIANCE_SIZE = 32`, `DEFAULT_PREFILTER_SIZE = 128`, `DEFAULT_PREFILTER_MIPS = 5`, `DEFAULT_LUT_SIZE = 512` (GUI seeds only — nothing may derive a shape from them)
  - `save_maps(maps: dict, path: str | Path) -> None` — unchanged signature; `maps["meta"]["settings"]` must be present
  - `load_maps(path: str | Path) -> dict` — unchanged signature; returned dict gains `maps["settings"]` as a `BakeSettings`
  - `prefilter_key` stays importable from `ibl_maps` (re-exported from `bake_settings`) so existing importers keep working

**Note on the old constants:** `ENV_SIZE`, `IRRADIANCE_SIZE`, `PREFILTER_SIZE`, `PREFILTER_MIPS` and `LUT_SIZE` are **removed** from `ibl_maps`, not aliased. `hdri_demo.py:36-41` imports them and is fixed in Task 5; `bake_ibl.py` uses them and is fixed in Task 3. Removing them makes any missed caller fail loudly at import rather than silently bake the wrong shape. Tasks 2, 3 and 5 therefore land as a group — expect `hdri_demo.py` to be broken between Task 2 and Task 5.

- [ ] **Step 1: Write the failing test**

Replace the whole of `PBR/HDRIBaker/tests/test_ibl_maps.py` with:

```python
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
        env_size=64, irradiance_size=16, prefilter_size=32, prefilter_mips=3,
        lut_size=128, prefilter_samples=32, brdf_samples=16,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest PBR/HDRIBaker/tests/test_ibl_maps.py -v
```

Expected: collection error — `ImportError: cannot import name 'SCHEMA_VERSION'` is not it; `SCHEMA_VERSION` already exists, so expect failures on `test_round_trip_at_non_default_settings` (`KeyError: 'settings'`) and `test_v1_file_without_settings_still_loads_at_legacy_shape`.

- [ ] **Step 3: Write minimal implementation**

Replace the whole of `PBR/HDRIBaker/ibl_maps.py` with:

```python
"""Save and load the baked split-sum IBL maps as a single ``.npz``.

One file holds every map the shader needs: the environment cube (for the
skybox), the irradiance cube (diffuse ambient), the prefiltered specular
cube's mip chain, and the BRDF lookup table. Arrays are float16 to match the
GPU bake format.

The ``meta`` block is what makes a file self-describing: it carries the
:class:`~bake_settings.BakeSettings` the maps were baked at, so a loader sizes
its textures from the file it was handed rather than from whatever constants
its own build happens to hold. Files written before that block existed
(schema 1) are assumed to be the one fixed shape the baker could produce then.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from bake_settings import BakeSettings, expected_shapes, prefilter_key

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_ENV_SIZE",
    "DEFAULT_IRRADIANCE_SIZE",
    "DEFAULT_PREFILTER_SIZE",
    "DEFAULT_PREFILTER_MIPS",
    "DEFAULT_LUT_SIZE",
    "prefilter_key",
    "save_maps",
    "load_maps",
]

SCHEMA_VERSION = 2

# Seeds for the baker's GUI only. Nothing may derive a map's shape from these
# -- that is the settings block's job.
DEFAULT_ENV_SIZE = 512
DEFAULT_IRRADIANCE_SIZE = 32
DEFAULT_PREFILTER_SIZE = 128
DEFAULT_PREFILTER_MIPS = 5
DEFAULT_LUT_SIZE = 512


def _array_keys(settings: BakeSettings) -> list[str]:
    return ["env", "irradiance", "brdf_lut"] + [
        prefilter_key(m) for m in range(settings.prefilter_mips)
    ]


def save_maps(maps: dict, path: str | Path) -> None:
    """Write ``maps`` (arrays + a ``meta`` dict carrying ``settings``) to a ``.npz``."""
    meta = dict(maps["meta"])
    settings = BakeSettings.from_meta(meta)
    settings.validate()
    meta["schema_version"] = SCHEMA_VERSION
    meta["settings"] = settings.to_meta()
    meta["prefilter_roughness"] = settings.roughness_levels()

    arrays = {
        k: np.asarray(maps[k], dtype=np.float16) for k in _array_keys(settings)
    }
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load_maps(path: str | Path) -> dict:
    """Load a ``.npz`` written by :func:`save_maps`; validate and return it.

    The returned dict holds each array, the raw ``meta`` dict, and a
    ``settings`` :class:`BakeSettings` rebuilt from the file.
    """
    try:
        npz = np.load(path, allow_pickle=False)
    except ValueError as err:
        # A non-.npz file (e.g. a stray .py passed to --maps) makes np.load
        # fall back to pickle and raise a confusing "pickled data" message;
        # translate it into something a user can act on.
        raise ValueError(
            f"{path} is not a NumPy .npz IBL maps file "
            f"(expected one written by the baker's Save)"
        ) from err
    with npz:
        if "meta" not in npz:
            raise ValueError(f"maps file {path} is missing arrays: ['meta']")
        meta = json.loads(str(npz["meta"]))
        settings = BakeSettings.from_meta(meta)
        settings.validate()

        missing = [k for k in _array_keys(settings) if k not in npz]
        if missing:
            raise ValueError(f"maps file {path} is missing arrays: {missing}")
        out = {k: npz[k] for k in _array_keys(settings)}

    out["meta"] = meta
    out["settings"] = settings
    for key, shape in expected_shapes(settings).items():
        if out[key].shape != shape:
            raise ValueError(
                f"maps file {path}: {key} has shape {out[key].shape}, expected {shape}"
            )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest PBR/HDRIBaker/tests/ -v
```

Expected: all of `test_bake_settings.py`, `test_ibl_maps.py` and `test_hdri_input.py` pass.

- [ ] **Step 5: Confirm the existing v1 file on disk still loads**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'PBR/HDRIBaker')
from ibl_maps import load_maps
m = load_maps('PBR/HDRIBaker/ibl_maps.npz')
print('settings:', m['settings'])
print('env:', m['env'].shape)
"
```

Expected: prints the legacy settings and `env: (6, 512, 512, 4)`. If this raises, the v1 fallback in `from_meta` is wrong — fix before continuing.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check PBR/HDRIBaker/ibl_maps.py PBR/HDRIBaker/tests/test_ibl_maps.py
uv run ruff format PBR/HDRIBaker/ibl_maps.py PBR/HDRIBaker/tests/test_ibl_maps.py
git add PBR/HDRIBaker/ibl_maps.py PBR/HDRIBaker/tests/test_ibl_maps.py
git commit -m "feat(hdri): make the IBL .npz self-describing with a schema v2 settings block"
```

---

### Task 3: The bake obeys the settings' sizes

**Files:**
- Modify: `PBR/HDRIBaker/bake_ibl.py:53-83` (the `bake_maps` function), `:252-291` (`bake_prefilter`), `:316-354` (`bake_brdf`)
- Create: `PBR/HDRIBaker/tests/test_bake_ibl.py`

**Interfaces:**
- Consumes: `BakeSettings`, `prefilter_key` from Task 1.
- Produces: `bake_maps(image: np.ndarray, settings: BakeSettings | None = None, source: str = "") -> dict` — returns the same dict as before plus a `meta["settings"]` block ready for `save_maps`.

This task changes **sizes only**. Sample counts stay hardcoded in WGSL until Task 4, so `settings.prefilter_samples` is accepted and recorded but has no effect yet. That is deliberate: it keeps the GPU-shape change and the shader-plumbing change independently reviewable.

- [ ] **Step 1: Write the failing test**

Create `PBR/HDRIBaker/tests/test_bake_ibl.py`:

```python
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
        env_size=64, irradiance_size=8, prefilter_size=16, prefilter_mips=3,
        lut_size=32, prefilter_samples=16, brdf_samples=16,
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
        env_size=64, irradiance_size=8, prefilter_size=16, prefilter_mips=3,
        lut_size=32, prefilter_samples=16, brdf_samples=16,
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

    maps = bake_maps(_tiny_equirect(), BakeSettings(env_size=32, prefilter_size=16,
                                                    prefilter_mips=2, lut_size=32,
                                                    irradiance_size=8,
                                                    prefilter_samples=16,
                                                    brdf_samples=16,
                                                    irradiance_sample_delta=0.5))
    assert maps["meta"]["prefilter_roughness"] == [0.0, 1.0]


def test_omitting_settings_bakes_at_the_defaults(gpu):
    # The GUI always passes settings, but the default must stay a real bake --
    # a 512 env at 1024 samples, so this one is slow by nature.
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings, expected_shapes

    maps = bake_maps(_tiny_equirect())
    for key, shape in expected_shapes(BakeSettings()).items():
        assert maps[key].shape == shape, key
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest PBR/HDRIBaker/tests/test_bake_ibl.py -v
```

Expected: FAIL with `TypeError: bake_maps() takes ... positional argument` (it currently has signature `(image, source="")`, so a `BakeSettings` lands in `source`). If instead every test SKIPS, this machine has no GPU adapter — stop and report that, because the rest of this task cannot be verified here.

- [ ] **Step 3: Rewrite `bake_maps`**

In `PBR/HDRIBaker/bake_ibl.py`, replace the import of `ibl_maps` and the `bake_maps` function (lines 14 and 53-83) with:

```python
from bake_settings import BakeSettings, prefilter_key
```

(delete `import ibl_maps` — the module no longer needs it, and Task 2 removed the constants it used)

```python
def bake_maps(
    image: np.ndarray,
    settings: BakeSettings | None = None,
    source: str = "",
) -> dict:
    """Bake every IBL map from an ``(H, W, 3)`` float32 equirect image."""
    settings = settings or BakeSettings()
    settings.validate()

    device = wgpu.utils.get_default_device()
    baker = _Baker(device, settings)
    rgba = np.dstack([image, np.ones(image.shape[:2], np.float32)]).astype(np.float32)
    equirect = baker.upload_2d(rgba, BAKE_FORMAT)

    env = baker.bake_cube("Equirect2Cube.wgsl", settings.env_size, "2d", equirect)
    irradiance = baker.bake_cube(
        "Irradiance.wgsl", settings.irradiance_size, "cube", env
    )
    prefilter = baker.bake_prefilter(env)
    lut = baker.bake_brdf()

    out = {
        "env": baker.read_cube(env, settings.env_size, 0),
        "irradiance": baker.read_cube(irradiance, settings.irradiance_size, 0),
        "brdf_lut": baker.read_2d(lut, settings.lut_size, settings.lut_size, 2, 0),
    }
    for mip in range(settings.prefilter_mips):
        size = settings.prefilter_size >> mip
        out[prefilter_key(mip)] = baker.read_cube(prefilter, size, mip)
    out["meta"] = {
        "source": source,
        "settings": settings.to_meta(),
        "prefilter_mips": settings.prefilter_mips,
        "prefilter_roughness": settings.roughness_levels(),
        "format": "rgba16float / rg16float",
    }
    return out
```

- [ ] **Step 4: Thread the settings through `_Baker`**

Replace `_Baker.__init__` (lines 87-101) so it takes and stores the settings:

```python
class _Baker:
    def __init__(self, device: "wgpu.GPUDevice", settings: BakeSettings) -> None:
        self.device = device
        self.settings = settings
        cube = PrimData.primitive(Prims.CUBE.value).astype(np.float32)
```

(the rest of `__init__` — `cube_buffer`, `cube_count`, `sampler` — is unchanged)

In `bake_prefilter` (lines 252-291), replace every `ibl_maps.PREFILTER_SIZE` with `self.settings.prefilter_size`, every `ibl_maps.PREFILTER_MIPS` with `self.settings.prefilter_mips`, and the roughness line with `roughness = self.settings.roughness_for_mip(mip)`:

```python
    def bake_prefilter(self, src) -> "wgpu.GPUTexture":
        pipe = self._make_cube_pipeline(
            "Prefilter.wgsl", "cube", _PREFILTER_CAPTURE_DTYPE
        )
        source_bind_group = self._source_bind_group(pipe["source_bgl"], src, "cube")

        size0 = self.settings.prefilter_size
        cube = self.device.create_texture(
            size=(size0, size0, 6),
            mip_level_count=self.settings.prefilter_mips,
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
        )

        for mip in range(self.settings.prefilter_mips):
            size = size0 >> mip
            uniforms = np.zeros((), dtype=_PREFILTER_CAPTURE_DTYPE)
            uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
            uniforms["roughness"] = self.settings.roughness_for_mip(mip)
            for face in range(6):
                uniforms["view"] = _CAPTURE_VIEWS[face].to_numpy()
                self.device.queue.write_buffer(
                    pipe["capture_buffer"], 0, uniforms.tobytes()
                )
                self._render_face(
                    pipe["pipeline"],
                    pipe["capture_bind_group"],
                    source_bind_group,
                    cube.create_view(
                        dimension="2d",
                        base_array_layer=face,
                        array_layer_count=1,
                        base_mip_level=mip,
                        mip_level_count=1,
                    ),
                    size,
                )
        return cube
```

In `bake_brdf` (lines 316-354), replace the three `ibl_maps.LUT_SIZE` uses:

```python
        lut_size = self.settings.lut_size
        lut = self.device.create_texture(
            size=(lut_size, lut_size, 1),
```

and

```python
        render_pass.set_viewport(0, 0, lut_size, lut_size, 0, 1)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest PBR/HDRIBaker/tests/test_bake_ibl.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Confirm nothing else still imports the dead constants**

```bash
grep -rn "ibl_maps\.\(ENV_SIZE\|IRRADIANCE_SIZE\|PREFILTER_SIZE\|PREFILTER_MIPS\|LUT_SIZE\)" PBR/HDRIBaker/
```

Expected: no matches. (`hdri_demo.py` imports the bare names, not the dotted ones — it is still broken here and is fixed in Task 5.)

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/tests/test_bake_ibl.py
uv run ruff format PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/tests/test_bake_ibl.py
git add PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/tests/test_bake_ibl.py
git commit -m "feat(hdri): bake every map at the size the BakeSettings asks for"
```

---

### Task 4: Sample counts become uniforms, not WGSL constants

**Files:**
- Modify: `PBR/HDRIBaker/shaders/Prefilter.wgsl` (the `CaptureUniforms` struct and `fragment_main`)
- Modify: `PBR/HDRIBaker/shaders/Irradiance.wgsl` (the `CaptureUniforms` struct and `fragment_main:35`)
- Modify: `PBR/HDRIBaker/shaders/BRDF.wgsl` (add a bind group)
- Modify: `PBR/HDRIBaker/bake_ibl.py` (the dtypes at `:35-50`, `bake_cube`, `bake_brdf`)
- Modify: `PBR/HDRIBaker/tests/test_bake_ibl.py` (add the quality assertions)

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: no new Python API. `settings.prefilter_samples`, `settings.brdf_samples` and `settings.irradiance_sample_delta` now actually change the output.

**Why uniforms and not WGSL `override` constants:** `override` would be the tidier fit (these never change within a bake), but wgpu-native's support for pipeline-overridable constants is uneven across backends, and this file already has a working uniform-struct pattern with `_PREFILTER_CAPTURE_DTYPE`. Follow the pattern that is already proven here.

**This task also fixes a latent bug:** `Prefilter.wgsl:101` hardcodes `let resolution = 512.0` with the comment "keep in sync with ENV_SIZE". Once `env_size` is adjustable, that constant silently biases the mip selection for every non-512 bake. It becomes a uniform here.

- [ ] **Step 1: Write the failing test**

Append to `PBR/HDRIBaker/tests/test_bake_ibl.py`:

```python
def test_sample_count_changes_the_prefiltered_result(gpu):
    # The whole point of exposing SAMPLE_COUNT: it must actually reach the
    # shader. A 4-sample GGX prefilter cannot match a 512-sample one.
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings, prefilter_key

    base = dict(env_size=64, irradiance_size=8, prefilter_size=16,
                prefilter_mips=3, lut_size=32, irradiance_sample_delta=0.5)
    noisy = bake_maps(_tiny_equirect(), BakeSettings(**base, prefilter_samples=4,
                                                     brdf_samples=16))
    clean = bake_maps(_tiny_equirect(), BakeSettings(**base, prefilter_samples=512,
                                                     brdf_samples=16))
    # mip 2 is the roughest level, where sampling noise shows up most
    a = np.asarray(noisy[prefilter_key(2)], np.float32)
    b = np.asarray(clean[prefilter_key(2)], np.float32)
    assert not np.allclose(a, b, atol=1e-3)


def test_brdf_sample_count_changes_the_lut(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    base = dict(env_size=32, irradiance_size=8, prefilter_size=16,
                prefilter_mips=2, lut_size=64, irradiance_sample_delta=0.5)
    coarse = bake_maps(_tiny_equirect(), BakeSettings(**base, brdf_samples=4,
                                                      prefilter_samples=16))
    fine = bake_maps(_tiny_equirect(), BakeSettings(**base, brdf_samples=512,
                                                    prefilter_samples=16))
    a = np.asarray(coarse["brdf_lut"], np.float32)
    b = np.asarray(fine["brdf_lut"], np.float32)
    assert not np.allclose(a, b, atol=1e-3)


def test_irradiance_sample_delta_changes_the_irradiance(gpu):
    from bake_ibl import bake_maps
    from bake_settings import BakeSettings

    base = dict(env_size=64, irradiance_size=8, prefilter_size=16,
                prefilter_mips=2, lut_size=32, prefilter_samples=16,
                brdf_samples=16)
    coarse = bake_maps(_tiny_equirect(),
                       BakeSettings(**base, irradiance_sample_delta=0.5))
    fine = bake_maps(_tiny_equirect(),
                     BakeSettings(**base, irradiance_sample_delta=0.05))
    a = np.asarray(coarse["irradiance"], np.float32)
    b = np.asarray(fine["irradiance"], np.float32)
    assert not np.allclose(a, b, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest PBR/HDRIBaker/tests/test_bake_ibl.py -k "sample" -v
```

Expected: all three FAIL on `assert not np.allclose(...)` — the settings are ignored, so both bakes are byte-identical.

- [ ] **Step 3: Update the Prefilter shader**

In `PBR/HDRIBaker/shaders/Prefilter.wgsl`, extend the struct:

```wgsl
struct CaptureUniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    roughness : f32,
    sampleCount : u32,
    envResolution : f32,  // source cube's per-face size, for the mip selection below
};
```

In `fragment_main`, delete the `const SAMPLE_COUNT : u32 = 1024u;` line and read from the uniform instead:

```wgsl
    let SAMPLE_COUNT = capture.sampleCount;
```

and replace the hardcoded resolution line:

```wgsl
            let resolution = capture.envResolution;
```

- [ ] **Step 4: Update the Irradiance shader**

In `PBR/HDRIBaker/shaders/Irradiance.wgsl`, extend the struct:

```wgsl
struct CaptureUniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    sampleDelta : f32,
};
```

and in `fragment_main` replace `let sampleDelta = 0.025;` with:

```wgsl
    let sampleDelta = capture.sampleDelta;
```

- [ ] **Step 5: Update the BRDF shader**

`BRDF.wgsl` currently has no bind groups at all. Add one at the top, after the existing `const PI` line's block (keep `PI` where it is):

```wgsl
struct BRDFUniforms {
    sampleCount : u32,
};
@group(0) @binding(0) var<uniform> brdf : BRDFUniforms;
```

and in `fragment_main` delete `const SAMPLE_COUNT : u32 = 1024u;`, replacing it with:

```wgsl
    let SAMPLE_COUNT = brdf.sampleCount;
```

- [ ] **Step 6: Give each shader its own uniform dtype**

In `PBR/HDRIBaker/bake_ibl.py`, replace the dtype block at lines 35-50. `Equirect2Cube.wgsl` keeps the plain 128-byte struct; Irradiance and Prefilter get their own. Every `itemsize` is a multiple of 16 as WGSL requires.

```python
_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4))],
        "offsets": [0, 64],
        "itemsize": 128,
    }
)
_IRRADIANCE_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "sample_delta"],
        "formats": [(np.float32, (4, 4)), (np.float32, (4, 4)), np.float32],
        "offsets": [0, 64, 128],
        "itemsize": 144,
    }
)
_PREFILTER_CAPTURE_DTYPE = np.dtype(
    {
        "names": ["projection", "view", "roughness", "sample_count", "env_resolution"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            np.float32,
            np.uint32,
            np.float32,
        ],
        "offsets": [0, 64, 128, 132, 136],
        "itemsize": 144,
    }
)
_BRDF_UNIFORM_DTYPE = np.dtype(
    {
        "names": ["sample_count"],
        "formats": [np.uint32],
        "offsets": [0],
        "itemsize": 16,
    }
)
```

- [ ] **Step 7: Let `bake_cube` carry the extra uniform fields**

`bake_cube` hardcodes `_CAPTURE_DTYPE`; Irradiance now needs its own. Replace `bake_cube` (lines 217-250) with:

```python
    def bake_cube(
        self,
        shader_name: str,
        size: int,
        src_view_dim: str,
        src,
        capture_dtype: np.dtype = _CAPTURE_DTYPE,
        extra: dict | None = None,
    ) -> "wgpu.GPUTexture":
        """Bake `shader_name` into all six faces of a new `size`^2 cube texture.

        `extra` sets any shader-specific uniform fields beyond projection/view.
        """
        pipe = self._make_cube_pipeline(shader_name, src_view_dim, capture_dtype)
        source_bind_group = self._source_bind_group(
            pipe["source_bgl"], src, src_view_dim
        )

        cube = self.device.create_texture(
            size=(size, size, 6),
            format=BAKE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_SRC,
        )

        uniforms = np.zeros((), dtype=capture_dtype)
        uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
        for name, value in (extra or {}).items():
            uniforms[name] = value
        for face in range(6):
            uniforms["view"] = _CAPTURE_VIEWS[face].to_numpy()
            self.device.queue.write_buffer(
                pipe["capture_buffer"], 0, uniforms.tobytes()
            )
            self._render_face(
                pipe["pipeline"],
                pipe["capture_bind_group"],
                source_bind_group,
                cube.create_view(
                    dimension="2d", base_array_layer=face, array_layer_count=1
                ),
                size,
            )
        return cube
```

and in `bake_maps`, pass the irradiance shader its sample delta:

```python
    irradiance = baker.bake_cube(
        "Irradiance.wgsl",
        settings.irradiance_size,
        "cube",
        env,
        capture_dtype=_IRRADIANCE_CAPTURE_DTYPE,
        extra={"sample_delta": settings.irradiance_sample_delta},
    )
```

- [ ] **Step 8: Fill the prefilter's new uniform fields**

In `bake_prefilter`, the `uniforms` block gains the two new members:

```python
            uniforms = np.zeros((), dtype=_PREFILTER_CAPTURE_DTYPE)
            uniforms["projection"] = _CAPTURE_PROJECTION.to_numpy()
            uniforms["roughness"] = self.settings.roughness_for_mip(mip)
            uniforms["sample_count"] = self.settings.prefilter_samples
            uniforms["env_resolution"] = float(self.settings.env_size)
```

- [ ] **Step 9: Give the BRDF pass its bind group**

In `bake_brdf`, replace the pipeline creation and add the uniform buffer. The old `layout=self.device.create_pipeline_layout(bind_group_layouts=[])` becomes:

```python
        bgl = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ],
        )
        uniform_buffer = self.device.create_buffer(
            size=_BRDF_UNIFORM_DTYPE.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        uniforms = np.zeros((), dtype=_BRDF_UNIFORM_DTYPE)
        uniforms["sample_count"] = self.settings.brdf_samples
        self.device.queue.write_buffer(uniform_buffer, 0, uniforms.tobytes())
        bind_group = self.device.create_bind_group(
            layout=bgl,
            entries=[{"binding": 0, "resource": {"buffer": uniform_buffer}}],
        )

        pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[bgl]),
            vertex={"module": shader, "entry_point": "vertex_main"},
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": LUT_FORMAT}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
```

and inside the render pass, bind it before the draw:

```python
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)
```

- [ ] **Step 10: Run the whole bake suite**

```bash
uv run pytest PBR/HDRIBaker/tests/ -v
```

Expected: every test passes, including the three new `sample` tests.

- [ ] **Step 11: Verify a default bake still looks right**

Re-bake the shipped HDRI at defaults and check it against the committed v1 file. They will not be bit-identical (the `envResolution` fix changes mip selection only if env_size ≠ 512, so at defaults they should be very close):

```bash
uv run python -c "
import sys; sys.path.insert(0, 'PBR/HDRIBaker')
import numpy as np
from bake_ibl import bake_maps
from bake_settings import BakeSettings
from hdri_input import load_equirect_hdr
from ibl_maps import load_maps
img = load_equirect_hdr('PBR/HDRIBaker/images/historic_cloister_passage_1k.exr')
fresh = bake_maps(img, BakeSettings(), source='historic_cloister_passage_1k.exr')
old = load_maps('PBR/HDRIBaker/ibl_maps.npz')
for k in ('irradiance', 'brdf_lut', 'prefilter_0'):
    a = np.asarray(fresh[k], np.float32); b = np.asarray(old[k], np.float32)
    print(k, 'max abs diff', float(np.max(np.abs(a - b))))
"
```

Expected: max abs diff near zero (under ~1e-2) for each. A large diff means a uniform is misaligned — check the dtype offsets against the WGSL structs before continuing.

- [ ] **Step 12: Lint and commit**

```bash
uv run ruff check PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/tests/test_bake_ibl.py
uv run ruff format PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/tests/test_bake_ibl.py
git add PBR/HDRIBaker/bake_ibl.py PBR/HDRIBaker/shaders/Prefilter.wgsl \
        PBR/HDRIBaker/shaders/Irradiance.wgsl PBR/HDRIBaker/shaders/BRDF.wgsl \
        PBR/HDRIBaker/tests/test_bake_ibl.py
git commit -m "feat(hdri): drive sample counts and env resolution from uniforms not WGSL consts"
```

---

### Task 5: The demo sizes its textures from the file

**Files:**
- Modify: `PBR/HDRIBaker/hdri_demo.py:34-41` (imports), `:353-360` (`_upload_maps`), `:113` (`_PBR_SCENE_DTYPE`)
- Modify: `PBR/HDRIBaker/shaders/PBR.wgsl:11` (`MAX_REFLECTION_LOD`) and its scene uniform struct

**Interfaces:**
- Consumes: `load_maps` returning `maps["settings"]` (Task 2).
- Produces: no new API. `hdri_demo.py` stops importing `ENV_SIZE`, `IRRADIANCE_SIZE`, `PREFILTER_SIZE`, `PREFILTER_MIPS`.

**Why the shader must change too:** `PBR.wgsl:11` has `const MAX_REFLECTION_LOD : f32 = 4.0; // PREFILTER_MIPS - 1`. Load a 3-mip file and the shader keeps sampling roughness up to LOD 4 on a chain that stops at 2, clamping every rough reflection to the last mip. It has to follow `prefilter_mips`.

- [ ] **Step 1: Add the uniform to the shader**

In `PBR/HDRIBaker/shaders/PBR.wgsl`, delete the `const MAX_REFLECTION_LOD : f32 = 4.0;` line at line 11, and append the value to the `Scene` struct at `@group(1)` instead:

```wgsl
// @group(1) scene lights, camera and the IBL toggle, shared by every draw this frame
struct Scene {
    lightPositions : array<vec4<f32>, 4>,
    lightColors : array<vec4<f32>, 4>,
    camPos : vec4<f32>,
    useIBL : u32,
    // prefilter_mips - 1: the roughest mip the chain actually has. Follows the
    // loaded map set, which is free to be baked with a shorter chain.
    maxReflectionLod : f32,
};
@group(1) @binding(0) var<uniform> scene : Scene;
```

Then change the prefilter sample near line 156:

```wgsl
        let prefilteredColor = textureSampleLevel(
            prefilterMap, iblSampler, R, roughness * scene.maxReflectionLod
        ).rgb;
```

- [ ] **Step 2: Mirror it in the numpy dtype**

In `PBR/HDRIBaker/hdri_demo.py`, replace `_PBR_SCENE_DTYPE` at line 113. `useIBL` sits at offset 144 with `itemsize` 160, so the new `f32` lands in the existing tail padding at 148 and the struct does not grow:

```python
_PBR_SCENE_DTYPE = np.dtype(
    {
        "names": ["lightPositions", "lightColors", "camPos", "useIBL",
                  "maxReflectionLod"],
        "formats": [
            (np.float32, (4, 4)),
            (np.float32, (4, 4)),
            (np.float32, 4),
            np.uint32,
            np.float32,
        ],
        "offsets": [0, 64, 128, 144, 148],
        "itemsize": 160,
    }
)
```

- [ ] **Step 3: Make `_upload_maps` read the settings**

Replace the `ibl_maps` import block (lines 34-41):

```python
from bake_settings import BakeSettings, prefilter_key
from ibl_maps import load_maps
```

and replace `_upload_maps` (lines 353-360):

```python
    def _upload_maps(self, maps: dict) -> None:
        """Upload a loaded map set to GPU cube/2D textures.

        Sizes come from the file's own settings, not from a constant here --
        a map set baked at any resolution has to land correctly.
        """
        settings: BakeSettings = maps["settings"]
        self.settings = settings
        self.env_cube = self._upload_cube(maps["env"], settings.env_size)
        self.irradiance_cube = self._upload_cube(
            maps["irradiance"], settings.irradiance_size
        )
        prefilter_mips = [
            maps[prefilter_key(m)] for m in range(1, settings.prefilter_mips)
        ]
        self.prefilter_cube = self._upload_cube(
            maps[prefilter_key(0)], settings.prefilter_size, prefilter_mips
        )
        self.brdf_lut = self._upload_lut(maps["brdf_lut"])
```

- [ ] **Step 4: Feed `maxReflectionLod` into the scene uniform**

Two sites set it, because the scene UBO and the map load happen in either order depending on startup path.

First, at the end of `_upload_maps` (the method you just wrote), keep the UBO in step with a freshly loaded file:

```python
        # On startup the maps may load before the scene UBO exists; the
        # pipeline-creation site below covers that case.
        if hasattr(self, "pbr_scene_uniforms"):
            self.pbr_scene_uniforms["maxReflectionLod"] = float(
                settings.prefilter_mips - 1
            )
```

Second, at the `self.pbr_scene_uniforms = np.zeros((), dtype=_PBR_SCENE_DTYPE)` block (around line 494), seed it from the settings that `_upload_maps` recorded:

```python
        self.pbr_scene_uniforms = np.zeros((), dtype=_PBR_SCENE_DTYPE)
        self.pbr_scene_uniforms["maxReflectionLod"] = float(
            self.settings.prefilter_mips - 1
        )
        for i, pos in enumerate(_LIGHT_POSITIONS):
```

Confirm which runs first before trusting either:

```bash
grep -n "_load_baked_maps()\|pbr_scene_uniforms = np.zeros" PBR/HDRIBaker/hdri_demo.py
```

If `_load_baked_maps()` is called *after* the UBO is built, the `hasattr` guard is dead but harmless and the seed does the work; if *before*, the seed reads a `self.settings` that already exists. Both sites together are correct either way. If neither runs first because `self.settings` is missing at the seed site, initialise `self.settings = BakeSettings()` in `__init__`.

- [ ] **Step 5: Run the demo against the legacy v1 file**

```bash
uv run PBR/HDRIBaker/hdri_demo.py --smoketest 2>&1 | tail -5
```

Expected: `SMOKETEST OK` (check `hdri_demo.py`'s argparse for the exact smoketest flag first — `grep -n smoketest PBR/HDRIBaker/hdri_demo.py`). If it has no smoketest flag, run it interactively and confirm the skybox and spheres render as before, then close it.

- [ ] **Step 6: Bake a small map set and load it in the demo**

This is the actual proof the schema change works end to end:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'PBR/HDRIBaker')
from bake_ibl import bake_maps
from bake_settings import BakeSettings
from hdri_input import load_equirect_hdr
from ibl_maps import save_maps
img = load_equirect_hdr('PBR/HDRIBaker/images/historic_cloister_passage_1k.exr')
s = BakeSettings(env_size=128, irradiance_size=16, prefilter_size=64,
                 prefilter_mips=3, lut_size=128, prefilter_samples=128)
save_maps(bake_maps(img, s, source='historic_cloister_passage_1k.exr'), '/tmp/small.npz')
print('baked')
"
uv run PBR/HDRIBaker/hdri_demo.py --maps /tmp/small.npz
```

Expected: the demo opens and renders with the low-res environment — visibly blurrier skybox, but correctly oriented and correctly lit, with no wgpu validation errors on stderr. This is the case that would have been silently broken without Tasks 2 and 5. Close the window when satisfied.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check PBR/HDRIBaker/hdri_demo.py
uv run ruff format PBR/HDRIBaker/hdri_demo.py
git add PBR/HDRIBaker/hdri_demo.py PBR/HDRIBaker/shaders/PBR.wgsl
git commit -m "feat(hdri): size the demo's IBL textures from the loaded file's settings"
```

---

### Task 6: The GUI controls

**Files:**
- Modify: `PBR/HDRIBaker/hdri_baker.py` (whole file — it is 175 lines)

**Interfaces:**
- Consumes: `BakeSettings` (Task 1), `DEFAULT_*` (Task 2), `bake_maps(image, settings, source)` (Task 3).
- Produces: a `SettingsPanel(QGroupBox)` with `settings() -> BakeSettings`, and a window that bakes at whatever the panel says.

**Design:** sizes are combo boxes of powers of two (a spin box invites 500 and a validation error). Sample counts are spin boxes. The sample delta is a double spin box. A `QLabel` shows the last bake's wall time — the whole point of the sample-count controls is the time/quality trade-off, and it is invisible without a number.

- [ ] **Step 1: Add the settings panel**

In `PBR/HDRIBaker/hdri_baker.py`, extend the imports:

```python
import time

from bake_settings import BakeSettings, prefilter_key
from ibl_maps import (
    DEFAULT_ENV_SIZE,
    DEFAULT_IRRADIANCE_SIZE,
    DEFAULT_LUT_SIZE,
    DEFAULT_PREFILTER_MIPS,
    DEFAULT_PREFILTER_SIZE,
    save_maps,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
```

(the existing `from ibl_maps import prefilter_key, save_maps` line goes away — `prefilter_key` now comes from `bake_settings`)

Add the panel class above `HDRIBakerWindow`:

```python
def _power_of_two_combo(choices: list[int], current: int) -> QComboBox:
    """A combo of power-of-two sizes -- a free-text spin box would only invite
    values the bake has to reject."""
    box = QComboBox()
    for value in choices:
        box.addItem(str(value), value)
    box.setCurrentIndex(choices.index(current))
    return box


class SettingsPanel(QGroupBox):
    """The bake's knobs. Sizes trade file size and detail; sample counts trade
    bake time against noise."""

    def __init__(self) -> None:
        super().__init__("Bake settings")
        self.env = _power_of_two_combo([128, 256, 512, 1024, 2048], DEFAULT_ENV_SIZE)
        self.irradiance = _power_of_two_combo([8, 16, 32, 64], DEFAULT_IRRADIANCE_SIZE)
        self.prefilter = _power_of_two_combo([32, 64, 128, 256], DEFAULT_PREFILTER_SIZE)
        self.lut = _power_of_two_combo([64, 128, 256, 512], DEFAULT_LUT_SIZE)

        self.mips = QSpinBox()
        self.mips.setRange(2, 8)  # 1 mip cannot span a roughness range
        self.mips.setValue(DEFAULT_PREFILTER_MIPS)

        self.prefilter_samples = QSpinBox()
        self.prefilter_samples.setRange(1, 8192)
        self.prefilter_samples.setValue(1024)

        self.brdf_samples = QSpinBox()
        self.brdf_samples.setRange(1, 8192)
        self.brdf_samples.setValue(1024)

        self.sample_delta = QDoubleSpinBox()
        self.sample_delta.setRange(0.005, 1.0)
        self.sample_delta.setSingleStep(0.005)
        self.sample_delta.setDecimals(3)
        self.sample_delta.setValue(0.025)

        form = QFormLayout()
        form.addRow("Environment cube", self.env)
        form.addRow("Irradiance cube", self.irradiance)
        form.addRow("Prefilter cube", self.prefilter)
        form.addRow("Prefilter mips", self.mips)
        form.addRow("BRDF LUT", self.lut)
        form.addRow("Prefilter samples", self.prefilter_samples)
        form.addRow("BRDF samples", self.brdf_samples)
        form.addRow("Irradiance sample delta", self.sample_delta)
        self.setLayout(form)

    def settings(self) -> BakeSettings:
        return BakeSettings(
            env_size=self.env.currentData(),
            irradiance_size=self.irradiance.currentData(),
            prefilter_size=self.prefilter.currentData(),
            prefilter_mips=self.mips.value(),
            lut_size=self.lut.currentData(),
            prefilter_samples=self.prefilter_samples.value(),
            brdf_samples=self.brdf_samples.value(),
            irradiance_sample_delta=self.sample_delta.value(),
        )
```

- [ ] **Step 2: Put the panel in the window**

In `HDRIBakerWindow.__init__`, after the thumb row is built, add the panel and a timing label, and put the preview and the panel side by side:

```python
        self.settings_panel = SettingsPanel()
        self.timing = QLabel("")
        self.timing.setAlignment(Qt.AlignCenter)

        side = QVBoxLayout()
        side.addWidget(self.settings_panel)
        side.addWidget(self.timing)
        side.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(self.preview, 1)
        top.addLayout(side)

        layout = QVBoxLayout()
        layout.addLayout(top, 1)
        layout.addLayout(thumb_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
```

(this replaces the existing `layout = QVBoxLayout()` … `self.setCentralWidget(central)` block)

- [ ] **Step 3: Bake at the panel's settings, and time it**

Replace the first half of `on_bake`:

```python
    def on_bake(self) -> None:
        if self.image is None:
            return
        settings = self.settings_panel.settings()
        try:
            settings.validate()
        except ValueError as err:
            self._show_error("Invalid bake settings", err)
            return

        # A big bake blocks the event loop for a while; at least let the
        # button look pressed and the cursor say so.
        self.bake_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        start = time.perf_counter()
        try:
            self.maps = bake_maps(self.image, settings, source=Path(self._source).name)
        except Exception as err:  # noqa: BLE001
            self._show_error("Bake failed", err)
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.bake_btn.setEnabled(True)
        self.timing.setText(f"baked in {time.perf_counter() - start:.2f}s")
```

The preview/thumbnail block that follows is unchanged **except** that the prefilter thumbnail must not assume mip 2 exists — a 2-mip bake has no `prefilter_2`. Replace that line:

```python
        thumb_mip = min(2, settings.prefilter_mips - 1)
        previews = (
            self.maps["irradiance"][0],
            self.maps[prefilter_key(thumb_mip)][0],
```

- [ ] **Step 4: Resize the window for the wider layout**

In `main()`, change the resize:

```python
    win.resize(1040, 640)
```

- [ ] **Step 5: Smoke-test the GUI**

```bash
uv run PBR/HDRIBaker/hdri_baker.py --smoketest
```

Expected: `SMOKETEST OK` and exit 0.

- [ ] **Step 6: Drive it by hand**

```bash
uv run PBR/HDRIBaker/hdri_baker.py
```

Then, in the window:
1. **Open HDRI…** → `PBR/HDRIBaker/images/historic_cloister_passage_1k.exr`. The panorama previews.
2. Set **Prefilter samples** to 8 and **Bake**. Note the time. The prefilter thumbnail should look visibly noisy/blotchy.
3. Set **Prefilter samples** to 2048 and **Bake** again. The time should rise noticeably and the thumbnail should clean up. *This is the demo the whole task exists for — if the two thumbnails look identical, the uniform from Task 4 is not reaching the shader.*
4. Set **Irradiance sample delta** to 0.5 and **Bake**. The irradiance thumbnail should go blocky.
5. Set **Prefilter mips** to 2 and **Bake**. It must not crash on the missing `prefilter_2` thumbnail.
6. **Save .npz…** → `/tmp/gui.npz`. Then `uv run PBR/HDRIBaker/hdri_demo.py --maps /tmp/gui.npz` and confirm it renders.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check PBR/HDRIBaker/hdri_baker.py
uv run ruff format PBR/HDRIBaker/hdri_baker.py
git add PBR/HDRIBaker/hdri_baker.py
git commit -m "feat(hdri): add GUI controls for map sizes and bake sample quality"
```

---

### Task 7: Docs and screenshot

**Files:**
- Modify: `PBR/HDRIBaker/README.md`
- Modify: `PBR/HDRIBaker/HDRIBaker.png` (replace)
- Modify: `README.md` (repo root) — only if the HDRIBaker entry's description no longer fits

**Interfaces:**
- Consumes: the finished GUI.
- Produces: docs.

- [ ] **Step 1: Load the writing skill**

Invoke the `jon-writing-style` skill before writing any prose in this task. The repo's CLAUDE.md requires it for all READMEs and comments.

- [ ] **Step 2: Retake the screenshot**

```bash
uv run PBR/HDRIBaker/hdri_baker.py
```

Open the shipped HDRI, bake at defaults so the thumbnails are populated and the timing label reads, then screenshot the window (`Cmd-Shift-4`, space, click the window) and save it over `PBR/HDRIBaker/HDRIBaker.png`. The settings panel must be visible — that is the point of the retake.

- [ ] **Step 3: Update the demo README**

Read `PBR/HDRIBaker/README.md` first, then add to it:
- a short section on the controls, grouped as the panel is: what the sizes cost (file size, bake time) and what the sample counts buy (less noise);
- the specific thing to try — bake at 8 prefilter samples, then at 2048, and compare the thumbnails and the times;
- a note that the `.npz` is schema v2 and carries its own settings, so the demo sizes its textures from the file, and that v1 files still load at the old fixed shape.

- [ ] **Step 4: Check the root README entry**

```bash
grep -n "HDRIBaker" README.md
```

If the existing description still describes the demo accurately, leave it. Only edit if it now undersells it.

- [ ] **Step 5: Full verification**

```bash
uv run pytest PBR/HDRIBaker/tests/ -v
uv run ruff check PBR/HDRIBaker/
uv run ruff format --check PBR/HDRIBaker/
uv run PBR/HDRIBaker/hdri_baker.py --smoketest
```

Expected: tests pass, ruff clean, `SMOKETEST OK`.

- [ ] **Step 6: Commit**

```bash
git add PBR/HDRIBaker/README.md PBR/HDRIBaker/HDRIBaker.png README.md
git commit -m "docs(hdri): document the baker's settings panel and the v2 map schema"
```

---

## Known quirk, not fixed here

`bake_cube` creates the env cube with no mip chain (`mip_level_count` defaults to 1), but `Prefilter.wgsl` computes a `mipLevel` and calls `textureSampleLevel` with it — that sample silently clamps to mip 0. The prefilter therefore does its GGX filtering by brute-force sampling alone, with the mip-based variance reduction inert. It predates this work, it is the same in `PBR/HDRI/HDRIWebGPU.py`, and it makes the prefilter *slower and noisier* rather than wrong. Fixing it means generating env mips before the prefilter pass, which is its own change. Task 4 makes `envResolution` correct so that a later fix has a right number to work with. Do not fix it inside this plan.
